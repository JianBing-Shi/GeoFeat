import math
import json
import os
import sys
from typing import Dict, Optional, Tuple, cast

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make project modules importable when this file is used as a script.
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.model.GeoFeatModel import GeoFeatModel  # noqa: E402
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InterpolateSparse2d(nn.Module):
	"""Sample dense feature maps at sparse 2D coordinates."""

	def __init__(self, mode="bicubic", align_corners=False):
		super().__init__()
		self.mode = mode
		self.align_corners = align_corners

	def normgrid(self, x, height, width):
		"""Normalize pixel coordinates to the ``[-1, 1]`` grid range."""
		return 2.0 * (
			x / torch.tensor(
				[width - 1, height - 1],
				device=x.device,
				dtype=x.dtype,
			)
		) - 1.0

	def forward(self, x, pos, height, width):
		"""Return sampled features with shape ``[B, N, C]``."""
		grid = self.normgrid(pos, height, width).unsqueeze(-2).to(x.dtype)
		x = F.grid_sample(
			x,
			grid,
			mode=self.mode,
			align_corners=self.align_corners,
		)
		return x.permute(0, 2, 3, 1).squeeze(-2)


REQUIRED_MODEL_CONFIG_KEYS = {
	"backbone",
	"keypoint_encoder",
	"keypoint_dim",
	"descriptor_encoder",
	"descriptor_dim",
	"geometric_features",
	"depth_encoder",
	"depth_dim",
	"gradient_encoder",
	"gradient_dim",
	"curvature_encoder",
	"curvature_dim",
	"repeatability_alpha",
	"attention_layers",
	"refiner_type",
	"last_activation",
	"output_dim",
}


def validate_model_config(model_config: Optional[dict]) -> dict:
	"""Strictly validate model_config required by GeoFeatModel and wrapper contract."""
	if model_config is None:
		raise ValueError(
			"GeoFeat requires model_config, but got None. "
			"Please pass the config snapshot's model_config explicitly."
		)

	if not isinstance(model_config, dict):
		raise TypeError(f"model_config must be a dict, got {type(model_config)}")

	missing = sorted([k for k in REQUIRED_MODEL_CONFIG_KEYS if k not in model_config])
	if missing:
		raise ValueError(
			"model_config is missing required keys: " + ", ".join(missing)
		)

	cfg = dict(model_config)
	geo_cfg = cfg["geometric_features"]
	if not isinstance(geo_cfg, dict):
		raise ValueError("model_config['geometric_features'] must be a dict")

	geo_cfg = dict(geo_cfg)
	for key in ("depth", "gradients", "curvatures"):
		if key not in geo_cfg:
			raise ValueError(f"model_config['geometric_features'] missing key: {key}")
	geo_cfg.setdefault("normal", False)
	cfg["geometric_features"] = geo_cfg

	return cfg


def normalize_legacy_model_config(model_config: Optional[dict], state_dict: Optional[dict] = None) -> dict:
	"""Patch older config snapshots so evaluation/inference can load archived checkpoints."""
	if model_config is None:
		raise ValueError("model_config is required, got None")
	if not isinstance(model_config, dict):
		raise TypeError(f"model_config must be a dict, got {type(model_config)}")

	cfg = dict(model_config)
	fallbacks = {
		"repeatability_alpha": 0.0,
		"geo_preprocess_mode": "no_unfold_lite",
		"aft_config": {
			"num_groups": 4,
			"local_kernel_sizes": [3, 5, 7],
			"domain_adapt": True,
			"temperature_min": 0.70,
			"temperature_max": 1.60,
			"local_base_gain": 1.0,
			"corr_base_gain": 0.5,
		},
		"geo_reliability_gate": {
			"enabled": False,
			"hidden_dim": 12,
			"gate_floor": 0.20,
			"bias_init": 2.0,
			"fusion_gain": 0.50,
		},
	}
	for key, value in fallbacks.items():
		if key not in cfg:
			cfg[key] = value
			print(f"Warning: model_config missing '{key}'; fallback to {value!r}.")
	cfg.setdefault("gcdr_config", {})
	cfg.setdefault("loss_config", {})
	if "backbone_channels" not in cfg:
		if int(cfg["descriptor_dim"]) >= 128:
			cfg["backbone_channels"] = [32, 32, 64, 128, 128]
		else:
			cfg["backbone_channels"] = [20, 20, 40, 80, 96]
	if "head_in_channels" not in cfg and isinstance(state_dict, dict):
		head_weight = state_dict["keypoint_head.layer1.layer.0.weight"] if "keypoint_head.layer1.layer.0.weight" in state_dict else None
		if head_weight is not None and getattr(head_weight, "shape", None) is not None:
			cfg["head_in_channels"] = int(head_weight.shape[1])
	if "descriptor_head_channels" not in cfg and isinstance(state_dict, dict):
		desc_key = "descriptor_head.layer.0.layer.0.weight"
		desc_weight = state_dict[desc_key] if desc_key in state_dict else None
		if desc_weight is not None and getattr(desc_weight, "shape", None) is not None:
			cfg["descriptor_head_channels"] = [int(desc_weight.shape[0]), 64]
	if "keypoint_head_channels" not in cfg and isinstance(state_dict, dict):
		kpt_channels = []
		for layer_idx in range(1, 6):
			kpt_key = f"keypoint_head.layer{layer_idx}.layer.0.weight"
			kpt_weight = state_dict[kpt_key] if kpt_key in state_dict else None
			if kpt_weight is not None and getattr(kpt_weight, "shape", None) is not None:
				kpt_channels.append(int(kpt_weight.shape[0]))
		if kpt_channels:
			cfg["keypoint_head_channels"] = kpt_channels
	geo_cfg = cfg["geometric_features"]
	if isinstance(geo_cfg, dict):
		geo_cfg = dict(geo_cfg)
		geo_cfg.setdefault("normal", False)
		if bool(geo_cfg["normal"]):
			# Old checkpoints used a standalone normal branch while the current
			# mainline packs normals into gradients. Preserve the checkpoint's
			# real contract when its weights prove it was trained that way.
			legacy_grad_key = "attn_fusion.geo_encoders.gradients.encoder.0.weight"
			legacy_grad_weight = state_dict[legacy_grad_key] if isinstance(state_dict, dict) and legacy_grad_key in state_dict else None
			legacy_standalone_normal = (
				legacy_grad_weight is not None
				and getattr(legacy_grad_weight, "shape", None) is not None
				and int(legacy_grad_weight.shape[1]) in (2, 128)
			)
			if not legacy_standalone_normal:
				geo_cfg["gradients"] = True
				geo_cfg["normal"] = False
		cfg["geometric_features"] = geo_cfg
	gate_cfg = cfg["geo_reliability_gate"]
	if isinstance(gate_cfg, dict):
		gate_cfg = dict(gate_cfg)
		gate_cfg.setdefault("fusion_gain", 0.50)
		cfg["geo_reliability_gate"] = gate_cfg
	gcdr_cfg = cfg["gcdr_config"]
	if isinstance(gcdr_cfg, dict):
		gcdr_cfg = dict(gcdr_cfg)
		gcdr_cfg.setdefault("residual_clip", 0.25)
		gcdr_cfg.setdefault("complementarity_enabled", False)
		gcdr_cfg.setdefault("texture_kernel_size", 5)
		gcdr_cfg.setdefault("texture_temperature", 8.0)
		gcdr_cfg.setdefault("complementarity_floor", 0.25)
		cfg["gcdr_config"] = gcdr_cfg
	loss_cfg = cfg["loss_config"]
	if isinstance(loss_cfg, dict):
		loss_cfg = dict(loss_cfg)
		loss_cfg.setdefault("descriptor_protect_weight", 0.0)
		loss_cfg.setdefault("reliable_enhance_margin", 0.02)
		loss_cfg.setdefault("reliable_enhance_weight", 0.0)
		loss_cfg.setdefault("reliable_enhance_threshold", 0.60)
		loss_cfg.setdefault("reliable_enhance_quantile", 0.50)
		loss_cfg.setdefault("unreliable_suppress_weight", 0.0)
		loss_cfg.setdefault("unreliable_suppress_threshold", 0.45)
		loss_cfg.setdefault("unreliable_suppress_quantile", 0.25)
		cfg["loss_config"] = loss_cfg
	return cfg


def load_model_config_snapshot(snapshot_path: str, state_dict: Optional[dict] = None) -> dict:
	"""Load and normalize a config snapshot produced by training."""
	if not os.path.isfile(snapshot_path):
		raise FileNotFoundError(f"Config snapshot not found: {snapshot_path}")

	with open(snapshot_path, "r", encoding="utf-8") as handle:
		payload = json.load(handle)
	if isinstance(payload, dict) and "model_config" in payload:
		payload = payload["model_config"]
	return normalize_legacy_model_config(payload, state_dict=state_dict)


def load_model_config_for_weights(weight_path: str) -> dict:
	"""Load config_snapshot.json from the same run directory as a checkpoint."""
	if not weight_path:
		raise ValueError("weight_path is required to locate config_snapshot.json")
	snapshot_path = os.path.join(os.path.dirname(os.path.abspath(weight_path)), "config_snapshot.json")
	state_dict = None
	if os.path.isfile(weight_path):
		state_dict = _read_checkpoint_state_dict(weight_path)
	return load_model_config_snapshot(snapshot_path, state_dict=state_dict)


def _read_checkpoint_state_dict(weight_path: str) -> dict:
	"""Read a checkpoint state dict and normalize common wrapper keys."""
	state = torch.load(weight_path, map_location="cpu")
	if isinstance(state, dict):
		if "model_state_dict" in state:
			state = state["model_state_dict"]
		elif "state_dict" in state:
			state = state["state_dict"]

	if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
		state = {k.replace("module.", "", 1): v for k, v in state.items()}
	return state


def load_model(model: nn.Module, weight_path: Optional[str]) -> nn.Module:
	"""Load weights if a valid path is provided."""
	if not weight_path:
		print("Warning: weight_path is None, using randomly initialized GeoFeat.")
		return model

	if not os.path.isfile(weight_path):
		print(f"Warning: weight file not found: {weight_path}, using randomly initialized GeoFeat.")
		return model

	state = _read_checkpoint_state_dict(weight_path)

	model_sd = model.state_dict()

	# Drop keys with shape mismatches to allow loading older checkpoints.
	filtered_state = {}
	dropped = []
	for k, v in state.items():
		if k not in model_sd:
			continue
		if model_sd[k].shape != v.shape:
			dropped.append(k)
			continue
		filtered_state[k] = v

	# Relaxed load: ignore missing/unexpected keys quietly to avoid noisy warnings
	if dropped:
		print(f"Warning: dropped {len(dropped)} keys due to shape mismatch: {dropped}")

	model.load_state_dict(filtered_state, strict=False)

	# Optional concise summary (comment out detailed lists to avoid clutter)
	model_keys = set(model_sd.keys())
	weight_keys = set(filtered_state.keys())
	missing = model_keys - weight_keys
	if missing:
		print(f"Warning: missing {len(missing)} keys in GeoFeat weights (suppressed list, enable debug if needed)")

	return model


class NonMaxSuppression(nn.Module):
	def __init__(self, rep_thr: float = 0.1, top_k: int = 0, kernel_size: int = 5):
		super().__init__()
		self.rep_thr = rep_thr
		self.top_k = top_k
		self.kernel_size = kernel_size
		self.padding = kernel_size // 2

	def forward(self, score: torch.Tensor) -> torch.Tensor:
		"""Return keypoint indices (B, N, 2) after NMS on a heatmap."""
		B, _, H, W = score.shape
		local_max = F.max_pool2d(score, kernel_size=self.kernel_size, stride=1, padding=self.padding)
		pos_mask = (score == local_max) & (score > self.rep_thr)

		pos_list = []
		for b in range(B):
			coords = pos_mask[b].nonzero(as_tuple=False)[..., 1:].flip(-1)  # (N, 2) -> (x, y)
			if coords.numel() == 0:
				pos_list.append(coords)
				continue

			# Rank by score and keep top_k if requested
			scores_b = score[b, 0, coords[:, 1], coords[:, 0]]
			if self.top_k and scores_b.numel() > self.top_k:
				topk_idx = torch.topk(scores_b, self.top_k).indices
				coords = coords[topk_idx]
			pos_list.append(coords)

		max_len = max((len(c) for c in pos_list), default=0)
		out = torch.zeros((B, max_len, 2), dtype=torch.long, device=score.device)
		for b, coords in enumerate(pos_list):
			if coords.numel() == 0:
				continue
			out[b, : len(coords)] = coords
		return out


class GeoFeat:
	def __init__(
		self,
		model_config: Optional[dict] = None,
		weight_path: Optional[str] = None,
		top_k: int = 0,
		detect_threshold: float = 0.1,
		nms_kernel_size: int = 5,
		match_min_cossim: float = -1.0,
		match_ratio_thresh: float = -1.0,
		descriptor_source: str = "refined",
		keypoint_source: str = "refined",
		descriptor_confidence_gain: float = 0.0,
		repeatability_adaptive_gamma: float = 0.0,
		force_bn_train: bool = False,
	):
		self.device = device
		self.top_k = top_k
		self.detect_threshold = detect_threshold
		self.match_min_cossim = match_min_cossim
		self.match_ratio_thresh = match_ratio_thresh
		self.descriptor_source = str(descriptor_source).lower()
		self.keypoint_source = str(keypoint_source).lower()
		valid_sources = {"raw", "refined"}
		if self.descriptor_source not in valid_sources:
			raise ValueError(f"descriptor_source must be one of {sorted(valid_sources)}, got {descriptor_source}")
		if self.keypoint_source not in valid_sources:
			raise ValueError(f"keypoint_source must be one of {sorted(valid_sources)}, got {keypoint_source}")
		base_config = validate_model_config(model_config)
		self.model_config = dict(base_config)
		self.model_config["descriptor_confidence_gain"] = float(descriptor_confidence_gain)
		self.model_config["repeatability_adaptive_gamma"] = float(repeatability_adaptive_gamma)

		self.net = GeoFeatModel(self.model_config).to(self.device).eval()
		self.net = cast(GeoFeatModel, load_model(self.net, weight_path))
		self.force_bn_train = force_bn_train
		self._apply_bn_train_mode()

		self.detector = NonMaxSuppression(
			rep_thr=detect_threshold,
			top_k=top_k,
			kernel_size=nms_kernel_size,
		).to(self.device)
		self.sampler = InterpolateSparse2d("bicubic").to(self.device)

	def _apply_bn_train_mode(self) -> None:
		if getattr(self, "force_bn_train", False):
			for m in self.net.modules():
				if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
					m.train()
					m.track_running_stats = True

	@staticmethod
	def _pad_size(length: int, divisor: int = 32) -> Tuple[int, int]:
		padded = math.ceil(length / divisor) * divisor
		return padded, padded - length

	def image_preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, list]:
		"""Pad to a multiple of 32 and convert to tensor on the correct device."""
		if image.ndim == 2:
			H, W = image.shape
		else:
			H, W, _ = image.shape

		_H, pad_h = self._pad_size(H)
		_W, pad_w = self._pad_size(W)

		if image.ndim == 2:
			image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, None, 0)
			image = image[None, ...]  # (1, H, W)
		else:
			image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, None, (0, 0, 0))
			image = image.transpose(2, 0, 1)  # (C, H, W)

		pad_info = [0, pad_h, 0, pad_w]
		tensor = torch.tensor(image, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0
		return tensor, pad_info

	@staticmethod
	def _build_heatmap_from_logits(kpt_logits: torch.Tensor) -> torch.Tensor:
		"""Convert 8x8 cell logits into full-resolution heatmap."""
		scores = F.softmax(kpt_logits, dim=1)[:, :64]
		heatmap = scores.permute(0, 2, 3, 1).reshape(scores.shape[0], scores.shape[2], scores.shape[3], 8, 8)
		heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(scores.shape[0], 1, scores.shape[2] * 8, scores.shape[3] * 8)
		return heatmap

	def _run_model_two_stage(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Run forward1 -> forward2 and expose raw/refined ablations."""
		stage1 = self.net.forward1(image)
		if not isinstance(stage1, (tuple, list)) or len(stage1) < 3:
			raise RuntimeError(
				"GeoFeatModel.forward1 must return at least (keypoint_logits, descriptor_map, geo_map)."
			)

		kpt_logits = stage1[0]
		descs_raw = stage1[1]
		geo_feats = stage1[2]
		raw_heatmap = self._build_heatmap_from_logits(kpt_logits)
		raw_descs_map = F.normalize(descs_raw, p=2, dim=1)

		need_refined_desc = self.descriptor_source == "refined"
		need_refined_kpt = self.keypoint_source == "refined"
		if not need_refined_desc and not need_refined_kpt:
			return raw_descs_map, raw_heatmap

		stage2 = self.net.forward2(kpt_logits, descs_raw, geo_feats, return_refined_keypoints=True)
		if isinstance(stage2, (tuple, list)) and len(stage2) == 2:
			descs_refined, kpt_logits_refined = stage2
		else:
			raise RuntimeError(
				"GeoFeatModel.forward2 must return (descriptors_refined, keypoint_logits_refined) "
				"when return_refined_keypoints=True."
			)

		if descs_refined.dim() != 4:
			raise RuntimeError(
				f"Expected descriptors_refined as [B, C, H, W], got shape={tuple(descs_refined.shape)}"
			)
		if kpt_logits_refined.dim() != 4:
			raise RuntimeError(
				f"Expected keypoint_logits_refined as [B, C, H, W], got shape={tuple(kpt_logits_refined.shape)}"
			)

		descs_map = raw_descs_map if self.descriptor_source == "raw" else F.normalize(descs_refined, p=2, dim=1)
		heatmap = raw_heatmap if self.keypoint_source == "raw" else self._build_heatmap_from_logits(kpt_logits_refined)
		return descs_map, heatmap

	@staticmethod
	def _crop_padding_keypoints(kpts: torch.Tensor, padded_h: int, padded_w: int, pad_info: list) -> torch.Tensor:
		"""Remove keypoints that fall inside right/bottom padding area."""
		if kpts.numel() == 0:
			return kpts

		valid_w = padded_w - pad_info[-1]
		valid_h = padded_h - pad_info[1]
		mask_w = kpts[..., 0] < valid_w
		kpts = kpts[mask_w]
		mask_h = kpts[..., 1] < valid_h
		kpts = kpts[mask_h]
		return kpts

	def _sample_sparse_outputs(self, heatmap: torch.Tensor, descs_map: torch.Tensor, kpts: torch.Tensor, padded_h: int, padded_w: int) -> Dict[str, torch.Tensor]:
		"""Sample scores/descriptors at sparse keypoints and pack output dict."""
		if kpts.numel() == 0:
			empty_desc = torch.empty((0, descs_map.shape[1]), device=self.device)
			empty_scores = torch.empty((0,), device=self.device)
			return {"descriptors": empty_desc, "keypoints": kpts, "scores": empty_scores}

		scores = self.sampler(heatmap, kpts.unsqueeze(0), padded_h, padded_w).squeeze(0).reshape(-1)
		descs = self.sampler(descs_map, kpts.unsqueeze(0), padded_h, padded_w).squeeze(0)
		descs = F.normalize(descs, p=2, dim=1)

		if self.top_k and scores.numel() > self.top_k:
			topk = torch.topk(scores, self.top_k)
			scores = topk.values
			kpts = kpts[topk.indices]
			descs = descs[topk.indices]

		return {"descriptors": descs, "keypoints": kpts, "scores": scores}

	@torch.inference_mode()
	def extract(self, image: np.ndarray) -> dict:
		image_tensor, pad_info = self.image_preprocess(image)
		if image_tensor.shape[0] != 1:
			raise RuntimeError(
				f"GeoFeat wrapper currently supports batch size 1 only, got B={image_tensor.shape[0]}"
			)

		_, _, padded_h, padded_w = image_tensor.shape
		self._apply_bn_train_mode()
		descs_map, heatmap = self._run_model_two_stage(image_tensor)

		pos = self.detector(heatmap)
		kpts = pos.squeeze(0)
		kpts = self._crop_padding_keypoints(kpts, padded_h, padded_w, pad_info)

		return self._sample_sparse_outputs(heatmap, descs_map, kpts, padded_h, padded_w)

	def match_featnet(
		self,
		img1: np.ndarray,
		img2: np.ndarray,
		min_cossim: Optional[float] = None,
		ratio_thresh: Optional[float] = None,
	) -> Tuple[np.ndarray, np.ndarray]:
		data1 = self.extract(img1)
		data2 = self.extract(img2)

		kpts1, feats1 = data1["keypoints"], data1["descriptors"]
		kpts2, feats2 = data2["keypoints"], data2["descriptors"]

		if kpts1.numel() == 0 or kpts2.numel() == 0:
			return np.empty((0, 2)), np.empty((0, 2))

		def nn_argmax(a: torch.Tensor, b: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
			"""Return argmax indices in b for each row in a using chunked matmul."""
			n = a.shape[0]
			out = torch.empty(n, dtype=torch.long, device=a.device)
			for i in range(0, n, chunk):
				j = min(i + chunk, n)
				sims = a[i:j] @ b.t()
				_, idx = sims.max(dim=1)
				out[i:j] = idx
			return out

		match12 = nn_argmax(feats1, feats2, chunk=1024)
		match21 = nn_argmax(feats2, feats1, chunk=1024)

		idx0 = torch.arange(len(match12), device=self.device)
		mutual = match21[match12] == idx0

		min_cossim = self.match_min_cossim if min_cossim is None else min_cossim
		ratio_thresh = self.match_ratio_thresh if ratio_thresh is None else ratio_thresh

		def max_scores(a: torch.Tensor, b: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
			"""Return per-row max similarity using chunked matmul."""
			n = a.shape[0]
			vals = torch.empty(n, dtype=a.dtype, device=a.device)
			for i in range(0, n, chunk):
				j = min(i + chunk, n)
				sims = a[i:j] @ b.t()
				m, _ = sims.max(dim=1)
				vals[i:j] = m
			return vals

		def ratio_mask(a: torch.Tensor, b: torch.Tensor, thr: float, chunk: int = 1024) -> torch.Tensor:
			"""Lowe-style ratio test on cosine similarities via distance ratio."""
			n = a.shape[0]
			keep = torch.zeros(n, dtype=torch.bool, device=a.device)
			for i in range(0, n, chunk):
				j = min(i + chunk, n)
				sims = a[i:j] @ b.t()
				if sims.shape[1] < 2:
					keep[i:j] = True
					continue
				top2 = torch.topk(sims, k=2, dim=1).values
				d1 = (1.0 - top2[:, 0]).clamp_min(1e-9)
				d2 = (1.0 - top2[:, 1]).clamp_min(1e-9)
				r = d1 / d2
				keep[i:j] = r < thr
			return keep

		if min_cossim > 0:
			cvals = max_scores(feats1, feats2, chunk=1024)
			good = cvals > min_cossim
			mutual = mutual & good

		if ratio_thresh > 0:
			rgood = ratio_mask(feats1, feats2, ratio_thresh, chunk=1024)
			mutual = mutual & rgood

		idx0 = idx0[mutual]
		idx1 = match12[mutual]

		mkpts1 = kpts1[idx0].cpu().numpy()
		mkpts2 = kpts2[idx1].cpu().numpy()

		return mkpts1, mkpts2
