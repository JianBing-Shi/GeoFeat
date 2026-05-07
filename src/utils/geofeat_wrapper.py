import math
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
from src.model.interpolator import InterpolateSparse2d  # noqa: E402


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


REQUIRED_MODEL_CONFIG_KEYS = {
	"backbone",
	"keypoint_encoder",
	"keypoint_dim",
	"descriptor_encoder",
	"descriptor_dim",
	"geometric_features",
	"depth_encoder",
	"depth_dim",
	"normal_encoder",
	"normal_dim",
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

	geo_cfg = model_config.get("geometric_features")
	if not isinstance(geo_cfg, dict):
		raise ValueError("model_config['geometric_features'] must be a dict")

	for key in ("depth", "normal", "gradients", "curvatures"):
		if key not in geo_cfg:
			raise ValueError(f"model_config['geometric_features'] missing key: {key}")

	return model_config


def load_model(model: nn.Module, weight_path: Optional[str]) -> nn.Module:
	"""Load weights if a valid path is provided."""
	if not weight_path:
		print("Warning: weight_path is None, using randomly initialized GeoFeat.")
		return model

	if not os.path.isfile(weight_path):
		print(f"Warning: weight file not found: {weight_path}, using randomly initialized GeoFeat.")
		return model

	state = torch.load(weight_path, map_location="cpu")
	if isinstance(state, dict):
		if "model_state_dict" in state:
			state = state["model_state_dict"]
		elif "state_dict" in state:
			state = state["state_dict"]

	# Strip potential DataParallel prefixes for compatibility
	if any(k.startswith("module.") for k in state.keys()):
		state = {k.replace("module.", "", 1): v for k, v in state.items()}

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
	def __init__(self, rep_thr: float = 0.1, top_k: int = 4096, kernel_size: int = 5):
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
		top_k: int = 4096,
		detect_threshold: float = 0.05,
		nms_kernel_size: int = 7,
		match_min_cossim: float = 0.1,
		match_ratio_thresh: float = 0.82,
		descriptor_source: str = "refined",
		keypoint_source: str = "refined",
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
		self.model_config = validate_model_config(model_config)

		self.net = GeoFeatModel(self.model_config).to(self.device).eval()
		self.net = cast(GeoFeatModel, load_model(self.net, weight_path))

		self.detector = NonMaxSuppression(
			rep_thr=detect_threshold,
			top_k=top_k,
			kernel_size=nms_kernel_size,
		).to(self.device)
		self.sampler = InterpolateSparse2d("bicubic").to(self.device)

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
