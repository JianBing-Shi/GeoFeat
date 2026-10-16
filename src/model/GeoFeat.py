"""
	"GeoFeat: 3D Geometry-Aware Local Feature Matching"
"""

import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
import torch_dct as DCT
from einops import rearrange

import tqdm
import math
import cv2
import time
from typing import Any, cast

import lightning.pytorch as pl

import sys
from typing import List, Tuple, Optional
from src.loss.loss import GeoFeatLoss, get_loss_config
from src.data import megadepth_wrapper
from src.utils.alike_wrapper import ALikeExtractor
from src.utils.depth_anything_wrapper import DepthAnythingExtractor
from src.model.GeometricExtractor import GeometricExtractor

"""
foundational functions
"""
def simple_nms(scores, radius):
	"""Perform non maximum suppression on the heatmap using max-pooling.
	This method does not suppress contiguous points that have the same score.
	Args:
		scores: the score heatmap of size `(B, H, W)`.
		radius: an integer scalar, the radius of the NMS window.
	"""

	def max_pool(x):
		return torch.nn.functional.max_pool2d(
			x, kernel_size=radius * 2 + 1, stride=1, padding=radius
		)

	zeros = torch.zeros_like(scores)
	max_mask = scores == max_pool(scores)
	for _ in range(2):
		supp_mask = max_pool(max_mask.float()) > 0
		supp_scores = torch.where(supp_mask, zeros, scores)
		new_max_mask = supp_scores == max_pool(supp_scores)
		max_mask = max_mask | (new_max_mask & (~supp_mask))
	return torch.where(max_mask, scores, zeros)

def top_k_keypoints(keypoints, scores, k):
	if k >= len(keypoints):
		return keypoints, scores
	scores, indices = torch.topk(scores, k, dim=0, sorted=True)
	return keypoints[indices], scores

def sample_k_keypoints(keypoints, scores, k):
	if k >= len(keypoints):
		return keypoints, scores
	indices = torch.multinomial(scores, k, replacement=False)
	return keypoints[indices], scores[indices]

def soft_argmax_refinement(keypoints, scores, radius: int):
	width = 2 * radius + 1
	sum_ = torch.nn.functional.avg_pool2d(
		scores[:, None], width, 1, radius, divisor_override=1
	)
	ar = torch.arange(-radius, radius + 1).to(scores)
	kernel_x = ar[None].expand(width, -1)[None, None]
	dx = torch.nn.functional.conv2d(scores[:, None], kernel_x, padding=radius)
	dy = torch.nn.functional.conv2d(
		scores[:, None], kernel_x.transpose(2, 3), padding=radius
	)
	dydx = torch.stack([dy[:, 0], dx[:, 0]], -1) / sum_[:, 0, :, :, None]
	refined_keypoints = []
	for i, kpts in enumerate(keypoints):
		delta = dydx[i][tuple(kpts.t())]
		refined_keypoints.append(kpts.float() + delta)
	return refined_keypoints

def sample_descriptors_fix_sampling(keypoints, descriptors, s: int = 8):
	"""Interpolate descriptors at keypoint locations"""
	b, c, h, w = descriptors.shape
	keypoints = keypoints / (keypoints.new_tensor([w, h]) * s)
	keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
	descriptors = torch.nn.functional.grid_sample(
		descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=False
	)
	descriptors = torch.nn.functional.normalize(
		descriptors.reshape(b, c, -1), p=2, dim=1
	)
	return descriptors

# --------- Backbone Classes ---------
# ========== StandardBackbone ==========
# 标准卷积骨干网络，输出三个不同分辨率的特征图供后续融合和头部使用。
class StandardBackbone(nn.Module):
	"""Plain CNN pyramid used when MODEL.BACKBONE == "Standard".

	The forward pass returns x3/x4/x5 at progressively lower resolutions. GeoFeat
	fuses these maps back to the x3 stride so detector, descriptor, and geometry
	heads share one dense feature grid.
	"""
	def __init__(self):
		super().__init__()
		c1, c2, c3, c4, c5 = 20, 20, 40, 80, 96
		self.output_channels = (c3, c4, c5)
		self.relu = nn.ReLU(inplace=True)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
		self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
		self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
		self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
		self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
		self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
		self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
		self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
		self.conv5a = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
		self.conv5b = nn.Conv2d(c5, c5, kernel_size=3, stride=1, padding=1)

	def forward(self, x):
		x1 = self.relu(self.conv1a(x))
		x1 = self.relu(self.conv1b(x1))
		x1 = self.pool(x1)
		x2 = self.relu(self.conv2a(x1))
		x2 = self.relu(self.conv2b(x2))
		x2 = self.pool(x2)
		x3 = self.relu(self.conv3a(x2))
		x3 = self.relu(self.conv3b(x3))
		x3 = self.pool(x3)
		x4 = self.relu(self.conv4a(x3))
		x4 = self.relu(self.conv4b(x4))
		x4 = self.pool(x4)
		x5 = self.relu(self.conv5a(x4))
		x5 = self.relu(self.conv5b(x5))
		x5 = self.pool(x5)
		return x3, x4, x5

# ========== RepVGG Backbone ==========
# RepVGG骨干网络，使用可转换为单分支卷积的多分支模块。训练时使用多分支结构以增加容量，部署时可以转换为单3x3卷积以加速推理。
def _fuse_conv_and_bn(conv, bn):
	"""Fuse conv and bn layer params to a single conv kernel and bias."""
	with torch.no_grad():
		# conv: nn.Conv2d, bn: nn.BatchNorm2d
		w = conv.weight
		if conv.bias is None:
			b = torch.zeros(w.size(0), device=w.device)
		else:
			b = conv.bias

		bn_w = bn.weight
		bn_b = bn.bias
		bn_rm = bn.running_mean
		bn_rv = bn.running_var
		if bn_w is None or bn_b is None or bn_rm is None or bn_rv is None:
			raise RuntimeError("BatchNorm2d must have affine params and running stats before fusion.")
		eps = bn.eps

		std = torch.sqrt(bn_rv + eps)
		w_fold = w * (bn_w / std).reshape(-1, 1, 1, 1)
		b_fold = (b - bn_rm) / std * bn_w + bn_b
		return w_fold, b_fold

class RepVGGBlock(nn.Module):
	"""RepVGG-like block (training multi-branch, can be converted to single conv for deploy).

	This implementation keeps no activation inside so it matches calling code that applies activations.
	"""
	def __init__(self, in_channels, out_channels, stride=1, deploy=False):
		super().__init__()
		self.deploy = deploy
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.stride = stride

		if deploy:
			self.rbr_reparam = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
		else:
			# 3x3 branch
			self.rbr_dense = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False),
				nn.BatchNorm2d(out_channels)
			)

			# 1x1 branch
			self.rbr_1x1 = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, 1, stride, padding=0, bias=False),
				nn.BatchNorm2d(out_channels)
			)

			# identity branch (if applicable)
			if out_channels == in_channels and stride == 1:
				self.rbr_identity = nn.BatchNorm2d(in_channels)
			else:
				self.rbr_identity = None

	def forward(self, x):
		if self.deploy:
			return self.rbr_reparam(x)
		out = self.rbr_dense(x) + self.rbr_1x1(x)
		if self.rbr_identity is not None:
			out += self.rbr_identity(x)
		return out

	def get_equivalent_kernel_bias(self):
		"""Return kernel and bias to create a single conv equivalent to the block."""
		if self.deploy:
			return self.rbr_reparam.weight, self.rbr_reparam.bias

		kernel3x3, bias3x3 = _fuse_conv_and_bn(self.rbr_dense[0], self.rbr_dense[1])
		kernel1x1, bias1x1 = _fuse_conv_and_bn(self.rbr_1x1[0], self.rbr_1x1[1])

		# pad 1x1 kernel to 3x3
		kernel1x1 = F.pad(kernel1x1, [1,1,1,1])

		if self.rbr_identity is not None:
			bn = self.rbr_identity
			running_var = bn.running_var
			gamma = bn.weight
			beta = bn.bias
			running_mean = bn.running_mean
			if running_var is None or gamma is None or beta is None or running_mean is None:
				raise RuntimeError("Identity BatchNorm2d must have affine params and running stats before fusion.")
			std = torch.sqrt(running_var + bn.eps)
			# identity kernel and bias
			id_kernel = torch.zeros_like(kernel3x3)
			for i in range(self.out_channels):
				id_kernel[i, i, 1, 1] = (gamma[i] / std[i]).item()
			id_bias = (- running_mean / std * gamma + beta)
		else:
			id_kernel = torch.zeros_like(kernel3x3)
			id_bias = torch.zeros(self.out_channels, device=kernel3x3.device)

		kernel = kernel3x3 + kernel1x1 + id_kernel
		bias = bias3x3 + bias1x1 + id_bias
		return kernel, bias

	def switch_to_deploy(self):
		if self.deploy:
			return
		kernel, bias = self.get_equivalent_kernel_bias()
		if bias is None:
			raise RuntimeError("RepVGG fused bias must not be None during deploy conversion.")
		self.rbr_reparam = nn.Conv2d(self.in_channels, self.out_channels, 3, self.stride, padding=1, bias=True)
		if self.rbr_reparam.bias is None:
			raise RuntimeError("RepVGG deploy conv must have a bias parameter.")
		with torch.no_grad():
			self.rbr_reparam.weight.copy_(kernel)
			self.rbr_reparam.bias.copy_(bias)

		# delete unneeded branches
		delattr(self, 'rbr_dense')
		delattr(self, 'rbr_1x1')
		if self.rbr_identity is not None:
			delattr(self, 'rbr_identity')
		self.deploy = True

class RepVGGBackbone(nn.Module):
	"""RepVGG-style pyramid with deployable convolution blocks.

	Training uses multi-branch RepVGGBlock modules. The branch fusion helpers above
	can convert each block into one 3x3 conv for faster deployment if needed.
	"""
	def __init__(self):
		super().__init__()
		c1, c2, c3, c4, c5 = 20, 20, 40, 80, 96
		self.output_channels = (c3, c4, c5)
		self.relu = nn.ReLU(inplace=True)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv1a = RepVGGBlock(1, c1, stride=1, deploy=False)
		self.conv1b = RepVGGBlock(c1, c1, stride=1, deploy=False)
		self.conv2a = RepVGGBlock(c1, c2, stride=1, deploy=False)
		self.conv2b = RepVGGBlock(c2, c2, stride=1, deploy=False)
		self.conv3a = RepVGGBlock(c2, c3, stride=1, deploy=False)
		self.conv3b = RepVGGBlock(c3, c3, stride=1, deploy=False)
		self.conv4a = RepVGGBlock(c3, c4, stride=1, deploy=False)
		self.conv4b = RepVGGBlock(c4, c4, stride=1, deploy=False)
		self.conv5a = RepVGGBlock(c4, c5, stride=1, deploy=False)
		self.conv5b = RepVGGBlock(c5, c5, stride=1, deploy=False)

	def forward(self, x):
		x1 = self.relu(self.conv1a(x))
		x1 = self.relu(self.conv1b(x1))
		x1 = self.pool(x1)
		x2 = self.relu(self.conv2a(x1))
		x2 = self.relu(self.conv2b(x2))
		x2 = self.pool(x2)
		x3 = self.relu(self.conv3a(x2))
		x3 = self.relu(self.conv3b(x3))
		x3 = self.pool(x3)
		x4 = self.relu(self.conv4a(x3))
		x4 = self.relu(self.conv4b(x4))
		x4 = self.pool(x4)
		x5 = self.relu(self.conv5a(x4))
		x5 = self.relu(self.conv5b(x5))
		x5 = self.pool(x5)
		return x3, x4, x5

# --------- UpsampleLayer Classes ---------
# ========== UpsampleLayer ==========
# 双线性2x上采样，后跟通道数减少和局部平滑。增加容量以更好地恢复几何细节，同时保持FPN式融合的紧凑。
class UpsampleLayer(nn.Module):
	"""Bilinear 2x upsampling followed by channel reduction and local smoothing."""
	def __init__(self, in_channels):
		super().__init__()
		# Reduce channels after upsampling so later FPN-style fusion stays compact.
		self.conv = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1)
		self.bn = nn.BatchNorm2d(in_channels//2)
		self.leaky_relu = nn.LeakyReLU(0.1)

	def forward(self, x):
		x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
		x = self.leaky_relu(self.bn(self.conv(x)))

		return x

# --------- Head Classes ---------
# 多头分支路径，分别预测关键点热图、描述子和几何特征。每个头部都使用共享的融合特征图作为输入。
class BaseLayer(nn.Module):
	def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False,activation=True):
		super().__init__()
		if activation:
			self.layer=nn.Sequential(
				nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=bias),
				nn.BatchNorm2d(out_channels,affine=False),
				nn.ReLU(inplace=True)
			)
		else:
			self.layer=nn.Sequential(
				nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=bias),
				nn.BatchNorm2d(out_channels,affine=False)
			)

	def forward(self,x):
		return self.layer(x)

class KeypointHead(nn.Module):
	"""Dense detector head.

	Output channel count is normally 65: 64 cell offsets plus a dustbin-style
	channel, following the SuperPoint/LiftFeat dense detector convention.
	"""
	def __init__(self,in_channels,out_channels, hidden_channels: Optional[List[int]] = None):
		super().__init__()
		hidden_channels = [32, 32, 64] if hidden_channels is None else list(hidden_channels)
		if not hidden_channels:
			raise ValueError("KeypointHead hidden_channels must not be empty")
		prev_channels = int(in_channels)
		for idx, hidden in enumerate(hidden_channels, start=1):
			setattr(self, f'layer{idx}', BaseLayer(prev_channels, int(hidden)))
			prev_channels = int(hidden)
		self._layer_count = len(hidden_channels)

		self.conv=nn.Conv2d(prev_channels,out_channels,kernel_size=3,stride=1,padding=1)
		self.bn=nn.BatchNorm2d(out_channels)

	def forward(self,x):
		for idx in range(1, self._layer_count + 1):
			x = getattr(self, f'layer{idx}')(x)
		x=self.bn(self.conv(x))
		return x

class DescriptorHead(nn.Module):
	"""Dense descriptor head that predicts one descriptor vector per feature cell."""
	def __init__(self,in_channels,out_channels, hidden_channels: Optional[List[int]] = None):
		super().__init__()
		hidden_channels = [32, 64] if hidden_channels is None else list(hidden_channels)
		if len(hidden_channels) != 2:
			raise ValueError(f"DescriptorHead hidden_channels must contain 2 integers, got {hidden_channels}")
		h1, h2 = [int(v) for v in hidden_channels]
		# Compact descriptor head for the sub-1M-parameter mainline.
		self.layer=nn.Sequential(
			BaseLayer(in_channels, h1),
			BaseLayer(h1, h2, activation=True),
			BaseLayer(h2, out_channels, activation=False)
		)

	def forward(self,x):
		x=self.layer(x)
		# x=nn.functional.softmax(x,dim=1)
		return x

class GeoHead(nn.Module):
	"""LiftFeat-style geometry head with dynamic output branches.

	Depending on config, it predicts depth, gradients (dx, dy), normals
	(nx, ny, nz), and Hessian proxies (half-trace H, half-eigenvalue-gap D).
	The branch order is fixed because
	FeatureBooster later slices the concatenated geometry tensor in the same order.
	"""
	def __init__(self, in_channels: int, geo_cfg: dict):
		super().__init__()
		# Track branch names and per-branch channel counts for selective normalization
		branch_defs = [
			('depth', 1),
			('gradients', 2),
			('normals', 3),
			('curvatures', 2),
		]
		self._branch_names = []
		self._branch_channels = []
		out_ch = 0
		for name, ch in branch_defs:
			if geo_cfg[name]:
				self._branch_names.append(name)
				self._branch_channels.append(ch)
				out_ch += ch
		if out_ch == 0:
			raise ValueError("GeoHead requires at least one enabled geometric feature")

		self.out_channels = out_ch
		# three-stage upsample like LiftFeat DepthHead
		self.upsampleDa = UpsampleLayer(in_channels)
		self.upsampleDb = UpsampleLayer(in_channels//2)
		self.upsampleDc = UpsampleLayer(in_channels//4)

		self.convDepa = nn.Conv2d(in_channels//2+in_channels, in_channels//2, kernel_size=3, stride=1, padding=1)
		self.bnDepa = nn.BatchNorm2d(in_channels//2)
		self.convDepb = nn.Conv2d(in_channels//4+in_channels//2, in_channels//4, kernel_size=3, stride=1, padding=1)
		self.bnDepb = nn.BatchNorm2d(in_channels//4)
		self.convDepc = nn.Conv2d(in_channels//8+in_channels//4, self.out_channels, kernel_size=3, stride=1, padding=1)
		self.bnDepc = nn.BatchNorm2d(self.out_channels)
		self.leaky_relu = nn.LeakyReLU(0.1)

	def forward(self, x):
		# Restore geometry to image-level resolution in three 2x steps. Each step
		# concatenates an interpolated skip with a learned upsampled feature.
		x0 = F.interpolate(x, scale_factor=2,mode='bilinear',align_corners=False)
		x1 = self.upsampleDa(x)
		x1 = torch.cat([x0,x1],dim=1)
		x1 = self.leaky_relu(self.bnDepa(self.convDepa(x1)))

		x1_0 = F.interpolate(x1,scale_factor=2,mode='bilinear',align_corners=False)
		x2 = self.upsampleDb(x1)
		x2 = torch.cat([x1_0,x2],dim=1)
		x2 = self.leaky_relu(self.bnDepb(self.convDepb(x2)))

		x2_0 = F.interpolate(x2,scale_factor=2,mode='bilinear',align_corners=False)
		x3 = self.upsampleDc(x2)
		x3 = torch.cat([x2_0,x3],dim=1)
		x = self.leaky_relu(self.bnDepc(self.convDepc(x3)))

		# Only surface normals are constrained to unit length.
		parts = x.split(self._branch_channels, dim=1)
		out = []
		for i, name in enumerate(self._branch_names):
			p = parts[i]
			if name == 'normals':
				p = F.normalize(p, p=2, dim=1)
			out.append(p)
		return torch.cat(out, dim=1)

# class HeatmapHead(nn.Module):
# 	def __init__(self,in_channels,mid_channels,out_channels):
# 		super().__init__()
# 		self.convHa = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
# 		self.bnHa = nn.BatchNorm2d(mid_channels)
# 		self.convHb = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
# 		self.bnHb = nn.BatchNorm2d(out_channels)
# 		self.leaky_relu = nn.LeakyReLU(0.1)

# 	def forward(self,x):
# 		x = self.leaky_relu(self.bnHa(self.convHa(x)))
# 		x = self.leaky_relu(self.bnHb(self.convHb(x)))

# 		x = torch.sigmoid(x)
# 		return x
# --------- Encoder Classes ---------
# 编码器类，使用MLP将几何属性和描述子映射到特征空间。每个编码器都可以选择性地应用dropout以增强训练时的鲁棒性。
def MLP(channels: List[int], do_bn: bool = False) -> nn.Module:
	""" Multi-layer perceptron """
	n = len(channels)
	layers = []
	for i in range(1, n):
		layers.append(nn.Linear(channels[i - 1], channels[i]))
		if i < (n-1):
			if do_bn:
				layers.append(nn.BatchNorm1d(channels[i]))
			layers.append(nn.ReLU())
	return nn.Sequential(*layers)

def MLP_no_ReLU(channels: List[int], do_bn: bool = False) -> nn.Module:
	""" Multi-layer perceptron """
	n = len(channels)
	layers = []
	for i in range(1, n):
		layers.append(nn.Linear(channels[i - 1], channels[i]))
		if i < (n-1):
			if do_bn:
				layers.append(nn.BatchNorm1d(channels[i]))
	return nn.Sequential(*layers)

class KeypointEncoder(nn.Module):
	""" Encoding of geometric properties using MLP """
	def __init__(self, keypoint_dim: int, feature_dim: int, dropout: bool = False, p: float = 0.1) -> None:
		super().__init__()
		self.encoder = MLP([keypoint_dim, 128, 64, 64, feature_dim])
		self.use_dropout = dropout
		self.dropout = nn.Dropout(p=p)

	def forward(self, kpts):
		if self.use_dropout:
			return self.dropout(self.encoder(kpts))
		return self.encoder(kpts)

class DescriptorEncoder(nn.Module):
	""" Encoding of visual descriptor using MLP """
	def __init__(self, feature_dim: int, dropout: bool = False, p: float = 0.1) -> None:
		super().__init__()
		self.encoder = MLP([feature_dim, 96, feature_dim])
		self.use_dropout = dropout
		self.dropout = nn.Dropout(p=p)

	def forward(self, descs):
		residual = descs
		if self.use_dropout:
			return residual + self.dropout(self.encoder(descs))
		return residual + self.encoder(descs)

class GeometricEncoder(nn.Module):
	""" Encoding of geometric properties using MLP """
	def __init__(self, input_dim: int, feature_dim: int, layers: List[int], dropout: bool = False, p: float = 0.1) -> None:
		super().__init__()
		self.encoder = MLP_no_ReLU([input_dim] + layers + [feature_dim])
		self.use_dropout = dropout
		self.dropout = nn.Dropout(p=p)

	def forward(self, x):
		if self.use_dropout:
			return self.dropout(self.encoder(x))
		return self.encoder(x)

class GeometryReliabilityGate(nn.Module):
	"""Predict where geometry is trustworthy enough to guide refinement.

	The gate looks at both raw geometry and encoded geometry memory, but it does
	not inject that memory into descriptors. Geometry should decide where/how much
	to enhance, not what the descriptor content should be.
	"""

	def __init__(
		self,
		geo_dim: int,
		feature_dim: int,
		hidden_dim: int = 12,
		gate_floor: float = 0.20,
		bias_init: float = 2.0,
		fusion_gain: float = 0.50,
	) -> None:
		super().__init__()
		hidden_dim = max(1, int(hidden_dim))
		self.gate_floor = float(max(0.0, min(gate_floor, 1.0)))
		self.fusion_gain = float(max(0.0, fusion_gain))

		self.raw_proj = nn.Sequential(
			nn.Conv2d(geo_dim, hidden_dim, 1, bias=False),
			nn.BatchNorm2d(hidden_dim),
			nn.GELU(),
		)
		self.memory_proj = nn.Sequential(
			nn.Conv2d(feature_dim, hidden_dim, 1, bias=False),
			nn.BatchNorm2d(hidden_dim),
			nn.GELU(),
		)
		self.reliability_head = nn.Sequential(
			nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1, bias=False),
			nn.BatchNorm2d(hidden_dim),
			nn.GELU(),
			nn.Conv2d(hidden_dim, 1, 1),
		)

		out_proj = self.reliability_head[-1]
		if isinstance(out_proj, nn.Conv2d):
			nn.init.zeros_(out_proj.weight)
			if out_proj.bias is not None:
				nn.init.constant_(out_proj.bias, float(bias_init))

	def forward(self, geo_v: torch.Tensor, geo_memory: torch.Tensor):
		"""Estimate reliability from NCHW raw geometry and NHWC geometry memory."""
		if geo_v.dim() != 4 or geo_memory.dim() != 4:
			raise RuntimeError(
				"GeometryReliabilityGate expects geo_v as NCHW and geo_memory as NHWC, "
				f"got {tuple(geo_v.shape)} and {tuple(geo_memory.shape)}."
			)

		geo_memory_map = geo_memory.permute(0, 3, 1, 2)
		if geo_v.shape[-2:] != geo_memory_map.shape[-2:]:
			geo_v = F.interpolate(
				geo_v,
				size=geo_memory_map.shape[-2:],
				mode='bilinear',
				align_corners=False,
			)

		raw_feat = self.raw_proj(geo_v)
		memory_feat = self.memory_proj(geo_memory_map)
		reliability = torch.sigmoid(self.reliability_head(torch.cat([raw_feat, memory_feat], dim=1)))
		return reliability

# Geometry-conditioned descriptor residual refinement.
class GCDR(nn.Module):
	"""Geo-Conditional Descriptor Refinement.

	GCDR compresses raw geometry, predicts a reliability map, and uses geometry
	to generate FiLM scale/shift for descriptors. The reliability map gates the
	geometry residual so noisy geometry cannot dominate the descriptor stream.
	"""

	def __init__(
		self,
		dim: int,
		geo_dim: int,
		geo_compress: int = 8,
		reliability_hidden: int = 16,
		gate_floor: float = 0.10,
		residual_gain: float = 1.0,
		residual_gate_floor: float = 0.10,
		residual_clip: float = 0.25,
		complementarity_enabled: bool = False,
		texture_kernel_size: int = 5,
		texture_temperature: float = 8.0,
		complementarity_floor: float = 0.25,
	):
		super().__init__()
		self.gate_floor = float(max(0.0, min(gate_floor, 1.0)))
		self.residual_gain = float(max(0.0, residual_gain))
		self.residual_gate_floor = float(max(0.0, min(residual_gate_floor, 1.0)))
		self.residual_clip = float(max(0.0, residual_clip))
		self.complementarity_enabled = bool(complementarity_enabled)
		self.texture_kernel_size = int(texture_kernel_size)
		if self.texture_kernel_size < 3 or self.texture_kernel_size % 2 == 0:
			self.texture_kernel_size = 5
		self.texture_temperature = float(max(texture_temperature, 1e-3))
		self.complementarity_floor = float(max(0.0, min(complementarity_floor, 1.0)))

		self.geo_proj = nn.Sequential(
			nn.Conv2d(geo_dim, geo_compress, 1, bias=False),
			nn.BatchNorm2d(geo_compress),
			nn.GELU(),
		)

		self.film_gen = nn.Conv2d(geo_compress, dim * 2, 1)
		nn.init.zeros_(self.film_gen.weight)
		if self.film_gen.bias is not None:
			nn.init.zeros_(self.film_gen.bias)

		reliability_hidden = max(1, int(reliability_hidden))
		self.reliability_head = nn.Sequential(
			nn.Conv2d(dim + geo_compress, reliability_hidden, 1, bias=False),
			nn.BatchNorm2d(reliability_hidden),
			nn.GELU(),
			nn.Conv2d(reliability_hidden, 1, 1),
		)
		nn.init.zeros_(self.reliability_head[-1].weight)
		nn.init.zeros_(self.reliability_head[-1].bias)

		self.norm = nn.LayerNorm(dim)

	def _aux(self, effective_reliability, texture_weakness):
		return {
			'effective_reliability_map': effective_reliability,
			'texture_weakness_map': texture_weakness,
		}

	def forward_with_reliability(self, desc: torch.Tensor, geo_v: torch.Tensor, reliability_map: torch.Tensor):
		"""Refine descriptors using an externally estimated geometry reliability map."""
		geo_feat = self.geo_proj(geo_v)
		reliability_map = F.interpolate(
			reliability_map,
			size=geo_feat.shape[-2:],
			mode='bilinear',
			align_corners=False,
		).clamp(0.0, 1.0)
		out, effective, texture = self.refine_with_reliability(desc, geo_feat, reliability_map)
		return out, reliability_map, self._aux(effective, texture)

	def _texture_weakness(self, desc: torch.Tensor) -> torch.Tensor:
		"""Estimate where visual evidence is weak using local descriptor variance."""
		desc_map = desc.permute(0, 3, 1, 2)
		desc_energy = desc_map.pow(2).mean(dim=1, keepdim=True)
		k = self.texture_kernel_size
		local_mean = F.avg_pool2d(desc_energy, kernel_size=k, stride=1, padding=k // 2)
		local_var = F.avg_pool2d((desc_energy - local_mean).pow(2), kernel_size=k, stride=1, padding=k // 2)
		local_std = local_var.clamp_min(1e-8).sqrt()
		baseline = local_std.mean(dim=(2, 3), keepdim=True).detach()
		texture_conf = torch.sigmoid(self.texture_temperature * (local_std - baseline))
		return (1.0 - texture_conf).clamp(0.0, 1.0)

	def refine_with_reliability(self, desc: torch.Tensor, geo_feat: torch.Tensor, reliability_map: torch.Tensor):
		"""Apply a small geometry-conditioned residual under reliability control."""
		effective_reliability = reliability_map
		texture_weakness = None
		if self.complementarity_enabled:
			texture_weakness = self._texture_weakness(desc)
			effective_reliability = reliability_map * (
				self.complementarity_floor + (1.0 - self.complementarity_floor) * texture_weakness
			)
		gate = self.gate_floor + (1.0 - self.gate_floor) * effective_reliability

		scale, shift = self.film_gen(geo_feat).chunk(2, dim=1)
		gate = gate.permute(0, 2, 3, 1)
		scale = torch.tanh(scale).permute(0, 2, 3, 1) * gate
		shift = shift.permute(0, 2, 3, 1) * gate
		residual = desc * scale + shift
		if self.residual_clip > 0:
			residual = residual.clamp(min=-self.residual_clip, max=self.residual_clip)
		residual_gate = self.residual_gate_floor + (1.0 - self.residual_gate_floor) * effective_reliability
		out = desc + self.residual_gain * residual_gate.permute(0, 2, 3, 1) * residual
		out = self.norm(out)
		return out, effective_reliability, texture_weakness

	def forward(self, desc: torch.Tensor, geo_v: torch.Tensor):
		"""Refine descriptors and estimate geometry reliability."""
		desc_map = desc.permute(0, 3, 1, 2)
		geo_feat = self.geo_proj(geo_v)
		reliability_map = torch.sigmoid(
			self.reliability_head(torch.cat([desc_map, geo_feat], dim=1))
		)
		out, effective, texture = self.refine_with_reliability(desc, geo_feat, reliability_map)
		return out, reliability_map, self._aux(effective, texture)

# --------- Linear Attention with Local Correlation ---------
# 负责空间上下文传播和局部相关性增强
# 线性注意力模块，结合了局部相关性增强和域动态适应。保持了线性全局AFT路径，同时注入多尺度局部结构和局部自相关图。
# 在推理过程中，它根据当前特征统计适应注意力温度、局部分支增益和相关增益，而不是依赖训练域的运行统计。

# 单个注意力机制
class AFTAttention(nn.Module):
	"""
	Grouped AFT with explicit local-correlation awareness and per-sample domain adaptation.

	The block keeps the linear global AFT path, then injects multi-scale local structure
	and local autocorrelation maps. During inference it adapts attention temperature,
	local-branch gain, and correlation gain from the current feature statistics instead
	of relying on training-domain running statistics.
	"""
	def __init__(
		self,
		d_model: int,
		num_groups: int = 4,
		dropout: bool = False,
		p: float = 0.1,
		local_kernel_sizes: Tuple[int, ...] = (3, 5, 7),
		domain_adapt: bool = True,
		temperature_min: float = 0.70,
		temperature_max: float = 1.60,
		local_base_gain: float = 1.0,
		corr_base_gain: float = 0.5,
	) -> None:
		super().__init__()
		self.dim = d_model
		self.num_groups = num_groups
		assert d_model % num_groups == 0, "d_model must be divisible by num_groups"
		self.head_dim = d_model // num_groups
		self.domain_adapt = bool(domain_adapt)
		self.temperature_min = float(temperature_min)
		self.temperature_max = float(max(temperature_max, temperature_min + 1e-3))
		self.local_base_gain = float(local_base_gain)
		self.corr_base_gain = float(corr_base_gain)
		self.local_kernel_sizes = tuple(int(k) for k in local_kernel_sizes if int(k) > 1 and int(k) % 2 == 1)
		if not self.local_kernel_sizes:
			self.local_kernel_sizes = (3, 5)

		self.query = nn.Linear(d_model, d_model)
		self.key = nn.Linear(d_model, d_model)
		self.value = nn.Linear(d_model, d_model)
		self.proj = nn.Linear(d_model, d_model)
		self.use_dropout = dropout
		self.dropout = nn.Dropout(p=p)

		norm_groups = min(8, d_model)
		while d_model % norm_groups != 0:
			norm_groups -= 1

		# Multi-scale depthwise local modeling captures corners, short edges, and wider
		# layout cues without losing the low-cost AFT character.
		self.local_branches = nn.ModuleList([
			nn.Sequential(
				nn.Conv2d(d_model, d_model, kernel_size=k, padding=k // 2, groups=d_model, bias=False),
				nn.GroupNorm(norm_groups, d_model),
				nn.GELU(),
			)
			for k in self.local_kernel_sizes
		])
		self.local_mix = nn.Conv2d(d_model * len(self.local_branches), d_model, kernel_size=1, bias=False)

		# Local autocorrelation maps tell the block where a descriptor agrees with its
		# surrounding layout. This is stronger than a pure convolutional prior in
		# repetitive or low-texture indoor regions.
		self.corr_proj = nn.Sequential(
			nn.Conv2d(len(self.local_kernel_sizes), d_model, kernel_size=1, bias=True),
			nn.Sigmoid(),
		)

		adapt_hidden = max(16, d_model // 4)
		self.domain_adapter = nn.Sequential(
			nn.Linear(d_model * 2, adapt_hidden),
			nn.GELU(),
			nn.Linear(adapt_hidden, 4),
		)
		nn.init.zeros_(self.domain_adapter[-1].weight)
		nn.init.zeros_(self.domain_adapter[-1].bias)

	def _resolve_shape(self, n_tokens: int, shape: Optional[Tuple[int, int]]) -> Tuple[int, int]:
		"""Recover the 2D descriptor grid shape from token count.

		FeatureBooster normally passes the true (H, W). The fallback keeps the block
		usable for square synthetic tests or legacy callers that only provide tokens.
		"""
		if shape is not None:
			H, W = int(shape[0]), int(shape[1])
			if H > 0 and W > 0 and H * W == n_tokens:
				return H, W

		side = int(math.sqrt(n_tokens))
		if side * side == n_tokens:
			return side, side

		best_h = max(1, side)
		for h in range(side, 0, -1):
			if n_tokens % h == 0:
				best_h = h
				break
		return best_h, n_tokens // best_h

	def forward(self, x: torch.Tensor, shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
		"""Apply global grouped AFT plus local-correlation enhancement.

		Args:
			x: Dense descriptors flattened as (B, H*W, C).
			shape: Original descriptor-grid size. Required for non-square maps.
		Returns:
			Refined descriptors with the same shape as x.
		"""
		B, N, C = x.shape
		residual = x
		H, W = self._resolve_shape(N, shape)
		x_grid = x.transpose(1, 2).reshape(B, C, H, W)

		# Per-sample feature statistics act as a lightweight domain descriptor.
		# This lets the block adapt at inference time without changing weights.
		channel_mean = x_grid.mean(dim=(2, 3), keepdim=True)
		channel_std = x_grid.var(dim=(2, 3), unbiased=False, keepdim=True).add(1e-6).sqrt()
		x_norm_grid = (x_grid - channel_mean) / channel_std
		domain_stats = torch.cat([
			channel_mean.flatten(1),
			channel_std.flatten(1),
		], dim=1)

		if self.domain_adapt:
			adapt = torch.tanh(self.domain_adapter(domain_stats))
		else:
			adapt = x.new_zeros(B, 4)

		# Adapter outputs:
		#   0 -> attention temperature, 1 -> local gain,
		#   2 -> correlation gain,      3 -> normalized/raw feature blend.
		temperature = (1.0 + 0.35 * adapt[:, 0]).clamp(self.temperature_min, self.temperature_max)
		local_gain = (self.local_base_gain * (1.0 + 0.75 * adapt[:, 1])).clamp(0.05, 2.50)
		corr_gain = (self.corr_base_gain * (1.0 + 0.75 * adapt[:, 2])).clamp(0.00, 2.00)
		if self.domain_adapt:
			norm_mix = torch.sigmoid(adapt[:, 3]).view(B, 1, 1, 1)
		else:
			norm_mix = x.new_zeros(B, 1, 1, 1)

		x_domain_grid = (1.0 - norm_mix) * x_grid + norm_mix * x_norm_grid
		x_domain = x_domain_grid.flatten(2).transpose(1, 2)

		# 1. Global Context Branch (Grouped AFT).
		# Softmax is over spatial tokens, so every location receives a global
		# key-value summary while groups keep the cost below full self-attention.
		q = self.query(x_domain).view(B, N, self.num_groups, self.head_dim)
		k = self.key(x_domain).view(B, N, self.num_groups, self.head_dim)
		v = self.value(x_domain).view(B, N, self.num_groups, self.head_dim)

		# Linear Attention normalization over spatial tokens (N)
		k_attn = torch.softmax(k / temperature.view(B, 1, 1, 1), dim=1)

		# Global Context Pool (B, 1, G, D)
		kv = (k_attn * v).sum(dim=1, keepdim=True)

		# Global context modulation
		x_global = (q * kv).view(B, N, C)

		# 2. Local Correlation Branch (Spatial Inductive Bias).
		# Depthwise multi-kernel convs model nearby layouts; correlation maps then
		# amplify neighborhoods whose descriptors agree with their local average.
		local_feats = [branch(x_domain_grid) for branch in self.local_branches]
		x_local_grid = self.local_mix(torch.cat(local_feats, dim=1))

		corr_source = F.normalize(x_domain_grid, p=2, dim=1, eps=1e-6)
		corr_maps = []
		for k_size in self.local_kernel_sizes:
			local_avg = F.avg_pool2d(corr_source, kernel_size=k_size, stride=1, padding=k_size // 2)
			corr_maps.append((corr_source * local_avg).sum(dim=1, keepdim=True))
		corr_gate = self.corr_proj(torch.cat(corr_maps, dim=1))
		x_local_grid = x_local_grid * (1.0 + corr_gain.view(B, 1, 1, 1) * corr_gate)
		x_local = x_local_grid.flatten(2).transpose(1, 2)

		# 3. Dynamic Fusion of Global & Local representations.
		# local_gain changes per image, which helps indoor/outdoor or low/high
		# texture domains use different global-vs-local balances.
		fused = x_global + local_gain.view(B, 1, 1) * x_local

		x = self.proj(fused)
		if self.use_dropout:
			x = self.dropout(x)
		x += residual
		return x

# FFN 前向传播模块
class FeedForwardNetwork(nn.Module):
	def __init__(self, feature_dim: int, dropout: bool = False, p: float = 0.1) -> None:
		super().__init__()
		# weights_0326 checkpoints use 2x FFN hidden expansion (64 -> 128 -> 64).
		self.mlp = MLP([feature_dim, feature_dim*2, feature_dim])
		# self.layer_norm = nn.LayerNorm(feature_dim, eps=1e-6)
		self.use_dropout = dropout
		self.dropout = nn.Dropout(p=p)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		residual = x
		x = self.mlp(x)
		if self.use_dropout:
			x = self.dropout(x)
		x += residual
		# x = self.layer_norm(x)
		return x

# Attention + FFN 残差块
class AFTBlock(nn.Module):
	"""AFT attention followed by a residual feed-forward network."""

	def __init__(self, feature_dim: int, dropout: bool = False, p: float = 0.1):
		super().__init__()
		self.attn = AFTAttention(
			feature_dim,
			num_groups=4,
			dropout=dropout,
			p=p,
			local_kernel_sizes=(3, 5),
			domain_adapt=True,
			temperature_min=0.7,
			temperature_max=1.6,
			local_base_gain=1.0,
			corr_base_gain=0.5,
		)
		self.ffn = FeedForwardNetwork(feature_dim, dropout=dropout, p=p)


	def forward(self, x: torch.Tensor, shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
		x = self.attn(x, shape=shape)
		x = self.ffn(x)
		return x

# 多个AFTBlock堆叠
class AttentionalNN(nn.Module):
	def __init__(self, feature_dim: int, layer_num: int, dropout: bool = False, p: float = 0.1) -> None:
		super().__init__()
		self.layers = nn.ModuleList()

		for _ in range(layer_num):
			self.layers.append(AFTBlock(feature_dim, dropout=dropout, p=p))

	def forward(self, desc: torch.Tensor, shape: Optional[Tuple[int,int]] = None) -> torch.Tensor:
		for layer in self.layers:
			desc = layer(desc, shape=shape)
		return desc

class FeatureBooster(nn.Module):
	"""Fuse visual descriptors with geometry, then refine them with AFT attention.

	Input and output descriptors are dense maps. Internally, descriptor/geometry
	channels are moved to NHWC for MLP-style encoders, then flattened to tokens for
	AFTAttention, and finally converted back to NCHW for downstream sampling.
	"""
	def __init__(self, model_config, dropout=False, p=0.1):
		super().__init__()
		# GeoFeat's geometry path is part of the default model definition.
		# Keep the fields internally for downstream modules, but do not require
		# them in the external YAML model configuration.
		model_config = dict(model_config)
		ablation = model_config['ablation']
		model_config['geometric_features'] = {
			'depth': not ablation['disable_geo_head'],
			'gradients': not ablation['disable_geo_head'],
			'normals': not ablation['disable_geo_head'],
			'curvatures': not ablation['disable_geo_head'],
		}
		model_config['refiner_type'] = 'GCDR' if not ablation['disable_gcdr'] else 'None'
		self.use_dropout = dropout
		self.dropout = nn.Dropout(p=p)
		descriptor_dim = model_config['descriptor_dim']
		self.denc = DescriptorEncoder(descriptor_dim, dropout=dropout)

		# Geometric features setup
		geo_dims = {
			'depth': 1,
			'gradients': 2,
			'normals': 3,
			'curvatures': 2,
		}
		self.geo_encoders = nn.ModuleDict()
		self.geo_branch_gates = nn.ModuleDict()
		self.geo_specs = []

		geo_cfg = model_config['geometric_features']
		geo_input_dim = 0
		for name, input_dim in geo_dims.items():
			if geo_cfg[name]:
				self.geo_encoders[name] = GeometricEncoder(
					input_dim,
					descriptor_dim,
					[32, 48],
					dropout=dropout
				)
				self.geo_branch_gates[name] = nn.Sequential(
					nn.Linear(input_dim, 8),
					nn.ReLU(inplace=True),
					nn.Linear(8, 1),
				)
				self.geo_specs.append((name, input_dim))
				geo_input_dim += input_dim

		self.geo_reliability_gate = None
		if geo_input_dim:
			self.geo_reliability_gate = GeometryReliabilityGate(
				geo_dim=geo_input_dim,
				feature_dim=descriptor_dim,
				hidden_dim=12,
				gate_floor=0.2,
				bias_init=2.0,
				fusion_gain=0.5,
			)

		self.attn_proj = AttentionalNN(descriptor_dim, layer_num=2, dropout=dropout)
		self.repeatability_enabled = not ablation['disable_repeatability']
		self.repeatability_head = nn.Sequential(
			nn.Conv2d(descriptor_dim, descriptor_dim, kernel_size=3, padding=1, groups=descriptor_dim, bias=False),
			nn.Conv2d(descriptor_dim, 1, kernel_size=1, bias=True),
			nn.Sigmoid(),
		) if self.repeatability_enabled else None
		# Descriptors are normalized directly after the attention refinement.
		# There is no configurable final activation in the current model.
		refiner = GCDR if model_config['refiner_type'] == 'GCDR' else None
		self.pre_attn_refiner = None
		if refiner is not None:
			self.pre_attn_refiner = refiner(
				dim=descriptor_dim,
				geo_dim=geo_input_dim,
				geo_compress=12,
				reliability_hidden=12,
				gate_floor=0.1,
				residual_gain=0.35,
				residual_gate_floor=0.05,
				residual_clip=0.1,
				complementarity_enabled=False,
				texture_kernel_size=5,
				texture_temperature=8.0,
				complementarity_floor=0.25,
			)

	def encode_geometry(self, geo_v):
		"""Fuse independent NHWC geometry branches with per-pixel competition."""
		parts = geo_v.split([dim for _, dim in self.geo_specs], dim=-1)
		encoded, scores = [], []
		for (name, _), part in zip(self.geo_specs, parts):
			encoded.append(self.geo_encoders[name](part))
			scores.append(self.geo_branch_gates[name](part))
		# Mean gain remains one, while relative gains compete at every location.
		weights = len(scores) * torch.softmax(torch.cat(scores, dim=-1), dim=-1)
		memory = sum(feat * weights[..., i:i+1] for i, feat in enumerate(encoded))
		conditioned = torch.cat(
			[part * weights[..., i:i+1] for i, part in enumerate(parts)],
			dim=-1,
		)
		return memory, conditioned, weights.permute(0, 3, 1, 2)

	def estimate_geometry_reliability(self, geo_map, geo_memory):
		"""Estimate spatial reliability from raw and encoded geometry."""
		if self.use_dropout:
			geo_memory = self.dropout(geo_memory)
		return self.geo_reliability_gate(geo_map, geo_memory)

	def forward(self, desc, kpts, geo_v=None, shape=None):
		"""Run descriptor enhancement.

		Args:
			desc: Raw dense descriptors, (B, C, H, W).
			kpts: Detector logits. Kept for API compatibility; current mainline
				does not inject keypoint logits into the booster.
			geo_v: Geometry tensor aligned or alignable to descriptor resolution.
			shape: Optional spatial shape override for attention tokens.
		"""
		# Assume inputs are always 4D dense maps: (B, C, H, W)
		_, _, H, W = desc.shape
		geo_reliability_map = None
		geo_memory_reliability_map = None
		geo_branch_weights = None
		gcdr_aux = {}

		# Ensure geo_v matches desc spatial resolution
		if geo_v is not None and (geo_v.shape[2] != H or geo_v.shape[3] != W):
			geo_v = F.interpolate(geo_v, size=(H, W), mode='bilinear', align_corners=False)

		# Permute for linear encoding layers: (B, H, W, C)
		desc_lin = desc.permute(0, 2, 3, 1)
		geo_v_lin = geo_v.permute(0, 2, 3, 1) if geo_v is not None else None
		geo_conditioning = geo_v

		# 1. Descriptor Encodings
		d_encoded = self.denc(desc_lin)
		if self.use_dropout:
			d_encoded = self.dropout(d_encoded)

		# 2a. Encode geometry and fuse its branches spatially.
		if geo_v is not None and self.geo_specs:
			geo_encoded_sum, geo_conditioning_lin, geo_branch_weights = self.encode_geometry(geo_v_lin)
			geo_conditioning = geo_conditioning_lin.permute(0, 3, 1, 2)

			# 2b. Estimate spatial reliability from the fused geometry.
			if self.geo_reliability_gate is not None:
				geo_memory_reliability_map = self.estimate_geometry_reliability(geo_conditioning, geo_encoded_sum)

		fused_desc = d_encoded

		# 3. Apply reliability-gated GCDR residual refinement before attention.
		if self.pre_attn_refiner is not None and geo_conditioning is not None:
			if geo_memory_reliability_map is not None:
				fused_desc, geo_reliability_map, gcdr_aux = self.pre_attn_refiner.forward_with_reliability(
					fused_desc,
					geo_conditioning,
					geo_memory_reliability_map,
				)
			else:
				fused_desc, geo_reliability_map, gcdr_aux = self.pre_attn_refiner(fused_desc, geo_conditioning)
		desc = fused_desc

		# 4. AFT descriptor context refinement
		B_curr, H_curr, W_curr, C_curr = desc.shape
		desc_flat = desc.reshape(B_curr, H_curr * W_curr, C_curr)
		desc_flat = self.attn_proj(desc_flat, shape=(H_curr, W_curr))
		desc = desc_flat.reshape(B_curr, H_curr, W_curr, C_curr)

		# 5. Final projection path
		# Convert back to NCHW for final output
		desc_map = desc.permute(0, 3, 1, 2)
		# Normalize descriptors along the channel dimension.
		desc_map = F.normalize(desc_map, p=2, dim=1)
		repeat_map = self.repeatability_head(desc_map) if self.repeatability_head is not None else None

		aux_outputs = {}
		if geo_branch_weights is not None:
			aux_outputs['geo_branch_weights'] = geo_branch_weights
		if geo_memory_reliability_map is not None:
			aux_outputs['geo_memory_reliability_map'] = geo_memory_reliability_map
		if geo_reliability_map is not None:
			aux_outputs['geo_reliability_map'] = geo_reliability_map
		if geo_reliability_map is not None and geo_memory_reliability_map is not None:
			aux_outputs['geo_joint_reliability_map'] = (geo_reliability_map * geo_memory_reliability_map).sqrt()
		if gcdr_aux:
			effective_reliability = gcdr_aux['effective_reliability_map']
			texture_weakness = gcdr_aux['texture_weakness_map']
			if effective_reliability is not None:
				aux_outputs['geo_effective_reliability_map'] = effective_reliability
			if texture_weakness is not None:
				aux_outputs['texture_weakness_map'] = texture_weakness

		return desc_map, repeat_map, aux_outputs

class GeoFeatModel(nn.Module):
	"""Main GeoFeat network.

	The model is intentionally split into two public stages:
	- forward1(image): CNN backbone + dense heads, producing raw keypoint,
	  descriptor, and geometry maps.
	- forward2(raw maps): geometry-aware descriptor refinement and repeatability
	  based keypoint-logit refinement.
	This split lets evaluation code inspect raw and refined predictions separately.
	"""
	def __init__(self, model_config):
		super().__init__()
		self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		if model_config['backbone'] == "Standard":
			self.backbone = StandardBackbone()
		elif model_config['backbone'] == "RepVGG":
			self.backbone = RepVGGBackbone()
		else:
			raise ValueError(f"Unknown backbone: {model_config['backbone']}")
		x3_channels, x4_channels, x5_channels = self.backbone.output_channels
		self.norm = nn.InstanceNorm2d(1)

		# FPN-style top-down fusion: x5 -> x4 -> x3.
		self.upsample4 = UpsampleLayer(x4_channels)
		self.upsample5 = UpsampleLayer(x5_channels)

		self.conv_fusion45 = nn.Conv2d(
			x5_channels // 2 + x4_channels,
			x4_channels,
			kernel_size=3,
			stride=1,
			padding=1,
		)
		self.conv_fusion34 = nn.Conv2d(
			x4_channels // 2 + x3_channels,
			x3_channels,
			kernel_size=3,
			stride=1,
			padding=1,
		)

		# Dense prediction heads on the fused stride-8 feature map.
		self.keypoint_head = KeypointHead(
			in_channels=x3_channels,
			out_channels=model_config['keypoint_dim'],
		)
		# descriptor
		self.descriptor_head = DescriptorHead(
			in_channels=x3_channels,
			out_channels=model_config['descriptor_dim'],
		)
		# geometric features
		self.geo_config = model_config['geometric_features']
		self.geo_head = None if not any(bool(v) for v in self.geo_config.values()) else GeoHead(in_channels=x3_channels, geo_cfg=self.geo_config)
		self.repeatability_enabled = model_config['repeatability_enabled']
		# # heatmap
		# self.heatmap_head = HeatmapHead(in_channels=c3, mid_channels=c3,out_channels=1)

		self.repeatability_proj = nn.Conv2d(1, model_config['keypoint_dim'], kernel_size=1, bias=True) if self.repeatability_enabled else None

		self.attn_fusion = FeatureBooster(model_config)

	def fuse_multi_features(self,x3,x4,x5):
		# upsample x5 feature
		x5 = self.upsample5(x5)
		if x5.shape[-2:] != x4.shape[-2:]:		# Pooling rounding may cause deviation, align to x4 before stitching
			x5 = F.interpolate(x5, size=x4.shape[-2:], mode='bilinear', align_corners=False)
		x4 = torch.cat([x4,x5],dim=1)
		x4 = self.conv_fusion45(x4)

		# upsample x4 feature
		x4 = self.upsample4(x4)
		if x4.shape[-2:] != x3.shape[-2:]:		# Pooling rounding may cause deviation, align to x3 before stitching
			x4 = F.interpolate(x4, size=x3.shape[-2:], mode='bilinear', align_corners=False)
		x3 = torch.cat([x3,x4],dim=1)
		x = self.conv_fusion34(x3)
		return x

	def forward1(self, x):
		"""
			Stage 1: image-to-dense-map prediction.

			Args:
				x: Input image tensor, grayscale or RGB, (B, C, H, W).
			Returns:
				keypoint_map: Detector logits, (B, keypoint_dim, H/8, W/8).
				des_map: Raw dense descriptors, (B, descriptor_dim, H/8, W/8).
				geo_map: Geometry map at image-like resolution before forward2
					preprocessing. Its channel count follows MODEL.GEOMETRIC_FEATURES.
		"""
		with torch.no_grad():
			# GeoFeat operates on grayscale intensity. InstanceNorm reduces image
			# exposure/domain shifts before entering the CNN backbone.
			x = x.mean(dim=1, keepdim = True)
			x = self.norm(x)

		x3, x4, x5 = self.backbone(x)

		# features fusion
		x = self.fuse_multi_features(x3, x4, x5)

		# keypoint
		keypoint_map = self.keypoint_head(x)
		# descriptor
		des_map = self.descriptor_head(x)
		# geometry
		geo_map = self.geo_head(x) if self.geo_head is not None else None
		# # heatmap
		# heatmap = self.heatmap_head(x)

		return keypoint_map, des_map, geo_map

	def forward2(self, keypoint_map, des_map, geo_map, return_logits: bool = True, return_aux: bool = False):
		"""Refine descriptors and blend keypoint logits with repeatability."""
		if geo_map is not None:
			geo_map = F.interpolate(
				geo_map,
				size=des_map.shape[-2:],
				mode='bilinear',
				align_corners=False,
			)

		desc, repeat_map, aux_outputs = self.attn_fusion(des_map, keypoint_map, geo_map)
		kpt_logits = keypoint_map
		if self.repeatability_enabled:
			repeat_logits = self.repeatability_proj(repeat_map)
			alpha = 0.35
			if self.attn_fusion.pre_attn_refiner is not None:
				# Unfloored reliability limits feedback; ablations keep visual-only fusion.
				alpha = alpha * aux_outputs['geo_reliability_map']
			kpt_logits = (1.0 - alpha) * keypoint_map + alpha * repeat_logits

		if return_aux:
			aux_outputs = dict(aux_outputs)
			# Training callers omit repeatability supervision when this key is absent.
			if repeat_map is not None:
				aux_outputs['repeat_map'] = repeat_map
			aux_outputs['kpt_logits'] = kpt_logits
			if return_logits:
				return desc, kpt_logits, aux_outputs
			return desc, aux_outputs

		if return_logits:
			return desc, kpt_logits
		return desc

	def forward(self,x):
		"""Convenience full forward used by legacy callers."""
		keypoint_map, des_map, geo_map = self.forward1(x)
		desc = self.forward2(keypoint_map, des_map, geo_map, return_logits=False)
		return desc, keypoint_map, des_map, geo_map


class GeoFeat(pl.LightningModule):
	"""Lightning training module configured by ``src/config/train.yaml``."""

	def __init__(self, backbone: str, keypoint_dim: int, descriptor_dim: int,
				 optimizer: dict[str, Any], scheduler: dict[str, Any],
				 ablation: dict[str, bool]) -> None:
		super().__init__()
		self.save_hyperparameters()
		disable_geo_head = ablation["disable_geo_head"]
		disable_gcdr = ablation["disable_gcdr"] or disable_geo_head
		disable_repeatability = ablation["disable_repeatability"] or disable_gcdr
		effective_ablation = {
			"disable_geo_head": disable_geo_head,
			"disable_gcdr": disable_gcdr,
			"disable_repeatability": disable_repeatability,
		}
		self.model_config = {
			"backbone": backbone,
			"keypoint_dim": keypoint_dim,
			"descriptor_dim": descriptor_dim,
			"ablation": effective_ablation,
			"refiner_type": "GCDR" if not disable_gcdr else "None",
			"geometric_features": {
				"depth": not disable_geo_head,
				"gradients": not disable_geo_head,
				"normals": not disable_geo_head,
				"curvatures": not disable_geo_head,
			},
			"repeatability_enabled": not disable_repeatability,
		}
		self.optimizer_config = optimizer
		self.scheduler_config = scheduler
		self.net = GeoFeatModel(self.model_config)
		self.loss_fn: GeoFeatLoss | None = None
		self.depth_net: DepthAnythingExtractor | None = None
		self.dam_geometry_extractor: Any = None
		self.alike_net: ALikeExtractor | None = None
		self.save_ckpt_every = 1000
		self.use_dual_kpt_loss = True
		self.kpt_raw_weight_start = 0.8
		self.kpt_raw_weight_end = 0.2
		self.kpt_refined_weight_start = 0.2
		self.kpt_refined_weight_end = 0.8
		self.kpt_ramp_steps = 40_000
		self.homography_difficulty_start = 0.10
		self.homography_difficulty_end = 0.10
		self.homography_ramp_steps = 1
		self.tps_probability = 0.70

	def setup(self, stage: str) -> None:
		if stage != "fit":
			return
		device = self.device
		loss_cfg = get_loss_config()
		self.loss_fn = GeoFeatLoss(
			device, self.model_config["geometric_features"],
			lam_descs=1, lam_fb_descs=3, lam_kpts=2, lam_kpts_raw=5,
			lam_heatmap=1,
			lam_reliability=loss_cfg["reliability_loss_weight"] if self.model_config["refiner_type"] == "GCDR" else 0,
			lam_repeatability=loss_cfg["repeatability_loss_weight"] if self.model_config["repeatability_enabled"] else 0,
			lam_depth=loss_cfg["depth_weight"] if self.model_config["geometric_features"]["depth"] else 0,
			lam_gradients=loss_cfg["gradient_weight"] if self.model_config["geometric_features"]["gradients"] or self.model_config["geometric_features"]["normals"] else 0,
			lam_curvature=loss_cfg["curvature_weight"] if self.model_config["geometric_features"]["curvatures"] else 0,
			lam_descriptor_protect=loss_cfg["descriptor_protect_weight"],
			lam_reliable_enhance=loss_cfg["reliable_enhance_weight"],
			lam_unreliable_suppress=loss_cfg["unreliable_suppress_weight"],
			reliable_enhance_margin=loss_cfg["reliable_enhance_margin"],
			reliable_enhance_threshold=loss_cfg["reliable_enhance_threshold"],
			reliable_enhance_quantile=loss_cfg["reliable_enhance_quantile"],
			unreliable_suppress_threshold=loss_cfg["unreliable_suppress_threshold"],
			unreliable_suppress_quantile=loss_cfg["unreliable_suppress_quantile"],
			reliability_consistency_weight=loss_cfg["reliability_consistency_weight"],
			reliability_target_scale=loss_cfg["reliability_target_scale"],
			geo_loss_reliability_weighted=loss_cfg["geo_loss_reliability_weighted"],
			geo_loss_unreliable_floor=loss_cfg["geo_loss_unreliable_floor"],
		).to(device)
		self.depth_net = DepthAnythingExtractor("vits", device, 256)
		self.dam_geometry_extractor = GeometricExtractor(
			model_config=self.model_config, device=device, enable_depth_extractor=False
		).to(device).eval()
		self.alike_net = ALikeExtractor("alike-t", device).eval()
		self.alike_net.requires_grad_(False)

	def _data_module(self):
		return self.trainer.datamodule

	def _kpt_weights(self, step: int) -> tuple[float, float]:
		if not self.use_dual_kpt_loss or step > self.kpt_ramp_steps:
			return 0.2, 0.8
		ratio = min(max(step - 1, 0) / max(self.kpt_ramp_steps - 1, 1), 1.0)
		raw = self.kpt_raw_weight_start + ratio * (self.kpt_raw_weight_end - self.kpt_raw_weight_start)
		refined = self.kpt_refined_weight_start + ratio * (self.kpt_refined_weight_end - self.kpt_refined_weight_start)
		return raw / (raw + refined), refined / (raw + refined)

	def _build_training_batch(self):
		dm = self._data_module()
		images0, images1, np0, np1, positives = [], [], [], [], []
		if dm.use_coco:
			coco0, coco1, h0, h1 = dm.make_coco_batch(
				self.homography_difficulty_start, self.tps_probability
			)
			hc, wc = coco0.shape[-2] // 8, coco0.shape[-1] // 8
			_, coco_pos = dm.get_corresponding_points(coco0, coco1, h0, h1, hc, wc)
			images0.append(coco0.mean(1, keepdim=True))
			images1.append(coco1.mean(1, keepdim=True))
			positives += coco_pos
		if dm.use_megadepth:
			batch = dm.next_megadepth_batch()
			for key in batch:
				if isinstance(batch[key], torch.Tensor):
					batch[key] = batch[key].to(self.device)
			images0.append(batch["image0"].mean(1, keepdim=True))
			images1.append(batch["image1"].mean(1, keepdim=True))
			for index in range(batch["image0_np"].shape[0]):
				np0.append(batch["image0_np"][index].squeeze(0).cpu().numpy())
				np1.append(batch["image1_np"][index].squeeze(0).cpu().numpy())
			positives += megadepth_wrapper.spvs_coarse(batch, 8)
		return torch.cat(images0), torch.cat(images1), np0, np1, positives

	def _forward_pair(self, image0: torch.Tensor, image1: torch.Tensor):
		key0, desc0, geo0 = self.net.forward1(image0)
		key1, desc1, geo1 = self.net.forward1(image1)
		feedback0, feedback1 = [], []
		refined0, refined1 = [], []
		reliability0, reliability1 = [], []
		repeatability0, repeatability1 = [], []
		alike0, alike1 = [], []

		for index in range(desc0.shape[0]):
			output0 = self.net.forward2(
				key0[index:index + 1], desc0[index:index + 1],
				geo0[index:index + 1] if geo0 is not None else None,
				return_logits=True, return_aux=True,
			)
			output1 = self.net.forward2(
				key1[index:index + 1], desc1[index:index + 1],
				geo1[index:index + 1] if geo1 is not None else None,
				return_logits=True, return_aux=True,
			)
			feedback0.append(output0[0])
			refined0.append(output0[1])
			feedback1.append(output1[0])
			refined1.append(output1[1])
			self._append_auxiliary_outputs(output0[2], reliability0, repeatability0)
			self._append_auxiliary_outputs(output1[2], reliability1, repeatability1)

			image0_np = image0[index].permute(1, 2, 0).expand(-1, -1, 3).detach().cpu().numpy() * 255
			image1_np = image1[index].permute(1, 2, 0).expand(-1, -1, 3).detach().cpu().numpy() * 255
			alike0.append(torch.as_tensor(self.alike_net.extract_alike_kpts(image0_np), device=self.device))
			alike1.append(torch.as_tensor(self.alike_net.extract_alike_kpts(image1_np), device=self.device))

		return {
			"raw_keypoints0": key0,
			"raw_keypoints1": key1,
			"raw_descriptors0": desc0,
			"raw_descriptors1": desc1,
			"geometry0": geo0,
			"geometry1": geo1,
			"feedback0": torch.cat(feedback0),
			"feedback1": torch.cat(feedback1),
			"refined0": torch.cat(refined0),
			"refined1": torch.cat(refined1),
			"reliability0": torch.cat(reliability0) if reliability0 else None,
			"reliability1": torch.cat(reliability1) if reliability1 else None,
			"repeatability0": torch.cat(repeatability0) if repeatability0 else None,
			"repeatability1": torch.cat(repeatability1) if repeatability1 else None,
			"alike0": alike0,
			"alike1": alike1,
		}

	@staticmethod
	def _append_auxiliary_outputs(auxiliary: dict[str, torch.Tensor], reliability: list, repeatability: list) -> None:
		if "geo_joint_reliability_map" in auxiliary:
			reliability.append(auxiliary["geo_joint_reliability_map"])
		elif "geo_reliability_map" in auxiliary:
			reliability.append(auxiliary["geo_reliability_map"])
		if "repeat_map" in auxiliary:
			repeatability.append(auxiliary["repeat_map"])

	def _build_loss_info(self, outputs: dict[str, Any], positives: list, geometry_targets0: list, geometry_targets1: list):
		dm = self._data_module()
		raw_weight, refined_weight = self._kpt_weights(self.global_step + 1)
		return self.loss_fn(
			outputs["raw_descriptors0"], outputs["feedback0"], outputs["refined0"], outputs["geometry0"],
			outputs["raw_descriptors1"], outputs["feedback1"], outputs["refined1"], outputs["geometry1"],
			positives, outputs["alike0"], outputs["alike1"], [], [], geometry_targets0, geometry_targets1,
			dm.megadepth_batch_size, dm.coco_batch_size,
			reliability1=outputs["reliability0"], reliability2=outputs["reliability1"],
			repeatability1=outputs["repeatability0"], repeatability2=outputs["repeatability1"],
			raw_kpts1=outputs["raw_keypoints0"], raw_kpts2=outputs["raw_keypoints1"],
			refined_kpts1=outputs["refined0"], refined_kpts2=outputs["refined1"],
			use_dual_kpt_loss=True, kpt_raw_weight=raw_weight,
			kpt_refined_weight=refined_weight, kpt_single_source="raw",
		)

	def _loss_terms(self, loss_info: dict[str, torch.Tensor]) -> list[torch.Tensor]:
		loss_names = ["loss_descs", "loss_fb_descs", "loss_kpts", "loss_descriptor_protect"]
		if self.model_config["refiner_type"] == "GCDR":
			loss_names += ["loss_reliability", "loss_reliable_enhance", "loss_unreliable_suppress"]
		if self.model_config["repeatability_enabled"]:
			loss_names.append("loss_repeatability")
		if self.model_config["geometric_features"]["depth"]:
			loss_names.append("loss_depths")
		if self.model_config["geometric_features"]["gradients"] or self.model_config["geometric_features"]["normals"]:
			loss_names.append("loss_gradients")
		if self.model_config["geometric_features"]["curvatures"]:
			loss_names.append("loss_curvature")
		return [loss_info[name] for name in loss_names]

	def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
		del batch, batch_idx
		images0, images1, np0, np1, positives = self._build_training_batch()
		if any(len(points) < 30 for points in positives):
			return torch.zeros((), device=self.device, requires_grad=True)
		outputs = self._forward_pair(images0, images1)
		if self.model_config["geometric_features"]["depth"]:
			geometry_targets0 = [self.depth_net.extract(image, geometric_extractor=self.dam_geometry_extractor) for image in np0]
			geometry_targets1 = [self.depth_net.extract(image, geometric_extractor=self.dam_geometry_extractor) for image in np1]
		else:
			geometry_targets0 = []
			geometry_targets1 = []
		loss_info = self._build_loss_info(outputs, positives, geometry_targets0, geometry_targets1)
		loss = torch.stack(self._loss_terms(loss_info)).mean()
		self.log("loss", loss, on_step=True, on_epoch=False, logger=True, prog_bar=True)
		self.log("Step", self.global_step + 1, on_step=True, on_epoch=False, logger=False, prog_bar=True)
		self.log("Loss/total", loss, on_step=True, on_epoch=False, logger=True)
		self.log("acc_fb_coarse", loss_info["acc_fb_coarse"], on_step=True, on_epoch=False, logger=True)
		return loss

	def configure_optimizers(self):
		optimizer_name = self.optimizer_config["name"]
		optimizer = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[optimizer_name](self.net.parameters(), lr=self.optimizer_config["lr"])
		scheduler_name = self.scheduler_config["name"]
		scheduler = {"StepLR": torch.optim.lr_scheduler.StepLR}[scheduler_name](optimizer, step_size=self.scheduler_config["step_size"], gamma=self.scheduler_config["gamma"])
		return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


if __name__ == "__main__":
	import sys
	sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
	from src.config.config import get_cfg_defaults

	img_path = os.path.join(os.path.dirname(__file__), '../../assert/ref.jpg')
	# Setup configuration from the central config file
	cfg = get_cfg_defaults()

	model_config = get_model_config(cfg)

	# Image loading and processing
	img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	if img is None:
		print(f"Failed to load image from {img_path}")
		# fallback or exit
		# exit(1) # Commented out to allow partial execution if needed, or handle gracefully
		# Creating a dummy image for testing if file missing
		img = np.zeros((640, 800), dtype=np.uint8)

	img = cv2.resize(img, (800, 640))
	img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()/255.0
	img = img.cuda() if torch.cuda.is_available() else img

	geofeat_sp=GeoFeatModel(model_config=model_config).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
	keypoint_map, des_map, geo_map = geofeat_sp.forward1(img)
	des_fine=geofeat_sp.forward2(keypoint_map, des_map, geo_map, return_logits=False)
	if isinstance(des_fine, tuple):
		des_fine = des_fine[0]
	print(f"Output descriptor shape: {des_fine.shape}")
