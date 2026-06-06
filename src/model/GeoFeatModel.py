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

import sys
from typing import List, Tuple, Optional
from src.config.config import get_cfg_defaults, get_model_config


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

# The original keypoint sampling is incorrect. We patch it here but
# keep the original one above for legacy.
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
	def __init__(self, c1=24, c2=24, c3=64, c4=64, c5=128):
		super().__init__()
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
	def __init__(self, c1=24, c2=24, c3=64, c4=64, c5=128):
		super().__init__()
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
	def __init__(self,in_channels,out_channels):
		super().__init__()
		self.layer1=BaseLayer(in_channels,32)
		self.layer2=BaseLayer(32,32)
		self.layer3=BaseLayer(32,64)

		self.conv=nn.Conv2d(64,out_channels,kernel_size=3,stride=1,padding=1)
		self.bn=nn.BatchNorm2d(out_channels)

	def forward(self,x):
		x=self.layer1(x)
		x=self.layer2(x)
		x=self.layer3(x)
		x=self.bn(self.conv(x))
		return x

class DescriptorHead(nn.Module):
	"""Dense descriptor head that predicts one descriptor vector per feature cell."""
	def __init__(self,in_channels,out_channels):
		super().__init__()
		# Compact descriptor head for the sub-1M-parameter mainline.
		self.layer=nn.Sequential(
			BaseLayer(in_channels, 32),
			BaseLayer(32, 64, activation=True),
			BaseLayer(64, out_channels, activation=False)
		)

	def forward(self,x):
		x=self.layer(x)
		# x=nn.functional.softmax(x,dim=1)
		return x

class HeatmapHead(nn.Module):
	def __init__(self,in_channels,mid_channels,out_channels):
		super().__init__()
		self.convHa = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
		self.bnHa = nn.BatchNorm2d(mid_channels)
		self.convHb = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
		self.bnHb = nn.BatchNorm2d(out_channels)
		self.leaky_relu = nn.LeakyReLU(0.1)

	def forward(self,x):
		x = self.leaky_relu(self.bnHa(self.convHa(x)))
		x = self.leaky_relu(self.bnHb(self.convHb(x)))

		x = torch.sigmoid(x)
		return x

class GeoHead(nn.Module):
	"""LiftFeat-style geometry head with dynamic output branches.

	Depending on config, it predicts depth, first-order gradients, and/or
	curvature-like channels. Mainline gradients include
	(dx, dy, normal_x, normal_y, normal_z). The branch order is fixed because
	FeatureBooster later slices the concatenated geometry tensor in the same order.
	"""
	def __init__(self, in_channels: int, geo_cfg: dict):
		super().__init__()
		# Track branch names and per-branch channel counts for selective normalization
		branch_defs = [
			('depth', 1),
			('gradients', 5),
			('curvatures', 1),
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
			out_ch = 1  # fallback to avoid empty head
			self._branch_names = ['depth']
			self._branch_channels = [1]

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

		# Selective normalization: normalize the normal sub-vector embedded in
		# the gradients branch.
		parts = x.split(self._branch_channels, dim=1)
		out = []
		for i, name in enumerate(self._branch_names):
			p = parts[i]
			if name == 'gradients' and p.shape[1] >= 5:
				grad_xy = p[:, :2]
				normal_xyz = F.normalize(p[:, 2:5], p=2, dim=1)
				p = torch.cat([grad_xy, normal_xyz], dim=1)
			out.append(p)
		return torch.cat(out, dim=1)

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
	def __init__(self, keypoint_dim: int, feature_dim: int, layers: List[int], dropout: bool = False, p: float = 0.1) -> None:
		super().__init__()
		self.encoder = MLP([keypoint_dim] + layers + [feature_dim])
		self.use_dropout = dropout
		self.dropout = nn.Dropout(p=p)

	def forward(self, kpts):
		if self.use_dropout:
			return self.dropout(self.encoder(kpts))
		return self.encoder(kpts)

class DescriptorEncoder(nn.Module):
	""" Encoding of visual descriptor using MLP """
	def __init__(self, feature_dim: int, layers: List[int], dropout: bool = False, p: float = 0.1) -> None:
		super().__init__()
		self.encoder = MLP([feature_dim] + layers + [feature_dim])
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

# 负责 2D descriptor 主动读取 3D geometry
class GroupedLinearGeoCrossAttention(nn.Module):
	"""Linear cross-attention from descriptor queries to geometry memory.

	Descriptors provide Q; encoded geometry provides K/V. The attention is grouped
	and linear over spatial tokens, matching the cost profile of AFT instead of
	full O(N^2) cross-attention.
	"""

	def __init__(
		self,
		d_model: int,
		num_groups: int = 4,
		dropout: bool = False,
		p: float = 0.1,
		gate_init: float = -2.0,
	) -> None:
		super().__init__()
		assert d_model % num_groups == 0, "d_model must be divisible by num_groups"
		self.d_model = d_model
		self.num_groups = int(num_groups)
		self.head_dim = d_model // self.num_groups
		self.query = nn.Linear(d_model, d_model)
		self.key = nn.Linear(d_model, d_model)
		self.value = nn.Linear(d_model, d_model)
		self.proj = nn.Linear(d_model, d_model)
		self.norm = nn.LayerNorm(d_model)
		self.use_dropout = dropout
		self.dropout = nn.Dropout(p=p)
		self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

	def forward(self, desc_tokens: torch.Tensor, geo_tokens: torch.Tensor) -> torch.Tensor:
		"""Return geometry-conditioned descriptor tokens.

		Args:
			desc_tokens: (B, N, C), descriptor query tokens.
			geo_tokens: (B, N, C), geometry memory tokens.
		"""
		if desc_tokens.shape != geo_tokens.shape:
			raise RuntimeError(
				"GroupedLinearGeoCrossAttention expects desc_tokens and geo_tokens "
				f"to have the same shape, got {tuple(desc_tokens.shape)} and {tuple(geo_tokens.shape)}."
			)

		B, N, C = desc_tokens.shape
		q = self.query(desc_tokens).view(B, N, self.num_groups, self.head_dim)
		k = self.key(geo_tokens).view(B, N, self.num_groups, self.head_dim)
		v = self.value(geo_tokens).view(B, N, self.num_groups, self.head_dim)

		k_attn = torch.softmax(k, dim=1)
		geo_context = (k_attn * v).sum(dim=1, keepdim=True)
		cross = (q * geo_context).reshape(B, N, C)
		cross = self.proj(cross)
		if self.use_dropout:
			cross = self.dropout(cross)

		gate = torch.sigmoid(self.gate_logit)
		return self.norm(desc_tokens + gate * cross)

# 负责根据几何可靠性调制 descriptor
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
	):
		super().__init__()
		self.dim = dim
		self.gate_floor = float(max(0.0, min(gate_floor, 1.0)))
		self.residual_gain = float(max(0.0, residual_gain))
		self.residual_gate_floor = float(max(0.0, min(residual_gate_floor, 1.0)))

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
		rel_out = self.reliability_head[-1]
		if isinstance(rel_out, nn.Conv2d):
			nn.init.zeros_(rel_out.weight)
			if rel_out.bias is not None:
				nn.init.zeros_(rel_out.bias)

		self.norm = nn.LayerNorm(dim)

	def forward(self, desc: torch.Tensor, geo_v: torch.Tensor):
		"""Refine NHWC descriptors using NCHW geometry.

		Args:
			desc: Descriptor map in (B, H, W, C), matching linear encoder format.
			geo_v: Geometry map in (B, C_geo, H, W).
		Returns:
			A refined descriptor map in NHWC and a reliability map in NCHW.
		"""
		desc_map = desc.permute(0, 3, 1, 2)
		geo_feat = self.geo_proj(geo_v)

		# Reliability is estimated from both current descriptors and compressed
		# geometry, then reused as a gate for all geometry-conditioned residuals.
		reliability_logits = self.reliability_head(torch.cat([desc_map, geo_feat], dim=1))
		reliability_map = torch.sigmoid(reliability_logits)
		gate = self.gate_floor + (1.0 - self.gate_floor) * reliability_map

		film = self.film_gen(geo_feat)
		scale, shift = film.chunk(2, dim=1)
		scale = torch.tanh(scale) * gate
		shift = shift * gate

		scale = scale.permute(0, 2, 3, 1)
		shift = shift.permute(0, 2, 3, 1)

		residual = desc * scale + shift
		residual_gate = self.residual_gate_floor + (1.0 - self.residual_gate_floor) * reliability_map
		residual_gate = residual_gate.permute(0, 2, 3, 1)
		out = desc + (self.residual_gain * residual_gate) * residual
		out = self.norm(out)
		return out, reliability_map

# --------- Linear Attention with Local Correlation & Domain-Dynamic Adaptation ---------
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
	"""AFT attention with selectable FFN (PFFN or Swin)."""

	def __init__(self, feature_dim: int, model_config: Optional[dict] = None, dropout: bool = False, p: float = 0.1):
		super().__init__()
		aft_cfg = dict((model_config or {}).get('aft_config', {}))
		self.attn = AFTAttention(
			feature_dim,
			num_groups=int(aft_cfg.get('num_groups', 4)),
			dropout=dropout,
			p=p,
			local_kernel_sizes=tuple(aft_cfg.get('local_kernel_sizes', (3, 5, 7))),
			domain_adapt=bool(aft_cfg.get('domain_adapt', True)),
			temperature_min=float(aft_cfg.get('temperature_min', 0.70)),
			temperature_max=float(aft_cfg.get('temperature_max', 1.60)),
			local_base_gain=float(aft_cfg.get('local_base_gain', 1.0)),
			corr_base_gain=float(aft_cfg.get('corr_base_gain', 0.5)),
		)
		self.ffn = FeedForwardNetwork(feature_dim, dropout=dropout, p=p)


	def forward(self, x: torch.Tensor, shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
		x = self.attn(x, shape=shape)
		x = self.ffn(x)
		return x

# 多个AFTBlock堆叠
class AttentionalNN(nn.Module):
	def __init__(self, feature_dim: int, layer_num: int, model_config: dict, dropout: bool = False, p: float = 0.1) -> None:
		super().__init__()
		self.model_config = model_config
		self.layers = nn.ModuleList()

		for _ in range(layer_num):
			self.layers.append(AFTBlock(feature_dim, model_config=model_config, dropout=dropout, p=p))

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
		self.model_config = model_config
		self.geo_preprocess_mode = self.model_config.get('geo_preprocess_mode', 'unfold')
		self.use_dropout = dropout
		self.dropout = nn.Dropout(p=p)

		# self.kenc = KeypointEncoder(self.config['keypoint_dim'], self.config['descriptor_dim'], self.config['keypoint_encoder'], dropout=dropout)

		self.denc = DescriptorEncoder(self.model_config['descriptor_dim'], self.model_config['descriptor_encoder'], dropout=dropout)

		# Geometric features setup
		self.geo_raw_dims = {
			'depth': 1,
			'gradients': 5,
			'curvatures': 1,
		}
		self.geo_encoders = nn.ModuleDict()
		self.geo_branch_gates = nn.ModuleDict()
		self.geo_specs = []

		geo_cfg = self.model_config['geometric_features']
		# Order must match GeoHead: depth -> gradients -> curvatures.
		# Mainline gradients are [dx, dy, normal_x, normal_y, normal_z].
		features_map = [
			('depth', 'depth_encoder'),
			('gradients', 'gradient_encoder'),
			('curvatures', 'curvature_encoder'),
		]

		geo_input_dim = 0
		self.geo_preprocess_mode = self.model_config['geo_preprocess_mode']
		geo_token_mult = 64 if self.geo_preprocess_mode == 'unfold' else 1

		for name, enc_key in features_map:
			if geo_cfg[name]:
				input_dim = self.geo_raw_dims[name] * geo_token_mult

				# Each geometry branch is encoded independently to descriptor_dim.
				# The branch gate below suppresses unreliable or low-information
				# branches on a per-image basis before summing them.
				self.geo_encoders[name] = GeometricEncoder(
					input_dim,
					self.model_config['descriptor_dim'], # Target dimension
					self.model_config[enc_key],
					dropout=dropout
				)
				gate_hidden = max(8, input_dim // 8)
				gate_mlp = nn.Sequential(
					nn.Linear(input_dim, gate_hidden),
					nn.ReLU(inplace=True),
					nn.Linear(gate_hidden, 1),
					nn.Sigmoid(),
				)
				# Backward-compatible default: when old checkpoints do not contain
				# geo_branch_gates parameters, keep branch contribution near identity.
				gate_proj = gate_mlp[2]
				if isinstance(gate_proj, nn.Linear):
					nn.init.zeros_(gate_proj.weight)
					nn.init.constant_(gate_proj.bias, 4.0)
				self.geo_branch_gates[name] = gate_mlp
				self.geo_specs.append((name, input_dim)) # Store raw_dim for slicing
				geo_input_dim += input_dim

		cross_cfg = dict(self.model_config.get('geo_cross_attention', {}))
		self.use_geo_cross_attention = bool(cross_cfg.get('enabled', True))
		self.geo_cross_attn = None
		if self.use_geo_cross_attention:
			self.geo_cross_attn = GroupedLinearGeoCrossAttention(
				self.model_config['descriptor_dim'],
				num_groups=int(cross_cfg.get('num_groups', 4)),
				dropout=dropout,
				p=p,
				gate_init=float(cross_cfg.get('gate_init', -2.0)),
			)

		self.attn_proj = AttentionalNN(feature_dim=self.model_config['descriptor_dim'], layer_num=self.model_config['attention_layers'], model_config=self.model_config, dropout=dropout)
		self.repeatability_head = nn.Sequential(
			nn.Conv2d(self.model_config['descriptor_dim'], self.model_config['descriptor_dim'], kernel_size=3, padding=1, groups=self.model_config['descriptor_dim'], bias=False),
			nn.Conv2d(self.model_config['descriptor_dim'], 1, kernel_size=1, bias=True),
			nn.Sigmoid(),
		)

		# [Refiner Integration] Select Refiner based on config
		# Modes: 'None', 'GCDR'
		self.refiner_type = model_config['refiner_type']
		self.pre_attn_refiner = None  # Pre-attention refiner slot (GCDR)
		total_geo_channels = sum(
			dim for name, dim in self.geo_raw_dims.items()
			if model_config['geometric_features'][name]
		)
		if self.refiner_type == 'None':
			pass
		elif self.refiner_type == 'GCDR':
			# GCDR inserts BEFORE AttentionalNN: geometry conditions descriptors
			# early so attention builds on geometry-aware representations.
			gcdr_cfg = dict(self.model_config.get('gcdr_config', {}))
			self.pre_attn_refiner = GCDR(
				dim=self.model_config['descriptor_dim'],
				geo_dim=total_geo_channels * geo_token_mult,
				geo_compress=int(gcdr_cfg.get('geo_compress', 8)),
				reliability_hidden=int(gcdr_cfg.get('reliability_hidden', 16)),
				gate_floor=float(gcdr_cfg.get('gate_floor', 0.10)),
				residual_gain=float(gcdr_cfg.get('residual_gain', 1.0)),
				residual_gate_floor=float(gcdr_cfg.get('residual_gate_floor', 0.10)),
			)
		else:
			raise ValueError(
				f"Unsupported REFINER_TYPE '{self.refiner_type}'. "
				"Expected one of: 'None', 'GCDR'. "
				"Check MODEL.REFINER_TYPE in src/config/config.py or your --config_file override."
			)

	def validate_refiner_configuration(self):
		refiner_type = str(self.model_config.get('refiner_type', 'None'))
		if refiner_type == 'GCDR':
			if self.pre_attn_refiner is None:
				raise RuntimeError(
					"Config requests REFINER_TYPE='GCDR', but pre_attn_refiner is None. "
					"Check MODEL.REFINER_TYPE / MODEL.GCDR in src/config/config.py or your merged YAML override, "
					"then verify GCDR wiring in GeoFeatModel.py."
				)
		elif refiner_type != 'None':
			raise RuntimeError(
				f"Config requests unsupported REFINER_TYPE='{refiner_type}'. Only 'None' and 'GCDR' remain wired in GeoFeatModel.py."
			)

		# Final activation
		if self.model_config['last_activation'] == 'None':
			self.last_activation = None
		elif self.model_config['last_activation'] == 'relu':
			self.last_activation = nn.ReLU()
		elif self.model_config['last_activation'] == 'sigmoid':
			self.last_activation = nn.Sigmoid()
		elif self.model_config['last_activation'] == 'tanh':
			self.last_activation = nn.Tanh()
		else:
			raise Exception('Not supported activation "%s".' % self.model_config['last_activation'])

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
		B, C, H, W = desc.shape
		geo_reliability_map = None

		# Ensure geo_v matches desc spatial resolution
		if geo_v is not None and (geo_v.shape[2] != H or geo_v.shape[3] != W):
			geo_v = F.interpolate(geo_v, size=(H, W), mode='bilinear', align_corners=False)

		# Permute for linear encoding layers: (B, H, W, C)
		desc_lin = desc.permute(0, 2, 3, 1)
		geo_v_lin = geo_v.permute(0, 2, 3, 1) if geo_v is not None else None

		# 1. Descriptor Encodings
		d_encoded = self.denc(desc_lin)
		if self.use_dropout:
			d_encoded = self.dropout(d_encoded)

		# 2. Geometric Feature Fusion
		if geo_v is not None and self.geo_specs:
			if geo_v_lin is None:
				raise RuntimeError("geo_v is provided but geo_v_lin is None")
			geo_encoded_sum = torch.zeros_like(d_encoded)
			start_idx = 0
			for name, dim in self.geo_specs:
				# geo_specs stores the same branch order as GeoHead output, so this
				# slice recovers one semantic geometry branch at a time.
				feat_v = geo_v_lin[..., start_idx : start_idx + dim]
				encoded = self.geo_encoders[name](feat_v)
				branch_stats = feat_v.mean(dim=(1, 2))
				branch_gate = self.geo_branch_gates[name](branch_stats).view(-1, 1, 1, 1)
				geo_encoded_sum = geo_encoded_sum + encoded * branch_gate
				start_idx += dim

			if self.use_dropout:
				geo_encoded_sum = self.dropout(geo_encoded_sum)

			if self.geo_cross_attn is not None:
				desc_tokens = d_encoded.reshape(B, H * W, C)
				geo_tokens = geo_encoded_sum.reshape(B, H * W, C)
				fused_desc = self.geo_cross_attn(desc_tokens, geo_tokens).reshape(B, H, W, C)
			else:
				fused_desc = d_encoded + geo_encoded_sum
		else:
			fused_desc = d_encoded

		# 3. LiftFeat baseline path: no keypoint encoder injection
		# GCDR: pre-attention geometric conditioning (operates in NHWC format)
		if self.pre_attn_refiner is not None and geo_v is not None:
			fused_desc, geo_reliability_map = self.pre_attn_refiner(fused_desc, geo_v)
		desc = fused_desc

		# 4. Cross Attention Boosting
		B_curr, H_curr, W_curr, C_curr = desc.shape
		desc_flat = desc.reshape(B_curr, H_curr * W_curr, C_curr)
		desc_flat = self.attn_proj(desc_flat, shape=(H_curr, W_curr))
		desc = desc_flat.reshape(B_curr, H_curr, W_curr, C_curr)

		# 5. Final projection path
		# Convert back to NCHW for final output
		desc_map = desc.permute(0, 3, 1, 2)

		# Final projection / activation
		if self.last_activation is not None:
			desc_map = self.last_activation(desc_map)

		# Baseline LiftFeat enforces L2 normalization before returning descriptors
		# L2 norm along channel dim=1 because desc_map is (B, C, H, W)
		desc_map = F.normalize(desc_map, p=2, dim=1)
		repeatability_map = self.repeatability_head(desc_map)

		aux_outputs = {}
		if geo_reliability_map is not None:
			aux_outputs['geo_reliability_map'] = geo_reliability_map

		return desc_map, repeatability_map, aux_outputs

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
		self.model_config = model_config
		self.geo_config = self.model_config['geometric_features']

		# Backbone extracts a three-level pyramid. Later fusion brings x4/x5 back
		# to x3 resolution, so all heads predict on the same stride-8 grid.
		self.backbone = self.model_config['backbone']
		# Balanced compact backbone: keep the full model under 1M parameters while
		# preserving enough mid-level capacity for stable keypoints/descriptors.
		c1, c2, c3, c4, c5 = 20, 20, 40, 80, 96
		if self.backbone == "Standard":
			self.feature_extract = StandardBackbone(c1, c2, c3, c4, c5)
		elif self.backbone == "RepVGG":
			self.feature_extract = RepVGGBackbone(c1, c2, c3, c4, c5)
		else:
			raise ValueError(f"Unknown backbone: {self.backbone}")
		self.norm = nn.InstanceNorm2d(1)

		# FPN-style top-down fusion: x5 -> x4 -> x3.
		self.upsample4 = UpsampleLayer(c4)
		self.upsample5 = UpsampleLayer(c5)

		self.conv_fusion45 = nn.Conv2d(c5//2+c4,c4,kernel_size=3,stride=1,padding=1)
		self.conv_fusion34 = nn.Conv2d(c4//2+c3,c3,kernel_size=3,stride=1,padding=1)

		# Dense prediction heads on the fused stride-8 feature map.
		self.keypoint_dim = self.model_config['keypoint_dim']
		self.keypoint_head = KeypointHead(in_channels=c3, out_channels=self.keypoint_dim)
		# descriptor
		self.descriptor_dim = self.model_config['descriptor_dim']
		self.descriptor_head = DescriptorHead(in_channels=c3, out_channels=self.descriptor_dim)
		# geometric features
		self.geo_head = GeoHead(in_channels=c3, geo_cfg=self.geo_config)
		# # heatmap
		# self.heatmap_head = HeatmapHead(in_channels=c3, mid_channels=c3,out_channels=1)

		# repeatability_alpha blends detector logits with repeatability logits.
		# Optional adaptive knobs are injected by wrapper/evaluation code.
		self.repeatability_alpha = self.model_config['repeatability_alpha']
		self.repeatability_adaptive_gamma = float(self.model_config.get('repeatability_adaptive_gamma', 0.0))
		self.descriptor_confidence_gain = float(self.model_config.get('descriptor_confidence_gain', 0.0))
		self.geo_preprocess_mode = self.model_config['geo_preprocess_mode']
		self.repeatability_proj = nn.Conv2d(1, self.keypoint_dim, kernel_size=1, bias=True)

		self.attn_fusion = FeatureBooster(model_config)
		self.attn_fusion.validate_refiner_configuration()

	def fuse_multi_features(self,x3,x4,x5):
		"""Fuse backbone pyramid features into one stride-8 feature map."""
		# Upsample x5 to x4 resolution and fuse semantic context into x4.
		x5 = self.upsample5(x5)
		# align spatial size to x4 to avoid cat mismatch when input H/W not divisible by 32
		if x5.shape[-2:] != x4.shape[-2:]:
			x5 = F.interpolate(x5, size=x4.shape[-2:], mode='bilinear', align_corners=False)
		x4 = torch.cat([x4, x5], dim=1)
		x4 = self.conv_fusion45(x4)

		# Upsample fused x4 to x3 resolution for dense heads.
		x4 = self.upsample4(x4)
		# align spatial size to x3 before concat
		if x4.shape[-2:] != x3.shape[-2:]:
			x4 = F.interpolate(x4, size=x3.shape[-2:], mode='bilinear', align_corners=False)
		x3 = torch.cat([x3, x4], dim=1)
		x = self.conv_fusion34(x3)
		return x

	def _unfold2d(self, x, ws = 2):
		"""
			Unfold a geometry/image-level tensor into non-overlapping windows.

			For ws=8, (B, C, H, W) becomes (B, C*64, H/8, W/8), preserving local
			geometry patches as channel tokens aligned with descriptor cells.
		"""
		B, C, H, W = x.shape
		x = x.unfold(2,  ws , ws).unfold(3, ws,ws).reshape(B, C, H//ws, W//ws, ws**2)
		return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H//ws, W//ws)


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

		x3, x4, x5 = self.feature_extract(x)

		# features fusion
		x = self.fuse_multi_features(x3, x4, x5)

		# keypoint
		keypoint_map = self.keypoint_head(x)
		# descriptor
		des_map = self.descriptor_head(x)
		# geometry
		geo_map = self.geo_head(x)
		# # heatmap
		# heatmap = self.heatmap_head(x)

		return keypoint_map, des_map, geo_map

	def forward2(self, keypoint_map, des_map, geo_map, return_refined_keypoints: bool = True, return_aux: bool = False):
		"""Stage 2: geometry-aware descriptor/keypoint refinement.

		Args:
			keypoint_map: Raw detector logits from forward1.
			des_map: Raw dense descriptor map from forward1.
			geo_map: Geometry map from forward1.
			return_refined_keypoints: If true, also return refined detector logits.
			return_aux: If true, include repeatability/reliability diagnostic maps.
		"""
		# Pass feature maps directly to FeatureBooster:
		# - des_map: (B, C_desc, H/8, W/8)
		# - keypoint_map: (B, C_kpt, H/8, W/8)
		# - geo_map is either unfolded to descriptor resolution or resized directly.
		# no_unfold_lite is preferred when geometry already includes normals,
		# gradients, and curvatures. The legacy unfold path was designed to enrich
		# depth-only geometry; with richer derivative geometry it can add redundant
		# channels and noise.
		if self.geo_preprocess_mode == 'unfold':
			# Legacy path: keep each 8x8 geometry patch as channels aligned to one
			# descriptor cell. More expressive, but heavier and noisier.
			geo_map = self._unfold2d(geo_map, ws=8)
		else:
			# Mainline path: geometry derivatives are already semantically dense, so
			# direct resizing is cheaper and avoids redundant patch channels.
			geo_map = F.interpolate(
				geo_map,
				size=des_map.shape[-2:],
				mode='bilinear',
				align_corners=False,
			)

		descs_refine, repeatability_map, aux_outputs = self.attn_fusion(des_map, keypoint_map, geo_map)

		def _repeatability_confidence(rep_map, gamma):
			"""Convert repeatability prediction into a confidence map for blending."""
			if gamma <= 0:
				return torch.sigmoid(rep_map)
			centered = rep_map - rep_map.mean(dim=(2, 3), keepdim=True)
			return torch.sigmoid(centered * gamma)

		confidence_map = _repeatability_confidence(
			repeatability_map,
			self.repeatability_adaptive_gamma,
		)

		if self.descriptor_confidence_gain > 0:
			# Optional inference-time safeguard: blend refined descriptors with raw
			# descriptors where repeatability confidence is low.
			raw_desc_map = F.normalize(des_map, p=2, dim=1)
			weight_map = (confidence_map * self.descriptor_confidence_gain).clamp(0.0, 1.0)
			descs_refine = F.normalize((1.0 - weight_map) * raw_desc_map + weight_map * descs_refine, p=2, dim=1)

		repeatability_logits = self.repeatability_proj(repeatability_map)
		alpha = float(self.repeatability_alpha)
		if self.repeatability_adaptive_gamma > 0:
			# Adaptive alpha makes repeatability influence stronger only where its
			# confidence is high, preserving raw detector logits elsewhere.
			alpha_map = (alpha * confidence_map).clamp(0.0, 1.0)
			keypoint_logits_refined = (1.0 - alpha_map) * keypoint_map + alpha_map * repeatability_logits
		else:
			keypoint_logits_refined = (1.0 - alpha) * keypoint_map + alpha * repeatability_logits

		if return_aux:
			aux_outputs = dict(aux_outputs)
			aux_outputs['repeatability_map'] = repeatability_map
			aux_outputs['repeatability_confidence_map'] = confidence_map
			aux_outputs['keypoint_logits_refined'] = keypoint_logits_refined
			if return_refined_keypoints:
				return descs_refine, keypoint_logits_refined, aux_outputs
			return descs_refine, aux_outputs

		if return_refined_keypoints:
			return descs_refine, keypoint_logits_refined
		return descs_refine

	def forward(self,x):
		"""Convenience full forward used by legacy callers."""
		keypoint_map, des_map, geo_map = self.forward1(x)
		descs_refine = self.forward2(keypoint_map, des_map, geo_map, return_refined_keypoints=True)
		return descs_refine, des_map, geo_map, keypoint_map


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
	des_fine=geofeat_sp.forward2(keypoint_map, des_map, geo_map, return_refined_keypoints=False)
	if isinstance(des_fine, tuple):
		des_fine = des_fine[0]
	print(f"Output descriptor shape: {des_fine.shape}")
