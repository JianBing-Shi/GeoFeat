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
from src.config.config import get_cfg_defaults


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
class StandardBackbone(nn.Module):
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
		eps = bn.epss

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
			std = torch.sqrt(bn.running_var + bn.eps)
			gamma = bn.weight
			beta = bn.bias
			running_mean = bn.running_mean
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
		self.rbr_reparam = nn.Conv2d(self.in_channels, self.out_channels, 3, self.stride, padding=1, bias=True)
		self.rbr_reparam.weight.data = kernel
		self.rbr_reparam.bias.data = bias

		# delete unneeded branches
		delattr(self, 'rbr_dense')
		delattr(self, 'rbr_1x1')
		if self.rbr_identity is not None:
			delattr(self, 'rbr_identity')
		self.deploy = True

class RepVGGBackbone(nn.Module):
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
class UpsampleLayer(nn.Module):
	def __init__(self, in_channels):
		super().__init__()
		# 定义特征提取层，减少通道数同时增加特征提取能力
		self.conv = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1)
		# 使用BN层
		self.bn = nn.BatchNorm2d(in_channels//2)
		# 使用LeakyReLU激活函数
		self.leaky_relu = nn.LeakyReLU(0.1)

	def forward(self, x):
		x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
		x = self.leaky_relu(self.bn(self.conv(x)))

		return x

# --------- Head Classes ---------
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
	def __init__(self,in_channels,out_channels):
		super().__init__()
		self.layer1=BaseLayer(in_channels,32)
		self.layer2=BaseLayer(32,32)
		self.layer3=BaseLayer(32,64)
		self.layer4=BaseLayer(64,64)
		self.layer5=BaseLayer(64,128)
		
		self.conv=nn.Conv2d(128,out_channels,kernel_size=3,stride=1,padding=1)
		self.bn=nn.BatchNorm2d(65)
		
	def forward(self,x):
		x=self.layer1(x)
		x=self.layer2(x)
		x=self.layer3(x)
		x=self.layer4(x)
		x=self.layer5(x)
		x=self.bn(self.conv(x))
		return x

class DescriptorHead(nn.Module):
	def __init__(self,in_channels,out_channels):
		super().__init__()
		self.layer=nn.Sequential(
			BaseLayer(in_channels,32),
			BaseLayer(32,32,activation=False),
			BaseLayer(32,64,activation=False),
			BaseLayer(64,out_channels,activation=False)
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
	"""LiftFeat-style depth/geo head with dynamic channels and projection to desc dim."""
	def __init__(self, in_channels: int, geo_cfg: dict):
		super().__init__()
		# Track branch names and per-branch channel counts for selective normalization
		branch_defs = [
			('depth', 1),
			('normal', 3),
			('gradients', 2),
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
		
		# Selective normalization: only normalize the normal branch
		parts = x.split(self._branch_channels, dim=1)
		out = []
		for i, name in enumerate(self._branch_names):
			p = parts[i]
			if name == 'normal':
				p = F.normalize(p, p=2, dim=1)
			out.append(p)
		return torch.cat(out, dim=1)																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																								   

# --------- Encoder Classes ---------
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

# --------- Linear Attention ---------
class AFTAttention(nn.Module):
	""" Attention-free attention """
	def __init__(self, d_model: int, dropout: bool = False, p: float = 0.1) -> None:
		super().__init__()
		self.dim = d_model
		self.query = nn.Linear(d_model, d_model)
		self.key = nn.Linear(d_model, d_model)
		self.value = nn.Linear(d_model, d_model)
		self.proj = nn.Linear(d_model, d_model)
		self.use_dropout = dropout
		self.dropout = nn.Dropout(p=p)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		residual = x
		q = self.query(x)
		k = self.key(x)
		v = self.value(x)
		# q = torch.sigmoid(q)
		# k = k.T
		# k = torch.softmax(k, dim=-1)
		# k = k.T
		k = torch.softmax(k, dim=-2)
		kv = (k * v).sum(dim=-2, keepdim=True)
		x = q * kv
		x = self.proj(x)
		if self.use_dropout:
			x = self.dropout(x)
		x += residual
		# x = self.layer_norm(x)
		return x

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

class AFTBlock(nn.Module):
	"""AFT attention with selectable FFN (PFFN or Swin)."""

	def __init__(self, feature_dim: int, dropout: bool = False, p: float = 0.1):
		super().__init__()
		self.attn = AFTAttention(feature_dim, dropout=dropout, p=p)
		self.ffn = FeedForwardNetwork(feature_dim, dropout=dropout, p=p)


	def forward(self, x: torch.Tensor, shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
		x = self.attn(x)
		x = self.ffn(x)
		return x

class AttentionalNN(nn.Module):
	def __init__(self, feature_dim: int, layer_num: int, model_config: dict, dropout: bool = False, p: float = 0.1) -> None:
		super().__init__()
		self.model_config = model_config
		self.layers = nn.ModuleList()
		
		for _ in range(layer_num):
			self.layers.append(AFTBlock(feature_dim, dropout=dropout, p=p))

	def forward(self, desc: torch.Tensor, shape: Optional[Tuple[int,int]] = None) -> torch.Tensor:
		for layer in self.layers:
			desc = layer(desc)
		return desc

# --------- LocalRefiner ---------
# 通用特征增强模块，使用局部细化模块
class LocalRefiner(nn.Module):
	"""
	Local Refinement Module using Depthwise Separable Convolution.
	Injects local inductive bias after global attention.
	"""
	def __init__(self, dim, k=3):
		super().__init__()
		self.dw = nn.Conv2d(dim, dim, kernel_size=k, stride=1, padding=k//2, groups=dim)
		self.norm = nn.LayerNorm(dim)
		self.pw = nn.Conv2d(dim, dim, kernel_size=1)
		self.act = nn.GELU()

	def forward(self, x):
		# x: (B, C, H, W) or (B, N, C) if we handle reshaping
		input_x = x
		x = self.dw(x)
		x = x.permute(0, 2, 3, 1) # (B, H, W, C)
		x = self.norm(x)
		x = x.permute(0, 3, 1, 2) # (B, C, H, W)
		x = self.act(x)
		x = self.pw(x)
		return input_x + x


class GCDR(nn.Module):
	"""Geo-Conditional Descriptor Refinement with reliability-aware fusion.

	Geometry first predicts a per-location reliability score, and that score
	controls how strongly the FiLM modulation is allowed to change descriptors.
	This turns GCDR from unconditional geometry injection into trust-aware fusion.
	"""

	def __init__(
		self,
		dim: int,
		geo_dim: int,
		geo_compress: int = 8,
		reliability_hidden: int = 16,
		gate_floor: float = 0.10,
	):
		super().__init__()
		self.dim = dim
		self.gate_floor = float(max(0.0, min(gate_floor, 1.0)))

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
		desc_map = desc.permute(0, 3, 1, 2)
		geo_feat = self.geo_proj(geo_v)

		reliability_logits = self.reliability_head(torch.cat([desc_map, geo_feat], dim=1))
		reliability_map = torch.sigmoid(reliability_logits)
		gate = self.gate_floor + (1.0 - self.gate_floor) * reliability_map

		film = self.film_gen(geo_feat)
		scale, shift = film.chunk(2, dim=1)
		scale = torch.tanh(scale) * gate
		shift = shift * gate

		scale = scale.permute(0, 2, 3, 1)
		shift = shift.permute(0, 2, 3, 1)

		out = desc * (1.0 + scale) + shift
		out = self.norm(out)
		return out, reliability_map


class GCLA(nn.Module):
	"""Geometric-Coupled Local Aggregation v2 (GCLA-v2).
	
	Enhanced version with:
	- Multi-scale convolution towers (3x3, 5x5, 7x7)
	- Learnable scale weighting
	- Learnable cross-modal coupling gates (replacing fixed addition)
	
	Fuses 2D spatial features and 3D geometric features through
	multi-scale parallel aggregation with adaptive inter-modal weighting.
	"""
	def __init__(
		self,
		dim,
		geo_dim,
		geo_compress=16,
		k=3,
		k_scales=None,
		use_residual_scale=False,
		residual_init=0.1,
		symmetric_coupling=False,
		use_stack_sum=False,
		gate_hidden=32,
	):
		super().__init__()
		if k_scales is None:
			k_scales = [3, 5, 7]  # Multi-scale receptive fields
		
		self.dim = dim
		self.geo_compress = geo_compress
		self.k_scales = k_scales
		self.symmetric_coupling = bool(symmetric_coupling)
		self.use_stack_sum = bool(use_stack_sum)
		gate_hidden = max(1, int(gate_hidden))
		
		# Compress geometric features for efficient local aggregation
		self.geo_proj = nn.Sequential(
			nn.Conv2d(geo_dim, geo_compress, 1, bias=False),
			nn.BatchNorm2d(geo_compress),
			nn.GELU(),
		)
		
		# Multi-scale depthwise convolutions for spatial features
		self.dw_spatial_scales = nn.ModuleList([
			nn.Conv2d(dim, dim, ks, padding=ks // 2, groups=dim)
			for ks in k_scales
		])
		
		# Multi-scale depthwise convolutions for geometric features
		self.dw_geo_scales = nn.ModuleList([
			nn.Conv2d(geo_compress, geo_compress, ks, padding=ks // 2, groups=geo_compress)
			for ks in k_scales
		])
		
		# Learnable scale weights for spatial and geometric branches
		self.scale_weight_spatial = nn.Parameter(
			torch.ones(len(k_scales)) / len(k_scales)
		)
		self.scale_weight_geo = nn.Parameter(
			torch.ones(len(k_scales)) / len(k_scales)
		)
		
		# Cross-modal coupling (3D to 2D and 2D to 3D)
		self.geo_to_spatial = nn.Conv2d(geo_compress, dim, 1)
		self.spatial_to_geo = nn.Conv2d(dim, geo_compress, 1)
		
		# Learnable coupling gates: how much to trust each modality's contribution
		self.coupling_gate = nn.Sequential(
			nn.Conv2d(dim + geo_compress, gate_hidden, 1),
			nn.ReLU(inplace=True),
			nn.Conv2d(gate_hidden, 2, 1),  # Output: [gate_spatial, gate_geo]
			nn.Sigmoid(),
		)
		
		# Output projection
		self.pw = nn.Conv2d(dim + geo_compress, dim, 1)
		self.norm = nn.LayerNorm(dim)
		self.act = nn.GELU()
		self.residual_scale = None
		if use_residual_scale:
			self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))

	def forward(self, x, geo_map):
		residual = x

		# Compress 3D geometric features
		geo_feat = self.geo_proj(geo_map)

		# Multi-scale spatial aggregation: weighted combination of different receptive fields
		spatial_weights = F.softmax(self.scale_weight_spatial, dim=0)
		if self.use_stack_sum:
			spatial_terms = [
				w * conv(x)
				for w, conv in zip(spatial_weights, self.dw_spatial_scales)
			]
			f_spatial = torch.stack(spatial_terms, dim=0).sum(dim=0)
		else:
			f_spatial = sum(
				w * conv(x)
				for w, conv in zip(spatial_weights, self.dw_spatial_scales)
			)

		# Multi-scale geometric aggregation: weighted combination of different receptive fields
		geo_weights = F.softmax(self.scale_weight_geo, dim=0)
		if self.use_stack_sum:
			geo_terms = [
				w * conv(geo_feat)
				for w, conv in zip(geo_weights, self.dw_geo_scales)
			]
			f_geo = torch.stack(geo_terms, dim=0).sum(dim=0)
		else:
			f_geo = sum(
				w * conv(geo_feat)
				for w, conv in zip(geo_weights, self.dw_geo_scales)
			)

		# Learnable cross-modal coupling (replaces fixed addition)
		coupled = torch.cat([f_spatial, f_geo], dim=1)
		gates = self.coupling_gate(coupled)  # [B, 2, H, W]
		gate_spatial, gate_geo = gates[:, 0:1], gates[:, 1:2]

		# Adaptive coupling: apply learned gates to cross-modal contributions
		if self.symmetric_coupling:
			spatial_update = gate_geo * self.geo_to_spatial(f_geo)
			geo_update = gate_spatial * self.spatial_to_geo(f_spatial)
			f_spatial = f_spatial + spatial_update
			f_geo = f_geo + geo_update
		else:
			f_spatial = f_spatial + gate_geo * self.geo_to_spatial(f_geo)
			f_geo = f_geo + gate_spatial * self.spatial_to_geo(f_spatial)

		# Merge and project
		out = torch.cat([f_spatial, f_geo], dim=1)
		out = self.pw(out)
		out = out.permute(0, 2, 3, 1)
		out = self.norm(out)
		out = out.permute(0, 3, 1, 2)
		out = self.act(out)

		if self.residual_scale is not None:
			return residual + self.residual_scale * out
		return residual + out

class FeatureBooster(nn.Module):
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
			'normal': 3,
			'gradients': 2,
			'curvatures': 1,
		}
		self.geo_encoders = nn.ModuleDict()
		self.geo_branch_gates = nn.ModuleDict()
		self.geo_specs = []
		
		geo_cfg = self.model_config['geometric_features']
		# Order must match GeoHead: depth -> normal -> gradients -> curvatures
		features_map = [
			('depth', 'depth_encoder'),
			('normal', 'normal_encoder'),
			('gradients', 'gradient_encoder'),
			('curvatures', 'curvature_encoder'),
		]

		geo_input_dim = 0
		self.geo_preprocess_mode = self.model_config['geo_preprocess_mode']
		geo_token_mult = 64 if self.geo_preprocess_mode == 'unfold' else 1

		for name, enc_key in features_map:
			if geo_cfg[name]:
				input_dim = self.geo_raw_dims[name] * geo_token_mult
				
				# The GeometricEncoder should take raw_dim as input
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

		self.attn_proj = AttentionalNN(feature_dim=self.model_config['descriptor_dim'], layer_num=self.model_config['attention_layers'], model_config=self.model_config, dropout=dropout)
		self.repeatability_head = nn.Sequential(
			nn.Conv2d(self.model_config['descriptor_dim'], self.model_config['descriptor_dim'], kernel_size=3, padding=1, groups=self.model_config['descriptor_dim'], bias=False),
			nn.Conv2d(self.model_config['descriptor_dim'], 1, kernel_size=1, bias=True),
			nn.Sigmoid(),
		)

		# [Refiner Integration] Select Refiner based on config
		# Modes: 'None', 'Local', 'GCLA', 'GCDR'
		self.refiner_type = model_config['refiner_type']
		self.pre_attn_refiner = None  # Pre-attention refiner slot (GCDR)
		total_geo_channels = sum(
			dim for name, dim in self.geo_raw_dims.items()
			if model_config['geometric_features'][name]
		)
		if self.refiner_type == 'Local':
			self.refiner = LocalRefiner(dim=self.model_config['descriptor_dim'])
		elif self.refiner_type == 'GCLA':
			gcla_cfg = dict(self.model_config.get('gcla_config', {}))
			self.refiner = GCLA(
				dim=self.model_config['descriptor_dim'],
				geo_dim=total_geo_channels * geo_token_mult,
				geo_compress=int(gcla_cfg.get('geo_compress', 16)),
				k_scales=list(gcla_cfg.get('k_scales', [3, 5, 7])),
				use_residual_scale=bool(gcla_cfg.get('use_residual_scale', False)),
				residual_init=float(gcla_cfg.get('residual_init', 0.1)),
				symmetric_coupling=bool(gcla_cfg.get('symmetric_coupling', False)),
				use_stack_sum=bool(gcla_cfg.get('use_stack_sum', False)),
				gate_hidden=int(gcla_cfg.get('gate_hidden', 32)),
			)
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
			)
			self.refiner = None
		else:
			raise ValueError(
				f"Unsupported REFINER_TYPE '{self.refiner_type}'. "
				"Expected one of: 'None', 'Local', 'GCLA', 'GCDR'."
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
				feat_v = geo_v_lin[..., start_idx : start_idx + dim]
				encoded = self.geo_encoders[name](feat_v)
				branch_stats = feat_v.mean(dim=(1, 2))
				branch_gate = self.geo_branch_gates[name](branch_stats).view(-1, 1, 1, 1)
				geo_encoded_sum = geo_encoded_sum + encoded * branch_gate
				start_idx += dim
			
			if self.use_dropout:
				geo_encoded_sum = self.dropout(geo_encoded_sum)

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

		# 5. Local Refiner / GCLA
		# Convert back to NCHW for Conv2d based Refiner and final output
		desc_map = desc.permute(0, 3, 1, 2)
		if self.refiner is not None:
			if self.refiner_type == 'Local':
				desc_map = self.refiner(desc_map)
			elif self.refiner_type == 'GCLA' and geo_v is not None:
				desc_map = self.refiner(desc_map, geo_v)

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
	def __init__(self, model_config):
		super().__init__()
		self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_config = model_config
		self.geo_config = self.model_config['geometric_features']

		# backbone
		self.backbone = self.model_config['backbone']
		c1,c2,c3,c4,c5 = 24,24,64,64,128
		if self.backbone == "Standard":
			self.feature_extract = StandardBackbone(c1, c2, c3, c4, c5)
		elif self.backbone == "RepVGG":
			self.feature_extract = RepVGGBackbone(c1, c2, c3, c4, c5)
		else:
			raise ValueError(f"Unknown backbone: {self.backbone}")
		self.norm = nn.InstanceNorm2d(1)
		
		# feature upsample and fusion
		self.upsample4 = UpsampleLayer(c4)
		self.upsample5 = UpsampleLayer(c5)

		self.conv_fusion45 = nn.Conv2d(c5//2+c4,c4,kernel_size=3,stride=1,padding=1)
		self.conv_fusion34 = nn.Conv2d(c4//2+c3,c3,kernel_size=3,stride=1,padding=1)

		# detector
		self.keypoint_dim = self.model_config['keypoint_dim']
		self.keypoint_head = KeypointHead(in_channels=c3, out_channels=self.keypoint_dim)
		# descriptor
		self.descriptor_dim = self.model_config['descriptor_dim']
		self.descriptor_head = DescriptorHead(in_channels=c3, out_channels=self.descriptor_dim)
		# geometric features
		self.geo_head = GeoHead(in_channels=c3, geo_cfg=self.geo_config)
		# # heatmap
		# self.heatmap_head = HeatmapHead(in_channels=c3, mid_channels=c3,out_channels=1)
		
		self.repeatability_alpha = self.model_config['repeatability_alpha']
		self.geo_preprocess_mode = self.model_config['geo_preprocess_mode']
		self.repeatability_proj = nn.Conv2d(1, self.keypoint_dim, kernel_size=1, bias=True)

		self.attn_fusion = FeatureBooster(model_config)
	
	def fuse_multi_features(self,x3,x4,x5):
		# upsample x5 feature
		x5 = self.upsample5(x5)
		# align spatial size to x4 to avoid cat mismatch when input H/W not divisible by 32
		if x5.shape[-2:] != x4.shape[-2:]:
			x5 = F.interpolate(x5, size=x4.shape[-2:], mode='bilinear', align_corners=False)
		x4 = torch.cat([x4, x5], dim=1)
		x4 = self.conv_fusion45(x4)
		
		# upsample x4 feature
		x4 = self.upsample4(x4)
		# align spatial size to x3 before concat
		if x4.shape[-2:] != x3.shape[-2:]:
			x4 = F.interpolate(x4, size=x3.shape[-2:], mode='bilinear', align_corners=False)
		x3 = torch.cat([x3, x4], dim=1)
		x = self.conv_fusion34(x3)
		return x
	
	def _unfold2d(self, x, ws = 2):
		"""
			Unfolds tensor in 2D with desired ws (window size) and concat the channels
		"""
		B, C, H, W = x.shape
		x = x.unfold(2,  ws , ws).unfold(3, ws,ws).reshape(B, C, H//ws, W//ws, ws**2)
		return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H//ws, W//ws)


	def forward1(self, x):
		"""
			input:
				x -> torch.Tensor(B, C, H, W) grayscale or rgb images
			return:
				feats     ->  torch.Tensor(B, 64, H/8, W/8) dense local features
				keypoints ->  torch.Tensor(B, 65, H/8, W/8) keypoint logit map
				heatmap   ->  torch.Tensor(B,  1, H/8, W/8) reliability map
		"""
		with torch.no_grad():
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
		# Pass feature maps directly to FeatureBooster (which now handles spatial dims)
		# des_map: (B, C, H/8, W/8)
		# geo_map:
		# - unfold mode: (B, C_geo, H, W) -> (B, C_geo*64, H/8, W/8)
		# - no_unfold_lite mode: resize to (B, C_geo, H/8, W/8)
		# keypoint_map: (B, C_k, H/8, W/8)
		# no_unfold 更好的根本原因不是"unfold 有害"，而是"你的几何特征已经使 unfold 变得多余"。
		# unfold 是为 depth-only 场景设计的补偿机制，当你的特征集包含 normal/gradients/curvatures 这些微分几何量时，unfold 从"有用"变成了"冗余+干扰"。
		if self.geo_preprocess_mode == 'unfold':
			geo_map = self._unfold2d(geo_map, ws=8)
		else:
			geo_map = F.interpolate(
				geo_map,
				size=des_map.shape[-2:],
				mode='bilinear',
				align_corners=False,
			)
			
		descs_refine, repeatability_map, aux_outputs = self.attn_fusion(des_map, keypoint_map, geo_map)

		repeatability_logits = self.repeatability_proj(repeatability_map)
		alpha = float(self.repeatability_alpha)
		keypoint_logits_refined = (1.0 - alpha) * keypoint_map + alpha * repeatability_logits

		if return_aux:
			aux_outputs = dict(aux_outputs)
			aux_outputs['repeatability_map'] = repeatability_map
			aux_outputs['keypoint_logits_refined'] = keypoint_logits_refined
			if return_refined_keypoints:
				return descs_refine, keypoint_logits_refined, aux_outputs
			return descs_refine, aux_outputs

		if return_refined_keypoints:
			return descs_refine, keypoint_logits_refined
		return descs_refine
	
	def forward(self,x):
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
	
	model_config = {
		"backbone": cfg.MODEL.BACKBONE,									# 骨干网络
		# 关键点编码器
		'keypoint_encoder': list(cfg.MODEL.KEYPOINT_ENCODER),			# 关键点编码器
		'keypoint_dim': int(cfg.MODEL.KEYPOINT_DIM),					# 关键点维度
		# 描述子编码器
		'descriptor_encoder': list(cfg.MODEL.DESCRIPTOR_ENCODER),		# 描述子编码器
		'descriptor_dim': int(cfg.MODEL.DESCRIPTOR_DIM),				# 描述子维度
		# 几何特征配置
		'geometric_features': {
			'depth': bool(cfg.MODEL.GEOMETRIC_FEATURES.DEPTH),
			'normal': bool(cfg.MODEL.GEOMETRIC_FEATURES.NORMAL),
			'gradients': bool(cfg.MODEL.GEOMETRIC_FEATURES.GRADIENTS),
			'curvatures': bool(cfg.MODEL.GEOMETRIC_FEATURES.CURVATURES),
		},
		# 几何特征编码器
		# 深度编码器
		'depth_encoder': list(cfg.MODEL.DEPTH_ENCODER),
		'depth_dim': int(cfg.MODEL.DEPTH_DIM),
		# 法向量编码器
		'normal_encoder': list(cfg.MODEL.NORMAL_ENCODER),
		'normal_dim': int(cfg.MODEL.NORMAL_DIM),
		# 梯度编码器
		'gradient_encoder': list(cfg.MODEL.GRADIENT_ENCODER),
		'gradient_dim': int(cfg.MODEL.GRADIENT_DIM),
		# 曲率编码器
		'curvature_encoder': list(cfg.MODEL.CURVATURE_ENCODER),
		'curvature_dim': int(cfg.MODEL.CURVATURE_DIM),
		# 几何预处理模式
		'geo_preprocess_mode': str(cfg.MODEL.GEO_PREPROCESS_MODE),
		# 重复性图权重系数
		'repeatability_alpha': float(cfg.MODEL.REPEATABILITY_ALPHA),
		# 注意力机制配置
		"attention_layers": int(cfg.MODEL.ATTENTIONAL_LAYERS),
		"refiner_type": str(cfg.MODEL.REFINER_TYPE),
		'gcla_config': {
			'k_scales': list(cfg.MODEL.GCLA.K_SCALES),
			'geo_compress': int(cfg.MODEL.GCLA.GEO_COMPRESS),
			'use_residual_scale': bool(cfg.MODEL.GCLA.USE_RESIDUAL_SCALE),
			'residual_init': float(cfg.MODEL.GCLA.RESIDUAL_INIT),
			'symmetric_coupling': bool(cfg.MODEL.GCLA.SYMMETRIC_COUPLING),
			'use_stack_sum': bool(cfg.MODEL.GCLA.USE_STACK_SUM),
			'gate_hidden': int(cfg.MODEL.GCLA.GATE_HIDDEN),
		},
		'gcdr_config': {
			'geo_compress': int(cfg.MODEL.GCDR.GEO_COMPRESS),
			'reliability_hidden': int(cfg.MODEL.GCDR.RELIABILITY_HIDDEN),
			'gate_floor': float(cfg.MODEL.GCDR.GATE_FLOOR),
			'reliability_loss_weight': float(cfg.MODEL.GCDR.RELIABILITY_LOSS_WEIGHT),
			'reliability_consistency_weight': float(cfg.MODEL.GCDR.RELIABILITY_CONSISTENCY_WEIGHT),
			'reliability_target_scale': float(cfg.MODEL.GCDR.RELIABILITY_TARGET_SCALE),
		},
		# 细化匹配器配置
		"last_activation": str(cfg.MODEL.LAST_ACTIVATION),
		# 模型输出维度
		'output_dim': int(cfg.MODEL.OUTPUT_DIM),
	}

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
