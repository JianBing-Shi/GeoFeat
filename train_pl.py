"""
GeoFeat: 3D Geometry-Aware Local Feature Matching
PyTorch Lightning Training Script
References: train.py
"""

import sys
import os
import time
import json
import glob
import tqdm
import numpy as np

# Add src to path for imports (compatible with original train.py structure)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset, ConcatDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.config.config import get_cfg_defaults
from model.GeoFeatModel import GeoFeatModel
from loss.loss import GeoFeatLoss
from src.utils.depth_anything_wrapper import DepthAnythingExtractor
from utils.alike_wrapper import ALikeExtractor

from data.coco_augmentor import COCOAugmentor
from data import coco_wrapper
from src.data.megadepth import MegaDepthDataset
from src.data import megadepth_wrapper

# Arg Parsing matching train.py
def parse_arguments():
	import argparse
	parser = argparse.ArgumentParser(description="GeoFeat PL Training")
	parser.add_argument('--name', type=str, default='GeoFeat', 
					 	help='set process name')
	# 保存权重
	parser.add_argument('--ckpt_save_path', type=str, default='weights_win',
						help='Path to save the checkpoints.')
	# 训练步数
	parser.add_argument('--n_steps', type=int, default=160_000,
						help='Number of training steps. Default is 160000.')
	# 学习率
	parser.add_argument('--lr', type=float, default=3e-4,
						help='Learning rate. Default is 0.0003.')
	# 学习率衰减
	parser.add_argument('--gamma_steplr', type=float, default=0.5,
						help='Gamma value for StepLR scheduler. Default is 0.5.')
	# 训练尺寸
	parser.add_argument('--training_res', type=lambda s: tuple(map(int, s.split(','))),
						default=(800, 600), help='Training resolution as width,height. Default is (800, 600).')
	# 训练设备
	parser.add_argument('--device_num', type=str, default='cuda',
						help='Device number to use for training. Default is "cuda:0".')
	# debug模式，快速验证流程(创建虚拟数据集)	
	parser.add_argument('--dry_run', action='store_true',
						help='If set, perform a dry run training with a mini-batch for sanity check.')
	# 每多少步保存权重
	parser.add_argument('--save_ckpt_every', type=int, default=1000,
						help='Save checkpoints every N steps. Default is 1000.')

	# Status 监控参数
	parser.add_argument('--log_interval', type=int, default=1000, help='控制台指标刷新步频(平均)')
	parser.add_argument('--status_interval', type=int, default=0, help='status.txt 写入步频 (0=跟随log_interval)')
	
	# 训练目录管理
	parser.add_argument('--max_runs', type=int, default=30, help='只保留最近 N 次训练目录 (按创建时间)')
	return parser.parse_args()

# -----------------------------------------------------------------------------
# 1. Custom Dataset to mimic train.py's generate_train_data
# -----------------------------------------------------------------------------
class MixedBatchIterDataset(IterableDataset):
	def __init__(self, data_config, training_res, device_num):
		super().__init__()
		self.data_config = data_config
		self.training_res = training_res
		# Note: We prefer to handle device placement inside the PL Module, 
		# but COCOAugmentor in train.py takes a device. 
		# In PL, data loading usually happens on CPU, but COCOAugmentor uses GPU for warping.
		# Ideally we pass the device of the trainer, but here we just follow arguments.
		self.device = torch.device(device_num if torch.cuda.is_available() else 'cpu')
		
		self.use_megadepth = bool(data_config['use_megadepth'])
		self.megadepth_batch_size = int(data_config['megadepth_batch_size']) if self.use_megadepth else 0
		
		self.use_coco = bool(data_config['use_coco'])
		self.coco_batch_size = int(data_config['coco_batch_size']) if self.use_coco else 0
		
		# Initialized in __iter__ to be worker-safe if needed, 
		# but COCOAugmentor involves CUDA tensors so it might need to be on main process 
		# or spawn method. For simplicity, we assume num_workers=0 (main thread) matching train.py.
		self.augmentor = None
		self.megadepth_iter = None
		self.megadepth_loader = None

	def _init_sources(self):
		# COCO Init
		if self.use_coco and self.augmentor is None:
			self.augmentor = COCOAugmentor(
				img_dir=self.data_config['coco_root_path'],
				device=self.device,
				load_dataset=True,
				batch_size=self.coco_batch_size,
				out_resolution=self.training_res,
				warp_resolution=self.training_res,
				sides_crop=0.1,
				max_num_imgs=3000,
				num_test_imgs=5,
				photometric=True,
				geometric=True,
				reload_step=4000
			)

		# MegaDepth Init
		if self.use_megadepth and self.megadepth_loader is None:
			root = self.data_config['megadepth_root_path']
			TRAIN_BASE_PATH = f"{root}/megadepth_indices"
			TRAINVAL_DATA_SOURCE = f"{root}/MegaDepth_v1"
			# Adapting to structure as in train.py (commented out there, but let's be safe or strict)
			# TRAINVAL_DATA_SOURCE = f"{root}/MegaDepth_v1/phoenix/S6/zl548/MegaDepth_v1" 
			TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"
			
			npz_paths = glob.glob(TRAIN_NPZ_ROOT + '/*.npz')[:]
			# Use a simplified list if glob fails or for debugging
			if not npz_paths: 
				print(f"Warning: No npz found in {TRAIN_NPZ_ROOT}")
			
			datasets = [MegaDepthDataset(
					root_dir=TRAINVAL_DATA_SOURCE,
					npz_path=path,
					img_resize=self.training_res) 
				for path in tqdm.tqdm(npz_paths, desc="[MegaDepth] Loading metadata")]
			
			if len(datasets) > 0:
				megadepth_dataset = ConcatDataset(datasets)
				self.megadepth_loader = DataLoader(
					megadepth_dataset, 
					batch_size=self.megadepth_batch_size, 
					shuffle=True,
					num_workers=0 # Keep on main thread to mix safely with COCO CUDA ops
				)
				self.megadepth_iter = iter(self.megadepth_loader)
			else:
				self.use_megadepth = False

	def __iter__(self):
		self._init_sources()
		while True:
			imgs1_t, imgs2_t = [], []
			imgs1_np, imgs2_np = [], []
			positives_coarse = []
			
			# 1. Generate COCO Batch
			if self.use_coco:
				coco_imgs1, coco_imgs2, H1, H2 = coco_wrapper.make_batch(self.augmentor, 0.1)
				h_coarse, w_coarse = coco_imgs1[0].shape[-2] // 8, coco_imgs1[0].shape[-1] // 8
				_, positives_coco_coarse = coco_wrapper.get_corresponding_pts(
					coco_imgs1, coco_imgs2, H1, H2, self.augmentor, h_coarse, w_coarse
				)
				# Normalize: mean(1) to convert to grayscale 1 channel if needed by model input 
				# (GeoFeatModel forward1 usually takes grayscale)
				coco_imgs1 = coco_imgs1.mean(1, keepdim=True)
				coco_imgs2 = coco_imgs2.mean(1, keepdim=True)
				
				imgs1_t.append(coco_imgs1)
				imgs2_t.append(coco_imgs2)
				positives_coarse += positives_coco_coarse
			
			# 2. Generate MegaDepth Batch
			if self.use_megadepth:
				try:
					md_data = next(self.megadepth_iter)
				except StopIteration:
					self.megadepth_iter = iter(self.megadepth_loader)
					md_data = next(self.megadepth_iter)
				
				# Move to device handled by PL? No, we are in Dataset.
				# We push to self.device so we can concat with COCO tensors (which are on GPU)
				for k in md_data:
					if isinstance(md_data[k], torch.Tensor):
						md_data[k] = md_data[k].to(self.device)
				
				md_imgs1, md_imgs2 = md_data['image0'], md_data['image1']
				md_imgs1 = md_imgs1.mean(1, keepdim=True)
				md_imgs2 = md_imgs2.mean(1, keepdim=True)
				
				imgs1_t.append(md_imgs1)
				imgs2_t.append(md_imgs2)
				
				# Numpy images for DepthAnything (CPU)
				md_imgs1_np, md_imgs2_np = md_data['image0_np'], md_data['image1_np']
				for i in range(md_imgs1_np.shape[0]):
					img1_np = md_imgs1_np[i].squeeze(0).cpu().numpy()
					img2_np = md_imgs2_np[i].squeeze(0).cpu().numpy()
					imgs1_np.append(img1_np)
					imgs2_np.append(img2_np)
				
				positives_md = megadepth_wrapper.spvs_coarse(md_data, 8)
				positives_coarse += positives_md
			
			# Concat tensors
			# Note: COCO imgs are from augmentor (GPU), MegaDepth moved to GPU above.
			# If both enabled, we cat.
			if not imgs1_t:
				continue # Should not happen if config checked
			
			batch_imgs1 = torch.cat(imgs1_t, dim=0)
			batch_imgs2 = torch.cat(imgs2_t, dim=0)
			
			# Check corruption (few matches)
			is_corrupted = False
			for p in positives_coarse:
				if len(p) < 30:
					is_corrupted = True
			
			if is_corrupted:
				continue
				
			# Yield the constructed batch
			# We wrap numpy lists in a custom object or just yield dictionary
			# PL DataLoader with batch_size=None will return this dict directly
			yield {
				'imgs1': batch_imgs1,
				'imgs2': batch_imgs2,
				'imgs1_np': imgs1_np, # List of numpy arrays
				'imgs2_np': imgs2_np,
				'positives_coarse': positives_coarse # List of arrays
			}

class GeoFeatDataModule(pl.LightningDataModule):
	def __init__(self, data_config, training_res, device_num, batch_size=1):
		super().__init__()
		self.data_config = data_config
		self.training_res = training_res
		self.device_num = device_num
		self.batch_size = batch_size # Not used for batch construction, but for PL
	
	def train_dataloader(self):
		dataset = MixedBatchIterDataset(self.data_config, self.training_res, self.device_num)
		# batch_size=None means "autobatching disabled", dataset yields full batches
		return DataLoader(dataset, batch_size=None, num_workers=0)


# -----------------------------------------------------------------------------
# 2. Lightning Module
# -----------------------------------------------------------------------------
class GeoFeatTrainer(pl.LightningModule):
	def __init__(self, model_config, learning_rate=3e-4, gamma=0.5, step_size=10000):
		super().__init__()
		self.save_hyperparameters()
		self.model_config = model_config
		self.lr = learning_rate
		self.gamma = gamma
		self.step_size = step_size
		
		# Define Model
		self.net = GeoFeatModel(model_config=self.model_config)
		
		# Extract geometric config
		geo_config = self.model_config['geometric_features']
		
		# Define Loss
		# Passing 'cpu' initially, will move to device automatically? 
		# GeoFeatLoss stores device for tensor creation. capture it dynamically or pass property
		# We'll handle device in forward logic or create loss in setup
		self.loss_fn_params = {
			'lam_descs': 1,
			'lam_fb_descs': 2,
			'lam_kpts': 2,
			'lam_heatmap': 1,
			'lam_gradients': 10,
			'lam_curvature': 10,
			'geo_config': geo_config
		}
		# Note: GeoFeatLoss takes 'device' in init. We delay instantiation or update it.
		
		# Auxiliary models (not parameters of this module)
		self.depth_net = None
		self.alike_net = None
		self.loss_fn = None

	def setup(self, stage=None):
		# Initialize auxiliary networks here to ensure they are on the right device
		# But 'self.device' is available after trainer attaches accelerator.
		pass

	def configure_optimizers(self):
		optimizer = optim.Adam(
			filter(lambda x: x.requires_grad, self.net.parameters()), 
			lr=self.lr
		)
		scheduler = torch.optim.lr_scheduler.StepLR(
			optimizer, 
			step_size=self.step_size, 
			gamma=self.gamma
		)
		return [optimizer], [scheduler]

	def on_fit_start(self):
		# Init Helper Nets
		self.depth_net = DepthAnythingExtractor('vits', self.device, 256)
		self.alike_net = ALikeExtractor('alike-t', self.device)
		self.loss_fn = GeoFeatLoss(self.device, **self.loss_fn_params)
		
	def training_step(self, batch, batch_idx):
		# Unpack batch from custom dataset
		imgs1_t = batch['imgs1']
		imgs2_t = batch['imgs2']
		imgs1_np = batch['imgs1_np']
		imgs2_np = batch['imgs2_np']
		positives_coarse = batch['positives_coarse']
		
		# Ensure tensors are on correct device (PL handles this for imgs1_t if returned from loader, 
		# but since we used manual mixed logic, check it)
		# Actually PL moves Tensors in batch to device automatically.
		
		start_time = time.perf_counter()
		
		# 1. Forward Pass 1 (Coarse Features)
		keypoint_map1, des_feats1, geo_feats1 = self.net.forward1(imgs1_t)
		keypoint_map2, des_feats2, geo_feats2 = self.net.forward1(imgs2_t)
		
		# 2. Forward Pass 2 (Refined Features) & ALike Extraction
		alike_kpts1, alike_kpts2 = [], []
		fb_feats1, fb_feats2 = [], []
		
		# Note: In train.py, forward2 is called in a loop. We can batch it if shapes align,
		# but train.py logic suggests per-sample processing primarily for list building.
		# However, GeoFeatModel.forward2 supports batches. Let's try batching for speed.
		
		# Batch forward2
		# Need to unsqueeze? No, forward2 takes (B, C, H, W)
		fb_feats1_batch, keypoint_map1_refined = self.net.forward2(
			keypoint_map1,
			des_feats1,
			geo_feats1,
			return_refined_keypoints=True,
		)
		fb_feats2_batch, keypoint_map2_refined = self.net.forward2(
			keypoint_map2,
			des_feats2,
			geo_feats2,
			return_refined_keypoints=True,
		)
		
		# Split back to lists for loss function which expects lists often? 
		# train.py appends to fb_feats1 lists then cats.
		# loss_fn expects `fb_descs1, fb_descs2` as tensors?
		# Looking at loss.py: forward(..., fb_descs1, ...) uses standard tensor ops usually.
		# train.py does `fb_feats1=torch.cat(fb_feats1,dim=0)` which results in (B, C, H, W) or (Total_Kpts, C).
		# forward2 returns (B, C, H, W).
		# So we can just use the batch output directly.
		
		fb_feats1 = fb_feats1_batch
		fb_feats2 = fb_feats2_batch
		
		# ALike Extraction (per image, CPU->Numpy->Device)
		# This part is heavy and sequential in train.py
		for b in range(len(imgs1_t)):
			# Convert tensor back to numpy for ALike (expects 255-scale numpy)
			# imgs1_t is (B, 1, H, W) normalized? train.py: `* 255`. 
			# In train.py imgs1_t came from mean(1) of loaded tensors.
			img1 = imgs1_t[b].permute(1, 2, 0).expand(-1, -1, 3).cpu().numpy() * 255
			img2 = imgs2_t[b].permute(1, 2, 0).expand(-1, -1, 3).cpu().numpy() * 255
			
			ak1 = self.alike_net.extract_alike_kpts(img1)
			ak2 = self.alike_net.extract_alike_kpts(img2)
			
			alike_kpts1.append(torch.tensor(ak1, device=self.device))
			alike_kpts2.append(torch.tensor(ak2, device=self.device))

		# 3. Depth Anything Extraction (MegaDepth only, but list might be mixed? 
		# In train.py imgs1_np is only populated for MegaDepth samples!
		# Wait, train.py: "imgs1_np.append" inside MegaDepth block. 
		# But depth_net.extract is called "for b in range(len(imgs1_np))".
		# So if batch has 2 COCO and 2 MD, len(imgs1_np) is 2. 
		# The loss function expects aligned lists? 
		# train.py uses `megadepth_batch_size, coco_batch_size` in loss to split features.
		# It assumes the FIRST part of batch is COCO (since COCO appends first in generate_train_data),
		# and SECOND part is MD?
		# train.py: `imgs1_t.append(coco); ... append(md); torch.cat(...)`
		# So order is [COCO...COCO, MD...MD].
		# imgs1_np only has MD images.
		# DA_normals lists will only be length of MD batch size. 
		# loss.py `compute_normals_loss` handles slicing: `pred_normals1[coco_batch_size:]`.
		
		DA_normals1, DA_normals2 = [], []
		DA_depths1, DA_depths2 = [], []
		
		if len(imgs1_np) > 0:
			for b in range(len(imgs1_np)):
				# imgs1_np data might be converted to Tensor by DataLoader
				img1_in = imgs1_np[b]
				img2_in = imgs2_np[b]
				
				if isinstance(img1_in, torch.Tensor):
					img1_in = img1_in.cpu().numpy()
				if isinstance(img2_in, torch.Tensor):
					img2_in = img2_in.cpu().numpy()

				d1, n1, _, _ = self.depth_net.extract(img1_in)
				d2, n2, _, _ = self.depth_net.extract(img2_in)
				DA_normals1.append(n1)
				DA_normals2.append(n2)
				DA_depths1.append(d1)
				DA_depths2.append(d2)
		
		# 4. Compute Loss
		# Need config batch sizes
		md_bs = self.model_config.get('megadepth_batch_size', 0)
		coco_bs = self.model_config.get('coco_batch_size', 0)
		
		# Ensure the batch sizes passed to Dataset match what we tell Loss
		# In Dataset we used data_config.
		# Here we pass them.
		
		# Ensure positives_coarse elements are Tensors on the correct device
		# The loss function calls .long() on them, so they must be Tensors.
		positives_coarse = [
			p.to(self.device) if isinstance(p, torch.Tensor) 
			else torch.from_numpy(p).to(self.device).float() # ensure float first if numpy is varied, but usually it's indices? 
			# Actually pts often contains coordinates which might be float in some contexts, 
			# but used as indices after /8 division and .long().
			# train.py generates them. Let's just ensure Tensor.
			for p in positives_coarse
		]

		loss_info = self.loss_fn(
			des_feats1, fb_feats1, keypoint_map1_refined, geo_feats1,
			des_feats2, fb_feats2, keypoint_map2_refined, geo_feats2,
			positives_coarse,
			alike_kpts1, alike_kpts2,
			DA_normals1, DA_normals2,
			DA_depths1, DA_depths2,
			megadepth_batch_size=md_bs,
			coco_batch_size=coco_bs
		)
		
		# 5. Logging
		# Calculate total loss manually as in train.py
		# Note: loss_info values are already weighted by lambdas in loss.py
		loss_items = []
		loss_items.append(loss_info['loss_descs'].unsqueeze(0))
		loss_items.append(loss_info['loss_fb_descs'].unsqueeze(0))
		loss_items.append(loss_info['loss_kpts'].unsqueeze(0))
		
		# Add geometric losses if enabled (they are 0.0 tensors if not enabled in loss.py, 
		# but let's follow geo_config to be precise or just add them if they are non-zero/enabled)
		# loss.py returns them as 0.0 tensor if disabled logic path taken or just 0.0.
		# But train.py checks config.
		if self.loss_fn_params['geo_config']['depth']:
			loss_items.append(loss_info['loss_depths'].unsqueeze(0))
		if self.loss_fn_params['geo_config']['normal']:
			loss_items.append(loss_info['loss_normals'].unsqueeze(0))
		if self.loss_fn_params['geo_config']['gradients']:
			loss_items.append(loss_info['loss_gradients'].unsqueeze(0))
		if self.loss_fn_params['geo_config']['curvatures']:
			loss_items.append(loss_info['loss_curvature'].unsqueeze(0))
			
		loss = torch.cat(loss_items, -1).mean()

		acc_fb_coarse = loss_info['acc_fb_coarse']
		
		# Log main metrics
		self.log('train_loss', loss, prog_bar=True, logger=True)
		self.log('train_acc', acc_fb_coarse, prog_bar=True, logger=True)
		
		# Log components
		for k, v in loss_info.items():
			if isinstance(v, torch.Tensor):
				self.log(f'details/{k}', v.item() if v.numel()==1 else v.mean(), logger=True)
			elif isinstance(v, (int, float)):
				self.log(f'details/{k}', v, logger=True)
		
		return loss

if __name__ == '__main__':
	
	args = parse_arguments()
	
	# Load Config
	cfg = get_cfg_defaults()
	
	data_config = {
		'use_megadepth': bool(cfg.DATASET.USE_MEGADEPTH),
		'use_coco': bool(cfg.DATASET.USE_COCO),
		'megadepth_root_path': cfg.DATASET.MEGADEPTH_ROOT_PATH,
		'megadepth_batch_size': int(cfg.DATASET.MEGADEPTH_BATCH_SIZE),
		'coco_root_path': cfg.DATASET.COCO_ROOT_PATH,
		'coco_batch_size': int(cfg.DATASET.COCO_BATCH_SIZE),
	}
	
	model_config = {
		"backbone": cfg.MODEL.BACKBONE,
		'keypoint_encoder': list(cfg.MODEL.KEYPOINT_ENCODER),
		'keypoint_dim': int(cfg.MODEL.KEYPOINT_DIM),
		'descriptor_encoder': list(cfg.MODEL.DESCRIPTOR_ENCODER),
		'descriptor_dim': int(cfg.MODEL.DESCRIPTOR_DIM),
		'geometric_features': {
			'depth': bool(cfg.MODEL.GEOMETRIC_FEATURES.DEPTH),
			'normal': bool(cfg.MODEL.GEOMETRIC_FEATURES.NORMAL),
			'gradients': bool(cfg.MODEL.GEOMETRIC_FEATURES.GRADIENTS),
			'curvatures': bool(cfg.MODEL.GEOMETRIC_FEATURES.CURVATURES),
		},
		'depth_encoder': list(cfg.MODEL.DEPTH_ENCODER),
		'depth_dim': int(cfg.MODEL.DEPTH_DIM),
		'normal_encoder': list(cfg.MODEL.NORMAL_ENCODER),
		'normal_dim': int(cfg.MODEL.NORMAL_DIM),
		'gradient_encoder': list(cfg.MODEL.GRADIENT_ENCODER),
		'gradient_dim': int(cfg.MODEL.GRADIENT_DIM),
		'curvature_encoder': list(cfg.MODEL.CURVATURE_ENCODER),
		'curvature_dim': int(cfg.MODEL.CURVATURE_DIM),
		'repeatability_alpha': float(cfg.MODEL.REPEATABILITY_ALPHA),
		"attention_layers": int(cfg.MODEL.ATTENTIONAL_LAYERS),
		"refiner_type": str(cfg.MODEL.REFINER_TYPE),
		"last_activation": str(cfg.MODEL.LAST_ACTIVATION),
		'output_dim': int(cfg.MODEL.OUTPUT_DIM),
		# Add batch sizes to model config for loss usage
		'megadepth_batch_size': data_config['megadepth_batch_size'],
		'coco_batch_size': data_config['coco_batch_size']
	}

	# Create Data Module
	datasets = GeoFeatDataModule(data_config, args.training_res, args.device_num)
	
	# Create Model
	model = GeoFeatTrainer(
		model_config, 
		learning_rate=args.lr, 
		gamma=args.gamma_steplr,
		step_size=10000
	)
	
	# Logger
	stamp = time.strftime('%Y%m%d_%H%M%S')
	logger = TensorBoardLogger(args.ckpt_save_path, name=f"{args.name}_{stamp}")
	
	# Checkpointing
	checkpoint_callback = ModelCheckpoint(
		monitor='train_acc',       # Monitoring the metric logged in training_step
		mode='max',
		filename='{epoch}-{step}-{train_acc:.3f}',
		save_top_k=3,
		every_n_train_steps=args.save_ckpt_every,
		save_last=True
	)
	
	# Trainer
	trainer = pl.Trainer(
		max_steps=args.n_steps,
		logger=logger,
		callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
		accelerator='gpu' if torch.cuda.is_available() else 'cpu',
		devices=1,
		log_every_n_steps=args.log_interval,
		fast_dev_run=args.dry_run,  # Enable dry run if requested
		# detect_anomaly=True, # Optional debugging
	)
	
	print(f"Starting PL Training: {args.name}")
	trainer.fit(model, datamodule=datasets)