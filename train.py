"""
	"LiftFeat: 3D Geometry-Aware Local Feature Matching"
	training script
"""

import os
import time
import sys
from typing import cast

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from src.config.config import (
	get_cfg_defaults,
	get_data_config,
	get_model_config,
	get_train_config,
)

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

import numpy as np
import tqdm
import glob
import sys
import os

from model.GeoFeatModel import GeoFeatModel
from model.GeometricExtractor import GeometricExtractor
from loss.loss import GeoFeatLoss, get_loss_config
from utils.depth_anything_wrapper import DepthAnythingExtractor
from utils.alike_wrapper import ALikeExtractor

from data.coco_augmentor import COCOAugmentor
from data import coco_wrapper
from src.data.megadepth import MegaDepthDataset
from src.data import megadepth_wrapper

import setproctitle

class Trainer():
	def __init__(self, 
				data_config, model_config, train_config=None,
				model_name='GeoFeat',
				ckpt_save_path="./weights",
				n_steps=160_000,
				save_ckpt_every=1000,
				lr=3e-4, gamma_steplr=0.5,
				training_res=(800, 640), 
				device_num="cuda:0",
				use_dual_kpt_loss=False,
				dry_run=False,
				max_runs=10
				):
		
		# Store configs
		self.data_config = data_config
		self.model_config = model_config
		self.train_config = dict(train_config or {})
		self.use_dual_kpt_loss = bool(use_dual_kpt_loss)
		self.kpt_single_source = 'raw'
		self.kpt_raw_weight_start = 0.8
		self.kpt_raw_weight_end = 0.2
		self.kpt_refined_weight_start = 0.2
		self.kpt_refined_weight_end = 0.8
		
		# Extract dataset parameters from config (assumed validated)
		use_megadepth = bool(self.data_config['use_megadepth'])
		megadepth_batch_size = int(self.data_config['megadepth_batch_size']) if use_megadepth else 0
		megadepth_root_path = cast(str, self.data_config['megadepth_root_path']) if use_megadepth else None

		use_coco = bool(self.data_config['use_coco'])
		coco_batch_size = int(self.data_config['coco_batch_size']) if use_coco else 0
		coco_root_path = cast(str, self.data_config['coco_root_path']) if use_coco else None
		
		print(f'MegeDepth: {use_megadepth}-{megadepth_batch_size}')
		print(f'COCO20k: {use_coco}-{coco_batch_size}')

		self.dev = torch.device(device_num if torch.cuda.is_available() else 'cpu')
		print(f'Device: {self.dev}')

		# training model - Pass model config to GeoFeatModel
		self.net = GeoFeatModel(model_config=self.model_config).to(self.dev)
		refiner_type = str(self.model_config['refiner_type'])
		if refiner_type == 'GCDR':
			if getattr(self.net.attn_fusion, 'pre_attn_refiner', None) is None:
				raise RuntimeError(
					"Config requests REFINER_TYPE='GCDR', but model.attn_fusion.pre_attn_refiner is None. "
					"Please verify that server GeoFeatModel.py includes the GCDR branch."
				)
		
		# Extract geometric config (validated in __main__)
		geo_config = self.model_config['geometric_features']
		loss_cfg = get_loss_config()
		reliability_loss_weight = 0.0
		if refiner_type == 'GCDR':
			reliability_loss_weight = float(loss_cfg['reliability_loss_weight'])

		repeatability_loss_weight = float(loss_cfg['repeatability_loss_weight'])

		self.loss_fn = GeoFeatLoss(
			self.dev, 
			lam_descs=1, 
			lam_fb_descs=3,
			lam_kpts=2, 
			lam_kpts_raw=5,
			lam_heatmap=1,
			lam_reliability=reliability_loss_weight,
			lam_repeatability=repeatability_loss_weight,
			lam_depth=float(loss_cfg['depth_weight']),
			lam_gradients=float(loss_cfg['gradient_weight']),
			lam_curvature=float(loss_cfg['curvature_weight']),
			lam_descriptor_protect=float(loss_cfg['descriptor_protect_weight']),
			lam_reliable_enhance=float(loss_cfg['reliable_enhance_weight']),
			lam_unreliable_suppress=float(loss_cfg['unreliable_suppress_weight']),
			reliable_enhance_margin=float(loss_cfg['reliable_enhance_margin']),
			reliable_enhance_threshold=float(loss_cfg['reliable_enhance_threshold']),
			reliable_enhance_quantile=float(loss_cfg['reliable_enhance_quantile']),
			unreliable_suppress_threshold=float(loss_cfg['unreliable_suppress_threshold']),
			unreliable_suppress_quantile=float(loss_cfg['unreliable_suppress_quantile']),
			reliability_consistency_weight=float(loss_cfg['reliability_consistency_weight']),
			reliability_target_scale=float(loss_cfg['reliability_target_scale']),
			geo_loss_reliability_weighted=bool(loss_cfg['geo_loss_reliability_weighted']),
			geo_loss_unreliable_floor=float(loss_cfg['geo_loss_unreliable_floor']),
			geo_config=geo_config
		)

		# depth-anything model  # 深度估计提取器
		self.depth_net = DepthAnythingExtractor('vits', self.dev, 256)
		self.dam_geometry_extractor = GeometricExtractor(
			model_config=self.model_config,
			device=self.dev,
			enable_depth_extractor=False,
		).to(self.dev).eval()

		# alike model   # 高效的高精度局部特征提取模型
		self.alike_net = ALikeExtractor('alike-t', self.dev)

		#Setup optimizer # 设置优化器
		self.steps = 1 if dry_run else n_steps
		self.opt = optim.Adam(filter(lambda x: x.requires_grad, self.net.parameters()), lr=lr)
		# 每 step_size 步 削减 按 gamma 比例, 减少学习率
		self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=10_000, gamma=gamma_steplr)

		##################### COCO INIT ##########################
		self.use_coco = use_coco
		self.coco_batch_size = coco_batch_size
		if self.use_coco:  # 启用COCO
			coco_img_dir = cast(str, coco_root_path)
			self.augmentor = COCOAugmentor(
				img_dir=coco_img_dir,
				device=self.dev,
				load_dataset=True,
				batch_size=self.coco_batch_size,
				out_resolution=training_res,  # 输出分辨率
				warp_resolution=training_res,  # 几何变换后的输出分辨率
				sides_crop=0.1,
				max_num_imgs=3000,  # 最大训练图像数
				num_test_imgs=5,  # 测试图像数
				photometric=True,  # 启用光度增强
				geometric=True,  # 启用几何增强
				reload_step=4000,
				rotation_degrees=30.0, scale_min=0.35, scale_max=1.20,
				brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15,
				gamma_min=1.0, gamma_max=1.0, gamma_probability=0.0,
				equalize_probability=0.40, shadow_probability=0.40,
				shadow_min_factor=0.25, gaussian_blur_probability=0.30,
				noise_probability=0.50, noise_std=10.0 / 255.0,
			)
		##################### COCO END #######################

		##################### MEGADEPTH INIT ##########################
		self.use_megadepth = use_megadepth
		self.megadepth_batch_size = megadepth_batch_size
		if self.use_megadepth:
			TRAIN_BASE_PATH = f"{megadepth_root_path}/megadepth_indices"
			TRAINVAL_DATA_SOURCE = f"{megadepth_root_path}/MegaDepth_v1"
			# Adapting to local deeply nested structure
			# TRAINVAL_DATA_SOURCE = f"{megadepth_root_path}/MegaDepth_v1/phoenix/S6/zl548/MegaDepth_v1"

			TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"

			npz_paths = glob.glob(TRAIN_NPZ_ROOT + '/*.npz')[:]
			megadepth_dataset = torch.utils.data.ConcatDataset(
				[MegaDepthDataset(
					root_dir=TRAINVAL_DATA_SOURCE,
					npz_path=path,
					img_resize=training_res) 
				for path in tqdm.tqdm(npz_paths, desc="[MegaDepth] Loading metadata")])

			self.megadepth_dataloader = DataLoader(megadepth_dataset, batch_size=megadepth_batch_size, shuffle=True)
			self.megadepth_data_iter = iter(self.megadepth_dataloader)
		##################### MEGADEPTH INIT END #######################

		# 运行目录结构: <ckpt_save_path>/<model_name>_<timestamp>/
		os.makedirs(ckpt_save_path, exist_ok=True)
		stamp = time.strftime('%Y%m%d_%H%M%S')
		self.run_dir = os.path.join(ckpt_save_path, f'{model_name}_{stamp}')
		os.makedirs(self.run_dir, exist_ok=True)
		self.ckpt_dir = self.run_dir  # 后续可细分 ckpts/ 与 logs/
		logdir = os.path.join(self.run_dir, 'logdir')
		os.makedirs(logdir, exist_ok=True)

		self.model_name = model_name
		self.ckpt_save_path = ckpt_save_path
		self.save_ckpt_every = save_ckpt_every
		configured_kpt_ramp = 40000
		self.kpt_ramp_steps = max(1, configured_kpt_ramp if configured_kpt_ramp > 0 else int(0.8 * n_steps))
		self.homography_difficulty_start = 0.10
		self.homography_difficulty_end = 0.10
		self.homography_ramp_steps = 1
		self.tps_probability = 0.70
		self.best_metric_name = 'acc_fb_coarse'

		# 清理旧的训练目录
		self._cleanup_old_runs(ckpt_save_path, model_name, max_runs=max_runs)

		self.dry_run = dry_run
		self.writer = SummaryWriter(logdir)
		self.status_path = os.path.join(self.run_dir, 'status.txt')
		
		# 初始化status文件
		try:
			with open(self.status_path, 'w', encoding='utf-8') as f:
				f.write('global_step,loss,' \
						'loss_descs,acc_coarse,' \
						'loss_fb_descs,acc_fb_coarse,' \
						'loss_kpts,loss_kpts_raw,loss_kpts_refined,loss_reliability,loss_repeatability,loss_descriptor_protect,loss_reliable_enhance,loss_unreliable_suppress,' \
						'reliable_enhance_rate,reliable_region_ratio,reliable_mean_gain,unreliable_enhance_rate,unreliable_region_ratio,unreliable_mean_gain,kpt_raw_weight,kpt_refined_weight,' \
						'loss_depths,loss_gradients,loss_curvature,' \
						'total_pos,lr,forward_cost\n')
			print(f"Status file initialized: {self.status_path}")
		except Exception as e:
			print(f"Warning: Failed to initialize status file: {e}")

	def _compute_kpt_weights(self, global_step: int):
		if not self.use_dual_kpt_loss:
			# Baseline recovery path: always use raw branch.
			return 1.0, 0.0

		# Stage clock is aligned with checkpoint interval.
		# First 80%: 0.8/0.2 -> 0.2/0.8 with stair steps.
		# Last 20%: keep 0.2/0.8 instead of hard-switching to refined only.
		# ALike/raw supervision stays as a weak regularizer while GeoFeat/refined
		# keypoints become the main training signal.
		ramp_end = self.kpt_ramp_steps
		if global_step > ramp_end:
			raw_w = float(self.kpt_raw_weight_end)
			refined_w = float(self.kpt_refined_weight_end)
			s = max(raw_w + refined_w, 1e-6)
			return raw_w / s, refined_w / s

		step_clock = max(1, self.save_ckpt_every)
		stage_idx = max(0, (global_step - 1) // step_clock)
		stage_total = max(1, (ramp_end - 1) // step_clock)
		t = stage_idx / float(stage_total)

		raw_w = self.kpt_raw_weight_start + t * (self.kpt_raw_weight_end - self.kpt_raw_weight_start)
		refined_w = self.kpt_refined_weight_start + t * (self.kpt_refined_weight_end - self.kpt_refined_weight_start)

		s = max(raw_w + refined_w, 1e-6)
		return raw_w / s, refined_w / s

	def _compute_homography_difficulty(self, global_step: int) -> float:
		t = min(max(float(global_step), 0.0) / float(self.homography_ramp_steps), 1.0)
		return self.homography_difficulty_start + t * (
			self.homography_difficulty_end - self.homography_difficulty_start
		)

	def _cleanup_old_runs(self, base_dir: str, model_name: str, max_runs: int = 10):
		"""保留最近 max_runs 个以 model_name_ 开头的运行目录, 其余删除."""
		try:
			candidates = []
			for n in os.listdir(base_dir):
				full = os.path.join(base_dir, n)
				if os.path.isdir(full) and n.startswith(model_name + '_'):
					candidates.append((os.path.getctime(full), full))
			candidates.sort(key=lambda x: x[0])  # old -> new
			if len(candidates) > max_runs:
				import shutil
				for _, path in candidates[:len(candidates)-max_runs]:
					try:
						shutil.rmtree(path, ignore_errors=True)
						print(f'[cleanup] removed old run: {os.path.basename(path)}')
					except Exception as e:
						print(f'[cleanup][warn] failed to remove {path}: {e}')
				removed_count = len(candidates) - max_runs
				print(f'[cleanup] Removed {removed_count} old training directories, keeping {max_runs} most recent')
		except Exception as e:
			print(f'[cleanup][warn] cleanup failed: {e}')

	# 创建训练数据
	def generate_train_data(self, global_step: int):
		imgs1_t,imgs2_t=[],[]
		imgs1_np,imgs2_np=[],[]
		# norms0,norms1=[],[]
		positives_coarse=[]
		
		if self.use_coco:
				difficulty = self._compute_homography_difficulty(global_step)
				coco_imgs1, coco_imgs2, H1, H2 = coco_wrapper.make_batch(
					self.augmentor,
					difficulty,
					tps_probability=self.tps_probability,
				)
				h_coarse, w_coarse = coco_imgs1[0].shape[-2] // 8, coco_imgs1[0].shape[-1] // 8
				_ , positives_coco_coarse = coco_wrapper.get_corresponding_pts(coco_imgs1, coco_imgs2, H1, H2, self.augmentor, h_coarse, w_coarse)
				coco_imgs1=coco_imgs1.mean(1,keepdim=True);coco_imgs2=coco_imgs2.mean(1,keepdim=True)
				imgs1_t.append(coco_imgs1);imgs2_t.append(coco_imgs2)
				positives_coarse += positives_coco_coarse
					
		if self.use_megadepth:
			try:
				megadepth_data=next(self.megadepth_data_iter)
			except StopIteration:
				print('End of MD DATASET')
				self.megadepth_data_iter=iter(self.megadepth_dataloader)
				megadepth_data=next(self.megadepth_data_iter)
			if megadepth_data is not None:
				for k in megadepth_data.keys():
					if isinstance(megadepth_data[k],torch.Tensor):
						megadepth_data[k]=megadepth_data[k].to(self.dev)
				megadepth_imgs1_t,megadepth_imgs2_t=megadepth_data['image0'],megadepth_data['image1']
				megadepth_imgs1_t=megadepth_imgs1_t.mean(1,keepdim=True);megadepth_imgs2_t=megadepth_imgs2_t.mean(1,keepdim=True)
				imgs1_t.append(megadepth_imgs1_t);imgs2_t.append(megadepth_imgs2_t)
				megadepth_imgs1_np,megadepth_imgs2_np=megadepth_data['image0_np'],megadepth_data['image1_np']
				for np_idx in range(megadepth_imgs1_np.shape[0]):
					img1_np,img2_np=megadepth_imgs1_np[np_idx].squeeze(0).cpu().numpy(),megadepth_imgs2_np[np_idx].squeeze(0).cpu().numpy()
					imgs1_np.append(img1_np);imgs2_np.append(img2_np)
				positives_megadepth_coarse=megadepth_wrapper.spvs_coarse(megadepth_data,8)
				positives_coarse += positives_megadepth_coarse
				
		with torch.no_grad():
			imgs1_t=torch.cat(imgs1_t,dim=0)
			imgs2_t=torch.cat(imgs2_t,dim=0)
			
		return imgs1_t,imgs2_t,imgs1_np,imgs2_np,positives_coarse
		
	# 神经网络训练
	def train(self):
		self.net.train()
		best_metric = float('-inf')

		with tqdm.tqdm(total=self.steps) as pbar:
			for i in range(self.steps):
				global_step = i + 1
				# import pdb;pdb.set_trace()
				imgs1_t, imgs2_t, imgs1_np, imgs2_np, positives_coarse = self.generate_train_data(global_step)

				#Check if batch is corrupted with too few correspondences
				# 检查匹配点是否过少
				is_corrupted = False
				for p in positives_coarse:
					# 若匹配点对小于30个, 则认为无效
					if len(p) < 30:
						is_corrupted = True

				if is_corrupted:
					continue    # 跳过当前批次

				# import pdb;pdb.set_trace()
				start=time.perf_counter()
				#Forward pass
				
				# 初始描述子提取(forward1)
				# keypoint_map: 关键点响应图
				# des_feats: 描述子特征图
				# geo_feats: 几何特征图（按配置启用的几何通道）
				keypoint_map1, des_feats1, geo_feats1 = self.net.forward1(imgs1_t)
				keypoint_map2, des_feats2, geo_feats2 = self.net.forward1(imgs2_t)
				
				# 初始描述子特征匹配, 增强描述子匹配
				alike_kpts1, alike_kpts2 = [], []
				DA_normals1, DA_normals2 = [], []  # Legacy loss arguments; normals are packed into gradients.
				DA_geo_targets1, DA_geo_targets2 = [], []

				# import pdb;pdb.set_trace()

				fb_feats1, fb_feats2=[],[]
				geo_reliability1, geo_reliability2 = [], []
				repeatability1, repeatability2 = [], []
				kpt_maps1_refined, kpt_maps2_refined = [], []
				for b in range(des_feats1.shape[0]):
					feat1=des_feats1[b].permute(1,2,0).reshape(-1,des_feats1.shape[1])
					feat2=des_feats2[b].permute(1,2,0).reshape(-1,des_feats2.shape[1])
					
					result1 = cast(tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]], self.net.forward2(
						keypoint_map1[b].unsqueeze(0),
						des_feats1[b].unsqueeze(0),
						geo_feats1[b].unsqueeze(0),
						return_refined_keypoints=True,
						return_aux=True,
					))
					fb_feat1, kpt_refined1, aux1 = result1
					result2 = cast(tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]], self.net.forward2(
						keypoint_map2[b].unsqueeze(0),
						des_feats2[b].unsqueeze(0),
						geo_feats2[b].unsqueeze(0),
						return_refined_keypoints=True,
						return_aux=True,
					))
					fb_feat2, kpt_refined2, aux2 = result2
					
					fb_feats1.append(fb_feat1);fb_feats2.append(fb_feat2)
					kpt_maps1_refined.append(kpt_refined1);kpt_maps2_refined.append(kpt_refined2)
					if 'geo_joint_reliability_map' in aux1:
						geo_reliability1.append(aux1['geo_joint_reliability_map'])
					elif 'geo_reliability_map' in aux1:
						geo_reliability1.append(aux1['geo_reliability_map'])
					if 'geo_joint_reliability_map' in aux2:
						geo_reliability2.append(aux2['geo_joint_reliability_map'])
					elif 'geo_reliability_map' in aux2:
						geo_reliability2.append(aux2['geo_reliability_map'])
					if 'repeatability_map' in aux1:
						repeatability1.append(aux1['repeatability_map'])
					if 'repeatability_map' in aux2:
						repeatability2.append(aux2['repeatability_map'])
					
					img1,img2=imgs1_t[b],imgs2_t[b]
					img1=img1.permute(1,2,0).expand(-1,-1,3).cpu().numpy() * 255
					img2=img2.permute(1,2,0).expand(-1,-1,3).cpu().numpy() * 255
					alike_kpt1=torch.tensor(self.alike_net.extract_alike_kpts(img1),device=self.dev)
					alike_kpt2=torch.tensor(self.alike_net.extract_alike_kpts(img2),device=self.dev)
					alike_kpts1.append(alike_kpt1);alike_kpts2.append(alike_kpt2)
				
				# import pdb;pdb.set_trace()
				for b in range(len(imgs1_np)):
					geo_target1 = self.depth_net.extract(imgs1_np[b], geometric_extractor=self.dam_geometry_extractor,)
					geo_target2 = self.depth_net.extract(imgs2_np[b], geometric_extractor=self.dam_geometry_extractor,)
					DA_geo_targets1.append(geo_target1)
					DA_geo_targets2.append(geo_target2)
					
				# import pdb;pdb.set_trace()
				fb_feats1=torch.cat(fb_feats1,dim=0)
				fb_feats2=torch.cat(fb_feats2,dim=0)
				geo_reliability1 = torch.cat(geo_reliability1, dim=0) if geo_reliability1 else None
				geo_reliability2 = torch.cat(geo_reliability2, dim=0) if geo_reliability2 else None
				repeatability1 = torch.cat(repeatability1, dim=0) if repeatability1 else None
				repeatability2 = torch.cat(repeatability2, dim=0) if repeatability2 else None
				keypoint_map1_refined = torch.cat(kpt_maps1_refined, dim=0)
				keypoint_map2_refined = torch.cat(kpt_maps2_refined, dim=0)
				# Flattening/Reshaping removed as forward2 now returns (B, C, H, W) maps directly
				
				end=time.perf_counter()
				print(f"\n\nforward cost {end-start:.1f} seconds")
				
				# 损失计算
				loss_items = []
				kpt_raw_weight, kpt_refined_weight = self._compute_kpt_weights(global_step)

				# import pdb;pdb.set_trace()
				loss_info = self.loss_fn(
					des_feats1, fb_feats1, keypoint_map1_refined, geo_feats1,   # 图像1: 描述子 增强描述子 关键点图(融合后) 几何特征
					des_feats2, fb_feats2, keypoint_map2_refined, geo_feats2,   # 图像2: 描述子 增强描述子 关键点图(融合后) 几何特征
					positives_coarse,                                         	# 粗糙匹配点
					alike_kpts1, alike_kpts2,                                  	# ALIKE关键点
					DA_normals1, DA_normals2,                                  	# MegaDepth法向量
					DA_geo_targets1, DA_geo_targets2,                           	# DAM depth -> GeometricExtractor targets
					self.megadepth_batch_size, self.coco_batch_size,
					reliability1=geo_reliability1,
					reliability2=geo_reliability2,
					repeatability1=repeatability1,
					repeatability2=repeatability2,
					raw_kpts1=keypoint_map1,
					raw_kpts2=keypoint_map2,
					refined_kpts1=keypoint_map1_refined,
					refined_kpts2=keypoint_map2_refined,
					use_dual_kpt_loss=self.use_dual_kpt_loss,
					kpt_raw_weight=kpt_raw_weight,
					kpt_refined_weight=kpt_refined_weight,
					kpt_single_source=self.kpt_single_source)

				loss_descs = loss_info['loss_descs']
				acc_coarse = loss_info['acc_coarse']
				
				loss_fb_descs = loss_info['loss_fb_descs']
				acc_fb_coarse = loss_info['acc_fb_coarse']
				
				loss_kpts = loss_info['loss_kpts']
				loss_kpts_raw = loss_info['loss_kpts_raw']
				loss_kpts_refined = loss_info['loss_kpts_refined']
				loss_reliability = loss_info['loss_reliability']
				loss_repeatability = loss_info['loss_repeatability']
				loss_descriptor_protect = loss_info['loss_descriptor_protect']
				loss_reliable_enhance = loss_info['loss_reliable_enhance']
				loss_unreliable_suppress = loss_info['loss_unreliable_suppress']
				reliable_enhance_rate = loss_info['reliable_enhance_rate']
				reliable_region_ratio = loss_info['reliable_region_ratio']
				reliable_mean_gain = loss_info['reliable_mean_gain']
				unreliable_enhance_rate = loss_info['unreliable_enhance_rate']
				unreliable_region_ratio = loss_info['unreliable_region_ratio']
				unreliable_mean_gain = loss_info['unreliable_mean_gain']
				kpt_raw_weight = float(loss_info['kpt_raw_weight'])
				kpt_refined_weight = float(loss_info['kpt_refined_weight'])
				acc_kpt = loss_info['acc_kpt']

				loss_depths = loss_info['loss_depths']
				loss_gradients = loss_info['loss_gradients']
				loss_curvature = loss_info['loss_curvature']
				
				loss_items.append(loss_descs.unsqueeze(0))      # 描述子损失
				loss_items.append(loss_fb_descs.unsqueeze(0))   # 增强描述子损失
				loss_items.append(loss_kpts.unsqueeze(0))       # 关键点损失
				if self.model_config['refiner_type'] == 'GCDR':
					loss_items.append(loss_reliability.unsqueeze(0))
					loss_items.append(loss_reliable_enhance.unsqueeze(0))
					loss_items.append(loss_unreliable_suppress.unsqueeze(0))
				
				loss_items.append(loss_repeatability.unsqueeze(0))
				loss_items.append(loss_descriptor_protect.unsqueeze(0))

				# Add geometric losses if enabled
				geo_config = self.model_config['geometric_features']
				if geo_config['depth']:
					loss_items.append(loss_depths.unsqueeze(0))     	# 深度损失
				if geo_config['gradients']:
					loss_items.append(loss_gradients.unsqueeze(0))  	# first-order loss: dx/dy + normals
				if geo_config['curvatures']:
					loss_items.append(loss_curvature.unsqueeze(0))  	# 曲率损失

				# nb_coarse = len(m1)
				# nb_coarse = len(fb_m1)
				loss = torch.cat(loss_items, -1).mean() # 加权平均总损失

				# Compute Backward Pass
				# 反向传播
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.)   # 梯度裁剪
				self.opt.step()         # 参数更新
				self.opt.zero_grad()    # 梯度清零
				self.scheduler.step()   # 学习率调整

				# import pdb;pdb.set_trace()
				# 保存权重文件
				if global_step % self.save_ckpt_every == 0:
					print('saving iter ', global_step)
					ckpt_path = os.path.join(self.ckpt_dir, f'{self.model_name}_step{global_step}.pth')
					torch.save(self.net.state_dict(), ckpt_path)

					# Save the best in-loop checkpoint separately because the training
					# metric may deteriorate during the later part of a run. The metric
					# is a training proxy and should still be confirmed on benchmarks.
					metric_value = float(acc_fb_coarse)
					if metric_value > best_metric:
						best_metric = metric_value
						best_ckpt_path = os.path.join(self.ckpt_dir, f'{self.model_name}_best.pth')
						torch.save(self.net.state_dict(), best_ckpt_path)
						best_info_path = os.path.join(self.ckpt_dir, f'{self.model_name}_best.txt')
						with open(best_info_path, 'w', encoding='utf-8') as f:
							f.write(f'global_step={global_step}\n')
							f.write(f'{self.best_metric_name}={metric_value:.8f}\n')
						print(
							f'new best {self.best_metric_name}={metric_value:.6f} '
							f'at step {global_step}'
						)

				# 打印训练过程(损失函数) - 使用 tqdm.write 分行输出详细信息，避免单行过长
				pbar.set_description(f'Loss/total: {loss.item():.4f} Accuracy/acc_fb_coarse: {acc_fb_coarse:.3f}')
				pbar.update(1)
				# 为什么修改打印训练过程 会该改变训练时间？
				# tqdm 的 update(n) 方法只是进度条的计数器，它会让进度条前进n步。例如：
				# pbar.update(1)：进度条前进 1 步
				# pbar.update(5)：进度条前进 5 步
				# 它不会影响你的训练逻辑，只是显示上的变化。

				# Log metrics whenever a checkpoint is saved.
				if global_step % self.save_ckpt_every == 0:
					self.writer.add_scalar('Loss/total', loss.item(), global_step)
					self.writer.add_scalar('Loss/descs', loss_descs.item(), global_step)
					self.writer.add_scalar('Accuracy/acc_coarse', acc_coarse, global_step)
					self.writer.add_scalar('Loss/fb_descs', loss_fb_descs.item(), global_step)
					self.writer.add_scalar('Accuracy/acc_fb_coarse', acc_fb_coarse, global_step)
					self.writer.add_scalar('Loss/kpts', loss_kpts.item(), global_step)
					self.writer.add_scalar('Loss/kpts_raw', loss_kpts_raw.item(), global_step)
					self.writer.add_scalar('Loss/kpts_refined', loss_kpts_refined.item(), global_step)
					self.writer.add_scalar('Loss/reliability', loss_reliability.item(), global_step)
					self.writer.add_scalar('Loss/repeatability', loss_repeatability.item(), global_step)
					self.writer.add_scalar('Loss/descriptor_protect', loss_descriptor_protect.item(), global_step)
					self.writer.add_scalar('Loss/reliable_enhance', loss_reliable_enhance.item(), global_step)
					self.writer.add_scalar('Loss/unreliable_suppress', loss_unreliable_suppress.item(), global_step)
					self.writer.add_scalar('Reliability/enhance_rate', reliable_enhance_rate.item(), global_step)
					self.writer.add_scalar('Reliability/reliable_region_ratio', reliable_region_ratio.item(), global_step)
					self.writer.add_scalar('Reliability/reliable_mean_gain', reliable_mean_gain.item(), global_step)
					self.writer.add_scalar('Reliability/unreliable_enhance_rate', unreliable_enhance_rate.item(), global_step)
					self.writer.add_scalar('Reliability/unreliable_region_ratio', unreliable_region_ratio.item(), global_step)
					self.writer.add_scalar('Reliability/unreliable_mean_gain', unreliable_mean_gain.item(), global_step)
					self.writer.add_scalar('Loss/kpt_raw_weight', kpt_raw_weight, global_step)
					self.writer.add_scalar('Loss/kpt_refined_weight', kpt_refined_weight, global_step)
					self.writer.add_scalar('Loss/depths', loss_depths.item(), global_step)
					self.writer.add_scalar('Loss/gradients', loss_gradients.item(), global_step)
					self.writer.add_scalar('Loss/curvature', loss_curvature.item(), global_step)
				
				# status.txt 写入
				if global_step % self.save_ckpt_every == 0:
					total_pos = sum(len(p) for p in positives_coarse)
					with open(self.status_path, 'a', encoding='utf-8') as f:
						f.write(f"{global_step},{loss.item():.6f},"
								f"{loss_descs.item():.6f},{acc_coarse:.4f},"
								f"{loss_fb_descs.item():.6f},{acc_fb_coarse:.4f},"
								f"{loss_kpts.item():.6f},{loss_kpts_raw.item():.6f},{loss_kpts_refined.item():.6f},"
								f"{loss_reliability.item():.6f},{loss_repeatability.item():.6f},"
								f"{loss_descriptor_protect.item():.6f},{loss_reliable_enhance.item():.6f},{loss_unreliable_suppress.item():.6f},"
								f"{reliable_enhance_rate.item():.6f},{reliable_region_ratio.item():.6f},{reliable_mean_gain.item():.6f},"
								f"{unreliable_enhance_rate.item():.6f},{unreliable_region_ratio.item():.6f},{unreliable_mean_gain.item():.6f},"
								f"{kpt_raw_weight:.4f},{kpt_refined_weight:.4f},"
								f"{loss_depths.item():.6f},{loss_gradients.item():.6f},{loss_curvature.item():.6f},"
								f"{total_pos},{self.opt.param_groups[0]['lr']:.6e},{end-start:.3f}\n")


if __name__ == '__main__':
	cfg = get_cfg_defaults()

	data_config = get_data_config(cfg)
	train_config = get_train_config(cfg)
	model_config = get_model_config(cfg)

	setproctitle.setproctitle(train_config['model_name'])

	# # Show loaded configs explicitly
	# print("data_config:")
	trainer = Trainer(
		# Configuration files
		data_config=data_config,
		model_config=model_config,
		train_config=train_config,

		# Training parameters from config.py
		model_name=train_config['model_name'],
		ckpt_save_path=train_config['ckpt_save_path'],
		n_steps=train_config['n_steps'],
		save_ckpt_every=train_config['save_ckpt_every'],
		lr=train_config['lr'],
		gamma_steplr=train_config['gamma_steplr'],
		training_res=train_config['training_res'],
		device_num=train_config['device'],
		use_dual_kpt_loss=train_config['use_dual_kpt_loss'],

		# Smoke Test
		dry_run=train_config['dry_run'],
		
		# Training directory management
		max_runs=train_config['max_runs']
	)

	trainer.train()
