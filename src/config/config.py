from yacs.config import CfgNode as CN

## Configuration for data paths and settings
cfg = CN()

##############  ↓  DATA CONFIG  ↓  ##############
cfg.DATASET = CN()
cfg.DATASET.NUM_WORKERS = 0
cfg.DATASET.USE_MEGADEPTH = True
cfg.DATASET.USE_COCO = True
cfg.DATASET.MEGADEPTH_ROOT_PATH = "./datasets/MegaDepth"
cfg.DATASET.MEGADEPTH_BATCH_SIZE = 2
cfg.DATASET.COCO_ROOT_PATH = "./datasets/coco_20k"
cfg.DATASET.COCO_BATCH_SIZE = 2

cfg.TRAIN = CN()
cfg.TRAIN.USE_DUAL_KPT_LOSS = False

#############  ↓  MODEL CONFIG  ↓  ##############
cfg.MODEL = CN()

# Backbone options: "Standard", "RepVGG"
cfg.MODEL.BACKBONE = "Standard"        

cfg.MODEL.KEYPOINT_ENCODER = [128, 64, 64]
cfg.MODEL.KEYPOINT_DIM = 65

cfg.MODEL.DESCRIPTOR_ENCODER = [96]
cfg.MODEL.DESCRIPTOR_DIM = 96

# Geometry feature switches. Mainline compact geometry uses:
# - depth: 1 channel
# - gradients: 5 channels [dx, dy, normal_x, normal_y, normal_z]
# - curvatures: 1 channel mean curvature
# Normal is merged into GRADIENTS because both are first-order geometry.
cfg.MODEL.GEOMETRIC_FEATURES = CN()
cfg.MODEL.GEOMETRIC_FEATURES.DEPTH = True
cfg.MODEL.GEOMETRIC_FEATURES.GRADIENTS = True
cfg.MODEL.GEOMETRIC_FEATURES.CURVATURES = True

cfg.MODEL.DEPTH_ENCODER = [32, 48]
cfg.MODEL.DEPTH_DIM = 96

cfg.MODEL.GRADIENT_ENCODER = [48, 64]
cfg.MODEL.GRADIENT_DIM = 96

cfg.MODEL.CURVATURE_ENCODER = [32, 48]
cfg.MODEL.CURVATURE_DIM = 96

# Geometry preprocessing mode before FeatureBooster:
# - "unfold": original 8x8 unfold to tokenized geometry channels
# - "no_unfold_lite": resize geometry map to descriptor resolution directly
cfg.MODEL.GEO_PREPROCESS_MODE = "no_unfold_lite"

# Attention configuration
cfg.MODEL.ATTENTIONAL_LAYERS = 2
cfg.MODEL.AFT = CN()
cfg.MODEL.AFT.NUM_GROUPS = 4
cfg.MODEL.AFT.LOCAL_KERNEL_SIZES = [3, 5]
cfg.MODEL.AFT.DOMAIN_ADAPT = True
cfg.MODEL.AFT.TEMPERATURE_MIN = 0.70
cfg.MODEL.AFT.TEMPERATURE_MAX = 1.60
cfg.MODEL.AFT.LOCAL_BASE_GAIN = 1.0
cfg.MODEL.AFT.CORR_BASE_GAIN = 0.5

# Geometry reliability gate:
# raw geometry + encoded geometry memory -> reliability-filtered additive geometry.
cfg.MODEL.GEO_RELIABILITY_GATE = CN()
cfg.MODEL.GEO_RELIABILITY_GATE.ENABLED = True
cfg.MODEL.GEO_RELIABILITY_GATE.HIDDEN_DIM = 12
cfg.MODEL.GEO_RELIABILITY_GATE.GATE_FLOOR = 0.20
cfg.MODEL.GEO_RELIABILITY_GATE.BIAS_INIT = 2.0
cfg.MODEL.GEO_RELIABILITY_GATE.FUSION_GAIN = 0.50

# Refiner options: "None", "GCDR"
cfg.MODEL.REFINER_TYPE = "GCDR"
# GCDR mainline knobs (Geo-Conditional Descriptor Refinement).
# Fixed path: geometry compression -> reliability-aware FiLM -> residual gated injection.
cfg.MODEL.GCDR = CN()
# 压缩几何特征通道数，控制几何条件分支的信息瓶颈与计算量。
cfg.MODEL.GCDR.GEO_COMPRESS = 12      # geo compression channels (7->8 for no_unfold_lite)
# reliability head 的隐藏通道数，决定可靠性估计分支的容量。
cfg.MODEL.GCDR.RELIABILITY_HIDDEN = 12
# FiLM 门控下限，避免 reliability 很低时几何调制完全退化为 0。
cfg.MODEL.GCDR.GATE_FLOOR = 0.10
# 几何残差注入强度，值越大表示 descriptor refinement 越激进。
cfg.MODEL.GCDR.RESIDUAL_GAIN = 1.0
# residual gate 的最小保留比例，保证残差路径不会被完全关闭。
cfg.MODEL.GCDR.RESIDUAL_GATE_FLOOR = 0.10
cfg.MODEL.GCDR.RESIDUAL_CLIP = 0.25

cfg.LOSS = CN()
# reliability supervision 的总权重；为 0 时表示不显式监督可靠性图。
cfg.LOSS.RELIABILITY_LOSS_WEIGHT = 0.0
# 配对点两侧 reliability map 的一致性约束权重。
cfg.LOSS.RELIABILITY_CONSISTENCY_WEIGHT = 0.25
# 将 refined/raw confidence 差值映射为 reliability target 时的 sigmoid 缩放系数。
cfg.LOSS.RELIABILITY_TARGET_SCALE = 6.0
# 重复性一致性损失权重，约束配对图像间的关键点检测稳定性。
cfg.LOSS.REPEATABILITY_LOSS_WEIGHT = 0.0
cfg.LOSS.DEPTH_WEIGHT = 1.0
cfg.LOSS.GRADIENT_WEIGHT = 2.0
cfg.LOSS.CURVATURE_WEIGHT = 1.0
cfg.LOSS.DESCRIPTOR_PROTECT_WEIGHT = 0.10
cfg.LOSS.RELIABLE_ENHANCE_MARGIN = 0.02
cfg.LOSS.RELIABLE_ENHANCE_WEIGHT = 0.05
cfg.LOSS.RELIABLE_ENHANCE_THRESHOLD = 0.50
cfg.LOSS.RELIABLE_ENHANCE_QUANTILE = 0.50
cfg.LOSS.UNRELIABLE_SUPPRESS_WEIGHT = 0.05
cfg.LOSS.UNRELIABLE_SUPPRESS_THRESHOLD = 0.45
cfg.LOSS.UNRELIABLE_SUPPRESS_QUANTILE = 0.25
# 重复性图的权重系数，控制其对关键点预测的影响程度
# ALPHA小：更依赖原始关键点预测；
# ALPHA大：更依赖重复性图进行调整
# Weight for fusing repeatability_map into keypoint logits in forward2.
cfg.MODEL.REPEATABILITY_ALPHA = 0.35

# Final activation for descriptors: "None", "relu", "sigmoid", "tanh"
cfg.MODEL.LAST_ACTIVATION = "None"

cfg.MODEL.OUTPUT_DIM = 96

def get_data_config(cfg_node=None):
	"""Build the runtime data config dict used by entrypoints."""
	active_cfg = cfg if cfg_node is None else cfg_node
	return {
		'use_megadepth': bool(active_cfg.DATASET.USE_MEGADEPTH),
		'use_coco': bool(active_cfg.DATASET.USE_COCO),
		'megadepth_root_path': str(active_cfg.DATASET.MEGADEPTH_ROOT_PATH),
		'megadepth_batch_size': int(active_cfg.DATASET.MEGADEPTH_BATCH_SIZE),
		'coco_root_path': str(active_cfg.DATASET.COCO_ROOT_PATH),
		'coco_batch_size': int(active_cfg.DATASET.COCO_BATCH_SIZE),
	}


def get_loss_config(cfg_node=None):
	"""Build the runtime loss config dict used by training entrypoints."""
	active_cfg = cfg if cfg_node is None else cfg_node
	return {
		'reliability_loss_weight': float(active_cfg.LOSS.RELIABILITY_LOSS_WEIGHT),
		'reliability_consistency_weight': float(active_cfg.LOSS.RELIABILITY_CONSISTENCY_WEIGHT),
		'reliability_target_scale': float(active_cfg.LOSS.RELIABILITY_TARGET_SCALE),
		'repeatability_loss_weight': float(active_cfg.LOSS.REPEATABILITY_LOSS_WEIGHT),
		'depth_weight': float(active_cfg.LOSS.DEPTH_WEIGHT),
		'gradient_weight': float(active_cfg.LOSS.GRADIENT_WEIGHT),
		'curvature_weight': float(active_cfg.LOSS.CURVATURE_WEIGHT),
		'descriptor_protect_weight': float(active_cfg.LOSS.DESCRIPTOR_PROTECT_WEIGHT),
		'reliable_enhance_margin': float(active_cfg.LOSS.RELIABLE_ENHANCE_MARGIN),
		'reliable_enhance_weight': float(active_cfg.LOSS.RELIABLE_ENHANCE_WEIGHT),
		'reliable_enhance_threshold': float(active_cfg.LOSS.RELIABLE_ENHANCE_THRESHOLD),
		'reliable_enhance_quantile': float(active_cfg.LOSS.RELIABLE_ENHANCE_QUANTILE),
		'unreliable_suppress_weight': float(active_cfg.LOSS.UNRELIABLE_SUPPRESS_WEIGHT),
		'unreliable_suppress_threshold': float(active_cfg.LOSS.UNRELIABLE_SUPPRESS_THRESHOLD),
		'unreliable_suppress_quantile': float(active_cfg.LOSS.UNRELIABLE_SUPPRESS_QUANTILE),
	}


def get_model_config(cfg_node=None):
	"""Build the canonical GeoFeat model config dict for all mainline entrypoints."""
	active_cfg = cfg if cfg_node is None else cfg_node
	return {
		'backbone': str(active_cfg.MODEL.BACKBONE),
		'keypoint_encoder': list(active_cfg.MODEL.KEYPOINT_ENCODER),
		'keypoint_dim': int(active_cfg.MODEL.KEYPOINT_DIM),
		'descriptor_encoder': list(active_cfg.MODEL.DESCRIPTOR_ENCODER),
		'descriptor_dim': int(active_cfg.MODEL.DESCRIPTOR_DIM),
		'geometric_features': {
			'depth': bool(active_cfg.MODEL.GEOMETRIC_FEATURES.DEPTH),
			# Legacy compatibility field. Standalone normal is disabled because
			# normals are now packed into gradients as first-order geometry.
			'normal': False,
			'gradients': bool(active_cfg.MODEL.GEOMETRIC_FEATURES.GRADIENTS),
			'curvatures': bool(active_cfg.MODEL.GEOMETRIC_FEATURES.CURVATURES),
		},
		'depth_encoder': list(active_cfg.MODEL.DEPTH_ENCODER),
		'depth_dim': int(active_cfg.MODEL.DEPTH_DIM),
		'gradient_encoder': list(active_cfg.MODEL.GRADIENT_ENCODER),
		'gradient_dim': int(active_cfg.MODEL.GRADIENT_DIM),
		'curvature_encoder': list(active_cfg.MODEL.CURVATURE_ENCODER),
		'curvature_dim': int(active_cfg.MODEL.CURVATURE_DIM),
		'geo_preprocess_mode': str(active_cfg.MODEL.GEO_PREPROCESS_MODE),
		'repeatability_alpha': float(active_cfg.MODEL.REPEATABILITY_ALPHA),
		'attention_layers': int(active_cfg.MODEL.ATTENTIONAL_LAYERS),
		'aft_config': {
			'num_groups': int(active_cfg.MODEL.AFT.NUM_GROUPS),
			'local_kernel_sizes': list(active_cfg.MODEL.AFT.LOCAL_KERNEL_SIZES),
			'domain_adapt': bool(active_cfg.MODEL.AFT.DOMAIN_ADAPT),
			'temperature_min': float(active_cfg.MODEL.AFT.TEMPERATURE_MIN),
			'temperature_max': float(active_cfg.MODEL.AFT.TEMPERATURE_MAX),
			'local_base_gain': float(active_cfg.MODEL.AFT.LOCAL_BASE_GAIN),
			'corr_base_gain': float(active_cfg.MODEL.AFT.CORR_BASE_GAIN),
		},
		'geo_reliability_gate': {
			'enabled': bool(active_cfg.MODEL.GEO_RELIABILITY_GATE.ENABLED),
			'hidden_dim': int(active_cfg.MODEL.GEO_RELIABILITY_GATE.HIDDEN_DIM),
			'gate_floor': float(active_cfg.MODEL.GEO_RELIABILITY_GATE.GATE_FLOOR),
			'bias_init': float(active_cfg.MODEL.GEO_RELIABILITY_GATE.BIAS_INIT),
			'fusion_gain': float(active_cfg.MODEL.GEO_RELIABILITY_GATE.FUSION_GAIN),
		},
		'refiner_type': str(active_cfg.MODEL.REFINER_TYPE),
		'gcdr_config': {
			'geo_compress': int(active_cfg.MODEL.GCDR.GEO_COMPRESS),
			'reliability_hidden': int(active_cfg.MODEL.GCDR.RELIABILITY_HIDDEN),
			'gate_floor': float(active_cfg.MODEL.GCDR.GATE_FLOOR),
			'residual_gain': float(active_cfg.MODEL.GCDR.RESIDUAL_GAIN),
			'residual_gate_floor': float(active_cfg.MODEL.GCDR.RESIDUAL_GATE_FLOOR),
			'residual_clip': float(active_cfg.MODEL.GCDR.RESIDUAL_CLIP),
		},
		'loss_config': get_loss_config(active_cfg),
		'last_activation': str(active_cfg.MODEL.LAST_ACTIVATION),
		'output_dim': int(active_cfg.MODEL.OUTPUT_DIM),
	}

def get_cfg_defaults():
	"""Get a yacs CfgNode object with default values for my_project."""
	return cfg.clone()
