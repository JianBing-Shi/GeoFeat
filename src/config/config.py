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

cfg.MODEL.KEYPOINT_DIM = 65
cfg.MODEL.KEYPOINT_ENCODER = [128, 64, 64]

cfg.MODEL.DESCRIPTOR_DIM = 64
cfg.MODEL.DESCRIPTOR_ENCODER = [64, 64]

# Normal, Gradient, and Curvature feature encoders
cfg.MODEL.GEOMETRIC_FEATURES = CN()
cfg.MODEL.GEOMETRIC_FEATURES.DEPTH = True
cfg.MODEL.GEOMETRIC_FEATURES.NORMAL = True
cfg.MODEL.GEOMETRIC_FEATURES.GRADIENTS = True
cfg.MODEL.GEOMETRIC_FEATURES.CURVATURES = True

cfg.MODEL.DEPTH_ENCODER = [128, 64, 64]
cfg.MODEL.DEPTH_DIM = 64

cfg.MODEL.NORMAL_ENCODER = [128, 64, 64]
cfg.MODEL.NORMAL_DIM = 192

cfg.MODEL.GRADIENT_ENCODER = [64, 32, 32]
cfg.MODEL.GRADIENT_DIM = 128

cfg.MODEL.CURVATURE_ENCODER = [64, 32, 32]
cfg.MODEL.CURVATURE_DIM = 64

# Geometry preprocessing mode before FeatureBooster:
# - "unfold": original 8x8 unfold to tokenized geometry channels
# - "no_unfold_lite": resize geometry map to descriptor resolution directly
cfg.MODEL.GEO_PREPROCESS_MODE = "no_unfold_lite"

# Attention configuration
cfg.MODEL.ATTENTIONAL_LAYERS = 3

# Refiner options: "None", "GCDR"
cfg.MODEL.REFINER_TYPE = "GCDR"
# GCDR mainline knobs (Geo-Conditional Descriptor Refinement).
# Fixed path: geometry compression -> reliability-aware FiLM -> residual gated injection.
cfg.MODEL.GCDR = CN()
# 压缩几何特征通道数，控制几何条件分支的信息瓶颈与计算量。
cfg.MODEL.GCDR.GEO_COMPRESS = 16      # geo compression channels (7->8 for no_unfold_lite)
# reliability head 的隐藏通道数，决定可靠性估计分支的容量。
cfg.MODEL.GCDR.RELIABILITY_HIDDEN = 16
# FiLM 门控下限，避免 reliability 很低时几何调制完全退化为 0。
cfg.MODEL.GCDR.GATE_FLOOR = 0.10
# 几何残差注入强度，值越大表示 descriptor refinement 越激进。
cfg.MODEL.GCDR.RESIDUAL_GAIN = 1.0
# residual gate 的最小保留比例，保证残差路径不会被完全关闭。
cfg.MODEL.GCDR.RESIDUAL_GATE_FLOOR = 0.10

cfg.LOSS = CN()
# reliability supervision 的总权重；为 0 时表示不显式监督可靠性图。
cfg.LOSS.RELIABILITY_LOSS_WEIGHT = 0.0
# 配对点两侧 reliability map 的一致性约束权重。
cfg.LOSS.RELIABILITY_CONSISTENCY_WEIGHT = 0.25
# 将 refined/raw confidence 差值映射为 reliability target 时的 sigmoid 缩放系数。
cfg.LOSS.RELIABILITY_TARGET_SCALE = 6.0
# 重复性图的权重系数，控制其对关键点预测的影响程度
# ALPHA小：更依赖原始关键点预测；
# ALPHA大：更依赖重复性图进行调整
# Weight for fusing repeatability_map into keypoint logits in forward2.
cfg.MODEL.REPEATABILITY_ALPHA = 0.35

# Final activation for descriptors: "None", "relu", "sigmoid", "tanh"
cfg.MODEL.LAST_ACTIVATION = "None"

cfg.MODEL.OUTPUT_DIM = 64

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
			'normal': bool(active_cfg.MODEL.GEOMETRIC_FEATURES.NORMAL),
			'gradients': bool(active_cfg.MODEL.GEOMETRIC_FEATURES.GRADIENTS),
			'curvatures': bool(active_cfg.MODEL.GEOMETRIC_FEATURES.CURVATURES),
		},
		'depth_encoder': list(active_cfg.MODEL.DEPTH_ENCODER),
		'depth_dim': int(active_cfg.MODEL.DEPTH_DIM),
		'normal_encoder': list(active_cfg.MODEL.NORMAL_ENCODER),
		'normal_dim': int(active_cfg.MODEL.NORMAL_DIM),
		'gradient_encoder': list(active_cfg.MODEL.GRADIENT_ENCODER),
		'gradient_dim': int(active_cfg.MODEL.GRADIENT_DIM),
		'curvature_encoder': list(active_cfg.MODEL.CURVATURE_ENCODER),
		'curvature_dim': int(active_cfg.MODEL.CURVATURE_DIM),
		'geo_preprocess_mode': str(active_cfg.MODEL.GEO_PREPROCESS_MODE),
		'repeatability_alpha': float(active_cfg.MODEL.REPEATABILITY_ALPHA),
		'attention_layers': int(active_cfg.MODEL.ATTENTIONAL_LAYERS),
		'refiner_type': str(active_cfg.MODEL.REFINER_TYPE),
		'gcdr_config': {
			'geo_compress': int(active_cfg.MODEL.GCDR.GEO_COMPRESS),
			'reliability_hidden': int(active_cfg.MODEL.GCDR.RELIABILITY_HIDDEN),
			'gate_floor': float(active_cfg.MODEL.GCDR.GATE_FLOOR),
			'residual_gain': float(active_cfg.MODEL.GCDR.RESIDUAL_GAIN),
			'residual_gate_floor': float(active_cfg.MODEL.GCDR.RESIDUAL_GATE_FLOOR),
		},
		'loss_config': get_loss_config(active_cfg),
		'last_activation': str(active_cfg.MODEL.LAST_ACTIVATION),
		'output_dim': int(active_cfg.MODEL.OUTPUT_DIM),
	}

def get_cfg_defaults():
	"""Get a yacs CfgNode object with default values for my_project."""
	return cfg.clone()
