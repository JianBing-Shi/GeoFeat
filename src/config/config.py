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

##############  ↓  MODEL CONFIG  ↓  ##############
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

# Refiner options: "None", "Local", "GCLA"
# "Local" uses simple depthwise separable conv
# "GCLA" uses Geometric-Coupled Local Aggregation (Requires geometric features)
cfg.MODEL.REFINER_TYPE = "GCLA"
# 重复性图的权重系数，控制其对关键点预测的影响程度
# ALPHA小：更依赖原始关键点预测；
# ALPHA大：更依赖重复性图进行调整
# Weight for fusing repeatability_map into keypoint logits in forward2.
cfg.MODEL.REPEATABILITY_ALPHA = 0.35

# Final activation for descriptors: "None", "relu", "sigmoid", "tanh"
cfg.MODEL.LAST_ACTIVATION = "None"

cfg.MODEL.OUTPUT_DIM = 64

def get_cfg_defaults():
	"""Get a yacs CfgNode object with default values for my_project."""
	return cfg.clone()
