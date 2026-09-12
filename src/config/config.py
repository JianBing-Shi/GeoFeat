"""Public GeoFeat configuration."""
from yacs.config import CfgNode as CN

_CN = CN()

# Dataset
_CN.DATASET = CN()
_CN.DATASET.USE_MEGADEPTH = True
_CN.DATASET.USE_COCO = True
_CN.DATASET.MEGADEPTH_ROOT_PATH = "./datasets/MegaDepth"
_CN.DATASET.MEGADEPTH_BATCH_SIZE = 8
_CN.DATASET.COCO_ROOT_PATH = "./datasets/coco_20k"
_CN.DATASET.COCO_BATCH_SIZE = 4

# Training
_CN.TRAINER = CN()
_CN.TRAINER.MODEL_NAME = "GeoFeat"
_CN.TRAINER.N_STEPS = 160_000
_CN.TRAINER.LR = 3e-4
_CN.TRAINER.GAMMA_STEPLR = 0.5
_CN.TRAINER.TRAINING_RES = (800, 600)
_CN.TRAINER.DEVICE = "cuda:0"
_CN.TRAINER.CKPT_SAVE_PATH = "weights"
_CN.TRAINER.SAVE_CKPT_EVERY = 1000
_CN.TRAINER.USE_DUAL_KPT_LOSS = True

_CN.TRAINER.DRY_RUN = True
_CN.TRAINER.MAX_RUNS = 10


# Public model options
_CN.MODEL = CN()
_CN.MODEL.BACKBONE = "RepVGG"
_CN.MODEL.BACKBONE_CHANNELS = [20, 20, 40, 80, 96]
_CN.MODEL.KEYPOINT_ENCODER = [128, 64, 64]
_CN.MODEL.KEYPOINT_DIM = 65
_CN.MODEL.DESCRIPTOR_ENCODER = [96]
_CN.MODEL.DESCRIPTOR_DIM = 96
_CN.MODEL.GEOMETRIC_FEATURES = CN()
_CN.MODEL.GEOMETRIC_FEATURES.DEPTH = True
_CN.MODEL.GEOMETRIC_FEATURES.GRADIENTS = True
_CN.MODEL.GEOMETRIC_FEATURES.CURVATURES = True

# Controlled ablations
_CN.ABLATION = CN()
_CN.ABLATION.DISABLE_GCDR = False
_CN.ABLATION.DISABLE_GEO_HEAD = False


def get_cfg_defaults():
    return _CN.clone()


def get_data_config(cfg):
    data = cfg.DATASET
    return {
        "use_megadepth": data.USE_MEGADEPTH,
        "use_coco": data.USE_COCO,
        "megadepth_root_path": data.MEGADEPTH_ROOT_PATH,
        "megadepth_batch_size": data.MEGADEPTH_BATCH_SIZE,
        "coco_root_path": data.COCO_ROOT_PATH,
        "coco_batch_size": data.COCO_BATCH_SIZE,
    }


def get_train_config(cfg):
    train = cfg.TRAINER
    return {
        "model_name": train.MODEL_NAME,
        "n_steps": train.N_STEPS,
        "lr": train.LR,
        "gamma_steplr": train.GAMMA_STEPLR,
        "training_res": tuple(train.TRAINING_RES),
        "device": train.DEVICE,
        "ckpt_save_path": train.CKPT_SAVE_PATH,
        "save_ckpt_every": train.SAVE_CKPT_EVERY,
        "use_dual_kpt_loss": train.USE_DUAL_KPT_LOSS,
        "dry_run": train.DRY_RUN,
        "max_runs": train.MAX_RUNS,
    }


def get_model_config(cfg):
	model = cfg.MODEL
	model_config = {
		"refiner_type": "GCDR",
		"backbone": model.BACKBONE,
		"backbone_channels": list(model.BACKBONE_CHANNELS),
		"keypoint_encoder": list(model.KEYPOINT_ENCODER),
		"keypoint_dim": model.KEYPOINT_DIM,
		"descriptor_encoder": list(model.DESCRIPTOR_ENCODER),
		"descriptor_dim": model.DESCRIPTOR_DIM,
		"geometric_features": {
		"depth": model.GEOMETRIC_FEATURES.DEPTH,
		"normal": False,
		"gradients": model.GEOMETRIC_FEATURES.GRADIENTS,
		"curvatures": model.GEOMETRIC_FEATURES.CURVATURES,
		},
	}

	if cfg.ABLATION.DISABLE_GCDR:
		model_config["refiner_type"] = "None"
	if cfg.ABLATION.DISABLE_GEO_HEAD:
		model_config["geometric_features"] = {
			"depth": False,
			"normal": False,
			"gradients": False,
			"curvatures": False,
		}
		model_config["refiner_type"] = "None"
	return model_config
