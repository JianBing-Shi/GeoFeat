from argparse import ArgumentParser

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from src.datamodule import GeoFeatDataModule
from src.model import GeoFeat
from src.config import load_config

import setproctitle
import os

CURRENT_FILE_PATH = os.path.realpath(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_FILE_PATH)
os.environ["PROJECT_ROOT"] = PROJECT_ROOT

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(1024, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='src/config/train.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    setproctitle.setproctitle(config['model_name'])
    
    datamodule = GeoFeatDataModule(**config['datamodule'])
    periodic_checkpoint = ModelCheckpoint(**config['trainer']['ckpt'])
    best_checkpoint = ModelCheckpoint(**config['trainer']['best_ckpt'])
    lr_monitor = LearningRateMonitor(**config['trainer']['lr_monitor'])
    csv_logger = CSVLogger(**config['trainer']['csv_logger'])
    trainer = pl.Trainer(
        strategy=config['trainer']['strategy'],
        devices=config['trainer']['devices'],
        accelerator=config['trainer']['accelerator'],
        callbacks=[periodic_checkpoint, best_checkpoint, lr_monitor],
        max_steps=config['trainer']['max_steps'],
        logger=csv_logger
    )

    # Load the model from a checkpoint if provided
    if config['trainer']['ckpt_path']:
        print(f"Resume training from checkpoint: {config['trainer']['ckpt_path']}")
        model = GeoFeat.load_from_checkpoint(config['trainer']['ckpt_path'], **config['model'])
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=config['trainer']['ckpt_path'])
    else:
        print("No checkpoint path provided, initializing a new model.")
        model = GeoFeat(**config['model'])
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=config['trainer']['ckpt_path'])
