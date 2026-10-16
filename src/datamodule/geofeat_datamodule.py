"""COCO and MegaDepth data module for GeoFeat Lightning training."""

from __future__ import annotations

import glob
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from tqdm import tqdm
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from . import coco_wrapper
from .coco_augmentor import COCOAugmentor
from .megadepth import MegaDepthDataset


class StepDataset(Dataset[int]):
    """Finite tokens that drive Lightning's step-based training loop."""

    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        return 0


class GeoFeatDataModule(pl.LightningDataModule):
    """Construct COCO online pairs and cyclic MegaDepth training batches."""

    def __init__(
        self,
        use_megadepth: bool,
        use_coco: bool,
        megadepth_root_path: str,
        megadepth_batch_size: int,
        coco_root_path: str,
        coco_batch_size: int,
        training_res: tuple[int, int] | list[int],
        device: str = "cpu",
        num_workers: int = 0,
        steps_per_epoch: int = 1000,
    ) -> None:
        super().__init__()
        self.use_megadepth = use_megadepth
        self.use_coco = use_coco
        self.megadepth_root_path = Path(megadepth_root_path)
        self.megadepth_batch_size = megadepth_batch_size
        self.coco_root_path = Path(coco_root_path)
        self.coco_batch_size = coco_batch_size
        self.training_res = tuple(training_res)
        self.device = torch.device(device)
        self.num_workers = num_workers
        self.steps_per_epoch = steps_per_epoch
        self.augmentor: COCOAugmentor | None = None
        self.megadepth_dataset: ConcatDataset[Any] | None = None
        self.megadepth_loader: DataLoader[Any] | None = None
        self.megadepth_iterator: Iterator[Any] | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "fit"):
            return
        if self.use_coco:
            self.augmentor = COCOAugmentor(
                device=self.device,
                img_dir=str(self.coco_root_path),
                load_dataset=True,
                batch_size=self.coco_batch_size,
                warp_resolution=self.training_res,
                out_resolution=self.training_res,
                sides_crop=0.1,
                max_num_imgs=3000,
                num_test_imgs=5,
                photometric=True,
                geometric=True,
                reload_step=4000,
                rotation_degrees=30.0,
                scale_min=0.35,
                scale_max=1.20,
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.15,
                gamma_min=1.0,
                gamma_max=1.0,
                gamma_probability=0.0,
                equalize_probability=0.40,
                shadow_probability=0.40,
                shadow_min_factor=0.25,
                gaussian_blur_probability=0.30,
                noise_probability=0.50,
                noise_std=10.0 / 255.0,
            )
        if self.use_megadepth:
            metadata_root = self.megadepth_root_path / "megadepth_indices" / "scene_info_0.1_0.7"
            image_root = self.megadepth_root_path / "MegaDepth_v1"
            scene_paths = sorted(glob.glob(str(metadata_root / "*.npz")))
            megadepth_datasets = []
            for scene_path in tqdm(scene_paths, desc="loading MegaDepth"):
                megadepth_datasets.append(
                    MegaDepthDataset(
                        root_dir=str(image_root),
                        npz_path=scene_path,
                        mode="train",
                        min_overlap_score=0.3,
                        max_overlap_score=1.0,
                        img_resize=self.training_res,
                    )
                )
            self.megadepth_dataset = ConcatDataset(megadepth_datasets)
            self.megadepth_loader = DataLoader(
                self.megadepth_dataset,
                batch_size=self.megadepth_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=self.num_workers > 0,
            )
            self.megadepth_iterator = iter(self.megadepth_loader)

    def train_dataloader(self) -> DataLoader[int]:
        if self.megadepth_dataset is None:
            length = self.steps_per_epoch
        else:
            length = math.ceil(len(self.megadepth_dataset) / self.megadepth_batch_size)
        return DataLoader(StepDataset(length), batch_size=None, num_workers=0)

    def make_coco_batch(
        self,
        difficulty: float,
        tps_probability: float,
    ) -> tuple[torch.Tensor, torch.Tensor, Any, Any]:
        return coco_wrapper.make_batch(
            self.augmentor,
            difficulty=difficulty,
            tps_probability=tps_probability,
        )

    def next_megadepth_batch(self) -> dict[str, Any]:
        try:
            return next(self.megadepth_iterator)
        except StopIteration:
            self.megadepth_iterator = iter(self.megadepth_loader)
            return next(self.megadepth_iterator)

    def get_corresponding_points(
        self,
        image0: torch.Tensor,
        image1: torch.Tensor,
        transform0: Any,
        transform1: Any,
        coarse_height: int,
        coarse_width: int,
    ) -> tuple[list[Any], list[Any]]:
        return coco_wrapper.get_corresponding_pts(
            image0,
            image1,
            transform0,
            transform1,
            self.augmentor,
            coarse_height,
            coarse_width,
        )
