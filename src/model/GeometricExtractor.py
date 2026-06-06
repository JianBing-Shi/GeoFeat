import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.depth_anything_wrapper import DepthAnythingExtractor


class GeometricExtractor(nn.Module):
    """Compute and package depth-derived geometry for GeoFeat.

    This class replaces the old split between GeometricComputer and
    GeometricExtractor:
    - compute_all(depth) returns all intermediate geometry terms for losses/debug.
    - forward(..., return_all=False) returns model-facing compact geometry.

    Model-facing gradients are the full first-order geometry group:
    [grad_x, grad_y, normal_x, normal_y, normal_z].
    """

    DEFAULT_GEO_CFG = {
        "depth": True,
        "gradients": True,
        "curvatures": True,
    }

    def __init__(
        self,
        model_config: Optional[dict] = None,
        device: Optional[torch.device] = None,
        enable_depth_extractor: bool = True,
    ):
        super().__init__()
        self.geo_cfg = self._resolve_geo_cfg(model_config)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_depth_extractor = bool(enable_depth_extractor)
        self.depth_extractor = None
        if self.enable_depth_extractor:
            self.depth_extractor = DepthAnythingExtractor("vits", self.device, 256)

        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3),
        )

    @classmethod
    def _resolve_geo_cfg(cls, model_config: Optional[dict]) -> dict:
        if model_config is None:
            return dict(cls.DEFAULT_GEO_CFG)
        if "geometric_features" not in model_config:
            raise KeyError("model_config must contain 'geometric_features'")

        geo_cfg = dict(model_config["geometric_features"])
        resolved = dict(cls.DEFAULT_GEO_CFG)
        resolved.update({
            "depth": bool(geo_cfg.get("depth", resolved["depth"])),
            "gradients": bool(geo_cfg.get("gradients", resolved["gradients"])),
            "curvatures": bool(geo_cfg.get("curvatures", resolved["curvatures"])),
            "normal": False,
        })
        return resolved

    def compute_derivatives(self, depth: torch.Tensor):
        """Compute first- and second-order Sobel derivatives."""
        depth_pad = F.pad(depth, (1, 1, 1, 1), mode="replicate")
        grad_x = F.conv2d(depth_pad, self.sobel_x)
        grad_y = F.conv2d(depth_pad, self.sobel_y)

        grad_x_pad = F.pad(grad_x, (1, 1, 1, 1), mode="replicate")
        grad_y_pad = F.pad(grad_y, (1, 1, 1, 1), mode="replicate")
        grad_xx = F.conv2d(grad_x_pad, self.sobel_x)
        grad_yy = F.conv2d(grad_y_pad, self.sobel_y)
        grad_xy = F.conv2d(grad_x_pad, self.sobel_y)
        return grad_x, grad_y, grad_xx, grad_yy, grad_xy

    def compute_all(self, depth: torch.Tensor) -> dict:
        """Return all depth-derived geometry terms.

        Args:
            depth: Normalized or caller-prepared depth map, (B, 1, H, W).
        """
        depth = depth.to(self.sobel_x.device)
        grad_x, grad_y, grad_xx, grad_yy, grad_xy = self.compute_derivatives(depth)

        normal = torch.cat((-grad_x, -grad_y, torch.ones_like(grad_x)), dim=1)
        normal = F.normalize(normal, p=2, dim=1)

        trace = grad_xx + grad_yy
        det = grad_xx * grad_yy - grad_xy ** 2
        discriminant = torch.sqrt(torch.clamp(trace ** 2 - 4 * det, min=1e-6))
        k1 = (trace + discriminant) / 2.0
        k2 = (trace - discriminant) / 2.0

        gaussian_curv = k1 * k2
        mean_curv = (k1 + k2) / 2.0

        diff = k2 - k1
        diff = torch.where(torch.abs(diff) < 1e-6, torch.ones_like(diff) * 1e-6, diff)
        shape_index = (2.0 / 3.1415926) * torch.atan((k2 + k1) / diff)

        raw_gradients = torch.cat([grad_x, grad_y], dim=1)
        gradients = torch.cat([raw_gradients, normal], dim=1)
        return {
            "depth": depth,
            "grad_x": grad_x,
            "grad_y": grad_y,
            "raw_gradients": raw_gradients,
            "normal": normal,
            "gradients": gradients,
            "curvatures": mean_curv,
            "k1": k1,
            "k2": k2,
            "gaussian_curv": gaussian_curv,
            "mean_curv": mean_curv,
            "shape_index": shape_index,
        }

    def _depth_from_image(self, image):
        if self.depth_extractor is None:
            raise RuntimeError("Depth extractor is disabled; pass depth directly or enable it.")

        if isinstance(image, torch.Tensor):
            _, _, H, W = image.shape
            imgs_np = image.permute(0, 2, 3, 1).detach().cpu().numpy()
            if imgs_np.max() <= 1.0 + 1e-6:
                imgs_np = (imgs_np * 255).astype(np.uint8)
            else:
                imgs_np = imgs_np.astype(np.uint8)

            depth_list = []
            for img in imgs_np:
                d, _, _, _ = self.depth_extractor.extract(img, target_size=(H, W))
                depth_list.append(d)
            return torch.stack(depth_list).unsqueeze(1)

        if isinstance(image, np.ndarray):
            target_size = (image.shape[0], image.shape[1])
            d, _, _, _ = self.depth_extractor.extract(image, target_size=target_size)
            return d.unsqueeze(0).unsqueeze(0)

        raise ValueError("Image must be torch.Tensor or numpy.ndarray")

    def forward(self, depth=None, normal=None, image=None, return_all: bool = False):
        """Compute geometry from depth or image.

        Args:
            depth: Optional depth map, (B, 1, H, W).
            normal: Unused legacy argument kept for caller compatibility.
            image: Optional RGB image tensor or numpy image used to infer depth.
            return_all: If true, return all computed geometry terms.
        """
        del normal
        if depth is None:
            if image is None:
                raise ValueError("Must provide either depth or image")
            depth = self._depth_from_image(image)

        all_features = self.compute_all(depth.to(self.device))
        if return_all:
            return all_features

        geo_features = {}
        if self.geo_cfg["depth"]:
            geo_features["depth"] = all_features["depth"]
        if self.geo_cfg["gradients"]:
            geo_features["gradients"] = all_features["gradients"]
        if self.geo_cfg["curvatures"]:
            geo_features["curvatures"] = all_features["curvatures"]
        return geo_features

if __name__ == "__main__":
    print("=== Verifying GeometricExtractor ===")
    model_config = {
        "geometric_features": {
            "depth": True,
            "gradients": True,
            "curvatures": True,
        }
    }

    print("Initializing GeometricExtractor...")
    try:
        extractor = GeometricExtractor(model_config)
        if torch.cuda.is_available():
            extractor = extractor.cuda()
        print("[Success] Initialized")
    except Exception as e:
        print(f"[Error] Initialization failed: {e}")
        sys.exit(1)

    H, W = 640, 800
    dummy_img = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)

    print(f"Running inference on image shape {dummy_img.shape}...")
    features = extractor(image=dummy_img)

    print("[Success] Inference complete")
    print("Output features:")
    for k, v in features.items():
        print(f"  {k}: {v.shape}")

    if features["depth"].shape[-2:] == (H, W):
        print("[Pass] Output resolution matches input")
    else:
        print(f"[Fail] Output resolution mismatch: {features['depth'].shape[-2:]} vs {(H, W)}")
