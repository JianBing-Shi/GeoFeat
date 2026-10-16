import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.depth_anything_wrapper import DepthAnythingExtractor


class GeometricExtractor(nn.Module):
    """Compute and package depth-derived geometry for GeoFeat.

    The model-facing geometry groups are one depth channel, two gradient
    channels ``[grad_x, grad_y]``, three normal channels
    ``[normal_x, normal_y, normal_z]``, and two compact curvature channels
    ``[H, D]`` (depth-Hessian half-trace and half-eigenvalue-gap proxies).
    Gradients and normals are
    derived from the same depth surface but remain separate branches so their
    different geometric semantics can be encoded independently.
    """

    def __init__(
        self,
        model_config: dict,
        device: Optional[torch.device] = None,
        enable_depth_extractor: bool = True,
    ):
        super().__init__()
        self.geo_cfg = model_config["geometric_features"]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_extractor = None
        if enable_depth_extractor:
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
        gradients = torch.cat([grad_x, grad_y], dim=1)

        # H and D are the depth Hessian's half-trace and half-eigenvalue gap,
        # not calibrated 3D surface curvatures.
        curv_h = (grad_xx + grad_yy) / 2.0
        # The sum-of-squares form avoids cancellation in trace**2 - 4*det.
        discriminant = (grad_xx - grad_yy) ** 2 + 4 * grad_xy ** 2
        curv_d = torch.sqrt(torch.clamp(discriminant, min=1e-6)) / 2.0
        curvatures = torch.cat([curv_h, curv_d], dim=1)

        return {
            "depth": depth,
            "gradients": gradients,
            "normals": normal,
            "curvatures": curvatures,
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
                d = self.depth_extractor.extract_depth(img, target_size=(H, W))
                depth_list.append(d)
            return torch.stack(depth_list).unsqueeze(1)

        if isinstance(image, np.ndarray):
            target_size = (image.shape[0], image.shape[1])
            d = self.depth_extractor.extract_depth(image, target_size=target_size)
            return d.unsqueeze(0).unsqueeze(0)

        raise ValueError("Image must be torch.Tensor or numpy.ndarray")

    def forward(self, depth=None, image=None, return_all: bool = False):
        """Compute geometry from depth or image.

        Args:
            depth: Optional depth map, (B, 1, H, W).
            image: Optional RGB image tensor or numpy image used to infer depth.
            return_all: If true, return all computed geometry terms.
        """
        if depth is None:
            if image is None:
                raise ValueError("Must provide either depth or image")
            depth = self._depth_from_image(image)

        all_features = self.compute_all(depth)
        if return_all:
            return all_features

        geo_features = {}
        if self.geo_cfg["depth"]:
            geo_features["depth"] = all_features["depth"]
        if self.geo_cfg["gradients"]:
            geo_features["gradients"] = all_features["gradients"]
        if self.geo_cfg["normals"]:
            geo_features["normals"] = all_features["normals"]
        if self.geo_cfg["curvatures"]:
            geo_features["curvatures"] = all_features["curvatures"]
        return geo_features

if __name__ == "__main__":
    print("=== Verifying GeometricExtractor ===")
    model_config = {
        "geometric_features": {
            "depth": True,
            "gradients": True,
            "normals": True,
            "curvatures": True,
        }
    }
    print("Initializing GeometricExtractor...")

    extractor = GeometricExtractor(model_config)
    if torch.cuda.is_available():
        extractor = extractor.cuda()
    print("[Success] Initialized")

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
