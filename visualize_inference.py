import argparse
import glob
import os
from typing import Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize GeoFeat inference on an image or a synthetic checkerboard.")
    parser.add_argument("--run-dir", type=str, default="", help="Training run directory containing config_snapshot.json and a best checkpoint.")
    parser.add_argument("--config-path", type=str, default="", help="Path to config_snapshot.json.")
    parser.add_argument("--weights-path", type=str, default="", help="Path to a GeoFeat checkpoint.")
    parser.add_argument("--image-path", type=str, default="", help="Optional input image. If omitted, a synthetic checkerboard is used.")
    parser.add_argument("--output-path", type=str, default="visualize_inference.png", help="Path to save the visualization figure.")
    return parser.parse_args()


def resolve_artifacts(args: argparse.Namespace) -> Tuple[str, str]:
    config_path = args.config_path
    weights_path = args.weights_path

    if args.run_dir:
        run_dir = os.path.abspath(args.run_dir)
        config_path = os.path.join(run_dir, "config_snapshot.json")
        candidates = sorted(glob.glob(os.path.join(run_dir, "*best*.pth")))
        if not candidates:
            candidates = sorted(glob.glob(os.path.join(run_dir, "*.pth")))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found under run directory: {run_dir}")
        weights_path = candidates[-1]

    if not config_path or not weights_path:
        raise ValueError("Provide either --run-dir or both --config-path and --weights-path.")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config snapshot not found: {config_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    return config_path, weights_path


def build_checkerboard(height: int = 480, width: int = 640, step: int = 64):
    import cv2
    import numpy as np

    image = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(0, height, step):
        for x in range(0, width, step):
            if ((x // step) + (y // step)) % 2 == 0:
                image[y:y + step, x:x + step] = 255
    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    return cv2.add(image, noise)


def load_image(image_path: str):
    import cv2

    if image_path:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        return image, image_path
    return build_checkerboard(), "synthetic_checkerboard"


def visualize_keypoints(image, keypoints, scores, output_path: str, title: str) -> None:
    import cv2
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if len(keypoints) > 0:
        plt.scatter(keypoints[:, 0], keypoints[:, 1], c=scores, cmap="jet", s=5, alpha=0.5)
        plt.colorbar(label="Score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    import numpy as np
    from src.utils.geofeat_wrapper import GeoFeat, load_model_config_snapshot

    args = parse_args()
    config_path, weights_path = resolve_artifacts(args)
    model_config = load_model_config_snapshot(config_path)
    image, image_label = load_image(args.image_path)

    print(f"Loading config from {config_path}")
    print(f"Loading weights from {weights_path}")
    geofeat = GeoFeat(model_config, weights_path)
    result = geofeat.extract(image)

    keypoints = result["keypoints"].cpu().numpy()
    scores = result["scores"].cpu().numpy()
    descriptors = result["descriptors"].cpu().numpy()

    print(f"Keypoints detected: {len(keypoints)}")
    if len(keypoints) > 0:
        print(f"First 5 keypoints:\n{keypoints[:5]}")
        print(f"Score stats: Min={scores.min():.4f}, Max={scores.max():.4f}, Mean={scores.mean():.4f}")
        print(f"Descriptor shape: {descriptors.shape}")
        print(
            "Descriptor stats: "
            f"Min={descriptors.min():.4f}, Max={descriptors.max():.4f}, "
            f"Mean={descriptors.mean():.4f}, Norm={np.linalg.norm(descriptors[0]):.4f}"
        )

    visualize_keypoints(
        image,
        keypoints,
        scores,
        args.output_path,
        f"GeoFeat Keypoints ({image_label}, count={len(keypoints)})",
    )
    print(f"Saved visualization to {args.output_path}")


if __name__ == "__main__":
    main()
