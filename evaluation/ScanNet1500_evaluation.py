import argparse
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast


parser = argparse.ArgumentParser(description='ScanNet1500 evaluation based on the LoFTR ScanNet protocol.')
parser.add_argument('--name', type=str, default='auto', help='Experiment name shown in the summary output.')
parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
parser.add_argument('--weights', type=str, required=True, help='Path to checkpoint (.pth)')
parser.add_argument('--matcher', type=str, default='auto', choices=['auto', 'liftfeat', 'geofeat'], help='Matcher backend selection.')
parser.add_argument('--top_k', type=int, default=4096, help='Maximum number of detected keypoints to keep.')
parser.add_argument('--detect-threshold', type=float, default=0.05, help='Keypoint detection threshold.')
parser.add_argument('--nms-kernel-size', type=int, default=7, help='GeoFeat NMS kernel size.')
parser.add_argument('--match-min-cossim', type=float, default=0.0, help='Minimum cosine similarity for keeping a match.')
parser.add_argument('--match-ratio-thresh', type=float, default=0.0, help='Optional Lowe-style ratio threshold. <=0 disables it.')
parser.add_argument('--ransac-thr', type=float, default=0.5, help='LoFTR-style ScanNet RANSAC pixel threshold.')
parser.add_argument('--max-pairs', type=int, default=0, help='Optional cap for smoke tests; 0 means full 1500 pairs.')
parser.add_argument('--save-json', type=str, default='', help='Optional path to save the evaluation summary as JSON.')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import cv2
import numpy as np
import torch
import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

LIFTFEAT_ROOT = ROOT / 'LiftFeat'
if str(LIFTFEAT_ROOT) not in sys.path:
    sys.path.append(str(LIFTFEAT_ROOT))

from src.utils.geofeat_wrapper import GeoFeat, load_model_config_for_weights  # noqa: E402


LiftFeatModel = importlib.import_module('models.liftfeat_wrapper').LiftFeat

DATASET_ROOT = ROOT / 'data' / 'scannet_test_1500'
ANNOTATION_PATH = ROOT / 'data' / 'scannet_1500.npz'
INTRINSIC_PATH = ROOT / 'LoFTR' / 'assets' / 'scannet_test_1500' / 'intrinsics.npz'
IMAGE_SIZE = (640, 480)
EPI_ERR_THR = 5e-4
RANSAC_CONF = 0.99999


def read_scannet_gray_as_bgr(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f'Failed to read image: {image_path}')
    image = cv2.resize(image, IMAGE_SIZE)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def read_world_to_camera(pose_path: Path) -> np.ndarray:
    cam_to_world = np.loadtxt(str(pose_path), delimiter=' ')
    return np.linalg.inv(cam_to_world)


def relative_pose_error(T_0to1: np.ndarray, R: np.ndarray, t: np.ndarray, ignore_gt_t_thr: float = 0.0) -> Tuple[float, float]:
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180.0 - t_err)
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:
        t_err = 0.0

    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1.0) / 2.0
    cos = np.clip(cos, -1.0, 1.0)
    R_err = np.rad2deg(np.abs(np.arccos(cos)))
    return float(t_err), float(R_err)


def error_auc(errors: List[float], thresholds: List[int]) -> Dict[str, float]:
    errors = [0.0] + sorted(list(errors))
    recall = list(np.linspace(0.0, 1.0, len(errors)))
    aucs = []
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index - 1]]
        x = errors[:last_index] + [thr]
        aucs.append(float(np.trapezoid(y, x) / thr))
    return {f'auc@{thr}': auc for thr, auc in zip(thresholds, aucs)}


def normalize_keypoints(kpts: np.ndarray, K: np.ndarray) -> np.ndarray:
    return (kpts - K[[0, 1], [2, 2]][None]) / K[[0, 1], [0, 1]][None]


def estimate_pose_loftr(kpts0: np.ndarray, kpts1: np.ndarray, K0: np.ndarray, K1: np.ndarray, thresh: float, conf: float) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if len(kpts0) < 5 or len(kpts1) < 5:
        return None

    kpts0 = normalize_keypoints(kpts0, K0)
    kpts1 = normalize_keypoints(kpts1, K1)
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    E, mask = cv2.findEssentialMat(
        kpts0,
        kpts1,
        np.eye(3),
        threshold=float(ransac_thr),
        prob=conf,
        method=cv2.RANSAC,
    )
    if E is None:
        return None

    best_ret = None
    best_num_inliers = -1
    for candidate in np.split(E, len(E) // 3):
        recovered = cast(Any, cv2.recoverPose(candidate, kpts0, kpts1, np.eye(3), 1e9, mask=mask))
        if len(recovered) == 4:
            num_inliers, R, t, pose_mask = recovered
        else:
            num_inliers, R, t, pose_mask, _ = recovered
        if int(num_inliers) > best_num_inliers:
            best_num_inliers = int(num_inliers)
            best_ret = (R, t[:, 0], pose_mask.ravel() > 0)
    return best_ret


def symmetric_epipolar_distance(kpts0: np.ndarray, kpts1: np.ndarray, T_0to1: np.ndarray, K0: np.ndarray, K1: np.ndarray) -> np.ndarray:
    if len(kpts0) == 0 or len(kpts1) == 0:
        return np.empty((0,), dtype=np.float64)

    pts0 = normalize_keypoints(kpts0, K0)
    pts1 = normalize_keypoints(kpts1, K1)
    pts0_h = np.concatenate([pts0, np.ones((len(pts0), 1), dtype=np.float64)], axis=1)
    pts1_h = np.concatenate([pts1, np.ones((len(pts1), 1), dtype=np.float64)], axis=1)

    t = T_0to1[:3, 3]
    tx = np.array([
        [0.0, -t[2], t[1]],
        [t[2], 0.0, -t[0]],
        [-t[1], t[0], 0.0],
    ], dtype=np.float64)
    E = tx @ T_0to1[:3, :3]

    Ep0 = pts0_h @ E.T
    Etp1 = pts1_h @ E
    p1Ep0 = np.sum(pts1_h * Ep0, axis=1)
    denom = (1.0 / (Ep0[:, 0] ** 2 + Ep0[:, 1] ** 2 + 1e-12)) + (1.0 / (Etp1[:, 0] ** 2 + Etp1[:, 1] ** 2 + 1e-12))
    return (p1Ep0 ** 2) * denom


def epidist_prec(errors: List[np.ndarray], threshold: float) -> float:
    precisions = []
    for errs in errors:
        if len(errs) == 0:
            precisions.append(0.0)
        else:
            precisions.append(float(np.mean(errs < threshold)))
    return float(np.mean(precisions)) if precisions else 0.0


class LiftFeatMatcher:
    def __init__(self, weight_path: str, top_k: int, detect_threshold: float):
        self.model = LiftFeatModel(weight=weight_path, top_k=top_k, detect_threshold=detect_threshold)

    def match(self, image0: np.ndarray, image1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.model.match_liftfeat(image0, image1)


def build_matcher(weight_path: str):
    matcher_name = args.matcher.lower()
    snapshot_path = Path(weight_path).resolve().parent / 'config_snapshot.json'

    if matcher_name == 'geofeat' or (matcher_name == 'auto' and snapshot_path.exists()):
        model_config = load_model_config_for_weights(weight_path)
        matcher = GeoFeat(
            model_config=model_config,
            weight_path=weight_path,
            top_k=args.top_k,
            detect_threshold=args.detect_threshold,
            nms_kernel_size=args.nms_kernel_size,
            match_min_cossim=args.match_min_cossim,
            match_ratio_thresh=args.match_ratio_thresh,
        )
        return 'GeoFeat', matcher.match_featnet

    matcher = LiftFeatMatcher(weight_path=weight_path, top_k=args.top_k, detect_threshold=args.detect_threshold)
    return 'LiftFeat', matcher.match


class ScanNet1500:
    def __init__(self, dataset_root: Path, annotation_path: Path, intrinsic_path: Path, max_pairs: int = 0):
        if not dataset_root.exists():
            raise FileNotFoundError(f'Dataset root not found: {dataset_root}')
        if not annotation_path.exists():
            raise FileNotFoundError(f'Annotation file not found: {annotation_path}')
        if not intrinsic_path.exists():
            raise FileNotFoundError(f'Intrinsic file not found: {intrinsic_path}')

        annotations = np.load(str(annotation_path), allow_pickle=True)
        names = annotations['name']
        rel_poses = annotations['rel_pose']
        total = len(names) if max_pairs <= 0 else min(len(names), max_pairs)

        self.dataset_root = dataset_root
        self.data_names = names[:total]
        self.rel_poses = rel_poses[:total]
        self.intrinsics = dict(np.load(str(intrinsic_path), allow_pickle=True))

    def __len__(self) -> int:
        return len(self.data_names)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        scene_id, scene_sub_id, stem0, stem1 = self.data_names[idx]
        scene_name = f'scene{scene_id:04d}_{scene_sub_id:02d}'
        scene_root = self.dataset_root / scene_name

        image0 = read_scannet_gray_as_bgr(scene_root / 'color' / f'{stem0}.jpg')
        image1 = read_scannet_gray_as_bgr(scene_root / 'color' / f'{stem1}.jpg')
        pose0_path = scene_root / 'pose' / f'{stem0}.txt'
        pose1_path = scene_root / 'pose' / f'{stem1}.txt'

        if pose0_path.exists() and pose1_path.exists():
            pose0 = read_world_to_camera(pose0_path)
            pose1 = read_world_to_camera(pose1_path)
            T_0to1 = pose1 @ np.linalg.inv(pose0)
        else:
            T_0to1 = np.eye(4, dtype=np.float64)
            T_0to1[:3, :] = self.rel_poses[idx].reshape(3, 4)

        K = np.asarray(self.intrinsics[scene_name], dtype=np.float64).reshape(3, 3)
        return {
            'identifier': f'{scene_name}_{stem0}_{stem1}',
            'image0': image0,
            'image1': image1,
            'K0': K.copy(),
            'K1': K.copy(),
            'T_0to1': T_0to1,
        }


def evaluate(dataset: ScanNet1500, match_fn) -> Dict[str, Any]:
    identifiers: List[str] = []
    R_errs: List[float] = []
    t_errs: List[float] = []
    epi_errs: List[np.ndarray] = []
    empty_match_pairs = 0
    total_matches = 0

    for sample in tqdm.tqdm(dataset, desc='processing'):
        mkpts0, mkpts1 = match_fn(sample['image0'], sample['image1'])
        identifiers.append(sample['identifier'])
        total_matches += int(len(mkpts0))

        if len(mkpts0) == 0:
            empty_match_pairs += 1

        epi_errs.append(symmetric_epipolar_distance(mkpts0, mkpts1, sample['T_0to1'], sample['K0'], sample['K1']))
        ret = estimate_pose_loftr(mkpts0, mkpts1, sample['K0'], sample['K1'], args.ransac_thr, RANSAC_CONF)
        if ret is None:
            R_errs.append(float('inf'))
            t_errs.append(float('inf'))
            continue

        R, t, _ = ret
        t_err, R_err = relative_pose_error(sample['T_0to1'], R, t, ignore_gt_t_thr=0.0)
        R_errs.append(R_err)
        t_errs.append(t_err)

    pose_errors = np.max(np.stack([R_errs, t_errs]), axis=0)
    metrics = error_auc(pose_errors.tolist(), [5, 10, 20])
    metrics['prec@5e-4'] = epidist_prec(epi_errs, EPI_ERR_THR)

    return {
        'metrics': {key: value * 100.0 for key, value in metrics.items()},
        'num_pairs': len(identifiers),
        'failed_pairs': int(np.isinf(pose_errors).sum()),
        'empty_match_pairs': empty_match_pairs,
        'avg_matches': float(total_matches / len(identifiers)) if identifiers else 0.0,
    }


if __name__ == '__main__':
    weights = str(Path(args.weights).resolve())
    dataset = ScanNet1500(
        dataset_root=DATASET_ROOT,
        annotation_path=ANNOTATION_PATH,
        intrinsic_path=INTRINSIC_PATH,
        max_pairs=args.max_pairs,
    )
    model_type, match_fn = build_matcher(weights)
    summary = evaluate(dataset, match_fn)

    run_name = args.name if args.name != 'auto' else model_type
    payload = {
        'name': run_name,
        'model_type': model_type,
        'weights': weights,
        'protocol': 'loftr_scannet_1500',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'eval_args': {
            'matcher': args.matcher,
            'top_k': args.top_k,
            'detect_threshold': args.detect_threshold,
            'nms_kernel_size': args.nms_kernel_size,
            'match_min_cossim': args.match_min_cossim,
            'match_ratio_thresh': args.match_ratio_thresh,
            'ransac_thr': args.ransac_thr,
            'max_pairs': args.max_pairs,
        },
        **summary,
    }

    print(f"\n==={payload['timestamp']}==={run_name}===")
    print(f"model_type: {payload['model_type']}")
    print(f"protocol: {payload['protocol']}")
    print(f"samples: {payload['num_pairs']}")
    print(f"failed_pairs: {payload['failed_pairs']}")
    print(f"empty_match_pairs: {payload['empty_match_pairs']}")
    print(f"avg_matches: {payload['avg_matches']:.2f}")
    for key, value in payload['metrics'].items():
        print(f"{key}: {value:.4f}")

    if args.save_json:
        with open(args.save_json, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)