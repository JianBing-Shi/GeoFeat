import argparse
import datetime
import json
import os
import sys

import cv2
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset


parser = argparse.ArgumentParser(description='ScanNet-1500 evaluation script for GeoFeat')
parser.add_argument('--name', type=str, default='GeoFeat', help='experiment name')
parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
parser.add_argument('--weights', type=str, required=True, help='Path to GeoFeat checkpoint (.pth)')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from evaluation.eval_utils import compute_maa, estimate_pose, relative_pose_error  # noqa: E402
from src.utils.geofeat_wrapper import GeoFeat  # noqa: E402


DATASET_ROOT = os.path.join(os.path.dirname(__file__), '../data/scannet_test_1500')
DATASET_INDEX = os.path.join(os.path.dirname(__file__), '../data/scannet_1500.npz')
EPI_ERR_THR = 5e-4
POSE_RANSAC_THR = 4.0
POSE_RANSAC_CONF = 0.99999


def normalize_model_config(model_config: dict) -> dict:
    cfg = dict(model_config)
    if 'repeatability_alpha' not in cfg:
        cfg['repeatability_alpha'] = 0.0
        print("Warning: model_config missing 'repeatability_alpha'; fallback to 0.0.")
    if 'geo_preprocess_mode' not in cfg:
        cfg['geo_preprocess_mode'] = 'no_unfold_lite'
        print("Warning: model_config missing 'geo_preprocess_mode'; fallback to 'no_unfold_lite'.")
    return cfg


def load_intrinsic_matrix(scene_dir: str) -> np.ndarray:
    intrinsic_path = os.path.join(scene_dir, 'intrinsic', 'intrinsic_color.txt')
    if not os.path.exists(intrinsic_path):
        raise FileNotFoundError(f'Missing intrinsic file: {intrinsic_path}')

    matrix = np.loadtxt(intrinsic_path, dtype=np.float32)
    if matrix.shape == (4, 4):
        matrix = matrix[:3, :3]
    if matrix.shape != (3, 3):
        raise ValueError(f'Unexpected intrinsic shape {matrix.shape} at {intrinsic_path}')
    return matrix


def unpack_rel_pose(rel_pose: np.ndarray) -> np.ndarray:
    rel_pose = np.asarray(rel_pose, dtype=np.float32).reshape(-1)
    if rel_pose.shape[0] == 12:
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :] = rel_pose.reshape(3, 4)
        return pose
    if rel_pose.shape[0] == 16:
        return rel_pose.reshape(4, 4).astype(np.float32)
    raise ValueError(f'Unsupported rel_pose shape: {rel_pose.shape}')


def symmetric_epipolar_distance_numpy(pts0: np.ndarray, pts1: np.ndarray, e_mat: np.ndarray, k0: np.ndarray, k1: np.ndarray) -> np.ndarray:
    if len(pts0) == 0:
        return np.empty((0,), dtype=np.float32)

    pts0_norm = (pts0 - k0[[0, 1], [2, 2]][None]) / k0[[0, 1], [0, 1]][None]
    pts1_norm = (pts1 - k1[[0, 1], [2, 2]][None]) / k1[[0, 1], [0, 1]][None]

    pts0_h = np.concatenate([pts0_norm, np.ones((len(pts0_norm), 1), dtype=np.float32)], axis=1)
    pts1_h = np.concatenate([pts1_norm, np.ones((len(pts1_norm), 1), dtype=np.float32)], axis=1)

    ep0 = pts0_h @ e_mat.T
    etp1 = pts1_h @ e_mat
    p1ep0 = np.sum(pts1_h * ep0, axis=1)

    denom = 1.0 / (ep0[:, 0] ** 2 + ep0[:, 1] ** 2 + 1e-12)
    denom += 1.0 / (etp1[:, 0] ** 2 + etp1[:, 1] ** 2 + 1e-12)
    return (p1ep0 ** 2) * denom


def compute_precision_at_threshold(epi_errs_list, threshold: float) -> float:
    per_pair = []
    for errs in epi_errs_list:
        errs = np.asarray(errs, dtype=np.float32)
        if errs.size == 0:
            per_pair.append(0.0)
        else:
            per_pair.append(float(np.mean(errs < threshold)))
    return float(np.mean(per_pair)) if per_pair else 0.0


class ScanNet1500(Dataset):
    def __init__(self, dataset_root: str, index_file: str):
        if not os.path.exists(dataset_root):
            raise RuntimeError(f'Dataset root does not exist: {dataset_root}')
        if not os.path.exists(index_file):
            raise RuntimeError(f'ScanNet-1500 index file does not exist: {index_file}')

        with np.load(index_file) as data:
            self.pairs = data['name'].astype(np.int32)
            self.rel_poses = data['rel_pose'].astype(np.float32)

        self.dataset_root = dataset_root

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        scene_id, scene_sub_id, stem0, stem1 = [int(v) for v in self.pairs[idx]]
        scene_name = f'scene{scene_id:04d}_{scene_sub_id:02d}'
        scene_dir = os.path.join(self.dataset_root, scene_name)
        img0_rel = os.path.join(scene_name, 'color', f'{stem0}.jpg')
        img1_rel = os.path.join(scene_name, 'color', f'{stem1}.jpg')
        img0_path = os.path.join(self.dataset_root, img0_rel)
        img1_path = os.path.join(self.dataset_root, img1_rel)

        image0 = cv2.imread(img0_path)
        image1 = cv2.imread(img1_path)
        if image0 is None:
            raise ValueError(f'Failed to read image: {img0_path}')
        if image1 is None:
            raise ValueError(f'Failed to read image: {img1_path}')

        k_mat = load_intrinsic_matrix(scene_dir)
        t_0to1 = unpack_rel_pose(self.rel_poses[idx])

        return {
            'image0': torch.tensor(image0.astype(np.float32) / 255.0).permute(2, 0, 1),
            'image1': torch.tensor(image1.astype(np.float32) / 255.0).permute(2, 0, 1),
            'K0': torch.tensor(k_mat.copy(), dtype=torch.float32),
            'K1': torch.tensor(k_mat.copy(), dtype=torch.float32),
            'T_0to1': torch.tensor(t_0to1, dtype=torch.float32),
            'T_1to0': torch.tensor(np.linalg.inv(t_0to1), dtype=torch.float32),
            'scale0': torch.tensor(np.array([1.0, 1.0], dtype=np.float32)),
            'scale1': torch.tensor(np.array([1.0, 1.0], dtype=np.float32)),
            'dataset_name': 'ScanNet',
            'scene_id': scene_name,
            'pair_id': idx,
            'pair_names': (img0_rel, img1_rel),
        }


def evaluate_pair(geofeat: GeoFeat, batch: dict) -> dict:
    image0 = batch['image0'][0].permute(1, 2, 0).cpu().numpy()
    image1 = batch['image1'][0].permute(1, 2, 0).cpu().numpy()
    image0 = (image0 * 255).astype(np.uint8)
    image1 = (image1 * 255).astype(np.uint8)

    mkpts0, mkpts1 = geofeat.match_featnet(image0, image1)

    k0 = batch['K0'][0].cpu().numpy()
    k1 = batch['K1'][0].cpu().numpy()
    t_0to1 = batch['T_0to1'][0].cpu().numpy()

    tx = np.array([
        [0.0, -t_0to1[2, 3], t_0to1[1, 3]],
        [t_0to1[2, 3], 0.0, -t_0to1[0, 3]],
        [-t_0to1[1, 3], t_0to1[0, 3], 0.0],
    ], dtype=np.float32)
    e_mat = tx @ t_0to1[:3, :3]
    epi_errs = symmetric_epipolar_distance_numpy(mkpts0, mkpts1, e_mat, k0, k1)

    ret = estimate_pose(mkpts0, mkpts1, k0, k1, POSE_RANSAC_THR, conf=POSE_RANSAC_CONF) if len(mkpts0) >= 5 else None
    if ret is None:
        return {
            'R_err': np.inf,
            't_err': np.inf,
            'epi_errs': epi_errs,
            'num_matches': int(len(mkpts0)),
            'success': False,
        }

    r_mat, t_vec, inliers = ret
    t_err, r_err = relative_pose_error(t_0to1, r_mat, t_vec, ignore_gt_t_thr=0.0)
    return {
        'R_err': float(r_err),
        't_err': float(t_err),
        'epi_errs': epi_errs,
        'num_matches': int(len(mkpts0)),
        'num_inliers': int(np.asarray(inliers).sum()) if len(inliers) else 0,
        'success': True,
    }


if __name__ == '__main__':
    weights = os.path.abspath(args.weights)
    snapshot_path = os.path.join(os.path.dirname(weights), 'config_snapshot.json')
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(f'No config_snapshot.json found at {snapshot_path}')

    with open(snapshot_path, 'r') as f:
        snap = json.load(f)
    model_config = normalize_model_config(snap.get('model_config', snap))

    geofeat = GeoFeat(model_config=model_config, weight_path=weights)
    dataset = ScanNet1500(dataset_root=DATASET_ROOT, index_file=DATASET_INDEX)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = []
    epi_errs_list = []
    failed_pairs = 0
    empty_match_pairs = 0
    total_matches = 0

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for batch in tqdm.tqdm(loader, desc='processing'):
        pair_result = evaluate_pair(geofeat, batch)
        results.append({'R_err': pair_result['R_err'], 't_err': pair_result['t_err']})
        epi_errs_list.append(pair_result['epi_errs'])
        total_matches += pair_result['num_matches']

        if pair_result['num_matches'] == 0:
            empty_match_pairs += 1
        if not pair_result['success']:
            failed_pairs += 1

    d_err_auc, errors = compute_maa(results)
    prec = compute_precision_at_threshold(epi_errs_list, EPI_ERR_THR)

    print(f'\n==={cur_time}==={args.name}===')
    print(f'samples: {len(results)}')
    print(f'failed_pairs: {failed_pairs}')
    print(f'empty_match_pairs: {empty_match_pairs}')
    print(f'avg_matches: {total_matches / max(len(results), 1):.2f}')
    for s_k, s_v in d_err_auc.items():
        print(f'{s_k}: {s_v * 100:.2f}')
    print(f'prec@5e-4: {prec * 100:.2f}')