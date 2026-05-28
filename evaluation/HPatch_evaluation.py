import argparse
import datetime
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluation.eval_utils import homo_trans
from src.utils.geofeat_wrapper import GeoFeat, load_model_config_for_weights


parser = argparse.ArgumentParser(description='HPatch dataset evaluation script for GeoFeat')
parser.add_argument('--name', type=str, default='GeoFeat', help='experiment name')
parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
parser.add_argument('--weights', type=str, required=True, help='Path to GeoFeat checkpoint (.pth)')
parser.add_argument(
    '--dataset_root',
    type=str,
    default=os.path.join(os.path.dirname(__file__), '../data/hpatches-sequences-release'),
    help='Path to HPatches root directory',
)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

n_i = 52
n_v = 56

def benchmark_features(match_fn, dataset_root: str):
    lim = [1, 9]
    rng = np.arange(lim[0], lim[1] + 1)

    seq_names = sorted([n for n in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, n))])

    n_feats = []
    n_matches = []
    seq_type = []
    i_err = {thr: 0 for thr in rng}
    v_err = {thr: 0 for thr in rng}

    i_err_homo = {thr: 0 for thr in rng}
    v_err_homo = {thr: 0 for thr in rng}

    for _, seq_name in tqdm(enumerate(seq_names), total=len(seq_names)):
        ref_path = os.path.join(dataset_root, seq_name, '1.ppm')
        ref_img = cv2.imread(ref_path)
        if ref_img is None:
            print(f'Warning: failed to read reference image: {ref_path}')
            continue
        ref_img_shape = ref_img.shape

        # Query images: 2~6 with ground-truth H_1_i
        for im_idx in range(2, 7):
            h_path = os.path.join(dataset_root, seq_name, f'H_1_{im_idx}')
            q_path = os.path.join(dataset_root, seq_name, f'{im_idx}.ppm')
            if not os.path.exists(h_path) or not os.path.exists(q_path):
                continue

            homography = np.loadtxt(h_path)
            query_img = cv2.imread(q_path)
            if query_img is None:
                continue

            mkpts_a, mkpts_b = match_fn(ref_img, query_img)

            pos_a = mkpts_a
            pos_b = mkpts_b

            n_matches.append(pos_a.shape[0])
            seq_type.append(seq_name[0])

            if pos_a.shape[0] == 0:
                dist = np.array([float('inf')])
            else:
                pos_a_h = np.concatenate([pos_a, np.ones([pos_a.shape[0], 1])], axis=1)
                pos_b_proj_h = np.transpose(np.dot(homography, np.transpose(pos_a_h)))
                pos_b_proj = pos_b_proj_h[:, :2] / pos_b_proj_h[:, 2:]
                dist = np.sqrt(np.sum((pos_b - pos_b_proj) ** 2, axis=1))

            for thr in rng:
                if seq_name[0] == 'i':
                    i_err[thr] += np.mean(dist <= thr)
                else:
                    v_err[thr] += np.mean(dist <= thr)

            # Estimate homography from matches. Require >=4 points.
            gt_homo = homography
            if mkpts_a.shape[0] < 4:
                pred_homo = None
            else:
                pred_homo, _ = cv2.findHomography(mkpts_a, mkpts_b, cv2.USAC_MAGSAC)

            if pred_homo is None:
                homo_dist = np.array([float('inf')])
            else:
                corners = np.array(
                    [
                        [0, 0],
                        [ref_img_shape[1] - 1, 0],
                        [0, ref_img_shape[0] - 1],
                        [ref_img_shape[1] - 1, ref_img_shape[0] - 1],
                    ]
                )
                real_warped_corners = homo_trans(corners, gt_homo)
                warped_corners = homo_trans(corners, pred_homo)
                homo_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))

            for thr in rng:
                if seq_name[0] == 'i':
                    i_err_homo[thr] += np.mean(homo_dist <= thr)
                else:
                    v_err_homo[thr] += np.mean(homo_dist <= thr)

    seq_type = np.array(seq_type)
    n_feats = np.array(n_feats)
    n_matches = np.array(n_matches)

    return i_err, v_err, i_err_homo, v_err_homo, [seq_type, n_feats, n_matches]


if __name__ == '__main__':
    weights = os.path.abspath(args.weights)
    if not os.path.exists(weights):
        raise FileNotFoundError(f'Weight file not found: {weights}')

    dataset_root = os.path.abspath(args.dataset_root)
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f'HPatches root directory not found: {dataset_root}')

    model_config = load_model_config_for_weights(weights)

    geofeat = GeoFeat(model_config=model_config, weight_path=weights)

    errors = benchmark_features(geofeat.match_featnet, dataset_root)
    i_err, v_err, i_err_hom, v_err_hom, _ = errors

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print(f'\n==={cur_time}==={args.name}===')
    print('MHA@3 MHA@5 MHA@7')
    for thr in [3, 5, 7]:
        ill_err_hom = i_err_hom[thr] / (n_i * 5)
        view_err_hom = v_err_hom[thr] / (n_v * 5)
        print(f'{ill_err_hom * 100:.2f}%-{view_err_hom * 100:.2f}%')
