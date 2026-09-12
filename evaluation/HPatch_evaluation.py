"""HPatches homography evaluation for GeoFeat, including all sequences by default."""
import argparse
import datetime
import json
import os
import sys
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from evaluation.eval_utils import homo_trans
from src.utils.geofeat_wrapper import GeoFeat, load_model_config_for_weights

LIFTFEAT_EXCLUDED_SEQUENCES = {
    'i_contruction', 'i_crownnight', 'i_dc', 'i_pencils', 'i_whitebuilding',
    'v_artisans', 'v_astronautis', 'v_talent',
}


def parse_args():
    parser = argparse.ArgumentParser(description='LiftFeat-compatible HPatches evaluation for GeoFeat')
    parser.add_argument('--name', type=str, default='GeoFeat')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--dataset_root', type=str, default=os.path.join(os.path.dirname(__file__), '../data/hpatches-sequences-release'))
    parser.add_argument('--output_json', type=str, default='')
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--detect_threshold', type=float, default=0.05)
    parser.add_argument('--nms_kernel_size', type=int, default=5)
    parser.add_argument('--match_min_cossim', type=float, default=-1.0)
    parser.add_argument('--match_ratio_thresh', type=float, default=-1.0)
    parser.add_argument('--descriptor_source', choices=['raw', 'refined'], default='refined')
    parser.add_argument('--keypoint_source', choices=['raw', 'refined'], default='raw')
    parser.add_argument('--descriptor_confidence_gain', type=float, default=0.0)
    parser.add_argument('--repeatability_adaptive_gamma', type=float, default=0.0)
    parser.add_argument(
        '--exclude_high_resolution_sequences',
        action='store_true',
        help='Exclude the 8 high-resolution sequences listed in the LiftFeat README',
    )
    parser.add_argument(
        '--legacy_fixed_denominator',
        action='store_true',
        help='Diagnostic only: use 52x5/56x5 denominators while evaluating all sequences',
    )
    parser.add_argument('--homography_method', choices=['RANSAC', 'USAC_DEFAULT', 'USAC_ACCURATE', 'USAC_FAST', 'USAC_MAGSAC', 'USAC_PROSAC'], default='USAC_MAGSAC')
    parser.add_argument('--ransac_reproj_threshold', type=float, default=None, help='Default follows LiftFeat demo: 0.01*(image height+width).')
    parser.add_argument('--ransac_max_iters', type=int, default=1000)
    parser.add_argument('--ransac_confidence', type=float, default=0.999)
    return parser.parse_args()


def _method(name: str) -> int:
    return cv2.RANSAC if name == 'RANSAC' else getattr(cv2, name)


def _sequences(root: str, exclude_high_resolution: bool) -> Iterable[str]:
    names = sorted(name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name)))
    if exclude_high_resolution:
        names = [name for name in names if name not in LIFTFEAT_EXCLUDED_SEQUENCES]
    return names


def _corner_error(pred_h: np.ndarray, gt_h: np.ndarray, shape: Tuple[int, ...]) -> float:
    height, width = shape[:2]
    corners = np.asarray([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]], dtype=np.float32)
    pred = homo_trans(corners, pred_h)
    gt = homo_trans(corners, gt_h)
    return float('inf') if not np.isfinite(pred).all() else float(np.mean(np.linalg.norm(pred - gt, axis=1)))


def evaluate(geofeat: GeoFeat, root: str, args) -> Dict:
    method = _method(args.homography_method)
    seq_names = list(_sequences(root, args.exclude_high_resolution_sequences))
    pairs = []
    for seq in tqdm(seq_names, desc='HPatches'):
        seq_dir = os.path.join(root, seq)
        ref = cv2.imread(os.path.join(seq_dir, '1.ppm'))
        if ref is None:
            continue
        for index in range(2, 7):
            query = cv2.imread(os.path.join(seq_dir, f'{index}.ppm'))
            h_path = os.path.join(seq_dir, f'H_1_{index}')
            if query is None or not os.path.isfile(h_path):
                continue
            gt_h = np.loadtxt(h_path).astype(np.float64)
            pts0, pts1 = geofeat.match_featnet(ref, query)
            threshold = args.ransac_reproj_threshold
            if threshold is None:
                threshold = 0.01 * (ref.shape[0] + ref.shape[1])
            pred_h = None
            if len(pts0) >= 4:
                pred_h, _ = cv2.findHomography(pts0.astype(np.float32), pts1.astype(np.float32), method,
                                                ransacReprojThreshold=float(threshold),
                                                maxIters=args.ransac_max_iters, confidence=args.ransac_confidence)
            error = float('inf') if pred_h is None else _corner_error(pred_h, gt_h, ref.shape)
            pairs.append({'sequence': seq, 'error': error, 'matches': int(len(pts0))})

    groups = {name: [p for p in pairs if p['sequence'].startswith(prefix)] for name, prefix in [('illumination', 'i'), ('viewpoint', 'v')]}
    mha = {name: {str(t): float(np.mean([p['error'] <= t for p in items]) * 100.0) if items else 0.0 for t in (3, 5, 7)} for name, items in groups.items()}
    valid_mha = mha
    legacy_mha = {
        group: {
            str(t): float(sum(item['error'] <= t for item in items) / (denom * 5) * 100.0)
            for t in (3, 5, 7)
        }
        for group, items, denom in (
            ('illumination', groups['illumination'], 52),
            ('viewpoint', groups['viewpoint'], 56),
        )
    }
    return {
        'name': args.name,
        'weights': os.path.abspath(args.weights),
        'dataset_root': os.path.abspath(root),
        'sequence_filter': {
            'mode': 'liftfeat_readme' if args.exclude_high_resolution_sequences else 'all_sequences',
            'excluded_sequences': sorted(LIFTFEAT_EXCLUDED_SEQUENCES) if args.exclude_high_resolution_sequences else [],
            'num_sequences': len(seq_names),
        },
        'pair_counts': {name: len(items) for name, items in groups.items()},
        'hyperparameters': vars(args),
        'mha': valid_mha,
        'legacy_invalid_mha': legacy_mha if args.legacy_fixed_denominator else None,
        'mean_matches': {name: float(np.mean([p['matches'] for p in items])) if items else 0.0 for name, items in groups.items()},
        'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
    }


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    weights, root = os.path.abspath(args.weights), os.path.abspath(args.dataset_root)
    if not os.path.isfile(weights):
        raise FileNotFoundError(weights)
    if not os.path.isdir(root):
        raise FileNotFoundError(root)
    geofeat = GeoFeat(model_config=load_model_config_for_weights(weights), weight_path=weights,
                      top_k=args.top_k, detect_threshold=args.detect_threshold,
                      nms_kernel_size=args.nms_kernel_size, match_min_cossim=args.match_min_cossim,
                      match_ratio_thresh=args.match_ratio_thresh, descriptor_source=args.descriptor_source,
                      keypoint_source=args.keypoint_source,
                      descriptor_confidence_gain=args.descriptor_confidence_gain,
                      repeatability_adaptive_gamma=args.repeatability_adaptive_gamma)
    result = evaluate(geofeat, root, args)
    print(f"\n==={result['generated_at']}==={args.name}===")
    print(f"Sequences: {result['sequence_filter']['num_sequences']} | pairs: {result['pair_counts']}")
    for group in ('illumination', 'viewpoint'):
        print(group, *(f"{result['mha'][group][str(t)]:.2f}%" for t in (3, 5, 7)))
    if args.legacy_fixed_denominator:
        print('legacy invalid diagnostic (not a valid score):')
        for group in ('illumination', 'viewpoint'):
            print(group, *(f"{result['legacy_invalid_mha'][group][str(t)]:.2f}%" for t in (3, 5, 7)))
    if args.output_json:
        output = os.path.abspath(args.output_json)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, 'w', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        print(f'Results written to: {output}')
