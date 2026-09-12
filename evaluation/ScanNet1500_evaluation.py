"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    Camera pose metrics adapted from LoFTR https://github.com/zju3dv/LoFTR/blob/master/src/utils/metrics.py
    ScanNet pose estimation is fixed to PoseLib relative pose RANSAC.

"""

import argparse
import numpy as np
import os
import sys
import cv2
from tqdm import tqdm
import json
import multiprocessing as mp

POSE_ESTIMATOR = 'poselib'

# Ensure local imports work when running this file directly from the repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ACCELERATED_FEATURES_ROOT = os.path.join(REPO_ROOT, '3rdparty', 'accelerated_features')
LIFTFEAT_ROOT = os.path.join(REPO_ROOT, 'LiftFeat')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if ACCELERATED_FEATURES_ROOT not in sys.path:
    sys.path.insert(0, ACCELERATED_FEATURES_ROOT)
if LIFTFEAT_ROOT not in sys.path:
    sys.path.insert(0, LIFTFEAT_ROOT)

# Disable scientific notation
np.set_printoptions(suppress=True)
# NumPy 2.x removed trapz in some builds; keep the old code path working.
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

def intrinsics_to_camera(K):
    px, py = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    return {
        "model": "PINHOLE",
        "width": int(2 * px),
        "height": int(2 * py),
        "params": [fx, fy, px, py],
    }

def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))

def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))

def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R

def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999, max_iterations=10000):
    if len(kpts0) < 5:
        return None
    import poselib
    pose, details = poselib.estimate_relative_pose(
        kpts0.tolist(),
        kpts1.tolist(),
        intrinsics_to_camera(K0),
        intrinsics_to_camera(K1),
        ransac_opt={
            'max_iterations': max_iterations, # default 100000
            'success_prob': conf, # default 0.99999
            'max_epipolar_error': thresh, # default 1.0
        },
        bundle_opt={  # all defaults
        },
    )
    ret = (pose.R, pose.t, details['inliers'])
    return ret

def estimate_pose_parallel(args):
    return estimate_pose(*args)

def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs

def pose_accuracy(errors, thresholds):
    return [np.mean(errors < t) * 100 for t in thresholds]

def get_relative_transform(pose0, pose1):
    R0 = pose0[..., :3, :3] # Bx3x3
    t0 = pose0[..., :3, [3]] # Bx3x1

    R1 = pose1[..., :3, :3] # Bx3x3
    t1 = pose1[..., :3, [3]] # Bx3x1
    
    R_0to1 = R1.transpose(-1, -2) @ R0 # Bx3x3
    t_0to1 = R1.transpose(-1, -2) @ (t0 - t1) # Bx3x1
    T_0to1 = np.concatenate([R_0to1, t_0to1], axis=-1) # Bx3x4

    return T_0to1

class Scannet1500:
    default_config = {
        # Keep the dataset location fixed to the repo's data folder.
        'scannet_path': os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')),
        # ScanNet1500 ground-truth file stored alongside the dataset root.
        'gt_path': os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/scannet_1500.npz')),
        'pose_estimator': POSE_ESTIMATOR,
        'cache_images': True,
        'ransac_thresholds': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
        'ransac_confidence': 0.99999,
        'ransac_max_iterations': 10000,
        'ignore_gt_t_thr': 0.2,
        'pose_thresholds': [5, 10, 20],
        'max_pairs': -1,
        'output': './output/scannet/',
        'n_workers': 8,
    }

    def __init__(self, config={}) -> None:
        self.config = {**self.default_config, **config}
        if not os.path.exists(self.config['scannet_path']):
            raise RuntimeError(
                f"Dataset {self.config['scannet_path']} does not exist! \n \
                  > If you didn't download the dataset, use the downloader tool: python3 -m modules.dataset.download -h")
        
        self.pairs = self.read_gt()

        os.makedirs(self.config['output'], exist_ok=True)

        if self.config['n_workers'] == -1:
            self.config['n_workers'] = mp.cpu_count()

        self.image_cache = {}
        if self.config['cache_images']:
            self.load_images()

    def load_images(self):
        for pair in tqdm(self.pairs, desc='Caching images'):
            if pair['image0'] not in self.image_cache:
                self.image_cache[pair['image0']] = cv2.imread(pair['image0'])
            if pair['image1'] not in self.image_cache:
                self.image_cache[pair['image1']] = cv2.imread(pair['image1'])

    def read_image(self, path):
        if self.config['cache_images']:
            return self.image_cache[path]
        else:
            return cv2.imread(path)

    def read_gt(self):
        pairs = []
        gt_poses = np.load(self.config['gt_path'])
        names = gt_poses['name']
        
        for i in range(len(names)):
                scene_id = names[i, 0]
                scene_idx = names[i, 1]
                scene = f'scene{scene_id:04d}_{scene_idx:02d}'
            
                image0 = str(int(names[i, 2]))
                image1 = str(int(names[i, 3]))
                
                K0 = np.loadtxt(
                    os.path.join(self.config['scannet_path'], 'scannet_test_1500', scene,  'intrinsic/intrinsic_color.txt')
                )
                K1 = K0

                pose_0 = np.loadtxt(
                    os.path.join(self.config['scannet_path'], 'scannet_test_1500', scene, 'pose', image0 + '.txt')
                )
                pose_1 = np.loadtxt(
                    os.path.join(self.config['scannet_path'], 'scannet_test_1500', scene, 'pose', image1 + '.txt')
                )
                T_0to1 = get_relative_transform(pose_0, pose_1)
                
                pairs.append({
                    'image0': os.path.join(self.config['scannet_path'], 'scannet_test_1500', scene, 'color', image0 + '.jpg'),
                    'image1': os.path.join(self.config['scannet_path'], 'scannet_test_1500', scene, 'color', image1 + '.jpg'),
                    'K0': K0,
                    'K1': K1,
                    'T_0to1': T_0to1,
                })

        return pairs

    def extract_and_save_matches(self, matcher_fn, name='', force=False):
        all_matches = []
        if name == '':
            name = matcher_fn.__name__

        fname = os.path.join(self.config['output'], f'{name}_matches.npz')
        if not force and os.path.exists(fname):
            cached_matches = np.load(fname, allow_pickle=True)['all_matches']
            if len(cached_matches) == len(self.pairs):
                return cached_matches
            print(
                f"Cached matches length mismatch for {name}: "
                f"{len(cached_matches)} cached vs {len(self.pairs)} pairs. Recomputing."
            )

        for pair in tqdm(self.pairs, desc='Extracting matches'):
            image0 = self.read_image(pair['image0'])
            image1 = self.read_image(pair['image1'])
            mkpts0, mkpts1 = matcher_fn(image0, image1)

            all_matches.append({
                'image0': pair['image0'],
                'image1': pair['image1'],
                'mkpts0': mkpts0,
                'mkpts1': mkpts1,
            })
        
        np.savez(fname, all_matches=all_matches)
        
        return all_matches

    def run_benchmark(self, matcher_fn, name='', force=False):
        if name == '':
            name = matcher_fn.__name__

        all_matches = self.extract_and_save_matches(matcher_fn, name=name, force=force)

        aucs_by_thresh = {}
        accs_by_thresh = {}
        for ransac_thresh in self.config['ransac_thresholds']:

            fname = os.path.join(self.config['output'], f'{name}_{POSE_ESTIMATOR}_{ransac_thresh}.txt')
            # check if exists and has the right number of lines
            if not force and os.path.exists(fname) and len(open(fname, 'r').readlines()) == len(self.pairs):
                errors = []
                with open(fname, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.replace('\n', '')
                        err_t, err_R = line.split(' ')
                        errors.append([float(err_t), float(err_R)])
            # redo the benchmark
            else:
                errors = []
                pairs = self.pairs
                errors_file = open(fname, 'w')

                # do the benchmark in parallel
                if self.config['n_workers'] != 1:

                    pool = mp.Pool(self.config['n_workers'])
                    pool_args = [
                        (
                            all_matches[pair_idx]['mkpts0'],
                            all_matches[pair_idx]['mkpts1'],
                            pair['K0'],
                            pair['K1'],
                            ransac_thresh,
                            self.config['ransac_confidence'],
                            self.config['ransac_max_iterations'],
                        )
                        for pair_idx, pair in enumerate(pairs)
                    ]
                    results = list(tqdm(pool.imap(estimate_pose_parallel, pool_args), total=len(pool_args), desc=f'Running benchmark for th={ransac_thresh}', leave=False))
                    pool.close()
                    

                    for pair_idx, ret in enumerate(results):
                        if ret is None:
                            err_t, err_R = np.inf, np.inf
                        else:
                            R, t, inliers = ret
                            pair = pairs[pair_idx]
                            err_t, err_R = compute_pose_error(pair['T_0to1'], R, t)
                        errors_file.write(f'{err_t} {err_R}\n')
                        errors.append([err_t, err_R])
                # do the benchmark in serial
                else:
                    for pair_idx, pair in tqdm(enumerate(pairs), desc=f'Running benchmark for th={ransac_thresh}', leave=False, total=len(pairs)):
                        mkpts0 = all_matches[pair_idx]['mkpts0']
                        mkpts1 = all_matches[pair_idx]['mkpts1']
                        ret = estimate_pose(
                            mkpts0,
                            mkpts1,
                            pair['K0'],
                            pair['K1'],
                            ransac_thresh,
                            self.config['ransac_confidence'],
                            self.config['ransac_max_iterations'],
                        )
                        if ret is None:
                            err_t, err_R = np.inf, np.inf
                        else:
                            R, t, inliers = ret
                            err_t, err_R = compute_pose_error(pair['T_0to1'], R, t)
                        errors_file.write(f'{err_t} {err_R}\n')
                        errors_file.flush()
                        errors.append([err_t, err_R])

                errors_file.close()

            # compute AUCs
            errors = np.array(errors)
            ignore_gt_t_thr = float(self.config.get('ignore_gt_t_thr', 0.0))
            if ignore_gt_t_thr > 0:
                gt_t_norms = np.array([np.linalg.norm(pair['T_0to1'][:3, 3]) for pair in self.pairs])
                errors[gt_t_norms < ignore_gt_t_thr, 0] = 0.0
            errors = errors.max(axis=1) 
            aucs = pose_auc(errors, self.config['pose_thresholds'])
            accs = pose_accuracy(errors, self.config['pose_thresholds'])
            aucs = {k: v*100 for k, v in zip(self.config['pose_thresholds'], aucs)}
            accs = {k: v for k, v in zip(self.config['pose_thresholds'], accs)}
            aucs_by_thresh[ransac_thresh] = aucs
            accs_by_thresh[ransac_thresh] = accs
            df_scores = []
            for key, score in aucs_by_thresh.items():
                df_scores.append({
                    'ransac_threshold': float(key),
                    'auc5': float(score[5]),
                    'auc10': float(score[10]),
                    'auc20': float(score[20]),
                    'mean': float((score[5] + score[10] + score[20]) / 3.0),
                })
            best_by_auc5 = max(df_scores, key=lambda x: x['auc5'])
            best_by_mean = max(df_scores, key=lambda x: x['mean'])

            # dump summary for this method
            summary = {
                'name': name,
                'aucs_by_thresh': aucs_by_thresh,
                'accs_by_thresh': accs_by_thresh,
                'best_by_auc5': best_by_auc5,
                'best_by_mean': best_by_mean,
                'eval_config': {
                    'pose_estimator': POSE_ESTIMATOR,
                    'ransac_confidence': self.config['ransac_confidence'],
                    'ransac_max_iterations': self.config['ransac_max_iterations'],
                    'ignore_gt_t_thr': self.config.get('ignore_gt_t_thr', 0.0),
                    'pose_thresholds': self.config['pose_thresholds'],
                    'num_pairs': len(self.pairs),
                },
            }
            json.dump(summary, open(os.path.join(self.config['output'], f'{name}_{POSE_ESTIMATOR}_summary.json'), 'w'), indent=2)

        return aucs_by_thresh

def get_xfeat(weights=None):
    from modules.xfeat import XFeat
    xfeat = XFeat(weights=weights)
    return xfeat.match_xfeat

def get_xfeat_star():
    from modules.xfeat import XFeat
    xfeat = XFeat(top_k=10_000)
    return xfeat.match_xfeat_star

def get_alike():
    from third_party import alike_wrapper as alike
    return alike.match_alike

def get_liftfeat(weights=None):
    from models.liftfeat_wrapper import LiftFeat
    return LiftFeat(weight=weights).match_liftfeat

def get_geofeat(
    weights=None,
    config_file='',
    top_k=0,
    detect_threshold=0.1,
    nms_kernel_size=5,
    match_min_cossim=-1.0,
    match_ratio_thresh=0.90,
    descriptor_source='refined',
    keypoint_source='raw',
    descriptor_confidence_gain=0.0,
    repeatability_adaptive_gamma=0.0,
):
    from src.utils.geofeat_wrapper import GeoFeat, load_model_config_for_weights
    from src.config.config import get_cfg_defaults, get_model_config

    if weights is None:
        raise ValueError("GeoFeat evaluation requires --weights so the config snapshot can be loaded")

    weights = os.path.abspath(weights)
    if config_file:
        cfg = get_cfg_defaults()
        cfg.merge_from_file(config_file)
        model_config = get_model_config(cfg)
        print(f"Using override model config: {os.path.abspath(config_file)}")
        print("Checkpoint weights will be loaded with relaxed shape-compatible matching.")
    else:
        model_config = load_model_config_for_weights(weights)
    geofeat = GeoFeat(
        model_config=model_config,
        weight_path=weights,
        top_k=top_k,
        detect_threshold=detect_threshold,
        nms_kernel_size=nms_kernel_size,
        match_min_cossim=match_min_cossim,
        match_ratio_thresh=match_ratio_thresh,
        descriptor_source=descriptor_source,
        keypoint_source=keypoint_source,
        descriptor_confidence_gain=descriptor_confidence_gain,
        repeatability_adaptive_gamma=repeatability_adaptive_gamma,
    )
    return geofeat.match_featnet

def print_fancy(d):
    print(json.dumps(d, indent=2))

def apply_geofeat_preset(args):
    if args.preset == 'scannet_auc5':
        args.top_k = 0
        args.detect_threshold = 0.1
        args.nms_kernel_size = 5
        args.match_min_cossim = -1.0
        args.match_ratio_thresh = 0.90
        args.descriptor_source = 'refined'
        args.keypoint_source = 'raw'
        if not args.ransac_thresholds:
            args.ransac_thresholds = [2.5, 3.0, 3.5, 4.0, 4.5]
    elif args.preset == 'scannet_wide':
        args.top_k = 0
        args.detect_threshold = 0.08
        args.nms_kernel_size = 5
        args.match_min_cossim = -1.0
        args.match_ratio_thresh = 0.92
        args.descriptor_source = 'refined'
        args.keypoint_source = 'raw'
        if not args.ransac_thresholds:
            args.ransac_thresholds = [3.0, 3.5, 4.0, 4.5, 5.0]
    return args

def parse():
    parser = argparse.ArgumentParser()
    
    # Default to the repo-local dataset layout so the command can be short.
    parser.add_argument("--scannet_path", type=str, default='./data', help="Path to the ScanNet root containing scannet_test_1500")
    parser.add_argument("--output", type=str, default="./output/GRR-TGC50/", help="Path to the output directory")
    parser.add_argument("--max_pairs", type=int, default=-1, help="Maximum number of pairs to run the benchmark on")
    parser.add_argument("--force", action='store_true', help="Force running the benchmark again")
    parser.add_argument("--n_workers", type=int, default=1, help="Number of PoseLib workers; use 1 for serial debugging")
    parser.add_argument("--ransac_thresholds", type=float, nargs='+', default=None, help="PoseLib max epipolar error thresholds")
    parser.add_argument("--ransac_confidence", type=float, default=0.99999, help="PoseLib RANSAC success probability")
    parser.add_argument("--ransac_max_iterations", type=int, default=10000, help="PoseLib RANSAC max iterations")
    parser.add_argument("--ignore_gt_t_thr", type=float, default=0.2, help="Ignore translation-direction error when GT translation norm is below this ScanNet indoor threshold; 0 disables")
    # XFeat weights are now configurable from the CLI.
    parser.add_argument("--weights", type=str, default='./weights/xfeat.pt', help="Path to model checkpoint")
    # Default to xfeat so the benchmark does not require optional ALIKE deps.
    parser.add_argument("--method", type=str, default='geofeat', choices=['xfeat', 'xfeat_star', 'liftfeat', 'geofeat', 'alike', 'all'], help="Which matcher to evaluate")
    parser.add_argument("--config_file", type=str, default='', help="Optional GeoFeat model config YAML, e.g. src/config/experiments/GRR50.yaml or GRR-TGC50.yaml")
    parser.add_argument("--top_k", type=int, default=0, help="GeoFeat maximum keypoints to keep; <=0 keeps all NMS keypoints")
    parser.add_argument("--detect_threshold", type=float, default=0.1, help="GeoFeat keypoint detection threshold")
    parser.add_argument("--nms_kernel_size", type=int, default=5, help="GeoFeat NMS kernel size")
    parser.add_argument("--match_min_cossim", type=float, default=-1.0, help="GeoFeat minimum cosine similarity; <=0 disables")
    parser.add_argument("--match_ratio_thresh", type=float, default=0.90, help="GeoFeat Lowe-style ratio threshold; <=0 disables")
    parser.add_argument("--descriptor_source", type=str, default='refined', choices=['raw', 'refined'], help="GeoFeat descriptor source")
    parser.add_argument("--keypoint_source", type=str, default='raw', choices=['raw', 'refined'], help="GeoFeat keypoint source")
    parser.add_argument("--descriptor_confidence_gain", type=float, default=0.0, help="GeoFeat adaptive descriptor fusion strength [0,+inf), 0 disables")
    parser.add_argument("--repeatability_adaptive_gamma", type=float, default=0.0, help="GeoFeat adaptive repeatability sharpness [0,+inf), 0 disables")
    parser.add_argument("--preset", type=str, default='scannet_auc5', choices=['none', 'scannet_auc5', 'scannet_wide'], help="GeoFeat ScanNet evaluation preset")
    parser.add_argument("--compare_sources", action='store_true', help="Evaluate all raw/refined descriptor-keypoint source combinations")
    
    parser.add_argument("--show", action='store_true', help="Show the matches")
    parser.add_argument("--accuracy", action='store_true', help="Show the accuracy instead of AUC")
    parser.add_argument("--filter", type=str, nargs='+', help="Filter the results by the given names")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    args = apply_geofeat_preset(args)
    if args.ransac_thresholds is None:
        args.ransac_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

    if not args.show:
        scannet = Scannet1500({
            'scannet_path': args.scannet_path,
            'gt_path': os.path.join(args.scannet_path, 'scannet_1500.npz'),
            'cache_images': False,
            'output': args.output,
            'max_pairs': args.max_pairs,
            'pose_estimator': POSE_ESTIMATOR,
            'ransac_thresholds': args.ransac_thresholds,
            'ransac_confidence': args.ransac_confidence,
            'ransac_max_iterations': args.ransac_max_iterations,
            'ignore_gt_t_thr': args.ignore_gt_t_thr,
            'n_workers': args.n_workers,
        })

        if args.max_pairs > 0:
            scannet.pairs = scannet.pairs[:args.max_pairs]

        if args.method == 'xfeat':
            functions = {'xfeat': get_xfeat(weights=args.weights)}
        elif args.method == 'liftfeat':
            functions = {'liftfeat': get_liftfeat(weights=args.weights)}
        elif args.method == 'geofeat':
            source_pairs = [(args.descriptor_source, args.keypoint_source)]
            if args.compare_sources:
                source_pairs = [
                    ('raw', 'raw'),
                    ('raw', 'refined'),
                    ('refined', 'raw'),
                    ('refined', 'refined'),
                ]
            functions = {}
            for descriptor_source, keypoint_source in source_pairs:
                geofeat_name = (
                    f"geofeat_{args.preset}_tk{args.top_k}_dt{args.detect_threshold:g}_nms{args.nms_kernel_size}"
                    f"_mc{args.match_min_cossim:g}_r{args.match_ratio_thresh:g}"
                    f"_ds{descriptor_source}_ks{keypoint_source}"
                    f"_dcg{args.descriptor_confidence_gain:g}_rag{args.repeatability_adaptive_gamma:g}"
                )
                functions[geofeat_name] = get_geofeat(
                    weights=args.weights,
                    config_file=args.config_file,
                    top_k=args.top_k,
                    detect_threshold=args.detect_threshold,
                    nms_kernel_size=args.nms_kernel_size,
                    match_min_cossim=args.match_min_cossim,
                    match_ratio_thresh=args.match_ratio_thresh,
                    descriptor_source=descriptor_source,
                    keypoint_source=keypoint_source,
                    descriptor_confidence_gain=args.descriptor_confidence_gain,
                    repeatability_adaptive_gamma=args.repeatability_adaptive_gamma,
                )
        else:
            functions = {'xfeat_star': get_xfeat_star()}

        # save all results to a file
        all_results = {}

        for name, fn in functions.items():
            print(name)
            result = scannet.run_benchmark(matcher_fn = fn, name=name, force=args.force)
            all_results[name] = result

        json.dump(all_results, open(os.path.join(args.output, 'summary.json'), 'w'), indent=2)

    if args.show:
        import glob
        import pandas as pd
        dataset_name = 'scannet'
        all_summary_files = glob.glob(os.path.join(args.output, "**_summary.json"), recursive=True)

        if args.filter:
            all_summary_files = [f for f in all_summary_files if any([fil in f for fil in args.filter])]

        dfs = []
        names = []
        estimators = []
        metric_key = 'aucs_by_thresh'
        if args.accuracy:
            metric_key = 'accuracies_by_thresh'    
        for summary in all_summary_files:
            summary_data = json.load(open(summary, 'r'))
            if metric_key not in summary_data:
                continue
            aucs_by_thresh = summary_data[metric_key]

            estimator = POSE_ESTIMATOR

            #make sure everything is float
            for thresh in aucs_by_thresh:
                for k in aucs_by_thresh[thresh]:
                    if isinstance(aucs_by_thresh[thresh][k], str):
                        aucs_by_thresh[thresh][k] = float(aucs_by_thresh[thresh][k].replace(' ', ''))

            # find best threshold based on the 5, 10, 20 mAP and everything is float
            df = pd.DataFrame(aucs_by_thresh).T.astype(float)
            df['mean'] = df.mean(axis=1)
            # create a string column called estimator
            cols = df.columns.tolist()
            dfs.append(df)
            names.append(summary_data['name'])
            estimators.append(estimator)

        # use each col as the main col to determine the best threshold
        # for col in cols:
        col = 'mean'

        final_df = pd.DataFrame()
        # add cols
        final_df['name'] = names
        final_df['best_thresh'] = ''
        final_df['estimator'] = estimators
        final_df[cols] = -1.0

        for df, name, estimator in zip(dfs, names, estimators):
            best_thresh = df[col].idxmax()
            best_results = df.loc[best_thresh]

            # now update the best_thresh based on the estimator
            final_df.loc[(final_df['name'] == name) & (final_df['estimator'] == estimator), 'best_thresh'] = best_thresh
            for _col in cols:
                final_df.loc[(final_df['name'] == name) & (final_df['estimator'] == estimator), _col] = best_results[_col]

        # sort by mean
        final_df = final_df.sort_values(by=['mean'])
        # reset index
        final_df = final_df.reset_index(drop=True)

        # drop estimator column
        final_df = final_df.drop(columns=['estimator'])

        # set max float precision to 1
        final_df = final_df.round(1)

        print(f"Dataset: {dataset_name}")
        print(f"Sorting by {col}")
        print(final_df)
        print()

        final_df.to_csv(os.path.join(args.output, f"{dataset_name}_{col}.csv"), index=False)
