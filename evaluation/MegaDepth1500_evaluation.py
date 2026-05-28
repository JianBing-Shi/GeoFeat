import os
import sys
import cv2
from pathlib import Path
import numpy as np
import torch
import torch.utils.data as data
import tqdm
from copy import deepcopy
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import json

import scipy.io as scio
import poselib

import argparse
import datetime

parser=argparse.ArgumentParser(description='MegaDepth dataset evaluation script')
parser.add_argument('--name',type=str,default='GeoFeat',help='experiment name')
parser.add_argument('--gpu',type=str,default='0',help='GPU ID')
parser.add_argument('--weights', type=str, required=True, help='Path to GeoFeat checkpoint (.pth)')
parser.add_argument('--top_k', type=int, default=4096, help='Maximum number of detected keypoints to keep.')
parser.add_argument('--detect-threshold', type=float, default=0.05, help='Keypoint detection threshold.')
parser.add_argument('--nms-kernel-size', type=int, default=7, help='NMS kernel size for keypoint detection.')
parser.add_argument('--match-min-cossim', type=float, default=0.1, help='Minimum cosine similarity for accepting a match.')
parser.add_argument('--match-ratio-thresh', type=float, default=0.82, help='Lowe-style ratio threshold for match filtering.')
parser.add_argument('--pose-epipolar-error', type=float, default=4.0, help='Epipolar error threshold passed to relative pose estimation.')
parser.add_argument('--descriptor-source', type=str, default='refined', choices=['raw', 'refined'], help='Descriptor source used at evaluation time.')
parser.add_argument('--keypoint-source', type=str, default='refined', choices=['raw', 'refined'], help='Keypoint heatmap source used at evaluation time.')
parser.add_argument('--compare-sources', action='store_true', help='Evaluate all raw/refined descriptor-keypoint source combinations.')
parser.add_argument('--preset', type=str, default='none', choices=['none', 'gcdr_noloss_auc5_boost'], help='Apply an evaluation preset before building the wrapper.')
parser.add_argument('--save-json', type=str, default='', help='Optional path to save evaluation summaries as JSON.')
args=parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from src.utils.geofeat_wrapper import GeoFeat, load_model_config_for_weights
from evaluation.eval_utils import *

from torch.utils.data import Dataset,DataLoader

use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"

DATASET_ROOT = os.path.join(os.path.dirname(__file__),'../data/megadepth_test_1500')
DATASET_JSON = os.path.join(os.path.dirname(__file__),'../data/megadepth_1500.json')

class MegaDepth1500(Dataset):
    """
        Streamlined MegaDepth-1500 dataloader. The camera poses & metadata are stored in a formatted json for facilitating 
        the download of the dataset and to keep the setup as simple as possible.
    """
    def __init__(self, json_file, root_dir):
        # Load the info & calibration from the JSON
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.root_dir = root_dir

        if not os.path.exists(self.root_dir):
            raise RuntimeError(
            f"Dataset {self.root_dir} does not exist! \n \
              > If you didn't download the dataset, use the downloader tool: python3 -m modules.dataset.download -h")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = deepcopy(self.data[idx])

        h1, w1 = data['size0_hw']
        h2, w2 = data['size1_hw']

        # Here we resize the images to max_dim = 1200, as described in the paper, and adjust the image such that it is divisible by 32
        # following the protocol of the LoFTR's Dataloader (intrinsics are corrected accordingly). 
        # For adapting this with different resolution, you would need to re-scale intrinsics below.
        image0_raw = cv2.imread(f"{self.root_dir}/{data['pair_names'][0]}")
        image1_raw = cv2.imread(f"{self.root_dir}/{data['pair_names'][1]}")
        if image0_raw is None:
            raise ValueError(f"Failed to read image: {self.root_dir}/{data['pair_names'][0]}")
        if image1_raw is None:
            raise ValueError(f"Failed to read image: {self.root_dir}/{data['pair_names'][1]}")

        image0 = cv2.resize(image0_raw,(w1, h1))

        image1 = cv2.resize(image1_raw,(w2, h2))

        data['image0'] = torch.tensor(image0.astype(np.float32)/255).permute(2,0,1)
        data['image1'] = torch.tensor(image1.astype(np.float32)/255).permute(2,0,1)

        for k,v in data.items():
            if k not in ('dataset_name', 'scene_id', 'pair_id', 'pair_names', 'size0_hw', 'size1_hw', 'image0', 'image1'):
                data[k] = torch.tensor(np.array(v, dtype=np.float32))

        return data

def apply_eval_preset(parsed_args):
    if parsed_args.preset == 'gcdr_noloss_auc5_boost':
        if parsed_args.top_k == 4096:
            parsed_args.top_k = 4096
        if parsed_args.detect_threshold == 0.05:
            parsed_args.detect_threshold = 0.03
        if parsed_args.match_min_cossim == 0.1:
            parsed_args.match_min_cossim = 0.05
        if parsed_args.match_ratio_thresh == 0.82:
            parsed_args.match_ratio_thresh = 0.88
        if parsed_args.pose_epipolar_error == 4.0:
            parsed_args.pose_epipolar_error = 2.5
    return parsed_args


def evaluate_once(dataset, weights, model_config, descriptor_source, keypoint_source):
    geofeat = GeoFeat(
        model_config=model_config,
        weight_path=weights,
        top_k=args.top_k,
        detect_threshold=args.detect_threshold,
        nms_kernel_size=args.nms_kernel_size,
        match_min_cossim=args.match_min_cossim,
        match_ratio_thresh=args.match_ratio_thresh,
        descriptor_source=descriptor_source,
        keypoint_source=keypoint_source,
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    results = []

    for d in tqdm.tqdm(loader, desc="processing"):
        error_infos = compute_pose_error(
            geofeat.match_featnet,
            d,
            pose_epipolar_error=args.pose_epipolar_error,
        )
        results.append(error_infos)

    d_err_auc, errors = compute_maa(results)
    return d_err_auc, errors

if __name__ == "__main__":
    args = apply_eval_preset(args)
    weights = os.path.abspath(args.weights)
    model_config = load_model_config_for_weights(weights)
    
    dataset = MegaDepth1500(json_file = DATASET_JSON, root_dir = DATASET_ROOT)
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    source_pairs = [(args.descriptor_source, args.keypoint_source)]
    if args.compare_sources:
        source_pairs = [
            ('raw', 'raw'),
            ('raw', 'refined'),
            ('refined', 'raw'),
            ('refined', 'refined'),
        ]

    summaries = []
    for descriptor_source, keypoint_source in source_pairs:
        d_err_auc, errors = evaluate_once(
            dataset=dataset,
            weights=weights,
            model_config=model_config,
            descriptor_source=descriptor_source,
            keypoint_source=keypoint_source,
        )
        summary = {
            'descriptor_source': descriptor_source,
            'keypoint_source': keypoint_source,
            'metrics': {k: v * 100 for k, v in d_err_auc.items()},
            'num_pairs': int(len(errors)),
            'eval_args': {
                'top_k': args.top_k,
                'detect_threshold': args.detect_threshold,
                'nms_kernel_size': args.nms_kernel_size,
                'match_min_cossim': args.match_min_cossim,
                'match_ratio_thresh': args.match_ratio_thresh,
                'pose_epipolar_error': args.pose_epipolar_error,
                'preset': args.preset,
            },
        }
        summaries.append(summary)

        print(f'\n==={cur_time}==={args.name}===')
        print(f'descriptor_source: {descriptor_source}, keypoint_source: {keypoint_source}')
        for s_k, s_v in summary['metrics'].items():
            print(f'{s_k}: {s_v}')

    if args.save_json:
        with open(args.save_json, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, indent=2)
