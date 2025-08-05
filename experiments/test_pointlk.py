"""
    Example for testing PointNet-LK.

    No-noise version.
"""

import argparse
import os
import sys
import logging
import numpy
import torch
import torch.utils.data
import torchvision
import time
import random
import shutil
from plyfile import PlyData, PlyElement
import numpy as np
import traceback
import warnings
from scipy.spatial.transform import Rotation
import math
import glob

warnings.filterwarnings('ignore', category=UserWarning)

# addpath('../')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import ptlk
# from ptlk import attention_v1
# from ptlk import mamba3d_v1
# from ptlk import mamba3d_v3
# from ptlk import mamba3d_v4
# from ptlk import fast_point_attention
# from ptlk import cformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ptlk.data.datasets as datasets
from ptlk.data.datasets import ModelNet, ShapeNet2, C3VDDataset, C3VDset4tracking, C3VDset4tracking_test, VoxelizationConfig
#from ptlk.data.datasets import SinglePairDataset, SinglePairTrackingDataset
from ptlk.data.datasets import C3VDset4tracking_test_random_sample, CADset4tracking_fixed_perturbation_random_sample
import ptlk.data.transforms as transforms
import ptlk.pointlk as pointlk
from ptlk import so3, se3


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def options(argv=None):
    parser = argparse.ArgumentParser(description='PointNet-LK')

    # required.
    parser.add_argument('-o', '--outfile', required=True, type=str,
                        metavar='FILENAME', help='output filename (.csv)')
    parser.add_argument('-i', '--dataset-path', required=True, type=str,
                        metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
    parser.add_argument('-c', '--categoryfile', required=True, type=str,
                        metavar='PATH', help='path to the categories to be tested') # eg. './sampledata/modelnet40_half1.txt'
    parser.add_argument('-p', '--perturbations', required=False, type=str,
                        metavar='PATH', help='path to the perturbation file') # see. generate_perturbations.py

    # settings for input data
    parser.add_argument('--dataset-type', default='modelnet', choices=['modelnet', 'c3vd'],
                        metavar='DATASET', help='dataset type (default: modelnet)')
    parser.add_argument('--format', default='wv', choices=['wv', 'wt'],
                        help='perturbation format (default: wv (twist)) (wt: rotation and translation)') # the output is always in twist format
    parser.add_argument('--num-points', default=1024, type=int,
                        metavar='N', help='points in point-cloud (default: 1024)')
    # C3VD pairing mode settings
    parser.add_argument('--pair-mode', default='one_to_one', choices=['one_to_one', 'scene_reference'],
                        help='Point cloud pairing mode: one_to_one (each source cloud pairs with specific target cloud) or scene_reference (each scene uses one shared target cloud)')
    parser.add_argument('--reference-name', default=None, type=str,
                        help='Target cloud name used in scene reference mode, default uses first cloud in scene')

    # settings for PointNet-LK
    parser.add_argument('--max-iter', default=20, type=int,
                        metavar='N', help='max-iter on LK. (default: 20)')
    parser.add_argument('--pretrained', default='', type=str,
                        metavar='PATH', help='path to trained model file (default: null (no-use))')
    parser.add_argument('--transfer-from', default='', type=str,
                        metavar='PATH', help='path to classifier feature (default: null (no-use))')
    parser.add_argument('--dim-k', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')
    parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
                        help='symmetric function (default: max)')
    parser.add_argument('--delta', default=1.0e-2, type=float,
                        metavar='D', help='step size for approx. Jacobian (default: 1.0e-2)')
    
    parser.add_argument('--model-type', default='pointnet', choices=['pointnet', 'attention', 'mamba3d', 'mamba3d_v2', 'fast_attention', 'cformer', 'mamba3d_v3', 'mamba3d_v4'],
                        help='Select model type: pointnet, attention, mamba3d, mamba3d_v2, fast_attention or cformer (default: pointnet)')
    
    parser.add_argument('--num-attention-blocks', default=3, type=int,
                        metavar='N', help='Number of attention blocks in attention module (default: 3)')
    parser.add_argument('--num-heads', default=8, type=int,
                        metavar='N', help='Number of heads in multi-head attention (default: 8)')
    
    # Mamba3D model parameters
    parser.add_argument('--num-mamba-blocks', default=3, type=int,
                        metavar='N', help='Number of Mamba blocks in Mamba3D module (default: 3)')
    parser.add_argument('--d-state', default=16, type=int,
                        metavar='N', help='Mamba state space dimension (default: 16)')
    parser.add_argument('--expand', default=2, type=float,
                        metavar='N', help='Mamba expansion factor (default: 2)')
    
    # Fast point attention model parameters
    parser.add_argument('--num-fast-attention-blocks', default=2, type=int,
                        metavar='N', help='Number of attention blocks in fast point attention module (default: 2)')
    parser.add_argument('--fast-attention-scale', default=1, type=int,
                        metavar='N', help='Scale factor for fast point attention model (default: 1, larger values mean lighter model)')
    
    # Cformer model parameters
    parser.add_argument('--num-proxy-points', default=8, type=int,
                        metavar='N', help='Number of proxy points in Cformer model (default: 8)')
    parser.add_argument('--num-blocks', default=2, type=int,
                        metavar='N', help='Number of blocks in Cformer model (default: 2)')

    # settings for on testing
    parser.add_argument('-l', '--logfile', default='', type=str,
                        metavar='LOGNAME', help='path to logfile (default: null (no logging))')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--device', default='cpu', type=str,
                        metavar='DEVICE', help='use CUDA if available (default: cpu)')
    parser.add_argument('--max-samples', default=2000, type=int,
                        metavar='N', help='Maximum number of test samples (default: 2000)')
                        
    # Visualization settings
    parser.add_argument('--visualize-pert', default=None, type=str, nargs='*',
                        help='List of perturbation filenames to visualize (e.g. pert_010.csv pert_020.csv), path not included')
    parser.add_argument('--visualize-samples', default=1, type=int,
                        help='Number of samples to visualize per perturbation file (default: 1)')
                        

    # Voxelization related parameters
    parser.add_argument('--use-voxelization', action='store_true', default=True,
                        help='Enable voxelization preprocessing (default: True)')
    parser.add_argument('--no-voxelization', dest='use_voxelization', action='store_false',
                        help='Disable voxelization, use simple resampling')
    parser.add_argument('--voxel-size', default=0.05, type=float,
                        metavar='SIZE', help='Voxel size (default: 0.05)')
    parser.add_argument('--voxel-grid-size', default=32, type=int,
                        metavar='SIZE', help='Voxel grid size (default: 32)')
    parser.add_argument('--max-voxel-points', default=100, type=int,
                        metavar='N', help='Maximum points per voxel (default: 100)')
    parser.add_argument('--max-voxels', default=20000, type=int,
                        metavar='N', help='Maximum number of voxels (default: 20000)')
    parser.add_argument('--min-voxel-points-ratio', default=0.1, type=float,
                        metavar='RATIO', help='Minimum voxel points ratio threshold (default: 0.1)')


    # Add perturbation directory parameter
    parser.add_argument('--perturbation-dir', default=None, type=str,
                        metavar='PATH', help='Perturbation directory path, will process all perturbation files in the directory')
                        

    # Single perturbation file parameter support
    parser.add_argument('--perturbation-file', default=None, type=str,
                        metavar='PATH', help='Single perturbation file path (e.g., gt_poses.csv)')


    # Single pair point cloud input mode parameters
    parser.add_argument('--single-pair-mode', action='store_true', default=False,
                        help='Enable single pair point cloud input mode')
    parser.add_argument('--source-cloud', default=None, type=str,
                        metavar='PATH', help='Source point cloud file path')
    parser.add_argument('--target-cloud', default=None, type=str,
                        metavar='PATH', help='Target point cloud file path')
    parser.add_argument('--single-perturbation', default=None, type=str,
                        metavar='VALUES', help='Single perturbation values, comma-separated. Format: rx,ry,rz,tx,ty,tz')
    parser.add_argument('--enhanced-output', action='store_true', default=False,
                        help='Output enhanced information including input perturbation and predicted transformation')

    args = parser.parse_args(argv)
    return args

def main(args):

    # Single pair point cloud input mode processing
    if args.single_pair_mode:

        if not args.source_cloud or not args.target_cloud:
            print("Error: Single pair mode requires --source-cloud and --target-cloud parameters")
            return
        
        if not args.single_perturbation:
            print("Error: Single pair mode requires --single-perturbation parameter")
            return
        
        if not os.path.exists(args.source_cloud):
            print(f"Error: Source cloud file does not exist: {args.source_cloud}")
            return
        
        if not os.path.exists(args.target_cloud):
            print(f"Error: Target cloud file does not exist: {args.target_cloud}")
            return
        
        print(f"\n====== Single Pair Point Cloud Mode ======")
        print(f"Source cloud: {args.source_cloud}")
        print(f"Target cloud: {args.target_cloud}")
        print(f"Perturbation: {args.single_perturbation}")
        print(f"Enhanced output: {args.enhanced_output}")
        
        process_single_pair(args)
        return
    

    # Create empty list to store all perturbation files to process
    perturbation_files = []
    

    # If perturbation directory is specified, first add all .csv files in the directory
    if args.perturbation_dir and os.path.exists(args.perturbation_dir):
        print(f"\n====== Perturbation Directory ======")
        print(f"Scanning perturbation directory: {args.perturbation_dir}")
        for filename in sorted(os.listdir(args.perturbation_dir)):
            if filename.endswith('.csv'):
                full_path = os.path.join(args.perturbation_dir, filename)
                perturbation_files.append(full_path)
                print(f"Found perturbation file: {filename}")
    

    # If individual perturbation file is specified (via --perturbations), add it to the list
    if args.perturbations and os.path.exists(args.perturbations):
        if args.perturbations not in perturbation_files:
            perturbation_files.append(args.perturbations)
            print(f"Added individually specified perturbation file: {os.path.basename(args.perturbations)}")
    

    # If individual perturbation file is specified (via --perturbation-file), add it to the list
    if args.perturbation_file and os.path.exists(args.perturbation_file):
        if args.perturbation_file not in perturbation_files:
            perturbation_files.append(args.perturbation_file)
            print(f"Added perturbation file: {os.path.basename(args.perturbation_file)}")
    

    # Check if there are perturbation files to process
    if not perturbation_files:
        print("Error: No perturbation files found. Please use --perturbation-dir to specify perturbation directory, --perturbations to specify perturbation file, or --perturbation-file to specify perturbation file.")
        return
    
    print(f"Total found {len(perturbation_files)} perturbation files to process")
    
    act = Action(args)
    
    # Process each perturbation file sequentially
    for i, pert_file in enumerate(perturbation_files):
        filename = os.path.basename(pert_file)
        print(f"\n====== Processing perturbation file [{i+1}/{len(perturbation_files)}]: {filename} ======")
        
        # Save original parameter values
        original_perturbations = args.perturbations
        original_outfile = args.outfile
        original_logfile = args.logfile
        
        is_single_file = len(perturbation_files) == 1 and (args.perturbations == pert_file or args.perturbation_file == pert_file)
        
        # Extract perturbation angle information (if any)
        angle_str = ""
        if filename.startswith("pert_") and "_" in filename:
            parts = filename.split("_")
            if len(parts) >= 2:
                angle_str = parts[1].split(".")[0]
        
        # Create output filename for current perturbation file
        if angle_str:
            # Create separate directory for each angle
            output_dir = os.path.join(os.path.dirname(args.outfile), f"angle_{angle_str}")
            os.makedirs(output_dir, exist_ok=True)
            
            base_filename = os.path.splitext(filename)[0]
            current_outfile = os.path.join(output_dir, f"results_{base_filename}.log")
            log_file = os.path.join(output_dir, f"log_{base_filename}.log")
        else:
            output_dir = os.path.dirname(args.outfile)
            os.makedirs(output_dir, exist_ok=True)
            
            base_filename = os.path.splitext(filename)[0]
            current_outfile = os.path.join(output_dir, f"results_{base_filename}.log")
            log_file = os.path.join(output_dir, f"log_{base_filename}.log")
        
        # Set current parameter values
        args.perturbations = pert_file
        args.outfile = current_outfile
        args.logfile = log_file
        
        print(f"Output file: {args.outfile}")
        print(f"Log file: {args.logfile}")
        
        testset = get_datasets(args)
        
        act.update_perturbation(args.perturbations, current_outfile)
        
        run(args, testset, act)
        
        # Restore original parameters
        args.perturbations = original_perturbations
        args.outfile = original_outfile
        args.logfile = original_logfile
        
        # Clean up memory
        del testset
        torch.cuda.empty_cache()
        import gc
        gc.collect()


def run(args, testset, action):
    # Custom dataset wrapper that handles exceptions
    class DatasetWrapper(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.valid_indices = []
            self.pre_check_dataset()
            
        def pre_check_dataset(self):
            """Pre-check a few samples in the dataset"""
            print(f"Starting to check samples in the dataset...")
            total_samples = len(self.dataset)
            
            # Only check first 10 samples or 1% of total, whichever is smaller
            check_count = min(10, max(1, total_samples // 100))
            print(f"Checking {check_count} samples out of {total_samples} total samples...")
            
            # Quick check of few samples
            valid_count = 0
            for idx in range(check_count):
                try:
                    p0, p1, igt = self.dataset[idx]
                    
                    # Check point cloud format and size
                    if not isinstance(p0, torch.Tensor) or not isinstance(p1, torch.Tensor):
                        print(f"Warning: Sample {idx} has incorrect point cloud format")
                        continue
                        
                    # Check transformation matrix
                    if igt is None or not isinstance(igt, torch.Tensor):
                        print(f"Warning: Sample {idx} has invalid transformation matrix")
                        continue
                    
                    # Check if there are valid values
                    if not torch.isfinite(p0).all() or not torch.isfinite(p1).all() or not torch.isfinite(igt).all():
                        print(f"Warning: Sample {idx} contains non-finite values")
                        continue
                    
                    # Check if point cloud size is reasonable
                    if p0.shape[0] == 0 or p1.shape[0] == 0:
                        print(f"Warning: Sample {idx} has empty point cloud")
                        continue
                    
                    valid_count += 1
                    
                except Exception as e:
                    print(f"Warning: Sample {idx} failed validation: {str(e)}")
                    continue
            
            print(f"Validation completed: {valid_count}/{check_count} samples are valid")
            
            if valid_count > 0:
                print(f"Dataset appears healthy, assuming all {total_samples} samples are usable")
                self.valid_indices = list(range(total_samples))
            else:
                print(f"Warning: No valid samples found in initial check, but continuing with full dataset")
                self.valid_indices = list(range(total_samples))
        
        def __len__(self):
            return len(self.valid_indices)
        
        def __getitem__(self, idx):
            try:
                return self.dataset[self.valid_indices[idx]]
            except Exception as e:
                print(f"Warning: Failed to get sample {idx}: {str(e)}")
                return self.dataset[self.valid_indices[-1]]
            
        def get_cloud_info(self, idx):
            """Get point cloud information"""
            try:
                real_idx = self.valid_indices[idx]
                
                if hasattr(self.dataset, 'get_cloud_info'):
                    info = self.dataset.get_cloud_info(real_idx)
                    return info
                    
                elif hasattr(self.dataset, 'cloud_info'):
                    if real_idx in self.dataset.cloud_info:
                        return self.dataset.cloud_info[real_idx]
                
                elif hasattr(self.dataset, 'dataset'):
                    if hasattr(self.dataset.dataset, 'get_cloud_info'):
                        if hasattr(self.dataset, 'indices'):
                            orig_idx = self.dataset.indices[real_idx]
                            return self.dataset.dataset.get_cloud_info(orig_idx)
                    
                    elif hasattr(self.dataset.dataset, 'cloud_info'):
                        if hasattr(self.dataset, 'indices'):
                            orig_idx = self.dataset.indices[real_idx]
                            if orig_idx in self.dataset.dataset.cloud_info:
                                return self.dataset.dataset.cloud_info[orig_idx]
                
                # Build default information
                source_file = None
                target_file = None
                
                try:
                    if hasattr(self.dataset, 'pairs'):
                        if real_idx < len(self.dataset.pairs):
                            source_file, target_file = self.dataset.pairs[real_idx]
                    elif hasattr(self.dataset, 'dataset') and hasattr(self.dataset.dataset, 'pairs'):
                        if hasattr(self.dataset, 'indices'):
                            orig_idx = self.dataset.indices[real_idx]
                            if orig_idx < len(self.dataset.dataset.pairs):
                                source_file, target_file = self.dataset.dataset.pairs[orig_idx]
                except Exception as e:
                    print(f"Error getting file paths: {str(e)}")
                
                # Extract information from paths
                scene_name = "unknown"
                source_seq = f"{real_idx:04d}"
                
                if source_file:
                    norm_path = source_file.replace('\\', '/')
                    
                    if 'C3VD_ply_source' in norm_path:
                        parts = norm_path.split('/')
                        idx = [i for i, part in enumerate(parts) if part == 'C3VD_ply_source']
                        if idx and idx[0] + 1 < len(parts):
                            scene_name = parts[idx[0] + 1]
                    
                    basename = os.path.basename(source_file)
                    if basename.endswith("_depth_pcd.ply") and basename[:4].isdigit():
                        source_seq = basename[:4]
                    else:
                        import re
                        numbers = re.findall(r'\d+', basename)
                        if numbers:
                            source_seq = numbers[0].zfill(4)
                
                return {
                    'identifier': f"{scene_name}_{source_seq}",
                    'scene': scene_name,
                    'sequence': source_seq,
                    'source_file': source_file,
                    'target_file': target_file
                }
                
            except Exception as e:
                print(f"Warning: Failed to get point cloud info: {str(e)}")
                return None
            
        def get_original_clouds(self, idx):
            """Get original point clouds"""
            try:
                real_idx = self.valid_indices[idx]
                
                if hasattr(self.dataset, 'get_original_clouds'):
                    original_source, original_target = self.dataset.get_original_clouds(real_idx)
                    if original_source is not None and original_target is not None:
                        return original_source, original_target
                
                elif hasattr(self.dataset, 'dataset'):
                    if hasattr(self.dataset.dataset, 'get_original_clouds'):
                        if hasattr(self.dataset, 'indices'):
                            orig_idx = self.dataset.indices[real_idx]
                            return self.dataset.dataset.get_original_clouds(orig_idx)
                
                info = self.get_cloud_info(idx)
                if info and 'original_source' in info and 'original_target' in info:
                    return info['original_source'], info['original_target']
                
                try:
                    p0, p1, _ = self.dataset[real_idx]
                    return p1.clone(), p0.clone()  # Note: p1 is source, p0 is target
                except:
                    pass
                
                return None, None
                
            except Exception as e:
                print(f"Warning: Failed to get original point clouds: {str(e)}")
                return None, None
            
        def get_identifier(self, idx):
            """Get point cloud identifier"""
            try:
                real_idx = self.valid_indices[idx]
                
                if hasattr(self.dataset, 'get_identifier'):
                    identifier = self.dataset.get_identifier(real_idx)
                    if identifier:
                        return identifier
                
                elif hasattr(self.dataset, 'dataset'):
                    if hasattr(self.dataset.dataset, 'get_identifier'):
                        if hasattr(self.dataset, 'indices'):
                            orig_idx = self.dataset.indices[real_idx]
                            identifier = self.dataset.dataset.get_identifier(orig_idx)
                            if identifier:
                                return identifier
                
                info = self.get_cloud_info(idx)
                if info and 'identifier' in info:
                    return info['identifier']
                
                if info:
                    scene = info.get('scene', 'unknown')
                    seq = info.get('sequence', f"{real_idx:04d}")
                    return f"{scene}_{seq}"
                
                return f"unknown_{real_idx:04d}"
                
            except Exception as e:
                print(f"Warning: Failed to get identifier: {str(e)}")
                return f"unknown_{idx:04d}"

    # CUDA availability check
    print(f"\n====== CUDA Availability Check ======")
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        print(f"CUDA available: No (testing will run on CPU, which will be slow)")
        args.device = 'cpu'
    
    args.device = torch.device(args.device)
    print(f"Using device: {args.device}")

    LOGGER.debug('Testing (PID=%d), %s', os.getpid(), args)

    model = action.create_model()
    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
    model.to(args.device)

    # Display model information
    print(f"\n====== Model Information ======")
    print(f"Parameter count: {sum(p.numel() for p in model.parameters())}")
    print(f"Model parameters on CUDA: {next(model.parameters()).is_cuda}")
    if str(args.device) != 'cpu':
        print(f"Current GPU memory usage: {torch.cuda.memory_allocated()/1024**2:.1f}MB")

    # Wrap dataset to handle exceptions
    print(f"\n====== Dataset Preparation ======")
    print(f"Original dataset size: {len(testset)}")
    testset = DatasetWrapper(testset)
    print(f"Filtered dataset size: {len(testset)}")

    # Custom collate function to handle None values
    def custom_collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            raise ValueError("All samples in batch are invalid")
        return torch.utils.data.dataloader.default_collate(batch)

    if len(testset) == 0:
        print("Error: No valid samples in dataset, cannot continue testing.")
        return

    # dataloader
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=1, shuffle=False, 
        num_workers=min(args.workers, 1),  # Reduce worker count to lower error rate
        collate_fn=custom_collate_fn)
    
    print(f"\n====== Dataset Information ======")
    print(f"Test samples: {len(testset)}")
    print(f"Points per point cloud: {args.num_points}")
    print(f"Batch size: 1")

    # testing
    print(f"\n====== Starting Test ======")
    LOGGER.debug('tests, begin')

    os.makedirs(os.path.dirname(action.filename), exist_ok=True)
    
    print(f"\n====== Using Perturbation File Test Mode ======")
    success_count, total_count = action.eval_1(model, testloader, args.device)
    print(f"Perturbation file test evaluation completed, successfully processed {success_count}/{total_count} samples")
    LOGGER.debug('tests, end')


class Action:
    def __init__(self, args):
        self.filename = args.outfile
        # PointNet
        self.transfer_from = args.transfer_from
        self.dim_k = args.dim_k
        
        self.model_type = args.model_type
        self.num_attention_blocks = args.num_attention_blocks
        self.num_heads = args.num_heads
        
        # Mamba3D attributes
        self.num_mamba_blocks = args.num_mamba_blocks
        self.d_state = args.d_state
        self.expand = args.expand
        
        # Fast point attention attributes
        self.num_fast_attention_blocks = args.num_fast_attention_blocks
        self.fast_attention_scale = args.fast_attention_scale
        
        # Cformer attributes
        self.num_proxy_points = getattr(args, 'num_proxy_points', 8)
        self.num_blocks = getattr(args, 'num_blocks', 2)
        
        self.sym_fn = None
        if args.model_type == 'attention':
            from ptlk import attention_v1
            if args.symfn == 'max':
                self.sym_fn = attention_v1.symfn_max
            elif args.symfn == 'avg':
                self.sym_fn = attention_v1.symfn_avg
            else:
                self.sym_fn = attention_v1.symfn_attention_pool
        elif args.model_type == 'mamba3d':
            from ptlk import mamba3d_v1
            if args.symfn == 'max':
                self.sym_fn = mamba3d_v1.symfn_max
            elif args.symfn == 'avg':
                self.sym_fn = mamba3d_v1.symfn_avg
            elif args.symfn == 'selective':
                self.sym_fn = mamba3d_v1.symfn_selective
            else:
                self.sym_fn = mamba3d_v1.symfn_max
        elif args.model_type == 'mamba3d_v2':
            from ptlk import mamba3d_v2
            ptnet = mamba3d_v2.Mamba3D_features(
                dim_k=self.dim_k, 
                sym_fn=self.sym_fn,
                scale=1,
                num_mamba_blocks=self.num_mamba_blocks,
                d_state=self.d_state,
                expand=self.expand
            )
            if self.transfer_from and os.path.isfile(self.transfer_from):
                try:
                    pretrained_dict = torch.load(self.transfer_from, map_location='cpu')
                    ptnet.load_state_dict(pretrained_dict)
                    print(f"Successfully loaded Mamba3D_v2 pretrained weights: {self.transfer_from}")
                except Exception as e:
                    print(f"Failed to load Mamba3D_v2 pretrained weights: {e}")
                    print("Continue using random initialization weights")
        elif args.model_type == 'mamba3d_v3':
            from ptlk import mamba3d_v3
            ptnet = mamba3d_v3.Mamba3D_features(
                dim_k=self.dim_k, 
                sym_fn=self.sym_fn,
                scale=1,
                num_mamba_blocks=self.num_mamba_blocks,
                d_state=self.d_state,
                expand=self.expand
            )
            if self.transfer_from and os.path.isfile(self.transfer_from):
                try:
                    pretrained_dict = torch.load(self.transfer_from, map_location='cpu')
                    ptnet.load_state_dict(pretrained_dict)
                    print(f"Successfully loaded Mamba3D_v3 pretrained weights: {self.transfer_from}")
                except Exception as e:
                    print(f"Failed to load Mamba3D_v3 pretrained weights: {e}")
                    print("Continue using random initialization weights")
        elif args.model_type == 'mamba3d_v4':
            from ptlk import mamba3d_v4
            ptnet = mamba3d_v4.Mamba3D_features(
                dim_k=self.dim_k, 
                sym_fn=self.sym_fn,
                scale=1,
                num_mamba_blocks=self.num_mamba_blocks,
                d_state=self.d_state,
                expand=self.expand
            )
            if self.transfer_from and os.path.isfile(self.transfer_from):
                try:
                    pretrained_dict = torch.load(self.transfer_from, map_location='cpu')
                    ptnet.load_state_dict(pretrained_dict)
                    print(f"Successfully loaded Mamba3D_v4 pretrained weights: {self.transfer_from}")
                except Exception as e:
                    print(f"Failed to load Mamba3D_v4 pretrained weights: {e}")
                    print("Continue using random initialization weights")
        elif args.model_type == 'fast_attention':
            from ptlk import fast_point_attention
            if args.symfn == 'max':
                self.sym_fn = fast_point_attention.symfn_max
            elif args.symfn == 'avg':
                self.sym_fn = fast_point_attention.symfn_avg
            elif args.symfn == 'selective':
                self.sym_fn = fast_point_attention.symfn_fast_attention_pool
            else:
                self.sym_fn = fast_point_attention.symfn_max
        elif args.model_type == 'cformer':
            from ptlk import cformer
            if args.symfn == 'max':
                self.sym_fn = cformer.symfn_max
            elif args.symfn == 'avg':
                self.sym_fn = cformer.symfn_avg
            elif args.symfn == 'cd_pool':
                self.sym_fn = cformer.symfn_cd_pool
            else:
                self.sym_fn = cformer.symfn_max
        else:
            if args.symfn == 'max':
                self.sym_fn = ptlk.pointnet.symfn_max
            elif args.symfn == 'avg':
                self.sym_fn = ptlk.pointnet.symfn_avg
        
        # LK
        self.delta = args.delta
        self.max_iter = args.max_iter
        self.xtol = 1.0e-7
        self.p0_zero_mean = True
        self.p1_zero_mean = True
        self.num_points = args.num_points
        
        # Visualization parameters
        self.visualize_pert = args.visualize_pert
        self.visualize_samples = args.visualize_samples
        
        if args.perturbations:
            self.current_pert_file = os.path.basename(args.perturbations)
        else:
            self.current_pert_file = "unknown"
        
        # Visualization folder
        if self.visualize_pert is not None and self.current_pert_file in self.visualize_pert:
            vis_dir = os.path.join(os.path.dirname(self.filename), 'visualize')
            os.makedirs(vis_dir, exist_ok=True)
            self.vis_subdir = os.path.join(vis_dir, os.path.splitext(self.current_pert_file)[0])
            os.makedirs(self.vis_subdir, exist_ok=True)
            
            self.vis_log_file = os.path.join(self.vis_subdir, 'visualization_log.txt')
            with open(self.vis_log_file, 'w') as f:
                f.write("# PointNetLK Registration Visualization Log\n")
                f.write("# Perturbation file: {}\n".format(self.current_pert_file))
                f.write("# Creation time: {}\n\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
                f.write("Point cloud pair,Perturbation file,Predicted(w1,w2,w3,v1,v2,v3),Ground truth perturbation(w1,w2,w3,v1,v2,v3)\n")
                f.write("--------------------------------------------------------------------\n")
                
        self.vis_count = 0

    def create_model(self):
        ptnet = self.create_pointnet_features()
        return self.create_from_pointnet_features(ptnet)

    def create_pointnet_features(self):
        if self.model_type == 'attention':
            from ptlk import attention_v1
            ptnet = attention_v1.AttentionNet_features(
                dim_k=self.dim_k, 
                sym_fn=self.sym_fn,
                scale=1,
                num_attention_blocks=self.num_attention_blocks,
                num_heads=self.num_heads
            )
            if self.transfer_from and os.path.isfile(self.transfer_from):
                try:
                    pretrained_dict = torch.load(self.transfer_from, map_location='cpu')
                    ptnet.load_state_dict(pretrained_dict)
                    print(f"Successfully loaded attention pretrained weights: {self.transfer_from}")
                except Exception as e:
                    print(f"Failed to load attention pretrained weights: {e}")
                    print("Continue using random initialization weights")
        elif self.model_type == 'mamba3d':
            from ptlk import mamba3d_v1
            ptnet = mamba3d_v1.Mamba3D_features(
                dim_k=self.dim_k, 
                sym_fn=self.sym_fn,
                scale=1,
                num_mamba_blocks=self.num_mamba_blocks,
                d_state=self.d_state,
                expand=self.expand
            )
            if self.transfer_from and os.path.isfile(self.transfer_from):
                try:
                    pretrained_dict = torch.load(self.transfer_from, map_location='cpu')
                    ptnet.load_state_dict(pretrained_dict)
                    print(f"Successfully loaded Mamba3D pretrained weights: {self.transfer_from}")
                except Exception as e:
                    print(f"Failed to load Mamba3D pretrained weights: {e}")
                    print("Continue using random initialization weights")
        elif self.model_type == 'mamba3d_v2':
            from ptlk import mamba3d_v2
            ptnet = mamba3d_v2.Mamba3D_features(
                dim_k=self.dim_k, 
                sym_fn=self.sym_fn,
                scale=1,
                num_mamba_blocks=self.num_mamba_blocks,
                d_state=self.d_state,
                expand=self.expand
            )
            if self.transfer_from and os.path.isfile(self.transfer_from):
                try:
                    pretrained_dict = torch.load(self.transfer_from, map_location='cpu')
                    ptnet.load_state_dict(pretrained_dict)
                    print(f"Successfully loaded Mamba3D_v2 pretrained weights: {self.transfer_from}")
                except Exception as e:
                    print(f"Failed to load Mamba3D_v2 pretrained weights: {e}")
                    print("Continue using random initialization weights")
        elif self.model_type == 'mamba3d_v3':
            from ptlk import mamba3d_v3
            ptnet = mamba3d_v3.Mamba3D_features(
                dim_k=self.dim_k, 
                sym_fn=self.sym_fn,
                scale=1,
                num_mamba_blocks=self.num_mamba_blocks,
                d_state=self.d_state,
                expand=self.expand
            )
            if self.transfer_from and os.path.isfile(self.transfer_from):
                try:
                    pretrained_dict = torch.load(self.transfer_from, map_location='cpu')
                    ptnet.load_state_dict(pretrained_dict)
                    print(f"Successfully loaded Mamba3D_v3 pretrained weights: {self.transfer_from}")
                except Exception as e:
                    print(f"Failed to load Mamba3D_v3 pretrained weights: {e}")
                    print("Continue using random initialization weights")
        elif self.model_type == 'mamba3d_v4':
            from ptlk import mamba3d_v4
            ptnet = mamba3d_v4.Mamba3D_features(
                dim_k=self.dim_k, 
                sym_fn=self.sym_fn,
                scale=1,
                num_mamba_blocks=self.num_mamba_blocks,
                d_state=self.d_state,
                expand=self.expand
            )
            if self.transfer_from and os.path.isfile(self.transfer_from):
                try:
                    pretrained_dict = torch.load(self.transfer_from, map_location='cpu')
                    ptnet.load_state_dict(pretrained_dict)
                    print(f"Successfully loaded Mamba3D_v4 pretrained weights: {self.transfer_from}")
                except Exception as e:
                    print(f"Failed to load Mamba3D_v4 pretrained weights: {e}")
                    print("Continue using random initialization weights")
        elif self.model_type == 'fast_attention':
            from ptlk import fast_point_attention
            ptnet = fast_point_attention.FastPointAttention_features(
                dim_k=self.dim_k, 
                sym_fn=self.sym_fn,
                scale=self.fast_attention_scale,
                num_attention_blocks=self.num_fast_attention_blocks
            )
            if self.transfer_from and os.path.isfile(self.transfer_from):
                try:
                    pretrained_dict = torch.load(self.transfer_from, map_location='cpu')
                    ptnet.load_state_dict(pretrained_dict)
                    print(f"Successfully loaded fast point attention pretrained weights: {self.transfer_from}")
                except Exception as e:
                    print(f"Failed to load fast point attention pretrained weights: {e}")
                    print("Continue using random initialization weights")
        elif self.model_type == 'cformer':
            from ptlk import cformer
            ptnet = cformer.CFormer_features(
                dim_k=self.dim_k, 
                sym_fn=self.sym_fn,
                scale=1,
                num_proxy_points=self.num_proxy_points,
                num_blocks=self.num_blocks
            )
            if self.transfer_from and os.path.isfile(self.transfer_from):
                try:
                    pretrained_dict = torch.load(self.transfer_from, map_location='cpu')
                    ptnet.load_state_dict(pretrained_dict)
                    print(f"Successfully loaded Cformer pretrained weights: {self.transfer_from}")
                except Exception as e:
                    print(f"Failed to load Cformer pretrained weights: {e}")
                    print("Continue using random initialization weights")
        else:
            ptnet = ptlk.pointnet.PointNet_features(self.dim_k, sym_fn=self.sym_fn)
            if self.transfer_from and os.path.isfile(self.transfer_from):
                try:
                    pretrained_dict = torch.load(self.transfer_from, map_location='cpu')
                    
                    if any(key.startswith('features.') for key in pretrained_dict.keys()):
                        print(f"Detected classifier weights, extracting feature extractor part...")
                        feature_dict = {}
                        for key, value in pretrained_dict.items():
                            if key.startswith('features.'):
                                new_key = key[9:]
                                feature_dict[new_key] = value
                        
                        ptnet.load_state_dict(feature_dict)
                        print(f"Successfully extracted and loaded PointNet feature weights from classifier weights: {self.transfer_from}")
                    else:
                        ptnet.load_state_dict(pretrained_dict)
                        print(f"Successfully loaded PointNet pretrained weights: {self.transfer_from}")
                        
                except Exception as e:
                    print(f"Failed to load PointNet pretrained weights: {e}")
                    print("Continue using random initialization weights")
        
        return ptnet

    def create_from_pointnet_features(self, ptnet):
        return ptlk.pointlk.PointLK(ptnet, self.delta)

    def eval_1__header(self, fout):
        cols = ['sample_id', 'scene_name', 'sequence', 'rotation_error', 'translation_error', 'total_error']
        print(','.join(map(str, cols)), file=fout)
        fout.flush()

    def eval_1__write(self, fout, ig_gt, g_hat, sample_info=None):
        dg = g_hat.bmm(ig_gt)
        dx = ptlk.se3.log(dg)
        
        rot_error = dx[:, :3]
        trans_error = dx[:, 3:]
        
        rot_norm = rot_error.norm(p=2, dim=1)
        trans_norm = trans_error.norm(p=2, dim=1)
        total_norm = dx.norm(p=2, dim=1)
        
        for i in range(g_hat.size(0)):
            if sample_info:
                sample_id = sample_info.get('identifier', f'sample_{i}')
                scene_name = sample_info.get('scene', 'unknown')
                sequence = sample_info.get('sequence', f'{i:04d}')
            else:
                sample_id = f'sample_{i}'
                scene_name = 'unknown'
                sequence = f'{i:04d}'
            
            vals = [sample_id, scene_name, sequence, 
                   rot_norm[i].item(), trans_norm[i].item(), total_norm[i].item()]
            print(','.join(map(str, vals)), file=fout)
        fout.flush()

    def eval_1(self, model, testloader, device):
        model.eval()
        success_count = 0
        total_count = 0
        error_count = 0
        
        total_rot_error = 0.0
        total_trans_error = 0.0
        total_total_error = 0.0
        all_rot_errors = []
        all_trans_errors = []
        all_total_errors = []
        
        current_vis_count = 0
        need_visualization = (self.visualize_pert is not None and 
                             self.current_pert_file in self.visualize_pert and
                             current_vis_count < self.visualize_samples)
        
        if need_visualization:
            num_samples = min(len(testloader), 100)
            vis_indices = random.sample(range(num_samples), min(num_samples, self.visualize_samples))
            print(f"\n====== Visualization Settings ======")
            print(f"Will visualize the following sample indices for perturbation file {self.current_pert_file}: {vis_indices}")
        
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        
        with open(self.filename, 'w') as fout:
            print(f"# PointNetLK Registration Test Results", file=fout)
            print(f"# Perturbation file: {self.current_pert_file}", file=fout)
            print(f"# Test time: {time.strftime('%Y-%m-%d %H:%M:%S')}", file=fout)
            print(f"# Rotation error unit: radians", file=fout)
            print(f"# Translation error unit: meters", file=fout)
            print(f"# =====================================", file=fout)
            
            self.eval_1__header(fout)
            with torch.no_grad():
                total_samples = len(testloader)
                start_time = time.time()
                
                for i, data in enumerate(testloader):
                    batch_start_time = time.time()
                    total_count += 1
                    
                    do_visualize = need_visualization and i in vis_indices
                    
                    try:
                        if i > 0 and i % 100 == 0:
                            torch.cuda.empty_cache()
                            print(f"GPU memory cleanup at sample {i}")
                        
                        if i > 0 and i % 500 == 0:
                            if torch.cuda.is_available():
                                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                                memory_reserved = torch.cuda.memory_reserved() / 1024**3  
                                print(f"Sample {i}: GPU memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
                            else:
                                print(f"Warning: CUDA not available at sample {i}")
                        
                        p0, p1, igt = data
                        
                        if p0.shape[1] != self.num_points or p1.shape[1] != self.num_points:
                            print(f"Warning: Batch {i} point cloud shape does not match expected: p0={p0.shape}, p1={p1.shape}, expected points={self.num_points}")
                        
                        if not torch.isfinite(p0).all() or not torch.isfinite(p1).all():
                            print(f"Warning: Batch {i} contains NaN values, attempting to fix")
                            p0 = torch.nan_to_num(p0, nan=0.0, posinf=1.0, neginf=-1.0)
                            p1 = torch.nan_to_num(p1, nan=0.0, posinf=1.0, neginf=-1.0)
                        
                        if not torch.isfinite(igt).all():
                            print(f"Warning: Batch {i} perturbation matrix contains invalid values, skipping this sample")
                            error_count += 1
                            dummy_vals = ['nan'] * 6
                            print(','.join(dummy_vals), file=fout)
                            fout.flush()
                            continue
                        
                        scene_name = "unknown"
                        source_seq = "0000"
                        source_file_path = None
                        target_file_path = None
                        
                        sample_info = {}
                        if hasattr(testloader.dataset, 'get_cloud_info'):
                            cloud_info = testloader.dataset.get_cloud_info(i)
                            if cloud_info:
                                sample_info = cloud_info
                                scene_name = cloud_info.get('scene', scene_name)
                                source_seq = cloud_info.get('sequence', source_seq)
                                source_file_path = cloud_info.get('source_file')
                                target_file_path = cloud_info.get('target_file')
                        
                        if not sample_info:
                            try:
                                if hasattr(testloader.dataset, 'dataset') and hasattr(testloader.dataset.dataset, 'pairs'):
                                    idx = testloader.dataset.valid_indices[i] if hasattr(testloader.dataset, 'valid_indices') else i
                                    if idx < len(testloader.dataset.dataset.pairs):
                                        source_file_path, target_file_path = testloader.dataset.dataset.pairs[idx]
                                elif hasattr(testloader.dataset, 'dataset') and hasattr(testloader.dataset.dataset, 'dataset'):
                                    if hasattr(testloader.dataset.dataset.dataset, 'pairs'):
                                        idx = testloader.dataset.valid_indices[i] if hasattr(testloader.dataset, 'valid_indices') else i
                                        if idx < len(testloader.dataset.dataset.dataset.pairs):
                                            source_file_path, target_file_path = testloader.dataset.dataset.dataset.pairs[idx]
                            except Exception as e:
                                pass
                            
                            if source_file_path:
                                try:
                                    norm_path = source_file_path.replace('\\', '/')
                                    if 'C3VD_ply_source' in norm_path:
                                        parts = norm_path.split('/')
                                        idx = [i for i, part in enumerate(parts) if part == 'C3VD_ply_source']
                                        if idx and idx[0] + 1 < len(parts):
                                            scene_name = parts[idx[0] + 1]
                                    
                                    basename = os.path.basename(source_file_path)
                                    if basename.endswith("_depth_pcd.ply") and basename[:4].isdigit():
                                        source_seq = basename[:4]
                                    else:
                                        import re
                                        numbers = re.findall(r'\d+', basename)
                                        if numbers:
                                            source_seq = numbers[0].zfill(4)
                                except Exception as e:
                                    pass
                            
                            sample_info = {
                                'identifier': f"{scene_name}_{source_seq}",
                                'scene': scene_name,
                                'sequence': source_seq,
                                'source_file': source_file_path,
                                'target_file': target_file_path
                            }
                        
                        try:
                            res = self.do_estimate(p0, p1, model, device)
                            
                            if not torch.isfinite(res).all():
                                print(f"Warning: Batch {i} registration result contains invalid values")
                                error_count += 1
                                dummy_vals = ['nan'] * 6
                                print(','.join(dummy_vals), file=fout)
                                fout.flush()
                                
                        except Exception as registration_error:
                            print(f"Registration error: Batch {i} registration failed: {str(registration_error)}")
                            error_count += 1
                            dummy_vals = ['nan'] * 6
                            print(','.join(dummy_vals), file=fout)
                            fout.flush()
                            continue
                        
                        ig_gt = igt.cpu().contiguous().view(-1, 4, 4)
                        g_hat = res.cpu().contiguous().view(-1, 4, 4)

                        try:
                            dg = g_hat.bmm(ig_gt)
                            dx = ptlk.se3.log(dg)
                            
                            if not torch.isfinite(dx).all():
                                print(f"Warning: Batch {i} error calculation contains invalid values")
                                error_count += 1
                                dummy_vals = ['nan'] * 6
                                print(','.join(dummy_vals), file=fout)
                                fout.flush()
                                continue
                            
                            rot_error = dx[:, :3]
                            trans_error = dx[:, 3:]
                            
                            rot_norm = rot_error.norm(p=2, dim=1)
                            trans_norm = trans_error.norm(p=2, dim=1)
                            total_norm = dx.norm(p=2, dim=1)
                            
                            if not (torch.isfinite(rot_norm).all() and torch.isfinite(trans_norm).all() and torch.isfinite(total_norm).all()):
                                print(f"Warning: Batch {i} error norm calculation contains invalid values")
                                error_count += 1
                                dummy_vals = ['nan'] * 6
                                print(','.join(dummy_vals), file=fout)
                                fout.flush()
                                continue
                            
                        except Exception as error_calc_error:
                            print(f"Error calculating error: Batch {i} error calculation failed: {str(error_calc_error)}")
                            error_count += 1
                            dummy_vals = ['nan'] * 6
                            print(','.join(dummy_vals), file=fout)
                            fout.flush()
                            continue
                        
                        total_rot_error += rot_norm.item()
                        total_trans_error += trans_norm.item()
                        total_total_error += total_norm.item()
                        all_rot_errors.append(rot_norm.item())
                        all_trans_errors.append(trans_norm.item())
                        all_total_errors.append(total_norm.item())
                        
                        self.eval_1__write(fout, ig_gt, g_hat, sample_info)
                        success_count += 1
                        
                        if do_visualize and current_vis_count < self.visualize_samples:
                            try:
                                print(f"\n====== Processing visualization sample {i} ======")
                                
                                x_hat = ptlk.se3.log(g_hat)[0]
                                mx_gt = ptlk.se3.log(ig_gt)[0]
                                
                                cloud_info = {}
                                identifier = f"unknown_{i:04d}"
                                scene_name = "unknown"
                                source_seq = f"{i:04d}"
                                source_file_path = None
                                target_file_path = None
                                
                                if hasattr(testloader.dataset, 'get_cloud_info'):
                                    cloud_info = testloader.dataset.get_cloud_info(i)
                                    if cloud_info:
                                        identifier = cloud_info.get('identifier', identifier)
                                        scene_name = cloud_info.get('scene', scene_name)
                                        source_seq = cloud_info.get('sequence', source_seq)
                                        source_file_path = cloud_info.get('source_file')
                                        target_file_path = cloud_info.get('target_file')
                                        
                                        print(f"Obtained point cloud information from test dataset:")
                                        print(f"  - Identifier: {identifier}")
                                        print(f"  - Scene: {scene_name}")
                                        print(f"  - Sequence: {source_seq}")
                                        print(f"  - Source file: {source_file_path}")
                                        print(f"  - Target file: {target_file_path}")
                                else:
                                    try:
                                        if hasattr(testloader.dataset, 'dataset') and hasattr(testloader.dataset.dataset, 'pairs'):
                                            idx = testloader.dataset.valid_indices[i] if hasattr(testloader.dataset, 'valid_indices') else i
                                            if idx < len(testloader.dataset.dataset.pairs):
                                                source_file_path, target_file_path = testloader.dataset.dataset.pairs[idx]
                                                print(f"Obtained source file: {source_file_path}")
                                                print(f"Obtained target file: {target_file_path}")
                                        elif hasattr(testloader.dataset, 'dataset') and hasattr(testloader.dataset.dataset, 'dataset'):
                                            if hasattr(testloader.dataset.dataset.dataset, 'pairs'):
                                                idx = testloader.dataset.valid_indices[i] if hasattr(testloader.dataset, 'valid_indices') else i
                                                if idx < len(testloader.dataset.dataset.dataset.pairs):
                                                    source_file_path, target_file_path = testloader.dataset.dataset.dataset.pairs[idx]
                                                    print(f"Obtained source file from subset: {source_file_path}")
                                                    print(f"Obtained target file from subset: {target_file_path}")
                                    except Exception as e:
                                        print(f"Error getting file paths: {e}")
                                        traceback.print_exc()
                                    
                                    scene_name = None
                                    source_seq = None
                                    source_file_path = None
                                    target_file_path = None
                                    
                                    if source_file_path:
                                        try:
                                            norm_path = source_file_path.replace('\\', '/')
                                            
                                            if 'C3VD_ply_source' in norm_path:
                                                parts = norm_path.split('/')
                                                idx = [i for i, part in enumerate(parts) if part == 'C3VD_ply_source']
                                                if idx and idx[0] + 1 < len(parts):
                                                    scene_name = parts[idx[0] + 1]
                                                    print(f"Successfully extracted scene name from path: {scene_name}")
                                            
                                            basename = os.path.basename(source_file_path)
                                            
                                            if basename.endswith("_depth_pcd.ply") and basename[:4].isdigit():
                                                source_seq = basename[:4]
                                                print(f"Extracted sequence number from file name: {source_seq}")
                                            else:
                                                import re
                                                numbers = re.findall(r'\d+', basename)
                                                if numbers:
                                                    source_seq = numbers[0].zfill(4)
                                                    print(f"Extracted sequence number from file name: {source_seq}")
                                        except Exception as e:
                                            print(f"Error extracting scene name or sequence: {e}")
                                            traceback.print_exc()
                                
                                source_filename = os.path.join(self.vis_subdir, f"{scene_name}_{source_seq}_source.ply")
                                target_filename = os.path.join(self.vis_subdir, f"{scene_name}_{source_seq}_target.ply")
                                
                                original_source_filename = os.path.join(self.vis_subdir, f"{scene_name}_{source_seq}_original_source.ply")
                                original_target_filename = os.path.join(self.vis_subdir, f"{scene_name}_{source_seq}_original_target.ply")
                                
                                try:
                                    with open(self.vis_log_file, 'a') as f:
                                        predicted = ",".join([f"{val:.6f}" for val in x_hat.tolist()])
                                        gt_values = ",".join([f"{val:.6f}" for val in (-mx_gt).tolist()])
                                        
                                        f.write(f"{identifier},{self.current_pert_file},{predicted},{gt_values}\n")
                                        
                                        f.write(f"Source file: {source_file_path}\n")
                                        f.write(f"Target file: {target_file_path}\n")
                                        f.write(f"Source point cloud size: {p1.shape}, Target point cloud size: {p0.shape}\n")
                                        f.write(f"Registration error: Rotation={rot_norm.item():.6f}, Translation={trans_norm.item():.6f}, Total={total_norm.item():.6f}\n\n")
                                    
                                    print(f"Registration information recorded to log file: {self.vis_log_file}")
                                except Exception as e:
                                    print(f"Error writing to log file: {e}")
                                    traceback.print_exc()
                                
                                success_copy = True
                                if source_file_path and os.path.exists(source_file_path):
                                    try:
                                        print(f"Copying source point cloud: {source_file_path} -> {source_filename}")
                                        shutil.copy2(source_file_path, source_filename)
                                    except Exception as e:
                                        print(f"Failed to copy source point cloud file: {e}")
                                        traceback.print_exc()
                                        success_copy = False
                                else:
                                    print(f"Warning: Unable to find source point cloud file path: {source_file_path}")
                                    success_copy = False
                                
                                if target_file_path and os.path.exists(target_file_path):
                                    try:
                                        print(f"Copying target point cloud: {target_file_path} -> {target_filename}")
                                        shutil.copy2(target_file_path, target_filename)
                                    except Exception as e:
                                        print(f"Failed to copy target point cloud file: {e}")
                                        traceback.print_exc()
                                        success_copy = False
                                else:
                                    print(f"Warning: Unable to find target point cloud file path: {target_file_path}")
                                    success_copy = False
                                
                                try:
                                    original_source, original_target = None, None
                                    
                                    if hasattr(testloader.dataset, 'get_original_clouds'):
                                        original_source, original_target = testloader.dataset.get_original_clouds(i)
                                        if original_source is not None and original_target is not None:
                                            print(f"Successfully obtained original point cloud data")
                                    
                                    def create_ply(points, filename):
                                        if points is None:
                                            print(f"Warning: Unable to save empty point cloud to {filename}")
                                            return False
                                            
                                        if isinstance(points, torch.Tensor):
                                            points_np = points.cpu().numpy()
                                            if len(points_np.shape) > 2:
                                                points_np = points_np.squeeze(0)
                                        else:
                                            points_np = points
                                        
                                        vertex = np.zeros(points_np.shape[0], dtype=[
                                            ('x', 'f4'), ('y', 'f4'), ('z', 'f4')
                                        ])
                                        vertex['x'] = points_np[:, 0]
                                        vertex['y'] = points_np[:, 1]
                                        vertex['z'] = points_np[:, 2]
                                        
                                        el = PlyElement.describe(vertex, 'vertex')
                                        
                                        PlyData([el], text=True).write(filename)
                                        print(f"Point cloud saved to: {filename}")
                                        return True
                                    
                                    if original_source is not None:
                                        create_ply(original_source, original_source_filename)
                                    if original_target is not None:
                                        create_ply(original_target, original_target_filename)
                                    
                                    with open(self.vis_log_file, 'a') as f:
                                        if original_source is not None and original_target is not None:
                                            f.write(f"Original point cloud data saved to: {os.path.basename(original_source_filename)} and {os.path.basename(original_target_filename)}\n\n")
                                        else:
                                            f.write(f"Unable to obtain original point cloud data\n\n")
                                
                                except Exception as e:
                                    print(f"Error saving original point cloud: {e}")
                                    traceback.print_exc()
                                
                                if not success_copy:
                                    print("Attempting to save point cloud data directly...")
                                    try:
                                        p0_np = p0.cpu().numpy().squeeze(0)
                                        p1_np = p1.cpu().numpy().squeeze(0)
                                        
                                        def create_ply(points, filename):
                                            vertex = np.zeros(points.shape[0], dtype=[
                                                ('x', 'f4'), ('y', 'f4'), ('z', 'f4')
                                            ])
                                            vertex['x'] = points[:, 0]
                                            vertex['y'] = points[:, 1]
                                            vertex['z'] = points[:, 2]
                                            
                                            el = PlyElement.describe(vertex, 'vertex')
                                            
                                            PlyData([el], text=True).write(filename)
                                            print(f"Point cloud saved directly to: {filename}")
                                        
                                        create_ply(p1_np, source_filename)
                                        create_ply(p0_np, target_filename)
                                        
                                        with open(self.vis_log_file, 'a') as f:
                                            f.write(f"Note: Unable to copy original file, point cloud data saved directly\n\n")
                                        
                                        success_copy = True
                                    except Exception as e:
                                        print(f"Failed to save point cloud: {e}")
                                        traceback.print_exc()
                                
                                current_vis_count += 1
                                print(f"Visualization completed for sample {identifier} ({current_vis_count}/{self.visualize_samples})")
                                
                                g_matrix = g_hat[0].cpu().numpy()
                                gt_matrix = ig_gt[0].cpu().numpy()

                                with open(self.vis_log_file, 'a') as f:
                                    f.write("\nPredicted transformation matrix (4x4):\n")
                                    for row in g_matrix:
                                        f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f}\n")
                                    
                                    f.write("\nTrue transformation matrix (4x4):\n")
                                    for row in gt_matrix:
                                        f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f}\n")
                                
                            except Exception as e:
                                print(f"Error processing visualization sample: {e}")
                                traceback.print_exc()
                                try:
                                    with open(self.vis_log_file, 'a') as f:
                                        f.write(f"Error processing sample_{i},{self.current_pert_file},error information: {str(e)}\n")
                                        f.write(f"Error stack trace: {traceback.format_exc()}\n\n")
                                except Exception as log_err:
                                    print(f"Failed to record error to log file: {log_err}")
                                
                                current_vis_count += 1
                        
                        batch_time = time.time() - batch_start_time
                        elapsed_time = time.time() - start_time
                        estimated_total = elapsed_time / (i + 1) * total_samples
                        remaining_time = max(0, estimated_total - elapsed_time)
                        
                        error_level = ""
                        if total_norm.item() > 0.5:
                            error_level = "【Registration failed】"
                            error_count += 1
                        elif total_norm.item() > 0.1:
                            error_level = "【Large error】"
                        
                        print(f"Testing: [{i+1}/{total_samples}] {(i+1)/total_samples*100:.1f}% | "
                              f"Rotation error: {rot_norm.item():.6f}, Translation error: {trans_norm.item():.6f} | "
                              f"Total error: {total_norm.item():.6f} {error_level} | "
                              f"Time: {batch_time:.2f} seconds | "
                              f"Remaining: {remaining_time/60:.1f} minutes")
                        
                        LOGGER.info('test, %d/%d, rot_error: %f, trans_error: %f, total_error: %f', 
                                   i, total_samples, rot_norm.item(), trans_norm.item(), total_norm.item())
                        
                    except Exception as e:
                        print(f"Error processing batch {i}: {str(e)}")
                        error_str = str(e).lower()
                        if 'cuda' in error_str or 'cublas' in error_str or 'cudnn' in error_str:
                            print(f"Detected CUDA error, attempting to clear GPU memory...")
                            torch.cuda.empty_cache()
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                                print(f"GPU memory cleanup completed")
                        
                        LOGGER.error('Error in batch %d: %s', i, str(e), exc_info=True)
                        error_count += 1
                        dummy_vals = ['nan'] * 6
                        print(','.join(dummy_vals), file=fout)
                        fout.flush()
        
        total_time = time.time() - start_time
        
        with open(self.filename, 'a') as fout:
            print(f"", file=fout)
            print(f"# =====================================", file=fout)
            print(f"# Test Statistics", file=fout)
            print(f"# =====================================", file=fout)
            print(f"# Total time: {total_time:.2f} seconds", file=fout)
            print(f"# Successfully processed samples: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)", file=fout)
            
            if success_count > 0:
                avg_rot_error = total_rot_error / success_count
                avg_trans_error = total_trans_error / success_count
                avg_total_error = total_total_error / success_count
                
                all_rot_errors.sort()
                all_trans_errors.sort()
                all_total_errors.sort()
                
                median_rot_error = all_rot_errors[len(all_rot_errors)//2] if all_rot_errors else 0
                median_trans_error = all_trans_errors[len(all_trans_errors)//2] if all_trans_errors else 0
                median_total_error = all_total_errors[len(all_total_errors)//2] if all_total_errors else 0
                
                if len(all_rot_errors) > 1:
                    std_rot_error = math.sqrt(sum((x - avg_rot_error)**2 for x in all_rot_errors) / (len(all_rot_errors) - 1))
                    std_trans_error = math.sqrt(sum((x - avg_trans_error)**2 for x in all_trans_errors) / (len(all_trans_errors) - 1))
                    std_total_error = math.sqrt(sum((x - avg_total_error)**2 for x in all_total_errors) / (len(all_total_errors) - 1))
                else:
                    std_rot_error = 0
                    std_trans_error = 0
                    std_total_error = 0
                
                print(f"#", file=fout)
                print(f"# Error Statistics (radians/meters):", file=fout)
                print(f"# Average rotation error: {avg_rot_error:.6f}", file=fout)
                print(f"# Average translation error: {avg_trans_error:.6f}", file=fout)
                print(f"# Average total error: {avg_total_error:.6f}", file=fout)
                print(f"#", file=fout)
                print(f"# Median errors:", file=fout)
                print(f"# Median rotation error: {median_rot_error:.6f}", file=fout)
                print(f"# Median translation error: {median_trans_error:.6f}", file=fout)
                print(f"# Median total error: {median_total_error:.6f}", file=fout)
                print(f"#", file=fout)
                print(f"# Standard deviation:", file=fout)
                print(f"# Rotation error std: {std_rot_error:.6f}", file=fout)
                print(f"# Translation error std: {std_trans_error:.6f}", file=fout)
                print(f"# Total error std: {std_total_error:.6f}", file=fout)
                print(f"#", file=fout)
                print(f"# Min/Max errors:", file=fout)
                print(f"# Min rotation error: {min(all_rot_errors):.6f}", file=fout)
                print(f"# Max rotation error: {max(all_rot_errors):.6f}", file=fout)
                print(f"# Min translation error: {min(all_trans_errors):.6f}", file=fout)
                print(f"# Max translation error: {max(all_trans_errors):.6f}", file=fout)
                print(f"# Min total error: {min(all_total_errors):.6f}", file=fout)
                print(f"# Max total error: {max(all_total_errors):.6f}", file=fout)
            else:
                print(f"# No successfully processed samples, cannot calculate error statistics", file=fout)
            
            if error_count > 0:
                print(f"# Registration failed samples: {error_count}/{success_count} ({error_count/success_count*100:.1f}%)", file=fout)
            
            print(f"# Completion time: {time.strftime('%Y-%m-%d %H:%M:%S')}", file=fout)
            print(f"# =====================================", file=fout)
        
        print(f"\n====== Test Completed ======")
        print(f"Total time: {total_time:.2f} seconds (average {total_time/total_count:.2f} seconds per sample)")
        print(f"Successfully processed samples: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        
        if success_count > 0:
            avg_rot_error = total_rot_error / success_count
            avg_trans_error = total_trans_error / success_count
            print(f"Average rotation error: {avg_rot_error:.6f}")
            print(f"Average translation error: {avg_trans_error:.6f}")
            
        if error_count > 0:
            print(f"Registration failed samples: {error_count}/{success_count} ({error_count/success_count*100:.1f}%)")
        print(f"Results saved to: {self.filename}")
        
        if need_visualization:
            print(f"\n====== Visualization Completed ======")
            print(f"Visualized {current_vis_count} samples for perturbation file {self.current_pert_file}")
            print(f"Visualization results saved to: {self.vis_subdir}")
            print(f"Visualization log: {self.vis_log_file}")
        
        return success_count, total_count

    def do_estimate(self, p0, p1, model, device):
        p0 = p0.to(device)
        p1 = p1.to(device)
        r = ptlk.pointlk.PointLK.do_forward(model, p0, p1, self.max_iter, self.xtol,
                                            self.p0_zero_mean, self.p1_zero_mean)
        est_g = model.g

        return est_g

    def test_metrics(self, rotations_gt, translation_gt, rotations_ab, translation_ab, filename):
        rotations_gt = np.concatenate(rotations_gt, axis=0).reshape(-1, 3)
        translation_gt = np.concatenate(translation_gt, axis=0).reshape(-1, 3)
        rotations_ab = np.concatenate(rotations_ab, axis=0).reshape(-1, 3)
        translation_ab = np.concatenate(translation_ab, axis=0).reshape(-1,3)

        rot_err = np.sqrt(np.mean((np.degrees(rotations_ab) - np.degrees(rotations_gt)) ** 2, axis=1))
        trans_err = np.sqrt(np.mean((translation_ab - translation_gt) ** 2, axis=1))

        suc_tab = np.zeros(11)
        
        rot_err_tab = np.arange(11) * 0.5
        trans_err_tab = np.arange(11) * 0.05
        
        err_count_tab = np.triu(np.ones((11, 11)))
        
        for i in range(rot_err.shape[0]):
            if rot_err[i] <= rot_err_tab[0] and trans_err[i] <= trans_err_tab[0]:
                suc_tab = suc_tab + err_count_tab[0]
            elif rot_err[i] <= rot_err_tab[1] and trans_err[i] <= trans_err_tab[1]:
                suc_tab = suc_tab + err_count_tab[1]
            elif rot_err[i] <= rot_err_tab[2] and trans_err[i] <= trans_err_tab[2]:
                suc_tab = suc_tab + err_count_tab[2]
            elif rot_err[i] <= rot_err_tab[3] and trans_err[i] <= trans_err_tab[3]:
                suc_tab = suc_tab + err_count_tab[3]
            elif rot_err[i] <= rot_err_tab[4] and trans_err[i] <= trans_err_tab[4]:
                suc_tab = suc_tab + err_count_tab[4]
            elif rot_err[i] <= rot_err_tab[5] and trans_err[i] <= trans_err_tab[5]:
                suc_tab = suc_tab + err_count_tab[5]
            elif rot_err[i] <= rot_err_tab[6] and trans_err[i] <= trans_err_tab[6]:
                suc_tab = suc_tab + err_count_tab[6]
            elif rot_err[i] <= rot_err_tab[7] and trans_err[i] <= trans_err_tab[7]:
                suc_tab = suc_tab + err_count_tab[7]
            elif rot_err[i] <= rot_err_tab[8] and trans_err[i] <= trans_err_tab[8]:
                suc_tab = suc_tab + err_count_tab[8]
            elif rot_err[i] <= rot_err_tab[9] and trans_err[i] <= trans_err_tab[9]:
                suc_tab = suc_tab + err_count_tab[9]
            elif rot_err[i] <= rot_err_tab[10] and trans_err[i] <= trans_err_tab[10]:
                suc_tab = suc_tab + err_count_tab[10]

        rot_mse_ab = np.mean((np.degrees(rotations_ab) - np.degrees(rotations_gt)) ** 2)
        rot_rmse_ab = np.sqrt(rot_mse_ab)
        rot_mae_ab = np.mean(np.abs(np.degrees(rotations_ab) - np.degrees(rotations_gt)))

        trans_mse_ab = np.mean((translation_ab - translation_gt) ** 2)
        trans_rmse_ab = np.sqrt(trans_mse_ab)
        trans_mae_ab = np.mean(np.abs(translation_ab - translation_gt))
        
        rot_mse_ab_02 = np.median((np.degrees(rotations_ab) - np.degrees(rotations_gt)) ** 2)
        rot_rmse_ab_02 = np.sqrt(rot_mse_ab_02)
        rot_mae_ab_02 = np.median(np.abs(np.degrees(rotations_ab) - np.degrees(rotations_gt)))
        
        trans_mse_ab_02 = np.median((translation_ab - translation_gt) ** 2)
        trans_rmse_ab_02 = np.sqrt(trans_mse_ab_02)
        trans_mae_ab_02 = np.median(np.abs(translation_ab - translation_gt))

        log_message = f'Source to Template:\n{filename}\n'
        log_message += '********************mean********************\n'
        log_message += f'rot_MSE: {rot_mse_ab}, rot_RMSE: {rot_rmse_ab}, rot_MAE: {rot_mae_ab}, trans_MSE: {trans_mse_ab}, trans_RMSE: {trans_rmse_ab}, trans_MAE: {trans_mae_ab}, rot_err: {np.mean(rot_err)}, trans_err: {np.mean(trans_err)}\n'
        log_message += '********************median********************\n'
        log_message += f'rot_MSE: {rot_mse_ab_02}, rot_RMSE: {rot_rmse_ab_02}, rot_MAE: {rot_mae_ab_02}, trans_MSE: {trans_mse_ab_02}, trans_RMSE: {trans_rmse_ab_02}, trans_MAE: {trans_mae_ab_02}\n'
        log_message += f'success cases are {suc_tab}\n'
        
        LOGGER.info(log_message)
        
        metrics_filename = f"{os.path.splitext(filename)[0]}_metrics.txt"
        with open(metrics_filename, 'w') as f:
            f.write(log_message)
        
        print(f"Test metrics saved to: {metrics_filename}")
        
        return


    def update_perturbation(self, perturbation_file, outfile):
        self.filename = outfile
        self.current_pert_file = os.path.basename(perturbation_file)
        
        if self.visualize_pert is not None and self.current_pert_file in self.visualize_pert:
            vis_dir = os.path.join(os.path.dirname(self.filename), 'visualize')
            os.makedirs(vis_dir, exist_ok=True)
            
            self.vis_subdir = os.path.join(vis_dir, os.path.splitext(self.current_pert_file)[0])
            os.makedirs(self.vis_subdir, exist_ok=True)
            
            self.vis_log_file = os.path.join(self.vis_subdir, 'visualization_log.txt')
            with open(self.vis_log_file, 'w') as f:
                f.write("# PointNetLK Registration Visualization Log\n")
                f.write("# Perturbation file: {}\n".format(self.current_pert_file))
                f.write("# Creation time: {}\n\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
                f.write("Point cloud pair,Perturbation file,Predicted(w1,w2,w3,v1,v2,v3),Ground truth perturbation(w1,w2,w3,v1,v2,v3)\n")
                f.write("--------------------------------------------------------------------\n")


def get_datasets(args):
    cinfo = None
    if args.categoryfile and os.path.exists(args.categoryfile):
        try:
            categories = [line.rstrip('\n') for line in open(args.categoryfile)]
            categories.sort()
            c_to_idx = {categories[i]: i for i in range(len(categories))}
            cinfo = (categories, c_to_idx)
        except Exception as e:
            LOGGER.warning(f"Failed to load category file: {e}")
            if args.dataset_type != 'c3vd':
                raise

    perturbations = None
    fmt_trans = False
    is_gt_poses_mode = False
    if args.perturbations:
        if not os.path.exists(args.perturbations):
            raise FileNotFoundError(f"{args.perturbations} not found.")
        perturbations = numpy.loadtxt(args.perturbations, delimiter=',')
        
        perturbation_filename = os.path.basename(args.perturbations)
        if perturbation_filename == 'gt_poses.csv' or 'gt_poses' in perturbation_filename:
            is_gt_poses_mode = True
            print(f"\n🎯 GT_POSES mode activated!")
            print(f"Perturbation file: {args.perturbations}")
            print(f"Number of perturbations: {len(perturbations)}")
            print(f"Test mode: Each perturbation selects a test sample randomly")
            print(f"Total test cases: {len(perturbations)} (equal to number of perturbations)")
        else:
            print(f"\n📋 Standard test mode")
            print(f"Perturbation file: {args.perturbations}")
            print(f"Number of perturbations: {len(perturbations)}")
            print(f"Test mode: Iterate over all test samples, using one perturbation per sample")
            
    if args.format == 'wt':
        fmt_trans = True

    if args.dataset_type == 'modelnet':
        transform = torchvision.transforms.Compose([
            ptlk.data.transforms.Mesh2Points(),
            ptlk.data.transforms.OnUnitCube(),
            ptlk.data.transforms.Resampler(args.num_points)
        ])

        testdata = ptlk.data.datasets.ModelNet(args.dataset_path, train=0, transform=transform, classinfo=cinfo)

        if is_gt_poses_mode:
            print(f"Using ModelNet random sampling mode...")
            testset = ptlk.data.datasets.CADset4tracking_fixed_perturbation_random_sample(
                testdata, perturbations, fmt_trans=fmt_trans, random_seed=42)
        else:
            testset = ptlk.data.datasets.CADset4tracking_fixed_perturbation(
                testdata, perturbations, fmt_trans=fmt_trans)
    
    elif args.dataset_type == 'c3vd':
        transform = torchvision.transforms.Compose([
            # No longer includes any point cloud processing, will be handled in C3VDset4tracking
        ])
        
        use_voxelization = getattr(args, 'use_voxelization', True)
        voxel_config = None
        if use_voxelization:
            voxel_config = ptlk.data.datasets.VoxelizationConfig(
                voxel_size=getattr(args, 'voxel_size', 0.05),
                voxel_grid_size=getattr(args, 'voxel_grid_size', 32),
                max_voxel_points=getattr(args, 'max_voxel_points', 100),
                max_voxels=getattr(args, 'max_voxels', 20000),
                min_voxel_points_ratio=getattr(args, 'min_voxel_points_ratio', 0.1)
            )
            print(f"\n====== Voxelization Configuration ======")
            print(f"Voxelization configuration: Voxel size={voxel_config.voxel_size}, Grid size={voxel_config.voxel_grid_size}")
            print(f"Maximum points per voxel: {voxel_config.max_voxel_points}, Maximum number of voxels: {voxel_config.max_voxels}")
            print(f"Minimum voxel points ratio: {voxel_config.min_voxel_points_ratio}")
        else:
            print(f"\n====== Sampling Configuration ======")
            print("Using simple resampling method")
        
        print(f"\n====== C3VD Dataset Configuration ======")
        print(f"Pairing mode: {args.pair_mode}")
        
        source_root = os.path.join(args.dataset_path, 'C3VD_ply_source')
        
        if args.pair_mode == 'scene_reference':
            if args.reference_name:
                print(f"Reference point cloud name: {args.reference_name}")
            else:
                print(f"Reference point cloud: First point cloud in each scene")
            target_path = os.path.join(args.dataset_path, 'C3VD_ref')
            print(f"Target point cloud directory: {target_path}")
        else:  # one_to_one mode
            target_path = os.path.join(args.dataset_path, 'visible_point_cloud_ply_depth')
            print(f"Target point cloud directory: {target_path}")
            print(f"Pairing method: Each source point cloud matches target point cloud with corresponding frame number")
        
        c3vd_dataset = ptlk.data.datasets.C3VDDataset(
            source_root=source_root,
            transform=transform,
            pair_mode=args.pair_mode,
            reference_name=args.reference_name
        )
        
        if len(c3vd_dataset.pairs) == 0:
            print(f"Error: No paired point clouds found, please check pairing mode and data paths")
            source_root = os.path.join(args.dataset_path, 'C3VD_ply_source')
            print(f"Source point cloud directory: {source_root}")
            print(f"Directory exists: {os.path.exists(source_root)}")
            if os.path.exists(source_root):
                scenes = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]
                print(f"Found scenes: {scenes}")
                if scenes:
                    for scene in scenes[:2]:  # Only show info for first two scenes
                        scene_dir = os.path.join(source_root, scene)
                        files = os.listdir(scene_dir)
                        print(f"Files in scene {scene}: {files[:5]}{'...' if len(files) > 5 else ''}")
            
            target_root = target_path
            print(f"Target point cloud directory: {target_root}")
            print(f"Directory exists: {os.path.exists(target_root)}")
            if os.path.exists(target_root):
                scenes = [d for d in os.listdir(target_root) if os.path.isdir(os.path.join(target_root, d))]
                print(f"Found scenes: {scenes}")
                if scenes:
                    for scene in scenes[:2]:  # Only show info for first two scenes
                        scene_dir = os.path.join(target_root, scene)
                        files = os.listdir(scene_dir)
                        print(f"Files in scene {scene}: {files[:5]}{'...' if len(files) > 5 else ''}")
            
            raise RuntimeError("Cannot find paired point clouds, please check dataset structure and pairing mode settings")
        
        print(f"\n====== Dataset Splitting ======")
        
        all_scenes = []
        source_root_for_split = os.path.join(args.dataset_path, 'C3VD_ply_source')
        for scene_dir in glob.glob(os.path.join(source_root_for_split, "*")):
            if os.path.isdir(scene_dir):
                all_scenes.append(os.path.basename(scene_dir))
        
        import random
        random.seed(42)
        test_scenes = random.sample(all_scenes, 4)
        train_scenes = [scene for scene in all_scenes if scene not in test_scenes]
        
        print(f"All scenes ({len(all_scenes)}): {sorted(all_scenes)}")
        print(f"Training scenes ({len(train_scenes)}): {sorted(train_scenes)}")
        print(f"Test scenes ({len(test_scenes)}): {sorted(test_scenes)}")
        
        test_indices = []
        
        for idx, (source_file, target_file) in enumerate(c3vd_dataset.pairs):
            scene_name = None
            for scene in all_scenes:
                if f"/{scene}/" in source_file:
                    scene_name = scene
                    break
            
            if scene_name in test_scenes:
                test_indices.append(idx)
        
        testdata = torch.utils.data.Subset(c3vd_dataset, test_indices)
        
        print(f"Total paired point clouds: {len(c3vd_dataset.pairs)}")
        print(f"Test set point cloud pairs (test scenes only): {len(testdata)}")
        
        if len(testdata) > 0:
            print(f"\nTest set scene verification:")
            sample_scenes = set()
            for i in range(min(10, len(testdata))):
                idx = testdata.indices[i]
                source_file, target_file = c3vd_dataset.pairs[idx]
                for scene in all_scenes:
                    if f"/{scene}/" in source_file:
                        sample_scenes.add(scene)
                        break
            print(f"Scenes in test set samples: {sorted(sample_scenes)}")
            print(f"Expected test scenes: {sorted(test_scenes)}")
            if sample_scenes.issubset(set(test_scenes)):
                print("✅ Scene splitting verification passed")
            else:
                print("❌ Warning: Scene splitting verification failed")
        else:
            print("❌ Error: No test samples found after scene splitting")
            raise RuntimeError("No test samples found after applying scene-based splitting")
        
        class SimpleRigidTransform:
            def __init__(self, perturbations_data, fmt_trans=False):
                self.perturbations = perturbations_data
                self.fmt_trans = fmt_trans
                self.igt = None
                self.current_perturbation_index = 0
            
            def __call__(self, tensor):
                try:
                    if self.perturbations is None or len(self.perturbations) == 0:
                        print("Warning: No perturbations, returning original point cloud")
                        return tensor
                    
                    if self.current_perturbation_index >= len(self.perturbations):
                        self.current_perturbation_index = 0
                    
                    twist = torch.from_numpy(numpy.array(self.perturbations[self.current_perturbation_index])).contiguous().view(1, 6)
                    self.current_perturbation_index += 1
                    
                    x = twist.to(tensor)
                    
                    if not self.fmt_trans:
                        g = ptlk.se3.exp(x).to(tensor)
                        p1 = ptlk.se3.transform(g, tensor)
                        self.igt = g.squeeze(0)
                    else:
                        w = x[:, 0:3]
                        q = x[:, 3:6]
                        R = ptlk.so3.exp(w).to(tensor)
                        g = torch.zeros(1, 4, 4)
                        g[:, 3, 3] = 1
                        g[:, 0:3, 0:3] = R
                        g[:, 0:3, 3] = q
                        p1 = ptlk.se3.transform(g, tensor)
                        self.igt = g.squeeze(0)
                    
                    return p1
                except Exception as e:
                    print(f"Error during rigid transform: {e}")
                    return tensor
        
        rigid_transform = SimpleRigidTransform(perturbations, fmt_trans)

        if is_gt_poses_mode:
            print(f"Using C3VD random sampling mode...")
            testset = ptlk.data.datasets.C3VDset4tracking_test_random_sample(
                testdata, rigid_transform, num_points=args.num_points,
                use_voxelization=use_voxelization, voxel_config=voxel_config, random_seed=42)
        else:
            print(f"Using C3D standard test mode...")
            testset = ptlk.data.datasets.C3VDset4tracking_test(
                testdata, rigid_transform, num_points=args.num_points,
                use_voxelization=use_voxelization, voxel_config=voxel_config)

    else:
        raise ValueError('Unsupported dataset type: {}'.format(args.dataset_type))

    if hasattr(args, 'max_samples') and args.max_samples is not None and args.max_samples > 0:
        if len(testset) > args.max_samples:
            print(f"Limiting test set sample count to {args.max_samples} (original sample count: {len(testset)})")
            testset = torch.utils.data.Subset(testset, range(args.max_samples))
            print(f"Sample count after limiting: {len(testset)}")
        else:
            print(f"Test set sample count {len(testset)} is less than or equal to maximum sample count limit {args.max_samples}, using all samples")
    else:
        print(f"Maximum sample count limit not set, or set to 0, using all test set samples: {len(testset)} samples")

    return testset


def process_single_pair(args):
    import time
    from ptlk.data.datasets import SinglePairTrackingDataset, VoxelizationConfig
    import ptlk.se3 as se3
    
    try:
        print(f"\n========== Starting single pair processing ==========")
        
        perturbation_values = [float(x.strip()) for x in args.single_perturbation.split(',')]
        if len(perturbation_values) != 6:
            raise ValueError(f"Perturbation values must be 6 numbers (rx,ry,rz,tx,ty,tz), provided {len(perturbation_values)}")
        
        print(f"Perturbation values: {perturbation_values}")
        
        voxel_config = VoxelizationConfig(
            voxel_size=args.voxel_size,
            voxel_grid_size=args.voxel_grid_size,
            max_voxel_points=args.max_voxel_points,
            max_voxels=args.max_voxels,
            min_voxel_points_ratio=args.min_voxel_points_ratio
        )
        
        print(f"\nCreating single pair point cloud tracking dataset...")
        testset = SinglePairTrackingDataset(
            source_cloud_path=args.source_cloud,
            target_cloud_path=args.target_cloud,
            perturbation=perturbation_values,
            num_points=args.num_points,
            use_voxelization=args.use_voxelization,
            voxel_config=voxel_config,
            fmt_trans=(args.format == 'wt')
        )
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=1, shuffle=False, num_workers=1
        )
        
        act = Action(args)
        
        print(f"\nLoading model...")
        model = act.create_model()
        device = torch.device(args.device)
        model = model.to(device)
        model.eval()
        
        print(f"Model type: {args.model_type}")
        print(f"Device: {device}")
        
        print(f"\nStarting prediction...")
        with torch.no_grad():
            for i, (template, source, igt) in enumerate(testloader):
                template = template.to(device)
                source = source.to(device)
                igt = igt.to(device)
                
                print(f"\nProcessing batch {i+1}/1:")
                print(f"  Template shape: {template.shape}")
                print(f"  Source point cloud shape: {source.shape}")
                print(f"  True transformation matrix shape: {igt.shape}")
                
                g_hat = act.do_estimate(template, source, model, device)
                
                print(f"  Predicted transformation matrix shape: {g_hat.shape}")
                
                print(f"\n========== Registration Results ==========")
                
                print(f"Input perturbation:")
                print(f"  Vector form: [{', '.join([f'{x:.6f}' for x in perturbation_values])}]")
                print(f"  Rotation part (rx,ry,rz): [{', '.join([f'{x:.6f}' for x in perturbation_values[:3]])}]")
                print(f"  Translation part (tx,ty,tz): [{', '.join([f'{x:.6f}' for x in perturbation_values[3:]])}]")
                
                print(f"\nPredicted transformation:")
                g_hat_np = g_hat.cpu().numpy().squeeze()
                print(f"  Transformation matrix:")
                for row in range(4):
                    print(f"    [{', '.join([f'{g_hat_np[row, col]:8.6f}' for col in range(4)])}]")
                
                predicted_twist = se3.log(g_hat).cpu().numpy().squeeze()
                print(f"  Twist vector form: [{', '.join([f'{x:.6f}' for x in predicted_twist])}]")
                print(f"  Rotation part: [{', '.join([f'{x:.6f}' for x in predicted_twist[:3]])}]")
                print(f"  Translation part: [{', '.join([f'{x:.6f}' for x in predicted_twist[3:]])}]")
                
                igt_np = igt.cpu().numpy().squeeze()
                g_hat_np = g_hat.cpu().numpy().squeeze()
                
                g_rel = np.linalg.inv(igt_np) @ g_hat_np
                
                R_rel = g_rel[:3, :3]
                trace_R = np.trace(R_rel)
                cos_angle = (trace_R - 1) / 2
                cos_angle = np.clip(cos_angle, -1, 1)
                rotation_error_rad = np.arccos(cos_angle)
                rotation_error_deg = np.degrees(rotation_error_rad)
                
                t_rel = g_rel[:3, 3]
                translation_error = np.linalg.norm(t_rel)
                
                print(f"\nRegistration error:")
                print(f"  Rotation error: {rotation_error_rad:.6f} radians = {rotation_error_deg:.6f} degrees")
                print(f"  Translation error: {translation_error:.6f}")
                
                if args.enhanced_output:
                    print(f"\nSaving enhanced output to file: {args.outfile}")
                    
                    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
                    
                    with open(args.outfile, 'w') as f:
                        f.write("# Single pair point cloud registration results\n")
                        f.write(f"# Source point cloud: {args.source_cloud}\n")
                        f.write(f"# Target point cloud: {args.target_cloud}\n")
                        f.write(f"# Model type: {args.model_type}\n")
                        f.write(f"# Processing time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write("\n")
                        
                        f.write("# Input perturbation\n")
                        f.write(f"input_perturbation_vector,{','.join([f'{x:.6f}' for x in perturbation_values])}\n")
                        f.write(f"input_rotation_part,{','.join([f'{x:.6f}' for x in perturbation_values[:3]])}\n")
                        f.write(f"input_translation_part,{','.join([f'{x:.6f}' for x in perturbation_values[3:]])}\n")
                        f.write("\n")
                        
                        f.write("# Predicted transformation\n")
                        f.write(f"predicted_twist_vector,{','.join([f'{x:.6f}' for x in predicted_twist])}\n")
                        f.write(f"predicted_rotation_part,{','.join([f'{x:.6f}' for x in predicted_twist[:3]])}\n")
                        f.write(f"predicted_translation_part,{','.join([f'{x:.6f}' for x in predicted_twist[3:]])}\n")
                        f.write("\n")
                        
                        f.write("# Predicted transformation matrix\n")
                        for row in range(4):
                            f.write(f"transformation_matrix_row_{row},{','.join([f'{g_hat_np[row, col]:.6f}' for col in range(4)])}\n")
                        f.write("\n")
                        
                        f.write("# Registration error\n")
                        f.write(f"rotation_error_rad,{rotation_error_rad:.6f}\n")
                        f.write(f"rotation_error_deg,{rotation_error_deg:.6f}\n")
                        f.write(f"translation_error,{translation_error:.6f}\n")
                
                print(f"\n========== Single pair processing completed ==========")
                break
                
    except Exception as e:
        print(f"Failed to process single pair: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    ARGS = options()

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s:%(name)s, %(asctime)s, %(message)s',
        filename=ARGS.logfile)
    LOGGER.debug('Testing (PID=%d), %s', os.getpid(), ARGS)

    main(ARGS)
    LOGGER.debug('done (PID=%d)', os.getpid())

#EOF