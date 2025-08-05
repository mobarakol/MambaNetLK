"""
    Example for training a tracker (PointNet-LK).

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
import gc
import copy
import glob
import random
import math
import traceback

# addpath('../')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import ptlk
from ptlk import attention_v1
# Import Mamba-related modules on demand to avoid importing missing modules when not in use
# from ptlk import mamba3d_v1  # Import Mamba3D module
# from ptlk import mamba3d_v2
# from ptlk import mamba3d_v3
# from ptlk import mamba3d_v4
from ptlk import fast_point_attention  # Import fast point attention module
from ptlk import cformer  # Import Cformer module
# Remove adversarial module imports
# from ptlk.adversarial import GradReverse, DomainDiscriminator # Import adversarial modules
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def options(argv=None):
    parser = argparse.ArgumentParser(description='PointNet-LK')

    # required.
    parser.add_argument('-o', '--outfile', required=True, type=str,
                        metavar='BASENAME', help='output filename (prefix)') # the result: ${BASENAME}_model_best.pth
    parser.add_argument('-i', '--dataset-path', required=True, type=str,
                        metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
    parser.add_argument('-c', '--categoryfile', required=True, type=str,
                        metavar='PATH', help='path to the categories to be trained') # eg. './sampledata/modelnet40_half1.txt'

    # settings for input data
    parser.add_argument('--dataset-type', default='modelnet', choices=['modelnet', 'shapenet2', 'c3vd'],
                        metavar='DATASET', help='dataset type (default: modelnet)')
    parser.add_argument('--num-points', default=1024, type=int,
                        metavar='N', help='points in point-cloud (default: 1024)')
    parser.add_argument('--mag', default=0.8, type=float,
                        metavar='T', help='max. mag. of twist-vectors (perturbations) on training (default: 0.8)')
    # C3VD pairing mode settings
    parser.add_argument('--pair-mode', default='one_to_one', 
                        choices=['one_to_one', 'scene_reference', 'source_to_source', 'target_to_target', 'all'],
                        help='Point cloud pairing mode: one_to_one (each source point cloud corresponds to a specific target point cloud), '
                             'scene_reference (each scene uses a shared target point cloud), '
                             'source_to_source (source point clouds paired with source point clouds), '
                             'target_to_target (target point clouds paired with target point clouds), '
                             'all (includes all pairing methods)')
    parser.add_argument('--reference-name', default=None, type=str,
                        help='Target point cloud name used in scene reference mode, defaults to the first point cloud in the scene')

    # settings for PointNet
    parser.add_argument('--pointnet', default='tune', type=str, choices=['fixed', 'tune'],
                        help='train pointnet (default: tune)')
    parser.add_argument('--use-tnet', dest='use_tnet', action='store_true',
                        help='flag for setting up PointNet with TNet')
    parser.add_argument('--dim-k', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')
    parser.add_argument('--symfn', default='max', choices=['max', 'avg', 'selective'],
                        help='symmetric function (default: max)')
    
    # Add model selection parameters (consistent with train_classifier.py)
    parser.add_argument('--model-type', default='pointnet', choices=['pointnet', 'attention', 'mamba3d', 'mamba3d_v2', 'fast_attention', 'cformer', 'mamba3d_v3', 'mamba3d_v4'],
                        help='Select model type: pointnet, attention, mamba3d, mamba3d_v2, fast_attention, cformer, mamba3d_v3, mamba3d_v4 (default: pointnet)')
    
    # Add attention model specific parameters (consistent with train_classifier.py)
    parser.add_argument('--num-attention-blocks', default=3, type=int,
                        metavar='N', help='Number of attention blocks in attention module (default: 3)')
    parser.add_argument('--num-heads', default=8, type=int,
                        metavar='N', help='Number of heads in multi-head attention (default: 8)')
    
    # Add Mamba3D model specific parameters
    parser.add_argument('--num-mamba-blocks', default=3, type=int,
                        metavar='N', help='Number of Mamba blocks in Mamba3D module (default: 3)')
    parser.add_argument('--d-state', default=16, type=int,
                        metavar='N', help='Mamba state space dimension (default: 16)')
    parser.add_argument('--expand', default=2, type=float,
                        metavar='N', help='Mamba expansion factor (default: 2)')
    
    # Add fast point attention model specific parameters
    parser.add_argument('--num-fast-attention-blocks', default=2, type=int,
                        metavar='N', help='Number of attention blocks in fast point attention module (default: 2)')
    parser.add_argument('--fast-attention-scale', default=1, type=int,
                        metavar='N', help='Scale factor for fast point attention model (default: 1, larger values mean lighter model)')
    
    # Add Cformer model specific parameters
    parser.add_argument('--num-proxy-points', default=8, type=int,
                        metavar='N', help='Number of proxy points in Cformer model (default: 8)')
    parser.add_argument('--num-blocks', default=2, type=int,
                        metavar='N', help='Number of blocks in Cformer model (default: 2)')
    
    parser.add_argument('--transfer-from', default='', type=str,
                        metavar='PATH', help='path to pointnet features file')

    # settings for LK
    parser.add_argument('--max-iter', default=10, type=int,
                        metavar='N', help='max-iter on LK. (default: 10)')
    parser.add_argument('--delta', default=1.0e-2, type=float,
                        metavar='D', help='step size for approx. Jacobian (default: 1.0e-2)')
    parser.add_argument('--learn-delta', dest='learn_delta', action='store_true',
                        help='flag for training step size delta')

    # settings for on training
    parser.add_argument('-l', '--logfile', default='', type=str,
                        metavar='LOGNAME', help='path to logfile (default: null (no logging))')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--epochs', default=200, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        metavar='METHOD', help='name of an optimizer (default: Adam)')
    parser.add_argument('--resume', default='', type=str,
                        metavar='PATH', help='path to latest checkpoint (default: null (no-use))')
    parser.add_argument('--pretrained', default='', type=str,
                        metavar='PATH', help='path to pretrained model file (default: null (no-use))')
    parser.add_argument('--device', default='cuda:0', type=str,
                        metavar='DEVICE', help='use CUDA if available')

    # Additional parameters
    parser.add_argument('--verbose', action='store_true',
                        help='Display detailed logs')
    parser.add_argument('--drop-last', action='store_true',
                        help='Drop the last incomplete batch')

    # Scene split parameter
    parser.add_argument('--scene-split', action='store_true',
                        help='Split train and validation sets by scene')
    
    # Voxelization related parameters
    parser.add_argument('--use-voxelization', action='store_true', default=True,
                        help='Enable voxelization preprocessing method (default: True)')
    parser.add_argument('--no-voxelization', dest='use_voxelization', action='store_false',
                        help='Disable voxelization and use simple resampling method')
    parser.add_argument('--voxel-size', default=0.05, type=float,
                        metavar='SIZE', help='Voxel size (default: 0.05)')
    parser.add_argument('--voxel-grid-size', default=32, type=int,
                        metavar='SIZE', help='Voxel grid size (default: 32)')
    parser.add_argument('--max-voxel-points', default=100, type=int,
                        metavar='N', help='Maximum number of points per voxel (default: 100)')
    parser.add_argument('--max-voxels', default=20000, type=int,
                        metavar='N', help='Maximum number of voxels (default: 20000)')
    parser.add_argument('--min-voxel-points-ratio', default=0.1, type=float,
                        metavar='RATIO', help='Minimum voxel points ratio threshold (default: 0.1)')
    
    # Add learning rate scheduling parameters (consistent with train_classifier.py)
    parser.add_argument('--base-lr', default=None, type=float,
                        help='Base learning rate, automatically set to optimizer initial learning rate')
    parser.add_argument('--warmup-epochs', default=5, type=int,
                        metavar='N', help='Number of learning rate warmup epochs (default: 5)')
    parser.add_argument('--cosine-annealing', action='store_true',
                        help='Use cosine annealing learning rate strategy')

    # Add global feature consistency loss weight parameter
    parser.add_argument('--global-consistency-weight', default=0.1, type=float,
                        metavar='W', help='Weight of global feature consistency loss (default: 0.1)')

    # Remove domain adversarial and geometric correspondence loss parameters
    # parser.add_argument('--adversarial-lambda', default=0.1, type=float,
    #                     metavar='L', help='Weight of domain adversarial loss (default: 0.1)')
    # parser.add_argument('--correspondence-lambda', default=0.05, type=float,
    #                     metavar='L', help='Weight of feature correspondence loss (default: 0.05)')

    args = parser.parse_args(argv)
    return args

# Remove helper function: compute Chamfer Distance in feature space
# def feature_chamfer_loss(feat_a, feat_b):
#     """
#     Compute Chamfer distance in feature space.
#     feat_a: [B, N, K]
#     feat_b: [B, M, K]
#     """
#     dist_matrix = torch.cdist(feat_a, feat_b, p=2)  # [B, N, M]
    
#     # For each point in feat_a, find the nearest point in feat_b
#     dist_a_to_b, _ = torch.min(dist_matrix, dim=2)
#     # For each point in feat_b, find the nearest point in feat_a
#     dist_b_to_a, _ = torch.min(dist_matrix, dim=1)
    
#     loss = torch.mean(dist_a_to_b) + torch.mean(dist_b_to_a)
#     return loss

def main(args):
    # dataset
    trainset, testset = get_datasets(args)

    # Reset logging configuration to ensure logs are correctly written to the specified file
    if args.logfile:
        # Completely reset the logging system
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        # Configure root logger
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(name)s, %(asctime)s, %(message)s',
            filename=args.logfile,
            filemode='w'  # Use 'w' mode to overwrite any existing log file
        )
        
        # Configure module-specific logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        
        print(f"Log will be written to: {args.logfile}")
    
    # training
    act = Action(args)
    run(args, trainset, testset, act)


def run(args, trainset, testset, action):
    # Custom dataset wrapper that handles exceptions
    class DatasetWrapper(torch.utils.data.Dataset):
        """Wrapper for safely loading dataset samples that might cause exceptions.
        
        This wrapper catches exceptions during sample loading and returns None instead,
        which will be filtered out by the custom collate function.
        """
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            try:
                return self.dataset[idx]
            except Exception as e:
                print(f"Warning: Error loading sample at index {idx}: {str(e)}")
                return None

    # Custom collate function to handle None values
    def custom_collate_fn(batch):
        """Custom collate function to filter out None values and check batch size"""
        # Remove None values
        batch = list(filter(lambda x: x is not None, batch))
        
        # Check if batch is empty
        if len(batch) == 0:
            raise ValueError("All samples in the batch are invalid")
            
        # Check if each element contains None values
        for item in batch:
            if None in item:
                print(f"Warning: Found None in batch item: {item}")
        
        # Use default collate function to process remaining samples
        return torch.utils.data.dataloader.default_collate(batch)
    
    # CUDA availability check
    print(f"\n====== CUDA Availability Check ======")
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"Number of devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        print(f"CUDA Available: No (training will run on CPU, which will be slow)")
        args.device = 'cpu'
    
    args.device = torch.device(args.device)
    print(f"Using device: {args.device}")
    

    # Dataset statistics
    print("\n====== Detailed Dataset Statistics ======")
    
    # Get original datasets (before wrapping)
    original_trainset = trainset.dataset if isinstance(trainset, DatasetWrapper) else trainset
    original_testset = testset.dataset if isinstance(testset, DatasetWrapper) else testset
    
    if hasattr(original_trainset, 'pairs') and hasattr(original_trainset, 'scenes'):
        print(f"Training set scenes: {len(original_trainset.scenes)}")
        print(f"Training set point cloud pairs: {len(original_trainset.pairs)}")
        print(f"Point cloud pairs distribution per scene:")
        scene_counts = {}
        for scene in original_trainset.pair_scenes:
            scene_counts[scene] = scene_counts.get(scene, 0) + 1
        for scene, count in scene_counts.items():
            print(f"  - {scene}: {count} point cloud pairs")
    
    # Validation set statistics
    if hasattr(original_testset, 'pairs') and hasattr(original_testset, 'scenes'):
        print(f"\nValidation set scenes: {len(original_testset.scenes)}")
        print(f"Validation set point cloud pairs: {len(original_testset.pairs)}")
    
    # Calculate expected batches
    total_samples = len(trainset)
    expected_batches = total_samples // args.batch_size
    if not args.drop_last and total_samples % args.batch_size != 0:
        expected_batches += 1
    
    print(f"\nBatch statistics:")
    print(f"Total samples: {total_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Drop last batch: {args.drop_last}")
    print(f"Expected number of batches: {expected_batches}")
    
    # Basic dataset information
    print(f"\n====== Dataset Information ======")
    print(f"Training set: {len(trainset)} samples, Test set: {len(testset)} samples")
    print(f"Batch size: {args.batch_size}, Points per cloud: {args.num_points}, Drop last batch: {args.drop_last}")
    
    # Model initialization and loading
    model = action.create_model()
    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
    model.to(args.device)
    
    # Confirm model is on correct device
    print(f"\n====== Model Information ======")
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Model parameters on CUDA: {next(model.parameters()).is_cuda}")
    if str(args.device) != 'cpu':
        print(f"Current GPU memory usage: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        print(f"Current GPU memory cached: {torch.cuda.memory_reserved()/1024**2:.1f}MB")

    checkpoint = None
    if args.resume:
        assert os.path.isfile(args.resume)
        print(f"🔄 Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')  # Load to CPU first to avoid device mismatch issues
        
        # Check checkpoint format
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            # Complete checkpoint format (including training state)
            args.start_epoch = checkpoint.get('epoch', 0)
            model.load_state_dict(checkpoint['model'])
            print(f"   - Checkpoint type: Complete training state")
            print(f"   - Resuming to epoch: {checkpoint.get('epoch', 0)}")
            print(f"   - Previous best loss: {checkpoint.get('min_loss', 'N/A')}")
            if 'best_epoch' in checkpoint:
                print(f"   - Previous best epoch: {checkpoint['best_epoch']}")
        else:
            # Model weights only format
            model.load_state_dict(checkpoint)
            args.start_epoch = 0  # Start from epoch 0 but use pretrained weights
            checkpoint = None  # Set to None so optimizer state won't be loaded later
            print(f"   - Checkpoint type: Model weights only")
            print(f"   - Will start training from epoch 0 (using pretrained model weights)")
            print(f"   - Note: Optimizer state will be reinitialized")

    # Wrap datasets
    trainset = DatasetWrapper(trainset)
    testset = DatasetWrapper(testset)
    
    # Data loaders
    print(f"\n====== Data Loaders ======")
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=1,  # Reduce number of workers
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=min(args.workers, 2),
        drop_last=args.drop_last,
        pin_memory=(str(args.device) != 'cpu'),
        collate_fn=custom_collate_fn
    )
    
    print(f"Training batches: {len(trainloader)}, Test batches: {len(testloader)}")
    
    
    # Optimizer
    best_val_loss = float('inf')  # Initialize best validation loss
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(learnable_params, lr=0.0001, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=0.001, momentum=0.9, weight_decay=1e-4)

    # Restore training state
    best_epoch = 0
    if checkpoint is not None:
        best_val_loss = checkpoint.get('min_loss', float('inf'))  # Load best validation loss
        best_epoch = checkpoint.get('best_epoch', 0)
        # Only restore when checkpoint contains optimizer state
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"✅ Successfully restored training state:")
            print(f"   - Current best validation loss: {best_val_loss}")
            print(f"   - Current best epoch: {best_epoch}")
            print(f"   - Optimizer state restored")
        else:
            print(f"⚠️  Partially restored training state:")
            print(f"   - Current best validation loss: {best_val_loss}")
            print(f"   - Current best epoch: {best_epoch}")
            print(f"   - Optimizer state will be reinitialized (using default learning rate)")
    
    # Use stronger learning rate scheduling strategy
    if args.epochs > 50:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, 
            threshold=0.01, min_lr=1e-6)

    # ========================
    # Debug mode: try processing a single batch
    # ========================
    print("\n====== Debug mode: Single batch test ======")
    try:
        print("Getting single batch data...")
        debug_batch = next(iter(trainloader))
        print(f"Batch data shapes: {[x.shape for x in debug_batch]}")
        
        print("\nTesting forward pass...")
        model.train()  # Set to training mode
        
        try:
            # Capture any errors when testing single batch
            with torch.autograd.detect_anomaly():
                loss, loss_g = action.compute_loss(model, debug_batch, args.device)
                print(f"Forward pass successful! loss={loss.item():.4f}, loss_g={loss_g.item():.4f}")
                
                print("\nTesting backward pass...")
                optimizer.zero_grad()
                loss.backward()
                print("Backward pass successful!")
                
                print("\nTesting parameter update...")
                optimizer.step()
                print("Parameter update successful!")
                
                print("\nSingle batch test all successful!")
        except Exception as e:
            print(f"Single batch test failed: {e}")
            traceback.print_exc()
    except Exception as e:
        print(f"Unable to get test batch: {e}")
        traceback.print_exc()
    
    print("\nContinue with full training? (Auto-continue in 10 seconds, press Ctrl+C to interrupt)")
    try:
        # Set 10-second pause for user to check output and decide whether to continue
        time.sleep(10)
    except KeyboardInterrupt:
        print("User interrupted training")
        return
        
    # Training
    print("\n====== Starting Training ======")
    LOGGER.debug('train, begin')
    
    # Add data loader test
    print("Testing data loader...")
    try:
        print("Trying to get first training batch...")
        test_iter = iter(trainloader)
        first_batch = next(test_iter)
        print(f"First batch loaded successfully, shapes: {[x.shape for x in first_batch]}")
        del test_iter, first_batch  # Clean up memory
    except Exception as e:
        print(f"Data loader test failed: {e}")
        traceback.print_exc()
        return
    
    total_start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = time.time()
        
        consecutive_nan = 0
        max_consecutive_nan = 20
        last_valid_state = None
        
        # Save state at the beginning of each epoch
        if epoch_start:
            last_valid_state = {
                'model': copy.deepcopy(model.state_dict()),
                'optimizer': copy.deepcopy(optimizer.state_dict())
            }
        
        running_loss, running_info = action.train_1(model, trainloader, optimizer, args.device)
        val_loss, val_info = action.eval_1(model, testloader, args.device)
        
        # Detect consecutive NaNs and recover
        if not isinstance(val_loss, torch.Tensor):
            # If val_loss is a Python float and not a tensor
            if not (isinstance(val_loss, float) and math.isfinite(val_loss)):
                consecutive_nan += 1
                if consecutive_nan >= max_consecutive_nan and last_valid_state is not None:
                    print(f"Warning: Detected {consecutive_nan} consecutive NaN batches, recovering to last valid state")
                    model.load_state_dict(last_valid_state['model'])
                    optimizer.load_state_dict(last_valid_state['optimizer'])
                    consecutive_nan = 0
        else:
            # If val_loss is a tensor
            if not torch.isfinite(val_loss).all():
                consecutive_nan += 1
                if consecutive_nan >= max_consecutive_nan and last_valid_state is not None:
                    print(f"Warning: Detected {consecutive_nan} consecutive NaN batches, recovering to last valid state")
                    model.load_state_dict(last_valid_state['model'])
                    optimizer.load_state_dict(last_valid_state['optimizer'])
                    consecutive_nan = 0
        
        # Update learning rate
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Model saving strategy based on lowest validation loss
        is_best = False
        if val_loss < best_val_loss:
            is_best = True
            best_val_loss = val_loss
            best_epoch = epoch + 1
            print(f"[Save] Found better model with validation loss: {val_loss:.4f}")

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        print(f"[Time] Epoch {epoch+1}: {epoch_time:.2f} sec | Loss: {running_loss:.4f} | Val Loss: {val_loss:.4f} | Val Geo Loss: {val_info.get('loss_g', 0):.4f}")
        print(f"[Info] Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}, Current LR: {current_lr:.6f}")
        
        # Modify log output, add best_epoch and current_lr
        LOGGER.info('epoch, %04d, %f, %f, %f, %f, %04d, %f, %f', 
                   epoch + 1, running_loss, val_loss, running_info.get('loss_g', -1), val_info.get('loss_g', -1), best_epoch, current_lr, best_val_loss)
        snap = {'epoch': epoch + 1,
                'model': model.state_dict(),
                'min_loss': best_val_loss,  # Use best validation loss
                'best_epoch': best_epoch,   # Save best epoch information
                'optimizer' : optimizer.state_dict(),}
        
        if is_best:
            save_checkpoint(snap, args.outfile, 'snap_best')
            save_checkpoint(model.state_dict(), args.outfile, 'model_best')
            print(f"[Save] Best model saved")

        # Clear cache after each epoch
        if str(args.device) != 'cpu':
            torch.cuda.empty_cache()
            gc.collect()
        
        # Display estimated remaining time
        elapsed = time.time() - total_start_time
        estimated_total = elapsed / (epoch + 1 - args.start_epoch) * (args.epochs - args.start_epoch)
        remaining = estimated_total - elapsed
        print(f"[Progress] {epoch+1}/{args.epochs} epochs | Elapsed: {elapsed/60:.1f} min | Remaining: {remaining/60:.1f} min")

    total_time = time.time() - total_start_time
    print(f"\n====== Training Complete ======")
    print(f"Total training time: {total_time/60:.2f} minutes ({total_time:.2f} seconds)")
    print(f"Average time per epoch: {total_time/(args.epochs-args.start_epoch):.2f} seconds")
    LOGGER.debug('train, end')

def save_checkpoint(state, filename, suffix):
    torch.save(state, '{}_{}.pth'.format(filename, suffix))


class Action:
    def __init__(self, args):
        # PointNet related parameters
        self.pointnet = args.pointnet # tune or fixed
        self.transfer_from = args.transfer_from
        self.dim_k = args.dim_k
        self.use_tnet = args.use_tnet
        
        # Add new attributes (consistent with train_classifier.py)
        self.model_type = args.model_type
        self.num_attention_blocks = args.num_attention_blocks
        self.num_heads = args.num_heads
        
        # Add Mamba3D attributes
        self.num_mamba_blocks = args.num_mamba_blocks
        self.d_state = args.d_state
        self.expand = args.expand
        
        # Add fast point attention attributes
        self.num_fast_attention_blocks = args.num_fast_attention_blocks
        self.fast_attention_scale = args.fast_attention_scale
        
        # Add Cformer attributes
        self.num_proxy_points = getattr(args, 'num_proxy_points', 8)
        self.num_blocks = getattr(args, 'num_blocks', 2)
        
        # Add global feature consistency loss weight
        self.global_consistency_weight = args.global_consistency_weight
        
        # Remove domain adversarial and geometric correspondence parameters
        # self.adversarial_lambda = args.adversarial_lambda
        # self.correspondence_lambda = args.correspondence_lambda
        # self.discriminator = None
        
        # Aggregation function settings (consistent with train_classifier.py)
        self.sym_fn = None
        if args.model_type == 'attention':
            # Set aggregation function for attention model
            if args.symfn == 'max':
                self.sym_fn = attention_v1.symfn_max
            elif args.symfn == 'avg':
                self.sym_fn = attention_v1.symfn_avg
            else:
                self.sym_fn = attention_v1.symfn_attention_pool  # Attention-specific aggregation
        elif args.model_type == 'mamba3d':
            # Dynamically import and set aggregation function for Mamba3D model
            try:
                from ptlk import mamba3d_v1
                if args.symfn == 'max':
                    self.sym_fn = mamba3d_v1.symfn_max
                elif args.symfn == 'avg':
                    self.sym_fn = mamba3d_v1.symfn_avg
                elif args.symfn == 'selective':
                    self.sym_fn = mamba3d_v1.symfn_selective
                else:
                    self.sym_fn = mamba3d_v1.symfn_max  # Default to max pooling
            except ImportError:
                print("Warning: Unable to import mamba3d_v1 module, will handle when creating model")
                self.sym_fn = None
        elif self.model_type == 'mamba3d_v2':
            # Dynamically import Mamba3D_v2 module to prevent import errors
            try:
                from ptlk import mamba3d_v2
                # Set aggregation function for Mamba3D_v2 model
                if args.symfn == 'max':
                    self.sym_fn = mamba3d_v2.symfn_max
                elif args.symfn == 'avg':
                    self.sym_fn = mamba3d_v2.symfn_avg
                elif args.symfn == 'selective':
                    self.sym_fn = mamba3d_v2.symfn_selective
                else:
                    self.sym_fn = mamba3d_v2.symfn_max  # Default to max pooling
            except ImportError:
                print("Warning: Unable to import mamba3d_v2 module, will handle when creating model")
                self.sym_fn = None
        elif self.model_type == 'mamba3d_v3':
            # Dynamically import Mamba3D_v3 module
            try:
                from ptlk import mamba3d_v3
                if args.symfn == 'max':
                    self.sym_fn = mamba3d_v3.symfn_max
                elif args.symfn == 'avg':
                    self.sym_fn = mamba3d_v3.symfn_avg
                elif args.symfn == 'selective':
                    self.sym_fn = mamba3d_v3.symfn_selective
                else:
                    self.sym_fn = mamba3d_v3.symfn_max
            except ImportError:
                print("Warning: Unable to import mamba3d_v3 module, will handle when creating model")
                self.sym_fn = None
        elif self.model_type == 'mamba3d_v4':
            # Dynamically import Mamba3D_v4 module
            try:
                from ptlk import mamba3d_v4
                if args.symfn == 'max':
                    self.sym_fn = mamba3d_v4.symfn_max
                elif args.symfn == 'avg':
                    self.sym_fn = mamba3d_v4.symfn_avg
                elif args.symfn == 'selective':
                    self.sym_fn = mamba3d_v4.symfn_selective
                else:
                    self.sym_fn = mamba3d_v4.symfn_max
            except ImportError:
                print("Warning: Unable to import mamba3d_v4 module, will handle when creating model")
                self.sym_fn = None
        elif args.model_type == 'fast_attention':
            # Set aggregation function for fast point attention model
            if args.symfn == 'max':
                self.sym_fn = fast_point_attention.symfn_max
            elif args.symfn == 'avg':
                self.sym_fn = fast_point_attention.symfn_avg
            elif args.symfn == 'selective':
                self.sym_fn = fast_point_attention.symfn_fast_attention_pool  # Fast attention specific aggregation
            else:
                self.sym_fn = fast_point_attention.symfn_max  # Default to max pooling
        elif args.model_type == 'cformer':
            # Set aggregation function for Cformer model
            if args.symfn == 'max':
                self.sym_fn = cformer.symfn_max
            elif args.symfn == 'avg':
                self.sym_fn = cformer.symfn_avg
            elif args.symfn == 'cd_pool':
                self.sym_fn = cformer.symfn_cd_pool
            else:
                self.sym_fn = cformer.symfn_max  # Default to max pooling
        else:
            # Set aggregation function for pointnet model
            if args.symfn == 'max':
                self.sym_fn = ptlk.pointnet.symfn_max
            elif args.symfn == 'avg':
                self.sym_fn = ptlk.pointnet.symfn_avg
                
        # LK parameters
        self.delta = args.delta
        self.learn_delta = args.learn_delta
        self.max_iter = args.max_iter
        self.xtol = 1.0e-7
        self.p0_zero_mean = True
        self.p1_zero_mean = True

        self._loss_type = 1 # see. self.compute_loss()

    def create_model(self):
        # Remove discriminator initialization
        # self.discriminator = DomainDiscriminator(input_dim=self.dim_k)

        if self.model_type == 'attention':
            # Create attention model
            ptnet = attention_v1.AttentionNet_features(
                dim_k=self.dim_k, 
                sym_fn=self.sym_fn,
                scale=1,
                num_attention_blocks=self.num_attention_blocks,
                num_heads=self.num_heads
            )
            # Support loading pretrained weights from attention classifier
            if self.transfer_from and os.path.isfile(self.transfer_from):
                try:
                    pretrained_dict = torch.load(self.transfer_from, map_location='cpu')
                    ptnet.load_state_dict(pretrained_dict)
                    print(f"Successfully loaded attention pretrained weights: {self.transfer_from}")
                except Exception as e:
                    print(f"Failed to load attention pretrained weights: {e}")
                    print("Continue using randomly initialized weights")
        elif self.model_type == 'mamba3d':
            # Dynamically import Mamba3D module
            try:
                from ptlk import mamba3d_v1
                # If sym_fn is None, reset it
                if self.sym_fn is None:
                    self.sym_fn = mamba3d_v1.symfn_max  # Default to max pooling
                # Create Mamba3D model
                ptnet = mamba3d_v1.Mamba3D_features(
                    dim_k=self.dim_k, 
                    sym_fn=self.sym_fn,
                    scale=1,
                    num_mamba_blocks=self.num_mamba_blocks,
                    d_state=self.d_state,
                    expand=self.expand
                )
                # Support loading pretrained weights from Mamba3D classifier
                if self.transfer_from and os.path.isfile(self.transfer_from):
                    try:
                        pretrained_dict = torch.load(self.transfer_from, map_location='cpu')
                        ptnet.load_state_dict(pretrained_dict)
                        print(f"Successfully loaded Mamba3D pretrained weights: {self.transfer_from}")
                    except Exception as e:
                        print(f"Failed to load Mamba3D pretrained weights: {e}")
                        print("Continue using randomly initialized weights")
            except ImportError as e:
                print(f"Error: Unable to import Mamba3D module: {e}")
                print("Please ensure mamba_ssm library is installed or use other model types")
                raise
        elif self.model_type == 'mamba3d_v2':
            # Dynamically import Mamba3D module
            try:
                from ptlk import mamba3d_v2
                # If sym_fn is None, reset it
                if self.sym_fn is None:
                    self.sym_fn = mamba3d_v2.symfn_max  # Default to max pooling
                # Create Mamba3D model
                ptnet = mamba3d_v2.Mamba3D_features(
                    dim_k=self.dim_k, 
                    sym_fn=self.sym_fn,
                    scale=1,
                    num_mamba_blocks=self.num_mamba_blocks,
                    d_state=self.d_state,
                    expand=self.expand
                )
                # Support loading pretrained weights from Mamba3D classifier
                if self.transfer_from and os.path.isfile(self.transfer_from):
                    try:
                        pretrained_dict = torch.load(self.transfer_from, map_location='cpu')
                        ptnet.load_state_dict(pretrained_dict)
                        print(f"Successfully loaded Mamba3D pretrained weights: {self.transfer_from}")
                    except Exception as e:
                        print(f"Failed to load Mamba3D pretrained weights: {e}")
                        print("Continue using randomly initialized weights")
            except ImportError as e:
                print(f"Error: Unable to import Mamba3D module: {e}")
                print("Please ensure mamba_ssm library is installed or use other model types")
                raise
        elif self.model_type == 'mamba3d_v3':
            # Dynamically import Mamba3D_v3 module
            try:
                from ptlk import mamba3d_v3
                # If sym_fn is None, reset it
                if self.sym_fn is None:
                    self.sym_fn = mamba3d_v3.symfn_max  # Default to max pooling
                # Create Mamba3D_v3 (SE-Net) model
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
            except ImportError as e:
                print(f"Error: Unable to import Mamba3D_v3 module: {e}")
                print("Please ensure mamba_ssm library is installed or use other model types")
                raise
        elif self.model_type == 'mamba3d_v4':
            # Dynamically import Mamba3D_v4 module
            try:
                from ptlk import mamba3d_v4
                # If sym_fn is None, reset it
                if self.sym_fn is None:
                    self.sym_fn = mamba3d_v4.symfn_max  # Default to max pooling
                # Create Mamba3D_v4 (CBAM) model
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
            except ImportError as e:
                print(f"Error: Unable to import Mamba3D_v4 module: {e}")
                print("Please ensure mamba_ssm library is installed or use other model types")
                raise
        elif self.model_type == 'fast_attention':
            # Create fast point attention model
            ptnet = fast_point_attention.FastPointAttention_features(
                dim_k=self.dim_k, 
                sym_fn=self.sym_fn,
                scale=self.fast_attention_scale,
                num_attention_blocks=self.num_fast_attention_blocks
            )
            # Support loading pretrained weights from fast point attention classifier
            if self.transfer_from and os.path.isfile(self.transfer_from):
                try:
                    pretrained_dict = torch.load(self.transfer_from, map_location='cpu')
                    ptnet.load_state_dict(pretrained_dict)
                    print(f"Successfully loaded fast point attention pretrained weights: {self.transfer_from}")
                except Exception as e:
                    print(f"Failed to load fast point attention pretrained weights: {e}")
                    print("Continue using randomly initialized weights")
        elif self.model_type == 'cformer':
            # Create Cformer model
            ptnet = cformer.CFormer_features(
                dim_k=self.dim_k, 
                sym_fn=self.sym_fn,
                scale=1,
                num_proxy_points=self.num_proxy_points,
                num_blocks=self.num_blocks
            )
            # Support loading pretrained weights from Cformer classifier
            if self.transfer_from and os.path.isfile(self.transfer_from):
                try:
                    pretrained_dict = torch.load(self.transfer_from, map_location='cpu')
                    ptnet.load_state_dict(pretrained_dict)
                    print(f"Successfully loaded Cformer pretrained weights: {self.transfer_from}")
                except Exception as e:
                    print(f"Failed to load Cformer pretrained weights: {e}")
                    print("Continue using randomly initialized weights")
        else:
            # Create original pointnet model
            ptnet = ptlk.pointnet.PointNet_features(self.dim_k, sym_fn=self.sym_fn)
            if self.transfer_from and os.path.isfile(self.transfer_from):
                try:
                    pretrained_dict = torch.load(self.transfer_from, map_location='cpu')
                    ptnet.load_state_dict(pretrained_dict)
                    print(f"Successfully loaded PointNet pretrained weights: {self.transfer_from}")
                except Exception as e:
                    print(f"Failed to load PointNet pretrained weights: {e}")
                    print("Continue using randomly initialized weights")
        
        if self.pointnet == 'tune':
            pass
        elif self.pointnet == 'fixed':
            for param in ptnet.parameters():
                param.requires_grad_(False)
                
        return ptlk.pointlk.PointLK(ptnet, self.delta, self.learn_delta)

    def train_1(self, model, trainloader, optimizer, device):
        model.train()
        # Remove discriminator related code
        # self.discriminator.to(device) # Ensure discriminator is on the correct device
        # self.discriminator.train()

        vloss = 0.0
        gloss = 0.0
        # Remove correspondence and domain losses
        # closs = 0.0 # Correspondence Loss
        # dloss = 0.0 # Domain Loss
        count = 0
        nan_batch_count = 0
        nan_loss_count = 0
        nan_grad_count = 0
        
        batch_times = []
        data_times = []
        forward_times = []
        backward_times = []
        
        print("=========== Training loop started ===========")
        print("Total number of batches: {}".format(len(trainloader)))
        print("Device: {}".format(device))
        print("Current CUDA memory: {:.1f}MB".format(torch.cuda.memory_allocated()/1024**2 if str(device) != 'cpu' else 0))
        
        batch_start = time.time()
        
        for i, data in enumerate(trainloader):
            print("\n----- Starting to process batch {}/{} -----".format(i+1, len(trainloader)))
            data_time = time.time() - batch_start
            data_times.append(data_time)
            print("Data loading time: {:.4f} seconds".format(data_time))
            
            # Check data integrity
            if len(data) != 3:
                print("Warning: Batch data incomplete, should have 3 elements, actually has {}".format(len(data)))
                batch_start = time.time()
                continue
                
            print("Data shapes: {}".format([x.shape for x in data]))
            print("Check if data contains NaN: p0={}, p1={}, igt={}".format(
                torch.isnan(data[0]).any(), torch.isnan(data[1]).any(), torch.isnan(data[2]).any()))
            
            # Forward pass
            print("Starting forward pass...")
            forward_start = time.time()
            
            try:
                loss, loss_g = self.compute_loss(model, data, device)

                print("Loss calculation completed: loss={:.4f}, loss_g={:.4f}".format(
                    loss.item(), loss_g.item()))

            except Exception as e:
                print("Forward pass or loss calculation error: {}".format(e))
                traceback.print_exc()
                nan_batch_count += 1
                batch_start = time.time()
                continue
            
            # Check if loss is NaN, if so skip this batch
            if not torch.isfinite(loss) or not torch.isfinite(loss_g):
                print(f"Warning: Batch {i} loss values are non-finite {loss.item() if torch.isfinite(loss) else 'NaN'}/{loss_g.item() if torch.isfinite(loss_g) else 'NaN'}, skipping")
                nan_loss_count += 1
                nan_batch_count += 1
                batch_start = time.time()
                continue
                
            if str(device) != 'cpu':
                torch.cuda.synchronize()
            forward_time = time.time() - forward_start
            forward_times.append(forward_time)
            print("Forward pass time: {:.4f} seconds".format(forward_time))
            
            # Backward pass
            print("Starting backward pass...")
            backward_start = time.time()
            
            try:
                optimizer.zero_grad()
                print("Gradients cleared")
                
                loss.backward()
                print("Backward pass completed")
                
                # Stronger gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                print("Gradient clipping completed")
            except Exception as e:
                print("Backward pass error: {}".format(e))
                traceback.print_exc()
                nan_batch_count += 1
                batch_start = time.time()
                continue
            
            # Check if gradients contain NaN or Inf values
            do_step = True
            grad_check_start = time.time()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if (1 - torch.isfinite(param.grad).long()).sum() > 0:
                        do_step = False
                        print(f"Warning: Parameter {name} gradients contain NaN/Inf values")
                        nan_grad_count += 1
                        break
            print("Gradient check time: {:.4f} seconds".format(time.time() - grad_check_start))
            
            # Only update parameters when gradients are normal
            if do_step:
                print("Updating parameters...")
                try:
                    optimizer.step()
                    print("Parameter update completed")
                except Exception as e:
                    print("Parameter update error: {}".format(e))
                    traceback.print_exc()
                    nan_batch_count += 1
                    batch_start = time.time()
                    continue
            else:
                # If gradients are abnormal, don't include in average loss
                print("Skipping parameter update due to gradient issues")
                nan_batch_count += 1
                batch_start = time.time()
                continue
                
            if str(device) != 'cpu':
                torch.cuda.synchronize()
                print("Current CUDA memory: {:.1f}MB".format(torch.cuda.memory_allocated()/1024**2))
                
            backward_time = time.time() - backward_start
            backward_times.append(backward_time)
            print("Backward pass time: {:.4f} seconds".format(backward_time))

            # Only normal batches count toward total loss and count
            vloss += loss.item()
            gloss += loss_g.item()
            # Remove correspondence and domain loss accumulation
            # closs += loss_corr.item()
            # dloss += loss_domain.item()
            count += 1
            
            # Record total batch time
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            print(f"----- Batch {i+1}/{len(trainloader)} completed | Loss: {loss.item():.4f} | Time: {batch_time:.4f} seconds -----")
            
            # Display progress every 5 batches
            if i % 5 == 0:
                if str(device) != 'cpu':
                    mem_used = torch.cuda.memory_allocated()/1024**2
                    mem_total = torch.cuda.get_device_properties(0).total_memory/1024**2
                    print(f"Batch {i+1}/{len(trainloader)} | Loss: {loss.item():.4f} | GPU Memory: {mem_used:.1f}/{mem_total:.1f}MB | Time: {batch_time:.4f} seconds")
                else:
                    print(f"Batch {i+1}/{len(trainloader)} | Loss: {loss.item():.4f} | Time: {batch_time:.4f} seconds")
            
            batch_start = time.time()
            
            # Add checkpoint saving every 10 batches for debugging purposes
            if i > 0 and i % 10 == 0:
                print(f"Saving intermediate checkpoint (batch {i+1}/{len(trainloader)})")
                temp_checkpoint = {
                    'epoch': 0,  # First epoch
                    'batch': i,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(temp_checkpoint, '/SAN/medic/MRpcr/results/mamba_v3_se_c3vd/mamba3d_pointlk_temp_checkpoint.pth')

        # Display NaN batch statistics
        if nan_batch_count > 0:
            print(f"\nWarning: {nan_batch_count} batches were skipped ({nan_batch_count/len(trainloader)*100:.2f}%)")
            print(f"- NaN loss batches: {nan_loss_count}")
            print(f"- NaN gradient batches: {nan_grad_count}")
            
        # Safely calculate averages
        ave_vloss = float(vloss)/count if count > 0 else float('inf')
        ave_gloss = float(gloss)/count if count > 0 else float('inf')
        # Remove correspondence and domain loss average calculation
        # ave_closs = float(closs)/count if count > 0 else float('inf')
        # ave_dloss = float(dloss)/count if count > 0 else float('inf')
        
        # Calculate average times
        avg_batch = sum(batch_times) / len(batch_times) if batch_times else 0
        avg_data = sum(data_times) / len(data_times) if data_times else 0
        avg_forward = sum(forward_times) / len(forward_times) if forward_times else 0
        avg_backward = sum(backward_times) / len(backward_times) if backward_times else 0
        
        print(f"\nPerformance statistics:")
        print(f"Valid batches: {count}/{len(trainloader)} ({count/len(trainloader)*100:.2f}%)")
        print(f"Average batch time: {avg_batch:.4f} seconds = Data loading: {avg_data:.4f} seconds + Forward pass: {avg_forward:.4f} seconds + Backward pass: {avg_backward:.4f} seconds")
        print(f"Training results: Total loss={ave_vloss:.4f}, Geometric loss={ave_gloss:.4f}")
        
        # Remove correspondence and domain loss return
        return ave_vloss, {'loss_g': ave_gloss}

    def eval_1(self, model, testloader, device):
        model.eval()
        # Remove discriminator related code
        # self.discriminator.to(device)
        # self.discriminator.eval()

        vloss = 0.0
        gloss = 0.0
        # Remove correspondence and domain losses
        # closs = 0.0 # Correspondence Loss
        # dloss = 0.0 # Domain Loss
        count = 0
        nan_count = 0
        
        print("\n====== Starting Validation ======")
        
        with torch.no_grad():
            for i, data in enumerate(testloader):
                try:
                    loss, loss_g = self.compute_loss(model, data, device)
                    
                    # Skip NaN losses
                    if not torch.isfinite(loss) or not torch.isfinite(loss_g):
                        print(f"Validation batch {i}: Loss has non-finite values, skipping")
                        nan_count += 1
                        continue

                    vloss += loss.item()
                    gloss += loss_g.item()
                    # Remove correspondence and domain loss accumulation
                    # closs += loss_corr.item()
                    # dloss += loss_domain.item()
                    count += 1
                    
                    # Display progress every 10 batches
                    if i % 10 == 0:
                        print(f"Validation batch {i}/{len(testloader)} | Loss: {loss.item():.4f}")
                        
                except Exception as e:
                    print(f"Error processing validation batch {i}: {e}")
                    nan_count += 1
                    continue

        # Safely calculate averages
        if count > 0:
            ave_vloss = float(vloss)/count
            ave_gloss = float(gloss)/count
            # Remove correspondence and domain loss average calculation
            # ave_closs = float(closs)/count
            # ave_dloss = float(dloss)/count
        else:
            print("Warning: All validation batches failed!")
            ave_vloss = 1e5  # Use a large value instead of inf
            ave_gloss = 1e5
            # Remove correspondence and domain loss default values
            # ave_closs = 1e5
            # ave_dloss = 1e5
        
        print(f"\nValidation statistics:")
        print(f"Valid batches: {count}/{len(testloader)} ({count/len(testloader)*100:.2f}%)")
        print(f"Validation results: Total loss={ave_vloss:.4f}, Geometric loss={ave_gloss:.4f}")
        
        if nan_count > 0:
            print(f"Evaluation: {nan_count} batches had NaN values ({nan_count/len(testloader)*100:.2f}%)")
            
        # Remove correspondence and domain loss return
        return ave_vloss, {'loss_g': ave_gloss}

    def compute_loss(self, model, data, device):
        """
        compute_loss dual loss version + global feature constraints
        Uses loss_r (feature residual loss) + loss_g (geometric transformation loss) + global feature consistency loss
        """
        print("====== Starting dual loss + global feature constraint calculation ======")
        # p0: template, p1: source
        p0, p1, igt = data
        p0, p1, igt = p0.to(device), p1.to(device), igt.to(device)
        
        # Call PointLK core forward propagation function
        r = ptlk.pointlk.PointLK.do_forward(model, p0, p1, self.max_iter, self.xtol,
                                             self.p0_zero_mean, self.p1_zero_mean)
        
        # Get final estimated transformation matrix
        est_g = model.g
        
        # Calculate geometric loss (loss_g)
        loss_g = ptlk.pointlk.PointLK.comp(est_g, igt)
        
        # Calculate feature residual loss (loss_r)
        loss_r = ptlk.pointlk.PointLK.rsq(r)
        
        # Add global feature consistency loss
        # Get global features of p0 and transformed p1
        p1_transformed = ptlk.se3.transform(est_g.unsqueeze(1), p1)  # Use estimated transformation
        
        # Extract global features
        f0_out = model.ptnet(p0)
        f0 = f0_out[0] if isinstance(f0_out, tuple) else f0_out
        
        f1_out = model.ptnet(p1_transformed)
        f1 = f1_out[0] if isinstance(f1_out, tuple) else f1_out
        
        # Global feature consistency loss (MSE)
        loss_global_consistency = torch.nn.functional.mse_loss(f0, f1)
        
        # Combined loss: loss_r + loss_g + 0.1 * loss_global_consistency
        global_consistency_weight = self.global_consistency_weight # Get weight from args
        loss = loss_r + loss_g + global_consistency_weight * loss_global_consistency
        
        print(f"loss_r: {loss_r.item():.4f}, loss_g: {loss_g.item():.4f}, loss_global_consistency: {loss_global_consistency.item():.4f}")
        print("====== Loss calculation completed ======")

        # Return total loss and geometric loss
        return loss, loss_g


class ShapeNet2_transform_coordinate:
    def __init__(self):
        pass
    def __call__(self, mesh):
        return mesh.clone().rot_x()

def get_datasets(args):

    cinfo = None
    if args.categoryfile:
        #categories = numpy.loadtxt(args.categoryfile, dtype=str, delimiter="\n").tolist()
        categories = [line.rstrip('\n') for line in open(args.categoryfile)]
        categories.sort()
        c_to_idx = {categories[i]: i for i in range(len(categories))}
        cinfo = (categories, c_to_idx)

    if args.dataset_type == 'modelnet':
        transform = torchvision.transforms.Compose([\
                ptlk.data.transforms.Mesh2Points(),\
                ptlk.data.transforms.OnUnitCube(),\
                ptlk.data.transforms.Resampler(args.num_points),\
            ])

        traindata = ptlk.data.datasets.ModelNet(args.dataset_path, train=1, transform=transform, classinfo=cinfo)
        testdata = ptlk.data.datasets.ModelNet(args.dataset_path, train=0, transform=transform, classinfo=cinfo)

        mag_randomly = True
        trainset = ptlk.data.datasets.CADset4tracking(traindata,\
                        ptlk.data.transforms.RandomTransformSE3(args.mag, mag_randomly))
        testset = ptlk.data.datasets.CADset4tracking(testdata,\
                        ptlk.data.transforms.RandomTransformSE3(args.mag, mag_randomly))

    elif args.dataset_type == 'shapenet2':
        transform = torchvision.transforms.Compose([\
                ShapeNet2_transform_coordinate(),\
                ptlk.data.transforms.Mesh2Points(),\
                ptlk.data.transforms.OnUnitCube(),\
                ptlk.data.transforms.Resampler(args.num_points),\
            ])

        dataset = ptlk.data.datasets.ShapeNet2(args.dataset_path, transform=transform, classinfo=cinfo)
        traindata, testdata = dataset.split(0.8)

        mag_randomly = True
        trainset = ptlk.data.datasets.CADset4tracking(traindata,\
                        ptlk.data.transforms.RandomTransformSE3(args.mag, mag_randomly))
        testset = ptlk.data.datasets.CADset4tracking(testdata,\
                        ptlk.data.transforms.RandomTransformSE3(args.mag, mag_randomly))

    elif args.dataset_type == 'c3vd':
        # Remove transform since C3VD no longer needs it
        # transform = torchvision.transforms.Compose([
        #     # Remove normalization since C3VD no longer needs it
        #     # ptlk.data.transforms.OnUnitCube(),
        # ])
        
        # Configure voxelization parameters
        use_voxelization = getattr(args, 'use_voxelization', True)
        voxel_config = None
        if use_voxelization:
            # Create voxelization configuration
            voxel_config = ptlk.data.datasets.VoxelizationConfig(
                voxel_size=getattr(args, 'voxel_size', 1),
                voxel_grid_size=getattr(args, 'voxel_grid_size', 32),
                max_voxel_points=getattr(args, 'max_voxel_points', 100),
                max_voxels=getattr(args, 'max_voxels', 20000),
                min_voxel_points_ratio=getattr(args, 'min_voxel_points_ratio', 0.1)
            )
            print(f"Voxelization config: voxel size={voxel_config.voxel_size}, grid size={voxel_config.voxel_grid_size}")
        else:
            print("Using simple resampling method")
        
        # Create C3VD dataset
        c3vd_dataset = ptlk.data.datasets.C3VDDataset(
            source_root=os.path.join(args.dataset_path, 'C3VD_ply_source'),
            target_root=os.path.join(args.dataset_path, 'visible_point_cloud_ply_depth'),
            transform=None, # Remove transform
            pair_mode=getattr(args, 'pair_mode', 'one_to_one'),
            reference_name=getattr(args, 'reference_name', None)
        )
        
        # Split based on scene or randomly
        if args.scene_split:
            # Get all scenes
            all_scenes = []
            source_root = os.path.join(args.dataset_path, 'C3VD_ply_source')
            for scene_dir in glob.glob(os.path.join(source_root, "*")):
                if os.path.isdir(scene_dir):
                    all_scenes.append(os.path.basename(scene_dir))
            
            # Randomly select 4 scenes for validation (fixed random seed to ensure consistency with classifier)
            random.seed(42)
            test_scenes = random.sample(all_scenes, 4)
            train_scenes = [scene for scene in all_scenes if scene not in test_scenes]
            
            print(f"Training scenes ({len(train_scenes)}): {train_scenes}")
            print(f"Validation scenes ({len(test_scenes)}): {test_scenes}")
            
            # Split data by scene
            train_indices = []
            test_indices = []
            
            for idx, (source_file, target_file) in enumerate(c3vd_dataset.pairs):
                # Extract scene name
                scene_name = None
                for scene in all_scenes:
                    if f"/{scene}/" in source_file:
                        scene_name = scene
                        break
                
                if scene_name in train_scenes:
                    train_indices.append(idx)
                elif scene_name in test_scenes:
                    test_indices.append(idx)
            
            # Create subsets
            traindata = torch.utils.data.Subset(c3vd_dataset, train_indices)
            testdata = torch.utils.data.Subset(c3vd_dataset, test_indices)
            
            print(f"Scene-based split: Training samples: {len(traindata)}, Validation samples: {len(testdata)}")
        else:
            # Original random split method
            dataset_size = len(c3vd_dataset)
            train_size = int(dataset_size * 0.8)
            test_size = dataset_size - train_size
            traindata, testdata = torch.utils.data.random_split(c3vd_dataset, [train_size, test_size])
            print(f"Random split: Training samples: {len(traindata)}, Validation samples: {len(testdata)}")
        
        # Create tracking datasets for training and testing with voxelization support
        mag_randomly = True
        trainset = ptlk.data.datasets.C3VDset4tracking(
            traindata, 
            ptlk.data.transforms.RandomTransformSE3(args.mag, mag_randomly),
            num_points=args.num_points,
            use_voxelization=use_voxelization,
            voxel_config=voxel_config
        )
        testset = ptlk.data.datasets.C3VDset4tracking(
            testdata, 
            ptlk.data.transforms.RandomTransformSE3(args.mag, mag_randomly),
            num_points=args.num_points,
            use_voxelization=use_voxelization,
            voxel_config=voxel_config
        )
    
    return trainset, testset


if __name__ == '__main__':
    ARGS = options()

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s:%(name)s, %(asctime)s, %(message)s',
        filename=ARGS.logfile)
    LOGGER.debug('Training (PID=%d), %s', os.getpid(), ARGS)

    main(ARGS)
    LOGGER.debug('done (PID=%d)', os.getpid())

#EOF