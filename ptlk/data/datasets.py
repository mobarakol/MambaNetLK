"""
    Dataset classes for point cloud processing and tracking.
"""

import os
import sys
import glob
import numpy as np
import torch
import torch.utils.data
from plyfile import PlyData, PlyElement
import random
import math
from scipy.spatial.transform import Rotation
import warnings
import copy

import ptlk.data.globset as globset
import ptlk.data.transforms as transforms

# Import Lie group operations
try:
    import ptlk.se3 as se3
    import ptlk.so3 as so3
    SE3_AVAILABLE = True
except ImportError:
    SE3_AVAILABLE = False
    print("Warning: se3/so3 modules not available")

# Optional imports for advanced voxelization
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    o3d = None

# Voxelization configuration class
class VoxelizationConfig:
    """Configuration for voxelization parameters"""
    def __init__(self, voxel_size=0.01, min_points_per_voxel=1, use_open3d=True):
        self.voxel_size = voxel_size
        self.min_points_per_voxel = min_points_per_voxel
        self.use_open3d = use_open3d and OPEN3D_AVAILABLE

def voxelize_point_clouds(source_points, target_points, num_points, voxel_config=None, fallback_to_sampling=True):
    """
    Voxelize point clouds using Open3D or fallback to numpy-based voxelization
    
    Args:
        source_points: numpy array of source points [N, 3]
        target_points: numpy array of target points [M, 3]
        num_points: target number of points after voxelization
        voxel_config: VoxelizationConfig object
        fallback_to_sampling: if True, fallback to random sampling when voxelization fails
        
    Returns:
        tuple: (voxelized_source, voxelized_target)
    """
    if voxel_config is None:
        voxel_config = VoxelizationConfig()
    
    def voxelize_single_cloud(points, num_points, config):
        if config.use_open3d and OPEN3D_AVAILABLE:
            try:
                # Use Open3D for voxelization
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                
                # Voxel downsampling
                downsampled = pcd.voxel_down_sample(voxel_size=config.voxel_size)
                voxelized_points = np.asarray(downsampled.points)
                
                # Resample to target number of points
                if len(voxelized_points) >= num_points:
                    indices = np.random.choice(len(voxelized_points), num_points, replace=False)
                    return voxelized_points[indices]
                else:
                    # Pad with random selection from original points
                    remaining = num_points - len(voxelized_points)
                    additional_indices = np.random.choice(len(points), remaining, replace=True)
                    return np.vstack([voxelized_points, points[additional_indices]])
                    
            except Exception as e:
                if not fallback_to_sampling:
                    raise e
                # Fallback to numpy-based voxelization
                pass
        
        # Use numpy-based voxelization as fallback
        return voxel_down_sample_numpy(points, num_points)
    
    try:
        voxelized_source = voxelize_single_cloud(source_points, num_points, voxel_config)
        voxelized_target = voxelize_single_cloud(target_points, num_points, voxel_config)
        return voxelized_source, voxelized_target
    except Exception as e:
        if fallback_to_sampling:
            # Ultimate fallback: random sampling
            source_indices = np.random.choice(len(source_points), num_points, replace=len(source_points) < num_points)
            target_indices = np.random.choice(len(target_points), num_points, replace=len(target_points) < num_points)
            return source_points[source_indices], target_points[target_indices]
        else:
            raise e


def voxel_down_sample_numpy(points, num_points):
    """
    Voxel downsampling for point clouds using pure numpy
    """
    # Calculate voxel grid
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
        
    # Calculate voxel grid size
    n_side = int(np.ceil(num_points ** (1/3)))
        
    # Prevent zero range
    extents = max_coords - min_coords
    extents[extents == 0] = 1e-6
    
    # Calculate voxel size
    voxel_size = extents / n_side
    
    # Calculate voxel indices
    coords = np.floor((points - min_coords) / voxel_size).astype(np.int64)
    coords = np.minimum(coords, n_side - 1)
    
    # Map 3D voxel indices to a 1D key
    keys = coords[:,0] + coords[:,1] * n_side + coords[:,2] * (n_side ** 2)
    
    # Get unique voxels and mapping
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    
    # Calculate centroid of each voxel
    counts = np.bincount(inverse)
    sums = np.zeros((unique_keys.shape[0], 3), dtype=np.float64)
    
    for i in range(3):
        sums[:, i] = np.bincount(inverse, weights=points[:, i])
    
    centroids = sums / counts[:, np.newaxis]
    
    if len(centroids) >= num_points:
        selected_indices = np.random.choice(len(centroids), num_points, replace=False)
        return centroids[selected_indices]
    else:
        # Pad with random selection
        remaining = num_points - len(centroids)
        additional_indices = np.random.choice(len(points), remaining, replace=True)
        return np.vstack([centroids, points[additional_indices]])


def plyread(file_path):
    """
    Read PLY file and return point coordinates
    """
    ply_data = PlyData.read(file_path)
    pc = np.vstack([
        ply_data['vertex']['x'],
        ply_data['vertex']['y'],
        ply_data['vertex']['z']
    ]).T
    return pc.astype(np.float32)


class ModelNet(globset.Globset):
    
    def __init__(self, dataset_path, train=1, transform=None, classinfo=None):
        super().__init__(dataset_path, transform, classinfo)


class ShapeNet2(globset.Globset):
    
    def __init__(self, dataset_path, transform=None, classinfo=None):
        super().__init__(dataset_path, transform, classinfo)


class CADset4tracking(torch.utils.data.Dataset):
    def __init__(self, dataset, rigid_transform, source_modifier=None, template_modifier=None):
        self.dataset = dataset
        self.rigid_transform = rigid_transform
        self.source_modifier = source_modifier
        self.template_modifier = template_modifier

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        template = self.dataset[idx].copy()
        
        if self.template_modifier:
            template = self.template_modifier(template)
        
        source = self.rigid_transform(template)
        
        if self.source_modifier:
            source = self.source_modifier(source)
        
        igt = self.rigid_transform.igt
        return template, source, igt


class CADset4tracking_fixed_perturbation(torch.utils.data.Dataset):
    @staticmethod
    def generate_perturbations(batch_size, mag, randomly=False):
        if randomly:
            amp = torch.rand(batch_size, 1) * mag
        else:
            amp = mag
        x = torch.randn(batch_size, 6)
        x = x / x.norm(p=2, dim=1, keepdim=True) * amp
        return x.numpy()

    @staticmethod
    def generate_rotations(batch_size, mag, randomly=False):
        if randomly:
            amp = torch.rand(batch_size, 1) * mag
        else:
            amp = mag
        w = torch.randn(batch_size, 3)
        w = w / w.norm(p=2, dim=1, keepdim=True) * amp
        v = torch.zeros(batch_size, 3)
        x = torch.cat((w, v), dim=1)
        return x.numpy()

    def __init__(self, dataset, perturbation, source_modifier=None, template_modifier=None,
                 fmt_trans=False):
        self.dataset = dataset
        self.perturbation = perturbation
        self.source_modifier = source_modifier
        self.template_modifier = template_modifier
        self.fmt_trans = fmt_trans

    def do_transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6]
        if not self.fmt_trans:
            # x: twist-vector
            g = se3.exp(x).to(p0) # [1, 4, 4]
            p1 = se3.transform(g, p0)
            igt = g.squeeze(0) # igt: p0 -> p1
        else:
            # x: rotation and translation
            w = x[:, 0:3]
            q = x[:, 3:6]
            R = so3.exp(w).to(p0) # [1, 3, 3]
            g = torch.zeros(1, 4, 4)
            g[:, 3, 3] = 1
            g[:, 0:3, 0:3] = R # rotation
            g[:, 0:3, 3] = q   # translation
            p1 = se3.transform(g, p0)
            igt = g.squeeze(0) # igt: p0 -> p1
        return p1, igt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        twist = torch.from_numpy(np.array(self.perturbation[idx])).contiguous().view(1, 6)
        pm, _ = self.dataset[idx]
        x = twist.to(pm)
        if self.source_modifier:
            p_ = self.source_modifier(pm)
            p1, igt = self.do_transform(p_, x)
        else:
            p1, igt = self.do_transform(pm, x)

        if self.template_modifier:
            p0 = self.template_modifier(pm)
        else:
            p0 = pm

        # p0: template, p1: source, igt: transform matrix from p0 to p1
        return p0, p1, igt


class CADset4tracking_fixed_perturbation_random_sample(torch.utils.data.Dataset):
    """
    Random sampling version of fixed perturbation dataset
    """
    
    def __init__(self, dataset, perturbation, source_modifier=None, template_modifier=None,
                 fmt_trans=False, random_seed=42):
        self.dataset = dataset
        self.perturbation = perturbation
        self.source_modifier = source_modifier
        self.template_modifier = template_modifier
        self.fmt_trans = fmt_trans
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Pre-generate random sample indices for each perturbation
        self.sample_indices = []
        for _ in range(len(perturbation)):
            idx = random.randint(0, len(dataset) - 1)
            self.sample_indices.append(idx)

    def do_transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6]
        if not self.fmt_trans:
            # x: twist-vector
            g = se3.exp(x).to(p0) # [1, 4, 4]
            p1 = se3.transform(g, p0)
            igt = g.squeeze(0) # igt: p0 -> p1
        else:
            # x: rotation and translation
            w = x[:, 0:3]
            q = x[:, 3:6]
            R = so3.exp(w).to(p0) # [1, 3, 3]
            g = torch.zeros(1, 4, 4)
            g[:, 3, 3] = 1
            g[:, 0:3, 0:3] = R # rotation
            g[:, 0:3, 3] = q   # translation
            p1 = se3.transform(g, p0)
            igt = g.squeeze(0) # igt: p0 -> p1
        return p1, igt

    def __len__(self):
        # Return perturbation count, not dataset size
        return len(self.perturbation)

    def __getitem__(self, idx):
        # idx is perturbation index (0 to len(perturbation)-1)
        perturbation_idx = idx
        
        # Use pre-generated random sample index
        sample_idx = self.sample_indices[perturbation_idx]
        
        twist = torch.from_numpy(np.array(self.perturbation[perturbation_idx])).contiguous().view(1, 6)
        pm, _ = self.dataset[sample_idx]
        
        x = twist.to(pm)
        if self.source_modifier:
            p_ = self.source_modifier(pm)
            p1, igt = self.do_transform(p_, x)
        else:
            p1, igt = self.do_transform(pm, x)

        if self.template_modifier:
            p0 = self.template_modifier(pm)
        else:
            p0 = pm

        # p0: template, p1: source, igt: transform matrix from p0 to p1
        return p0, p1, igt


class C3VDDataset(torch.utils.data.Dataset):
    """
    C3VD Dataset for point cloud pairs
    
    Args:
        source_root (str): Source point cloud root directory (C3VD_ply_source)
        target_root (str, optional): Target point cloud root directory. If None, auto-set based on pair_mode
        transform: Point cloud transformation
        pair_mode (str): Pairing mode options:
            - 'one_to_one': Each source corresponds to specific target (original mode)
            - 'scene_reference': Each scene uses shared target (original mode)
            - 'source_to_source': Source-to-source pairing (data augmentation)
            - 'target_to_target': Target-to-target pairing (data augmentation)
            - 'all': Include all pairing modes (full data augmentation)
        reference_name (str, optional): Target cloud name for scene reference mode, None uses first cloud
        use_prefix (bool): Whether to use scene name prefix (before first underscore) as category
    """
    
    def __init__(self, source_root, target_root=None, transform=None, pair_mode='one_to_one', reference_name=None, use_prefix=False):
        self.source_root = source_root
        self.transform = transform
        self.pair_mode = pair_mode
        self.reference_name = reference_name
        self.use_prefix = use_prefix
        
        # Set target path based on pairing mode
        if target_root is None:
            if pair_mode == 'scene_reference':
                # Scene reference mode uses C3VD_ref directory
                parent_dir = os.path.dirname(source_root.rstrip('/'))
                self.target_root = os.path.join(parent_dir, 'C3VD_ref')
            else:
                # Other modes use visible_point_cloud_ply_depth directory
                self.target_root = source_root.replace('C3VD_ply_source', 'visible_point_cloud_ply_depth')
        else:
            self.target_root = target_root
        
        # Get all scene directories
        self.scene_dirs = []
        for item in os.listdir(self.source_root):
            scene_path = os.path.join(self.source_root, item)
            if os.path.isdir(scene_path):
                self.scene_dirs.append(item)
        
        self.scene_dirs.sort()
        
        # Build scene to class mapping
        self.scene_to_class = {}
        self.classes = set()
        self.scene_prefixes = {}  # Map scene to prefix
        
        for scene in self.scene_dirs:
            if self.use_prefix:
                # Extract scene prefix (part before first underscore)
                if '_' in scene:
                    prefix = scene.split('_')[0]
                else:
                    prefix = scene
                self.scene_prefixes[scene] = prefix
                self.scene_to_class[scene] = prefix
                self.classes.add(prefix)
            else:
                self.scene_to_class[scene] = scene
                self.classes.add(scene)
        
        # Record pairing type for data augmentation modes
        self.pair_types = []
        
        # Get all point cloud files in each scene for data augmentation modes
        self.scene_source_files = {}
        self.scene_target_files = {}
        
        for scene in self.scene_dirs:
            # Get all source point clouds in scene
            source_scene_path = os.path.join(self.source_root, scene)
            source_files = sorted(glob.glob(os.path.join(source_scene_path, '*.ply')))
            self.scene_source_files[scene] = [os.path.basename(f) for f in source_files]
            
            # Get all target point clouds in scene
            target_scene_path = os.path.join(self.target_root, scene)
            if os.path.exists(target_scene_path):
                target_files = sorted(glob.glob(os.path.join(target_scene_path, '*.ply')))
                self.scene_target_files[scene] = [os.path.basename(f) for f in target_files]
            else:
                self.scene_target_files[scene] = []
        
        # Create point cloud pairs based on pairing mode
        self.pairs = []
        if pair_mode == 'one_to_one':
            # Handle original one_to_one mode
            self._create_one_to_one_pairs()
        elif pair_mode == 'scene_reference':
            # Handle original scene_reference mode
            self._create_scene_reference_pairs()
        elif pair_mode == 'source_to_source':
            # New: source-to-source pairing
            self._create_source_to_source_pairs()
        elif pair_mode == 'target_to_target':
            # New: target-to-target pairing
            self._create_target_to_target_pairs()
        elif pair_mode == 'all':
            # If all mode, create all pairings
            self._create_one_to_one_pairs()
            # Add other pairings after one_to_one
            self._create_scene_reference_pairs()
            self._create_source_to_source_pairs()
            self._create_target_to_target_pairs()
        
        # Print pairing counts by type
        if pair_mode == 'all':
            type_counts = {}
            for pair_type in self.pair_types:
                type_counts[pair_type] = type_counts.get(pair_type, 0) + 1
            print(f"Total pairs: {len(self.pairs)}")
            for ptype, count in type_counts.items():
                print(f"  {ptype}: {count}")
        else:
            print(f"Created {len(self.pairs)} {pair_mode} pairs")

    def _create_one_to_one_pairs(self):
        """Create one-to-one source to target pairing"""
        for scene in self.scene_dirs:
            source_files = self.scene_source_files[scene]
            target_files = self.scene_target_files[scene]
            
            for source_file in source_files:
                # Extract frame index from source filename
                basename = os.path.splitext(source_file)[0]
                frame_idx = basename[:4]  # Extract first 4 digits as frame index
                
                # Build corresponding target filename
                target_file = f"{frame_idx}.ply"
                
                # Confirm target file exists
                if target_file in target_files:
                    source_path = os.path.join(self.source_root, scene, source_file)
                    target_path = os.path.join(self.target_root, scene, target_file)
                    self.pairs.append((source_path, target_path, self.scene_to_class[scene]))
                    
                    # Record pairing type (for all mode)
                    if hasattr(self, 'pair_types'):
                        self.pair_types.append('one_to_one')

    def _create_scene_reference_pairs(self):
        """Create source to reference target pairing"""
        for scene in self.scene_dirs:
            source_files = self.scene_source_files[scene]
            target_files = self.scene_target_files[scene]
            
            if not target_files:
                continue
            
            # Find reference point cloud
            if self.reference_name:
                # Use specified reference name
                reference_file = self.reference_name
            else:
                # Use first target as reference
                if self.pair_mode == 'scene_reference':
                    # For scene_reference mode, use first file
                    reference_file = target_files[0]  # Already sorted, take first
                
            # Confirm reference file exists
            if reference_file in target_files:
                # Each source corresponds to same reference
                for source_file in source_files:
                    # If all mode, avoid duplicate pairings
                    if self.pair_mode == 'all':
                        # Check if this pairing already exists in one_to_one mode
                        basename = os.path.splitext(source_file)[0]
                        frame_idx = basename[:4]
                        corresponding_target = f"{frame_idx}.ply"
                        
                        # If reference is exactly the corresponding target, skip
                        if reference_file == corresponding_target:
                            continue
                    
                    source_path = os.path.join(self.source_root, scene, source_file)
                    target_path = os.path.join(self.target_root, scene, reference_file)
                    self.pairs.append((source_path, target_path, self.scene_to_class[scene]))
                    
                    # Record pairing type (for all mode)
                    if hasattr(self, 'pair_types'):
                        self.pair_types.append('scene_reference')

    def _create_source_to_source_pairs(self):
        """Create source-to-source pairing (data augmentation)"""
        for scene in self.scene_dirs:
            source_files = self.scene_source_files[scene]
            
            if len(source_files) < 2:
                continue
            
            # Need at least two source files for pairing
            source_files = sorted(source_files)
            
            # Pair adjacent frame sources
            for i in range(len(source_files) - 1):
                source_file1 = source_files[i]
                source_file2 = source_files[i + 1]
                
                source_path1 = os.path.join(self.source_root, scene, source_file1)
                source_path2 = os.path.join(self.source_root, scene, source_file2)
                self.pairs.append((source_path1, source_path2, self.scene_to_class[scene]))
                
                # Record pairing type (for all mode)
                if hasattr(self, 'pair_types'):
                    self.pair_types.append('source_to_source')

    def _create_target_to_target_pairs(self):
        """Create target-to-target pairing (data augmentation)"""
        for scene in self.scene_dirs:
            target_files = self.scene_target_files[scene]
            
            if len(target_files) < 2:
                continue
            
            # Need at least two target files for pairing
            target_files = sorted(target_files)
            
            # Pair adjacent frame targets
            for i in range(len(target_files) - 1):
                target_file1 = target_files[i]
                target_file2 = target_files[i + 1]
                
                target_path1 = os.path.join(self.target_root, scene, target_file1)
                target_path2 = os.path.join(self.target_root, scene, target_file2)
                self.pairs.append((target_path1, target_path2, self.scene_to_class[scene]))
                
                # Record pairing type (for all mode)
                if hasattr(self, 'pair_types'):
                    self.pair_types.append('target_to_target')

    def get_scene_indices(self, scene_names):
        """Get indices for specific scenes"""
        indices = []
        for i, (_, _, scene_class) in enumerate(self.pairs):
            if scene_class in scene_names:
                indices.append(i)
        return indices

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source_path, target_path, scene_class = self.pairs[idx]
        
        source_pc = plyread(source_path)
        target_pc = plyread(target_path)
        
        if self.transform:
            source_pc = self.transform(source_pc)
            target_pc = self.transform(target_pc)
        
        return source_pc, target_pc, scene_class


class C3VDset4tracking(torch.utils.data.Dataset):
    
    def __init__(self, dataset, rigid_transform, num_points=1024, use_voxelization=False, voxel_config=None):
        self.dataset = dataset
        self.rigid_transform = rigid_transform
        self.num_points = num_points
        self.use_voxelization = use_voxelization
        # Set voxelization configuration
        if self.use_voxelization and voxelize_point_clouds is not None:
            if voxel_config is None and VoxelizationConfig is not None:
                self.voxel_config = VoxelizationConfig()
            else:
                self.voxel_config = voxel_config
        else:
            self.voxel_config = None
        
        # Keep original resampler as fallback
        self.resampler = transforms.Resampler(num_points)
        
        print(f"C3VDset4tracking initialized:")
        print(f"  Target points: {num_points}")
        print(f"  Voxelization: {'Complex voxelization' if use_voxelization else 'Open3D style VoxelGrid downsampling'}")
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        try:
            # Get original point cloud pair
            source, target, _ = self.dataset[idx]  # Ignore original igt
            
            # Convert to numpy array for processing
            source_np = source.numpy() if isinstance(source, torch.Tensor) else source
            target_np = target.numpy() if isinstance(target, torch.Tensor) else target
            
            # Data cleaning: remove invalid points
            source_mask = np.isfinite(source_np).all(axis=1)
            target_mask = np.isfinite(target_np).all(axis=1)
            
            source_clean = source_np[source_mask]
            target_clean = target_np[target_mask]
            
            if len(source_clean) < 100 or len(target_clean) < 100:
                raise ValueError(f"Point cloud too small after cleaning at index {idx}: {len(source_clean)}")
            
            # Select processing strategy: complex voxelization vs Open3D style VoxelGrid downsampling
            if self.use_voxelization and voxelize_point_clouds is not None:
                try:
                    processed_source, processed_target = voxelize_point_clouds(
                        source_clean, target_clean, self.num_points, self.voxel_config, fallback_to_sampling=True)
                except Exception as e:
                    print(f"Voxelization failed, falling back to resampling: {e}")
                    processed_source = self.resampler(torch.from_numpy(source_clean).float()).numpy()
                    processed_target = self.resampler(torch.from_numpy(target_clean).float()).numpy()
            else:
                processed_source = voxel_down_sample_numpy(source_clean, self.num_points)
                processed_target = voxel_down_sample_numpy(target_clean, self.num_points)

            # New: Ensure point count equals self.num_points
            if processed_source.shape[0] != self.num_points:
                processed_source = voxel_down_sample_numpy(processed_source, self.num_points)
            if processed_target.shape[0] != self.num_points:
                processed_target = voxel_down_sample_numpy(processed_target, self.num_points)

            # Normalize based on use_voxelization branch
            if self.use_voxelization and voxelize_point_clouds is not None:
                # Normalize source separately
                source_tensor = torch.from_numpy(processed_source).float()
                min_vals = source_tensor.min(dim=0)[0]
                max_vals = source_tensor.max(dim=0)[0]
                center_s = (min_vals + max_vals) / 2.0
                scale_s = (max_vals - min_vals).max() if (max_vals - min_vals).max() > 1e-6 else 1.0
                source_tensor = (source_tensor - center_s) / scale_s

                # Normalize target separately
                target_tensor = torch.from_numpy(processed_target).float()
                min_vals_t = target_tensor.min(dim=0)[0]
                max_vals_t = target_tensor.max(dim=0)[0]
                center_t = (min_vals_t + max_vals_t) / 2.0
                scale_t = (max_vals_t - min_vals_t).max() if (max_vals_t - min_vals_t).max() > 1e-6 else 1.0
                target_tensor = (target_tensor - center_t) / scale_t
            else:
                # Joint normalization (unify centering + scaling in numpy)
                all_pts = np.vstack([processed_source, processed_target])
                min_bounds = all_pts.min(axis=0)
                max_bounds = all_pts.max(axis=0)
                center = (min_bounds + max_bounds) / 2.0
                scales = max_bounds - min_bounds
                scale = scales.max() if scales.max() > 1e-6 else 1.0
                processed_source = (processed_source - center) / scale
                processed_target = (processed_target - center) / scale
                source_tensor = torch.from_numpy(processed_source).float()
                target_tensor = torch.from_numpy(processed_target).float()

            # Apply random rigid transformation
            transformed_source = self.rigid_transform(source_tensor)
            igt = self.rigid_transform.igt  # Get true transformation matrix

            # Return: template, transformed source, transformation matrix
            return target_tensor, transformed_source, igt
        except Exception as e:
            print(f"Error processing point cloud at index {idx}: {str(e)}")
            raise


class C3VDset4tracking_test(C3VDset4tracking):
    
    def __init__(self, dataset, rigid_transform, num_points=1024, 
                 use_voxelization=False, voxel_config=None):
        # Call parent constructor
        super().__init__(dataset, rigid_transform, num_points, use_voxelization, voxel_config)
        
        # Save file index and point cloud pair information dictionary
        self.cloud_info = {}
        
        # Collect all possible point cloud pair information
        if hasattr(dataset, 'pairs'):
            self.original_pairs = dataset.pairs
            self.original_pair_scenes = dataset.pair_scenes if hasattr(dataset, 'pair_scenes') else None
        elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'pairs'):
            self.original_pairs = dataset.dataset.pairs
            self.original_pair_scenes = dataset.dataset.pair_scenes if hasattr(dataset.dataset, 'pair_scenes') else None
        else:
            print("Warning: Unable to find original point cloud pair information")
            self.original_pairs = None
            self.original_pair_scenes = None
    
    def __getitem__(self, index):
        """Get test data item, while saving original point cloud information"""
        try:
            # 1. Get original point cloud pair
            source, target, _ = self.dataset[index]

            # 2. Convert to numpy array and clean invalid points
            source_np = source.numpy() if isinstance(source, torch.Tensor) else source
            target_np = target.numpy() if isinstance(target, torch.Tensor) else target
            
            source_mask = np.isfinite(source_np).all(axis=1)
            target_mask = np.isfinite(target_np).all(axis=1)
            
            source_clean = source_np[source_mask]
            target_clean = target_np[target_mask]
            
            if len(source_clean) < 100 or len(target_clean) < 100:
                raise ValueError(f"Point cloud too small after cleaning at index {index}: {len(source_clean)}")

            # 3. Select processing strategy: complex voxelization vs Open3D style VoxelGrid downsampling
            if self.use_voxelization and voxelize_point_clouds is not None:
                try:
                    processed_source, processed_target = voxelize_point_clouds(
                        source_clean, target_clean, self.num_points, self.voxel_config, fallback_to_sampling=True)
                except Exception as e:
                    print(f"Voxelization failed, falling back to resampling: {e}")
                    source_tensor_tmp = torch.from_numpy(source_clean).float()
                    target_tensor_tmp = torch.from_numpy(target_clean).float()
                    processed_source = self.resampler(source_tensor_tmp).numpy()
                    processed_target = self.resampler(target_tensor_tmp).numpy()
            else:
                processed_source = voxel_down_sample_numpy(source_clean, self.num_points)
                processed_target = voxel_down_sample_numpy(target_clean, self.num_points)

            # 4. Ensure point count equals self.num_points (key fix)
            if processed_source.shape[0] != self.num_points:
                processed_source = voxel_down_sample_numpy(processed_source, self.num_points)
            if processed_target.shape[0] != self.num_points:
                processed_target = voxel_down_sample_numpy(processed_target, self.num_points)

            # Convert back to torch tensor
            source_tensor = torch.from_numpy(processed_source).float()
            target_tensor = torch.from_numpy(processed_target).float()
            
            # 5. Normalize separately
            source_min_vals = source_tensor.min(dim=0)[0]
            source_max_vals = source_tensor.max(dim=0)[0]
            source_center = (source_min_vals + source_max_vals) / 2
            source_scale = (source_max_vals - source_min_vals).max()
            if source_scale < 1e-10:
                raise ValueError(f"Source point cloud normalization scale too small at index {index}: {source_scale}")
            source_normalized = (source_tensor - source_center) / source_scale
            
            target_min_vals = target_tensor.min(dim=0)[0]
            target_max_vals = target_tensor.max(dim=0)[0]
            target_center = (target_min_vals + target_max_vals) / 2
            target_scale = (target_max_vals - target_min_vals).max()
            if target_scale < 1e-10:
                raise ValueError(f"Target point cloud normalization scale too small at index {index}: {target_scale}")
            target_normalized = (target_tensor - target_center) / target_scale
            
            # 6. Apply rigid transformation
            # In test mode, rigid_transform is a special class that applies a fixed perturbation based on index
            transformed_source = self.rigid_transform(source_normalized)
            igt = self.rigid_transform.igt

            # 7. Collect original point cloud information
            if hasattr(self.dataset, 'indices') and self.original_pairs:
                orig_index = self.dataset.indices[index]
                source_file, target_file = self.original_pairs[orig_index]
                scene = self.original_pair_scenes[orig_index] if self.original_pair_scenes else "unknown"
            elif self.original_pairs and index < len(self.original_pairs):
                source_file, target_file = self.original_pairs[index]
                scene = self.original_pair_scenes[index] if self.original_pair_scenes else "unknown"
            else:
                source_file, target_file, scene = None, None, "unknown"

            scene_name = scene
            source_seq = "0000"
            if source_file:
                try:
                    norm_path = source_file.replace('\\', '/')
                    if scene_name == "unknown" and 'C3VD_ply_source' in norm_path:
                        parts = norm_path.split('/')
                        idx_part = [i for i, part in enumerate(parts) if part == 'C3VD_ply_source']
                        if idx_part and idx_part[0] + 1 < len(parts):
                            scene_name = parts[idx_part[0] + 1]
                    basename = os.path.basename(source_file)
                    if basename.endswith("_depth_pcd.ply") and basename[:4].isdigit():
                        source_seq = basename[:4]
                    else:
                        import re
                        numbers = re.findall(r'\\d+', basename)
                        if numbers:
                            source_seq = numbers[0].zfill(4)
                except Exception as e:
                    print(f"Warning: Error extracting scene name: {str(e)}")
            
            identifier = f"{scene_name}_{source_seq}_pert{index:04d}"

            # Use current_perturbation_index might be a custom implementation in test_pointlk.py, here for compatibility
            pert_idx = index
            if hasattr(self.rigid_transform, 'current_perturbation_index'):
                pert_idx = self.rigid_transform.current_perturbation_index - 1

            self.cloud_info[index] = {
                'identifier': identifier,
                'scene': scene_name,
                'sequence': source_seq,
                'source_file': source_file,
                'target_file': target_file,
                'sample_index': index,
                'perturbation_index': pert_idx,
                'original_source': source_normalized.clone(),
                'original_target': target_normalized.clone(),
                'igt': igt.clone() if igt is not None else None
            }

            return target_normalized, transformed_source, igt

        except Exception as e:
            print(f"Error processing point cloud at index {index}: {str(e)}")
            raise

    def get_cloud_info(self, index):
        """Get point cloud information for a specific index"""
        return self.cloud_info.get(index, {})
    
    def get_original_clouds(self, index):
        """Get original point cloud pair for a specific index"""
        info = self.cloud_info.get(index, {})
        return info.get('original_source'), info.get('original_target')
    
    def get_identifier(self, index):
        """Get point cloud identifier for a specific index"""
        info = self.cloud_info.get(index, {})
        return info.get('identifier', f"unknown_{index:04d}")


class C3VDset4tracking_test_random_sample(C3VDset4tracking_test):
    """
    Random sampling version of C3VD test dataset
    """
    
    def __init__(self, dataset, rigid_transform, num_points=1024, 
                 use_voxelization=False, voxel_config=None, random_seed=42):
        # Call parent constructor
        super().__init__(dataset, rigid_transform, num_points, use_voxelization, voxel_config)
        
        # Get perturbations from rigid_transform
        if hasattr(rigid_transform, 'perturbations'):
            self.perturbations = rigid_transform.perturbations
        else:
            raise ValueError("rigid_transform must have perturbations attribute for random sampling mode")
        
        self.original_dataset_size = len(dataset)
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Pre-generate random sample indices for each perturbation
        self.sample_indices = []
        for _ in range(len(self.perturbations)):
            idx = random.randint(0, self.original_dataset_size - 1)
            self.sample_indices.append(idx)
        
        print(f"C3VD Random sampling mode activated for gt_poses.csv:")
        print(f"- Total perturbations: {len(self.perturbations)}")
        print(f"- Original dataset size: {self.original_dataset_size}")
        print(f"- Random seed: {random_seed}")
        print(f"- Test iterations: {len(self.perturbations)} (one per perturbation)")
    
    def __len__(self):
        # Return perturbation count, not dataset size
        return len(self.perturbations)
    
    def __getitem__(self, index):
        """
        Get test data item, using randomly selected samples and specified perturbation
        Args:
            index: Perturbation index (0 to len(perturbations)-1)
        """
        # Use pre-generated random sample index
        sample_idx = self.sample_indices[index]
        
        try:
            # 1. Get original point cloud pair (from randomly selected sample)
            source, target, _ = self.dataset[sample_idx]
            
            # 2. Convert to numpy array and clean invalid points
            source_np = source.numpy() if isinstance(source, torch.Tensor) else source
            target_np = target.numpy() if isinstance(target, torch.Tensor) else target
            
            source_mask = np.isfinite(source_np).all(axis=1)
            target_mask = np.isfinite(target_np).all(axis=1)
            
            source_clean = source_np[source_mask]
            target_clean = target_np[target_mask]
            
            if len(source_clean) < 100 or len(target_clean) < 100:
                raise ValueError(f"Point cloud too small at sample index {sample_idx}: {len(source_clean)}")
            
            # 3. Select processing strategy: complex voxelization vs Open3D style VoxelGrid downsampling
            if self.use_voxelization and voxelize_point_clouds is not None:
                try:
                    processed_source, processed_target = voxelize_point_clouds(
                        source_clean, target_clean, self.num_points, self.voxel_config, fallback_to_sampling=True)
                except Exception as e:
                    print(f"Voxelization failed, falling back to resampling: {e}")
                    source_tensor_tmp = torch.from_numpy(source_clean).float()
                    target_tensor_tmp = torch.from_numpy(target_clean).float()
                    processed_source = self.resampler(source_tensor_tmp).numpy()
                    processed_target = self.resampler(target_tensor_tmp).numpy()
            else:
                processed_source = voxel_down_sample_numpy(source_clean, self.num_points)
                processed_target = voxel_down_sample_numpy(target_clean, self.num_points)
            
            # 4. Ensure point count equals self.num_points (key fix)
            if processed_source.shape[0] != self.num_points:
                processed_source = voxel_down_sample_numpy(processed_source, self.num_points)
            if processed_target.shape[0] != self.num_points:
                processed_target = voxel_down_sample_numpy(processed_target, self.num_points)

            # Convert back to torch tensor
            source_tensor = torch.from_numpy(processed_source).float()
            target_tensor = torch.from_numpy(processed_target).float()
            
            # 5. Normalize separately
            source_min_vals = source_tensor.min(dim=0)[0]
            source_max_vals = source_tensor.max(dim=0)[0]
            source_center = (source_min_vals + source_max_vals) / 2
            source_scale = (source_max_vals - source_min_vals).max()
            if source_scale < 1e-10:
                raise ValueError(f"Source point cloud normalization scale too small at sample index {sample_idx}: {source_scale}")
            source_normalized = (source_tensor - source_center) / source_scale
            
            target_min_vals = target_tensor.min(dim=0)[0]
            target_max_vals = target_tensor.max(dim=0)[0]
            target_center = (target_min_vals + target_max_vals) / 2
            target_scale = (target_max_vals - target_min_vals).max()
            if target_scale < 1e-10:
                raise ValueError(f"Target point cloud normalization scale too small at sample index {sample_idx}: {target_scale}")
            target_normalized = (target_tensor - target_center) / target_scale
            
            # 6. Apply specific perturbation
            perturbation = self.perturbations[index]
            twist = torch.from_numpy(np.array(perturbation)).contiguous().view(1, 6)
            x = twist.to(source_normalized)
            
            if not getattr(self.rigid_transform, 'fmt_trans', False):
                g = se3.exp(x).to(source_normalized)
                transformed_source = se3.transform(g, source_normalized)
                igt = g.squeeze(0)
            else:
                w = x[:, 0:3]
                q = x[:, 3:6]
                R = so3.exp(w).to(source_normalized)
                g = torch.zeros(1, 4, 4)
                g[:, 3, 3] = 1
                g[:, 0:3, 0:3] = R
                g[:, 0:3, 3] = q
                transformed_source = se3.transform(g, source_normalized)
                igt = g.squeeze(0)

            # 7. Collect original point cloud information
            if hasattr(self.dataset, 'indices') and self.original_pairs:
                orig_index = self.dataset.indices[sample_idx]
                source_file, target_file = self.original_pairs[orig_index]
                scene = self.original_pair_scenes[orig_index] if self.original_pair_scenes else "unknown"
            elif self.original_pairs and sample_idx < len(self.original_pairs):
                source_file, target_file = self.original_pairs[sample_idx]
                scene = self.original_pair_scenes[sample_idx] if self.original_pair_scenes else "unknown"
            else:
                source_file, target_file, scene = None, None, "unknown"

            scene_name = scene
            source_seq = "0000"
            if source_file:
                try:
                    norm_path = source_file.replace('\\', '/')
                    if scene_name == "unknown" and 'C3VD_ply_source' in norm_path:
                        parts = norm_path.split('/')
                        idx_part = [i for i, part in enumerate(parts) if part == 'C3VD_ply_source']
                        if idx_part and idx_part[0] + 1 < len(parts):
                            scene_name = parts[idx_part[0] + 1]
                    basename = os.path.basename(source_file)
                    if basename.endswith("_depth_pcd.ply") and basename[:4].isdigit():
                        source_seq = basename[:4]
                    else:
                        import re
                        numbers = re.findall(r'\\d+', basename)
                        if numbers:
                            source_seq = numbers[0].zfill(4)
                except Exception as e:
                    print(f"Warning: Error extracting scene name: {str(e)}")

            identifier = f"{scene_name}_{source_seq}_pert{index:04d}"

            self.cloud_info[index] = {
                'identifier': identifier,
                'scene': scene_name,
                'sequence': source_seq,
                'source_file': source_file,
                'target_file': target_file,
                'sample_index': sample_idx,
                'perturbation_index': index,
                'original_source': source_normalized.clone(),
                'original_target': target_normalized.clone(),
                'igt': igt.clone() if igt is not None else None
            }

            return target_normalized, transformed_source, igt

        except Exception as e:
            print(f"Error processing perturbation index {index} (sample index {sample_idx}): {str(e)}")
            raise

    def get_cloud_info(self, index):
        """Get point cloud information for a specific index"""
        return self.cloud_info.get(index, {})
    
    def get_original_clouds(self, index):
        """Get original point cloud pair for a specific index"""
        info = self.cloud_info.get(index, {})
        return info.get('original_source'), info.get('original_target')
    
    def get_identifier(self, index):
        """Get point cloud identifier for a specific index"""
        info = self.cloud_info.get(index, {})
        return info.get('identifier', f"unknown_pert{index:04d}")

# EOF 