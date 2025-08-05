""" gives some transform methods for 3d points """
import math

import numpy as np
import torch
import torch.utils.data

from . import mesh
from .. import so3
from .. import se3


class Mesh2Points:
    def __init__(self):
        pass

    def __call__(self, mesh):
        mesh = mesh.clone()
        v = mesh.vertex_array
        return torch.from_numpy(v).type(dtype=torch.float)

class OnUnitSphere:
    def __init__(self, zero_mean=False):
        self.zero_mean = zero_mean

    def __call__(self, tensor):
        if self.zero_mean:
            m = tensor.mean(dim=0, keepdim=True) # [N, D] -> [1, D]
            v = tensor - m
        else:
            v = tensor
        nn = v.norm(p=2, dim=1) # [N, D] -> [N]
        nmax = torch.max(nn)
        return v / nmax

class OnUnitCube:
    def __init__(self):
        pass

    def method1(self, tensor):
        m = tensor.mean(dim=0, keepdim=True) # [N, D] -> [1, D]
        v = tensor - m
        s = torch.max(v.abs())
        v = v / s * 0.5
        return v

    def method2(self, tensor):
        c = torch.max(tensor, dim=0)[0] - torch.min(tensor, dim=0)[0] # [N, D] -> [D]
        s = torch.max(c) # -> scalar
        v = tensor / s
        return v - v.mean(dim=0, keepdim=True)

    def __call__(self, tensor):
        #return self.method1(tensor)
        return self.method2(tensor)


class OnJointUnitCube:
    """Transformation class for normalizing a pair of point clouds to a joint unit cube"""
    
    def __init__(self):
        self.cached_params = None  # For storing normalization parameters
    
    def __call__(self, points_pair):
        """
        Normalize a pair of point clouds using a common bounding box
        
        Args:
            points_pair: Tuple (points1, points2), two point cloud tensors
            
        Returns:
            Normalized point cloud pair: (normalized_points1, normalized_points2)
        """
        points1, points2 = points_pair
        
        try:
            # If cached parameters exist, apply them directly
            if self.cached_params is not None:
                scale, offset = self.cached_params
                return (points1 / scale - offset, points2 / scale - offset)
            
            # Check if point clouds are valid
            if not torch.isfinite(points1).all() or not torch.isfinite(points2).all():
                print("Warning: Input point clouds contain invalid values, fixing")
                points1 = torch.nan_to_num(points1, nan=0.0, posinf=1.0, neginf=-1.0)
                points2 = torch.nan_to_num(points2, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Calculate joint bounding box
            # Combine both point clouds
            all_points = torch.cat([points1, points2], dim=0)
            
            # Calculate joint bounding box dimensions
            min_coords = torch.min(all_points, dim=0)[0]  # [x_min, y_min, z_min]
            max_coords = torch.max(all_points, dim=0)[0]  # [x_max, y_max, z_max]
            bbox_size = max_coords - min_coords  # [x_size, y_size, z_size]
            
            # Check if bounding box size is valid
            if torch.any(bbox_size <= 0) or not torch.isfinite(bbox_size).all():
                print("Warning: Invalid bounding box calculation, using default scaling")
                scale_factor = torch.tensor(1.0, device=points1.device)
                center = torch.zeros(3, device=points1.device)
            else:
                # Find maximum dimension for scaling
                scale_factor = torch.max(bbox_size)
                
                # Additional safety check
                if scale_factor <= 0 or not torch.isfinite(scale_factor):
                    print("Warning: Invalid scale factor, using default value 1.0")
                    scale_factor = torch.tensor(1.0, device=points1.device)
                
                # Calculate normalization center
                center = (min_coords + max_coords) / 2
                
                # Check if center point is valid
                if not torch.isfinite(center).all():
                    print("Warning: Invalid center point calculation, using origin")
                    center = torch.zeros(3, device=points1.device)
            
            # Save transformation parameters for future use
            self.cached_params = (scale_factor, center)
            
            # Apply normalization to both point clouds
            normalized_points1 = (points1 - center) / scale_factor
            normalized_points2 = (points2 - center) / scale_factor
            
            # Final check for valid results
            if not torch.isfinite(normalized_points1).all() or not torch.isfinite(normalized_points2).all():
                print("Warning: Normalized results contain invalid values, fixing")
                normalized_points1 = torch.nan_to_num(normalized_points1, nan=0.0, posinf=1.0, neginf=-1.0)
                normalized_points2 = torch.nan_to_num(normalized_points2, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return (normalized_points1, normalized_points2)
            
        except Exception as e:
            print(f"Joint normalization error: {e}, returning original point clouds")
            # Return original point clouds in case of error
            return points_pair
    
    def reset_params(self):
        """Reset cached parameters"""
        self.cached_params = None


class Resampler:
    """ [N, D] -> [M, D] """
    def __init__(self, num):
        self.num = num

    def __call__(self, tensor):
        # Check input point cloud
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        
        if tensor.dim() != 2:
            raise ValueError(f"Input must be a 2D tensor, got {tensor.dim()}D")
        
        num_points, dim_p = tensor.size()
        
        # Check if point cloud is empty
        if num_points == 0:
            raise ValueError("Input point cloud is empty")
        
        # Check if point cloud contains invalid values
        if not torch.isfinite(tensor).all():
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # If point count already matches target, return as is
        if num_points == self.num:
            return tensor
        
        # If point count exceeds target, sample randomly
        if num_points > self.num:
            idx = torch.randperm(num_points)[:self.num]
            return tensor[idx]
        
        # If point count is less than target, use repeated sampling
        # First copy the entire point cloud
        repeats = self.num // num_points
        remainder = self.num % num_points
        
        # Copy the complete part
        out = tensor.repeat(repeats, 1)
        
        # Add the remaining part
        if remainder > 0:
            idx = torch.randperm(num_points)[:remainder]
            out = torch.cat([out, tensor[idx]], dim=0)
        
        return out

class RandomTranslate:
    def __init__(self, mag=None, randomly=True):
        self.mag = 1.0 if mag is None else mag
        self.randomly = randomly
        self.igt = None

    def __call__(self, tensor):
        # tensor: [N, 3]
        amp = torch.rand(1) if self.randomly else 1.0
        t = torch.randn(1, 3).to(tensor)
        t = t / t.norm(p=2, dim=1, keepdim=True) * amp * self.mag

        g = torch.eye(4).to(tensor)
        g[0:3, 3] = t[0, :]
        self.igt = g # [4, 4]

        p1 = tensor + t
        return p1

class RandomRotator:
    def __init__(self, mag=None, randomly=True):
        self.mag = math.pi if mag is None else mag
        self.randomly = randomly
        self.igt = None

    def __call__(self, tensor):
        # tensor: [N, 3]
        amp = torch.rand(1) if self.randomly else 1.0
        w = torch.randn(1, 3)
        w = w / w.norm(p=2, dim=1, keepdim=True) * amp * self.mag

        g = so3.exp(w).to(tensor) # [1, 3, 3]
        self.igt = g.squeeze(0) # [3, 3]

        p1 = so3.transform(g, tensor) # [1, 3, 3] x [N, 3] -> [N, 3]
        return p1

class RandomRotatorZ:
    def __init__(self):
        self.mag = 2 * math.pi

    def __call__(self, tensor):
        # tensor: [N, 3]
        w = torch.Tensor([0, 0, 1]).view(1, 3) * torch.rand(1) * self.mag

        g = so3.exp(w).to(tensor) # [1, 3, 3]

        p1 = so3.transform(g, tensor)
        return p1

class RandomJitter:
    """ generate perturbations """
    def __init__(self, scale=0.01, clip=0.05):
        self.scale = scale
        self.clip = clip
        self.e = None

    def jitter(self, tensor):
        noise = torch.zeros_like(tensor).to(tensor) # [N, 3]
        noise.normal_(0, self.scale)
        noise.clamp_(-self.clip, self.clip)
        self.e = noise
        return tensor.add(noise)

    def __call__(self, tensor):
        return self.jitter(tensor)


class RandomTransformSE3:
    """ rigid motion """
    def __init__(self, mag=1, mag_randomly=False):
        self.mag = mag
        self.randomly = mag_randomly

        self.gt = None
        self.igt = None

    def generate_transform(self):
        # return: a twist-vector
        amp = self.mag
        if self.randomly:
            amp = torch.rand(1, 1) * self.mag
        x = torch.randn(1, 6)
        x = x / x.norm(p=2, dim=1, keepdim=True) * amp

        return x # [1, 6]

    def apply_transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6]
        g = se3.exp(x).to(p0)   # [1, 4, 4]
        gt = se3.exp(-x).to(p0) # [1, 4, 4]

        p1 = se3.transform(g, p0)
        self.gt = gt.squeeze(0) #  gt: p1 -> p0
        self.igt = g.squeeze(0) # igt: p0 -> p1
        return p1

    def transform(self, tensor):
        x = self.generate_transform()
        return self.apply_transform(tensor, x)

    def __call__(self, tensor):
        return self.transform(tensor)



#EOF