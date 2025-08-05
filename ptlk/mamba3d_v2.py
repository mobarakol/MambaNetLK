"""
    Mamba3D for Point Cloud Processing v2
    
    Simplified version using mamba-ssm package.
    Focuses on core functionality with cleaner implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Import mamba-ssm if available
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not available, using fallback implementation")


def flatten(x):
    return x.view(x.size(0), -1)


class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return flatten(x)


def mlp_layers(nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
    """Create multi-layer perceptron layers"""
    layers = []
    last = nch_input
    for i, outp in enumerate(nch_layers):
        if b_shared:
            weights = torch.nn.Conv1d(last, outp, 1)
        else:
            weights = torch.nn.Linear(last, outp)
        layers.append(weights)
        layers.append(torch.nn.BatchNorm1d(outp, momentum=bn_momentum))
        layers.append(torch.nn.ReLU())
        if b_shared == False and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp
    return layers


class MLPNet(torch.nn.Module):
    """Multi-layer perceptron network"""
    
    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
        super().__init__()
        self.b_shared = b_shared
        list_layers = mlp_layers(nch_input, nch_layers, b_shared, bn_momentum, dropout)
        self.layers = torch.nn.Sequential(*list_layers)

    def forward(self, inp):
        return self.layers(inp)


class PositionalEncoding3D(nn.Module):
    """3D positional encoding for point clouds"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding projection layer
        self.pos_projection = nn.Linear(3, d_model)
        
    def forward(self, points):
        """
        Args:
            points: [B, N, 3] point cloud coordinates
        Returns:
            pos_encoding: [B, N, d_model] positional encoding
        """
        # Directly use linear layer to map 3D coordinates to d_model dimension
        pos_encoding = self.pos_projection(points)  # [B, N, d_model]
        return pos_encoding


class Mamba3DBlock(nn.Module):
    """Mamba3D block using mamba-ssm package"""
    def __init__(self, d_model, d_state=16, d_ff=None, expand=2, dropout=0.1):
        super().__init__()
        
        # Completely adopt v1's parameter optimization strategy
        stable_d_state = max(8, d_state // 2)  # Optimization: reduce state dimension
        stable_expand = max(1.5, expand * 0.75)  # Optimization: reduce expansion ratio
        
        if d_ff is None:
            d_ff = int(d_model * 2)  # Optimization: reduce FFN size
            
        # Use mamba-ssm library's Mamba layer to replace v1's S6Layer
        self.mamba_layer = Mamba(
            d_model=d_model,
            d_state=stable_d_state,
            d_conv=4,
            expand=stable_expand
        )
        
        # Completely adopt v1's feed-forward network structure
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Add pre-normalization layer, consistent with v1's S6Layer
        self.pre_norm = nn.LayerNorm(d_model)  # Pre-normalization before Mamba layer
        self.norm = nn.LayerNorm(d_model)      # Normalization before FFN
        
    def forward(self, x):
        # Add pre-normalization, consistent with v1's S6Layer
        residual = x
        x_prenorm = self.pre_norm(x)  # Pre-normalization
        x = self.mamba_layer(x_prenorm)  # Corresponds to v1's self.s6_layer(x)
        x = x + residual  # Residual connection
        
        # FFN part maintains original structure
        x_norm = self.norm(x)
        x = x + self.feed_forward(x_norm)
        return x


def symfn_max(x):
    """Max pooling symmetry function"""
    return torch.max(x, dim=1)[0]


def symfn_avg(x):
    """Average pooling symmetry function"""
    return torch.mean(x, dim=1)


def symfn_selective(x):
    """Selective pooling based on attention mechanism"""
    # Use softmax to get weight for each point
    weights = torch.softmax(torch.sum(x, dim=-1), dim=-1)  # [B, N]
    weights = weights.unsqueeze(-1)  # [B, N, 1]
    
    # Weighted sum
    aggregated = torch.sum(x * weights, dim=1)  # [B, K]
    return aggregated


class Mamba3D_features(torch.nn.Module):
    """
    Mamba3D feature extractor v2
    Uses mamba-ssm for improved efficiency
    """
    
    def __init__(self, dim_k=1024, sym_fn=symfn_max, scale=1, num_mamba_blocks=3, 
                 d_state=16, expand=2):
        super().__init__()
        
        # Feature dimension settings - completely adopt v1's optimized version
        self.d_model = max(64, int(128 / scale))  # Ensure minimum dimension
        self.dim_k = int(dim_k / scale)   # Final output feature dimension
        self.num_mamba_blocks = min(num_mamba_blocks, 3)  # Limit number of blocks
        
        # Input embedding layer: map 3D coordinates to high-dimensional feature space
        self.input_projection = nn.Linear(3, self.d_model)
        
        # 3D positional encoding - use v1's coordinate-based positional encoding
        self.pos_encoding = PositionalEncoding3D(self.d_model)
        
        # Multi-layer Mamba blocks - use optimized parameters, consistent with v1
        self.mamba_blocks = nn.ModuleList([
            Mamba3DBlock(self.d_model, d_state=max(8, d_state//2), expand=max(1.5, expand*0.75))
            for _ in range(self.num_mamba_blocks)
        ])
        
        # Feature transformation layer: map from mamba dimension to final feature dimension
        self.feature_transform = MLPNet(
            self.d_model, 
            [int(256/scale), self.dim_k], 
            b_shared=True  # Use Conv1d layer to support [B, d_model, N] format input
        )
        
        # Aggregation function
        self.sy = sym_fn
        
        # Maintain compatibility attributes with PointNet_features
        self.t_out_t2 = None
        self.t_out_h1 = None

    def forward(self, points):
        """
        Forward propagation - completely consistent flow with v1 version
        
        Args:
            points: [B, N, 3] input point cloud
            
        Returns:
            (global_features, point_features): ([B, K], [B, N, K])
        """
        batch_size, num_points, _ = points.size()
        
        # Input projection: 3D coordinates -> high-dimensional features
        x = self.input_projection(points)  # [B, N, d_model]
        
        # Add coordinate-based positional encoding - consistent with v1
        pos_encoding = self.pos_encoding(points)  # [B, N, d_model]
        x = x + pos_encoding
        
        # Save intermediate features (for compatibility)
        self.t_out_h1 = x.transpose(1, 2)  # [B, d_model, N] format for compatibility
        
        # Pass through multi-layer Mamba blocks
        for mamba_block in self.mamba_blocks:
            x = mamba_block(x)  # [B, N, d_model]
        
        # Feature transformation
        # Convert to [B, d_model, N] format for Conv1d processing
        x = x.transpose(1, 2)  # [B, d_model, N]
        x = self.feature_transform(x)  # [B, dim_k, N]
        
        # Convert back to [B, N, dim_k] format for aggregation
        x = x.transpose(1, 2)  # [B, N, dim_k]
        point_features = x
        
        # Global aggregation
        if self.sy == symfn_max:
            global_features = symfn_max(point_features)  # [B, dim_k]
        elif self.sy == symfn_avg:
            global_features = symfn_avg(point_features)  # [B, dim_k]
        elif self.sy == symfn_selective:
            global_features = symfn_selective(point_features)  # [B, dim_k]
        else:
            # Default to max pooling
            global_features = symfn_max(point_features)  # [B, dim_k]
        
        return global_features, point_features


class Mamba3D_classifier(torch.nn.Module):
    """
    Mamba3D classifier v2
    """
    
    def __init__(self, num_c, mambafeat, dim_k):
        """
        Args:
            num_c: number of classes
            mambafeat: Mamba3D_features instance
            dim_k: feature dimension
        """
        super().__init__()
        self.features = mambafeat
        
        # Classification head: feature vector -> classification result
        list_layers = mlp_layers(dim_k, [512, 256], b_shared=False, bn_momentum=0.1, dropout=0.0)
        list_layers.append(torch.nn.Linear(256, num_c))
        self.classifier = torch.nn.Sequential(*list_layers)

    def forward(self, points):
        """
        Forward propagation
        
        Args:
            points: [B, N, 3] input point cloud
            
        Returns:
            out: [B, num_c] classification output
        """
        global_feat, _ = self.features(points)
        out = self.classifier(global_feat)
        return out

    def loss(self, out, target, w=0.001):
        """
        Compute loss function
        
        Args:
            out: [B, num_c] classification output
            target: [B] ground truth labels
            w: regularization weight
            
        Returns:
            loss: total loss
        """
        # Classification loss
        loss_c = torch.nn.functional.nll_loss(
            torch.nn.functional.log_softmax(out, dim=1), target, size_average=False)

        # Note: For mamba model, we currently don't add transformation matrix regularization
        # because the structure of mamba mechanism is different from PointNet, no corresponding t_out_t2
        # If needed, other forms of regularization can be added
        t2 = self.features.t_out_t2
        if (t2 is None) or (w == 0):
            return loss_c

        # If transformation matrix exists, add regularization term
        batch = t2.size(0)
        K = t2.size(1)  # [B, K, K]
        I = torch.eye(K).repeat(batch, 1, 1).to(t2)
        A = t2.bmm(t2.transpose(1, 2))
        loss_m = torch.nn.functional.mse_loss(A, I, size_average=False)
        loss = loss_c + w * loss_m
        return loss

#EOF 