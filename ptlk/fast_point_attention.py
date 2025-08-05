"""
    Fast Point Attention for Point Cloud Processing
    
    A lightweight attention mechanism for point cloud feature extraction.
    Optimized for efficiency with simplified attention computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class SimplifiedPositionalEncoding(nn.Module):
    """Simplified positional encoding for point clouds"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Use simpler positional encoding: only use a linear layer
        self.pos_projection = nn.Linear(3, d_model // 4)  # Reduce positional encoding dimension
        
    def forward(self, points):
        """
        Args:
            points: [B, N, 3] point cloud coordinates
        Returns:
            pos_encoding: [B, N, d_model//4] simplified positional encoding
        """
        pos_encoding = self.pos_projection(points)  # [B, N, d_model//4]
        return pos_encoding


class FastAttention(nn.Module):
    """Fast attention mechanism for point clouds"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.scale = 1.0 / math.sqrt(d_model)
        
        # Use single-head attention to reduce computation
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: [B, N, d_model] input features
        Returns:
            output: [B, N, d_model] attention output
        """
        batch_size, seq_len, d_model = x.size()
        
        # Input for residual connection
        residual = x
        
        # Compute Q, K, V - single-head attention
        Q = self.query(x)  # [B, N, d_model]
        K = self.key(x)    # [B, N, d_model]
        V = self.value(x)  # [B, N, d_model]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, N, N]
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights
        context = torch.matmul(attention_weights, V)  # [B, N, d_model]
        
        # Residual connection and layer normalization
        output = self.layer_norm(context + residual)
        
        return output


class SimpleFeedForward(nn.Module):
    """Simple feed-forward network"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        # Use smaller hidden layer dimension
        d_ff = d_model * 2  # Changed from 4x to 2x
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        residual = x
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.layer_norm(x + residual)


class FastAttentionBlock(nn.Module):
    """Fast attention block with residual connections"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.fast_attention = FastAttention(d_model, dropout)
        self.feed_forward = SimpleFeedForward(d_model, dropout)
        
    def forward(self, x):
        x = self.fast_attention(x)
        x = self.feed_forward(x)
        return x


def symfn_max(x):
    """Max pooling symmetry function"""
    return torch.max(x, dim=1)[0]


def symfn_avg(x):
    """Average pooling symmetry function"""
    return torch.mean(x, dim=1)


def symfn_fast_attention_pool(x):
    """Fast attention-based pooling"""
    # Use simplified attention weight computation
    batch_size, seq_len, d_model = x.size()
    
    # Compute global average feature as query
    global_feat = torch.mean(x, dim=1, keepdim=True)  # [B, 1, K]
    
    # Compute attention weights - use dot-product attention
    attention_scores = torch.sum(x * global_feat, dim=-1)  # [B, N]
    attention_weights = F.softmax(attention_scores, dim=-1).unsqueeze(-1)  # [B, N, 1]
    
    # Weighted sum
    aggregated = torch.sum(x * attention_weights, dim=1)  # [B, K]
    return aggregated


class FastPointAttention_features(torch.nn.Module):
    """
    Fast Point Attention feature extractor
    Lightweight alternative to transformer-based approaches
    """
    
    def __init__(self, dim_k=1024, sym_fn=symfn_max, scale=1, num_attention_blocks=2):
        super().__init__()
        
        # Feature dimension settings - use smaller hidden dimensions
        self.d_model = int(64 / scale)    # Significantly reduce hidden dimension
        self.dim_k = int(dim_k / scale)   # Final output feature dimension
        self.num_attention_blocks = num_attention_blocks
        
        # Input embedding layer: 3D coordinates -> feature space
        self.input_projection = nn.Linear(3, self.d_model - self.d_model // 4)
        
        # Simplified 3D positional encoding
        self.pos_encoding = SimplifiedPositionalEncoding(self.d_model)
        
        # Fast attention blocks (fewer in number)
        self.attention_blocks = nn.ModuleList([
            FastAttentionBlock(self.d_model)
            for _ in range(num_attention_blocks)
        ])
        
        # Feature transformation layer: use simpler MLP
        self.feature_transform = MLPNet(
            self.d_model, 
            [int(128/scale), self.dim_k], 
            b_shared=True
        )
        
        # Aggregation function
        self.sy = sym_fn
        
        # Maintain compatibility attributes with PointNet_features
        self.t_out_t2 = None
        self.t_out_h1 = None

    def forward(self, points):
        """
        Forward propagation
        
        Args:
            points: [B, N, 3] input point cloud
            
        Returns:
            features: [B, K] global feature vector
        """
        batch_size, num_points, _ = points.size()
        
        # Input projection: 3D coordinates -> feature space
        x = self.input_projection(points)  # [B, N, d_model-d_model//4]
        
        # Add simplified positional encoding
        pos_encoding = self.pos_encoding(points)  # [B, N, d_model//4]
        x = torch.cat([x, pos_encoding], dim=-1)  # [B, N, d_model]
        
        # Save intermediate features (for compatibility)
        self.t_out_h1 = x.transpose(1, 2)  # [B, d_model, N] format for compatibility
        
        # Pass through fast attention blocks
        for attention_block in self.attention_blocks:
            x = attention_block(x)  # [B, N, d_model]
        
        # Feature transformation
        # Convert to [B, d_model, N] format for Conv1d processing
        x = x.transpose(1, 2)  # [B, d_model, N]
        x = self.feature_transform(x)  # [B, dim_k, N]
        
        # Convert back to [B, N, dim_k] format for aggregation
        x = x.transpose(1, 2)  # [B, N, dim_k]
        
        # Global aggregation
        if self.sy == symfn_max:
            global_features = symfn_max(x)  # [B, dim_k]
        elif self.sy == symfn_avg:
            global_features = symfn_avg(x)  # [B, dim_k]
        elif self.sy == symfn_fast_attention_pool:
            global_features = symfn_fast_attention_pool(x)  # [B, dim_k]
        else:
            # Default to max pooling
            global_features = symfn_max(x)  # [B, dim_k]
        
        return global_features


class FastPointAttention_classifier(torch.nn.Module):
    """
    Fast Point Attention classifier
    """
    
    def __init__(self, num_c, fast_feat, dim_k):
        """
        Args:
            num_c: number of classes
            fast_feat: FastPointAttention_features instance
            dim_k: feature dimension
        """
        super().__init__()
        self.features = fast_feat
        
        # Classification head: feature vector -> classification result
        list_layers = mlp_layers(dim_k, [256, 128], b_shared=False, bn_momentum=0.1, dropout=0.0)
        list_layers.append(torch.nn.Linear(128, num_c))
        self.classifier = torch.nn.Sequential(*list_layers)

    def forward(self, points):
        """
        Forward propagation
        
        Args:
            points: [B, N, 3] input point cloud
            
        Returns:
            out: [B, num_c] classification output
        """
        feat = self.features(points)  # [B, dim_k]
        out = self.classifier(feat)   # [B, num_c]
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

        # For fast attention model, do not add additional regularization terms for now
        # Focus mainly on computational efficiency
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