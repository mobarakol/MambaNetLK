"""
    CFormer (Context-aware Transformer) for Point Cloud Processing
    
    A context-aware transformer architecture for point cloud feature extraction.
    Combines local self-attention with global cross-attention mechanisms.
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


class ContextAwarePositionEncoding(nn.Module):
    """Context-aware positional encoding for point clouds"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # 局部位置编码投影层
        self.local_pos_proj = nn.Linear(3, d_model)
        
        # 全局位置编码投影层
        self.global_pos_proj = nn.Linear(3, d_model)
        
    def forward(self, points, is_global=False):
        """
        Args:
            points: [B, N, 3] 点云坐标
            is_global: 是否用于全局上下文
        Returns:
            pos_encoding: [B, N, d_model] 位置编码
        """
        if is_global:
            # 全局位置编码 - 强调点之间的长距离关系
            pos_encoding = self.global_pos_proj(points)
        else:
            # 局部位置编码 - 关注近邻点的几何结构
            pos_encoding = self.local_pos_proj(points)
            
        return pos_encoding


class LocalSelfAttention(nn.Module):
    """Local self-attention mechanism"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.scale = math.sqrt(d_model)
        
        # Q, K, V投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: [B, N, d_model] 输入特征
        Returns:
            output: [B, N, d_model] 注意力输出
        """
        # 残差连接输入
        residual = x
        
        # 投影Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 计算注意力权重
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力
        out = torch.matmul(attn, v)
        out = self.out_proj(out)
        
        # 残差连接和层归一化
        out = self.layer_norm(residual + out)
        
        return out


class GlobalCrossAttention(nn.Module):
    """Global cross-attention with learnable proxy points"""
    def __init__(self, d_model, num_proxy_points=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_proxy_points = num_proxy_points
        self.scale = math.sqrt(d_model)
        
        # 代理点初始化 - 可学习参数
        self.proxy_points = nn.Parameter(torch.randn(1, num_proxy_points, d_model))
        
        # 收集过程投影
        self.collect_q_proj = nn.Linear(d_model, d_model)
        self.collect_k_proj = nn.Linear(d_model, d_model)
        self.collect_v_proj = nn.Linear(d_model, d_model)
        
        # 分发过程投影
        self.distribute_q_proj = nn.Linear(d_model, d_model)
        self.distribute_k_proj = nn.Linear(d_model, d_model)
        self.distribute_v_proj = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: [B, N, d_model] 输入特征
        Returns:
            output: [B, N, d_model] 注意力输出
        """
        batch_size = x.size(0)
        residual = x
        
        # 准备代理点
        proxy = self.proxy_points.expand(batch_size, -1, -1)  # [B, P, d_model]
        
        # 收集过程 - 局部特征聚合到代理点
        collect_q = self.collect_q_proj(proxy)                # [B, P, d_model]
        collect_k = self.collect_k_proj(x)                    # [B, N, d_model]
        collect_v = self.collect_v_proj(x)                    # [B, N, d_model]
        
        collect_attn = torch.matmul(collect_q, collect_k.transpose(-2, -1)) / self.scale
        collect_attn = F.softmax(collect_attn, dim=-1)
        collect_attn = self.dropout(collect_attn)
        
        # 应用收集注意力，更新代理点特征
        proxy_updated = torch.matmul(collect_attn, collect_v)  # [B, P, d_model]
        
        # 分发过程 - 代理点分发全局信息到局部点
        distribute_q = self.distribute_q_proj(x)                     # [B, N, d_model]
        distribute_k = self.distribute_k_proj(proxy_updated)         # [B, P, d_model]
        distribute_v = self.distribute_v_proj(proxy_updated)         # [B, P, d_model]
        
        distribute_attn = torch.matmul(distribute_q, distribute_k.transpose(-2, -1)) / self.scale
        distribute_attn = F.softmax(distribute_attn, dim=-1)
        distribute_attn = self.dropout(distribute_attn)
        
        # 应用分发注意力
        out = torch.matmul(distribute_attn, distribute_v)
        out = self.out_proj(out)
        
        # 残差连接和层归一化
        out = self.layer_norm(residual + out)
        
        return out


def symfn_max(x):
    """Max pooling symmetry function"""
    return torch.max(x, dim=1)[0]


def symfn_avg(x):
    """Average pooling symmetry function"""
    return torch.mean(x, dim=1)


def symfn_cd_pool(x):
    """Context-aware pooling with learnable aggregation"""
    batch_size, num_points, dim = x.size()
    
    # 全局平均作为全局上下文
    global_context = torch.mean(x, dim=1, keepdim=True)  # [B, 1, K]
    
    # 计算点与全局上下文的相似度作为权重
    attn_weights = torch.sum(x * global_context, dim=-1)  # [B, N]
    attn_weights = F.softmax(attn_weights, dim=-1).unsqueeze(-1)  # [B, N, 1]
    
    # 加权聚合
    pooled = torch.sum(x * attn_weights, dim=1)  # [B, K]
    return pooled


# 使symbfn_cd_pool成为symbfn_max的别名，与训练脚本中的'cd_pool'参数兼容
symbfn_cd_pool = symfn_max


class CFormer_features(torch.nn.Module):
    """
    CFormer feature extractor
    Context-aware transformer for point cloud processing
    """
    def __init__(self, dim_k=1024, sym_fn=symfn_max, scale=1, num_proxy_points=8, num_blocks=2):
        super().__init__()
        
        # 特征维度设置
        self.d_model = int(128 / scale)  # 隐藏层特征维度
        self.dim_k = int(dim_k / scale)  # 输出特征维度
        self.num_proxy_points = num_proxy_points  # 代理点数量
        self.num_blocks = num_blocks  # 块数量
        
        # 输入嵌入层：3D坐标 -> 高维特征空间
        self.input_projection = nn.Linear(3, self.d_model)
        
        # 上下文感知位置编码
        self.position_encoding = ContextAwarePositionEncoding(self.d_model)
        
        # 局部特征提取层
        self.local_blocks = nn.ModuleList([
            LocalSelfAttention(self.d_model)
            for _ in range(num_blocks)
        ])
        
        # 全局信息交互层
        self.global_block = GlobalCrossAttention(self.d_model, num_proxy_points)
        
        # 特征变换层：从注意力维度映射到输出维度
        self.feature_transform = MLPNet(
            self.d_model, 
            [int(256 / scale), self.dim_k], 
            b_shared=True  # 使用Conv1d层
        )
        
        # 聚合函数
        self.sy = sym_fn
        
        # 保持与PointNet_features兼容的属性
        self.t_out_t2 = None
        self.t_out_h1 = None

    def forward(self, points):
        """
        前向传播
        
        Args:
            points: [B, N, 3] 输入点云
            
        Returns:
            features: [B, K] 全局特征向量
        """
        batch_size, num_points, _ = points.size()
        
        # 输入投影：3D坐标 -> 高维特征
        x = self.input_projection(points)  # [B, N, d_model]
        
        # 添加局部位置编码
        local_pos_encoding = self.position_encoding(points, is_global=False)
        x = x + local_pos_encoding
        
        # 保存中间特征（兼容性）
        self.t_out_h1 = x.transpose(1, 2)  # [B, d_model, N] 格式用于兼容
        
        # 局部特征提取 - 局部自注意力
        for local_block in self.local_blocks:
            x = local_block(x)  # [B, N, d_model]
        
        # 全局信息交互 - 收集分发机制
        x = self.global_block(x)  # [B, N, d_model]
        
        # 特征变换
        # 转换为 [B, d_model, N] 格式用于Conv1d处理
        x = x.transpose(1, 2)  # [B, d_model, N]
        x = self.feature_transform(x)  # [B, dim_k, N]
        
        # 转回 [B, N, dim_k] 格式进行聚合
        x = x.transpose(1, 2)  # [B, N, dim_k]
        
        # 全局聚合
        if self.sy == symfn_max:
            global_features = symfn_max(x)  # [B, dim_k]
        elif self.sy == symfn_avg:
            global_features = symfn_avg(x)  # [B, dim_k]
        elif self.sy == symfn_cd_pool:
            global_features = symfn_cd_pool(x)  # [B, dim_k]
        else:
            global_features = symfn_max(x)  # 默认使用最大池化
        
        return global_features


class CFormer_classifier(torch.nn.Module):
    """
    CFormer classifier
    """
    def __init__(self, num_c, cfeat, dim_k):
        """
        Args:
            num_c: 分类数量
            cfeat: CFormer_features实例
            dim_k: 特征维度
        """
        super().__init__()
        self.features = cfeat
        
        # 分类头：特征向量 -> 分类结果
        list_layers = mlp_layers(dim_k, [512, 256], b_shared=False, bn_momentum=0.1, dropout=0.0)
        list_layers.append(torch.nn.Linear(256, num_c))
        self.classifier = torch.nn.Sequential(*list_layers)

    def forward(self, points):
        """
        前向传播
        
        Args:
            points: [B, N, 3] 输入点云
            
        Returns:
            out: [B, num_c] 分类输出
        """
        feat = self.features(points)  # [B, dim_k]
        out = self.classifier(feat)   # [B, num_c]
        return out

    def loss(self, out, target, w=0.001):
        """
        计算损失函数
        
        Args:
            out: [B, num_c] 分类输出
            target: [B] 真实标签
            w: 正则化权重
            
        Returns:
            loss: 总损失
        """
        # 分类损失
        loss_c = torch.nn.functional.nll_loss(
            torch.nn.functional.log_softmax(out, dim=1), target, size_average=False)

        # 收集分发模型不需要变换矩阵正则化
        t2 = self.features.t_out_t2
        if (t2 is None) or (w == 0):
            return loss_c

        # 如果存在变换矩阵，添加正则化项
        batch = t2.size(0)
        K = t2.size(1)  # [B, K, K]
        I = torch.eye(K).repeat(batch, 1, 1).to(t2)
        A = t2.bmm(t2.transpose(1, 2))
        loss_m = torch.nn.functional.mse_loss(A, I, size_average=False)
        loss = loss_c + w * loss_m
        return loss


# EOF 