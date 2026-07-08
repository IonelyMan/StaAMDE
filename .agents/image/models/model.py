import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn import MultiheadAttention
from torchvision.models import resnet18
from models.inception import inception_v3
from models.attention import SelfAttentionLayer
from models.normalflow import NormalFlow

import os
import sys
parent_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

class SequenceStripPooling(nn.Module):
    """
    针对 1D 字节流折叠为 2D 图像的特性，提取水平长距离的连续指令特征
    """
    def __init__(self, in_channels):
        super(SequenceStripPooling, self).__init__()
        # 1xN 卷积，专门捕获横向（连续字节）的特征
        self.horizontal_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 5), padding=(0, 2))
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # 提取连续水平指令特征
        out = self.horizontal_conv(x)
        out = self.bn(out)
        out = self.act(out)
        # 与原 2D 宏观特征形成残差组合
        return x + out


class FGP(nn.Module):
    """
        Fine grained Pooling模块
        常用于细粒度分类
    """
    def __init__(self, pool=None):
        super(FGP, self).__init__()
        self.pool = pool
        if pool is not None:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        # attention 和 feature 尺寸必须一致
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        if self.pool is None:
            # 每个 attention map = 一个“局部专家”
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W))
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + 1e-12 )

        feature_matrix = F.normalize(feature_matrix, dim=-1)
        return feature_matrix

class CoordAttentionMask(nn.Module):
    """
    专为二进制折叠图像设计的坐标感知掩码生成器
    利用 X 和 Y 方向的独立池化，捕获 DEX 字节流的绝对物理偏移特征
    CoordMask 解决了 DEX 转 RGB 后一维语义丢失的问题，让网络能精确捕捉行列结构信息
    """
    def __init__(self, in_channels, num_masks, reduction=32):
        super(CoordAttentionMask, self).__init__()
        self.num_masks = num_masks
        
        # X 和 Y 方向的 1D 自适应池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()  # SiLU 在这类任务中通常比 ReLU 表现更好

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        
        # 最终映射到 num_masks 通道
        self.mask_proj = nn.Sequential(
            nn.Conv2d(in_channels, num_masks, kernel_size=1),
            nn.Softplus() # 使用 Softplus 代替 ReLU，保证梯度的平滑性，避免死神经元
        )


    def forward(self, x):
        n, c, h, w = x.size()
        
        # 1. 沿行列提取 1D 物理分布特征
        x_h = self.pool_h(x)  # [N, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [N, C, W, 1]

        # 2. 拼接并进行特征交互
        y = torch.cat([x_h, x_w], dim=2)  # [N, C, H+W, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 

        # 3. 分离回 X 和 Y 方向
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 4. 生成坐标注意力权重
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        # 5. 将坐标权重施加到原特征图上
        out = x * a_h * a_w
        
        # 6. 生成最终的 num_masks
        masks = self.mask_proj(out)
        return masks

class IDDCNF(nn.Module):
    "imbalanced dataset detection using contrastive normalizing flow"
    def __init__(self, num_classes, num_masks=32, num_attn_layers=3,latent_input_dim=32):
        super(IDDCNF, self).__init__()
        self.num_classes = num_classes
        # 注意力图个数
        self.num_masks = num_masks
        self.num_atten_layer = num_attn_layers
        self.latent_input_dim = latent_input_dim
        # 原始图像特征提取器
        inception_model = inception_v3(pretrained=True)
        # 一个方法同时返回深浅两层特征
        self.stage1, self.stage2 = inception_model.get_features_multiscale()
        
        # Mixed_5d 有 288 通道, Mixed_6e 有 768 通道
        # 合并后通道数
        self.num_features = 288 + 768 #1056
        
        # 将深浅层特征映射到一个统一尺寸的对齐层（利用自适应池化对齐空间分辨率）
        self.align_pool = nn.AdaptiveAvgPool2d((14, 14)) # 假设对齐到 14x14

        # 注意力层
        self.atten_layer = nn.ModuleList(
            [
                SelfAttentionLayer(model_dim=self.num_features, feed_forward_dim=2048, num_heads=8, dropout=0.1, mask=False)
                for _ in range(num_attn_layers)
            ]
        )

        # 水平条带池化（Horizontal Strip Pooling），专门用于捕获长距离的连续指令依赖。
        self.strip_pool = SequenceStripPooling(self.num_features)

        # 注意力掩码生成器
        self.amg = CoordAttentionMask(self.num_features, self.num_masks)

        # 细粒度提取模块
        self.fgp = FGP()

        # 类别token信息，用于后续分类条件
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.num_features))  # [1, 1, 1056]
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_masks + 1, self.num_features))  # [1, 33, 1056]
        
        self.feature_proj = nn.Sequential(
            nn.Linear(self.num_features*(self.num_masks+1), 8192),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 512),
            nn.LayerNorm(512)
        )

        # 学习不同类别的对数方差 log_var_gc,因为数据集不平衡，加入这个是有必要的
        self.log_var_gc = nn.Parameter(torch.zeros(num_classes, self.latent_input_dim))
        # Gaussian 模型的均值，[num_classes, gc_dim]，num_classes个高斯组件，混合高斯分布, 学习不同类别的均值
        self.mu_gc = nn.Parameter(torch.randn(num_classes, self.latent_input_dim))
        
        # self.cond_proj = nn.Sequential(
        #     nn.Linear(1056, 256),
        #     nn.ReLU(),
        #     nn.Linear(256,latent_input_dim)
        # )
        
        self.gc_proj = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.latent_input_dim),
            nn.LayerNorm(self.latent_input_dim)
        )
        self.normalflow = NormalFlow(
            dim=self.latent_input_dim, # 根据后续修改
            hidden_dim=512,
            n_blocks=2,
            clamp=2.,
            act_norm=1.,
            act_norm_type='SOFTPLUS',
            permute_soft=True
        )
        nn.init.normal_(self.mu_gc, mean=0.0, std=0.02)

    def forward(self, x):
        batch_size = x.size(0)
        # 对图像进行卷积特征提取
        # print(f'x shape: {x.shape}')
        # [修改处] 提取多尺度特征
        feat_shallow = self.stage1(x)  # [B, 288, 53, 53]
        feat_deep = self.stage2(feat_shallow) # [B, 768, 26, 26]
        
        # 空间分辨率对齐
        feat_shallow_aligned = self.align_pool(feat_shallow) # [B, 288, 14, 14]
        feat_deep_aligned = self.align_pool(feat_deep) # [B, 768, 14, 14]
        
        # 沿着通道维度拼接，融合微观指令级特征与宏观结构级特征
        combined_features = torch.cat([feat_shallow_aligned, feat_deep_aligned], dim=1) # [B, 1056, 14, 14]

        # [新增] 专门强化水平维度的序列字节特征
        enhanced_features = self.strip_pool(combined_features) # [B, 1056, 14, 14]
        # 后续的 amg (注意力生成器) 和 fgp 等不需要改，因为 self.num_features 已经更新为 1056
        attention_maps = self.amg(enhanced_features) # [B,32,14,14]

        # 聚合“看到了什么”
        fg_features = self.fgp(enhanced_features, attention_maps) # [B,32,1056]

        # 类别token信息，后续作为条件流的输入
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, 1056]
        # 特征与类别的嵌入表示
        embeddings = torch.cat((fg_features, cls_tokens), dim=1)  # [B, 33, 1056]
        embeddings = embeddings + self.pos_embedding # (B,33,1056)
        cls_features = embeddings[:, -1, :]  # [B, 1056]
        
        for attn in self.atten_layer:
            embeddings = attn(embeddings)

        flattended_features = embeddings.view(batch_size, -1) # [B, 25344] # 固定，不论你如何改变img_size
        # print(f"complex_features shape: {complex_features.shape}")
        proj_features = self.feature_proj(flattended_features)  # [B, 512]

        # z_gc[B, 512] 是 经过标准Flow 变换后的 latent 表示，Gaussian Conditional,条件流到高斯混合模型
        # log_det是每个样本对应的 log-determinant，用于 计算 log-likelihood
        z_gc_input = self.gc_proj(proj_features) # [B, self.latent_input_dim]
        
        # cond = self.cond_proj(cls_features)
        # flow_input = torch.cat(
        #     [z_gc_input, cond],
        #     dim=1
        # )
        z_gc, log_det = self.normalflow(z_gc_input)

        # 其他不变
        return cls_features,attention_maps, z_gc, log_det