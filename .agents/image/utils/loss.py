import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
class FeatureMetric(nn.Module):
    "马氏距离"
    def __init__(self, dim):
        super().__init__()
        self.L = nn.Parameter(torch.eye(dim) + 0.01 * torch.randn(dim, dim))
    def forward(self, x1, x2):
        M = self.L.T @ self.L  
        diff = x1 - x2  # [B * B, D]
        dist = torch.sum((diff @ M) * diff, dim=1)  # [B * B]
        return dist
def get_orthogonal_mask_loss(attention_masks):
    """
    强制不同的 attention mask 关注 DEX 图像的不同区域，避免注意力坍塌
    attention_masks shape: [B, num_masks, H, W]
    正交正则化强制这 32 个 Mask 必须关注图像的不同区域，像 32 个不同的“专家”一样，最大化特征的多样性，从而大幅提升少数类（恶意）的召回率（Recall）。
    正交掩码正则化损失 (Orthogonal Mask Regularization)
    """
    B, M, H, W = attention_masks.size()
    
    # 展平空间维度 -> [B, M, H*W]
    masks_flat = attention_masks.view(B, M, -1)
    
    # L2 归一化
    masks_norm = F.normalize(masks_flat, p=2, dim=-1)
    
    # 计算 M 个 mask 之间的余弦相似度矩阵 -> [B, M, M]
    sim_matrix = torch.bmm(masks_norm, masks_norm.transpose(1, 2))
    
    # 创建单位矩阵，我们希望非对角线元素（不同mask的相似度）尽可能趋近于 0
    identity = torch.eye(M, device=attention_masks.device).unsqueeze(0).expand(B, -1, -1)
    
    # 计算正交损失（只惩罚非对角线元素）
    ortho_loss = F.mse_loss(sim_matrix, identity)
    
    return ortho_loss
def get_feature_contrastive_loss(features, labels, feature_metric, feature_margin=0.4, pos_weight=1.0, neg_weight=5.0):
    
    B = features.size(0)
    labels = labels.view(-1, 1)
  
    pos_mask = (labels == labels.T).float()
    neg_mask = 1.0 - pos_mask
    diag = torch.eye(B, device=features.device)
    pos_mask -= diag
    neg_mask *= (1 - diag)
    
    feat1 = features.unsqueeze(1).expand(B, B, -1).reshape(-1, features.size(1))  # [B*B, D]
    feat2 = features.unsqueeze(0).expand(B, B, -1).reshape(-1, features.size(1))  # [B*B, D]
 
    dists = feature_metric(feat1, feat2).reshape(B, B)  # [B, B]
    pos_loss = dists * pos_mask
    pos_loss = pos_loss.sum() / (pos_mask.sum() + 1e-8)
    # 这是特征图像的马氏距离对比损失，注意和流模型的对比损失概念区分开
    neg_margin = torch.clamp(feature_margin - dists, min=0.0)
    neg_loss = neg_margin * neg_mask
    neg_loss = neg_loss.sum() / (neg_mask.sum() + 1e-8)
    loss = pos_weight * pos_loss + neg_weight * neg_loss
    return loss
def get_flow_contrastive_loss(z_gc, log_det, mu_gc, log_var_gc, labels, class_prior=None, 
                                flow_margin=1.0):
    """
    结合异方差高斯流 (Heteroscedastic Gaussian) 与 自适应边界 (LDAM) 的强化版对比流损失
    """
    B, D = z_gc.size()
    K = mu_gc.size(0)
    
    # ==========================================
    # 1: 异方差马氏距离计算
    # ==========================================
    z_expand = z_gc.unsqueeze(1)         # [B, 1, D]
    mu_expand = mu_gc.unsqueeze(0)       # [1, K, D]
    
    # 限制 log_var_gc 的范围，防止训练初期方差极小导致除以0，或极大导致NaN (数值稳定保护)
    var_expand = torch.exp(torch.clamp(log_var_gc, min=-4, max=2)).unsqueeze(0)  # [1, K, D]
    
    # 计算马氏距离平方: (z - mu)^2 / var
    # [B, K, D] -> sum -> [B, K]
    dist_sq_mahalanobis = torch.sum(((z_expand - mu_expand) ** 2) / var_expand, dim=-1)
    
    # ==========================================
    # 2. 异方差 NLL 生成损失
    # ==========================================
    # 高斯分布的对数概率 PDF，需要加上协方差的对数行列式惩罚项
    log_det_var = torch.sum(torch.clamp(log_var_gc, min=-4, max=2), dim=-1).unsqueeze(0) # [1, K]
    
    # log p(z|y) ∝ -0.5 * (D_M^2 + log|Var|)
    log_p_z_given_y_all = -0.5 * (dist_sq_mahalanobis + log_det_var) # [B, K]
    
    loss_latent = F.cross_entropy(log_p_z_given_y_all, labels)  # 交叉熵损失，监督学习的核心目标：在隐空间中分得开类1和类2
    if class_prior is None:
        log_prior = torch.zeros(K, device=z_gc.device)
    else:
        log_prior = class_prior.to(z_gc.device)
        
    # 联合对数概率 log p(x, y) = log(x|y) + log p(y) = log p(z|y) + log|det(J)| + log p(y)
    log_p_xy_all = log_p_z_given_y_all + log_det.unsqueeze(1) + log_prior.unsqueeze(0)
    
    # 提取真实类别的联合对数概率
    log_p_xy_true = log_p_xy_all.gather(1, labels.unsqueeze(1)).squeeze(1)
    # nll损失，负对数似然，越大越好，所以取负号
    loss_nll = -log_p_xy_true.mean()
    adaptive_margins = torch.full((K,), flow_margin, device=z_gc.device)
        
    # 获取当前批次每个样本对应正确类别的 margin [B]
    batch_margins = adaptive_margins.gather(0, labels)
    
    # 提取正样本与最难负样本距离 (基于马氏距离)
    pos_dist_sq = dist_sq_mahalanobis.gather(1, labels.unsqueeze(1)).squeeze(1) 
    
    # 构建负样本
    mask = torch.zeros_like(dist_sq_mahalanobis).scatter_(1, labels.unsqueeze(1), 1.0).bool()
    neg_dist_sq = dist_sq_mahalanobis.masked_fill(mask, float('inf')) 
    hardest_neg_dist_sq, _ = neg_dist_sq.min(dim=1)  # [B]
    
    # 总的对比损失
    loss_contrast = F.relu(pos_dist_sq - hardest_neg_dist_sq + batch_margins).mean()
    # ==========================================
    # 损失汇总
    # ==========================================
    total_loss = loss_nll +loss_contrast
    
    # 返回分离的loss，方便用wandb监控
    return loss_latent, loss_nll, loss_contrast
