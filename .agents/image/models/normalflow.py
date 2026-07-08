import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import special_ortho_group
import numpy as np
from math import exp
class GlowCouplingBlock(nn.Module):
    def __init__(self, dims_in, dims_c=[], subnet_constructor=None, clamp=2.):
        super().__init__()
        channels = dims_in[0][0]
        if dims_c:
            raise ValueError('does not support conditioning yet')
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2
        self.splits = [self.split_len1, self.split_len2]
        self.in_channels = channels
        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)
        self.conditional = False
        self.s1 = subnet_constructor(self.split_len1, 2 * self.split_len2)
        self.s2 = subnet_constructor(self.split_len2, 2 * self.split_len1)
        self.last_jac = None
    def log_e(self, s):
        return self.clamp * torch.tanh(0.2 * s)
    def affine(self, x, a, rev=False):
        ch = x.shape[1]
        sub_jac = self.log_e(a[:,:ch])
        if not rev:
            return (x * torch.exp(sub_jac) + a[:,ch:],
                    torch.sum(sub_jac, dim=(1,2,3)))
        else:
            return ((x - a[:,ch:]) * torch.exp(-sub_jac),
                    -torch.sum(sub_jac, dim=(1,2,3)))
    def forward(self, x, c=[], rev=False):
        x1, x2 = torch.split(x[0], self.splits, dim=1)
        if not rev:
            a1 = self.s1(x1)
            y2, j2 = self.affine(x2, a1)
            a2 = self.s2(y2)
            y1, j1 = self.affine(x1, a2)
        else: # names of x and y are swapped!
            a2 = self.s2(x2)
            y1, j1 = self.affine(x1, a2, rev=True)
            a1 = self.s1(y1)
            y2, j2 = self.affine(x2, a1, rev=True)
        self.last_jac = j1 + j2
        return [torch.cat((y1, y2), 1)]
    def jacobian(self, x, c=[], rev=False):
        return self.last_jac
    def output_dims(self, input_dims):
        return input_dims
class AIO_GlowCouplingBlock(GlowCouplingBlock):
    "基于 Glow 架构的一种 affine coupling layer（仿射耦合层） 的实现版本"
    def __init__(self,
        dims_in,
        dims_c=[],
        subnet_constructor=None,
        clamp=2.,
        act_norm=1.,
        act_norm_type='SOFTPLUS',
        permute_soft=False
    ):
        super().__init__(dims_in, dims_c=dims_c, subnet_constructor=subnet_constructor, clamp=clamp)
        if act_norm_type == 'SIGMOID':
            act_norm = np.log(act_norm)
            self.actnorm_activation = (lambda a: 10 * torch.sigmoid(a - 2.))
        elif act_norm_type == 'SOFTPLUS':
            act_norm = 10. * act_norm
            self.softplus = nn.Softplus(beta=0.5)
            self.actnorm_activation = (lambda a: 0.1 * self.softplus(a))
        elif act_norm_type == 'EXP':
            act_norm = np.log(act_norm)
            self.actnorm_activation = (lambda a: torch.exp(a))
        else:
            raise ValueError('Please, SIGMOID, SOFTPLUS or EXP, as actnorm type')
        assert act_norm > 0., "please, this is not allowed. don't do it. take it... and go."
        channels = self.in_channels
        self.act_norm = nn.Parameter(torch.ones(1, channels, 1, 1) * float(act_norm))
        self.act_offset = nn.Parameter(torch.zeros(1, channels, 1, 1))
        if permute_soft:
            w = special_ortho_group.rvs(channels)
        else:
            w = np.zeros((channels,channels))
            for i,j in enumerate(np.random.permutation(channels)):
                w[i,j] = 1.
        w_inv = w.T
        self.w = nn.Parameter(torch.FloatTensor(w).view(channels, channels, 1, 1), requires_grad=False)
        self.w_inv = nn.Parameter(torch.FloatTensor(w_inv).view(channels, channels, 1, 1), requires_grad=False)
    def permute(self, x, rev=False):
        scale = self.actnorm_activation( self.act_norm)
        if rev:
            return (F.conv2d(x, self.w_inv) - self.act_offset) / scale
        else:
            return F.conv2d(x * scale + self.act_offset, self.w)
    def forward(self, x, c=[], rev=False):
        if rev:
            x = [self.permute(x[0], rev=True)]
        x_out = super().forward(x, c=[], rev=rev)[0]
        if not rev:
            x_out = self.permute(x_out, rev=False)
        n_pixels = x_out.shape[2] * x_out.shape[3]
        self.last_jac += ((-1)**rev * n_pixels) * (torch.log(self.actnorm_activation(self.act_norm) + 1e-12).sum())
        return [x_out]
if __name__ == '__main__':
    import numpy as np
    N = 8
    c = 32
    x = torch.FloatTensor(128, c, N, N)
    x.normal_(0,1)
    def constr(c_in, c_out):
        layer = torch.nn.Conv2d(c_in, c_out, 1)
        layer.weight.data *= 0.
        layer.bias.data *= 0.
        return layer
    actnorm = 5.26
    layer = AIO_GlowCouplingBlock(
        [(c, N, N)],
        subnet_constructor=constr,
        clamp=2.,
        act_norm=actnorm,
        permute_soft=True
    )
    transf = layer([x])
    jac = layer.jacobian([x])
    x_inv = layer(transf, rev=True)[0]
    err = torch.abs(x - x_inv)
    print(transf[0].shape)
    print(jac.mean().item(), np.log(actnorm) * x.numel() / 128)
    print(err.max().item())
    print(err.mean().item())
class NormalFlow(nn.Module):
    def __init__(self, dim=512, hidden_dim=512, n_blocks=2,
                 clamp=2., act_norm=1., act_norm_type='SOFTPLUS', permute_soft=True):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        """
            Glow模块 本质上是：
            在 RealNVP 的 affine coupling flow 基础上，引入 ActNorm 和可学习的 invertible 1×1 convolution
            从而显著增强 channel mixing 与生成能力的升级版 normalizing flow。
        """
        def subnet_constructor(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, hidden_dim, kernel_size=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, c_out, kernel_size=1, padding=0, bias=True),
            )
        self.blocks = nn.ModuleList([
            AIO_GlowCouplingBlock(
                dims_in=[(dim, 1, 1)],
                subnet_constructor=subnet_constructor,
                clamp=clamp,
                act_norm=act_norm,
                act_norm_type=act_norm_type,
                permute_soft=permute_soft
            )
            for _ in range(n_blocks)
        ])
    def forward(self, h, rev=False):
        """
            :params h：是特征向量,复杂的数据分布
            :returns:
                z: 是把复杂数据“拉直”之后的表示，让它服从简单分布（通常是高斯）,被 Flow “变换后”的 latent 表示
            它可以将输入数据 $x$ 映射到一个潜空间z
            并且能够无损地变换回原始输入。
        """
        # B=批次样本数，D：特征维度
        B, D = h.shape
        assert D == self.dim, f'Expected dim={self.dim}, got {D}'
        # 把每个特征看作 1x1 的“图像通道”。
        x = h.view(B, D, 1, 1) # 后续的block（通常是卷积层、1x1 卷积或 ActNorm 等）需要 4D 张量 [B, C, H, W]
        
        # 计算 log-likelihood 需要用到 Jacobian 的行列式
        # 用来累积 每个 block 的 log-det(Jacobian)
        log_det_total = 0.0 # 累积 每个 block 的 log-det(Jacobian)。每个样本有自己的 log-det。
        if not rev:
            for block in self.blocks:
                x = block([x], rev=False)[0]
                log_det_total = log_det_total + block.last_jac  # [B]
        else:
            for block in reversed(self.blocks):
                x = block([x], rev=True)[0]
                log_det_total = log_det_total + block.last_jac
        # 把 [B, D, 1, 1] 张量 reshape 回 [B, D]。这样输出的 latent 或重构后的数据维度和输入一致。
        # 返回的Z 是 正则化流“变换后的空间”
        z = x.view(B, D)
        return z, log_det_total
        # log p(x) = log p(z) + log_det_total
        # log p(z) = -0.5 * (z ** 2).sum(dim=1)
