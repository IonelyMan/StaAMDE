from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ConvNeXtV2Config:
    depths: List[int]
    dims: List[int]


CONVNEXTV2_CONFIGS: Dict[str, ConvNeXtV2Config] = {
    "atto": ConvNeXtV2Config(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320]),
    "femto": ConvNeXtV2Config(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384]),
    "pico": ConvNeXtV2Config(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512]),
    "nano": ConvNeXtV2Config(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640]),
    "tiny": ConvNeXtV2Config(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]),
    "base": ConvNeXtV2Config(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]),
    "large": ConvNeXtV2Config(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536]),
}


def convnextv2_names() -> List[str]:
    return [f"convnextv2_{name}" for name in CONVNEXTV2_CONFIGS]


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class GRN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


class ConvNeXtV2(nn.Module):
    def __init__(
        self,
        depths: List[int],
        dims: List[int],
        num_classes: int = 2,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for index in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[index], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[index], dims[index + 1], kernel_size=2, stride=2),
                )
            )

        self.stages = nn.ModuleList()
        drop_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cursor = 0
        for stage_index in range(4):
            blocks = [
                ConvNeXtV2Block(dims[stage_index], drop_path=drop_rates[cursor + block_index])
                for block_index in range(depths[stage_index])
            ]
            cursor += depths[stage_index]
            self.stages.append(nn.Sequential(*blocks))

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward_features(self, x):
        for index in range(4):
            x = self.downsample_layers[index](x)
            x = self.stages[index](x)
        return self.norm(x.mean(dim=(-2, -1)))

    def forward(self, x):
        return self.head(self.forward_features(x))


def build_convnextv2(
    arch: str,
    num_classes: int = 2,
    drop_path_rate: float = 0.1,
) -> ConvNeXtV2:
    name = arch.replace("convnextv2_", "")
    if name not in CONVNEXTV2_CONFIGS:
        valid = ", ".join(convnextv2_names())
        raise ValueError(f"未知 ConvNeXt V2 模型: {arch}，可选: {valid}")
    config = CONVNEXTV2_CONFIGS[name]
    return ConvNeXtV2(
        depths=list(config.depths),
        dims=list(config.dims),
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )
