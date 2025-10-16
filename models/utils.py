"""
Utility Modules for C3Net Architecture .

This file consolidates fundamental building blocks used across different
components of the C3Net model, promoting code reuse and consistency.
It includes:
- ConvGNReLU: Standard Convolution -> Group Normalization -> ReLU block.
- ECABlock: Efficient Channel Attention module.
- AttentionGate: Standard Attention Gate mechanism for feature modulation.
- DySample: The original learnable Dynamic Upsampling module (Liu et al., ICCV 2023),
            chosen based on experimental results showing a slight advantage over
            fixed interpolation for the C3Net baseline.

Author  : Baber Jan
Date    : 09-05-2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Type # Added Type
import logging
import math

# Initialize logger for this module
logger = logging.getLogger(__name__)

# --- Common Convolutional Block ---
class ConvGNReLU(nn.Sequential):
    """
    A standard sequential block: Convolution -> Group Normalization -> ReLU.

    Uses GroupNorm with 32 groups when possible, otherwise 1 group.
    ReLU activation is inplace.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel. Default: 3.
        stride (int): Stride of the convolution. Default: 1.
        padding (int): Padding added to the input. Default: 1 (for kernel_size=3).
        dilation (int): Spacing between kernel elements. Default: 1.
        groups (int): Convolution groups. Default: 1 (standard convolution).
        bias (bool): Whether to include bias in convolution. Default: False.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: Optional[int] = None, # Default padding based on kernel_size
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = False):

        # --- Validate Channels ---
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError(f"'in_channels' must be a positive integer, got {in_channels}.")
        if not isinstance(out_channels, int) or out_channels <= 0:
             raise ValueError(f"'out_channels' must be a positive integer, got {out_channels}.")

        # Default padding to keep spatial size same for stride=1, kernel=3
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        # --- Determine Number of Groups for GroupNorm ---
        # Use 32 groups if channels >= 32 and divisible by 32, otherwise 1 group.
        if out_channels >= 32 and out_channels % 32 == 0:
            num_groups_norm = 32
        else:
            num_groups_norm = 1

        # --- Define the Sequential Layers ---
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.GroupNorm(num_groups=num_groups_norm, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

# --- Efficient Channel Attention (ECA) Block ---
class ECABlock(nn.Module):
    """
    Efficient Channel Attention Block (ECA-Net, CVPR 2020).

    Captures local cross-channel interactions efficiently using an adaptive 1D convolution.
    Kernel size `k` is adaptively determined based on channel dimension `C` by default.

    Args:
        channels (int): Number of input channels.
        gamma (int): Mapping parameter for adaptive kernel size. Default: 2.
        b (int): Mapping parameter for adaptive kernel size. Default: 1.
        use_adaptive_kernel (bool): If True, adapt kernel size `k` based on `channels`.
                                     If False, use fixed `kernel_size`. Default: True.
        kernel_size (int): Fixed kernel size if `use_adaptive_kernel` is False. Must be odd. Default: 3.
    """
    def __init__(self,
                 channels: int,
                 gamma: int = 2,
                 b: int = 1,
                 use_adaptive_kernel: bool = True,
                 kernel_size: int = 3):
        super().__init__()
        if not isinstance(channels, int) or channels <= 0:
            raise ValueError(f"'channels' must be a positive integer, got {channels}.")
        self.channels = channels

        # --- Determine Kernel Size (k) ---
        if use_adaptive_kernel:
            # k = |log2(C)/gamma + b/gamma|_odd
            t = int(abs((math.log2(channels) + b) / gamma))
            k = t if t % 2 else t + 1 # Ensure k is odd
        else:
            k = kernel_size
        if k % 2 == 0 or k <= 0: # Ensure positive odd
            raise ValueError(f"ECA Block kernel size k ({k}) must be odd and positive.")

        # --- Layers ---
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # Global Average Pooling (B, C, 1, 1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False) # 1D Conv across channels
        self.sigmoid = nn.Sigmoid() # Channel weights [0, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply ECA attention. Input: (B, C, H, W), Output: (B, C, H, W). """
        b, c, h, w = x.size()
        y = self.avg_pool(x) # (B, C, 1, 1)
        # Reshape and apply 1D conv
        y = self.conv1d(y.squeeze(-1).transpose(-1, -2)) # (B, C, 1) -> (B, 1, C) -> (B, 1, C)
        # Reshape back and apply sigmoid
        y = y.transpose(-1, -2).unsqueeze(-1) # (B, 1, C) -> (B, C, 1) -> (B, C, 1, 1)
        channel_attention_weights = self.sigmoid(y)
        # Apply attention
        return x * channel_attention_weights # Broadcasting: (B, C, H, W) * (B, C, 1, 1)

# --- Attention Gate Module (Included as a standard utility) ---
class AttentionGate(nn.Module):
    """
    Attention Gate mechanism (Attention U-Net, MIDL 2018 style).

    Modulates input features 'x' based on a gating signal 'g' from potentially
    a different source but same spatial resolution.

    Args:
        F_x (int): Number of channels in the input feature map 'x'.
        F_g (int): Number of channels in the gating signal 'g'.
        F_int (int): Intermediate channel dimension. Defaults to max(1, F_x // 2).
        norm_layer (Type[nn.Module]): Normalization layer. Default: nn.GroupNorm.
    """
    def __init__(self,
                 F_x: int,
                 F_g: int,
                 F_int: Optional[int] = None,
                 norm_layer: Type[nn.Module] = nn.GroupNorm): # Changed Type hint
        super().__init__()
        if F_int is None: F_int = max(1, F_x // 2)
        if not all(isinstance(f, int) and f > 0 for f in [F_x, F_g, F_int]):
            raise ValueError("F_x, F_g, F_int must be positive integers.")

        self.F_int = F_int

        # Learnable projections for x and g
        self.W_x = nn.Conv2d(F_x, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True)

        # Final convolution to get attention coefficients
        self.psi = nn.Sequential(
            norm_layer(num_groups=32 if F_int >= 32 and F_int % 32 == 0 else 1, num_channels=F_int),
            nn.ReLU(inplace=True),
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid() # Attention coefficients alpha [0, 1]
        )

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """ Apply attention gating. Input x: (B, F_x, H, W), g: (B, F_g, H, W). Output: (B, F_x, H, W)."""
        # Check spatial dims - ideally they should match for this AG variant
        if x.shape[-2:] != g.shape[-2:]:
             logger.warning(f"AG spatial dimensions mismatch: x={x.shape}, g={g.shape}. Resizing 'g'.")
             g = F.interpolate(g, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # Project x and g, sum, pass through psi to get alpha
        theta_x = self.W_x(x) # (B, F_int, H, W)
        phi_g = self.W_g(g)   # (B, F_int, H, W)
        combined = theta_x + phi_g
        alpha = self.psi(combined) # (B, 1, H, W)

        # Modulate input x
        return alpha * x # (B, F_x, H, W)

# --- Dynamic Upsampling Module (DySample) ---
# Original learnable dynamic upsampling module (Liu et al., ICCV 2023).
# This module learns to upsample features dynamically based on the input.
def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

