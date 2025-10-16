"""
Implements the Edge Refinement Pathway (ERP) for the C3Net architecture.

This pathway specializes in extracting fine-grained boundary details and general
edge information critical for distinguishing camouflaged objects. It processes
early-stage, high-resolution feature maps (specifically Stage 1 and Stage 2)
extracted by the DINOv2 encoder backbone.

Key characteristics:
- Hierarchical Processing: Progressively fuses features from Stage 1 and 2
  while increasing spatial resolution.
- Detail Preservation: Utilizes shared upsampler instances designed to maintain fidelity.
- Advanced Feature Enhancement: Employs specialized `EdgeEnhanceBlock` modules
  that incorporate learnable convolutional filters *initialized* with Sobel
  (gradient) and Laplacian kernels, alongside standard convolutions and
  Efficient Channel Attention (ECA), to explicitly focus on and refine
  edge-related signals from a strong prior.

Input:
- f_s1 (torch.Tensor): Feature map from Encoder Stage 1 (B, D, Hf, Wf).
- f_s2 (torch.Tensor): Feature map from Encoder Stage 2 (B, D, Hf, Wf).
                       Expected to have the same dimensions as f_s1.
- shared_upsamplers (nn.ModuleDict): A dictionary containing pre-initialized,
  shared upsampler modules keyed by the string representation of their expected
  input channel dimension.

Output:
- F_edge_features (torch.Tensor): High-resolution feature map rich in boundary
  information (B, final_feature_channels, H_out, W_out). Typically H_out = Hf * 8,
  W_out = Wf * 8. This output is fed into the Final Fusion Head.

This branch is typically supervised during training via an auxiliary loss comparing
an intermediate prediction derived from its output to a ground truth edge map.


Author  : Baber Jan
Date    : 01-05-2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import logging

from models.utils import ConvGNReLU, ECABlock

# Initialize logger for this module
logger = logging.getLogger(__name__)

# --- Edge Enhancement Block ---
class EdgeEnhanceBlock(nn.Module):
    """
    Edge Enhancement Module (EEM) - Paper Section 3.2.1.

    Refined convolutional block designed to enhance edge-related features with
    gradient-initialized convolutions (Sobel and Laplacian kernels).

    This block integrates three distinct feature processing paths:
    1. Context Path (Path A): Uses depthwise separable convolutions for efficient
       capture of general spatial patterns.
    2. Gradient Path (Path B): Uses learnable 3x3 convolutions initialized with
       Sobel X and Y kernels applied per input channel, followed by normalization,
       activation, and a 1x1 refinement convolution.
    3. Laplacian Path (Path C): Uses a learnable 3x3 convolution initialized with
       a Laplacian kernel applied per input channel, followed by normalization
       and activation.

    These paths are combined, processed with Efficient Channel Attention (ECA)
    to re-weight channel importance, and added back to the input via a residual
    connection. The Sobel/Laplacian initializations provide a strong prior for
    edge detection, which the network can then fine-tune.

    Args:
        channels (int): Number of input and output channels for the block.
        grad_out_channels_ratio (int): Divisor determining the number of output
                                       channels for the *refined* gradient path
                                       (Path B) after the 1x1 convolution
                                       (channels // ratio). Default: 4.
    """
    def __init__(self, channels: int, grad_out_channels_ratio: int = 4):
        super().__init__()
        # --- Validate Input Channels ---
        if not isinstance(channels, int) or channels <= 0:
            raise ValueError(f"Input 'channels' must be a positive integer, got {channels}.")
        if not isinstance(grad_out_channels_ratio, int) or grad_out_channels_ratio <= 0:
             raise ValueError(f"'grad_out_channels_ratio' must be a positive integer, got {grad_out_channels_ratio}.")
        self.channels = channels # Store input/output channel dimension

        # --- Calculate Output Channels for Refined Gradient Path ---
        # Ensure refined grad path has at least 1 channel.
        grad_refined_channels = max(1, channels // grad_out_channels_ratio)

        # --- Define Kernels for Initialization ---
        # Sobel Gx Kernel (detects vertical edges)
        sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) # Shape (1, 1, 3, 3)
        # Sobel Gy Kernel (detects horizontal edges)
        sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) # Shape (1, 1, 3, 3)
        # Laplacian Kernel (detects general edges/corners) - Using a common 8-connectivity version
        laplacian_kernel = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) # Shape (1, 1, 3, 3)

        # --- Path A: Context Path (Depthwise Separable Conv) ---
        # Efficiently learns general spatial features. 
        self.path_a = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(max(1, channels // 32), channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(max(1, channels // 32), channels),
            nn.ReLU(inplace=True)
        )

        # --- Path B: Learned Gradient Path (Sobel Initialized) ---
        # 1. Base Convolution: Apply Sobel X/Y per channel. Output channels = C * 2.
        self.grad_conv = nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, groups=channels, bias=False)
        # Initialize weights: Stack Sobel X and Y for each input channel group.
        # Weight shape: (out_channels, in_channels // groups, kH, kW) = (C*2, C // C, 3, 3) = (C*2, 1, 3, 3)
        grad_weights = torch.zeros_like(self.grad_conv.weight)
        for i in range(channels):
            grad_weights[i*2, 0, :, :] = sobel_x_kernel      # Gx for channel i
            grad_weights[i*2 + 1, 0, :, :] = sobel_y_kernel  # Gy for channel i
        self.grad_conv.weight = nn.Parameter(grad_weights, requires_grad=True)
        # 2. Normalization and Activation for the raw gradient features.
        # Use GroupNorm suitable for C*2 channels.
        self.grad_norm_relu = nn.Sequential(
            nn.GroupNorm(max(1, (channels * 2) // 32), channels * 2),
            nn.ReLU(inplace=True)
        )
        # 3. Refinement Convolution: 1x1 Conv to refine/mix the per-channel Gx/Gy features.
        # Output channels = grad_refined_channels.
        self.grad_refine = ConvGNReLU(channels * 2, grad_refined_channels, kernel_size=1, padding=0)

        # --- Path C: Learned Laplacian Path (Laplacian Initialized) ---
        # 1. Base Convolution: Apply Laplacian per channel. Output channels = C.
        self.laplacian_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        # Initialize weights: Repeat Laplacian kernel for each input channel group.
        # Weight shape: (out_channels, in_channels // groups, kH, kW) = (C, C // C, 3, 3) = (C, 1, 3, 3)
        lap_weights = laplacian_kernel.repeat(channels, 1, 1, 1) # Shape (C, 1, 3, 3)
        self.laplacian_conv.weight = nn.Parameter(lap_weights, requires_grad=True)
        # 2. Normalization and Activation for Laplacian features.
        self.laplacian_norm_relu = nn.Sequential(
            nn.GroupNorm(max(1, channels // 32), channels),
            nn.ReLU(inplace=True)
        )

        # --- Fusion and Attention ---
        # Calculate total channels after concatenating the three paths.
        # Path A: channels, Path B refined: grad_refined_channels, Path C: channels
        total_channels_before_mix = channels + grad_refined_channels + channels
        # Mixer: 1x1 ConvGNReLU to fuse concatenated features back to original 'channels'.
        self.mixer = ConvGNReLU(total_channels_before_mix, channels, kernel_size=1, padding=0)
        # ECA Block: Applies Efficient Channel Attention.
        self.eca = ECABlock(channels=channels)

        # --- Final Norm and Activation (Applied before residual connection) ---
        self.final_norm = nn.GroupNorm(max(1, channels // 32), channels)
        self.final_act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the EdgeEnhanceBlock.

        Args:
            x (torch.Tensor): Input feature map (B, channels, H, W).

        Returns:
            torch.Tensor: Output enhanced feature map (B, channels, H, W).
        """
        # Store the input for the residual connection.
        residual = x # Shape: (B, C, H, W)

        # --- Process through parallel paths ---
        # Path A (Context): Learns general spatial patterns.
        out_a = self.path_a(x) # Shape: (B, C, H, W)

        # Path B (Gradient): Applies Sobel-initialized conv, norm+relu, then refines.
        grad_raw = self.grad_conv(x) # Shape: (B, C*2, H, W)
        grad_activated = self.grad_norm_relu(grad_raw) # Shape: (B, C*2, H, W)
        out_b = self.grad_refine(grad_activated) # Shape: (B, C_grad_refined, H, W)

        # Path C (Laplacian): Applies Laplacian-initialized conv, then norm+relu.
        lap_raw = self.laplacian_conv(x) # Shape: (B, C, H, W)
        out_c = self.laplacian_norm_relu(lap_raw) # Shape: (B, C, H, W)

        # --- Fusion ---
        # Concatenate features from Path A, refined Path B, and Path C.
        combined = torch.cat([out_a, out_b, out_c], dim=1) # Shape: (B, C + C_grad_refined + C, H, W)
        # Mix concatenated features back to the original channel dimension.
        mixed = self.mixer(combined) # Shape: (B, C, H, W)

        # --- Attention ---
        # Apply Efficient Channel Attention.
        attended = self.eca(mixed) # Shape: (B, C, H, W)

        # --- Residual Connection ---
        # Apply final normalization *before* adding the residual.
        normalized_attended = self.final_norm(attended) # Shape: (B, C, H, W)
        # Add the original input.
        summed = normalized_attended + residual # Shape: (B, C, H, W)
        # Apply the final activation.
        output = self.final_act(summed) # Shape: (B, C, H, W)

        return output

# --- Main Edge Branch Module ---
class EdgeBranch(nn.Module):
    """
    Edge Refinement Pathway (ERP) - Paper Section 3.2.

    Edge Refinement Pathway (ERP) implementation for C3Net.
    Referred to as "Edge Refinement Pathway" in the IEEE TAI paper.

    Processes early DINOv2 features (Stage 1 & 2) through a series of adaptation,
    upsampling, fusion, and enhancement steps to produce a high-resolution feature map
    emphasizing object boundaries and edge details. Uses EdgeEnhanceBlocks with
    gradient-initialized (Sobel/Laplacian) convolutions.

    Architecture Flow (Simplified):
    1. Adapt F_s1 and F_s2 features to `chan_adapt` using 1x1 Convs.
    2. Upsample (2x) both adapted features using `upsampler_adapt`.
    3. Fuse the upsampled features via concatenation and a 1x1 Conv (`fuse1`) -> `chan_fuse1`.
    4. Upsample (2x) the fused features using `upsampler_fuse1`.
    5. Process through `num_enhance_blocks` of `EdgeEnhanceBlock` (operates at `chan_fuse1`).
    6. Adapt channels if needed (`adapt_enhance_ch`) -> `chan_enhance`.
    7. Adapt channels to `chan_adapt_final` using a 1x1 Conv (`adapt_final`).
    8. Upsample (2x) the final adapted features using `upsampler_adapt_final`.

    Resulting spatial resolution is typically input Hf * 8, Wf * 8.

    Args:
        encoder_channels (int): Channel dimension of the DINOv2 encoder output features.
        intermediate_channels (List[int]): Channel dimensions for decoder stages:
                                           [`chan_adapt`, `chan_fuse1`, `chan_enhance`].
        output_channels (int): Channel dimension immediately *before* the final upsampling stage
                               (`chan_adapt_final`). This determines the final output channel dimension.
        num_enhance_blocks (int): Number of `EdgeEnhanceBlock` modules to apply. Default: 2.
        shared_upsamplers (nn.ModuleDict): Dictionary mapping channel size (str) to
                                           pre-initialized shared upsampler instances. Must contain keys
                                           corresponding to `chan_adapt`, `chan_fuse1`, and `output_channels`.
    """
    def __init__(self,
                 encoder_channels: int,
                 intermediate_channels: List[int],
                 output_channels: int,
                 num_enhance_blocks: int = 2,
                 shared_upsamplers: nn.ModuleDict = None):
        super().__init__()

        # --- Input Validation & Channel Configuration ---
        if not isinstance(intermediate_channels, list) or len(intermediate_channels) != 3:
            raise ValueError("Argument 'intermediate_channels' must be a list containing exactly "
                             "three integers: [chan_adapt, chan_fuse1, chan_enhance].")
        self.chan_adapt, self.chan_fuse1, self.chan_enhance = intermediate_channels

        if not isinstance(output_channels, int) or output_channels <= 0:
             raise ValueError("Argument 'output_channels' (chan_adapt_final) must be a positive integer.")
        self.chan_adapt_final = output_channels

        # Validate and store shared upsamplers dictionary
        if not isinstance(shared_upsamplers, nn.ModuleDict):
             raise TypeError("Argument 'shared_upsamplers' must be an instance of nn.ModuleDict.")
        self.upsamplers = shared_upsamplers

        # Verify necessary upsampler keys exist
        needed_upsampler_keys = {str(self.chan_adapt), str(self.chan_fuse1), str(self.chan_adapt_final)}
        missing_keys = needed_upsampler_keys - set(self.upsamplers.keys())
        if missing_keys:
            raise KeyError(f"The 'shared_upsamplers' dictionary is missing required upsampler instances "
                           f"for channel keys: {missing_keys}. Available keys: {list(self.upsamplers.keys())}")

        # --- Define Architectural Layers ---

        # 1. Adaptation Layers: Adapt encoder features (D channels) to chan_adapt.
        self.adapt_s1 = ConvGNReLU(encoder_channels, self.chan_adapt, kernel_size=1, padding=0)
        self.adapt_s2 = ConvGNReLU(encoder_channels, self.chan_adapt, kernel_size=1, padding=0)

        # 2. Fusion Layer (Fuse 1): Fuses the first upsampled streams.
        self.fuse1 = ConvGNReLU(self.chan_adapt * 2, self.chan_fuse1, kernel_size=1, padding=0)

        # 3. Enhancement Blocks: Apply sequential EdgeEnhanceBlocks.
        # Using the updated EdgeEnhanceBlock with fixed initializations.
        # grad_out_channels_ratio can be hardcoded (e.g., 4) or passed if needed.
        enhance_blocks_list = []
        for _ in range(num_enhance_blocks):
             enhance_blocks_list.append(EdgeEnhanceBlock(channels=self.chan_fuse1, grad_out_channels_ratio=4))
        self.enhance_blocks = nn.Sequential(*enhance_blocks_list)


        # 4. Adapt Enhancement Channels: Adapt from 'chan_fuse1' (output of enhance_blocks)
        # to the target 'chan_enhance' if they differ.
        if self.chan_fuse1 != self.chan_enhance:
            self.adapt_enhance_ch = ConvGNReLU(self.chan_fuse1, self.chan_enhance, kernel_size=1, padding=0)
        else:
            self.adapt_enhance_ch = nn.Identity()

        # 5. Final Adaptation Layer: Adapt features from 'chan_enhance' to 'chan_adapt_final'.
        self.adapt_final = ConvGNReLU(self.chan_enhance, self.chan_adapt_final, kernel_size=1, padding=0)

        # Final output channel dimension is determined by chan_adapt_final.
        self.final_feature_channels = self.chan_adapt_final



    def forward(self,
                f_s1: torch.Tensor,
                f_s2: torch.Tensor
               ) -> torch.Tensor:
        """
        Forward pass through the Edge Detection Branch.

        Args:
            f_s1 (torch.Tensor): Features from Encoder Stage 1 (B, D, Hf, Wf).
            f_s2 (torch.Tensor): Features from Encoder Stage 2 (B, D, Hf, Wf).

        Returns:
            torch.Tensor: Final edge feature map (B, final_feature_channels, Hf*8, Wf*8).
        """
        # --- Retrieve necessary upsampler instances ---
        try:
            upsampler_adapt = self.upsamplers[str(self.chan_adapt)]
            upsampler_fuse1 = self.upsamplers[str(self.chan_fuse1)]
            upsampler_adapt_final = self.upsamplers[str(self.chan_adapt_final)]
        except KeyError as e:
            logger.error(f"Missing required upsampler instance during EdgeBranch forward pass: {e}")
            raise e

        # --- Stage 1: Input Adaptation ---
        # Input shapes: (B, D, Hf, Wf)
        x_s1_adapted = self.adapt_s1(f_s1) # Output shape: (B, chan_adapt, Hf, Wf)
        x_s2_adapted = self.adapt_s2(f_s2) # Output shape: (B, chan_adapt, Hf, Wf)

        # --- Stage 2: First Upsampling (2x) & Fusion ---
        x_s1_up1 = upsampler_adapt(x_s1_adapted) # Output shape: (B, chan_adapt, Hf*2, Wf*2)
        x_s2_up1 = upsampler_adapt(x_s2_adapted) # Output shape: (B, chan_adapt, Hf*2, Wf*2)
        x_concat1 = torch.cat([x_s1_up1, x_s2_up1], dim=1) # Shape: (B, chan_adapt*2, Hf*2, Wf*2)
        x_fused1 = self.fuse1(x_concat1) # Output shape: (B, chan_fuse1, Hf*2, Wf*2)

        # --- Stage 3: Second Upsampling (2x) & Enhancement ---
        x_up2 = upsampler_fuse1(x_fused1) # Output shape: (B, chan_fuse1, Hf*4, Wf*4)
        x_enhanced = self.enhance_blocks(x_up2) # Output shape: (B, chan_fuse1, Hf*4, Wf*4)
        x_enhanced_adapted = self.adapt_enhance_ch(x_enhanced) # Output shape: (B, chan_enhance, Hf*4, Wf*4)

        # --- Stage 4: Final Adaptation & Third Upsampling (2x) ---
        x_adapted_final = self.adapt_final(x_enhanced_adapted) # Output shape: (B, chan_adapt_final, Hf*4, Wf*4)
        f_edge_features = upsampler_adapt_final(x_adapted_final) # Output shape: (B, chan_adapt_final, Hf*8, Wf*8)

        # Return the final high-resolution edge feature map.
        return f_edge_features