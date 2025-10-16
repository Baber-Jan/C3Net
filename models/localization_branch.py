"""
Implements the Contextual Localization Pathway (CLP) for C3Net.

This pathway focuses on semantic context and localization using late-stage encoder
features (Stages N-1, N). It incorporates an **enhanced Image-based Context
Guidance (ICG)** module combining Guided Contrast and  Iterative Attention
Gating enhancements, while retaining the original simpler ImageAppearanceCNN.

Rationale for Final Configuration:
---------------------------------
Following extensive experiments:
- Position P1 (within this branch) was confirmed as preferable to P2 (Fusion Head).
- Complex ICG enhancements E1 (Frequency-Aware CNN) proved ineffective.
- This version tests the hypothesis that combining E2 (improved contrast calculation
  via GuidedContrastModule with GFA-inspiration & context) and E3 (improved
  modulation via Iterative Attention Gates) might yield synergistic benefits,
  even when using the original simpler ImageAppearanceCNN (as E1 was detrimental).
- The original learnable DySample upsampler is used based on EXP-Upsample results.
- The loss function (defined separately) is tuned for precision here (post-ICG)
  and recall finally.

Key characteristics:
- Processes late, semantically rich encoder features (F_n-1, F_n).
- Uses shared learnable upsamplers (`DySample`).
- Employs `SemanticEnhanceBlock` for feature refinement.
- Integrates the **Enhanced ICG module (E2+E3)**:
    - Uses the *original* simple `ImageAppearanceCNN`.
    - Calculates contrast using `GuidedContrastModule` (E2).
    - Uses a two-stage Iterative Refinement with `AttentionGate`s (E3) for modulation.
- Outputs high-resolution localization features (`F_loc_features`) with potentially
  improved saliency suppression, ready for fusion.

Author  : Baber Jan
Date    : 09-05-2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Type
import logging
 
from models.utils import ConvGNReLU, ECABlock, AttentionGate
 

logger = logging.getLogger(__name__)

# --- Sub-Modules (Standard Helpers) ---

# --- Refined Spatial Attention Module (CBAM Style) ---
class SpatialAttentionModule(nn.Module):
    """
    Spatial Attention Module, inspired by the CBAM architecture.

    Computes a spatial attention map highlighting salient spatial locations.
    It achieves this by applying average and max pooling across the channel
    dimension, concatenating the results, and passing them through a
    convolutional layer followed by a sigmoid activation.

    Args:
        kernel_size (int): Kernel size for the spatial convolution layer.
                           Must be an odd integer. Default: 7.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        # Ensure the kernel size is odd to allow for 'same' padding.
        if not isinstance(kernel_size, int) or kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size ({kernel_size}) must be a positive odd integer.")
        # Calculate padding size to maintain spatial dimensions.
        padding = kernel_size // 2

        # Convolutional layer to process the concatenated pooled features.
        # Input has 2 channels (AvgPool result + MaxPool result). Output is 1 channel.
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False # Bias is often omitted in attention mechanisms.
        )
        # Sigmoid activation function to scale attention weights between 0 and 1.
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention to the input feature map.

        Args:
            x (torch.Tensor): Input feature map (B, C, H, W).

        Returns:
            torch.Tensor: Attention-weighted feature map (B, C, H, W).
        """
        # --- Compute Channel Pooling ---
        # Average pooling across the channel dimension. Result shape: (B, 1, H, W).
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        # Max pooling across the channel dimension. Result shape: (B, 1, H, W).
        max_pool, _ = torch.max(x, dim=1, keepdim=True)

        # --- Generate Attention Map ---
        # Concatenate the pooled features along the channel dimension. Shape: (B, 2, H, W).
        pooled_features = torch.cat([avg_pool, max_pool], dim=1)
        # Apply convolution to learn spatial importance. Shape: (B, 1, H, W).
        attn_map_logits = self.conv(pooled_features)
        # Apply sigmoid to get the final attention map weights. Shape: (B, 1, H, W).
        attn_map = self.sigmoid(attn_map_logits)

        # --- Apply Attention ---
        # Multiply the input feature map element-wise by the spatial attention map.
        # Broadcasting automatically applies the (B, 1, H, W) map to each channel C.
        output = x * attn_map # Shape: (B, C, H, W)
        return output


# --- Semantic Enhancement Block ---
class SemanticEnhanceBlock(nn.Module):
    """
    Semantic Enhancement Unit (SEU) - Paper Section 3.3.1.

    Refines high-level semantic features using Spatial and Channel Attention.
    Referred to as "Semantic Enhancement Unit" in the IEEE TAI paper.

    Applies a sequence of operations including depthwise separable convolutions
    for efficiency, Spatial Attention (CBAM-style), Efficient Channel Attention (ECA),
    and a residual connection.

    Args:
        channels (int): Number of input and output channels for the block.
        spatial_attn_kernel_size (int): Kernel size for the internal
                                        SpatialAttentionModule. Default: 7.
    """
    def __init__(self, channels: int, spatial_attn_kernel_size: int = 7):
        super().__init__()
        # --- Validate Input Channels ---
        if not isinstance(channels, int) or channels <= 0:
            raise ValueError(f"Input 'channels' must be a positive integer, got {channels}.")
        self.channels = channels

        # --- Convolutional Block 1 (Depthwise Separable) ---
        # Consists of Depthwise Conv -> GN -> ReLU -> Pointwise Conv -> GN -> ReLU
        self.conv1_dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.norm1_dw = nn.GroupNorm(max(1, channels // 32), channels) # Adapt group size
        self.act1_dw = nn.ReLU(inplace=True)
        self.conv1_pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        # Group Norm and ReLU applied after the pointwise convolution
        self.norm1_pw_act = nn.Sequential(nn.GroupNorm(max(1, channels // 32), channels), nn.ReLU(inplace=True))

        # --- Attention Modules ---
        # Spatial Attention Module (CBAM Style)
        self.spatial_attn = SpatialAttentionModule(kernel_size=spatial_attn_kernel_size)
        # Efficient Channel Attention Module (ECA-Net)
        self.channel_attn = ECABlock(channels=channels)

        # --- Convolutional Block 2 (Depthwise Separable) ---
        # Applied after attention, before the residual connection.
        self.conv2_dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.norm2_dw = nn.GroupNorm(max(1, channels // 32), channels)
        self.act2_dw = nn.ReLU(inplace=True)
        self.conv2_pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        # Group Norm applied before adding the residual.
        self.norm2_pw = nn.GroupNorm(max(1, channels // 32), channels)

        # --- Final Activation ---
        # Applied after the residual connection.
        self.final_act = nn.ReLU(inplace=True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SemanticEnhanceBlock.

        Args:
            x (torch.Tensor): Input feature map (B, channels, H, W).

        Returns:
            torch.Tensor: Output enhanced feature map (B, channels, H, W).
        """
        # Store the input for the residual connection.
        residual = x # Shape: (B, C, H, W)

        # --- First Convolutional Block ---
        # Apply depthwise, norm, act, pointwise, norm+act sequence.
        out = self.conv1_dw(x)        # Shape: (B, C, H, W)
        out = self.act1_dw(self.norm1_dw(out)) # Shape: (B, C, H, W)
        out = self.conv1_pw(out)      # Shape: (B, C, H, W)
        out = self.norm1_pw_act(out)  # Shape: (B, C, H, W)

        # --- Attention Modules ---
        # Apply spatial attention first, then channel attention.
        out = self.spatial_attn(out) # Shape: (B, C, H, W)
        out = self.channel_attn(out) # Shape: (B, C, H, W)

        # --- Second Convolutional Block ---
        # Apply depthwise, norm, act, pointwise, norm sequence.
        out = self.conv2_dw(out)      # Shape: (B, C, H, W)
        out = self.act2_dw(self.norm2_dw(out)) # Shape: (B, C, H, W)
        out = self.conv2_pw(out)      # Shape: (B, C, H, W)
        out = self.norm2_pw(out)      # Shape: (B, C, H, W)

        # --- Residual Connection ---
        # Add the original input (residual).
        out = out + residual          # Shape: (B, C, H, W)
        # Apply the final activation function.
        output = self.final_act(out)  # Shape: (B, C, H, W)

        return output


# --- Image Appearance CNN (for ICG Module) ---
class ImageAppearanceCNN(nn.Module):
    """
    A lightweight Convolutional Neural Network (CNN) designed to extract
    low-level appearance features (like color and texture) from the input image.
    These features are used by the Image-based Context Guidance (ICG) module.

    Args:
        image_channels (int): Number of channels in the input image (typically 3 for RGB). Default: 3.
        out_channels (int): Number of output channels for the extracted appearance features. Default: 32.
    """
    def __init__(self, image_channels: int = 3, out_channels: int = 32):
        super().__init__()
        # --- Validate Channels ---
        if not isinstance(image_channels, int) or image_channels <= 0:
            raise ValueError(f"'image_channels' must be a positive integer, got {image_channels}.")
        if not isinstance(out_channels, int) or out_channels <= 0:
             raise ValueError(f"'out_channels' must be a positive integer, got {out_channels}.")

        # Define a simple two-layer CNN using ConvGNReLU blocks.
        # No pooling is used to maintain the spatial resolution.
        # Layer 1: Input channels -> out_channels / 2
        self.conv1 = ConvGNReLU(image_channels, out_channels // 2, kernel_size=3, padding=1)
        # Layer 2: out_channels / 2 -> out_channels
        self.conv2 = ConvGNReLU(out_channels // 2, out_channels, kernel_size=3, padding=1)

    def forward(self, resized_image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract appearance features.

        Args:
            resized_image (torch.Tensor): Input image tensor, typically resized to match the
                                          spatial dimensions of the features in the ICG module
                                          (B, image_channels, H, W).

        Returns:
            torch.Tensor: Extracted appearance features (B, out_channels, H, W).
        """
        # Pass the image through the two convolutional blocks.
        features_conv1 = self.conv1(resized_image) # Shape: (B, out_channels/2, H, W)
        appearance_features = self.conv2(features_conv1) # Shape: (B, out_channels, H, W)
        return appearance_features


# Include GuidedContrastModule 
class GuidedContrastModule(nn.Module):
    """ Enhanced contrast calculation (E2): Uses learned spatial aggregation and global context. """
    def __init__(self, feature_channels: int, contrast_channels: int, aggregation_kernel_size: int = 5, global_bg_threshold: float = 0.1, eps: float = 1e-7):
        super().__init__(); self.feature_channels = feature_channels; self.contrast_channels = contrast_channels; self.aggregation_padding = aggregation_kernel_size // 2; self.global_bg_threshold = global_bg_threshold; self.eps = eps
        # Layers for Spatial Aggregation
        self.fg_agg_dw = nn.Conv2d(feature_channels, feature_channels, aggregation_kernel_size, padding=self.aggregation_padding, groups=feature_channels, bias=False)
        self.fg_agg_pw_norm_relu = ConvGNReLU(feature_channels, feature_channels, 1, padding=0, bias=False)
        self.bg_agg_dw = nn.Conv2d(feature_channels, feature_channels, aggregation_kernel_size, padding=self.aggregation_padding, groups=feature_channels, bias=False)
        self.bg_agg_pw_norm_relu = ConvGNReLU(feature_channels, feature_channels, 1, padding=0, bias=False)
        # Layers for Combining Local and Global Background
        self.bg_fusion = ConvGNReLU(feature_channels * 2, feature_channels, 1, padding=0, bias=False)
        # Final Contrast Refinement
        self.contrast_refiner = ConvGNReLU(feature_channels, self.contrast_channels, 1, padding=0, bias=False)

    def forward(self, appearance_features: torch.Tensor, guidance_map: torch.Tensor) -> torch.Tensor:
        if guidance_map.shape[1] != 1: guidance_map = guidance_map[:, :1, :, :]; guidance_map = torch.clamp(guidance_map, 0.0, 1.0)
        # Weight, Aggregate FG/BG
        app_feat_fg = appearance_features * guidance_map; app_feat_bg = appearance_features * (1.0 - guidance_map)
        agg_local_fg = self.fg_agg_pw_norm_relu(self.fg_agg_dw(app_feat_fg))
        agg_local_bg = self.bg_agg_pw_norm_relu(self.bg_agg_dw(app_feat_bg))
        # Global BG Context
        with torch.no_grad(): bg_mask_confident = (guidance_map < self.global_bg_threshold).float(); bg_pixel_count = bg_mask_confident.sum(dim=[1, 2, 3], keepdim=True)
        global_bg_sum = (appearance_features * bg_mask_confident).sum(dim=[2, 3], keepdim=True); global_bg_feat_avg = global_bg_sum / (bg_pixel_count + self.eps)
        global_bg_feat_expanded = global_bg_feat_avg.expand_as(appearance_features)
        # Fuse BG and Calculate Contrast
        effective_bg_feat = self.bg_fusion(torch.cat([agg_local_bg, global_bg_feat_expanded], dim=1))
        raw_contrast = torch.abs(agg_local_fg - effective_bg_feat)
        learned_contrast_map = self.contrast_refiner(raw_contrast)
        return learned_contrast_map

class ImageBasedContextGuidance(nn.Module):
    """
    Image-based Context Guidance (ICG) Module - Paper Section 3.3.2.

    Enhanced ICG implementation combining Guided Contrast Module and Iterative
    Attention Gating for intrinsic saliency suppression.

    Combines E2 (Guided Contrast Module) and E3 (Iterative Attention Gating),
    using the original simple ImageAppearanceCNN based on experimental findings.

    Args:
        feature_channels (int): Channels of input semantic features (F_enhanced_N).
        image_channels (int): Input image channels. Default: 3.
        ia_cnn_channels (int): Output channels of internal ImageAppearanceCNN. Default: 32.
        contrast_channels (int): Output channels of the GuidedContrastModule. Default: 16.
        contrast_agg_kernel (int): Kernel size for GFA aggregation in contrast module. Default: 5.
        contrast_bg_thresh (float): Threshold for global BG context in contrast module. Default: 0.1.
        ag_intermediate_channels (Optional[int]): Intermediate channels for AGs. Default: None (feat_ch // 2).
        eps (float): Epsilon for numerical stability. Default: 1e-7.
        **kwargs: Absorbs unused parameters from config.
    """
    def __init__(self,
                 feature_channels: int,
                 image_channels: int = 3,
                 ia_cnn_channels: int = 32, # For original simple CNN
                 contrast_channels: int = 16, # Output of GuidedContrastModule
                 contrast_agg_kernel: int = 5, # Param for GuidedContrastModule
                 contrast_bg_thresh: float = 0.1, # Param for GuidedContrastModule
                 ag_intermediate_channels: Optional[int] = None, # Param for AttentionGates
                 eps: float = 1e-7,
                 **kwargs): # Absorb potentially unused enhancement params from config
        super().__init__()

        # --- Input Validation ---
        if not isinstance(feature_channels, int) or feature_channels <= 0: raise ValueError("'feature_channels' must be positive.")
        if not isinstance(ia_cnn_channels, int) or ia_cnn_channels <= 0: raise ValueError("'ia_cnn_channels' must be positive.")
        if not isinstance(contrast_channels, int) or contrast_channels <= 0: raise ValueError("'contrast_channels' must be positive.")

        self.feature_channels = feature_channels
        self.eps = eps

        # --- Define ICG Components ---
        # 1. Initial Mask Head 
        self.initial_mask_head = nn.Conv2d(feature_channels, 1, kernel_size=1)

        # 2. Appearance CNN 
        self.ia_cnn = ImageAppearanceCNN(image_channels, ia_cnn_channels)

        # 3. Contrast Module
        self.contrast_module = GuidedContrastModule(
            feature_channels=ia_cnn_channels, # Input from ia_cnn
            contrast_channels=contrast_channels,
            aggregation_kernel_size=contrast_agg_kernel,
            global_bg_threshold=contrast_bg_thresh,
            eps=eps
        )

        # 4. Attention Gates 
        if ag_intermediate_channels is None:
            ag_intermediate_channels = max(1, feature_channels // 2)

        self.attn_gate1 = AttentionGate( F_x=feature_channels, F_g=contrast_channels, F_int=ag_intermediate_channels )
        self.attn_gate2 = AttentionGate( F_x=feature_channels, F_g=contrast_channels, F_int=ag_intermediate_channels )

        # Log ignored parameters from kwargs if any unexpected ones were passed
        # Example: Check for freq/octave params which are not used here
        known_params = {'feature_channels', 'image_channels', 'ia_cnn_channels',
                        'contrast_channels', 'contrast_agg_kernel', 'contrast_bg_thresh',
                        'ag_intermediate_channels', 'eps'}
        ignored_params = {k:v for k,v in kwargs.items() if k not in known_params}
        if ignored_params:
            logger.warning(f"Enhanced ICG (E23) ignoring parameters: {ignored_params}")


    def forward(self, f_enhanced_n: torch.Tensor, resized_image: torch.Tensor) -> torch.Tensor:
        """ Forward pass implementing E2 (GuidedContrast) + E3 (Iterative AG). """
        # --- Shared Computations ---
        image_appearance_features = self.ia_cnn(resized_image) # Original CNN

        # --- Stage 1 ---
        P_init = torch.sigmoid(self.initial_mask_head(f_enhanced_n))
        # Calculate contrast using E2 module
        contrast_map_s1 = self.contrast_module(image_appearance_features, P_init)
        # Apply First Attention Gate (E3)
        f_modulated_s1 = self.attn_gate1(x=f_enhanced_n, g=contrast_map_s1)

        # --- Stage 2 (Refinement) ---
        # Generate refined mask guidance P_refined (not directly used if contrast map is reused)
        # P_refined = torch.sigmoid(self.initial_mask_head(f_modulated_s1)) # Optional: use if recalculating contrast
        # Reuse contrast map from Stage 1 (decisive choice)
        contrast_map_s2 = contrast_map_s1
        # Apply Second Attention Gate (E3)
        f_final_modulated = self.attn_gate2(x=f_modulated_s1, g=contrast_map_s2)

        return f_final_modulated

# --- Light Refinement Block ---
class LightRefineBlock(nn.Sequential):
    """
    A simple two-layer refinement block using depthwise separable convolutions.

    Args:
        channels (int): Number of input and output channels.
    """
    def __init__(self, channels: int):
        # --- Validate Channels ---
        if not isinstance(channels, int) or channels <= 0:
            raise ValueError(f"'channels' must be a positive integer, got {channels}.")
        # Define the sequential block
        super().__init__(
            # Layer 1: Depthwise Conv -> GN -> ReLU -> Pointwise Conv
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(max(1, channels // 32), channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            # Layer 2: GN -> ReLU (applied after pointwise of Layer 1)
            nn.GroupNorm(max(1, channels // 32), channels),
            nn.ReLU(inplace=True)
        )

# --- Main Object Localization Branch ---
class ObjectLocalizationBranch(nn.Module):
    """
    Contextual Localization Pathway (CLP) - Paper Section 3.3.

    Finalized Object Localization Branch using Enhanced ICG.
    Referred to as "Contextual Localization Pathway" in the IEEE TAI paper.
    """
    def __init__(self,
                 encoder_channels: int,
                 intermediate_channels: List[int],
                 output_channels: int,
                 num_enhance_blocks: int = 1,
                 icg_params: dict = None, # Pass params for ICG E2+E3 here
                 shared_upsamplers: nn.ModuleDict = None):
        super().__init__()
        # --- Input Validation & Channel Config (Keep as before) ---
        if not isinstance(intermediate_channels, list) or len(intermediate_channels) != 3: raise ValueError("intermediate_channels: [adapt, fuse1, refine2]")
        self.chan_adapt, self.chan_fuse1, self.chan_refine2 = intermediate_channels
        if not isinstance(output_channels, int) or output_channels <= 0: raise ValueError("output_channels must be positive int.")
        self.chan_adapt_final = output_channels
        if not isinstance(shared_upsamplers, nn.ModuleDict): raise TypeError("shared_upsamplers must be nn.ModuleDict.")
        self.upsamplers = shared_upsamplers
        needed_keys = {str(self.chan_adapt), str(self.chan_fuse1), str(self.chan_adapt_final)}
        if not needed_keys.issubset(self.upsamplers.keys()): raise KeyError(f"shared_upsamplers missing keys: {needed_keys - set(self.upsamplers.keys())}")

        # --- Define Layers ---
        self.adapt_n = ConvGNReLU(encoder_channels, self.chan_adapt, 1, padding=0)
        self.adapt_n_minus_1 = ConvGNReLU(encoder_channels, self.chan_adapt, 1, padding=0)
        self.enhance_blocks = nn.Sequential(*[SemanticEnhanceBlock(self.chan_adapt) for _ in range(num_enhance_blocks)])

        # Instantiate the  ICG module 
        self.icg_module = ImageBasedContextGuidance(
            feature_channels=self.chan_adapt, # Operates on enhanced features
            **(icg_params or {}) # Pass all relevant params for E2+E3 config
        )

        self.fuse1 = ConvGNReLU(self.chan_adapt * 2, self.chan_fuse1, 1, padding=0)
        self.refine_block2 = LightRefineBlock(channels=self.chan_fuse1)
        self.adapt_after_refine = ConvGNReLU(self.chan_fuse1, self.chan_refine2, 1, padding=0) if self.chan_fuse1 != self.chan_refine2 else nn.Identity()
        self.adapt_final = ConvGNReLU(self.chan_refine2, self.chan_adapt_final, 1, padding=0)
        self.final_feature_channels = self.chan_adapt_final

    def forward(self, f_n_minus_1: torch.Tensor, f_n: torch.Tensor, x_resized_map: Dict[Tuple[int, int], torch.Tensor]) -> torch.Tensor:
        """ Forward pass using Enhanced ICG (E2+E3). """
        # Retrieve Upsamplers
        upsampler_adapt = self.upsamplers[str(self.chan_adapt)]; upsampler_fuse1 = self.upsamplers[str(self.chan_fuse1)]; upsampler_adapt_final = self.upsamplers[str(self.chan_adapt_final)]
        # Get Resized Image
        icg_h, icg_w = f_n.shape[-2:]; resized_image_for_icg = x_resized_map.get((icg_h, icg_w));
        if resized_image_for_icg is None: raise ValueError(f"Resized image ({icg_h}, {icg_w}) not found for ICG.")

        # --- Process F_N Stream with Enhanced ICG ---
        f_n_adapted = self.adapt_n(f_n)
        f_n_enhanced = self.enhance_blocks(f_n_adapted)
        # Apply ENHANCED ICG (E2+E3)
        f_n_modulated = self.icg_module(f_n_enhanced, resized_image_for_icg)

        # --- Upsample, Fuse with F_N-1, Refine ---
        f_n_up1 = upsampler_adapt(f_n_modulated)
        f_n_minus_1_adapted = self.adapt_n_minus_1(f_n_minus_1)
        f_n_minus_1_up1 = upsampler_adapt(f_n_minus_1_adapted)
        f_concat1 = torch.cat([f_n_up1, f_n_minus_1_up1], dim=1)
        f_fused1 = self.fuse1(f_concat1)
        f_up2 = upsampler_fuse1(f_fused1)
        f_refined2 = self.refine_block2(f_up2)
        f_post_refine = self.adapt_after_refine(f_refined2)
        f_adapted_final = self.adapt_final(f_post_refine)
        f_loc_features = upsampler_adapt_final(f_adapted_final)

        return f_loc_features