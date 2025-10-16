# filename: models/fusion_head.py
"""
Implements the Attentive Fusion Module (AFM) for the C3Net architecture.

This module serves as the terminal stage of the C3Net decoder, responsible for
synergistically integrating the outputs from the two specialized pathways:
1. Edge Refinement Pathway (ERP): Provides high-resolution features rich in boundary details (`f_edge_features`).
2. Contextual Localization Pathway (CLP): Provides features containing semantic context and
   object localization information, crucially with salient distractors suppressed
   by the ICG module (`f_loc_features`).

The core mechanism is an attention-gating strategy where the context-aware
localization features generate a spatial attention map. This map modulates the
edge features, effectively emphasizing boundary details that fall within the
localized regions of interest while suppressing irrelevant edge information.
The gated edge features and the localization features are then combined and
processed through refinement layers to produce the final, high-fidelity
segmentation prediction.

Input:
- f_edge_features (torch.Tensor): Feature map from the Edge Detection Branch
                                  (B, C_edge, H, W).
- f_loc_features (torch.Tensor): Feature map from the Object Localization Branch
                                 (B, C_loc, H, W). Expected to have the same
                                 spatial dimensions (H, W) as f_edge_features.

Output:
- final_logits (torch.Tensor): Raw output logits for the final segmentation map
                               (B, 1, H, W). Suitable for use with a loss function
                               expecting logits (e.g., BCEWithLogitsLoss).


Author  : Baber Jan
Date    : 01-05-2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
import logging

from models.utils import ConvGNReLU

# Initialize logger for this module
logger = logging.getLogger(__name__)

class FinalFusionHead(nn.Module):
    """
    Attentive Fusion Module (AFM) - Paper Section 3.4.

    Final Fusion Head for C3Net using Attention Gating.
    Referred to as "Attentive Fusion Module" in the IEEE TAI paper.

    Combines Edge Branch and Localization Branch features. Uses localization features
    to generate a spatial attention gate, applies this gate to edge features,
    concatenates gated edge and localization features, refines the result,
    and predicts the final segmentation logits.

    Args:
        edge_channels (int): Number of channels in the input feature map from the
                             Edge Branch.
        loc_channels (int): Number of channels in the input feature map from the
                            Localization Branch.
        fuse_channels (int): Intermediate channel dimension used after initial
                             adaptation and within the refinement blocks. Default: 64.
        num_refine_blocks (int): Number of `ConvGNReLU` refinement blocks applied
                                 after combining features. Must be >= 1. Default: 2.
        attn_gen_intermediate_channels (int): Intermediate channels within the
                                              convolutional pathway used to generate
                                              the attention gate. Default: 32.
    """
    def __init__(self,
                 edge_channels: int,
                 loc_channels: int,
                 fuse_channels: int = 64,
                 num_refine_blocks: int = 2,
                 attn_gen_intermediate_channels: int = 32):
        super().__init__()

        # --- Input Validation ---
        if not isinstance(edge_channels, int) or edge_channels <= 0:
             raise ValueError(f"'edge_channels' must be a positive integer, got {edge_channels}.")
        if not isinstance(loc_channels, int) or loc_channels <= 0:
             raise ValueError(f"'loc_channels' must be a positive integer, got {loc_channels}.")
        if not isinstance(fuse_channels, int) or fuse_channels <= 0:
             raise ValueError(f"'fuse_channels' must be a positive integer, got {fuse_channels}.")
        if not isinstance(num_refine_blocks, int) or num_refine_blocks < 1:
            raise ValueError(f"'num_refine_blocks' must be an integer >= 1, got {num_refine_blocks}.")
        if not isinstance(attn_gen_intermediate_channels, int) or attn_gen_intermediate_channels <= 0:
             raise ValueError(f"'attn_gen_intermediate_channels' must be a positive integer, got {attn_gen_intermediate_channels}.")

        # Store configuration
        self.edge_channels = edge_channels
        self.loc_channels = loc_channels
        self.fuse_channels = fuse_channels

        # --- Define Architectural Layers ---

        # 1. Adaptation Layers: Project features from each branch to the common 'fuse_channels' dimension.
        # Uses 1x1 ConvGNReLU blocks.
        self.adapt_edge = ConvGNReLU(edge_channels, fuse_channels, kernel_size=1, padding=0)
        self.adapt_loc = ConvGNReLU(loc_channels, fuse_channels, kernel_size=1, padding=0)

        # 2. Attention Gate Generation Pathway: Generates a spatial attention map from adapted localization features.
        # Implements the "refined convolutional pathway": 3x3 ConvGNReLU -> 1x1 Conv -> Sigmoid.
        self.attention_generator = nn.Sequential(
            # First layer uses 3x3 Conv to incorporate local spatial context for better gating.
            ConvGNReLU(fuse_channels, attn_gen_intermediate_channels, kernel_size=3, padding=1),
            # Second layer is a 1x1 Conv to output the single attention channel (logits).
            nn.Conv2d(attn_gen_intermediate_channels, 1, kernel_size=1),
            # Sigmoid activation scales the attention weights to the range [0, 1].
            nn.Sigmoid()
        )

        # 3. Refinement Layers: Processes the combined (gated edge + localization) features.
        # Uses a sequence of ConvGNReLU blocks (typically 3x3 kernel).
        refinement_layers = []
        # The first block adapts the channel dimension from the concatenated size (2 * fuse_channels)
        # down to 'fuse_channels'.
        refinement_layers.append(ConvGNReLU(fuse_channels * 2, fuse_channels, kernel_size=3, padding=1))
        # Add subsequent refinement blocks, each operating at 'fuse_channels'.
        for _ in range(num_refine_blocks - 1): # Subtract 1 because the first block is already added
            refinement_layers.append(ConvGNReLU(fuse_channels, fuse_channels, kernel_size=3, padding=1))
        # Create the sequential module for refinement.
        self.refinement_blocks = nn.Sequential(*refinement_layers)

        # 4. Final Prediction Layer: Predicts single-channel logits from the final refined features.
        # Uses a 1x1 convolution. No activation function here, as loss functions typically expect raw logits.
        self.final_pred = nn.Conv2d(fuse_channels, 1, kernel_size=1)


    def forward(self,
                f_edge_features: torch.Tensor, # Output from Edge Branch
                f_loc_features: torch.Tensor   # Output from Localization Branch
               ) -> torch.Tensor:
        """
        Forward pass for the Final Fusion Head.

        Args:
            f_edge_features (torch.Tensor): Feature map from Edge Branch (B, C_edge, H_edge, W_edge).
            f_loc_features (torch.Tensor): Feature map from Localization Branch (B, C_loc, H_loc, W_loc).

        Returns:
            torch.Tensor: Raw logits for the final segmentation map (B, 1, H_out, W_out),
                          where H_out, W_out match the spatial dimensions of f_loc_features.
        """

        # --- 1. Adapt Input Features ---
        # Project features from both branches to the common 'fuse_channels' dimension using 1x1 Convs.
        # Input shapes: (B, C_edge, H, W), (B, C_loc, H, W)
        f_edge_adapted = self.adapt_edge(f_edge_features) # Output shape: (B, fuse_channels, H, W)
        f_loc_adapted = self.adapt_loc(f_loc_features)   # Output shape: (B, fuse_channels, H, W)
        # logger.debug(f"Fusion Head - Edge Adapted: {f_edge_adapted.shape}")
        # logger.debug(f"Fusion Head - Loc Adapted: {f_loc_adapted.shape}")

        # --- 2. Generate Spatial Attention Gate ---
        # Use the adapted localization features (rich in context) to generate the spatial attention map.
        # Input shape: (B, fuse_channels, H, W)
        attention_map = self.attention_generator(f_loc_adapted) # Output shape: (B, 1, H, W), values in [0, 1]
        # logger.debug(f"Fusion Head - Attention Map: {attention_map.shape}")

        # --- 3. Modulate Edge Features ---
        # Apply the spatial attention gate element-wise to the adapted edge features.
        # This enhances edge details in regions deemed important by the localization branch.
        # Input shapes: (B, fuse_channels, H, W) * (B, 1, H, W)
        f_edge_gated = f_edge_adapted * attention_map # Output shape: (B, fuse_channels, H, W)
        # logger.debug(f"Fusion Head - Edge Gated: {f_edge_gated.shape}")

        # --- 4. Combine Gated Edge and Localization Features ---
        # Concatenate the spatially gated edge features and the adapted localization features
        # along the channel dimension.
        # Input shapes: (B, fuse_channels, H, W) and (B, fuse_channels, H, W)
        f_combined = torch.cat([f_edge_gated, f_loc_adapted], dim=1) # Output shape: (B, fuse_channels * 2, H, W)
        # logger.debug(f"Fusion Head - Combined: {f_combined.shape}")

        # --- 5. Refine Combined Features ---
        # Pass the combined features through the sequence of refinement ConvGNReLU blocks.
        # Learns to effectively integrate the context-selected edge details and localization context.
        # Input shape: (B, fuse_channels * 2, H, W)
        f_refined = self.refinement_blocks(f_combined) # Output shape: (B, fuse_channels, H, W)
        # logger.debug(f"Fusion Head - Refined: {f_refined.shape}")

        # --- 6. Final Prediction ---
        # Generate the final single-channel segmentation logits using a 1x1 convolution.
        # Input shape: (B, fuse_channels, H, W)
        final_logits = self.final_pred(f_refined) # Output shape: (B, 1, H, W)
        # logger.debug(f"Fusion Head - Final Logits: {final_logits.shape}")

        # Return the raw logits. Activation (e.g., sigmoid) should be applied either
        # within the loss function or during post-processing/inference.
        return final_logits