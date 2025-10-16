"""
Implements the main C3Net (Context-Contrast COD Network) model architecture.

This model serves as the central orchestrator, integrating several key components:
1.  A powerful DINOv2 encoder (`DinoV2WithRegistersEncoder`) for hierarchical
    feature extraction.
2.  A shared upsampling module used across the decoder for detail-preserving, content-
    aware upsampling based on learned sampling locations (Liu et al., ICCV 2023).
3.  A specialized dual-branch decoder:
    - `EdgeBranch` (Edge Refinement Pathway - ERP): Focuses on fine boundary
      details using early encoder features and Sobel/Laplacian initialized
      enhancement blocks (EEMs).
    - `ObjectLocalizationBranch` (Contextual Localization Pathway - CLP): Focuses
      on semantic context and localization using late encoder features, incorporating
      the novel Image-based Context Guidance (ICG) module for integrated salient
      object suppression.
4.  A `FinalFusionHead` (Attentive Fusion Module - AFM) that synergistically
    combines the outputs of the two decoder branches using an attention-gating
    mechanism.
5.  Intermediate prediction heads for enabling deep supervision during training.

The entire network is designed to be trained end-to-end, guided by the C3NetLoss
function which leverages the intermediate and final outputs.


Author  : Baber Jan
Date    : 01-05-2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Set
import logging

from models.backbone import DinoV2WithRegistersEncoder
from models.utils import DySample
from models.edge_branch import EdgeBranch
from models.localization_branch import ObjectLocalizationBranch
from models.fusion_head import FinalFusionHead

# Initialize logger for this module
logger = logging.getLogger(__name__)

class C3Net(nn.Module):
    """
    C3Net: Context-Contrast Camouflaged Object Detection Network.

    Integrates a DINOv2 backbone, a dual-branch decoder (Edge Refinement Pathway/ERP
    and Contextual Localization Pathway/CLP) using DySample for upsampling, an ICG
    module for saliency suppression, and an Attentive Fusion Module (AFM).

    Args:
        config (Dict): A configuration dictionary containing parameters for all
                       sub-modules (encoder, decoder branches, upsampler, fusion head).
                       Requires specific keys and structures as defined by the
                       component modules' initializers.
        img_size (int): The expected square input image size (H=W), e.g., 392.
                        Must be divisible by the encoder's patch size (typically 14).
    """
    def __init__(self, config: Dict):
        super().__init__()
        logger.info("--- Initializing C3Net Model ---")
        # --- Validate and Store Basic Config ---
        if not isinstance(config, dict):
            raise TypeError("`config` must be a dictionary.")
        img_size = config['image_processing'].get('target_size', 392)
        if not isinstance(img_size, int) or img_size <= 0:
             raise ValueError("`img_size` must be a positive integer.")
        self.config = config
        self.img_size = img_size

        # Validate input image size divisibility by patch size 
        self.patch_size = config.get('encoder', {}).get('patch_size', 14)
        if self.img_size % self.patch_size != 0:
            msg = (f"Input image size {self.img_size} is not divisible by "
                   f"encoder patch size ({self.patch_size}).")
            logger.error(msg)
            raise ValueError(msg)

        # --- 1. Encoder Initialization ---
        encoder_cfg = config.get('encoder', {})
        # Default stages often used for ViT-L/14 based architectures
        default_output_stages = [1, 2, 23, 24]
        self.output_stages = encoder_cfg.get('output_stages', default_output_stages)
        if len(self.output_stages) != 4:
            logger.warning(f"Expected 4 output stages from encoder for typical C3Net setup, "
                           f"but got {len(self.output_stages)}. Using: {self.output_stages}.")

        # Instantiate the DINOv2 encoder
        self.encoder = DinoV2WithRegistersEncoder(
            model_name=encoder_cfg.get('model_name', "facebook/dinov2-with-registers-large"), 
            variant=encoder_cfg.get('variant', 'large'), 
            output_stages=self.output_stages
        )
        self.encoder_channels: int = self.encoder.hidden_dim # Get channel dimension from encoder
        # Calculate spatial dimensions of the base feature map (Hf, Wf)
        self.feature_h, self.feature_w = self.encoder.get_patch_grid_size(self.img_size, self.img_size)
        

        # --- 2. Shared Upsamplers (DySample) Initialization ---
        decoder_cfg = config.get('decoder', {})
        # Extract DySample specific parameters from config
        upsampler_cfg = decoder_cfg.get('upsampler', {})
        dysample_style = upsampler_cfg.get('style', 'lp')
        dysample_groups = upsampler_cfg.get('groups', 4)
        dysample_dyscope = upsampler_cfg.get('dyscope', False)

        # Determine the unique input channel dimensions needed for DySample across both branches.
        # Each branch upsamples 3 times. Input channels are [AdaptC, Fuse1C, AdaptFinalC].
        edge_cfg = decoder_cfg.get('edge_branch', {})
        loc_cfg = decoder_cfg.get('loc_branch', {})
        # Get channel lists from config, providing defaults if missing
        edge_intermediate_channels = edge_cfg.get('intermediate_channels', [256, 256, 128]) # AdaptC, Fuse1C, EnhanceC
        edge_output_channels = edge_cfg.get('output_channels', 128) # AdaptFinalC
        loc_intermediate_channels = loc_cfg.get('intermediate_channels', [256, 256, 128]) # AdaptC, Fuse1C, Refine2C
        loc_output_channels = loc_cfg.get('output_channels', 128) # AdaptFinalC

        upsampler_input_channels_needed: Set[int] = set()
        # Edge branch upsample inputs: AdaptC, Fuse1C, AdaptFinalC
        upsampler_input_channels_needed.add(edge_intermediate_channels[0]) # chan_adapt
        upsampler_input_channels_needed.add(edge_intermediate_channels[1]) # chan_fuse1
        upsampler_input_channels_needed.add(edge_output_channels)          # chan_adapt_final
        # Loc branch upsample inputs: AdaptC, Fuse1C, AdaptFinalC
        upsampler_input_channels_needed.add(loc_intermediate_channels[0]) # chan_adapt
        upsampler_input_channels_needed.add(loc_intermediate_channels[1]) # chan_fuse1
        upsampler_input_channels_needed.add(loc_output_channels)          # chan_adapt_final

        # Filter out any invalid channel dimensions (e.g., None or 0 if config errors)
        valid_channels = {ch for ch in upsampler_input_channels_needed if isinstance(ch, int) and ch > 0}

        # Create a ModuleDict to hold shared DySample instances, keyed by stringified channel number.
        self.upsamplers = nn.ModuleDict({
            str(ch): DySample(in_channels=ch,
                              scale=2, # 2x upsampling at each stage
                              style=dysample_style,
                              groups=dysample_groups,
                              dyscope=dysample_dyscope)
            for ch in sorted(list(valid_channels)) # Sort for deterministic ordering
        })
        

        # --- 3. Decoder Branches Initialization ---
        # Extract final output channel dimensions for consistency check and fusion head.
        self.edge_final_channels = edge_output_channels
        self.loc_final_channels = loc_output_channels

        # Instantiate branches, passing the *entire dictionary* of shared upsamplers.
        self.edge_branch = EdgeBranch(
            encoder_channels=self.encoder_channels,
            intermediate_channels=edge_intermediate_channels,
            output_channels=edge_output_channels,
            num_enhance_blocks=edge_cfg.get('num_enhance_blocks', 2),
            shared_upsamplers=self.upsamplers  
        )
        self.loc_branch = ObjectLocalizationBranch(
            encoder_channels=self.encoder_channels,
            intermediate_channels=loc_intermediate_channels,
            output_channels=loc_output_channels,
            num_enhance_blocks=loc_cfg.get('num_enhance_blocks', 1),
            icg_params=loc_cfg.get('icg_params', {}),  
            shared_upsamplers=self.upsamplers  
        )
        

        # --- 4. Intermediate Prediction Heads (for deep supervision) ---
        # Lightweight 1x1 Convolutions applied to the output features of each branch before fusion.
        self.edge_pred_head = nn.Conv2d(self.edge_final_channels, 1, kernel_size=1)
        self.loc_pred_head = nn.Conv2d(self.loc_final_channels, 1, kernel_size=1)
        

        # --- 5. Final Fusion Head Initialization ---
        fusion_cfg = config.get('fusion_head', {})
        self.fusion_head = FinalFusionHead(
            edge_channels=self.edge_final_channels, # Input channel from Edge Branch
            loc_channels=self.loc_final_channels,   # Input channel from Loc Branch
            fuse_channels=fusion_cfg.get('fuse_channels', 64),
            num_refine_blocks=fusion_cfg.get('num_refine_blocks', 2),
            attn_gen_intermediate_channels=fusion_cfg.get('attn_gen_intermediate_channels', 32)
        )
        

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the full C3Net model.

        Args:
            x (torch.Tensor): Input image tensor (B, 3, H, W). Assumed to be normalized
                              and match the `img_size` specified during initialization.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing output logits maps:
                - 'final_mask': Logits from Final Fusion Head (B, 1, H_out, W_out).
                - 'object_mask': Logits from Object Localization Branch intermediate
                                 head (B, 1, H_out, W_out).
                - 'edge_mask': Logits from Edge Detection Branch intermediate
                               head (B, 1, H_out, W_out).
                Where H_out, W_out are the final output spatial dimensions after upsampling
                (typically H/patch_size * 8, W/patch_size * 8).
        """
        # --- Input Shape Logging (Optional) ---
        # logger.debug(f"C3Net Input shape: {x.shape}")

        # --- Prepare Resized Image for ICG Module ---
        # The ICG module in the Localization Branch requires the input image resized
        # to the spatial dimensions of the features it processes (base feature resolution Hf, Wf).
        icg_input_res: Tuple[int, int] = (self.feature_h, self.feature_w)
        # Perform resizing using bilinear interpolation.
        try:
             x_resized_for_icg = F.interpolate(x, size=icg_input_res, mode='bilinear', align_corners=False)
        except Exception as e:
             logger.error(f"Failed to resize input image for ICG: {e}")
             raise e
        # Create the dictionary format expected by the Localization Branch forward method.
        x_resized_map: Dict[Tuple[int, int], torch.Tensor] = { icg_input_res: x_resized_for_icg }
        # logger.debug(f"Prepared resized image map for ICG at {icg_input_res} resolution.")

        # --- 1. Encoder: Extract Hierarchical Features ---
        # Input: (B, 3, H, W)
        # Output: List of feature maps from specified stages, e.g., 4 tensors of shape (B, D, Hf, Wf)
        features: List[torch.Tensor] = self.encoder(x)
        # Assign features based on the expected order [S1, S2, S_N-1, S_N] corresponding to output_stages.
        # This assumes output_stages = [early1, early2, late1, late2]. Adapt if stage order changes.
        f_s1, f_s2, f_n_minus_1, f_n = features[0], features[1], features[2], features[3]
        # logger.debug(f"Encoder outputs: S1={f_s1.shape}, S2={f_s2.shape}, SN-1={f_n_minus_1.shape}, SN={f_n.shape}")

        # --- 2. Decoder Branches ---
        # The branches now internally access the shared `self.upsamplers` ModuleDict.
        # Edge Refinement Pathway (ERP): Processes early features (S1, S2).
        # Input: (B, D, Hf, Wf), (B, D, Hf, Wf)
        # Output: (B, edge_final_channels, H_out, W_out)
        f_edge_features = self.edge_branch(f_s1, f_s2)
        # logger.debug(f"Edge Branch output feature shape: {f_edge_features.shape}")

        # Contextual Localization Pathway (CLP): Processes late features (N-1, N) and the resized image map.
        # Input: (B, D, Hf, Wf), (B, D, Hf, Wf), Dict
        # Output: (B, loc_final_channels, H_out, W_out)
        f_loc_features = self.loc_branch(f_n_minus_1, f_n, x_resized_map)
        # logger.debug(f"Localization Branch output feature shape: {f_loc_features.shape}")

        # --- 3. Intermediate Predictions (for Deep Supervision) ---
        # Apply lightweight 1x1 prediction heads to the final features from each branch.
        # Input: (B, edge_final_channels, H_out, W_out) -> Output: (B, 1, H_out, W_out)
        edge_logits = self.edge_pred_head(f_edge_features)
        # Input: (B, loc_final_channels, H_out, W_out) -> Output: (B, 1, H_out, W_out)
        loc_logits = self.loc_pred_head(f_loc_features)
        # logger.debug(f"Intermediate logits: Edge={edge_logits.shape}, Loc={loc_logits.shape}")

        # --- 4. Final Fusion ---
        # Combine the features from both branches using the attention-gated fusion head.
        # Input: (B, edge_final_channels, H_out, W_out), (B, loc_final_channels, H_out, W_out)
        # Output: (B, 1, H_out, W_out)
        final_logits = self.fusion_head(f_edge_features, f_loc_features)
        # logger.debug(f"Final logits shape: {final_logits.shape}")

        # --- 5. Assemble Output Dictionary ---
        # Collect all logits maps required by the loss function or for inference.
        # Ensure keys match those expected by C3NetLoss ('final_mask', 'object_mask', 'edge_mask').
        output_dict = {
            'final_mask': final_logits,   # Logits from the final fusion head
            'object_mask': loc_logits,    # Intermediate logits from Localization Branch
            'edge_mask': edge_logits      # Intermediate logits from Edge Branch
        }

        return output_dict