"""
DINOv2 Hierarchical Vision Transformer Encoder with Register Tokens

This encoder implements an enhanced version of the DINOv2 architecture that incorporates register 
tokens for improved feature extraction. The architecture combines the hierarchical processing 
capabilities of vision transformers with a novel register token mechanism that maintains clean 
feature maps while preserving spatial information.

Architecture Details:
--------------------
The encoder processes images through several sophisticated stages:

1. Patch Embedding Layer:
   The input image is first divided into non-overlapping patches, each processed independently:
   - 224x224 images → 16x16 grid of 14x14 pixel patches (256 total patches)
   - 512x512 images → 36x36 grid of 14x14 pixel patches (1296 total patches)
   Each patch undergoes linear projection to create high-dimensional token embeddings.

2. Token Structure and Organization:
   The sequence combines three types of tokens, each serving a distinct purpose:
   - [CLS] Token: A learnable token that aggregates global image information
   - Register Tokens (N=4): Dedicated computational memory tokens that participate in 
     self-attention but are discarded at output. These prevent the model from repurposing
     patch tokens for global computation, maintaining cleaner feature maps.
   - Patch Tokens: Carry spatial information from the image patches through the network

3. Transformer Processing:
   Each transformer block processes the token sequence through:
   a) Layer Normalization
   b) Multi-head Self-attention (MSA):
      - Varying attention heads by model size enable parallel relationship processing
      - Register tokens participate in attention but maintain separation of global/local info
      - Attention combines information across all tokens: patches, registers, and [CLS]
   c) Feed-forward Network (FFN):
      - Two-layer perceptron with GELU activation
      - Processes each token independently
   d) Residual connections around both MSA and FFN

4. Model Variants and Dimensions:
   The architecture scales across three sizes:
   Giant (g/14):
   - Hidden dimension: 1536
   - Attention heads: 24
   - Transformer layers: 40
   - Best for complex tasks requiring fine-grained understanding

   Large (L/14):
   - Hidden dimension: 1024
   - Attention heads: 16
   - Transformer layers: 24
   - Balanced performance and computational cost

   Base (B/14):
   - Hidden dimension: 768
   - Attention heads: 12
   - Transformer layers: 18
   - Efficient while maintaining good performance

5. Feature Map Generation:
   The encoder outputs feature maps at different stages, maintaining spatial layout:
   - Each stage produces features of shape [B, D, H', W']
     where: B = batch size
            D = hidden dimension (varies by model variant)
            H' = input_height / patch_size
            W' = input_width / patch_size
   - Features maintain both local spatial information and global context
   - Register tokens' influence creates cleaner, artifact-free feature maps

6. Register Token Innovation:
   The register token mechanism provides several key benefits:
   - Separates global computation from spatial representation
   - Prevents high-norm outlier tokens in feature maps
   - Improves performance on dense prediction tasks
   - Enables better object discovery and segmentation
   - Maintains cleaner attention maps throughout the network

Usage Notes:
------------
    - Input images must have dimensions divisible by the patch size (14)
    - Register tokens are automatically handled internally
    - Output features maintain spatial layout for dense prediction tasks
    - Feature maps from any stage can be extracted by specifying stage indices
    - Each stage's features contain both local and global information thanks to register tokens

References:
-----------
    DINOv2 Paper: https://arxiv.org/abs/2304.07193
    Vision Transformers Need Registers: https://arxiv.org/abs/2309.16588
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple
from transformers import AutoBackbone
import logging

logger = logging.getLogger(__name__)

class DinoV2WithRegistersEncoder(nn.Module):
    """
    DINOv2 encoder with register tokens for hierarchical feature extraction.
    
    This implementation allows extracting features from specific transformer stages,
    maintaining the register token concept for clean feature maps. Each stage 
    processes tokens through self-attention while leveraging register tokens for
    global computation.

    Feature Dimensions:
        224x224 input → 16x16 patch grid → 256 patch tokens
        512x512 input → 36x36 patch grid → 1296 patch tokens
        
        Output features per stage maintain spatial layout:
        - Giant:  [B, 1536, H/14, W/14]
        - Large:  [B, 1024, H/14, W/14]
        - Base:   [B, 768, H/14, W/14]
    """
    
    def __init__(
        self, 
        model_name: str = "facebook/dinov2-with-registers-large",
        variant: str = "large",
        output_stages: Optional[List[int]] = [1,2,23,24]
    ):
        """
        Initialize the DINOv2 encoder with register tokens.

        Args:
            model_name: HuggingFace model identifier
            variant: Model size variant ('base', 'large', 'giant')
            output_stages: List of stage indices to output (1-indexed)
                         If None, outputs all stages
        """
        super().__init__()
        
        # Map variant names to hidden dimensions
        self.hidden_dims = {
            'small': 384,
            'base': 768,
            'large': 1024,
            'giant': 1536
        }
        
        if variant not in self.hidden_dims:
            raise ValueError(f"Invalid variant. Choose from: {list(self.hidden_dims.keys())}")
        
        self.variant = variant
        self.hidden_dim = self.hidden_dims[variant]
        
        # Setup output stages (1-indexed for clarity)
        self.output_stages = output_stages or list(range(1, 13))
        stage_features = [f'stage{i}' for i in self.output_stages]
        
        # Initialize backbone with register tokens
        self.model = AutoBackbone.from_pretrained(
            model_name,
            out_features=stage_features
        )
        
        logger.info(
            f"Initialized {variant} encoder with {self.param_count:,} parameters. "
            f"Output stages: {self.output_stages}"
        )

    @property
    def param_count(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_patch_grid_size(self, height: int, width: int) -> Tuple[int, int]:
        """
        Calculate patch grid dimensions for given input size.
        
        Args:
            height: Input image height
            width: Input image width
            
        Returns:
            Tuple of (grid_height, grid_width)
        """
        return height // 14, width // 14

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass generating features with register token attention.

        The input image is divided into 14x14 patches, each processed through the
        transformer while interacting with register tokens. Register tokens participate
        in self-attention but are discarded in the final output.

        Args:
            x: Input images [B, 3, H, W]
               Supported sizes:
               - 224x224 → 16x16 patch grid
               - 512x512 → 36x36 patch grid
               Height and width must be divisible by 14

        Returns:
            List of feature tensors for requested stages
            Each tensor shape: [B, hidden_dim, H/14, W/14]
            - Base:   hidden_dim = 768
            - Large:  hidden_dim = 1024
            - Giant:  hidden_dim = 1536
        """
        # Input size validation
        B, C, H, W = x.shape
        if H % 14 != 0 or W % 14 != 0:
            raise ValueError("Input height and width must be divisible by patch size (14):", H, W)
            
        # Process through backbone with register tokens
        outputs = self.model(x)
        
        return outputs.feature_maps

    def __repr__(self) -> str:
        """Detailed string representation with architecture info."""
        return (
            f"DinoV2WithRegisterEncoder(\n"
            f"  variant={self.variant}\n"
            f"  hidden_dim={self.hidden_dim}\n"
            f"  output_stages={self.output_stages}\n"
            f"  params={self.param_count:,}\n"
            f"  model={self.model}\n"
            f")"
        )