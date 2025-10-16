"""
Implements tailored loss functions for the C3Net architecture, optimized based on
experimental findings and analysis.

This module provides the necessary loss components for training C3Net, focusing on
addressing challenges specific to Camouflaged Object Detection (COD). It includes:
- C3NetLoss: The main loss class orchestrating deep supervision and combining
  specialized loss terms for different model outputs (edge, localization, final).
- Helper functions for:
    - Focal Loss: To handle class imbalance between foreground and background.
    - Tversky Loss: To provide a tunable balance between penalizing False Positives
      and False Negatives, with different settings applied to intermediate vs. final outputs.
    - Dice Loss on Edges: A term added to the final loss to specifically improve
      boundary alignment using ground truth edges with a controlled weight.
    - Total Variation Loss: Used as a regularization term to encourage smoothness
      in predicted background regions of the intermediate edge map.

Key features of the finalized loss strategy:
- Deep Supervision: Standard application with weights w_edge, w_loc, w_final.
- Instance Weighting: Applied to localization and final segmentation losses based on object size.
- Specialized Intermediate Edge Supervision: Unchanged (Weighted Focal + TV Reg).
- Tunable Precision/Recall Balance:
    - Intermediate Localization Loss (`loss_loc`): Uses Tversky loss prioritizing *Precision*
      (alpha=0.6, beta=0.4) to strongly penalize salient false positives potentially
      missed by the ICG module operating just before this loss point (Position P1).
    - Final Segmentation Loss (`loss_final`): Uses Tversky loss prioritizing *Recall*
      (alpha=0.4, beta=0.6) to ensure the complete object is captured after fusion,
      complementing the precision focus of the intermediate loss.
- Final Edge Guidance: Augments the final segmentation loss with a low-weighted Dice
  loss calculated only on GT edge pixels to promote sharper boundaries without
  overpowering the region segmentation.

Author  : Baber Jan
Date : 09-05-2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

# Initialize logger for this module
logger = logging.getLogger(__name__)

# --- Helper Loss Functions ---

def focal_loss_with_logits(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Computes the mean Focal Loss between predicted logits and binary targets.
    Handles optional spatial weighting internally. Always returns mean loss.
    Ensures inputs are 4D (B, C, H, W) for calculation.
    """
    # --- Input Dimension Handling & Type Conversion ---
    input_dim = inputs.dim()
    if input_dim == 2: inputs = inputs.unsqueeze(0).unsqueeze(0) # (H, W) -> (1, 1, H, W)
    elif input_dim == 3: inputs = inputs.unsqueeze(0)           # (C, H, W) -> (1, C, H, W)

    target_dim = targets.dim()
    if target_dim == 2: targets = targets.unsqueeze(0).unsqueeze(0) # (H, W) -> (1, 1, H, W)
    elif target_dim == 3: targets = targets.unsqueeze(0)           # (C, H, W) -> (1, C, H, W)
    targets = targets.float()

    # --- Channel / Shape Checks ---
    if inputs.shape[1] != 1 or targets.shape[1] != 1:
         logger.warning(f"Focal loss expects C=1. Got Input {inputs.shape}, Target {targets.shape}. Using C=1.")
         inputs = inputs[:,:1,:,:]
         targets = targets[:,:1,:,:]
    if inputs.shape[-2:] != targets.shape[-2:]:
         raise ValueError(f"Focal loss spatial dimension mismatch: Input {inputs.shape}, Target {targets.shape}")


    # --- Calculate Base BCE Loss (Per Element) ---
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none') # (B, 1, H, W)

    # --- Calculate Focal Loss Components (Per Element) ---
    probs = torch.sigmoid(inputs)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    # Add epsilon inside power for stability, especially with gamma
    modulating_factor = torch.pow(1.0 - p_t + eps, gamma)
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
    focal_loss_per_pixel = alpha_factor * modulating_factor * bce_loss # (B, 1, H, W)

    # --- Apply Spatial Weights and Reduce to Mean ---
    if weight is not None:
        try:
            if weight.dim() == 2: weight = weight.unsqueeze(0).unsqueeze(0) # H, W -> 1, 1, H, W
            elif weight.dim() == 3: weight = weight.unsqueeze(0)           # C, H, W -> 1, C, H, W (assume C=1)
            weight = weight.expand_as(focal_loss_per_pixel) # Ensure broadcastable
            # Compute weighted mean: sum(loss*w) / sum(w)
            weighted_loss_per_pixel = focal_loss_per_pixel * weight
            # Clamp weight sum to avoid division by zero if weights are all zero
            weight_sum = weight.sum().clamp(min=eps)
            mean_loss = weighted_loss_per_pixel.sum() / weight_sum
        except Exception as e:
            logger.error(f"Failed spatial weight application in focal loss: {e}. Returning unweighted mean.", exc_info=True)
            mean_loss = focal_loss_per_pixel.mean()
    else:
        mean_loss = focal_loss_per_pixel.mean()

    return mean_loss


def tversky_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float, # Weight for FP - Passed as argument
    beta: float,  # Weight for FN - Passed as argument
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Computes the Tversky loss with specified alpha and beta.
    Assumes inputs are probabilities [0, 1]. Averages loss across batch dim.
    """
    inputs = torch.clamp(inputs, min=0.0, max=1.0)
    targets = targets.float()

    if inputs.shape != targets.shape:
        raise ValueError(f"Shape mismatch: inputs {inputs.shape}, targets {targets.shape}")

    # Flatten spatial dimensions but keep batch dimension
    inputs_flat = inputs.view(inputs.shape[0], -1)
    targets_flat = targets.view(targets.shape[0], -1)

    # Calculate TP, FP, FN per sample
    TP = (inputs_flat * targets_flat).sum(1)
    FP = (inputs_flat * (1 - targets_flat)).sum(1)
    FN = ((1 - inputs_flat) * targets_flat).sum(1)

    denominator = TP + alpha * FP + beta * FN + smooth
    tversky_index = (TP + smooth) / denominator # Per sample Tversky Index

    # Clamp index for stability
    tversky_index = torch.clamp(tversky_index, min=0.0, max=1.0)
    loss = 1.0 - tversky_index # Per sample loss

    return loss.mean() # Average loss across batch


def dice_loss_on_edges(
    probs: torch.Tensor,
    gt_mask: torch.Tensor,
    gt_edges: torch.Tensor,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Computes Dice loss between predicted probabilities and the GT mask,
    considering only pixels marked by gt_edges. Averages loss across batch.
    Expects inputs shape (B, 1, H, W).
    """
    probs = probs.float()
    gt_mask = gt_mask.float()
    gt_edges = gt_edges.float() # Ensure edges are float {0., 1.}

    # --- Input Dimension Handling & Type Conversion ---
    input_dim = probs.dim()
    if input_dim == 2: probs = probs.unsqueeze(0).unsqueeze(0) # (H, W) -> (1, 1, H, W)
    elif input_dim == 3: probs = probs.unsqueeze(0)           # (C, H, W) -> (1, C, H, W)

    target_dim = gt_mask.dim()
    if target_dim == 2: gt_mask = gt_mask.unsqueeze(0).unsqueeze(0) # (H, W) -> (1, 1, H, W)
    elif target_dim == 3: gt_mask = gt_mask.unsqueeze(0)           # (C, H, W) -> (1, C, H, W)
    gt_mask = gt_mask.float()

    target_dim = gt_edges.dim()
    if target_dim == 2: gt_edges = gt_edges.unsqueeze(0).unsqueeze(0) # (H, W) -> (1, 1, H, W)
    elif target_dim == 3: gt_edges = gt_edges.unsqueeze(0)           # (C, H, W) -> (1, C, H, W)
    gt_edges = gt_edges.float()

    if not (probs.shape == gt_mask.shape == gt_edges.shape and probs.shape[1] == 1):
         raise ValueError(f"Inputs must have shape (B, 1, H, W) and match. Got P:{probs.shape}, G:{gt_mask.shape}, E:{gt_edges.shape}")

    # Apply edge mask
    P_edge = probs * gt_edges
    G_edge = gt_mask * gt_edges

    # Calculate sums per sample in batch (sum over H, W)
    intersection = (P_edge * G_edge).sum(dim=[2, 3]) # Shape: (B, 1)
    denominator = P_edge.sum(dim=[2, 3]) + G_edge.sum(dim=[2, 3]) # Shape: (B, 1)

    # Calculate Dice score per sample
    # Add smooth to numerator AND denominator
    dice_score = (2. * intersection + smooth) / (denominator + smooth) # Shape: (B, 1)
    dice_loss = 1.0 - dice_score # Shape: (B, 1)

    # Average loss over the batch
    mean_dice_loss = dice_loss.mean()

    return mean_dice_loss


def total_variation_loss(
    img: torch.Tensor,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Computes the mean anisotropic Total Variation (TV) Loss.
    Operates on the last two dimensions (H, W). Calculates mean loss across batch/channels.
    """
    if img.dim() < 4: # Expect B, C, H, W
        logger.debug(f"TV Loss input dim < 4 ({img.shape}), returning 0.")
        return torch.tensor(0.0, device=img.device, dtype=img.dtype)

    dh = torch.abs(img[..., 1:, :] - img[..., :-1, :])
    dw = torch.abs(img[..., :, 1:] - img[..., :, :-1])

    # Mean per pixel difference
    # Sum over all dimensions, divide by total number of difference elements computed
    total_diff_elements = dh.numel() + dw.numel()
    if total_diff_elements < eps:
        return torch.tensor(0.0, device=img.device, dtype=img.dtype)

    loss = (dh.sum() + dw.sum()) / total_diff_elements
    return loss


# --- Main C3Net Loss Class ---

class C3NetLoss(nn.Module):
    """
    Combined loss function for C3Net with deep supervision and finalized settings.

    Implements:
    - Deep supervision weights (w_edge, w_loc, w_final).
    - Instance weighting for loc and final losses based on object size.
    - Intermediate edge loss: Spatially weighted Focal + TV regularization.
    - Intermediate localization loss: Instance Weighted (Focal + Tversky (Precision Focus)).
    - Final loss: Instance Weighted (Focal + Tversky (Recall Focus) + Weighted Edge Dice).

    Args:
        w_edge (float): Weight for intermediate edge loss. Default: 1.0.
        w_loc (float): Weight for intermediate localization loss. Default: 1.0.
        w_final (float): Weight for final fusion head loss. Default: 1.0.
        focal_alpha (float): Alpha for Focal Loss component. Default: 0.25.
        focal_gamma (float): Gamma for Focal Loss component. Default: 2.0.
        tversky_alpha_loc (float): Alpha for Tversky in loc loss (Precision). Default: 0.7.
        tversky_beta_loc (float): Beta for Tversky in loc loss (Precision). Default: 0.3.
        tversky_alpha_final (float): Alpha for Tversky in final loss (Recall). Default: 0.3.
        tversky_beta_final (float): Beta for Tversky in final loss (Recall). Default: 0.7.
        final_edge_dice_weight (float): Weight for edge Dice in final loss. Default: 0.1.
        edge_focal_weight (float): Max spatial weight near GT edges for interm. edge loss. Default: 5.0.
        edge_reg_weight (float): Weight for TV regularization in interm. edge loss. Default: 0.1.
        edge_dist_cutoff (int): Iterations for edge weight approx. Default: 3.
        instance_weight_k (float): Factor for instance weighting (0 disables). Default: 2.0.
        instance_weight_low_thresh (float): Lower FG ratio threshold. Default: 0.01.
        instance_weight_high_thresh (float): Upper FG ratio threshold. Default: 0.9.
        eps (float): Epsilon for numerical stability. Default: 1e-7.
    """
    def __init__(
        self,
        w_edge: float = 1.0,
        w_loc: float = 1.0,
        w_final: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        # Separate Tversky parameters based on reasoning
        tversky_alpha_loc: float = 0.6, # Precision focus for Loc Branch loss
        tversky_beta_loc: float = 0.4,
        tversky_alpha_final: float = 0.4, # Recall focus for Final loss
        tversky_beta_final: float = 0.6,
        # Final edge loss weight (decisive low value)
        final_edge_dice_weight: float = 0.1,
        # Intermediate edge loss params 
        edge_focal_weight: float = 5.0,
        edge_reg_weight: float = 0.1,
        edge_dist_cutoff: int = 3,
        # Instance weighting params
        instance_weight_k: float = 2.0,
        instance_weight_low_thresh: float = 0.01,
        instance_weight_high_thresh: float = 0.9,
        eps: float = 1e-7
    ):
        super().__init__()
        # --- Validation ---
        if not all(w >= 0 for w in [w_edge, w_loc, w_final, final_edge_dice_weight]):
            raise ValueError("Loss weights must be >= 0.")
        if not (0 < instance_weight_low_thresh < instance_weight_high_thresh < 1):
             raise ValueError("Instance weight thresholds: 0 < low < high < 1.")
        if not (tversky_alpha_loc >= 0 and tversky_beta_loc >= 0 and (tversky_alpha_loc + tversky_beta_loc) > eps):
             raise ValueError("Tversky loc alpha/beta must be non-negative and sum > eps.")
        if not (tversky_alpha_final >= 0 and tversky_beta_final >= 0 and (tversky_alpha_final + tversky_beta_final) > eps):
             raise ValueError("Tversky final alpha/beta must be non-negative and sum > eps.")
        if not (0 <= focal_alpha <= 1): raise ValueError("focal_alpha must be in [0, 1]")
        if not (focal_gamma >= 0): raise ValueError("focal_gamma must be >= 0")
        if not (edge_focal_weight >= 1.0): raise ValueError("edge_focal_weight must be >= 1.0")
        if not (edge_reg_weight >= 0): raise ValueError("edge_reg_weight must be >= 0")
        if not (edge_dist_cutoff >= 1): raise ValueError("edge_dist_cutoff must be >= 1")
        if not (instance_weight_k >= 0): raise ValueError("instance_weight_k must be >= 0")

        # --- Store Parameters ---
        self.w_edge, self.w_loc, self.w_final = w_edge, w_loc, w_final
        self.focal_alpha, self.focal_gamma = focal_alpha, focal_gamma
        # Store separate Tversky params
        self.tversky_alpha_loc, self.tversky_beta_loc = tversky_alpha_loc, tversky_beta_loc
        self.tversky_alpha_final, self.tversky_beta_final = tversky_alpha_final, tversky_beta_final
        # Store final edge weight
        self.final_edge_dice_weight = final_edge_dice_weight
        # Store other params
        self.edge_focal_weight = edge_focal_weight
        self.edge_reg_weight = edge_reg_weight
        self.edge_dist_cutoff = max(1, edge_dist_cutoff) # Ensure at least 1 iter
        self.instance_weight_k = instance_weight_k
        self.instance_weight_low_thresh = instance_weight_low_thresh
        self.instance_weight_high_thresh = instance_weight_high_thresh
        self.eps = eps

        # Avg Pooling for intermediate edge loss spatial weights
        self.edge_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False)

    # --- Helper Methods ---

    def _compute_instance_weight(self, gt_mask: torch.Tensor) -> float:
        """ Calculates sample weight based on foreground ratio. """
        if self.instance_weight_k <= 0: return 1.0
        if gt_mask.dim() == 3 and gt_mask.shape[0] == 1: gt_mask_2d = gt_mask.squeeze(0)
        elif gt_mask.dim() == 2: gt_mask_2d = gt_mask
        else: logger.warning(f"Instance weight GT mask shape unexpected: {gt_mask.shape}. Using weight 1.0."); return 1.0
        num_pixels = float(gt_mask_2d.numel());
        if num_pixels == 0: return 1.0
        foreground_pixels = gt_mask_2d.sum().float()
        fg_ratio = torch.clamp(foreground_pixels / (num_pixels + self.eps), self.eps, 1.0 - self.eps)
        max_weight = 1.0 + self.instance_weight_k; weight = 1.0
        if fg_ratio <= self.instance_weight_low_thresh or fg_ratio >= self.instance_weight_high_thresh: weight = max_weight
        else:
            mid = 0.5; scale = (mid - fg_ratio) / (mid - self.instance_weight_low_thresh + self.eps) if fg_ratio < mid else (fg_ratio - mid) / (self.instance_weight_high_thresh - mid + self.eps)
            weight = 1.0 + self.instance_weight_k * scale
        return min(max(weight.item() if isinstance(weight, torch.Tensor) else weight, 1.0), max_weight)

    @torch.no_grad()
    def _compute_spatial_weights_edge(self, gt_edges: torch.Tensor) -> torch.Tensor:
        """ Approximates distance transform via Average Pooling for edge spatial weights. """
        if gt_edges.dim() == 3 and gt_edges.shape[0] == 1: gt_edges_in = gt_edges # Input is (1, H, W)
        elif gt_edges.dim() == 2: gt_edges_in = gt_edges.unsqueeze(0) # H, W -> 1, H, W
        else: logger.warning(f"Edge spatial weight GT edges shape unexpected: {gt_edges.shape}. Using as is."); gt_edges_in = gt_edges
        blurred_edges = gt_edges_in.unsqueeze(1).float() # Add C dim: (1, 1, H, W)
        for _ in range(self.edge_dist_cutoff): blurred_edges = self.edge_pool(blurred_edges)
        blurred_edges = blurred_edges.squeeze(1).clamp(0.0, 1.0) # Remove C dim -> (1, H, W)
        spatial_weight_map = 1.0 + (self.edge_focal_weight - 1.0) * blurred_edges
        return torch.clamp(spatial_weight_map, min=1.0, max=self.edge_focal_weight) # Shape (1, H, W)

    def _compute_edge_loss_single(self, pred_edge_logits: torch.Tensor, gt_edges: torch.Tensor) -> torch.Tensor:
        """ Computes Weighted Focal + TV loss for a single edge sample (inputs B=1, C=1, H, W). """
        # 1. Compute Spatial Weights
        spatial_weight_map = self._compute_spatial_weights_edge(gt_edges) # (1, H, W)

        # 2. Calculate Weighted Focal Loss (Mean)
        focal_loss_val = focal_loss_with_logits(
            inputs=pred_edge_logits, targets=gt_edges,
            alpha=self.focal_alpha, gamma=self.focal_gamma,
            weight=spatial_weight_map, eps=self.eps # Pass spatial weights
        )

        # 3. Calculate Background TV Regularization (Mean)
        tv_reg_mean = torch.tensor(0.0, device=pred_edge_logits.device, dtype=pred_edge_logits.dtype)
        if self.edge_reg_weight > 0:
            with torch.no_grad():
                # Background mask where spatial weight is close to 1
                background_mask = (spatial_weight_map <= 1.0 + self.eps).float() # (1, H, W)
            # Need to ensure pred_edge_logits is B=1,C=1,H,W for TV helper
            if pred_edge_logits.dim() == 3: pred_edge_logits_bchw = pred_edge_logits.unsqueeze(0)
            else: pred_edge_logits_bchw = pred_edge_logits # Assume already BCHW
            pred_edge_probs = torch.sigmoid(pred_edge_logits_bchw) # (1, 1, H, W)
            # Apply background mask (needs compatible shape)
            tv_input = pred_edge_probs * background_mask.unsqueeze(1) # Add C dim to mask
            tv_reg_mean = total_variation_loss(tv_input, eps=self.eps)

        loss = focal_loss_val + self.edge_reg_weight * tv_reg_mean
        return loss

    # Modified to accept specific alpha/beta for Tversky
    def _compute_segmentation_loss_single(
        self,
        pred_logits: torch.Tensor,
        gt_mask: torch.Tensor,
        tversky_alpha: float, # Pass alpha
        tversky_beta: float   # Pass beta
        ) -> torch.Tensor:
        """ Computes Focal + Tversky loss for a single sample with specific alpha/beta. """
        # Assume inputs are B=1, C=1, H, W
        # 1. Calculate Focal Loss (Mean)
        focal = focal_loss_with_logits(
            inputs=pred_logits, targets=gt_mask,
            alpha=self.focal_alpha, gamma=self.focal_gamma,
            weight=None, eps=self.eps # No spatial weight here
        )
        # 2. Calculate Tversky Loss with passed alpha/beta
        probs = torch.sigmoid(pred_logits)
        tversky = tversky_loss(
            inputs=probs, targets=gt_mask,
            alpha=tversky_alpha, beta=tversky_beta, # Use passed params
            smooth=self.eps
        )
        # 3. Combine Focal and Tversky losses
        loss = focal + tversky
        return loss

    def forward(
        self,
        predictions: Dict[str, List[torch.Tensor]], # Dict: stage -> List[Tensor(C=1, H, W)]
        gt_masks: List[torch.Tensor],              # List[Tensor(C=1, H, W)]
        gt_edges: List[torch.Tensor]               # List[Tensor(C=1, H, W)]
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the final combined C3Net loss across a batch using list inputs.
        Uses different Tversky params for Loc vs Final loss.
        Includes weighted edge Dice loss for Final loss.
        """
        final_preds_list = predictions['final_mask']
        loc_preds_list = predictions['object_mask']
        edge_preds_list = predictions['edge_mask']

        batch_size = len(gt_masks)
        if not (len(final_preds_list) == batch_size and len(loc_preds_list) == batch_size and len(edge_preds_list) == batch_size and len(gt_edges) == batch_size):
            raise ValueError(f"Input lists must have same length (batch size). Mismatch found.")

        # --- Loss Accumulation Initialization ---
        output_device = final_preds_list[0].device if batch_size > 0 else torch.device('cpu')
        zero_loss = torch.tensor(0.0, device=output_device, dtype=torch.float32)
        accumulated_losses = {'loc': [], 'final': [], 'edge': []}
        instance_weights_list = []

        # --- Per-Sample Loss Calculation Loop ---
        for i in range(batch_size):
            # Get tensors for sample i, move to device, ensure float
            final_pred_i = final_preds_list[i] 
            loc_pred_i = loc_preds_list[i] 
            edge_pred_i = edge_preds_list[i] 
            gt_mask_i = gt_masks[i] 
            gt_edge_i = gt_edges[i] 


            # --- Calculate Instance Weight ---
            instance_weight = self._compute_instance_weight(gt_mask_i)
            instance_weights_list.append(instance_weight)

            # --- Intermediate Edge Loss ---
            loss_edge_i = zero_loss.clone()
            if self.w_edge > 0:
                loss_edge_i = self._compute_edge_loss_single(edge_pred_i, gt_edge_i)
            accumulated_losses['edge'].append(loss_edge_i)

            # --- Intermediate Localization Loss (Precision Focus) ---
            loss_loc_i = self._compute_segmentation_loss_single(
                loc_pred_i, gt_mask_i,
                tversky_alpha=self.tversky_alpha_loc,
                tversky_beta=self.tversky_beta_loc
            )
            accumulated_losses['loc'].append(loss_loc_i)

            # --- Final Segmentation Loss (Recall Focus + Edge Dice) ---
            # 1. Region Loss Component
            loss_final_region_i = self._compute_segmentation_loss_single(
                final_pred_i, gt_mask_i,
                tversky_alpha=self.tversky_alpha_final,
                tversky_beta=self.tversky_beta_final
            )
            # 2. Edge Dice Loss Component
            loss_final_edge_i = zero_loss.clone()
            if self.final_edge_dice_weight > 0:
                loss_final_edge_i = dice_loss_on_edges(
                    torch.sigmoid(final_pred_i), gt_mask_i, gt_edge_i, smooth=self.eps
                )
            # 3. Combine final loss components
            loss_final_i = loss_final_region_i + self.final_edge_dice_weight * loss_final_edge_i
            accumulated_losses['final'].append(loss_final_i)

        
        # --- Average Losses Across Batch ---
        instance_weights_tensor = torch.tensor(instance_weights_list, device=output_device).float()
        total_instance_weight_accum = instance_weights_tensor.sum()

        # Calculate weighted average for loc and final loss
        avg_loss_loc = (instance_weights_tensor * torch.stack(accumulated_losses['loc'])).sum() / (total_instance_weight_accum + self.eps)
        avg_loss_final = (instance_weights_tensor * torch.stack(accumulated_losses['final'])).sum() / (total_instance_weight_accum + self.eps)
        # Simple average for edge loss
        avg_loss_edge = torch.stack(accumulated_losses['edge']).mean() if self.w_edge > 0 else zero_loss.clone()

        # --- Combine Losses with Overall Branch Weights ---
        total_loss = (self.w_final * avg_loss_final +
                      self.w_loc * avg_loss_loc +
                      self.w_edge * avg_loss_edge)

        # --- Return Dictionary of Losses ---
        return {
            'loss': total_loss,
            'loss_final': avg_loss_final.detach(),
            'loss_loc': avg_loss_loc.detach(),
            'loss_edge': avg_loss_edge.detach()
        }