"""
Distributed training utilities for multi-GPU model optimization.

This module provides functions and classes to optimize distributed model training:
1. Activation checkpointing for transformer layers
2. Gradient synchronization and reduction
3. Memory-efficient parameter initialization
4. Performance monitoring

Usage:
    from utils.distributed_utils import apply_activation_checkpointing, gather_metrics
"""

import logging
import torch
import torch.distributed as dist
from typing import Dict, Any, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

def is_dist_initialized():
    """Check if distributed training is initialized."""
    return dist.is_available() and dist.is_initialized()

def get_world_size():
    """Get the world size (total number of processes)."""
    if is_dist_initialized():
        return dist.get_world_size()
    return 1

def get_rank():
    """Get the rank of the current process."""
    if is_dist_initialized():
        return dist.get_rank()
    return 0

def is_main_process():
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0

def apply_activation_checkpointing(model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply activation checkpointing to transformer layers.
    
    Args:
        model: PyTorch model with transformer blocks
        
    Returns:
        Model with activation checkpointing applied
        
    Notes:
        - Significantly reduces memory usage during training
        - Slightly increases computation time (recomputes activations in backward pass)
        - Works with DINOv2, ViT, and similar transformer architectures
    """
    try:
        from torch.utils.checkpoint import checkpoint
        
        # Handle DINOv2 model structure
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'model'):
            base_model = model.encoder.model
            
            # Apply to transformer blocks
            if hasattr(base_model, 'blocks'):
                blocks = base_model.blocks
                
                # Apply checkpointing to each block
                for i, block in enumerate(blocks):
                    # Store the original forward method
                    block._original_forward = block.forward
                    
                    # Define a checkpointed forward method
                    def make_checkpointed_forward(block):
                        original_forward = block._original_forward
                        
                        def checkpointed_forward(*args, **kwargs):
                            return checkpoint(original_forward, *args, **kwargs)
                        
                        return checkpointed_forward
                    
                    # Replace the forward method with the checkpointed version
                    block.forward = make_checkpointed_forward(block)
                
                logger.info(f"Applied activation checkpointing to {len(blocks)} transformer blocks")
            else:
                logger.warning("Could not find transformer blocks for checkpointing")
        else:
            logger.warning("Model structure not compatible with automatic activation checkpointing")
    
    except ImportError:
        logger.warning("torch.utils.checkpoint not available, skipping activation checkpointing")
    except Exception as e:
        logger.warning(f"Error applying activation checkpointing: {e}")
    
    return model

def gather_metrics(metrics: Dict[str, Union[float, torch.Tensor]]) -> Dict[str, float]:
    """
    Gather and average metrics from all processes.
    
    Args:
        metrics: Dictionary of metrics to gather
        
    Returns:
        Dictionary of averaged metrics
    """
    if not is_dist_initialized():
        return metrics
    
    gathered_metrics = {}
    
    for key, value in metrics.items():
        # Skip non-numeric values
        if not isinstance(value, (float, int, torch.Tensor)):
            gathered_metrics[key] = value
            continue
        
        # Convert to tensor if needed
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32, device=f'cuda:{get_rank()}')
        
        # Move to correct device
        if value.device.type != 'cuda':
            value = value.to(f'cuda:{get_rank()}')
        
        # Sum across all processes
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        
        # Average by world size
        value = value / get_world_size()
        
        # Convert back to Python scalar
        gathered_metrics[key] = value.item()
    
    return gathered_metrics

def sync_model_parameters(model: torch.nn.Module) -> None:
    """
    Ensure model parameters are synchronized across all processes.
    
    Args:
        model: PyTorch model to synchronize
    """
    if not is_dist_initialized():
        return
    
    # Get model parameters
    params = [p.data for p in model.parameters() if p.requires_grad]
    
    # Synchronize parameters from rank 0
    for p in params:
        dist.broadcast(p, src=0)
    
    # Add barrier to ensure synchronization
    dist.barrier()

def all_gather_object(obj: Any) -> List[Any]:
    """
    Gather objects from all processes.
    
    Args:
        obj: Python object to gather
        
    Returns:
        List of objects from all processes
    """
    if not is_dist_initialized():
        return [obj]
    
    world_size = get_world_size()
    gathered_objs = [None for _ in range(world_size)]
    
    # Use built-in all_gather_object if available
    if hasattr(dist, 'all_gather_object'):
        dist.all_gather_object(gathered_objs, obj)
        return gathered_objs
    
    # Fallback implementation
    import pickle
    
    # Serialize the object
    buffer = pickle.dumps(obj)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(f'cuda:{get_rank()}')
    
    # Get the size of the tensor for each process
    local_size = torch.tensor([tensor.numel()], device=f'cuda:{get_rank()}')
    size_list = [torch.tensor([0], device=f'cuda:{get_rank()}') for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    
    # Create a list of tensors to hold all gathered data
    size_list = [size.item() for size in size_list]
    max_size = max(size_list)
    
    # Pad tensor to max size for gathering
    if local_size != max_size:
        padding = torch.zeros(max_size - local_size, dtype=torch.uint8, device=f'cuda:{get_rank()}')
        tensor = torch.cat((tensor, padding), dim=0)
    
    # Gather all tensors
    tensor_list = [torch.zeros(max_size, dtype=torch.uint8, device=f'cuda:{get_rank()}') 
                  for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    
    # Deserialize the gathered data
    for i, (size, tensor) in enumerate(zip(size_list, tensor_list)):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        gathered_objs[i] = pickle.loads(buffer)
    
    return gathered_objs