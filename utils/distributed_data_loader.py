"""
Distributed data loading utilities for Camouflaged Object Detection (COD).

This module extends the base data_loader functionality with distributed training support:
1. DistributedSampler integration for efficient multi-GPU data partitioning
2. Gradient accumulation support for effective larger batch sizes
3. Optimized worker initialization for multi-node training

Key Features:
- Per-GPU data sharding to eliminate redundant processing
- Deterministic data distribution with epoch-based shuffling
- Support for gradient accumulation steps
- Memory-efficient data prefetching
"""

import logging
import math
import os
from typing import Dict, List, Tuple, Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, ConcatDataset, DistributedSampler

from utils.data_loader import CODDataset, collate_fn

logger = logging.getLogger(__name__)

def worker_init_fn(worker_id: int) -> None:
    """
    Initialize worker processes with different random seeds.
    
    Args:
        worker_id: Worker process ID
    """
    # Set random seed based on worker_id and process rank
    worker_seed = torch.initial_seed() % 2**32
    worker_seed = worker_seed + worker_id + torch.distributed.get_rank() * 1000
    torch.manual_seed(worker_seed)
    
    # Set environment variable for worker
    os.environ["WORKER_ID"] = str(worker_id)


def get_distributed_training_loaders(
    dataset_dirs: List[str],
    model_config: Dict,
    batch_size: int = 16,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    grad_accum_steps: int = 1,
    prefetch_factor: int = 2,
    world_size: int = 1,
    rank: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create distributed training and validation data loaders.
    
    Args:
        dataset_dirs: List of dataset root directories
        model_config: Model configuration containing:
            - image_processing.target_size: Size to resize images to
            - image_processing.normalize_mean: RGB normalization means
            - image_processing.normalize_std: RGB normalization stds
        batch_size: Per-GPU batch size for data loading
        num_workers: Number of worker processes per GPU
        val_ratio: Ratio of training data for validation (0 for no validation)
        grad_accum_steps: Number of gradient accumulation steps (effective bs = batch_size * steps * world_size)
        prefetch_factor: Number of batches to prefetch per worker
        world_size: Total number of processes in distributed training
        rank: Rank of current process
        seed: Random seed for dataset splitting and sampling
        
    Returns:
        Tuple containing:
        - Distributed training DataLoader
        - Distributed validation DataLoader (None if val_ratio=0)
    
    Notes:
        - Each GPU receives a distinct subset of the data
        - Effective batch size = batch_size * grad_accum_steps * world_size
        - Set drop_last=True for stable training with fixed-size batches
    """
    # Extract processor parameters from config
    processor_params = {
        'target_size': model_config['image_processing']['target_size'],
        'normalize_mean': tuple(model_config['image_processing']['normalize_mean']),
        'normalize_std': tuple(model_config['image_processing']['normalize_std'])
    }
    
    # Create datasets from all training directories
    train_datasets = []
    for train_dir in dataset_dirs:
        train_path = os.path.join(train_dir, 'train')
        if not os.path.exists(train_path):
            continue
            
        train_datasets.append(
            CODDataset(
                root_dir=train_path,
                processor_params=processor_params,
                is_train=True
            )
        )
    
    if not train_datasets:
        raise ValueError("No valid training datasets found")
        
    # Combine all training datasets
    combined_dataset = ConcatDataset(train_datasets)
    
    # Calculate effective batch size
    effective_batch_size = batch_size * grad_accum_steps * world_size
    logger.info(f"Effective batch size: {effective_batch_size} " 
               f"(per-GPU batch: {batch_size}, accum steps: {grad_accum_steps}, GPUs: {world_size})")
    
    if val_ratio > 0:
        # Split combined dataset with fixed seed for reproducibility
        train_size = int((1 - val_ratio) * len(combined_dataset))
        val_size = len(combined_dataset) - train_size
        
        generator = torch.Generator().manual_seed(seed)
        train_subset, val_subset = torch.utils.data.random_split(
            combined_dataset,
            [train_size, val_size],
            generator=generator
        )
        
        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_subset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed
        )
        
        val_sampler = DistributedSampler(
            val_subset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=seed
        )
        
        # Create distributed loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=False,  # Sampler handles shuffling
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,  # Drop incomplete batches for stable training
            collate_fn=collate_fn,
            sampler=train_sampler,
            prefetch_factor=prefetch_factor,
            worker_init_fn=worker_init_fn,
            persistent_workers=(num_workers > 0)
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,  # Keep all validation samples
            collate_fn=collate_fn,
            sampler=val_sampler,
            prefetch_factor=prefetch_factor,
            worker_init_fn=worker_init_fn,
            persistent_workers=(num_workers > 0)
        )
        
        return train_loader, val_loader
        
    else:
        # No validation split
        train_sampler = DistributedSampler(
            combined_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed
        )
        
        train_loader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=False,  # Sampler handles shuffling
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
            prefetch_factor=prefetch_factor,
            worker_init_fn=worker_init_fn,
            persistent_workers=(num_workers > 0)
        )
        return train_loader, None


def get_distributed_test_loaders(
    dataset_dirs: List[str],
    model_config: Dict,
    batch_size: int = 16,
    num_workers: int = 4,
    world_size: int = 1,
    rank: int = 0
) -> Dict[str, DataLoader]:
    """
    Create distributed test data loaders for each dataset.
    
    Args:
        dataset_dirs: List of dataset root directories
        model_config: Model configuration
        batch_size: Per-GPU batch size
        num_workers: Number of worker processes per GPU
        world_size: Total number of processes in distributed training
        rank: Rank of current process
        
    Returns:
        Dictionary mapping dataset names to distributed DataLoaders
    """
    # Extract processor parameters from config
    processor_params = {
        'target_size': model_config['image_processing']['target_size'],
        'normalize_mean': tuple(model_config['image_processing']['normalize_mean']),
        'normalize_std': tuple(model_config['image_processing']['normalize_std'])
    }
    
    test_loaders = {}
    for test_dir in dataset_dirs:
        test_path = os.path.join(test_dir, 'test')
        if not os.path.exists(test_path):
            continue
            
        dataset_name = os.path.basename(test_dir)
        test_dataset = CODDataset(
            root_dir=test_path,
            processor_params=processor_params,
            is_train=False
        )
        
        # Create distributed sampler for test set
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        
        test_loaders[dataset_name] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,  # Keep all test samples
            collate_fn=collate_fn,
            sampler=test_sampler,
            prefetch_factor=2,
            persistent_workers=(num_workers > 0)
        )
    
    if not test_loaders:
        raise ValueError("No valid test datasets found")
        
    return test_loaders