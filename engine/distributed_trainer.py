"""
Distributed Trainer for Camouflaged Object Detection models.

This module extends the base Trainer with distributed training capabilities:
1. PyTorch DistributedDataParallel (DDP) integration
2. Multi-node/multi-GPU coordination
3. Gradient accumulation for effective larger batch sizes
4. Activation checkpointing for memory efficiency
5. Distributed validation and metrics collection

Key Optimizations:
- Layer-wise learning rate with warmup
- Automatic mixed precision with gradient scaling
- Activation checkpointing for transformer layers
- Per-rank file I/O and metrics reduction
- Efficient checkpoint handling
"""

import logging
import json
import time
import os
import pickle
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from pathlib import Path
import copy

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from utils.metrics import MetricsProcessor
from utils.loss_functions import C3NetLoss
from utils.run_manager import DirectoryManager
from models.c3net import C3Net
from utils.distributed_data_loader import get_distributed_training_loaders

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self, name='', fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize_between_processes(self):
        """Synchronize meter across distributed processes."""
        if not dist.is_available() or not dist.is_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = t[0]
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class DistributedTrainer:
    """
    Manages distributed training across multiple GPUs and nodes.
    
    Features:
    1. Automatic scaling of batch size based on GPU count
    2. Synchronization of metrics across devices
    3. Efficient model checkpointing from rank 0
    4. Gradient accumulation and normalization
    5. Memory-optimized transformer training
    """
    
    def __init__(
        self,
        config: Dict,
        dir_manager: DirectoryManager,
        local_rank: int,
        world_size: int,
        grad_accum_steps: int = 1,
    ):
        """
        Initialize the distributed trainer.
        
        Args:
            config: Training configuration dictionary
            dir_manager: Directory manager for outputs
            local_rank: Local rank of this process
            world_size: Total number of processes
            grad_accum_steps: Number of gradient accumulation steps
        """
        # Basic setup
        self.config = config['training']
        self.model_config = config['model']
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main_process = (local_rank == 0)
        self.device = torch.device(f'cuda:{local_rank}')
        self.grad_accum_steps = grad_accum_steps
        self.mode_config = self.config.get('mode', {'type': 'train'})
        
        # Only log from main process
        if self.is_main_process:
            logging.info(f"Initializing distributed trainer with {world_size} processes")
            logging.info(f"Gradient accumulation steps: {grad_accum_steps}")
            
        # Initialize model and training components
        self.model = self._initialize_model()
        
        # Training parameters
        self.batch_size = self.config['batch_size']
        self.num_epochs = self.config['num_epochs']
        self.grad_clip = self.config.get('gradient_clip', 1.0)
        self.early_stop_patience = self.config.get('early_stop_patience', 15)
        self.save_freq = self.config.get('save_freq', 1)
        
        # Initialize components
        self._setup_optimization()
        self.criterion = C3NetLoss(**self.config['loss']).to(self.device)
        if self.is_main_process:
            self.monitor = DistributedTrainingMonitor(dir_manager)
        self.metrics_processor = MetricsProcessor()
        
        # Mixed precision setup
        self.use_amp = self.config.get('use_amp', True)
        self.scaler = amp.GradScaler(enabled=self.use_amp)

        # Create metric trackers
        self.train_meters = {
            'loss': AverageMeter('Loss', ':.4f'),
            'loss_edge': AverageMeter('Edge Loss', ':.4f'),
            'loss_loc': AverageMeter('Loc Loss', ':.4f'),
            'loss_final': AverageMeter('Final Loss', ':.4f'),
            'data_time': AverageMeter('Data', ':.3f'),
            'batch_time': AverageMeter('Batch', ':.3f'),
        }
        
        self.val_meters = {
            'loss': AverageMeter('Loss', ':.4f'),
            'weighted_f': AverageMeter('F-measure', ':.4f'),
            's_alpha': AverageMeter('S-alpha', ':.4f'),
            'mae': AverageMeter('MAE', ':.4f'),
            'data_time': AverageMeter('Data', ':.3f'),
            'batch_time': AverageMeter('Batch', ':.3f'),
        }
        
        # Set starting epoch
        self.start_epoch = 0
        
        # Load checkpoint if resuming or fine-tuning
        if self.mode_config.get('checkpoint_path'):
            self._load_checkpoint()
    
    def _initialize_model(self) -> torch.nn.Module:
        """Initialize model with DDP and memory optimizations."""
        # Create base model
        model = C3Net(self.model_config).to(self.device)
        
        # Apply memory optimizations
        self._apply_memory_optimizations(model)
        
        # Setup for fine-tuning if needed
        if self.mode_config['type'] == 'finetune':
            self._setup_finetuning(model)
        
        # Wrap with DDP
        model = DDP(
            model, 
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            broadcast_buffers=False,  # Improves performance
            find_unused_parameters=False  # Set to True if needed for partial model update
        )
        
        return model
    
    def _apply_memory_optimizations(self, model: torch.nn.Module) -> None:
        """Apply memory optimizations like activation checkpointing."""
        # Enable activation checkpointing for transformer layers if available
        if hasattr(model.encoder, 'model') and hasattr(model.encoder.model, 'blocks'):
            try:
                from torch.utils.checkpoint import checkpoint_sequential
                blocks = model.encoder.model.blocks
                # Enable checkpointing on transformer blocks
                for block in blocks:
                    block.enable_checkpoint = True
                
                if self.is_main_process:
                    logging.info("Enabled activation checkpointing for transformer blocks")
            except (ImportError, AttributeError) as e:
                if self.is_main_process:
                    logging.warning(f"Could not apply activation checkpointing: {e}")
    
    def _setup_finetuning(self, model: torch.nn.Module) -> None:
        """Configure model for fine-tuning."""
        # Freeze encoder if specified
        if self.mode_config.get('freeze_encoder', False):
            for param in model.encoder.parameters():
                param.requires_grad = False
            
            if self.is_main_process:
                logging.info("Encoder layers frozen for fine-tuning")
        
        # Freeze specified layers
        for layer_name in self.mode_config.get('frozen_layers', []):
            if hasattr(model, layer_name):
                for param in getattr(model, layer_name).parameters():
                    param.requires_grad = False
                
                if self.is_main_process:
                    logging.info(f"Layer {layer_name} frozen for fine-tuning")
    
    def _setup_optimization(self):
        """Configure optimization with layerwise learning rates and warmup."""
        param_groups = self._get_param_groups()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config['optimizer'].get('weight_decay', 0.01)
        )
        
        # LR Scheduler with warmup
        self.scheduler = self._get_scheduler_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=100,  # Can be configured in config
            num_training_steps=self.num_epochs
        )
    
    def _get_param_groups(self) -> List[Dict]:
        """Create parameter groups for different learning rates."""
        # Parameter groups
        encoder_params = {'params': [], 'weight_decay': 0.0}
        encoder_norm_params = {'params': [], 'weight_decay': 0.0}
        decoder_params = {'params': [], 
                          'weight_decay': self.config['optimizer'].get('weight_decay', 0.01)}
        decoder_norm_params = {'params': [], 'weight_decay': 0.0}
        
        # Group parameters - access through module when using DDP
        base_model = self.model.module if hasattr(self.model, 'module') else self.model
        
        for name, p in base_model.named_parameters():
            if 'encoder' in name:
                if 'norm' in name or 'bn' in name:
                    encoder_norm_params['params'].append(p)
                else:
                    encoder_params['params'].append(p)
            else:
                if 'norm' in name or 'bn' in name:
                    decoder_norm_params['params'].append(p)
                else:
                    decoder_params['params'].append(p)
        
        # Set learning rates
        base_lr = self.config['optimizer']['learning_rate']
        encoder_lr_scale = self.config['optimizer'].get('encoder_lr_ratio', 0.1)
        
        encoder_params['lr'] = base_lr * encoder_lr_scale
        encoder_norm_params['lr'] = base_lr * encoder_lr_scale
        decoder_params['lr'] = base_lr
        decoder_norm_params['lr'] = base_lr
        
        return [encoder_params, encoder_norm_params, 
                decoder_params, decoder_norm_params]
    
    def _get_scheduler_with_warmup(self, optimizer, num_warmup_steps, num_training_steps):
        """Create a learning rate scheduler with linear warmup."""
        # Define a custom lambda function for the LambdaLR scheduler
        def lr_lambda(current_step):
            # Linear warmup
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            
            # After warmup, use ReduceLROnPlateau-like behavior by using a constant scale factor
            # This will be adjusted manually based on validation metrics
            return 1.0
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def _load_checkpoint(self) -> None:
        """Load and validate checkpoint for distributed training."""
        checkpoint_path = Path(self.mode_config['checkpoint_path'])
        
        if self.is_main_process:
            logging.info(f"Loading checkpoint from: {checkpoint_path}")
        
        # Load checkpoint only on local_rank 0 and broadcast to others
        if self.is_main_process:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                if 'model_state_dict' not in checkpoint:
                    raise ValueError("Invalid checkpoint: missing model state")
                
            except Exception as e:
                logging.error(f"Error loading checkpoint: {e}", exc_info=True)
                raise RuntimeError(f"Failed to load checkpoint: {checkpoint_path}")
        else:
            # Other ranks wait for the checkpoint to be broadcast
            checkpoint = None
        
        # Broadcast checkpoint from rank 0 to all other ranks
        if self.world_size > 1:
            checkpoint = self._broadcast_object(checkpoint)
            
        # Now all ranks have the checkpoint, load it
        try:
            # DDP requires loading to module
            self.model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            # Load other states (optimizer, scheduler, etc.) - only needed on resuming
            if self.mode_config['type'] == 'resume':
                self.start_epoch = checkpoint.get('epoch', -1) + 1
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.scaler.load_state_dict(checkpoint['scaler'])
                
                if self.is_main_process:
                    self.monitor.history = checkpoint.get('history', self.monitor.history)
                    logging.info(f"Resumed training from epoch {self.start_epoch}")
            else:  # finetune
                # Apply learning rate scaling
                lr_scale = self.mode_config.get('finetune_lr_scale', 0.1)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * lr_scale
                
                if self.is_main_process:
                    logging.info(f"Fine-tuning with learning rate scale: {lr_scale}")
                    # Reset history for finetune
                    self.monitor.history = {
                        'epochs': [],
                        'best_metrics': {'weighted_f': 0.0, 's_alpha': 0.0, 'mae': float('inf')}
                    }
        except Exception as e:
            if self.is_main_process:
                logging.error(f"Error applying checkpoint: {e}", exc_info=True)
            raise RuntimeError("Failed to apply checkpoint")
    
    def _broadcast_object(self, obj):
        """Broadcast a Python object from rank 0 to all other ranks."""
        if not dist.is_available() or not dist.is_initialized():
            return obj
        
        if self.local_rank == 0:
            buffer = torch.ByteStorage.from_buffer(pickle.dumps(obj))
            tensor = torch.ByteTensor(buffer)
            size = torch.LongTensor([tensor.numel()])
            dist.broadcast(size, 0)
            dist.broadcast(tensor, 0)
            return obj
        else:
            size = torch.LongTensor([0])
            dist.broadcast(size, 0)
            tensor = torch.ByteTensor(size.item())
            dist.broadcast(tensor, 0)
            buffer = tensor.cpu().numpy().tobytes()
            return pickle.loads(buffer)
    
    def _process_batch(self, batch: Dict[str, torch.Tensor], 
                  is_train: bool, step: int) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Process a single batch through complete training pipeline.
        
        Args:
            batch: Input data batch
            is_train: Whether in training mode
            step: Current step within epoch (for gradient accumulation)
            
        Returns:
            metrics: Performance measurements
            timing: Processing time breakdown
        """
        # Initialize timing and metrics
        timing = {}
        batch_start = time.time()
        
        # Data loading
        data_start = time.time()
        images = batch['images'].to(self.device, non_blocking=True)
        masks = [m.to(self.device, non_blocking=True) for m in batch['masks']]
        edges = [e.to(self.device, non_blocking=True) for e in batch['edges']]
        timing['data_time'] = time.time() - data_start
        
        # Determine if we need to accumulate gradients
        need_backward = is_train
        optimize_step = is_train and ((step + 1) % self.grad_accum_steps == 0)
        
        # Clear gradients only at the beginning of accumulation cycle
        if is_train and (step % self.grad_accum_steps == 0):
            self.optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        forward_start = time.time()
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            with torch.set_grad_enabled(need_backward):
                outputs = self.model(images)
        timing['forward_time'] = time.time() - forward_start
        
        # Prediction resizing
        resize_start = time.time()
        resized_dict = {}
        for key, value in outputs.items():
            resized_preds = [F.interpolate(
                value[i:i+1], size=masks[i].shape[-2:], mode='bilinear', align_corners=False
                ).squeeze(0) for i, _ in enumerate(value)]
            resized_dict[key] = resized_preds
        timing['resize_time'] = time.time() - resize_start
        
        # Loss computation
        loss_start = time.time()
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            loss_dict = self.criterion(
                predictions=resized_dict,
                gt_masks=masks,
                gt_edges=edges
            )
        
        # Scale loss for gradient accumulation
        if need_backward:
            loss_dict['loss'] = loss_dict['loss'] / self.grad_accum_steps
            loss_dict['loss_edge'] = loss_dict['loss_edge'] / self.grad_accum_steps
            loss_dict['loss_loc'] = loss_dict['loss_loc'] / self.grad_accum_steps
            loss_dict['loss_final'] = loss_dict['loss_final'] / self.grad_accum_steps
        
        timing['loss_time'] = time.time() - loss_start
        
        # Optimization (backward and step)
        if need_backward:
            backward_start = time.time()
            self.scaler.scale(loss_dict['loss']).backward()
            
            if optimize_step:
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            timing['backward_time'] = time.time() - backward_start
            metrics = loss_dict
        else:
            # Metrics computation for validation
            metric_start = time.time()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    metrics = self.metrics_processor.compute_metrics(
                        seg_pred=resized_dict['final_mask'],
                        seg_gt=masks,
                        edge_pred=resized_dict['edge_mask'],
                        edge_gt=edges
                    )
            metrics.update(loss_dict)
            timing['metric_time'] = time.time() - metric_start
        
        timing['batch_time'] = time.time() - batch_start
        
        return metrics, timing
    
    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Execute single training epoch with distributed coordination."""
        self.model.train()
        
        # Reset metric trackers
        for meter in self.train_meters.values():
            meter.reset()
        
        # Only use tqdm on rank 0
        if self.is_main_process:
            pbar = tqdm(total=len(loader), desc=f'Train Epoch {epoch+1}/{self.num_epochs}')
        
        # Track epoch start time
        epoch_start = time.time()
        
        for step, batch in enumerate(loader):
            # Process batch
            metrics, timing = self._process_batch(batch, is_train=True, step=step)
            
            # Update metrics (locally)
            batch_size = len(batch['images'])
            self.train_meters['loss'].update(metrics['loss'].item() * self.grad_accum_steps, batch_size)
            self.train_meters['loss_edge'].update(metrics['loss_edge'].item() * self.grad_accum_steps, batch_size)
            self.train_meters['loss_loc'].update(metrics['loss_loc'].item() * self.grad_accum_steps, batch_size)
            self.train_meters['loss_final'].update(metrics['loss_final'].item() * self.grad_accum_steps, batch_size)
            self.train_meters['data_time'].update(timing['data_time'], batch_size)
            self.train_meters['batch_time'].update(timing['batch_time'], batch_size)
            
            # Update progress bar on main process
            if self.is_main_process and ((step + 1) % 10 == 0 or step == len(loader) - 1):
                pbar.update(10 if (step + 1) % 10 == 0 else len(loader) % 10)
                pbar.set_postfix({
                    'loss': f"{self.train_meters['loss'].avg:.4f}",
                    'edge_loss': f"{self.train_meters['loss_edge'].avg:.4f}",
                    'loc_loss': f"{self.train_meters['loss_loc'].avg:.4f}",
                    'data_time': f"{self.train_meters['data_time'].avg:.3f}s",
                    'batch_time': f"{self.train_meters['batch_time'].avg:.3f}s",
                })
        
        # Synchronize metrics across processes
        for meter in self.train_meters.values():
            meter.synchronize_between_processes()
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        
        # Only gather and log metrics on main process
        if self.is_main_process:
            pbar.close()
            
            # Create the results dict
            results = {
                'loss': self.train_meters['loss'].avg,
                'loss_edge': self.train_meters['loss_edge'].avg,
                'loss_loc': self.train_meters['loss_loc'].avg,
                'loss_final': self.train_meters['loss_final'].avg,
                'epoch_time': epoch_time,
                'data_time': self.train_meters['data_time'].avg,
                'batch_time': self.train_meters['batch_time'].avg,
            }
            
            logging.info(
                f"Train Epoch {epoch+1} - "
                f"Loss: {results['loss']:.4f}, "
                f"Edge Loss: {results['loss_edge']:.4f}, "
                f"Loc Loss: {results['loss_loc']:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Update monitor
            self.monitor.save_epoch(epoch, 'train', results)
            
            return results
        else:
            # Non-main processes return None
            return None
    
    @torch.no_grad()
    def validate(self, loader: DataLoader, epoch: int) -> Optional[Dict[str, float]]:
        """Perform validation with synchronized metrics across processes."""
        self.model.eval()
        
        # Reset metric trackers
        for meter in self.val_meters.values():
            meter.reset()
        
        # Only use tqdm on rank 0
        if self.is_main_process:
            pbar = tqdm(total=len(loader), desc=f'Validate Epoch {epoch+1}')
        
        # Track epoch start time
        epoch_start = time.time()
        
        for step, batch in enumerate(loader):
            # Process batch (no gradients, no accumulation)
            metrics, timing = self._process_batch(batch, is_train=False, step=0)
            
            # Update metrics (locally)
            batch_size = len(batch['images'])
            self.val_meters['loss'].update(metrics['loss'].item(), batch_size)
            self.val_meters['weighted_f'].update(metrics['weighted_f'], batch_size)
            self.val_meters['s_alpha'].update(metrics['s_alpha'], batch_size)
            self.val_meters['mae'].update(metrics['mae'], batch_size)
            self.val_meters['data_time'].update(timing['data_time'], batch_size)
            self.val_meters['batch_time'].update(timing['batch_time'], batch_size)
            
            # Update progress bar on main process
            if self.is_main_process and ((step + 1) % 10 == 0 or step == len(loader) - 1):
                pbar.update(10 if (step + 1) % 10 == 0 else len(loader) % 10)
                pbar.set_postfix({
                    'f_measure': f"{self.val_meters['weighted_f'].avg:.4f}",
                    's_alpha': f"{self.val_meters['s_alpha'].avg:.4f}",
                    'mae': f"{self.val_meters['mae'].avg:.4f}",
                    'loss': f"{self.val_meters['loss'].avg:.4f}",
                })
        
        # Synchronize metrics across processes
        for meter in self.val_meters.values():
            meter.synchronize_between_processes()
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        
        # Only gather and log metrics on main process
        if self.is_main_process:
            pbar.close()
            
            # Create the results dict
            results = {
                'loss': self.val_meters['loss'].avg,
                'weighted_f': self.val_meters['weighted_f'].avg,
                's_alpha': self.val_meters['s_alpha'].avg,
                'mae': self.val_meters['mae'].avg,
                'epoch_time': epoch_time,
                'data_time': self.val_meters['data_time'].avg,
                'batch_time': self.val_meters['batch_time'].avg,
            }
            
            logging.info(
                f"Val Epoch {epoch+1} - "
                f"F-measure: {results['weighted_f']:.4f}, "
                f"S-alpha: {results['s_alpha']:.4f}, "
                f"MAE: {results['mae']:.4f}, "
                f"Loss: {results['loss']:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Update monitor
            self.monitor.save_epoch(epoch, 'val', results)
            
            return results
        else:
            # Non-main processes return None
            return None
    
    def train(self, dataset_dirs: List[str]):
        """
        Main distributed training loop with proper coordination.
        """
        try:
            # Get effective batch size for logging
            effective_batch_size = self.batch_size * self.grad_accum_steps * self.world_size
            
            if self.is_main_process:
                logging.info(f"Effective batch size: {effective_batch_size} "
                           f"(per-GPU: {self.batch_size}, accum: {self.grad_accum_steps}, GPUs: {self.world_size})")
            
            # Setup data loaders with distributed samplers
            train_loader, val_loader = get_distributed_training_loaders(
                dataset_dirs=dataset_dirs,
                model_config=self.model_config,
                batch_size=self.batch_size,
                num_workers=self.config['num_workers'],
                val_ratio=self.config['val_ratio'],
                grad_accum_steps=self.grad_accum_steps,
                world_size=self.world_size,
                rank=self.local_rank,
                seed=42  # Fix seed for reproducibility
            )
            
            # Log dataset sizes on main process
            if self.is_main_process:
                logging.info(f"Training samples: {len(train_loader.dataset)} - "
                           f"Batches: {len(train_loader)}")
                if val_loader:
                    logging.info(f"Validation samples: {len(val_loader.dataset)} - "
                               f"Batches: {len(val_loader)}")
            
            # Initialize tracking variables (only on main process)
            if self.is_main_process:
                best_weighted_f = self.monitor.history['best_metrics']['weighted_f']
                early_stop_counter = 0
                min_delta = self.config.get('min_delta', 1e-4)
            
            # Create completion barrier to sync processes
            dist.barrier()
            
            # Training loop
            for epoch in range(self.start_epoch, self.num_epochs):
                # Set epochs in samplers for proper shuffling
                train_loader.sampler.set_epoch(epoch)
                if val_loader:
                    val_loader.sampler.set_epoch(epoch)
                
                # Training phase
                train_metrics = self.train_epoch(train_loader, epoch)
                
                # Step LR scheduler based on epoch (not validation metric)
                self.scheduler.step()
                
                # Validation phase
                if val_loader:
                    val_metrics = self.validate(val_loader, epoch)
                    
                    # Early stopping check (only on main process)
                    if self.is_main_process and val_metrics:
                        if val_metrics['weighted_f'] - best_weighted_f > min_delta:
                            best_weighted_f = val_metrics['weighted_f']
                            early_stop_counter = 0
                            # Save model if improved
                            self._save_checkpoint(epoch, val_metrics, is_best=True)
                        else:
                            early_stop_counter += 1
                        
                        if early_stop_counter >= self.early_stop_patience:
                            logging.info("Early stopping triggered")
                            # Signal all processes to stop
                            break
                
                # Regular checkpointing (only on main process)
                if self.is_main_process and (epoch + 1) % self.save_freq == 0:
                    self._save_checkpoint(
                        epoch,
                        val_metrics if val_loader else train_metrics,
                        is_best=False
                    )
                
                # Synchronize processes at end of epoch
                dist.barrier()
                
                # Memory cleanup
                torch.cuda.empty_cache()
                
        except Exception as e:
            if self.is_main_process:
                logging.error(f"Training error: {str(e)}", exc_info=True)
            raise
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save training checkpoint (only on main process)."""
        if not self.is_main_process:
            return
        
        # Extract model state dict (unwrap DDP)
        model_state_dict = self.model.module.state_dict()
        
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'metrics': metrics,
            'config': {
                'training': self.config,
                'model': self.model_config
            },
            'history': self.monitor.history,
            'mode': self.mode_config,
            'world_size': self.world_size
        }
        
        filename = 'model_best.pth' if is_best else f'checkpoint_{epoch:03d}.pth'
        save_path = self.monitor.checkpoint_dir / filename
        
        # Atomic save
        temp_path = save_path.with_suffix('.tmp')
        torch.save(save_dict, temp_path)
        temp_path.rename(save_path)
        
        logging.info(f"Saved checkpoint: {save_path}")


class DistributedTrainingMonitor:
    """
    Tracks and manages training metrics and model checkpoints for distributed training.
    Only active on the main process (rank 0).
    """
    
    def __init__(self, dir_manager: DirectoryManager):
        """Initialize training monitor with directory structure."""
        # Output paths
        self.metrics_file = dir_manager.run_dirs.metrics_file
        self.checkpoint_dir = dir_manager.run_dirs.checkpoints
        
        # Initialize or load history
        self.history = {
            "epochs": [],
            "best_metrics": {
                "weighted_f": 0.0,
                "s_alpha": 0.0,
                "mae": float("inf")
            }
        }
        
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                self.history = json.load(f)
    
    def check_best_model(self, current_metrics: Dict[str, float]) -> bool:
        """Determines if current model achieves best performance."""
        if current_metrics['weighted_f'] > self.history['best_metrics']['weighted_f']:
            self.history['best_metrics'] = current_metrics.copy()
            self.save_history()
            logging.info(
                f"New best model -> F-Measure: {current_metrics['weighted_f']:.4f}"
            )
            return True
        return False
    
    def save_history(self):
        """Atomic save of complete history structure."""
        tmp_path = self.metrics_file.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        tmp_path.rename(self.metrics_file)
    
    def save_epoch(self, epoch: int, phase: str, metrics: Dict[str, float]):
        """
        Save epoch results to metrics file.
        
        Args:
            epoch: Current epoch number
            phase: Training phase ('train' or 'val')
            metrics: Epoch metrics
        """
        # Split into metrics and timing
        timing = {k: v for k, v in metrics.items() if k.endswith('_time')}
        metrics = {k: v for k, v in metrics.items() if not k.endswith('_time')}
        
        # Ensure epoch entry exists
        while len(self.history['epochs']) <= epoch:
            self.history['epochs'].append({"epoch": len(self.history['epochs'])})
        
        # Update epoch data
        self.history['epochs'][epoch][phase] = {
            "metrics": metrics,
            "timing": timing
        }
        
        # Save complete history structure
        self.save_history()