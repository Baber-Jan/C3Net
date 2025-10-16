"""
Training engine for Camouflaged Object Detection models.

This module provides a comprehensive training implementation with:
1. Automatic mixed precision training
2. Multi-scale segmentation and edge supervision
3. Resource-optimized batch processing
4. Metric tracking and model checkpointing

Core Components:
- TrainingMonitor: Handles metric tracking and checkpoint management
- Trainer: Manages training loops and optimization
- Batch Processing: Efficient multi-scale prediction and loss computation

Processing Pipeline:
1. Data Loading: Non-blocking tensor transfers
2. Forward Pass: Mixed precision inference
3. Multi-scale Loss: Segmentation and edge predictions
4. Optimization: Gradient scaling and clipping
5. Metric Tracking: Comprehensive performance monitoring
"""

import logging
import json
import time
from typing import Dict, List, Tuple
from collections import defaultdict
from pathlib import Path

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict

from utils.metrics import MetricsProcessor
from utils.loss_functions import C3NetLoss
from utils.run_manager import DirectoryManager
from models.c3net import C3Net
from utils.data_loader import get_training_loaders

class TrainingMonitor:
    """
    Tracks and manages training metrics and model checkpoints.
    
    This class handles:
    1. Per-batch statistic aggregation
    2. Best model selection via F-measure
    3. Metric history management
    4. Atomic checkpoint saving
    
    Attributes:
        metrics_file (Path): JSON file for metric storage
        checkpoint_dir (Path): Directory for model checkpoints
        batch_stats (defaultdict): Running statistics tracker
        epoch_start (float): Epoch timing reference
        history (dict): Complete training history
            epochs: List of epoch statistics
            best_metrics: Best performance records
    """
    
    def __init__(self, dir_manager: DirectoryManager):
        """Initialize training monitor with directory structure."""
        # Output paths
        self.metrics_file = dir_manager.run_dirs.metrics_file
        self.checkpoint_dir = dir_manager.run_dirs.checkpoints
        
        # Statistics tracking
        self.batch_stats = defaultdict(lambda: {'sum': 0.0, 'count': 0})
        
        # Epoch tracking
        self.epoch_start = None
        
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

    def start_epoch(self):
        """Reset statistics and start timing for new epoch."""
        self.batch_stats.clear()
        self.epoch_start = time.time()

    def update_batch(self, metrics: Dict[str, float], timing: Dict[str, float], batch_size: int):
        """
        Updates running statistics with batch results.
        
        Accumulates weighted averages of metrics and timing statistics,
        accounting for variable batch sizes.
        
        Args:
            metrics: Current batch metric values
            timing: Timing measurements (data_time, forward_time, etc.)
            batch_size: Number of samples in batch
            
        Updates:
            - Running sums and counts for each statistic
            - Enables proper weighted averaging
        """
        # Update running statistics
        for key, value in {**metrics, **timing}.items():
            value = float(value) if torch.is_tensor(value) else value
            self.batch_stats[key]['sum'] += value * batch_size
            self.batch_stats[key]['count'] += batch_size

    def get_current_stats(self) -> Dict[str, float]:
        """Get current averaged statistics."""
        return {
            key: stats['sum'] / stats['count']
            for key, stats in self.batch_stats.items()
            if stats['count'] > 0
        }

    def check_best_model(self, current_metrics: Dict[str, float]) -> bool:
        """
        Determines if current model achieves best performance.
        
        Compares current F-measure against historical best and
        updates records if improved.
        
        Args:
            current_metrics: Current epoch evaluation metrics
        
        Returns:
            bool: True if new best model achieved
            
        Updates:
            - Best metrics history if improved
            - Saves updated history to disk
        """
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

    def save_epoch(self, epoch: int, phase: str):
        """
        Save epoch results to metrics file.
        
        Args:
            epoch: Current epoch number
            phase: Training phase ('train' or 'val')
        """
        stats = self.get_current_stats()
        epoch_time = time.time() - self.epoch_start

        # Split into metrics and timing
        metrics = {k: v for k, v in stats.items() if not k.endswith('_time')}
        timing = {k: v for k, v in stats.items() if k.endswith('_time')}
        timing['epoch_time'] = epoch_time

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
        if phase == 'val':
            # Log epoch summary
            logging.info(
                f"Epoch {epoch} ({phase}) - "
                f"F-measure: {stats['weighted_f']:.4f}, "
                f"S-alpha: {stats['s_alpha']:.4f}, "
                f"MAE: {stats['mae']:.4f}, "
                f"Loss: {stats['loss']:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
        else:
            logging.info(
                f"Epoch {epoch} ({phase}) - "
                f"Loss: {stats['loss']:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )

class Trainer:
    """
    Manages complete training pipeline for Camouflaged Object Detection.
    
    Implements:
    1. Multi-scale training with automatic mixed precision
    2. Joint segmentation and edge supervision
    3. Layer-wise learning rate optimization
    4. Comprehensive validation and checkpointing
    
    Processing Flow:
    1. Batch Loading: Images, masks, and edge maps
    2. Forward Pass: Generate multi-scale predictions
    3. Loss Computation: Combined segmentation and edge losses
    4. Optimization: Scaled gradients with clipping
    5. Validation: Multi-metric performance tracking
    
    Args:
        config (Dict): Model and training configuration
        dir_manager (DirectoryManager): Output directory handler
        device (torch.device): Computing device
    """
    
    def __init__(
        self,
        config: Dict,
        dir_manager: DirectoryManager,
        device: torch.device,
    ):
        # Basic setup
        self.config = config['training']
        self.model_config = config['model']
        self.device = device
        self.mode_config = self.config.get('mode', {'type': 'train'})
        
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
        self.criterion = C3NetLoss(**self.config['loss']).to(device)
        self.monitor = TrainingMonitor(dir_manager)
        self.metrics_processor = MetricsProcessor()
        
        # Mixed precision setup
        self.use_amp = self.config.get('use_amp', True)
        self.scaler = amp.GradScaler(enabled=self.use_amp)


        # Set starting epoch
        self.start_epoch = 0
        
        # Load checkpoint if resuming or fine-tuning
        if self.mode_config.get('checkpoint_path'):
            self._load_checkpoint()

    def _initialize_model(self) -> torch.nn.Module:
        """Initialize model and wrap with DataParallel if multiple GPUs are available."""
        model = C3Net(self.model_config) # Create the base model first

        # Check for multiple GPUs and wrap with DataParallel
        if torch.cuda.device_count() > 1:
            gpu_count = torch.cuda.device_count()
            logging.info(f"Using {gpu_count} GPUs with torch.nn.DataParallel.")
            model = torch.nn.DataParallel(model) # Wrap the model
        else:
            logging.info("Using single GPU or CPU.")

        model = model.to(self.device) # Move wrapped or unwrapped model to target device

        if self.mode_config['type'] == 'finetune':
             # Access the underlying model if wrapped for freezing
            base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            self._setup_finetuning(base_model) # Pass the base model to freeze layers

        return model

    def _setup_finetuning(self, model: torch.nn.Module) -> None:
        """
        Configure model for fine-tuning.
        
        Args:
            model: Model to be fine-tuned
        """
        # Freeze encoder if specified
        if self.mode_config.get('freeze_encoder', False):
            for param in model.encoder.parameters():
                param.requires_grad = False
            logging.info("Encoder layers frozen for fine-tuning")
        
        # Freeze specified layers
        for layer_name in self.mode_config.get('frozen_layers', []):
            if hasattr(model, layer_name):
                for param in getattr(model, layer_name).parameters():
                    param.requires_grad = False
                logging.info(f"Layer {layer_name} frozen for fine-tuning")

    def _load_checkpoint(self) -> None:
        """
        Load and validate checkpoint, handling DataParallel state dict keys.
        """
        checkpoint_path = Path(self.mode_config['checkpoint_path'])
        logging.info(f"Loading checkpoint from: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            if 'model_state_dict' not in checkpoint:
                raise ValueError("Invalid checkpoint: missing model state")

            saved_state_dict = checkpoint['model_state_dict']

            # --- Handle potential 'module.' prefix from DataParallel saved models ---
            # Create a new state dict without the 'module.' prefix if it exists
            new_state_dict = OrderedDict()
            is_saved_with_dp = False
            for k, v in saved_state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v # remove `module.` prefix
                    is_saved_with_dp = True
                else:
                    new_state_dict[k] = v
            if is_saved_with_dp:
                 logging.info("Checkpoint was saved from a DataParallel model. Adjusting keys.")
                 saved_state_dict = new_state_dict # Use the corrected state dict
            # -------------------------------------------------------------------

            # Load into the base model (access via .module if DP is currently active)
            base_model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
            missing_keys, unexpected_keys = base_model.load_state_dict(
                saved_state_dict, strict=False
            )

            if missing_keys:
                logging.warning(f"Missing keys when loading state_dict: {missing_keys}")
            if unexpected_keys:
                logging.warning(f"Unexpected keys when loading state_dict: {unexpected_keys}")

            # --- Load other states (optimizer, scheduler, etc.) ---
            if self.mode_config['type'] == 'resume':
                self.start_epoch = checkpoint.get('epoch', -1) + 1
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                else:
                    logging.warning("No optimizer state found, using fresh optimizer.")
                if 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                else:
                    logging.warning("No scheduler state found, using fresh scheduler.")
                if 'scaler' in checkpoint:
                    self.scaler.load_state_dict(checkpoint['scaler'])
                else:
                    logging.warning("No AMP scaler state found, using fresh scaler.")
                if 'history' in checkpoint:
                    self.monitor.history = checkpoint['history']
                # ... (rest of resume logic) ...
                logging.info(f"Resumed training from epoch {self.start_epoch}")
            else: # finetune
                lr_scale = self.mode_config.get('finetune_lr_scale', 0.1)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * lr_scale
                logging.info(f"Fine-tuning with learning rate scale: {lr_scale}")
                self.monitor.history = { # Reset history for finetune
                    'epochs': [],
                    'best_metrics': {'weighted_f': 0.0, 's_alpha': 0.0, 'mae': float('inf')}
                }

            # Validate configuration compatibility
            if 'config' in checkpoint:
                self._validate_config_compatibility(checkpoint['config'])

        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}", exc_info=True) # Added exc_info
            raise RuntimeError(f"Failed to load checkpoint: {checkpoint_path}")
        

    def _validate_config_compatibility(self, checkpoint_config: Dict) -> None:
        """
        Validate and log configuration differences between checkpoint and current setup.
        
        Args:
            checkpoint_config: Configuration from checkpoint
            
        Notes:
            Logs warnings for significant differences but doesn't block execution
        """
        try:
            # Check model configuration
            if 'model' in checkpoint_config:
                current_model = self.model_config
                saved_model = checkpoint_config['model']
                
                # Check critical parameters
                if saved_model.get('name') != current_model.get('name'):
                    logging.warning(f"Model architecture mismatch: saved={saved_model.get('name')}, current={current_model.get('name')}")
                    
                if saved_model.get('encoder', {}).get('variant') != current_model.get('encoder', {}).get('variant'):
                    logging.warning("Encoder variant mismatch between checkpoint and current configuration")
            
            # Check training configuration
            if 'training' in checkpoint_config:
                # Log significant training parameter changes
                current_train = self.config
                saved_train = checkpoint_config['training']
                
                # Check batch size compatibility
                if saved_train.get('batch_size') != current_train.get('batch_size'):
                    logging.info(f"Batch size changed: saved={saved_train.get('batch_size')}, current={current_train.get('batch_size')}")
                
                # Check learning rate differences
                if saved_train.get('optimizer', {}).get('learning_rate') != current_train.get('optimizer', {}).get('learning_rate'):
                    logging.info("Learning rate configuration has changed from checkpoint")
                    
        except Exception as e:
            logging.warning(f"Configuration validation error: {e}")
            logging.warning("Continuing with current configuration")


    def _setup_optimization(self):
        """Configure optimization with layerwise learning rates."""
        param_groups = self._get_param_groups()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config['optimizer'].get('weight_decay', 0.01)
        )
        
        # LR Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=self.config['scheduler'].get('factor', 0.5),
            patience=self.config['scheduler'].get('patience', 5),
            min_lr=self.config['scheduler'].get('min_lr', 1e-6)
        )

    def _get_param_groups(self) -> List[Dict]:
        """Create parameter groups for different learning rates."""
        # Parameter groups
        encoder_params = {'params': [], 'weight_decay': 0.0}
        encoder_norm_params = {'params': [], 'weight_decay': 0.0}
        decoder_params = {'params': [], 
                         'weight_decay': self.config['optimizer'].get('weight_decay', 0.01)}
        decoder_norm_params = {'params': [], 'weight_decay': 0.0}
        
        # Group parameters
        for name, p in self.model.named_parameters():
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
    
    def _process_batch(self, batch: Dict[str, torch.Tensor], 
                  is_train: bool) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Processes single batch through complete training pipeline.
        
        Steps:
        1. Data Transfer: Non-blocking device placement
        2. Forward Pass: Mixed precision inference
        3. Prediction Resizing: Multi-scale alignment
        4. Loss Computation: Segmentation and edge losses
        5. Optimization: Gradient scaling and updates (training only)
        6. Metric Computation: Performance measurements
        
        Args:
            batch: Input data containing:
                - images: Input images
                - masks: Ground truth segmentations
                - edges: Ground truth edge maps
            is_train: Whether in training mode
            
        Returns:
            metrics: Performance measurements
            timing: Processing time breakdown
        """

        timing = {}
        batch_start = time.time()
        
        # Data loading
        data_start = time.time()
        images = batch['images'].to(self.device, non_blocking=True)
        masks = [m.to(self.device, non_blocking=True) for m in batch['masks']]
        edges = [e.to(self.device, non_blocking=True) for e in batch['edges']]
        timing['data_time'] = time.time() - data_start
        
        # Forward pass
        forward_start = time.time()
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            with torch.set_grad_enabled(is_train):
                outputs = self.model(images)
        timing['forward_time'] = time.time() - forward_start

        # Prediction resizing
        resize_start = time.time()
        # Resize predictions to match target sizes
        resized_dict = {}
        for key, value in outputs.items(): 
            """
                {
                    'final_mask': final_mask,
                    'edge_mask': edge_mask,
                    'object_mask': object_mask
                }
            """
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
        timing['loss_time'] = time.time() - loss_start
        # Optimization
        if is_train:
            backward_start = time.time()
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss_dict['loss']).backward()
            
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            timing['backward_time'] = time.time() - backward_start
            metrics = loss_dict
        else:
            # Metrics computation
            metric_start = time.time()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    metrics = self.metrics_processor.compute_metrics(
                        seg_pred= resized_dict['final_mask'],
                        seg_gt=masks,
                        edge_pred= resized_dict['edge_mask'],
                        edge_gt=edges
                    )
            metrics.update(loss_dict)
            timing['metric_time'] = time.time() - metric_start
            
        timing['batch_time'] = time.time() - batch_start
        
        return metrics, timing


    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Executes single training epoch.
        
        Processing:
        1. Sets model to training mode
        2. Iterates through batches with progress tracking
        3. Processes each batch with optimization
        4. Updates monitoring statistics
        
        Args:
            loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            dict: Averaged epoch metrics
        """

        self.model.train()
        self.monitor.start_epoch()
        
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{self.num_epochs}')
        for batch_idx, batch in enumerate(pbar):
            # Process batch
            metrics, timing = self._process_batch(batch, is_train=True)
            
            # Update monitoring
            self.monitor.update_batch(
                metrics=metrics,
                timing=timing,
                batch_size=len(batch['images'])
            )
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'object_loss': f"{metrics['loss_loc']:.4f}",
                'edge_loss': f"{metrics['loss_edge']:.4f}",
                'final_loss': f"{metrics['loss_final']:.4f}",
                'time': f"{timing['batch_time']:.3f}s"
            })
        
        return self.monitor.get_current_stats()

    @torch.no_grad()
    def validate(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Performs validation pass.
        
        Processing:
        1. Sets model to evaluation mode
        2. Processes batches without gradients
        3. Computes comprehensive metrics
        4. Tracks performance statistics
        
        Args:
            loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            dict: Complete validation metrics
        """
        self.model.eval()
        self.monitor.start_epoch()
        
        pbar = tqdm(loader, desc='Validation')
        for batch_idx, batch in enumerate(pbar):
            metrics, timing = self._process_batch(batch, is_train=False)
            
            self.monitor.update_batch(
                metrics=metrics,
                timing=timing,
                batch_size=len(batch['images'])
            )
            
            pbar.set_postfix({
                'f_measure': f"{metrics['weighted_f']:.4f}",
                's_alpha': f"{metrics['s_alpha']:.4f}",
                'mae': f"{metrics['mae']:.4f}",
                'loss': f"{metrics['loss']:.4f}",
                'time': f"{timing['batch_time']:.3f}s"
            })
            
        return self.monitor.get_current_stats()
        
    def train(self, dataset_dirs: List[str]):
        """Main training loop."""
        try:
            # Setup data loaders
            train_loader, val_loader = get_training_loaders(
                dataset_dirs=dataset_dirs,
                model_config=self.model_config,
                batch_size=self.batch_size,
                num_workers=self.config['num_workers'],
                val_ratio=self.config['val_ratio']
            )
            
            # Log setup
            logging.info(f"Training samples: {len(train_loader.dataset)}")
            if val_loader:
                logging.info(f"Validation samples: {len(val_loader.dataset)}")

            # Initialize tracking variables
            best_weighted_f = self.monitor.history['best_metrics']['weighted_f']
            early_stop_counter = 0
            min_delta = self.config.get('min_delta', 1e-4)

            # Training loop
            for epoch in range(self.start_epoch, self.num_epochs):
                # Training phase
                train_metrics = self.train_epoch(train_loader, epoch)
                self.monitor.save_epoch(epoch, 'train')
                
                # Validation phase
                if val_loader:
                    val_metrics = self.validate(val_loader, epoch)
                    self.monitor.save_epoch(epoch, 'val')
                    
                    # Learning rate adjustment
                    self.scheduler.step(val_metrics['weighted_f'])
                    
                    # Early stopping check using validation loss
                    if val_metrics['weighted_f'] - best_weighted_f > min_delta:
                        best_weighted_f = val_metrics['weighted_f']
                        early_stop_counter = 0
                        # Save model if structure measure also improved
                        if self.monitor.check_best_model(val_metrics):
                            self._save_checkpoint(epoch, val_metrics, is_best=True)
                    
                    else:
                        early_stop_counter += 1
                        
                    if early_stop_counter >= self.early_stop_patience:
                        logging.info("Early stopping triggered")
                        break
                
                # Regular checkpointing
                if (epoch + 1) % self.save_freq == 0:
                    self._save_checkpoint(
                        epoch,
                        val_metrics if val_loader else train_metrics,
                        is_best=False
                    )
                
                # Memory cleanup
                torch.cuda.empty_cache()
                
        except Exception as e:
            logging.error(f"Training error: {str(e)}", exc_info=True)
            raise
            
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save training checkpoint, handling DataParallel."""
        # If using DataParallel, save the underlying model's state_dict
        if isinstance(self.model, torch.nn.DataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        save_dict = {
            'epoch': epoch,
            'model_state_dict': model_state_dict, # Use potentially unwrapped state_dict
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'metrics': metrics,
            'config': {
                'training': self.config,
                'model': self.model_config
            },
            'history': self.monitor.history,
            'mode': self.mode_config
        }

        filename = 'model_best.pth' if is_best else f'checkpoint_{epoch:03d}.pth'
        save_path = self.monitor.checkpoint_dir / filename

        # Atomic save
        temp_path = save_path.with_suffix('.tmp')
        torch.save(save_dict, temp_path)
        temp_path.rename(save_path)

        logging.info(f"Saved checkpoint: {save_path}")