"""
Custom nnUNet Trainer for Partial LDH Labels

问题背景：
- Dataset102中，普通样本没有LDH（12号）标签，但可能实际存在LDH却没有标注
- LDH样本只有12号LDH标签是真实的，其他标签是step1推理的伪标签

解决方案：
- 对于没有LDH标签的样本：正常计算脊柱结构的loss（0-11），忽略LDH通道（12）的loss
- 对于有LDH标签的样本：正常计算所有类别的loss
"""

import torch
import numpy as np
from torch import nn
from typing import Callable, Tuple

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.helpers import softmax_helper_dim1
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDAOrd0 import nnUNetTrainer_DASegOrd0_NoMirroring


class PartialLDH_SoftDiceLoss(nn.Module):
    """
    Dice Loss that can ignore specific class channels per sample based on whether 
    that class is present in the ground truth.
    
    For samples WITHOUT LDH label: ignore LDH channel in dice calculation
    For samples WITH LDH label: compute dice for all channels
    """
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, 
                 do_bg: bool = True, smooth: float = 1., ddp: bool = True,
                 ldh_class_idx: int = 12):
        super().__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.ldh_class_idx = ldh_class_idx  # LDH is class 12 in Dataset102

    def forward(self, x, y, loss_mask=None):
        """
        x: network output (B, C, D, H, W) - logits
        y: ground truth (B, 1, D, H, W) - label map
        """
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        
        batch_size = x.shape[0]
        num_classes = x.shape[1]
        axes = tuple(range(2, x.ndim))  # spatial dimensions
        
        # Determine which samples have LDH labels
        with torch.no_grad():
            if x.ndim != y.ndim:
                y_view = y.view((y.shape[0], 1, *y.shape[1:]))
            else:
                y_view = y
            
            # Check if each sample contains LDH label
            # has_ldh: (B,) boolean tensor
            has_ldh = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
            for b in range(batch_size):
                has_ldh[b] = (y_view[b] == self.ldh_class_idx).any()
            
            # Create one-hot encoding
            if x.shape == y_view.shape:
                y_onehot = y_view
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y_view.long(), 1)
            
            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
            
            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)
        
        # Compute predictions (outside no_grad)
        if not self.do_bg:
            x = x[:, 1:]
        
        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes)  # (B, C-1) or (B, C)
            sum_pred = x.sum(axes)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)
        
        # Compute Dice per class per sample: (B, C-1) or (B, C)
        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))
        
        # Create mask for LDH channel based on whether sample has LDH
        # For samples without LDH, mask out the LDH channel contribution
        # LDH class index adjustment: if do_bg=False, class indices shift by 1
        ldh_idx_adjusted = self.ldh_class_idx - 1 if not self.do_bg else self.ldh_class_idx
        
        if ldh_idx_adjusted < dc.shape[1]:  # Make sure LDH channel exists
            # Create class weight mask: (B, num_classes)
            class_weights = torch.ones_like(dc)
            # For samples WITHOUT LDH, set LDH channel weight to 0
            class_weights[~has_ldh, ldh_idx_adjusted] = 0.0
            
            # Apply weights and compute mean
            # Count valid classes per sample
            valid_class_count = class_weights.sum(dim=1, keepdim=True)  # (B, 1)
            
            # Weighted dice per sample
            weighted_dc = (dc * class_weights).sum(dim=1) / valid_class_count.squeeze(1)  # (B,)
            dc_mean = weighted_dc.mean()
        else:
            dc_mean = dc.mean()
        
        return -dc_mean


class PartialLDH_CELoss(nn.Module):
    """
    Cross Entropy Loss that ignores LDH class predictions for samples without LDH labels.
    
    For samples WITHOUT LDH label: replace GT with ignore_index where network might predict LDH
                                    OR zero out the LDH logits to prevent learning
    For samples WITH LDH label: compute CE normally
    """
    def __init__(self, ldh_class_idx: int = 12, ignore_index: int = -100):
        super().__init__()
        self.ldh_class_idx = ldh_class_idx
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    
    def forward(self, x, y):
        """
        x: network output (B, C, D, H, W) - logits
        y: ground truth (B, D, H, W) - label map (long tensor)
        """
        batch_size = x.shape[0]
        
        # Determine which samples have LDH labels
        with torch.no_grad():
            has_ldh = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
            for b in range(batch_size):
                has_ldh[b] = (y[b] == self.ldh_class_idx).any()
        
        # For samples WITHOUT LDH: we set the LDH logits to very negative values
        # This effectively prevents the network from being penalized for predicting LDH
        # on unlabeled samples
        x_modified = x.clone()
        
        for b in range(batch_size):
            if not has_ldh[b]:
                # Set LDH channel logits to very negative value for this sample
                # This makes softmax probability for LDH class ~0, so:
                # - If GT is not LDH, predicting LDH gets no penalty (since prob is ~0)
                # - Network won't be encouraged to predict LDH on these samples
                x_modified[b, self.ldh_class_idx] = -1e6
        
        # Compute CE loss
        loss_per_voxel = self.ce(x_modified, y.long())  # (B, D, H, W)
        
        # Mean over all voxels
        return loss_per_voxel.mean()


class PartialLDH_DC_and_CE_loss(nn.Module):
    """
    Combined Dice + Cross Entropy Loss with partial LDH label handling.
    """
    def __init__(self, soft_dice_kwargs: dict, ce_kwargs: dict, 
                 weight_ce: float = 1.0, weight_dice: float = 1.0,
                 ignore_label=None, dice_class=None,
                 ldh_class_idx: int = 12):
        super().__init__()
        
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label
        self.ldh_class_idx = ldh_class_idx
        
        # Custom Dice Loss
        self.dc = PartialLDH_SoftDiceLoss(
            apply_nonlin=softmax_helper_dim1,
            ldh_class_idx=ldh_class_idx,
            **soft_dice_kwargs
        )
        
        # Custom CE Loss
        ignore_index = ignore_label if ignore_label is not None else -100
        self.ce = PartialLDH_CELoss(
            ldh_class_idx=ldh_class_idx,
            ignore_index=ignore_index
        )
    
    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        net_output: (B, C, D, H, W) - logits
        target: (B, 1, D, H, W) - label map
        """
        # Handle ignore label if present
        if self.ignore_label is not None:
            assert target.shape[1] == 1
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
        
        # Dice Loss
        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) if self.weight_dice != 0 else 0
        
        # CE Loss
        ce_loss = self.ce(net_output, target[:, 0]) if self.weight_ce != 0 else 0
        
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class nnUNetTrainer_PartialLDH(nnUNetTrainer_DASegOrd0_NoMirroring):
    """
    Custom nnUNet Trainer for partial LDH label training.
    
    Inherits from nnUNetTrainer_DASegOrd0_NoMirroring (no mirroring for spine)
    and uses custom loss that handles missing LDH labels.
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, 
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        # LDH class index (12 in Dataset102)
        self.ldh_class_idx = 12
        
        self.print_to_log_file("")
        self.print_to_log_file("=" * 70)
        self.print_to_log_file("  Using Custom Trainer: nnUNetTrainer_PartialLDH")
        self.print_to_log_file("  - Handles missing LDH labels in training")
        self.print_to_log_file("  - LDH Class Index: {}".format(self.ldh_class_idx))
        self.print_to_log_file("=" * 70)
        self.print_to_log_file("")
    
    def _build_loss(self):
        """
        Build custom loss function that handles partial LDH labels.
        """
        # Create the partial label loss
        loss = PartialLDH_DC_and_CE_loss(
            soft_dice_kwargs={
                'batch_dice': self.configuration_manager.batch_dice,
                'smooth': 1e-5,
                'do_bg': False,
                'ddp': self.is_ddp
            },
            ce_kwargs={},
            weight_ce=1.0,
            weight_dice=1.0,
            ignore_label=self.label_manager.ignore_label,
            ldh_class_idx=self.ldh_class_idx
        )
        
        # Wrap with deep supervision if enabled
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            
            # Weights decrease exponentially with resolution
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0  # Don't use lowest resolution
            weights = weights / weights.sum()  # Normalize
            
            loss = DeepSupervisionWrapper(loss, weights)
        
        return loss


# Also create a variant that only updates on LDH samples (more aggressive approach)
class nnUNetTrainer_PartialLDH_OnlyLDHUpdate(nnUNetTrainer_DASegOrd0_NoMirroring):
    """
    More aggressive variant: Only perform gradient updates when batch contains LDH samples.
    
    For batches WITHOUT any LDH sample: skip gradient update entirely
    For batches WITH at least one LDH sample: normal training
    
    This approach ensures that the network only learns when we have reliable LDH supervision.
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, 
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        self.ldh_class_idx = 12
        self.skipped_batches = 0
        self.total_batches = 0
        
        self.print_to_log_file("")
        self.print_to_log_file("=" * 70)
        self.print_to_log_file("  Using Custom Trainer: nnUNetTrainer_PartialLDH_OnlyLDHUpdate")
        self.print_to_log_file("  - Only updates weights when batch contains LDH samples")
        self.print_to_log_file("  - LDH Class Index: {}".format(self.ldh_class_idx))
        self.print_to_log_file("=" * 70)
        self.print_to_log_file("")
    
    def train_step(self, batch: dict) -> dict:
        """
        Override train_step to conditionally update based on LDH presence.
        """
        data = batch['data']
        target = batch['target']
        
        self.total_batches += 1
        
        # Check if any sample in the batch has LDH label
        has_ldh_in_batch = False
        if isinstance(target, list):
            # Deep supervision: check first (highest res) target
            target_check = target[0]
        else:
            target_check = target
        
        with torch.no_grad():
            has_ldh_in_batch = (target_check == self.ldh_class_idx).any().item()
        
        if not has_ldh_in_batch:
            # Skip this batch - no gradient update
            self.skipped_batches += 1
            
            # Still need to return a loss dict for logging
            # Do a forward pass without gradient to get a loss value
            data = data.to(self.device, non_blocking=True)
            if isinstance(target, list):
                target = [i.to(self.device, non_blocking=True) for i in target]
            else:
                target = target.to(self.device, non_blocking=True)
            
            with torch.no_grad():
                output = self.network(data)
                l = self.loss(output, target)
            
            return {'loss': l.detach().cpu().numpy()}
        
        # Normal training step for batches with LDH
        return super().train_step(batch)
    
    def on_epoch_end(self):
        """Log statistics about skipped batches."""
        super().on_epoch_end()
        
        if self.total_batches > 0:
            skip_ratio = self.skipped_batches / self.total_batches
            self.print_to_log_file(
                f"Epoch stats: Skipped {self.skipped_batches}/{self.total_batches} "
                f"batches ({skip_ratio*100:.1f}%) without LDH labels"
            )
        
        # Reset counters
        self.skipped_batches = 0
        self.total_batches = 0

