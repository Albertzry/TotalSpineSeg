"""
Custom nnUNet Trainer for Partial LDH Labels

问题背景：
- Dataset102中，普通样本没有LDH（12号）标签，但可能实际存在LDH却没有标注
- LDH样本只有12号LDH标签是真实的，其他标签是step1推理的伪标签
- LDH是微小结构，位于椎间盘附近，需要特殊的loss设计

优化措施：
1. Partial Label Handling: 没有LDH标签的样本不计算LDH通道loss
2. LDH Class Weight (3x): 对LDH类别给予更高权重
3. Tversky Loss (alpha=0.3, beta=0.7): 提高recall，适合小目标
4. Focal Loss (gamma=2.0): 处理类别不平衡
5. Foreground Oversampling (50%): 增加LDH样本出现概率
6. Anatomical Attention: 利用椎间盘位置引导LDH检测（LDH只出现在椎间盘附近）
"""

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from typing import Callable, Optional, Tuple

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.helpers import softmax_helper_dim1
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDAOrd0 import nnUNetTrainer_DASegOrd0_NoMirroring


def create_disc_attention_mask(target: torch.Tensor, 
                                disc_classes: Tuple[int, ...] = (1, 2, 3, 4, 5),
                                dilation_radius: int = 5) -> torch.Tensor:
    """
    创建基于椎间盘位置的attention mask
    
    LDH只会出现在椎间盘附近，所以我们：
    1. 提取椎间盘区域（disc classes 1-5）
    2. 进行3D膨胀扩展到周围区域
    3. 返回这个区域作为attention mask
    
    Args:
        target: (B, 1, D, H, W) 标签图
        disc_classes: 椎间盘类别索引
        dilation_radius: 膨胀半径（体素）
    
    Returns:
        attention_mask: (B, 1, D, H, W) 椎间盘附近区域mask
    """
    batch_size = target.shape[0]
    device = target.device
    
    # 提取所有椎间盘区域
    disc_mask = torch.zeros_like(target, dtype=torch.float32)
    for dc in disc_classes:
        disc_mask = disc_mask + (target == dc).float()
    disc_mask = (disc_mask > 0).float()
    
    # 3D膨胀 - 使用max pooling实现
    # kernel size = 2 * radius + 1
    kernel_size = 2 * dilation_radius + 1
    
    # 确保kernel size不超过spatial dimensions
    spatial_shape = target.shape[2:]
    kernel_size = min(kernel_size, min(spatial_shape))
    if kernel_size % 2 == 0:
        kernel_size -= 1
    kernel_size = max(3, kernel_size)
    
    padding = kernel_size // 2
    
    # Max pooling 实现膨胀
    dilated_mask = F.max_pool3d(
        disc_mask, 
        kernel_size=kernel_size, 
        stride=1, 
        padding=padding
    )
    
    return dilated_mask


class TverskyLoss(nn.Module):
    """
    Tversky Loss - 适合小目标检测
    
    TL = 1 - (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
    
    alpha < beta → 更高的recall（对小目标有效）
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-5,
                 apply_nonlin: Callable = None, do_bg: bool = False):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.apply_nonlin = apply_nonlin
        self.do_bg = do_bg
    
    def forward(self, x: torch.Tensor, y: torch.Tensor,
                loss_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 转换为float32以避免fp16溢出
        x = x.float()
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        
        axes = tuple(range(2, x.ndim))
        
        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))
            
            if x.shape == y.shape:
                y_onehot = y.float()
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
                y_onehot.scatter_(1, y.long(), 1)
            
            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
        
        if not self.do_bg:
            x = x[:, 1:]
        
        if loss_mask is not None:
            loss_mask = loss_mask.float()
            tp = (x * y_onehot * loss_mask).sum(axes)
            fp = (x * (1 - y_onehot) * loss_mask).sum(axes)
            fn = ((1 - x) * y_onehot * loss_mask).sum(axes)
        else:
            tp = (x * y_onehot).sum(axes)
            fp = (x * (1 - y_onehot)).sum(axes)
            fn = ((1 - x) * y_onehot).sum(axes)
        
        tversky = (tp + self.smooth) / (torch.clamp(tp + self.alpha * fp + self.beta * fn + self.smooth, min=1e-8))
        return (1 - tversky).mean()


class PartialLDH_Loss(nn.Module):
    """
    针对LDH微小结构优化的组合Loss
    
    包含：Dice + CE + Tversky + Focal + Anatomical Attention
    """
    def __init__(self, soft_dice_kwargs: dict, ce_kwargs: dict,
                 weight_ce: float = 1.0, weight_dice: float = 1.0,
                 weight_tversky: float = 0.5, weight_focal: float = 0.5,
                 ignore_label=None, ldh_class_idx: int = 12,
                 ldh_class_weight: float = 3.0,
                 tversky_alpha: float = 0.3, tversky_beta: float = 0.7,
                 focal_gamma: float = 2.0,
                 use_anatomical_attention: bool = True,
                 disc_classes: Tuple[int, ...] = (1, 2, 3, 4, 5),
                 attention_dilation: int = 5,
                 outside_disc_penalty: float = 2.0):
        """
        Args:
            use_anatomical_attention: 是否使用解剖先验attention
            disc_classes: 椎间盘类别索引 (1-5)
            attention_dilation: 椎间盘区域膨胀半径
            outside_disc_penalty: 在椎间盘区域外预测LDH的惩罚系数
        """
        super().__init__()
        
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_tversky = weight_tversky
        self.weight_focal = weight_focal
        self.ignore_label = ignore_label
        self.ldh_class_idx = ldh_class_idx
        self.ldh_class_weight = ldh_class_weight
        self.focal_gamma = focal_gamma
        
        # Anatomical attention参数
        self.use_anatomical_attention = use_anatomical_attention
        self.disc_classes = disc_classes
        self.attention_dilation = attention_dilation
        self.outside_disc_penalty = outside_disc_penalty
        
        # Dice Loss参数
        self.batch_dice = soft_dice_kwargs.get('batch_dice', False)
        self.smooth = soft_dice_kwargs.get('smooth', 1e-5)
        self.do_bg = soft_dice_kwargs.get('do_bg', False)
        
        # Tversky Loss
        self.tversky = TverskyLoss(
            alpha=tversky_alpha, beta=tversky_beta,
            apply_nonlin=softmax_helper_dim1, do_bg=False
        )
        
        # CE ignore index
        self.ignore_index = ignore_label if ignore_label is not None else -100
    
    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        batch_size = net_output.shape[0]
        
        # 检测哪些样本有LDH标签
        with torch.no_grad():
            has_ldh = torch.zeros(batch_size, dtype=torch.bool, device=net_output.device)
            for b in range(batch_size):
                has_ldh[b] = (target[b] == self.ldh_class_idx).any()
            
            # 创建椎间盘区域attention mask
            if self.use_anatomical_attention:
                disc_attention = create_disc_attention_mask(
                    target, self.disc_classes, self.attention_dilation
                )
            else:
                disc_attention = None
        
        # 处理ignore label
        if self.ignore_label is not None:
            mask = target != self.ignore_label
            target_clean = torch.where(mask, target, 0)
        else:
            target_clean = target
            mask = None
        
        total_loss = 0.0
        
        # 1. Weighted Dice Loss (with partial label handling)
        if self.weight_dice > 0:
            dc_loss = self._compute_partial_dice(net_output, target_clean, has_ldh, mask)
            total_loss = total_loss + self.weight_dice * dc_loss
        
        # 2. CE Loss (with partial label handling)
        if self.weight_ce > 0:
            ce_loss = self._compute_partial_ce(net_output, target[:, 0], has_ldh)
            total_loss = total_loss + self.weight_ce * ce_loss
        
        # 3. Tversky Loss - 只对有LDH的样本计算
        if self.weight_tversky > 0 and has_ldh.any():
            ldh_indices = has_ldh.nonzero(as_tuple=True)[0]
            tversky_loss = self.tversky(
                net_output[ldh_indices], target_clean[ldh_indices],
                loss_mask=mask[ldh_indices] if mask is not None else None
            )
            total_loss = total_loss + self.weight_tversky * tversky_loss
        
        # 4. Focal Loss for LDH - 只对有LDH的样本计算
        if self.weight_focal > 0 and has_ldh.any():
            ldh_indices = has_ldh.nonzero(as_tuple=True)[0]
            disc_attn_subset = disc_attention[ldh_indices] if disc_attention is not None else None
            focal_loss = self._compute_ldh_focal(
                net_output[ldh_indices], target[ldh_indices, 0], disc_attn_subset
            )
            total_loss = total_loss + self.weight_focal * focal_loss
        
        # 5. Anatomical Penalty - 惩罚在椎间盘区域外预测LDH（只对有LDH标签的样本计算）
        if self.use_anatomical_attention and disc_attention is not None and has_ldh.any():
            ldh_indices = has_ldh.nonzero(as_tuple=True)[0]
            anat_penalty = self._compute_anatomical_penalty(
                net_output[ldh_indices], disc_attention[ldh_indices]
            )
            total_loss = total_loss + anat_penalty
        
        return total_loss
    
    def _compute_partial_dice(self, x: torch.Tensor, y: torch.Tensor,
                               has_ldh: torch.Tensor, loss_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """计算带partial label处理和LDH权重的Dice Loss"""
        # 转换为float32以避免fp16溢出
        x = softmax_helper_dim1(x.float())
        batch_size = x.shape[0]
        axes = tuple(range(2, x.ndim))
        
        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))
            
            # 使用float32类型以确保数值稳定性
            y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
            y_onehot.scatter_(1, y.long(), 1)
            
            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
            
            if loss_mask is not None:
                loss_mask_float = loss_mask.float()
                sum_gt = (y_onehot * loss_mask_float).sum(axes)
            else:
                sum_gt = y_onehot.sum(axes)
        
        if not self.do_bg:
            x = x[:, 1:]
        
        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes)
            sum_pred = x.sum(axes)
        else:
            intersect = (x * y_onehot * loss_mask_float).sum(axes)
            sum_pred = (x * loss_mask_float).sum(axes)
        
        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))
        
        # LDH index调整
        ldh_idx = self.ldh_class_idx - 1 if not self.do_bg else self.ldh_class_idx
        
        if ldh_idx < dc.shape[1]:
            class_weights = torch.ones_like(dc)
            # 有LDH的样本：LDH通道权重=ldh_class_weight
            # 没有LDH的样本：LDH通道权重=0（忽略）
            ldh_weight_values = torch.where(
                has_ldh,
                torch.full((batch_size,), self.ldh_class_weight, device=dc.device, dtype=dc.dtype),
                torch.zeros(batch_size, device=dc.device, dtype=dc.dtype)
            )
            class_weights[:, ldh_idx] = ldh_weight_values
            
            valid_weights = class_weights.sum(dim=1, keepdim=True)
            # 避免除零
            valid_weights = torch.clamp(valid_weights, min=1e-8)
            weighted_dc = (dc * class_weights).sum(dim=1) / valid_weights.squeeze(1)
            return -weighted_dc.mean()
        
        return -dc.mean()
    
    def _compute_partial_ce(self, x: torch.Tensor, y: torch.Tensor, has_ldh: torch.Tensor) -> torch.Tensor:
        """计算带partial label处理的CE Loss"""
        batch_size = x.shape[0]
        x_modified = x.clone()
        
        # 对没有LDH标签的样本，将LDH logits设为极负值
        # 注意：使用 -1e4 而不是 -1e6，因为 fp16 的范围约为 ±65504
        for b in range(batch_size):
            if not has_ldh[b]:
                x_modified[b, self.ldh_class_idx] = -1e4
        
        loss = F.cross_entropy(x_modified, y.long(), ignore_index=self.ignore_index, reduction='mean')
        return loss
    
    def _compute_ldh_focal(self, logits: torch.Tensor, target: torch.Tensor,
                           disc_attention: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算LDH的Focal Loss，带有解剖attention
        
        在椎间盘附近区域：正常计算focal loss
        在椎间盘区域外：增加对LDH预测的惩罚
        """
        # 使用float32计算以避免fp16溢出
        logits_fp32 = logits.float()
        probs = F.softmax(logits_fp32, dim=1)
        ldh_probs = probs[:, self.ldh_class_idx]
        ldh_target = (target == self.ldh_class_idx).float()
        
        alpha = 0.75
        gamma = self.focal_gamma
        # 使用更安全的clamp范围
        p = torch.clamp(ldh_probs, 1e-6, 1 - 1e-6)
        
        focal_weight_pos = alpha * ((1 - p) ** gamma)
        focal_weight_neg = (1 - alpha) * (p ** gamma)
        
        # 使用clamp限制log的输出范围，避免极端值
        log_p = torch.clamp(torch.log(p), min=-100)
        log_1_minus_p = torch.clamp(torch.log(1 - p), min=-100)
        
        bce_pos = -ldh_target * focal_weight_pos * log_p
        bce_neg = -(1 - ldh_target) * focal_weight_neg * log_1_minus_p
        
        focal_loss = bce_pos + bce_neg
        
        # 如果有解剖attention，在椎间盘区域内的loss给予更高权重
        if disc_attention is not None:
            # disc_attention: (B, 1, D, H, W), 需要squeeze
            disc_attn = disc_attention.squeeze(1)  # (B, D, H, W)
            
            # 椎间盘区域内权重=1，区域外权重更低（让网络更关注正确区域）
            # 但对于正样本（真实LDH），我们仍要学习，所以不降低权重
            spatial_weight = torch.where(
                ldh_target > 0,  # 真实LDH位置
                torch.ones_like(disc_attn),  # 权重=1
                disc_attn * 0.5 + 0.5  # 椎间盘外的负样本权重降低
            )
            focal_loss = focal_loss * spatial_weight
        
        return focal_loss.mean()
    
    def _compute_anatomical_penalty(self, net_output: torch.Tensor,
                                     disc_attention: torch.Tensor) -> torch.Tensor:
        """
        惩罚在椎间盘区域外预测LDH
        
        这个loss鼓励网络只在合理的解剖位置（椎间盘附近）预测LDH
        """
        # 使用float32计算以避免fp16溢出
        net_output_fp32 = net_output.float()
        probs = F.softmax(net_output_fp32, dim=1)
        ldh_probs = probs[:, self.ldh_class_idx]  # (B, D, H, W)
        
        # disc_attention: (B, 1, D, H, W) -> (B, D, H, W)
        disc_attn = disc_attention.squeeze(1).float()
        
        # 在椎间盘区域外（disc_attn=0）预测LDH的概率应该被惩罚
        outside_disc_mask = (1 - disc_attn)  # 椎间盘区域外=1
        
        # 惩罚 = 在区域外的LDH预测概率
        penalty = (ldh_probs * outside_disc_mask).mean() * self.outside_disc_penalty
        
        return penalty


class nnUNetTrainer_PartialLDH(nnUNetTrainer_DASegOrd0_NoMirroring):
    """
    针对LDH微小结构优化的nnUNet Trainer
    
    优化措施：
    1. Partial Label Handling: 没有LDH标签的样本不计算LDH通道loss
    2. LDH Class Weight (3x): 对LDH类别给予更高权重
    3. Tversky Loss (alpha=0.3, beta=0.7): 提高recall，适合小目标
    4. Focal Loss (gamma=2.0): 处理类别不平衡
    5. Foreground Oversampling (50%): 增加LDH样本出现概率
    6. Anatomical Attention: 利用椎间盘位置引导LDH检测
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        self.ldh_class_idx = 12
        
        # 提高前景采样比例，增加LDH样本出现概率
        self.oversample_foreground_percent = 0.5
    
    def _build_loss(self):
        loss = PartialLDH_Loss(
            soft_dice_kwargs={
                'batch_dice': self.configuration_manager.batch_dice,
                'smooth': 1e-5,
                'do_bg': False,
            },
            ce_kwargs={},
            weight_ce=1.0,
            weight_dice=1.0,
            weight_tversky=0.5,
            weight_focal=0.5,
            ignore_label=self.label_manager.ignore_label,
            ldh_class_idx=self.ldh_class_idx,
            ldh_class_weight=3.0,
            tversky_alpha=0.3,
            tversky_beta=0.7,
            focal_gamma=2.0,
            # Anatomical Attention参数
            use_anatomical_attention=True,
            disc_classes=(1, 2, 3, 4, 5),  # 椎间盘类别
            attention_dilation=5,  # 膨胀半径
            outside_disc_penalty=2.0  # 区域外惩罚系数
        )
        
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        
        return loss
