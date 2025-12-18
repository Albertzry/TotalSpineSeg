"""
Custom nnUNet Trainer for LDH Binary Segmentation (Step 5)

核心改进：将 Step1 二值掩码转换为 SDF（符号距离场）提供解剖先验

针对微小结构的优化：
1. Online Hard Example Mining (OHEM) - 聚焦难分割样本
2. 多通道解剖先验 - SDF + 距离通道
3. Log-Cosh Dice Loss - 对小偏移更稳定
4. GT 膨胀训练 - 扩大目标范围，提高召回
5. AdamW + Cosine Annealing - 更稳定的训练
"""

import torch
import numpy as np
from os.path import join
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple, Dict, List
from time import time
from scipy.ndimage import distance_transform_edt, binary_dilation, generate_binary_structure
import math

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDAOrd0 import nnUNetTrainer_DASegOrd0_NoMirroring
from nnunetv2.utilities.collate_outputs import collate_outputs


def dilate_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    膨胀二值掩码，用于 GT 膨胀训练
    
    对于微小结构，轻微膨胀 GT 可以：
    1. 提供更宽松的目标，减少完美对齐的压力
    2. 使边界更容易被检测
    3. 提高召回率
    """
    if mask.sum() == 0:
        return mask
    
    struct = generate_binary_structure(3, 1)  # 6-连通
    return binary_dilation(mask.astype(bool), structure=struct, iterations=iterations).astype(mask.dtype)


def compute_sdf_from_mask(mask: np.ndarray) -> np.ndarray:
    """
    计算符号距离场 (Signed Distance Field)
    
    SDF(x) = -d(x, boundary) if x in mask
           = +d(x, boundary) if x not in mask
    
    归一化到 [-1, 1] 范围
    
    Args:
        mask: 二值掩码 (D, H, W)
    
    Returns:
        sdf: 符号距离场，内部为负，外部为正，边界为0
    """
    mask = mask.astype(bool)
    
    if mask.sum() == 0:
        # 没有前景，返回全正值（表示都在外部）
        return np.ones_like(mask, dtype=np.float32)
    
    if (~mask).sum() == 0:
        # 全是前景，返回全负值（表示都在内部）
        return -np.ones_like(mask, dtype=np.float32)
    
    # 计算到前景边界的距离
    dist_outside = distance_transform_edt(~mask)  # 外部点到前景的距离
    dist_inside = distance_transform_edt(mask)    # 内部点到背景的距离
    
    # SDF: 外部为正，内部为负
    sdf = dist_outside - dist_inside
    
    # 归一化到 [-1, 1]
    max_abs = max(np.abs(sdf).max(), 1e-8)
    sdf = np.clip(sdf / max_abs, -1, 1)
    
    return sdf.astype(np.float32)


def compute_anatomical_prior(step1_pred: np.ndarray, 
                              disc_labels: Tuple[int, ...] = (1, 2, 3, 4, 5),
                              vertebrae_labels: Tuple[int, ...] = tuple(range(11, 25)),
                              canal_label: int = 8,
                              cord_label: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 Step1 预测计算解剖先验
    
    LDH 只会出现在椎间盘、椎骨、脊管三者之间的狭小区域
    
    返回:
        sdf_channel: 椎间盘 SDF，用作网络输入
        validity_map: 解剖有效性图 [0, 1]，用于 Loss 惩罚
                     值越高表示该位置越可能出现 LDH
    """
    # 1. 提取各结构掩码
    disc_mask = np.isin(step1_pred, disc_labels)
    vertebrae_mask = np.isin(step1_pred, vertebrae_labels)
    canal_cord_mask = (step1_pred == canal_label) | (step1_pred == cord_label)
    
    # 2. 计算到各结构的距离（欧氏距离）
    # 距离越小表示越靠近该结构
    if disc_mask.sum() > 0:
        dist_to_disc = distance_transform_edt(~disc_mask)
    else:
        dist_to_disc = np.ones_like(step1_pred, dtype=np.float32) * 100
    
    if vertebrae_mask.sum() > 0:
        dist_to_vertebrae = distance_transform_edt(~vertebrae_mask)
    else:
        dist_to_vertebrae = np.ones_like(step1_pred, dtype=np.float32) * 100
    
    if canal_cord_mask.sum() > 0:
        dist_to_canal = distance_transform_edt(~canal_cord_mask)
    else:
        dist_to_canal = np.ones_like(step1_pred, dtype=np.float32) * 100
    
    # 3. 计算椎间盘 SDF 作为网络输入
    disc_sdf = compute_sdf_from_mask(disc_mask)
    
    # 4. 计算解剖有效性图
    # LDH 的解剖约束：
    # - 必须靠近椎间盘边界（不在椎间盘深处，也不太远离椎间盘）
    # - 靠近脊管/脊髓
    # - 在椎骨间隙内
    
    # 定义距离阈值（单位：体素，可根据实际情况调整）
    disc_boundary_range = 15.0   # 距离椎间盘边界的有效范围
    canal_range = 20.0           # 距离脊管的有效范围
    vertebrae_range = 10.0       # 距离椎骨的有效范围
    
    # 各约束的软得分 (0-1)，使用高斯衰减
    # 椎间盘边界约束：在边界附近最高，向内向外衰减
    disc_boundary_dist = np.abs(disc_sdf) * dist_to_disc.max()  # 转回实际距离
    disc_score = np.exp(-0.5 * (disc_boundary_dist / disc_boundary_range) ** 2)
    
    # 脊管约束：越靠近脊管得分越高
    canal_score = np.exp(-0.5 * (dist_to_canal / canal_range) ** 2)
    
    # 椎骨约束：在椎骨附近（椎间孔区域）
    vertebrae_score = np.exp(-0.5 * (dist_to_vertebrae / vertebrae_range) ** 2)
    
    # 组合：三者都需要满足，使用几何平均
    validity_map = (disc_score * canal_score * vertebrae_score) ** (1/3)
    
    # 归一化到 [0, 1]
    if validity_map.max() > 0:
        validity_map = validity_map / validity_map.max()
    
    return disc_sdf.astype(np.float32), validity_map.astype(np.float32)


def compute_boundary_distance_map(mask: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    计算到 GT 边界的距离图（用于 Boundary Loss）
    
    对于微小结构，边界损失比纯 Dice 更宽容：
    - 预测位置稍有偏移时，惩罚较轻
    - 完全远离 GT 时，惩罚较重
    """
    mask_np = mask.cpu().numpy()
    batch_size = mask_np.shape[0]
    dist_maps = []
    
    for b in range(batch_size):
        m = mask_np[b].astype(bool)
        if m.sum() == 0:
            # 没有前景，距离设为大值
            dist = np.ones_like(m, dtype=np.float32) * 50.0
        elif (~m).sum() == 0:
            # 全是前景
            dist = np.zeros_like(m, dtype=np.float32)
        else:
            # 计算到边界的距离
            dist_outside = distance_transform_edt(~m)
            dist_inside = distance_transform_edt(m)
            # 外部为正，内部为负（类似 SDF）
            dist = dist_outside - dist_inside
        
        if normalize:
            # 归一化到 [-1, 1]
            max_abs = max(np.abs(dist).max(), 1e-8)
            dist = dist / max_abs
        
        dist_maps.append(dist)
    
    return torch.from_numpy(np.stack(dist_maps, axis=0)).to(mask.device).float()


class LDH_AnatomicalLoss(nn.Module):
    """
    LDH Loss with Anatomical Guidance + Micro-structure Optimizations
    
    针对微小结构的优化（目标 Dice > 0.8）：
    1. Log-Cosh Dice: 对小偏移更稳定，减少梯度爆炸
    2. Focal Tversky: 强调召回率
    3. Online Hard Example Mining: 聚焦难分割体素
    4. Boundary-aware Loss: 对边界附近更宽容
    5. Anatomical Guidance: 约束预测区域
    
    Loss = Log-Cosh Dice + Focal CE + Focal Tversky + OHEM + Anatomical Penalty
    """
    def __init__(self,
                 weight_dice: float = 1.0,
                 weight_ce: float = 1.0,
                 weight_tversky: float = 0.5,
                 weight_ohem: float = 0.3,          # OHEM 权重
                 weight_anatomical: float = 0.5,
                 tversky_alpha: float = 0.3,        # FP 权重
                 tversky_beta: float = 0.7,         # FN 权重（高召回）
                 focal_gamma: float = 2.0,          # Focal 参数
                 ce_pos_weight: float = 8.0,        # 正样本权重
                 ohem_ratio: float = 0.3,           # OHEM 选取比例
                 smooth: float = 1e-5,
                 use_log_cosh: bool = True,         # 使用 Log-Cosh Dice
                 ignore_label=None):
        super().__init__()
        
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_tversky = weight_tversky
        self.weight_ohem = weight_ohem
        self.weight_anatomical = weight_anatomical
        
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.focal_gamma = focal_gamma
        self.ce_pos_weight = ce_pos_weight
        self.ohem_ratio = ohem_ratio
        self.smooth = smooth
        self.use_log_cosh = use_log_cosh
        self.ignore_label = ignore_label
        
        self.current_validity_map = None
    
    def set_validity_map(self, validity_map: torch.Tensor):
        """设置当前 batch 的解剖有效性图"""
        self.current_validity_map = validity_map
    
    def log_cosh_dice_loss(self, probs: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Log-Cosh Dice Loss
        
        优势：
        - 对小偏移更平滑（梯度不会爆炸）
        - 接近 L1 的行为但处处可微
        - 对异常值更鲁棒
        """
        intersection = (probs * gt).sum()
        union = probs.sum() + gt.sum()
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice
        
        # log(cosh(x)) ≈ |x| for large x, ≈ x²/2 for small x
        return torch.log(torch.cosh(dice_loss + 1e-8))
    
    def focal_bce_loss(self, logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Focal BCE Loss
        
        聚焦于难分类的样本，降低易分类样本的权重
        """
        bce = F.binary_cross_entropy_with_logits(logits, gt, reduction='none')
        probs = torch.sigmoid(logits)
        
        # Focal 权重: (1-p)^γ for positive, p^γ for negative
        pt = probs * gt + (1 - probs) * (1 - gt)
        focal_weight = (1 - pt) ** self.focal_gamma
        
        # 正样本加权
        pos_weight = gt * self.ce_pos_weight + (1 - gt)
        
        return (focal_weight * pos_weight * bce).mean()
    
    def ohem_loss(self, probs: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Online Hard Example Mining
        
        只选择最难的 K% 体素计算 loss，聚焦于边界和难分区域
        """
        # 计算逐体素 loss
        per_voxel_loss = -gt * torch.log(probs + 1e-8) - (1 - gt) * torch.log(1 - probs + 1e-8)
        
        # 选择最难的 K% 体素
        k = max(int(per_voxel_loss.numel() * self.ohem_ratio), 1)
        top_k_loss, _ = torch.topk(per_voxel_loss.view(-1), k)
        
        return top_k_loss.mean()
    
    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        net_output_fp32 = net_output.float()
        
        # 获取预测概率和 logits
        if net_output_fp32.ndim == 5 and net_output_fp32.shape[1] == 2:
            probs = torch.softmax(net_output_fp32, dim=1)[:, 1]
            logits = net_output_fp32[:, 1]
        else:
            probs = torch.sigmoid(net_output_fp32)
            if probs.ndim == 5 and probs.shape[1] == 1:
                probs = probs[:, 0]
                logits = net_output_fp32[:, 0]
            else:
                logits = net_output_fp32
        
        # 获取 GT
        if target.ndim == 5:
            gt = (target[:, 0] == 1).float()
        else:
            gt = (target == 1).float()
        
        total_loss = 0.0
        has_foreground = gt.sum() > 0
        
        # 1. Log-Cosh Dice Loss（更稳定）
        if self.weight_dice > 0:
            if self.use_log_cosh:
                dice_loss = self.log_cosh_dice_loss(probs, gt)
            else:
                intersection = (probs * gt).sum()
                union = probs.sum() + gt.sum()
                dice = (2 * intersection + self.smooth) / (union + self.smooth)
                dice_loss = 1 - dice
            total_loss = total_loss + self.weight_dice * dice_loss
        
        # 2. Focal BCE Loss（聚焦难样本）
        if self.weight_ce > 0:
            focal_ce = self.focal_bce_loss(logits, gt)
            total_loss = total_loss + self.weight_ce * focal_ce
        
        # 3. Focal Tversky Loss（强调召回）
        if self.weight_tversky > 0:
            tp = (probs * gt).sum()
            fp = (probs * (1 - gt)).sum()
            fn = ((1 - probs) * gt).sum()
            
            tversky_index = (tp + self.smooth) / (
                tp + self.tversky_alpha * fp + self.tversky_beta * fn + self.smooth
            )
            # Focal 版本
            focal_tversky = torch.pow(1 - tversky_index, 1.0 / self.focal_gamma)
            total_loss = total_loss + self.weight_tversky * focal_tversky
        
        # 4. OHEM Loss（聚焦边界）
        if self.weight_ohem > 0 and has_foreground:
            ohem = self.ohem_loss(probs, gt)
            total_loss = total_loss + self.weight_ohem * ohem
        
        # 5. Anatomical Guidance Loss
        if self.weight_anatomical > 0 and self.current_validity_map is not None:
            validity = self.current_validity_map
            
            if validity.shape != probs.shape:
                validity = F.interpolate(
                    validity.unsqueeze(1), 
                    size=probs.shape[-3:], 
                    mode='trilinear', 
                    align_corners=False
                ).squeeze(1)
            
            invalid_region = 1.0 - validity
            anatomical_penalty = (probs * invalid_region).mean()
            total_loss = total_loss + self.weight_anatomical * anatomical_penalty
        
        return total_loss


class nnUNetTrainer_LDH(nnUNetTrainer_DASegOrd0_NoMirroring):
    """
    LDH Trainer with Anatomical Prior from Disc-Vertebrae-Canal Interface
    
    核心改进：
    1. 在线将 Step1 预测转换为 SDF（网络输入）
    2. 计算解剖有效性图（Loss 惩罚）
    3. LDH 预测被约束在椎间盘-椎骨-脊管三者之间的狭小区域
    
    Loss: Dice + CE + Focal Tversky + Anatomical Penalty
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        self.print_to_log_file("=" * 60)
        self.print_to_log_file("LDH Trainer: Anatomical Prior from Disc-Vertebrae-Canal")
        self.print_to_log_file("LDH region constrained to interface of 3 structures")
        self.print_to_log_file("DEBUG: Custom validation_step is enabled")
        self.print_to_log_file("=" * 60)
        
        self.oversample_foreground_percent = 0.9
        self.num_epochs = 1000
        
        # 解剖结构标签
        self.disc_labels = (1, 2, 3, 4, 5)
        self.vertebrae_labels = tuple(range(11, 25))  # 椎骨 C1-S1
        self.canal_label = 8
        self.cord_label = 9
        
        # 存储当前 batch 的 validity map
        self.current_validity_maps = None
    
    def _process_anatomical_prior(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理 Step1 预测，生成 SDF 和解剖有效性图
        
        Args:
            data: (B, C, D, H, W) 输入数据
                  C=0: MRI 图像
                  C=1: Step1 预测（整数标签）
        
        Returns:
            data: 第二通道已转换为 SDF
            validity_maps: (B, D, H, W) 解剖有效性图
        """
        if data.shape[1] < 2:
            return data, None
        
        batch_size = data.shape[0]
        device = data.device
        
        # 获取 Step1 通道
        step1_channel = data[:, 1].cpu().numpy()
        
        sdf_batch = []
        validity_batch = []
        
        for b in range(batch_size):
            sdf, validity = compute_anatomical_prior(
                step1_channel[b],
                disc_labels=self.disc_labels,
                vertebrae_labels=self.vertebrae_labels,
                canal_label=self.canal_label,
                cord_label=self.cord_label
            )
            sdf_batch.append(sdf)
            validity_batch.append(validity)
        
        # 转回 tensor
        sdf_tensor = torch.from_numpy(np.stack(sdf_batch, axis=0)).to(device).float()
        validity_tensor = torch.from_numpy(np.stack(validity_batch, axis=0)).to(device).float()
        
        # 替换第二通道为 SDF
        data = data.clone()
        data[:, 1] = sdf_tensor
        
        return data, validity_tensor
    
    def _apply_gt_dilation(self, target: torch.Tensor) -> torch.Tensor:
        """
        对 GT 进行膨胀（仅训练时）
        
        对于微小结构，轻微膨胀 GT 可以：
        - 提供更宽松的目标
        - 提高召回率
        - 减少边界对齐的严格要求
        """
        if not self.use_gt_dilation or self.gt_dilation_iterations == 0:
            return target
        
        if isinstance(target, list):
            # Deep supervision 情况
            dilated = []
            for t in target:
                t_np = t.cpu().numpy()
                batch_dilated = []
                for b in range(t_np.shape[0]):
                    mask = (t_np[b, 0] == 1).astype(np.float32)
                    if mask.sum() > 0:
                        mask = dilate_mask(mask, self.gt_dilation_iterations)
                    batch_dilated.append(mask)
                dilated_tensor = torch.from_numpy(np.stack(batch_dilated)[:, None]).to(t.device).float()
                dilated.append(dilated_tensor)
            return dilated
        else:
            t_np = target.cpu().numpy()
            batch_dilated = []
            for b in range(t_np.shape[0]):
                if target.ndim == 5:
                    mask = (t_np[b, 0] == 1).astype(np.float32)
                else:
                    mask = (t_np[b] == 1).astype(np.float32)
                if mask.sum() > 0:
                    mask = dilate_mask(mask, self.gt_dilation_iterations)
                batch_dilated.append(mask)
            
            if target.ndim == 5:
                return torch.from_numpy(np.stack(batch_dilated)[:, None]).to(target.device).float()
            else:
                return torch.from_numpy(np.stack(batch_dilated)).to(target.device).float()
    
    def train_step(self, batch: dict) -> dict:
        """重写训练步骤，添加解剖先验处理和GT膨胀"""
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        
        # 处理解剖先验
        data, validity_maps = self._process_anatomical_prior(data)
        
        # GT 膨胀（仅训练时）
        target_for_loss = self._apply_gt_dilation(target)
        
        # 设置 validity map 到 loss
        if hasattr(self.loss, 'set_validity_map'):
            self.loss.set_validity_map(validity_maps)
        elif hasattr(self.loss, 'loss') and hasattr(self.loss.loss, 'set_validity_map'):
            # DeepSupervisionWrapper 情况
            self.loss.loss.set_validity_map(validity_maps)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else torch.autocast('cpu', enabled=False):
            output = self.network(data)
            l = self.loss(output, target_for_loss)
        
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        
        return {'loss': l.detach().cpu().numpy()}
    
    def validation_step(self, batch: dict) -> dict:
        """重写验证步骤，添加解剖先验处理"""
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        
        # 处理解剖先验
        data, validity_maps = self._process_anatomical_prior(data)
        
        # 设置 validity map 到 loss
        if hasattr(self.loss, 'set_validity_map'):
            self.loss.set_validity_map(validity_maps)
        elif hasattr(self.loss, 'loss') and hasattr(self.loss.loss, 'set_validity_map'):
            self.loss.loss.set_validity_map(validity_maps)
        
        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else torch.autocast('cpu', enabled=False):
                output = self.network(data)
                l = self.loss(output, target)
                
                # 计算 tp, fp, fn 用于 pseudo dice
                if self.enable_deep_supervision:
                    output_seg = output[0]
                    target_seg = target[0]
                else:
                    output_seg = output
                    target_seg = target
                
                # 获取预测
                if output_seg.shape[1] == 2:
                    predicted_segmentation = output_seg.argmax(1)
                else:
                    predicted_segmentation = (torch.sigmoid(output_seg) > 0.5).long().squeeze(1)
                
                # 获取 GT
                if target_seg.ndim == 5:
                    gt_seg = target_seg[:, 0].long()
                else:
                    gt_seg = target_seg.long()
                
                # 计算 tp, fp, fn (形状需要是 [batch_size, num_classes])
                axes = tuple(range(1, predicted_segmentation.ndim))
                tp_hard = ((predicted_segmentation == 1) & (gt_seg == 1)).sum(dim=axes)
                fp_hard = ((predicted_segmentation == 1) & (gt_seg == 0)).sum(dim=axes)
                fn_hard = ((predicted_segmentation == 0) & (gt_seg == 1)).sum(dim=axes)
                
                # 增加类别维度: (batch_size,) -> (batch_size, 1) 表示1个前景类
                tp_hard = tp_hard.unsqueeze(1)
                fp_hard = fp_hard.unsqueeze(1)
                fn_hard = fn_hard.unsqueeze(1)
        
        return {
            'loss': l.detach().cpu().numpy(),
            'tp_hard': tp_hard.detach().cpu().numpy(),
            'fp_hard': fp_hard.detach().cpu().numpy(),
            'fn_hard': fn_hard.detach().cpu().numpy()
        }
    
    def _build_loss(self):
        loss = LDH_AnatomicalLoss(
            weight_dice=1.0,           # Log-Cosh Dice
            weight_ce=1.0,             # Focal BCE
            weight_tversky=0.5,        # Focal Tversky（强调召回）
            weight_ohem=0.3,           # OHEM（聚焦边界）
            weight_anatomical=0.5,     # 解剖约束
            tversky_alpha=0.3,         # FP 权重（低）
            tversky_beta=0.7,          # FN 权重（高召回）
            focal_gamma=2.0,           # Focal 参数
            ce_pos_weight=8.0,         # 正样本权重
            ohem_ratio=0.3,            # 选取最难的 30%
            smooth=1e-5,
            use_log_cosh=True,         # 使用 Log-Cosh Dice
            ignore_label=self.label_manager.ignore_label,
        )
        
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        
        return loss
    
    def configure_optimizers(self):
        if self.use_adamw:
            # AdamW: 更稳定，适合微小结构
            optimizer = torch.optim.AdamW(
                self.network.parameters(),
                lr=self.adamw_lr,
                weight_decay=self.adamw_weight_decay,
                betas=(0.9, 0.999)
            )
            # Cosine Annealing with Warm Restarts
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=50,           # 首次重启周期
                T_mult=2,         # 每次重启周期翻倍
                eta_min=1e-7      # 最小学习率
            )
        else:
            # 原始 SGD
            optimizer = torch.optim.SGD(
                self.network.parameters(),
                lr=self.initial_lr,
                momentum=0.99,
                weight_decay=3e-5,
                nesterov=True
            )
            lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
                optimizer, total_iters=self.num_epochs, power=0.9
            )
        
        return optimizer, lr_scheduler

