"""
Custom nnUNet Trainer for LDH Binary Segmentation (Step 5)

针对微小结构 LDH 的优化策略，利用解剖关系增强 Attention：

LDH (椎间盘突出) 的解剖特点：
1. 只发生在椎间盘边界区域（不是椎间盘内部）
2. 位于椎间盘和脊髓/椎管之间
3. 位于相邻椎骨之间的区域
4. 通常向后突出（posterior）

利用 Step 1 预测的解剖结构创建多层次 Attention：
- Disc (椎间盘): labels 1-5
- Vertebrae (椎骨): labels 6-7  
- Canal (椎管): label 8
- Cord (脊髓): label 9

Anatomical Attention 机制：
1. Disc Boundary Attention: 椎间盘边界区域（LDH 起源位置）
2. Disc-Cord Interface: 椎间盘和脊髓/椎管之间的区域
3. Inter-vertebral Region: 椎骨之间的区域
4. 综合 Attention Map: 以上区域的加权组合

Loss 函数组合：
1. Dice + CE (基础)
2. Focal Tversky Loss (小目标优化)
3. Boundary Loss (边界精度)
4. Multi-structure Anatomical Attention (解剖约束)

参考文献:
- Focal Tversky Loss: Abraham & Khan (2019)
- Boundary Loss: Kervadec et al. (2019)
"""

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple
from scipy.ndimage import distance_transform_edt

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDAOrd0 import nnUNetTrainer_DASegOrd0_NoMirroring


def compute_distance_transform(target: torch.Tensor) -> torch.Tensor:
    """计算距离变换，用于 Boundary Loss"""
    target_np = target.cpu().numpy()
    batch_size = target_np.shape[0]
    dist_maps = np.zeros_like(target_np, dtype=np.float32)
    
    for b in range(batch_size):
        seg = target_np[b]
        if seg.sum() > 0:
            pos_dist = distance_transform_edt(seg)
            neg_dist = distance_transform_edt(1 - seg)
            dist_maps[b] = neg_dist - pos_dist
        else:
            dist_maps[b] = distance_transform_edt(np.ones_like(seg))
    
    return torch.from_numpy(dist_maps).to(target.device).float()


def dilate_3d(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """3D 膨胀操作"""
    if mask.sum() < 1:
        return mask
    
    kernel_size = 2 * radius + 1
    if mask.ndim == 3:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 4:
        mask = mask.unsqueeze(1)
    
    spatial_shape = mask.shape[2:]
    kernel_size = min(kernel_size, min(spatial_shape))
    if kernel_size % 2 == 0:
        kernel_size -= 1
    kernel_size = max(3, kernel_size)
    padding = kernel_size // 2
    
    dilated = F.max_pool3d(mask.float(), kernel_size=kernel_size, stride=1, padding=padding)
    return dilated


def erode_3d(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """3D 腐蚀操作"""
    if mask.sum() < 1:
        return mask
    
    kernel_size = 2 * radius + 1
    if mask.ndim == 3:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 4:
        mask = mask.unsqueeze(1)
    
    spatial_shape = mask.shape[2:]
    kernel_size = min(kernel_size, min(spatial_shape))
    if kernel_size % 2 == 0:
        kernel_size -= 1
    kernel_size = max(3, kernel_size)
    padding = kernel_size // 2
    
    # 腐蚀 = 对反转图像做膨胀后再反转
    eroded = 1 - F.max_pool3d(1 - mask.float(), kernel_size=kernel_size, stride=1, padding=padding)
    return eroded


class AnatomicalAttention(nn.Module):
    """
    基于解剖结构的多层次 Attention 机制
    
    利用 Step 1 预测的解剖结构创建 LDH 可能出现区域的 attention map：
    
    Step 1 标签格式:
    - 1-5: 椎间盘 (disc)
    - 6-7: 椎骨 (vertebrae)
    - 8: 椎管 (canal)
    - 9: 脊髓 (cord)
    
    Attention 组成:
    1. disc_boundary: 椎间盘边界区域 (LDH 起源)
    2. disc_cord_interface: 椎间盘-脊髓/椎管接触区域 (LDH 突出方向)
    3. inter_vertebral: 椎骨之间区域
    """
    def __init__(self,
                 disc_labels: Tuple[int, ...] = (1, 2, 3, 4, 5),
                 vertebrae_labels: Tuple[int, ...] = (6, 7),
                 canal_label: int = 8,
                 cord_label: int = 9,
                 disc_dilation: int = 3,      # 椎间盘边界膨胀
                 interface_dilation: int = 5,  # 接触区域膨胀
                 vertebrae_dilation: int = 3): # 椎骨间区域膨胀
        super().__init__()
        self.disc_labels = disc_labels
        self.vertebrae_labels = vertebrae_labels
        self.canal_label = canal_label
        self.cord_label = cord_label
        self.disc_dilation = disc_dilation
        self.interface_dilation = interface_dilation
        self.vertebrae_dilation = vertebrae_dilation
    
    def forward(self, step1_pred: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        从 Step 1 预测创建解剖 attention map
        
        Args:
            step1_pred: (B, D, H, W) 或 (B, 1, D, H, W) Step 1 预测
        
        Returns:
            attention_map: (B, D, H, W) 综合 attention map
            components: dict 包含各组成部分
        """
        if step1_pred.ndim == 5:
            step1_pred = step1_pred[:, 0]  # (B, D, H, W)
        
        batch_size = step1_pred.shape[0]
        device = step1_pred.device
        
        # 提取各解剖结构
        disc_mask = torch.zeros_like(step1_pred, dtype=torch.float32)
        for d in self.disc_labels:
            disc_mask = disc_mask + (step1_pred == d).float()
        disc_mask = (disc_mask > 0).float()
        
        vertebrae_mask = torch.zeros_like(step1_pred, dtype=torch.float32)
        for v in self.vertebrae_labels:
            vertebrae_mask = vertebrae_mask + (step1_pred == v).float()
        vertebrae_mask = (vertebrae_mask > 0).float()
        
        canal_mask = (step1_pred == self.canal_label).float()
        cord_mask = (step1_pred == self.cord_label).float()
        
        # 脊髓/椎管组合区域
        cord_canal_mask = torch.clamp(cord_mask + canal_mask, 0, 1)
        
        components = {}
        
        # ==================== 1. 椎间盘边界区域 ====================
        # LDH 起源于椎间盘边界，不是内部
        if disc_mask.sum() > 0:
            disc_dilated = dilate_3d(disc_mask, self.disc_dilation)
            disc_eroded = erode_3d(disc_mask, 1)
            # 边界 = 膨胀 - 腐蚀
            disc_boundary = torch.clamp(disc_dilated.squeeze(1) - disc_eroded.squeeze(1), 0, 1)
        else:
            disc_boundary = torch.zeros_like(disc_mask)
        components['disc_boundary'] = disc_boundary
        
        # ==================== 2. 椎间盘-脊髓/椎管接触区域 ====================
        # LDH 通常向后突出，压迫脊髓/椎管
        if disc_mask.sum() > 0 and cord_canal_mask.sum() > 0:
            disc_dilated = dilate_3d(disc_mask, self.interface_dilation)
            cord_canal_dilated = dilate_3d(cord_canal_mask, self.interface_dilation)
            # 接触区域 = 两者膨胀后的交集
            interface_region = disc_dilated.squeeze(1) * cord_canal_dilated.squeeze(1)
        else:
            interface_region = torch.zeros_like(disc_mask)
        components['disc_cord_interface'] = interface_region
        
        # ==================== 3. 椎骨之间区域 ====================
        # LDH 发生在相邻椎骨之间
        if vertebrae_mask.sum() > 0:
            vertebrae_dilated = dilate_3d(vertebrae_mask, self.vertebrae_dilation)
            # 椎骨间 = 椎骨膨胀区域 - 椎骨本身
            inter_vertebral = torch.clamp(vertebrae_dilated.squeeze(1) - vertebrae_mask, 0, 1)
            # 还需要与椎间盘区域重叠
            if disc_mask.sum() > 0:
                disc_region = dilate_3d(disc_mask, self.disc_dilation + 2).squeeze(1)
                inter_vertebral = inter_vertebral * disc_region
        else:
            inter_vertebral = torch.zeros_like(disc_mask)
        components['inter_vertebral'] = inter_vertebral
        
        # ==================== 综合 Attention Map ====================
        # 加权组合各区域
        # disc_cord_interface 最重要（LDH 突出压迫的区域）
        # disc_boundary 次之（LDH 起源）
        # inter_vertebral 辅助
        attention_map = (
            0.5 * components['disc_cord_interface'] +
            0.3 * components['disc_boundary'] +
            0.2 * components['inter_vertebral']
        )
        
        # 归一化到 [0.1, 1]，避免完全为 0
        attention_map = torch.clamp(attention_map, 0, 1)
        attention_map = 0.1 + 0.9 * attention_map  # 最小值 0.1
        
        # 如果没有检测到任何解剖结构，返回全 1
        if disc_mask.sum() < 1:
            attention_map = torch.ones_like(step1_pred)
        
        return attention_map, components


class LDH_MicroStructure_Loss(nn.Module):
    """
    针对微小结构 LDH 优化的综合 Loss，利用解剖 Attention
    
    组成部分:
    1. Dice Loss + CrossEntropy Loss (基础)
    2. Focal Tversky Loss (小目标优化)
    3. Boundary Loss (边界精度)
    4. Anatomical Attention Weighted Loss (解剖约束)
    5. Outside Anatomical Region Penalty (区域外惩罚)
    """
    def __init__(self,
                 # 基础 loss 权重
                 weight_dice: float = 1.0,
                 weight_ce: float = 1.0,
                 # 小目标优化 loss 权重
                 weight_focal_tversky: float = 0.5,
                 weight_boundary: float = 0.5,
                 # Focal Tversky 参数
                 tversky_alpha: float = 0.3,
                 tversky_beta: float = 0.7,
                 focal_gamma: float = 0.75,
                 # CE 权重
                 ce_pos_weight: float = 10.0,
                 # 解剖 Attention 参数
                 use_anatomical_attention: bool = True,
                 anatomical_penalty_weight: float = 2.0,
                 # 其他
                 smooth: float = 1e-5,
                 ignore_label=None):
        super().__init__()
        
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_focal_tversky = weight_focal_tversky
        self.weight_boundary = weight_boundary
        
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.focal_gamma = focal_gamma
        
        self.ce_pos_weight = ce_pos_weight
        
        self.use_anatomical_attention = use_anatomical_attention
        self.anatomical_penalty_weight = anatomical_penalty_weight
        
        self.smooth = smooth
        self.ignore_label = ignore_label
        
        # 解剖 Attention 模块
        self.anatomical_attention = AnatomicalAttention(
            disc_labels=(1, 2, 3, 4, 5),
            vertebrae_labels=(6, 7),
            canal_label=8,
            cord_label=9,
            disc_dilation=3,
            interface_dilation=5,
            vertebrae_dilation=3
        )
    
    def forward(self, net_output: torch.Tensor, target: torch.Tensor,
                step1_channel: Optional[torch.Tensor] = None):
        """
        Args:
            net_output: (B, 1, D, H, W) 网络输出 logits
            target: (B, 1, D, H, W) GT 标签 (0=background, 1=LDH)
            step1_channel: (B, 1, D, H, W) Step 1 预测
        """
        # 获取 LDH 概率
        net_output_fp32 = net_output.float()
        
        # 处理 2通道 (Background, LDH) 或 1通道 (LDH) 输出
        if net_output_fp32.ndim == 5 and net_output_fp32.shape[1] == 2:
            # 2通道: 使用 Softmax
            ldh_probs = torch.softmax(net_output_fp32, dim=1)
            ldh_probs_squeezed = ldh_probs[:, 1]  # 取 LDH 通道
            logits_squeezed = net_output_fp32     # 保留完整 logits 用于 CE
        else:
            # 1通道: 使用 Sigmoid
            ldh_probs = torch.sigmoid(net_output_fp32)
            if ldh_probs.ndim == 5 and ldh_probs.shape[1] == 1:
                ldh_probs_squeezed = ldh_probs[:, 0]
                logits_squeezed = net_output_fp32[:, 0]
            else:
                ldh_probs_squeezed = ldh_probs
                logits_squeezed = net_output_fp32
        
        # 获取 LDH GT
        if target.ndim == 5:
            ldh_target = (target[:, 0] == 1).float()
        else:
            ldh_target = (target == 1).float()
        
        # 计算解剖 Attention
        attention_map = None
        attention_components = None
        if self.use_anatomical_attention and step1_channel is not None:
            attention_map, attention_components = self.anatomical_attention(step1_channel)
        
        total_loss = 0.0
        
        # ==================== 1. Anatomical Attention Weighted Dice ====================
        if self.weight_dice > 0:
            dice_loss = self._attention_weighted_dice(
                ldh_probs_squeezed, ldh_target, attention_map
            )
            total_loss = total_loss + self.weight_dice * dice_loss
        
        # ==================== 2. Weighted CE Loss ====================
        if self.weight_ce > 0:
            ce_loss = self._weighted_ce_loss(logits_squeezed, ldh_target)
            total_loss = total_loss + self.weight_ce * ce_loss
        
        # ==================== 3. Focal Tversky Loss ====================
        if self.weight_focal_tversky > 0:
            ft_loss = self._focal_tversky_loss(ldh_probs_squeezed, ldh_target)
            total_loss = total_loss + self.weight_focal_tversky * ft_loss
        
        # ==================== 4. Boundary Loss ====================
        if self.weight_boundary > 0 and ldh_target.sum() > 0:
            boundary_loss = self._boundary_loss(ldh_probs_squeezed, ldh_target)
            total_loss = total_loss + self.weight_boundary * boundary_loss
        
        # ==================== 5. Anatomical Region Penalty ====================
        if self.use_anatomical_attention and attention_map is not None:
            anat_penalty = self._anatomical_penalty(ldh_probs_squeezed, attention_map)
            total_loss = total_loss + anat_penalty
        
        return total_loss
    
    def _attention_weighted_dice(self, pred: torch.Tensor, target: torch.Tensor,
                                  attention: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Attention 加权的 Dice Loss
        在解剖相关区域给予更高权重
        """
        if attention is not None:
            # 在 attention 区域内的 dice 更重要
            weighted_pred = pred * attention
            weighted_target = target * attention
            
            intersection = (weighted_pred * weighted_target).sum()
            union = weighted_pred.sum() + weighted_target.sum()
        else:
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice
    
    def _weighted_ce_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """带权重的 CrossEntropyLoss"""
        # 确保 target 是 long 类型 (B, D, H, W)
        if target.ndim == 5:
            target = target[:, 0]
        
        # 权重: 背景=1.0, LDH=ce_pos_weight
        weights = torch.tensor([1.0, self.ce_pos_weight], device=logits.device)
        
        # 使用 CrossEntropyLoss
        return F.cross_entropy(logits, target.long(), weight=weights, reduction='mean')
    
    def _focal_tversky_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Focal Tversky Loss - 小目标优化"""
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        
        tversky_index = (tp + self.smooth) / (
            tp + self.tversky_alpha * fp + self.tversky_beta * fn + self.smooth
        )
        
        return torch.pow(1 - tversky_index, self.focal_gamma)
    
    def _boundary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Boundary Loss - 边界精度"""
        dist_map = compute_distance_transform(target)
        boundary_loss = (pred * dist_map).mean()
        return torch.tanh(boundary_loss / 10.0)
    
    def _anatomical_penalty(self, pred: torch.Tensor, 
                            attention_map: torch.Tensor) -> torch.Tensor:
        """
        惩罚在解剖不相关区域预测 LDH
        
        attention_map 值低的区域是解剖上不太可能出现 LDH 的地方
        """
        # attention_map: 高值表示可能区域，低值表示不可能区域
        # 惩罚在低 attention 区域的预测
        low_attention_region = 1 - attention_map  # 反转
        penalty = (pred * low_attention_region).mean() * self.anatomical_penalty_weight
        return penalty


class nnUNetTrainer_LDH(nnUNetTrainer_DASegOrd0_NoMirroring):
    """
    针对微小结构 LDH 优化的 nnUNet Trainer
    
    利用多结构解剖 Attention：
    1. 椎间盘边界区域 (disc boundary) - LDH 起源
    2. 椎间盘-脊髓/椎管接触区域 (disc-cord interface) - LDH 突出方向
    3. 椎骨之间区域 (inter-vertebral) - LDH 发生位置
    
    Loss 组合:
    - Dice + CE (基础)
    - Focal Tversky (小目标)
    - Boundary Loss (边界)
    - Anatomical Attention (解剖约束)
    
    网络输出: 只有 LDH (binary)
    输入通道:
      - Channel 0: MRI 图像
      - Channel 1: Step 1 预测 (解剖上下文)
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        self.print_to_log_file("=" * 60)
        self.print_to_log_file("LDH Micro-Structure Trainer with Anatomical Attention")
        self.print_to_log_file("=" * 60)
        self.print_to_log_file("")
        self.print_to_log_file("网络输出: 只有 LDH (binary)")
        self.print_to_log_file("输入通道:")
        self.print_to_log_file("  - Channel 0: MRI 图像")
        self.print_to_log_file("  - Channel 1: Step 1 预测")
        self.print_to_log_file("")
        self.print_to_log_file("解剖 Attention (利用 Step 1 预测):")
        self.print_to_log_file("  1. Disc Boundary: 椎间盘边界 (LDH 起源)")
        self.print_to_log_file("  2. Disc-Cord Interface: 椎间盘-脊髓接触区 (突出方向)")
        self.print_to_log_file("  3. Inter-vertebral: 椎骨之间区域")
        self.print_to_log_file("")
        self.print_to_log_file("Loss 组合:")
        self.print_to_log_file("  1. Attention Weighted Dice (weight=1.0)")
        self.print_to_log_file("  2. Weighted CE (weight=1.0, pos_weight=10)")
        self.print_to_log_file("  3. Focal Tversky (weight=0.5, α=0.3, β=0.7)")
        self.print_to_log_file("  4. Boundary Loss (weight=0.5)")
        self.print_to_log_file("  5. Anatomical Penalty (weight=2.0)")
        self.print_to_log_file("")
        self.print_to_log_file("其他优化: 85% 前景过采样")
        self.print_to_log_file("=" * 60)
        
        self.oversample_foreground_percent = 0.85
    
    def _build_loss(self):
        loss = LDH_MicroStructure_Loss(
            weight_dice=1.0,
            weight_ce=1.0,
            weight_focal_tversky=0.5,
            weight_boundary=0.5,
            tversky_alpha=0.3,
            tversky_beta=0.7,
            focal_gamma=0.75,
            ce_pos_weight=10.0,
            use_anatomical_attention=True,
            anatomical_penalty_weight=2.0,
            smooth=1e-5,
            ignore_label=self.label_manager.ignore_label,
        )
        
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (1.5 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            self.print_to_log_file(f"Deep Supervision 权重: {weights}")
            loss = DeepSupervisionWrapper(loss, weights)
        
        return loss
