"""
nnU-Net Extensions for TotalSpineSeg

针对微小结构 LDH 优化的 trainer，利用多结构解剖 Attention.

解剖 Attention 机制 (利用 Step 1 预测):
1. Disc Boundary: 椎间盘边界区域 (LDH 起源位置)
2. Disc-Cord Interface: 椎间盘-脊髓/椎管接触区域 (LDH 突出方向)
3. Inter-vertebral: 椎骨之间区域 (LDH 发生位置)

Loss 组合:
- Attention Weighted Dice + Weighted CE (基础)
- Focal Tversky Loss (小目标优化)
- Boundary Loss (边界精度)
- Anatomical Penalty (解剖区域外惩罚)

Usage:
    1. Copy trainer to nnUNet:
       cp totalspineseg/nnunet_extensions/nnUNetTrainer_LDH.py \\
          /path/to/nnunetv2/training/nnUNetTrainer/
    
    2. Train:
       nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainer_LDH
"""

from totalspineseg.nnunet_extensions.nnUNetTrainer_LDH import (
    nnUNetTrainer_LDH,
    LDH_MicroStructure_Loss,
    AnatomicalAttention,
)

__all__ = [
    'nnUNetTrainer_LDH',
    'LDH_MicroStructure_Loss',
    'AnatomicalAttention',
]

