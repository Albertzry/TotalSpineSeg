# Step 5: LDH 微小结构分割优化方案

## 概述

Step 5 是专门针对 **LDH (腰椎间盘突出)** 这一微小结构设计的训练流程。由于 LDH 体积小、类别不平衡严重，需要特殊的优化策略。

## 设计思路

### 问题分析

1. **LDH 是微小结构**：在整个图像中占比极小，导致严重的类别不平衡
2. **LDH 有特定的解剖位置**：只会出现在椎间盘、椎骨和脊髓之间
3. **普通样本可能存在未标注的 LDH**：在 Dataset 102 中训练时可能产生负面影响

### 解决方案

创建 **Dataset 105**，只包含 LDH 样本：
- 网络输出：**只有 LDH**（二分类：0=background, 1=LDH）
- 输入通道 0：MRI 图像
- 输入通道 1：Step 1 预测结果（提供解剖上下文）

---

## 数据流程

```
Dataset 100 (增强后的数据)
       ↓
提取 sub-LDH* 样本
       ↓
对 LDH 图像运行 Step 1 推理
       ↓
获得椎间盘、椎骨、椎管、脊髓的预测
       ↓
创建 Dataset 105:
  - 标签: 0=background, 1=LDH (来自 GT 的 label 101)
  - Channel 0: MRI 图像
  - Channel 1: Step 1 预测 (解剖上下文)
       ↓
使用 nnUNetTrainer_LDH 训练
```

---

## 多结构解剖 Attention 机制

### LDH 的解剖特点

```
                  ┌── 椎骨 (6-7) ──┐
                  │                │
     LDH 起源 ─→ 椎间盘边界 (1-5) │
                      │            │
                      ▼            │
                向后突出 ────────→ 脊髓 (9) / 椎管 (8)
                  │                │
                  └── 椎骨 (6-7) ──┘
```

### Step 1 预测标签

| 标签值 | 结构 |
|--------|------|
| 0 | background |
| 1 | disc (普通椎间盘) |
| 2 | disc_C2_C3 |
| 3 | disc_C7_T1 |
| 4 | disc_T12_L1 |
| 5 | disc_L5_S |
| 6 | vertebrae (普通椎骨) |
| 7 | vertebrae_C1 |
| 8 | canal (椎管) |
| 9 | cord (脊髓) |

### 三层 Attention 组成

1. **Disc Boundary Attention (权重 0.3)**
   - 计算：椎间盘膨胀 - 椎间盘腐蚀 = 边界区域
   - 作用：LDH 起源于椎间盘边界

2. **Disc-Cord Interface Attention (权重 0.5)** ⭐ 最重要
   - 计算：椎间盘膨胀 ∩ 脊髓/椎管膨胀 = 接触区域
   - 作用：LDH 向后突出压迫脊髓的位置

3. **Inter-vertebral Attention (权重 0.2)**
   - 计算：椎骨膨胀 - 椎骨本身 = 椎骨间区域
   - 作用：LDH 只发生在椎骨之间

### 综合 Attention Map

```python
attention = 0.5 * disc_cord_interface 
          + 0.3 * disc_boundary 
          + 0.2 * inter_vertebral
```

---

## Loss 函数设计

### 组合 Loss

```
Total Loss = Attention_Dice + CE + 0.5*FocalTversky + 0.5*Boundary + Anatomical_Penalty
```

### 各 Loss 说明

| Loss | 权重 | 参数 | 作用 |
|------|------|------|------|
| **Attention Weighted Dice** | 1.0 | - | 在解剖相关区域计算 Dice |
| **Weighted CE** | 1.0 | pos_weight=10 | LDH 正类 10 倍权重 |
| **Focal Tversky** | 0.5 | α=0.3, β=0.7, γ=0.75 | 高 recall，小目标优化 |
| **Boundary Loss** | 0.5 | 距离变换 | 边界精度 |
| **Anatomical Penalty** | 2.0 | - | 惩罚解剖不合理区域的预测 |

### Focal Tversky Loss 详解

专门为小目标设计 (Abraham & Khan 2019)：

```
Tversky Index (TI) = TP / (TP + α*FP + β*FN)
Focal Tversky Loss = (1 - TI)^γ
```

- **α=0.3, β=0.7**：对 FN (漏检) 给更高惩罚 → 提高 recall
- **γ=0.75**：关注困难样本

### Boundary Loss 详解

基于距离变换 (Kervadec et al. 2019)：

```
L_boundary = mean(pred * dist_map)
dist_map: 内部为负, 外部为正
```

---

## 其他优化策略

1. **85% 前景过采样**：确保更多 patch 包含 LDH
2. **Deep Supervision 浅层优先**：高分辨率层获得更高权重
3. **解剖惩罚**：在低 attention 区域预测 LDH 会受到惩罚

---

## 文件结构

```
/root/TotalSpineSeg-v2/
├── totalspineseg/
│   ├── nnunet_extensions/
│   │   ├── __init__.py
│   │   └── nnUNetTrainer_LDH.py      # LDH 专用 trainer
│   └── resources/
│       ├── datasets/
│       │   └── dataset_step5_ldh.json # Dataset 105 配置
│       └── labels_maps/
│           └── nnunet_step5_ldh.json  # 标签映射
├── scripts/
│   ├── prepare_dataset_105.py         # 数据准备脚本
│   └── train.sh                       # 训练脚本 (已支持 105)
└── docs/
    ├── Step5_LDH_Optimization.md      # 本文档
    └── Step5_Commands.md              # 执行命令
```

---

## 参考文献

1. Abraham, N., & Khan, N. M. (2019). A novel focal tversky loss function with improved attention u-net for lesion segmentation. ISBI 2019.
2. Kervadec, H., et al. (2019). Boundary loss for highly unbalanced segmentation. MIDL 2019.
3. Wong, K. C., et al. (2018). 3D segmentation with exponential logarithmic loss for highly unbalanced object sizes. MICCAI 2018.

