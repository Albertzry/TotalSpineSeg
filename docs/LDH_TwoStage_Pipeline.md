# LDH Two-Stage Segmentation Pipeline

## 📋 Executive Summary

本文档描述了 TotalSpineSeg-v2 中全新的**两阶段腰椎间盘突出（LDH）分割流水线**，该流水线利用解剖学先验（椎间盘索引）提升小目标分割性能，并采用表面感知损失函数优化边界精度。

### 核心改进
- ✅ **两阶段架构**：Stage A (检测) + Stage B (精细分割)
- ✅ **解剖学先验**：基于 Step2 椎骨标注生成归一化椎间盘索引图
- ✅ **表面感知监督**：SDM (Signed Distance Map) + Boundary Loss
- ✅ **强制四类采样**：确保 Stage A 训练样本多样性
- ✅ **完全移除旧流程**：不再依赖 Step1 prior 的单阶段 nnUNet 训练

---

## 🎯 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Preparation (Dataset 105)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  STEP 0: Generate Disc Index from Step2 Output                  │
│  ├─ Run Step1 inference → Step2 inference                       │
│  ├─ Post-process to full labels (C1..L5, disc 63..100)          │
│  └─ Transform Step2 labels → normalized disc_index_map          │
│                                                                   │
│  STEP 1: Per-Disc Sample Extraction                             │
│  ├─ Iterate each disc level (L1/L2 .. L5/S1)                    │
│  ├─ Generate Stage A patches (96³) with 4-class sampling:       │
│  │   • LDH center (if positive)                                 │
│  │   • LDH boundary (if positive)                               │
│  │   • Disc boundary (negative, mandatory)                      │
│  │   • Disc interior (negative, mandatory)                      │
│  └─ Generate Stage B ROIs (48³, positive cases only)            │
│      • Centered at LDH centroid                                  │
│      • Include disc_mask, disc_index, LDH_mask, SDM             │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         Stage A Training                         │
├─────────────────────────────────────────────────────────────────┤
│  Task: Disc-level LDH Detection (binary classification)         │
│  Input: image (1) + disc_mask (1) + disc_index (1) → 3 channels│
│  Output: has_ldh (0 or 1)                                        │
│  Architecture: 3D CNN (5 conv blocks + GAP + FC)                │
│  Loss: Focal Loss (γ=2, prioritize recall)                      │
│  Training: ~50 epochs, AdamW, cosine annealing                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         Stage B Training                         │
├─────────────────────────────────────────────────────────────────┤
│  Task: ROI-based Fine Segmentation (positive cases only)        │
│  Input: image (1) + disc_mask (1) + disc_index (1) → 3 channels│
│  Output: LDH mask (1 channel)                                    │
│  Architecture: Small 3D U-Net (4 levels, 16→128 channels)       │
│  Loss: FocalTversky + Boundary + L1(SDM)                        │
│  Training: ~200 epochs, AdamW, ReduceLROnPlateau                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        Inference Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│  For each test case:                                             │
│  1. Run Step1 + Step2 inference → disc_index_map               │
│  2. For each disc level (L1/L2 .. L5/S1):                       │
│     a. Stage A: Predict has_ldh (detection)                     │
│     b. If has_ldh == 1:                                          │
│        → Stage B: Segment LDH in ROI                            │
│  3. Aggregate all disc-level predictions → final LDH mask       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📂 Directory Structure

```
TotalSpineSeg-v2/
├── totalspineseg/
│   ├── ldh_twostage/              # 两阶段LDH核心模块
│   │   ├── __init__.py
│   │   ├── disc_index.py          # 椎间盘索引生成
│   │   ├── distance_maps.py       # SDM计算
│   │   ├── sampling.py            # 四类强制采样
│   │   ├── losses.py              # Focal/Tversky/Boundary损失
│   │   ├── models.py              # StageA检测器 + StageB UNet
│   │   ├── data.py                # Dataset类
│   │   └── metrics.py             # 评估指标
│   └── utils/
│       └── predict_nnunet.py      # nnUNet推理封装（含monkeypatch）
│
├── scripts/
│   ├── prepare_dataset_105.py     # Dataset 105准备（两阶段专用）
│   ├── train_ldh_stage_a.py       # Stage A训练脚本
│   ├── train_ldh_stage_b.py       # Stage B训练脚本
│   ├── eval_ldh_twostage.py       # 两阶段评估脚本
│   └── train.sh                   # 统一训练入口（Dataset 105分支）
│
└── data/nnUNet/raw/
    └── Dataset105_TotalSpineSeg_LDH/
        ├── imagesTr/               # MRI原始图像（_0000.nii.gz）
        ├── labelsTr/               # LDH二值标签（0=bg, 1=LDH）
        ├── disc_index_maps/        # 椎间盘索引图（Step0输出）
        └── ldh_twostage/
            ├── stageA_patches/     # Stage A训练补丁（.npz）
            └── stageB_rois/        # Stage B训练ROI（.npz）
```

---

## 🛠️ Key Technical Components

### 1. Disc Index Map Generation (`disc_index.py`)

**目的**：将 Step2 的离散椎骨标签（C1=13, ..., L5=41, S1=50）转换为连续归一化索引（0-1），作为空间先验输入网络。

**实现**：
```python
from totalspineseg.ldh_twostage.disc_index import make_disc_index_map_from_step2_full_labels

# Step2 full labels: C1=13, C2=14, ..., L5=41, S1=50
# Disc labels: C2-C3=64, ..., L5-S1=100
disc_index_nii = make_disc_index_map_from_step2_full_labels(
    step2_seg_nii,  # Step2 full segmentation (Nifti1Image)
    spec=DiscIndexSpec.default_lumbar()  # L1/L2 → L5/S1 mapping
)
```

**输出**：
- 每个椎间盘区域内填充归一化索引：`[0, 0.2, 0.4, 0.6, 0.8, 1.0]` 对应 `[L1/L2, L2/L3, ..., L5/S1]`
- 非椎间盘区域：0

### 2. Mandatory 4-Class Sampling (`sampling.py`)

**目的**：确保 Stage A 训练补丁覆盖所有关键场景，避免模型过拟合单一样本类型。

**四类定义**：
| 类别                  | 中心点选择                     | 标签 | 必须性 |
|-----------------------|--------------------------------|------|--------|
| `ldh_center`          | LDH mask的质心                 | 1    | 正例时 |
| `ldh_boundary`        | LDH边界上随机点                | 1    | 正例时 |
| `disc_boundary_neg`   | 椎间盘边界上随机点（无LDH）    | 0    | 总是   |
| `disc_interior_neg`   | 椎间盘内部随机点（无LDH）      | 0    | 总是   |

**实现**：
```python
from totalspineseg.ldh_twostage.sampling import sample_four_class_centers

centers = sample_four_class_centers(
    disc_region,  # 椎间盘掩码（稍膨胀，7x7x7）
    ldh_in_disc,  # 该椎间盘内的LDH掩码
    rng=np.random.RandomState(seed)
)
# 返回：{'ldh_center': Point3D(...), 'disc_boundary_neg': Point3D(...), ...}
```

### 3. Signed Distance Map (SDM) (`distance_maps.py`)

**目的**：为 Stage B 提供表面距离监督，增强边界精度。

**定义**：
- 对于 mask 内部点：SDM = 到最近边界的**负距离**
- 对于 mask 外部点：SDM = 到最近边界的**正距离**

**实现**：
```python
from totalspineseg.ldh_twostage.distance_maps import signed_distance_map

sdm = signed_distance_map(ldh_mask)  # shape: same as ldh_mask
# sdm[mask == 1] < 0 (内部)
# sdm[mask == 0] > 0 (外部)
```

### 4. Stage B Combined Loss (`losses.py`)

**组合**：
```
Total Loss = w1 × FocalTversky + w2 × Boundary + w3 × L1(SDM)
```

**各项作用**：
- **FocalTversky**：解决类别不平衡，关注小目标（α=0.7, β=0.3, γ=1.33）
- **Boundary Loss**：基于距离变换的边界权重，直接优化边界IoU
- **L1(SDM)**：回归真实 SDM，强制网络学习表面几何

**权重配置**：
```python
stage_b_total_loss(
    logits, target, sdm_target,
    w_tversky=1.0,
    w_boundary=0.5,
    w_sdm=0.2
)
```

---

## 🚀 Usage Guide

### Step 1: 数据准备

```bash
# 确保环境变量已设置
export TOTALSPINESEG="/root/TotalSpineSeg-v2"
export TOTALSPINESEG_DATA="/opt/data/private/data_sum"
export TOTALSPINESEG_JOBS=12

# 准备 Dataset 105（约需1-3小时，取决于样本数）
cd $TOTALSPINESEG
python scripts/prepare_dataset_105.py \
    --stagea-patch 96 \
    --stageb-roi 48 \
    --device cuda
```

**输出检查**：
```bash
DATA_DIR="$TOTALSPINESEG_DATA/nnUNet/raw/Dataset105_TotalSpineSeg_LDH"
echo "StageA patches: $(ls $DATA_DIR/ldh_twostage/stageA_patches/*.npz | wc -l)"
echo "StageB ROIs: $(ls $DATA_DIR/ldh_twostage/stageB_rois/*.npz | wc -l)"
echo "Disc index maps: $(ls $DATA_DIR/disc_index_maps/*.nii.gz | wc -l)"
```

### Step 2: 训练

```bash
# 两阶段训练（自动执行 Stage A → Stage B → Evaluation）
bash scripts/train.sh 105 0

# 或分别运行：
# Stage A (检测)
python scripts/train_ldh_stage_a.py \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-3

# Stage B (精细分割)
python scripts/train_ldh_stage_b.py \
    --epochs 200 \
    --batch-size 16 \
    --lr 1e-4
```

**训练监控**：
```bash
# 查看 Stage A 训练日志
tail -f $TOTALSPINESEG_DATA/ldh_stage_a/training.log

# 查看 Stage B 训练日志
tail -f $TOTALSPINESEG_DATA/ldh_stage_b/training.log

# TensorBoard (如果启用)
tensorboard --logdir $TOTALSPINESEG_DATA/ldh_stage_b/runs
```

### Step 3: 评估

```bash
# 两阶段联合评估
python scripts/eval_ldh_twostage.py \
    --model-a $TOTALSPINESEG_DATA/ldh_stage_a/best_model.pth \
    --model-b $TOTALSPINESEG_DATA/ldh_stage_b/best_model.pth \
    --data-dir $TOTALSPINESEG_DATA/nnUNet/raw/Dataset105_TotalSpineSeg_LDH
```

**关键指标**：
- **Disc-level Recall**：椎间盘级别的检出率（Stage A 性能）
- **Lesion-wise Detection Rate**：病灶级检出率（≥1 voxel overlap）
- **Dice Score**：分割重叠度（Stage B 性能）
- **ASD (Average Surface Distance)**：平均表面距离（边界精度）

---

## 🐛 Troubleshooting

### 问题 1：`ModuleNotFoundError: No module named 'totalspineseg.ldh_twostage'`

**原因**：Python 找不到仓库模块。

**解决**：
```bash
# 方法1：修改 PYTHONPATH（推荐）
export PYTHONPATH="$TOTALSPINESEG:${PYTHONPATH:-}"

# 方法2：在脚本开头添加（已在所有脚本中实现）
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### 问题 2：`ImportError: cannot import name 'models' from 'totalspineseg'`

**原因**：`totalspineseg/__init__.py` 中引用了不存在的 `models` 包。

**解决**：已修复，确认 `totalspineseg/__init__.py` 中**不包含** `from . import models`。

### 问题 3：nnUNet 推理时报错 `ImportError: cannot import name 'LDH_MicroStructure_Loss'`

**原因**：conda 环境的 `site-packages/nnunetv2/training/nnUNetTrainer/` 中有遗留的旧 trainer 文件。

**解决**：已在 `totalspineseg/utils/predict_nnunet.py` 中实现 **monkeypatch**，自动屏蔽问题模块：
```python
# 运行时自动生效，无需手动干预
_bad_trainer_modules = [
    'nnunetv2.training.nnUNetTrainer.nnUNetTrainer_LDH_Step6',
    # 如有新的问题模块，在此添加
]
```

### 问题 4：数据准备卡在 `[4/5] Building Step2 input` 且无进度显示

**原因**：3000+ 样本的并行处理需要时间，之前 `quiet=True` 隐藏了进度条。

**解决**：已全部改为 `quiet=False`，现在会显示 `tqdm` 进度条：
```
  - Cropping images to segmentation regions...
100%|████████████████████| 3627/3627 [05:23<00:00, 11.23it/s]
```

**监控进程**：
```bash
# 检查 Python 进程是否运行
ps aux | grep prepare_dataset_105

# 监控资源使用
watch -n 1 'free -h && nvidia-smi'
```

### 问题 5：Step1 labeling 失败（`Some label must be in the segmentation`）

**原因**：部分病例 Step1 分割质量差，无法提取有效的椎间盘/椎管标签。

**处理**：
- ✅ 脚本会**自动跳过**这些病例，不影响其他样本
- ✅ 控制台显示：`⚠ Step1 labeling failed or incomplete for X cases`
- ✅ 这些病例**不会**进入 Step2 推理和后续训练

**确认**：
```bash
# 查看有效病例数
ls $TOTALSPINESEG_DATA/nnUNet/raw/Dataset105_TotalSpineSeg_LDH/step2_input/*_0000.nii.gz | wc -l
```

---

## 📊 Performance Expectations

### 数据规模（基于典型 LDH 数据集）
| 阶段                | 样本数估算                     | 存储需求    |
|---------------------|--------------------------------|-------------|
| 原始 LDH 病例       | ~400 cases                     | ~20 GB      |
| Stage A patches     | ~8,000 patches (4/disc × 5 discs × 400 cases) | ~15 GB |
| Stage B ROIs        | ~1,200 ROIs (positive discs only) | ~3 GB  |
| 模型权重            | Stage A + Stage B              | ~200 MB     |

### 训练时间（单卡 V100/A100）
- **Stage A**：~2-4 小时（50 epochs, batch=32）
- **Stage B**：~8-12 小时（200 epochs, batch=16）

### 推理时间
- **单病例**：~15-30 秒（Step1 + Step2 + 两阶段 LDH）
- **批量（100 cases）**：~30-45 分钟

### 性能基准（需在实际数据上验证）
| 指标                        | 目标值     | 单阶段基线 |
|-----------------------------|------------|------------|
| Disc-level Recall (Stage A) | ≥95%       | N/A        |
| Lesion Detection Rate       | ≥90%       | ~85%       |
| Dice Score                  | ≥0.75      | ~0.68      |
| ASD (mm)                    | ≤2.0       | ~2.5       |

---

## 🔬 Design Rationale

### 为什么选择两阶段？
1. **解耦检测与分割**：Stage A 优化召回率，Stage B 优化精度，各司其职
2. **减少假阳性**：Stage A 筛选可疑椎间盘，避免在无关区域浪费计算
3. **数据效率**：Stage B 只在正例 ROI 上训练，缓解样本不平衡

### 为什么需要 Disc Index？
- **空间先验**：LDH 在 L4/L5 和 L5/S1 最常见，disc_index 提供位置信息
- **硬约束**：相比 Step1 的概率性 prior，Step2 椎骨编号是确定性的，更可靠

### 为什么用 SDM？
- **边界敏感**：纯 Dice Loss 对小位移不敏感，SDM 提供连续梯度信号
- **距离回归**：网络学习到"距离边界多远"，而非简单的二值分类

### 为什么强制四类采样？
- **类别平衡**：LDH 通常只占椎间盘体积的 5-10%，随机采样会过拟合负样本
- **边界学习**：显式采样边界点，强制网络学习判别边界特征

---

## 📝 Code Maintenance Notes

### 已移除的遗留代码
以下文件/逻辑已从仓库中**完全删除**，不应再使用：
- ❌ `totalspineseg/nnunet_extensions/nnUNetTrainer_LDH.py`（单阶段 nnUNet trainer）
- ❌ `scripts/prepare_dataset_105.py` 中的 `--legacy` 参数
- ❌ `scripts/train.sh` 中的 `TOTALSPINESEG_LDH_TWOSTAGE` 环境变量分支

### 关键约束
1. **Disc labels 范围**：L1/L2=91, L2/L3=92, ..., L5/S1=100（共5个）
2. **Patch size 限制**：Stage A 至少 96³，Stage B 至少 48³（确保覆盖典型 LDH）
3. **nnUNet 版本**：需要 ≥2.0，与 Step1/Step2 训练一致

### 扩展指南
#### 添加新损失函数：
1. 在 `totalspineseg/ldh_twostage/losses.py` 中定义新损失
2. 在 `scripts/train_ldh_stage_b.py` 中集成到 `stage_b_total_loss`
3. 更新 `--loss-weights` 命令行参数

#### 支持胸椎 LDH：
1. 修改 `DiscIndexSpec` 添加 T1/T2 - T12/L1 映射
2. 在 `prepare_dataset_105.py` 中扩展 `disc_labels` 列表
3. 重新训练 Stage A/B（无需修改网络结构）

---

## 📚 References

### Key Papers
1. **Focal Loss**: Lin et al. "Focal Loss for Dense Object Detection." ICCV 2017.
2. **Tversky Loss**: Salehi et al. "Tversky Loss Function for Image Segmentation." MLMI 2017.
3. **Boundary Loss**: Kervadec et al. "Boundary Loss for Highly Unbalanced Segmentation." MIDL 2019.
4. **Distance Maps**: Park et al. "Learning to Segment Medical Images with Scribble Supervision Alone." MICCAI 2019.

### Related Work
- **nnU-Net**: Isensee et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." Nature Methods 2021.
- **TotalSegmentator**: Wasserthal et al. "TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images." Radiology: AI 2023.

---

## 📧 Contact & Support

**Repository**: [TotalSpineSeg-v2](https://github.com/neuropoly/totalspineseg)  
**Issues**: GitHub Issues tab  
**Documentation**: `docs/` folder in repository

**维护者**：TotalSpineSeg Development Team  
**最后更新**：2025-12-13

---

## ✅ Quick Checklist

使用本流水线前，请确认以下条件：

- [ ] nnUNet 环境已安装（conda env: `tss2`）
- [ ] Step1 模型已训练（Dataset 101）
- [ ] Step2 模型已训练（Dataset 102）
- [ ] Dataset 100（augmented）已准备
- [ ] 环境变量已设置（`TOTALSPINESEG`, `TOTALSPINESEG_DATA`, `TOTALSPINESEG_JOBS`）
- [ ] GPU 可用且显存 ≥16GB（推荐 24GB+）
- [ ] 磁盘空间充足（至少 100GB 可用）

开始训练：
```bash
# 一键完成所有步骤
bash scripts/train.sh 105 0
```

Good luck! 🚀

