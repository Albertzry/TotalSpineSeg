# LDH 两阶段流水线快速参考

## 🎯 核心概念

### 流程概览
```
原始MRI → Step1/Step2推理 → 椎间盘索引图 → Stage A检测 → Stage B精分割 → 最终LDH掩码
```

### 三个关键先验
1. **Disc Index Map**（椎间盘索引）：归一化位置编码 [0-1]，区分 L1/L2 到 L5/S1
2. **Disc Mask**（椎间盘掩码）：来自 Step2 全标签，标识目标椎间盘区域
3. **Anatomical Context**（解剖学上下文）：周围脊柱结构作为空间参考

### 两个训练阶段
| 阶段     | 任务           | 输入尺寸 | 输出       | 损失函数              |
|----------|----------------|----------|------------|-----------------------|
| Stage A  | 椎间盘级检测   | 96³      | 0/1 (二分类) | Focal Loss (γ=2)      |
| Stage B  | ROI精细分割    | 48³      | LDH掩码    | Tversky+Boundary+SDM  |

---

## 🚀 快速开始

### 1. 环境设置
```bash
conda activate tss2
export TOTALSPINESEG="/root/TotalSpineSeg-v2"
export TOTALSPINESEG_DATA="/opt/data/private/data_sum"
export TOTALSPINESEG_JOBS=12
```

### 2. 数据准备（首次运行）
```bash
cd $TOTALSPINESEG
python scripts/prepare_dataset_105.py
```

**预期耗时**：1-3小时（取决于样本数）  
**输出检查**：
```bash
tree $TOTALSPINESEG_DATA/nnUNet/raw/Dataset105_TotalSpineSeg_LDH/ldh_twostage/
# 应该看到：
# ├── stageA_patches/  (~8000个 .npz 文件)
# └── stageB_rois/     (~1200个 .npz 文件)
```

### 3. 训练（一键执行）
```bash
bash scripts/train.sh 105 0
```

这会自动依次执行：
1. **Stage A 训练**（~2-4小时）
2. **Stage B 训练**（~8-12小时）
3. **联合评估**（~30分钟）

### 4. 单独运行各阶段（可选）
```bash
# 只训练 Stage A
python scripts/train_ldh_stage_a.py --epochs 50 --batch-size 32

# 只训练 Stage B
python scripts/train_ldh_stage_b.py --epochs 200 --batch-size 16

# 只评估
python scripts/eval_ldh_twostage.py \
    --model-a $TOTALSPINESEG_DATA/ldh_stage_a/best_model.pth \
    --model-b $TOTALSPINESEG_DATA/ldh_stage_b/best_model.pth
```

---

## 📊 监控训练

### 实时查看日志
```bash
# Stage A
tail -f $TOTALSPINESEG_DATA/ldh_stage_a/training.log

# Stage B
tail -f $TOTALSPINESEG_DATA/ldh_stage_b/training.log
```

### 检查 GPU 使用
```bash
watch -n 1 nvidia-smi
```

### 查看磁盘空间
```bash
df -h $TOTALSPINESEG_DATA
```

---

## 🔍 数据格式说明

### Stage A Patch (.npz)
```python
data = np.load("patch.npz")
# 包含字段：
# - image: (96,96,96) float32, MRI图像
# - disc_mask: (96,96,96) float32, 椎间盘掩码
# - disc_index: (96,96,96) float32, 归一化索引 [0-1]
# - ldh_mask: (96,96,96) float32, LDH真值掩码
# - has_ldh: int8, 0或1（该椎间盘是否有LDH）
# - disc_label: int16, 椎间盘编号（91-100）
# - sample_id: bytes, 病例ID
# - patch_type: bytes, 补丁类型（4类之一）
```

### Stage B ROI (.npz)
```python
data = np.load("roi.npz")
# 包含字段（前6项同Stage A）：
# - image, disc_mask, disc_index, ldh_mask, has_ldh, disc_label
# + sdm: (48,48,48) float32, 有符号距离图（内部<0，外部>0）
# + sample_id, patch_type
```

### 四类强制采样
| 类型                | 说明                     | 标签 | 数量/椎间盘 |
|---------------------|--------------------------|------|-------------|
| `ldh_center`        | LDH质心                  | 1    | 1（正例时）|
| `ldh_boundary`      | LDH边界随机点            | 1    | 1（正例时）|
| `disc_boundary_neg` | 椎间盘边界随机点（负例） | 0    | 1（总是）  |
| `disc_interior_neg` | 椎间盘内部随机点（负例） | 0    | 1（总是）  |

---

## 🛠️ 常见问题

### Q1: 报错 `ModuleNotFoundError: No module named 'totalspineseg.ldh_twostage'`
**A**: 确保 `PYTHONPATH` 包含仓库根目录：
```bash
export PYTHONPATH="$TOTALSPINESEG:${PYTHONPATH:-}"
```

### Q2: 数据准备卡住，没有进度显示
**A**: 已修复（v2.1+），现在所有步骤都有进度条。如果仍无显示：
```bash
# 检查进程是否还在运行
ps aux | grep prepare_dataset_105

# 杀掉并重新运行
pkill -f prepare_dataset_105
python scripts/prepare_dataset_105.py
```

### Q3: Step1 labeling 失败的病例怎么处理？
**A**: 脚本会**自动跳过**无效病例，不影响其他样本。查看跳过数量：
```bash
# 控制台会显示：
# ⚠ Step1 labeling failed or incomplete for X cases; they will be skipped
```

### Q4: 训练时 GPU OOM（显存溢出）
**A**: 减小 batch size：
```bash
# Stage A（默认32，可减到16/8）
python scripts/train_ldh_stage_a.py --batch-size 16

# Stage B（默认16，可减到8/4）
python scripts/train_ldh_stage_b.py --batch-size 8
```

### Q5: 如何恢复中断的训练？
**A**: 脚本会自动保存 checkpoint，重新运行即可续训：
```bash
# Stage A 自动加载最新 checkpoint（如果存在）
python scripts/train_ldh_stage_a.py --resume

# Stage B 同理
python scripts/train_ldh_stage_b.py --resume
```

---

## 📈 评估指标解读

### Disc-level Recall（椎间盘级召回率）
- **定义**：正确检测到 LDH 的椎间盘数 / 真实有 LDH 的椎间盘总数
- **目标**：≥95%（优先保证不漏检）
- **Stage**：反映 Stage A 性能

### Lesion Detection Rate（病灶检出率）
- **定义**：检测到的 LDH 病灶数 / 真实 LDH 病灶总数（≥1 voxel overlap）
- **目标**：≥90%
- **Stage**：反映 Stage A + Stage B 联合性能

### Dice Score（分割重叠度）
- **定义**：2 × |预测 ∩ 真值| / (|预测| + |真值|)
- **目标**：≥0.75
- **Stage**：反映 Stage B 分割精度

### ASD (Average Surface Distance)（平均表面距离）
- **定义**：预测边界到真值边界的平均距离（mm）
- **目标**：≤2.0 mm
- **Stage**：反映 Stage B 边界精度

---

## 🎓 高级用法

### 调整损失函数权重
```bash
# Stage B 默认权重：Tversky=1.0, Boundary=0.5, SDM=0.2
python scripts/train_ldh_stage_b.py \
    --w-tversky 1.0 \
    --w-boundary 0.8 \  # 增加边界权重
    --w-sdm 0.3          # 增加表面监督
```

### 修改网络结构
编辑 `totalspineseg/ldh_twostage/models.py`：
```python
# Stage A 检测器（5层卷积 + 全连接）
class StageADetector(nn.Module):
    def __init__(self, in_channels=3, base_channels=16):  # 可调整 base_channels
        ...

# Stage B U-Net（4层，16→128通道）
class SmallUNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=16):  # 可调整 init_features
        ...
```

### 自定义采样策略
编辑 `totalspineseg/ldh_twostage/sampling.py`：
```python
def sample_four_class_centers(disc_region, ldh_in_disc, rng, n_boundary=3):
    # 增加边界采样点数量（默认1，可改成3）
    ...
```

### 导出推理结果
```python
# 在 eval_ldh_twostage.py 中添加保存逻辑
import nibabel as nib

# 保存预测掩码
pred_nii = nib.Nifti1Image(pred_mask.astype(np.uint8), affine, header)
nib.save(pred_nii, f"predictions/{case_id}_pred.nii.gz")

# 保存 Stage A 检测结果（JSON）
import json
with open(f"predictions/{case_id}_detections.json", "w") as f:
    json.dump({"L1_L2": 0, "L2_L3": 1, ..., "L5_S1": 1}, f)
```

---

## 📁 重要路径速查

| 路径                                              | 用途                     |
|---------------------------------------------------|--------------------------|
| `$TOTALSPINESEG/totalspineseg/ldh_twostage/`      | 核心代码模块             |
| `$TOTALSPINESEG/scripts/train_ldh_stage_*.py`     | 训练脚本                 |
| `$TOTALSPINESEG_DATA/nnUNet/raw/Dataset105_*/`    | 数据集根目录             |
| `.../ldh_twostage/stageA_patches/`                | Stage A 训练数据         |
| `.../ldh_twostage/stageB_rois/`                   | Stage B 训练数据         |
| `.../disc_index_maps/`                            | 椎间盘索引图             |
| `$TOTALSPINESEG_DATA/ldh_stage_a/`                | Stage A 输出（模型/日志）|
| `$TOTALSPINESEG_DATA/ldh_stage_b/`                | Stage B 输出（模型/日志）|

---

## 🔗 相关资源

- **完整文档**：`docs/LDH_TwoStage_Pipeline.md`
- **代码仓库**：https://github.com/neuropoly/totalspineseg
- **Issue 追踪**：GitHub Issues

---

## ✅ 每日检查清单

### 训练前
- [ ] GPU 可用（`nvidia-smi`）
- [ ] 磁盘空间充足（`df -h`）
- [ ] 数据准备完成（检查 patch 数量）
- [ ] 日志目录可写

### 训练中
- [ ] 监控 GPU 使用率（应 >80%）
- [ ] 查看训练日志（loss 是否下降）
- [ ] 检查磁盘空间（checkpoint 会占空间）

### 训练后
- [ ] 查看评估指标（Dice, ASD, Recall）
- [ ] 保存最佳模型（`best_model.pth`）
- [ ] 可视化预测结果（抽查几个 case）
- [ ] 备份重要文件

---

**最后更新**：2025-12-13  
**版本**：v2.1

祝训练顺利！🎉

