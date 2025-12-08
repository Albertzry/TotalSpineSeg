# Step 5 LDH 模型评估指南

## 概述

`evaluate_step5.py` 脚本用于评估 Step 5 (Dataset 105) LDH 分割模型在测试集上的表现。

## 功能特性

### 评估指标

脚本会计算以下指标：

1. **Dice Score**: 分割重叠度（主要指标）
2. **IoU (Intersection over Union)**: 交并比
3. **Precision**: 精确率（减少假阳性）
4. **Recall**: 召回率（减少假阴性）
5. **F1 Score**: Precision 和 Recall 的调和平均
6. **Hausdorff Distance (95th percentile)**: 边界精度（对微小结构很重要）
7. **Average Surface Distance (ASD)**: 平均表面距离
8. **Volume Metrics**: 
   - 预测体积 vs 真实体积
   - 体积误差（体素和 mm³）

### 输出文件

评估完成后会生成以下文件：

1. **详细结果 CSV**: `step5_evaluation_fold{fold}_detailed.csv`
   - 每个测试案例的详细指标

2. **汇总统计 CSV**: `step5_evaluation_fold{fold}_summary.csv`
   - 所有指标的统计摘要（mean, std, min, max, quartiles）

3. **汇总 JSON**: `step5_evaluation_fold{fold}_summary.json`
   - 机器可读的汇总结果

## 使用方法

### 基本用法

```bash
# 评估 fold 0（默认，使用 checkpoint_best.pth）
python scripts/evaluate_step5.py

# 使用 checkpoint_best.pth（训练中评估）
python scripts/evaluate_step5.py --checkpoint checkpoint_best.pth

# 使用 checkpoint_final.pth（训练完成后评估）
python scripts/evaluate_step5.py --checkpoint checkpoint_final.pth

# 评估指定 fold
python scripts/evaluate_step5.py --fold 0

# 如果预测已完成，跳过预测步骤
python scripts/evaluate_step5.py --skip-prediction
```

### 完整参数

```bash
python scripts/evaluate_step5.py \
    --fold 0 \
    --trainer nnUNetTrainer_LDH \
    --configuration 3d_fullres \
    --plans nnUNetPlans_small \
    --device cuda \
    --checkpoint checkpoint_best.pth \
    --output-dir results/step5_evaluation \
    --skip-prediction
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--fold` | 0 | 要评估的 fold 编号 |
| `--trainer` | `nnUNetTrainer_LDH` | Trainer 名称 |
| `--configuration` | `3d_fullres` | nnUNet 配置 |
| `--plans` | `nnUNetPlans_small` | Plans 名称 |
| `--device` | `cuda` | 设备（cuda/cpu） |
| `--checkpoint` | `checkpoint_best.pth` | Checkpoint 名称（`checkpoint_best.pth` 或 `checkpoint_final.pth`） |
| `--output-dir` | `results/step5_evaluation` | 评估结果输出目录 |
| `--skip-prediction` | False | 如果预测已完成，跳过预测步骤 |
| `--dataset-id` | 105 | 数据集 ID |

### Checkpoint 选择

- **`checkpoint_best.pth`** (默认): 训练过程中验证集上表现最好的模型
  - 适合在训练过程中评估
  - 通常比 `checkpoint_final.pth` 表现更好
  
- **`checkpoint_final.pth`**: 训练结束时的最终模型
  - 需要等待训练完全结束
  - 可能不是最优模型

**注意**: 使用不同的 checkpoint 时，预测结果会保存在不同的目录中，避免冲突。

## 工作流程

1. **检查预测结果**
   - 如果预测结果已存在，跳过预测步骤
   - 否则，在测试集上运行推理

2. **加载预测和真实标签**
   - 从 `nnUNet_results/Dataset105_.../fold_{fold}/test/` 加载预测
   - 从 `nnUNet_raw/Dataset105_.../labelsTs/` 加载真实标签

3. **计算指标**
   - 对每个测试案例计算所有指标
   - 处理缺失或失败的案例

4. **生成报告**
   - 计算汇总统计
   - 保存详细结果和汇总报告

## 输出示例

```
================================================================================
Step 5 LDH Model Evaluation
================================================================================
Dataset: Dataset105_TotalSpineSeg_LDH
Fold: 0
Trainer: nnUNetTrainer_LDH
Configuration: 3d_fullres
Plans: nnUNetPlans_small
================================================================================
✓ Predictions already exist in /path/to/predictions

Found 50 test cases
Evaluating...
  ✓ sub-LDH1_20120911_T2w_a0: Dice=0.7234, IoU=0.6123, Precision=0.7891, Recall=0.6789
  ✓ sub-LDH2_20120911_T2w_a1: Dice=0.8123, IoU=0.7012, Precision=0.8456, Recall=0.7823
  ...

================================================================================
Evaluation Summary
================================================================================
Total cases evaluated: 50

Key Metrics (Mean ± Std):
  Dice Score:        0.7523 ± 0.0891
  IoU:               0.6234 ± 0.1023
  Precision:         0.8123 ± 0.0789
  Recall:            0.7234 ± 0.0912
  F1 Score:          0.7654 ± 0.0845
  Hausdorff Distance (95%): 2.34 ± 1.23 mm
  Average Surface Distance: 1.12 ± 0.67 mm
================================================================================

✓ Detailed results saved to: results/step5_evaluation/step5_evaluation_fold0_best_detailed.csv
✓ Summary statistics saved to: results/step5_evaluation/step5_evaluation_fold0_best_summary.csv
✓ Summary JSON saved to: results/step5_evaluation/step5_evaluation_fold0_best_summary.json
```

## 环境要求

### Python 包

- `numpy`
- `nibabel`
- `scipy`
- `pandas`

### 环境变量

确保以下环境变量已设置：

```bash
export TOTALSPINESEG=/root/TotalSpineSeg-v2
export TOTALSPINESEG_DATA=/opt/data/private/data_sum
export nnUNet_raw=/opt/data/private/data_sum/nnUNet/raw
export nnUNet_preprocessed=/opt/data/private/data_sum/nnUNet/preprocessed
export nnUNet_results=/opt/data/private/data_sum/nnUNet/results
```

## 注意事项

1. **预测结果位置**: 脚本会自动在 `nnUNet_results` 目录下查找预测结果
2. **测试集标签**: 确保 `labelsTs` 目录存在且包含测试标签
3. **GPU 使用**: 如果使用 `--device cuda`，确保 GPU 可用
4. **内存使用**: 评估过程会加载所有测试案例，确保有足够内存

## 故障排除

### 预测结果未找到

如果遇到 "Prediction not found" 警告：

1. 检查预测是否已完成：
   ```bash
   ls /opt/data/private/data_sum/nnUNet/results/Dataset105_*/nnUNetTrainer_LDH*/fold_0/test/
   ```

2. 如果没有预测结果，运行评估脚本（不带 `--skip-prediction`）会自动运行预测

### 形状不匹配

如果遇到形状不匹配警告：

- 脚本会自动裁剪到最小公共形状
- 如果问题持续，检查数据预处理是否正确

### 内存不足

如果遇到内存问题：

- 考虑分批处理测试案例
- 或者使用 `--skip-prediction` 先单独运行预测

## 与 nnUNet 官方评估的对比

nnUNet 的 `nnUNetv2_evaluate_folder` 命令也会生成评估结果，但本脚本提供：

1. **更详细的指标**: 包括 Hausdorff Distance、ASD、体积指标
2. **更好的可读性**: CSV 格式便于分析
3. **针对 LDH 优化**: 专门针对微小结构的评估指标
4. **自动化流程**: 自动运行预测（如果需要）

## 下一步

评估完成后，可以：

1. 分析详细结果 CSV，找出表现差的案例
2. 对比不同 fold 的结果
3. 使用结果优化模型参数
4. 生成可视化报告

