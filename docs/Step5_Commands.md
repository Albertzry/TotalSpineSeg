# Step 5: LDH 训练执行命令

## 环境信息

- **Conda 环境**: `tss2`
- **数据路径**: `/opt/data/private/data_sum/`
- **代码路径**: `/root/TotalSpineSeg-v2`

## 前置条件

已准备的数据集：
- ✅ Dataset 99 (原始数据)
- ✅ Dataset 100 (增强数据)
- ✅ Dataset 101 (Step 1 训练数据)
- ✅ Dataset 102 (Step 2 训练数据)

---

## 执行步骤

### 1. 激活环境

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate tss2
```

### 2. 设置环境变量

```bash
export TOTALSPINESEG=/root/TotalSpineSeg-v2
export TOTALSPINESEG_DATA=/opt/data/private/data_sum/
```

### 3. 确保 Trainer 已安装到 nnUNet

```bash
# 复制 trainer 到 nnUNet 安装目录
cp $TOTALSPINESEG/totalspineseg/nnunet_extensions/nnUNetTrainer_LDH.py \
   /opt/conda/envs/tss2/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/

# 验证
ls -la /opt/conda/envs/tss2/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_LDH.py
```

### 4. 准备 Dataset 105

```bash
cd $TOTALSPINESEG

# 运行数据准备脚本
# 这会：
#   1. 从 Dataset 100 提取 sub-LDH* 样本
#   2. 运行 Step 1 推理获得解剖结构预测
#   3. 创建二值 LDH 标签
#   4. 将 Step 1 预测作为第二输入通道
python scripts/prepare_dataset_105.py
```

**可选参数：**
```bash
# 跳过 Step 1 推理（如果已经有预测结果）
python scripts/prepare_dataset_105.py --skip-inference

# 调整测试集比例（默认 10%）
python scripts/prepare_dataset_105.py --test-ratio 0.15
```

### 5. 训练 Step 5

```bash
# 使用 GPU 1 训练
CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh 105

# 或者指定 fold
CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh 105 0
```

---

## 完整命令（一键执行）

```bash
# 复制以下命令到终端执行

source /opt/conda/etc/profile.d/conda.sh
conda activate tss2

export TOTALSPINESEG=/root/TotalSpineSeg-v2
export TOTALSPINESEG_DATA=/opt/data/private/data_sum/

# 确保 trainer 已安装
cp $TOTALSPINESEG/totalspineseg/nnunet_extensions/nnUNetTrainer_LDH.py \
   /opt/conda/envs/tss2/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/

cd $TOTALSPINESEG

# 准备数据
python scripts/prepare_dataset_105.py

# 训练
CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh 105
```

---

## 检查数据准备结果

```bash
# 检查 Dataset 105 是否创建成功
ls -la $TOTALSPINESEG_DATA/nnUNet/raw/Dataset105_TotalSpineSeg_LDH/

# 检查训练样本数量
ls $TOTALSPINESEG_DATA/nnUNet/raw/Dataset105_TotalSpineSeg_LDH/labelsTr/ | wc -l

# 检查测试样本数量
ls $TOTALSPINESEG_DATA/nnUNet/raw/Dataset105_TotalSpineSeg_LDH/labelsTs/ | wc -l

# 查看 dataset.json
cat $TOTALSPINESEG_DATA/nnUNet/raw/Dataset105_TotalSpineSeg_LDH/dataset.json
```

---

## 监控训练

```bash
# 查看训练日志
tail -f $TOTALSPINESEG_DATA/nnUNet/results/Dataset105_TotalSpineSeg_LDH/nnUNetTrainer_LDH__nnUNetPlans_small__3d_fullres/fold_0/training_log*.txt

# 使用 nvidia-smi 监控 GPU
watch -n 1 nvidia-smi
```

---

## 训练完成后

### 查看结果

```bash
# 查看测试结果
cat $TOTALSPINESEG_DATA/nnUNet/results/Dataset105_TotalSpineSeg_LDH/nnUNetTrainer_LDH__nnUNetPlans_small__3d_fullres/fold_0/test/summary.json
```

### 导出模型

```bash
# 模型会自动导出到
ls $TOTALSPINESEG_DATA/nnUNet/exports/Dataset105_*.zip
```

---

## 故障排除

### 问题 1: 找不到 LDH 样本

```bash
# 检查 Dataset 100 中是否有 LDH 样本
ls $TOTALSPINESEG_DATA/nnUNet/raw/Dataset100_TotalSpineSeg_Aug/imagesTr/sub-LDH* | head
```

### 问题 2: Step 1 模型不存在

```bash
# 检查 Step 1 模型是否训练完成
ls $TOTALSPINESEG_DATA/nnUNet/results/Dataset101_TotalSpineSeg_step1/
```

### 问题 3: Trainer 未找到

```bash
# 重新复制 trainer
cp $TOTALSPINESEG/totalspineseg/nnunet_extensions/nnUNetTrainer_LDH.py \
   /opt/conda/envs/tss2/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/

# 测试导入
python -c "from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_LDH import nnUNetTrainer_LDH; print('OK')"
```

### 问题 4: CUDA 内存不足

```bash
# 使用较小的 batch size（修改 plans）
# 或使用不同的 GPU
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh 105
```

---

## 预计时间

| 步骤 | 预计时间 |
|------|----------|
| 数据准备 | 10-30 分钟 (取决于样本数量和 Step 1 推理) |
| 训练 | 12-48 小时 (取决于 GPU 和样本数量) |

---

## 联系

如有问题，请检查：
1. 环境变量是否正确设置
2. 数据路径是否存在
3. GPU 是否可用
4. nnUNet 版本兼容性

