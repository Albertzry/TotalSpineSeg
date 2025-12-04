"""
创建迷你测试数据集，用于快速测试trainer

从Dataset102中抽取一小部分样本创建Dataset103_mini
包含LDH样本和普通样本，用于快速验证训练流程
"""

import os
import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# 配置
TOTALSPINESEG_DATA = Path(os.environ.get('TOTALSPINESEG_DATA', '/opt/data/private/data_sum'))
NNUNET_RAW = TOTALSPINESEG_DATA / "nnUNet/raw"

SOURCE_DATASET = "Dataset102_TotalSpineSeg_step2"
TARGET_DATASET = "Dataset103_TotalSpineSeg_mini"

# 抽取数量
NUM_LDH_SAMPLES = 30  # LDH样本数
NUM_NORMAL_SAMPLES = 30  # 普通样本数
NUM_TEST_SAMPLES = 10  # 测试集样本数

def main():
    source_dir = NNUNET_RAW / SOURCE_DATASET
    target_dir = NNUNET_RAW / TARGET_DATASET
    
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    
    # 检查源数据集
    if not source_dir.exists():
        print(f"Error: Source dataset not found at {source_dir}")
        return
    
    # 获取所有标签文件
    labels_dir = source_dir / "labelsTr"
    all_labels = sorted([f.name for f in labels_dir.glob("*.nii.gz")])
    
    # 分类LDH和普通样本
    ldh_samples = [f for f in all_labels if "sub-LDH" in f]
    normal_samples = [f for f in all_labels if "sub-LDH" not in f]
    
    print(f"Total samples: {len(all_labels)}")
    print(f"LDH samples: {len(ldh_samples)}")
    print(f"Normal samples: {len(normal_samples)}")
    
    # 随机抽取
    random.seed(42)  # 固定种子保证可重复
    selected_ldh = random.sample(ldh_samples, min(NUM_LDH_SAMPLES, len(ldh_samples)))
    selected_normal = random.sample(normal_samples, min(NUM_NORMAL_SAMPLES, len(normal_samples)))
    
    # 合并并分割训练/测试
    all_selected = selected_ldh + selected_normal
    random.shuffle(all_selected)
    
    train_samples = all_selected[:-NUM_TEST_SAMPLES]
    test_samples = all_selected[-NUM_TEST_SAMPLES:]
    
    print(f"\nSelected for training: {len(train_samples)}")
    print(f"  - LDH: {len([s for s in train_samples if 'sub-LDH' in s])}")
    print(f"  - Normal: {len([s for s in train_samples if 'sub-LDH' not in s])}")
    print(f"Selected for testing: {len(test_samples)}")
    
    # 创建目标目录
    if target_dir.exists():
        print(f"\nRemoving existing {target_dir}")
        shutil.rmtree(target_dir)
    
    (target_dir / "imagesTr").mkdir(parents=True)
    (target_dir / "labelsTr").mkdir(parents=True)
    (target_dir / "imagesTs").mkdir(parents=True)
    (target_dir / "labelsTs").mkdir(parents=True)
    
    # 复制训练数据
    print("\nCopying training data...")
    for label_file in tqdm(train_samples, desc="Training"):
        # 标签文件
        src_label = source_dir / "labelsTr" / label_file
        dst_label = target_dir / "labelsTr" / label_file
        shutil.copy2(src_label, dst_label)
        
        # 图像文件（两个通道）
        base_name = label_file.replace(".nii.gz", "")
        for channel in ["_0000.nii.gz", "_0001.nii.gz"]:
            src_img = source_dir / "imagesTr" / (base_name + channel)
            dst_img = target_dir / "imagesTr" / (base_name + channel)
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
    
    # 复制测试数据
    print("Copying test data...")
    for label_file in tqdm(test_samples, desc="Testing"):
        # 标签文件
        src_label = source_dir / "labelsTr" / label_file
        dst_label = target_dir / "labelsTs" / label_file
        shutil.copy2(src_label, dst_label)
        
        # 图像文件
        base_name = label_file.replace(".nii.gz", "")
        for channel in ["_0000.nii.gz", "_0001.nii.gz"]:
            src_img = source_dir / "imagesTr" / (base_name + channel)
            dst_img = target_dir / "imagesTs" / (base_name + channel)
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
    
    # 复制并修改dataset.json
    print("\nCreating dataset.json...")
    with open(source_dir / "dataset.json") as f:
        dataset_json = json.load(f)
    
    dataset_json["numTraining"] = len(train_samples)
    
    with open(target_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Mini dataset created at: {target_dir}")
    print(f"Training samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print(f"{'='*60}")
    print(f"\nTo train with mini dataset:")
    print(f"  bash scripts/train.sh 103 0")


if __name__ == "__main__":
    main()

