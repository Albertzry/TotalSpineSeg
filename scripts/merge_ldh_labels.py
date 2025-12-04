"""
Merge LDH labels with Step 1 predictions for Dataset 102.

优化措施：
1. 检查是否已有预测结果，跳过重复推理
2. 批量推理而不是逐个处理
3. 并行处理合并操作
"""

import os
import glob
import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Configuration
if 'TOTALSPINESEG_DATA' not in os.environ:
    print("Error: TOTALSPINESEG_DATA environment variable not set.")
    exit(1)

TOTALSPINESEG_DATA = Path(os.environ['TOTALSPINESEG_DATA'])
NNUNET_RESULTS = TOTALSPINESEG_DATA / "nnUNet/results"
NNUNET_RAW = TOTALSPINESEG_DATA / "nnUNet/raw"

DATASET_LDH_IMAGES = NNUNET_RAW / "Dataset99_TotalSpineSeg/imagesTr"
DATASET_LDH_LABELS = NNUNET_RAW / "Dataset99_TotalSpineSeg/labelsTr"
DATASET_102_LABELS_TR = NNUNET_RAW / "Dataset102_TotalSpineSeg_step2/labelsTr"

# Step 1 Model Config
DATASET_ID = 101
TRAINER = "nnUNetTrainer_DASegOrd0_NoMirroring"
PLANS = "nnUNetPlans_small"
CONFIG = "3d_fullres"
FOLD = 0


def merge_single_case(args):
    """合并单个case的预测和GT标签"""
    pred_path, gt_path, output_path = args
    
    try:
        pred_nii = nib.load(pred_path)
        pred_data = pred_nii.get_fdata().astype(np.int32)
        
        gt_nii = nib.load(gt_path)
        gt_data = gt_nii.get_fdata().astype(np.int32)
        
        # Merge: Step1预测 + LDH GT
        final_data = pred_data.copy()
        ldh_mask = (gt_data == 101)
        final_data[ldh_mask] = 12
        
        out_nii = nib.Nifti1Image(final_data, pred_nii.affine, pred_nii.header)
        nib.save(out_nii, output_path)
        
        return True, str(output_path)
    except Exception as e:
        return False, f"{output_path}: {e}"


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 获取CPU数量用于并行
    num_workers = min(multiprocessing.cpu_count(), 12)
    print(f"Using {num_workers} workers for parallel processing")

    # 1. 查找LDH图像
    ldh_images = sorted(glob.glob(str(DATASET_LDH_IMAGES / "sub-LDH*_0000.nii.gz")))
    print(f"Found {len(ldh_images)} LDH images to process.")

    if not ldh_images:
        print("No LDH images found. Check paths.")
        return

    # 预测输出目录
    temp_pred_dir = Path("temp_step1_preds_ldh")
    temp_input_dir = Path("temp_ldh_input")
    
    # 2. 检查是否需要运行推理
    need_inference = False
    for img in ldh_images:
        fname = os.path.basename(img)
        base_name = fname.replace("_0000.nii.gz", "")
        label_name = base_name + ".nii.gz"
        pred_path = temp_pred_dir / label_name
        
        if not pred_path.exists():
            need_inference = True
            break
    
    if need_inference:
        print("Running Step 1 inference on LDH images...")
        
        # 创建输入目录
        import shutil
        if temp_input_dir.exists():
        shutil.rmtree(temp_input_dir)
    os.makedirs(temp_input_dir)
        os.makedirs(temp_pred_dir, exist_ok=True)
    
        # 只链接需要预测的图像
        images_to_predict = []
    for img in ldh_images:
            fname = os.path.basename(img)
            base_name = fname.replace("_0000.nii.gz", "")
            label_name = base_name + ".nii.gz"
            pred_path = temp_pred_dir / label_name
            
            if not pred_path.exists():
                dst = temp_input_dir / fname
        if not dst.exists():
            os.symlink(img, dst)
                images_to_predict.append(img)
        
        if images_to_predict:
            print(f"Need to predict {len(images_to_predict)} images (skipping already predicted)")
            
            # 使用nnUNet批量推理
    cmd = [
        "nnUNetv2_predict",
        "-d", str(DATASET_ID),
        "-i", str(temp_input_dir),
        "-o", str(temp_pred_dir),
        "-f", str(FOLD),
        "-c", CONFIG,
        "-tr", TRAINER,
        "-p", PLANS,
        "-device", device.type
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    subprocess.check_call(cmd)
        else:
            print("All predictions already exist, skipping inference.")
    else:
        print("All Step 1 predictions already exist, skipping inference step.")
    
    # 3. 并行合并标签
    print("Merging Step 1 predictions with Ground Truth LDH labels...")
    os.makedirs(DATASET_102_LABELS_TR, exist_ok=True)
    
    # 准备合并任务
    merge_tasks = []
    skipped = 0
    
    for img_path in ldh_images:
        fname = os.path.basename(img_path)
        base_name = fname.replace("_0000.nii.gz", "")
        label_name = base_name + ".nii.gz"
        
        pred_path = temp_pred_dir / label_name
        gt_path = DATASET_LDH_LABELS / label_name 
        output_path = DATASET_102_LABELS_TR / label_name
        
        # 跳过已存在的输出
        if output_path.exists():
            skipped += 1
            continue
        
        if not pred_path.exists():
            print(f"Warning: Prediction not found for {label_name}")
            continue
            
        if not gt_path.exists():
            print(f"Warning: GT label not found for {label_name}")
            continue
        
        merge_tasks.append((str(pred_path), str(gt_path), str(output_path)))
    
    if skipped > 0:
        print(f"Skipped {skipped} already merged files")
    
    if merge_tasks:
        print(f"Merging {len(merge_tasks)} files in parallel...")
        
        # 并行处理
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(merge_single_case, task): task for task in merge_tasks}
            
            with tqdm(total=len(merge_tasks), desc="Merging") as pbar:
                for future in as_completed(futures):
                    success, msg = future.result()
                    if not success:
                        print(f"Error: {msg}")
                    pbar.update(1)
    else:
        print("All files already merged.")

    print("Merge complete.")
    

if __name__ == "__main__":
    main()
