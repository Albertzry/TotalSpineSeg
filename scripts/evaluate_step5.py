#!/usr/bin/env python3
"""
Evaluate Step 5 (Dataset 105) LDH Segmentation Model on Test Set

This script evaluates the Step 5 model performance on the test set, focusing on
LDH (Lumbar Disc Herniation) binary segmentation metrics.

Usage:
    python scripts/evaluate_step5.py [options]

Options:
    --fold: Fold number to evaluate (default: 0)
    --trainer: Trainer name (default: nnUNetTrainer_LDH)
    --configuration: Configuration (default: 3d_fullres)
    --plans: Plans name (default: nnUNetPlans_small)
    --device: Device (default: cuda)
    --checkpoint: Checkpoint name (default: checkpoint_best.pth)
    --output-dir: Output directory for evaluation results (optional)
    --skip-prediction: Skip prediction if already done
    --dataset-id: Dataset ID (default: 105)
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import nibabel as nib
from scipy import ndimage
import pandas as pd

# Fix for PyTorch 2.6+ weights_only issue
try:
    import torch
    import torch.serialization
    import numpy as np
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
except Exception:
    pass


def get_env_paths() -> Tuple[Path, Path, Path, Path]:
    """Get paths from environment variables"""
    totalspineseg = Path(os.environ.get('TOTALSPINESEG', '/root/TotalSpineSeg-v2'))
    data_root = Path(os.environ.get('TOTALSPINESEG_DATA', '/opt/data/private/data_sum'))
    
    nnunet_raw = Path(os.environ.get('nnUNet_raw', data_root / 'nnUNet' / 'raw'))
    nnunet_preprocessed = Path(os.environ.get('nnUNet_preprocessed', data_root / 'nnUNet' / 'preprocessed'))
    nnunet_results = Path(os.environ.get('nnUNet_results', data_root / 'nnUNet' / 'results'))
    
    return totalspineseg, nnunet_raw, nnunet_preprocessed, nnunet_results


def load_nifti(file_path: Path) -> Tuple[np.ndarray, Optional[nib.Nifti1Header]]:
    """Load NIfTI file and return array and header"""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    nii = nib.load(str(file_path))
    data = nii.get_fdata().astype(np.float32)
    return data, nii.header


def compute_dice(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    """Compute Dice coefficient"""
    pred_binary = (pred > 0.5).astype(np.float32)
    target_binary = target.astype(np.float32)
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return float(dice)


def compute_iou(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    """Compute Intersection over Union (IoU)"""
    pred_binary = (pred > 0.5).astype(np.float32)
    target_binary = target.astype(np.float32)
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = (intersection + smooth) / (union + smooth)
    return float(iou)


def compute_precision_recall(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> Tuple[float, float]:
    """Compute Precision and Recall"""
    pred_binary = (pred > 0.5).astype(np.float32)
    target_binary = target.astype(np.float32)
    
    tp = (pred_binary * target_binary).sum()
    fp = (pred_binary * (1 - target_binary)).sum()
    fn = ((1 - pred_binary) * target_binary).sum()
    
    precision = (tp + smooth) / (tp + fp + smooth) if (tp + fp) > 0 else 0.0
    recall = (tp + smooth) / (tp + fn + smooth) if (tp + fn) > 0 else 0.0
    
    return float(precision), float(recall)


def compute_hausdorff_distance(pred: np.ndarray, target: np.ndarray, percentile: float = 95.0) -> float:
    """Compute 95th percentile Hausdorff Distance"""
    pred_binary = (pred > 0.5).astype(bool)
    target_binary = target.astype(bool)
    
    if target_binary.sum() == 0:
        return np.inf if pred_binary.sum() > 0 else 0.0
    
    if pred_binary.sum() == 0:
        return np.inf
    
    # Get boundary points
    pred_boundary = np.argwhere(ndimage.binary_erosion(pred_binary) != pred_binary)
    target_boundary = np.argwhere(ndimage.binary_erosion(target_binary) != target_binary)
    
    if len(pred_boundary) == 0 or len(target_boundary) == 0:
        return np.inf
    
    # Compute directed Hausdorff distances
    distances = []
    for p_point in pred_boundary:
        min_dist = np.min(np.linalg.norm(target_boundary - p_point, axis=1))
        distances.append(min_dist)
    
    if len(distances) == 0:
        return np.inf
    
    # Use percentile
    k = max(1, int(len(distances) * (1 - percentile / 100)))
    hd_value = np.partition(distances, -k)[-k] if k < len(distances) else max(distances)
    
    return float(hd_value)


def compute_average_surface_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Average Surface Distance (ASD)"""
    pred_binary = (pred > 0.5).astype(bool)
    target_binary = target.astype(bool)
    
    if target_binary.sum() == 0:
        return np.inf if pred_binary.sum() > 0 else 0.0
    
    if pred_binary.sum() == 0:
        return np.inf
    
    # Get boundary points
    pred_boundary = np.argwhere(ndimage.binary_erosion(pred_binary) != pred_binary)
    target_boundary = np.argwhere(ndimage.binary_erosion(target_binary) != target_binary)
    
    if len(pred_boundary) == 0 or len(target_boundary) == 0:
        return np.inf
    
    # Compute average distance from pred boundary to target boundary
    distances = []
    for p_point in pred_boundary:
        min_dist = np.min(np.linalg.norm(target_boundary - p_point, axis=1))
        distances.append(min_dist)
    
    return float(np.mean(distances)) if distances else np.inf


def compute_volume_metrics(pred: np.ndarray, target: np.ndarray, 
                          voxel_spacing: Optional[Tuple[float, ...]] = None) -> Dict[str, float]:
    """Compute volume-related metrics"""
    pred_binary = (pred > 0.5).astype(bool)
    target_binary = target.astype(bool)
    
    pred_volume = pred_binary.sum()
    target_volume = target_binary.sum()
    
    # Convert to mm³ if spacing is provided
    if voxel_spacing is not None:
        voxel_volume = np.prod(voxel_spacing)
        pred_volume_mm3 = pred_volume * voxel_volume
        target_volume_mm3 = target_volume * voxel_volume
    else:
        pred_volume_mm3 = None
        target_volume_mm3 = None
    
    volume_error = abs(pred_volume - target_volume)
    volume_error_pct = (volume_error / target_volume * 100) if target_volume > 0 else np.inf
    
    if voxel_spacing is not None:
        volume_error_mm3 = abs(pred_volume_mm3 - target_volume_mm3)
    else:
        volume_error_mm3 = None
    
    return {
        'pred_volume_voxels': float(pred_volume),
        'target_volume_voxels': float(target_volume),
        'volume_error_voxels': float(volume_error),
        'volume_error_pct': float(volume_error_pct) if volume_error_pct != np.inf else np.nan,
        'pred_volume_mm3': float(pred_volume_mm3) if pred_volume_mm3 is not None else np.nan,
        'target_volume_mm3': float(target_volume_mm3) if target_volume_mm3 is not None else np.nan,
        'volume_error_mm3': float(volume_error_mm3) if volume_error_mm3 is not None else np.nan,
    }


def evaluate_case(pred_path: Path, target_path: Path) -> Dict[str, float]:
    """Evaluate a single case"""
    pred_data, pred_header = load_nifti(pred_path)
    target_data, target_header = load_nifti(target_path)
    
    # Ensure same shape
    if pred_data.shape != target_data.shape:
        print(f"Warning: Shape mismatch {pred_path.name}: pred {pred_data.shape} vs target {target_data.shape}")
        min_shape = tuple(min(p, t) for p, t in zip(pred_data.shape, target_data.shape))
        pred_data = pred_data[:min_shape[0], :min_shape[1], :min_shape[2]]
        target_data = target_data[:min_shape[0], :min_shape[1], :min_shape[2]]
    
    # Get voxel spacing
    if pred_header is not None and 'pixdim' in pred_header:
        spacing = tuple(pred_header['pixdim'][1:4])
    else:
        spacing = None
    
    # Compute metrics
    dice = compute_dice(pred_data, target_data)
    iou = compute_iou(pred_data, target_data)
    precision, recall = compute_precision_recall(pred_data, target_data)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    hd95 = compute_hausdorff_distance(pred_data, target_data, percentile=95.0)
    asd = compute_average_surface_distance(pred_data, target_data)
    
    volume_metrics = compute_volume_metrics(pred_data, target_data, spacing)
    
    results = {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'hd95': hd95 if hd95 != np.inf else np.nan,
        'asd': asd if asd != np.inf else np.nan,
        **volume_metrics
    }
    
    return results


def run_prediction(dataset_id: int, fold: int, trainer: str, configuration: str,
                  plans: str, device: str, nnunet_raw: Path, nnunet_results: Path,
                  checkpoint: str = 'checkpoint_best.pth',
                  skip_if_exists: bool = True) -> Path:
    """Run prediction on test set if not already done"""
    dataset_name = f"Dataset{dataset_id:03d}_TotalSpineSeg_LDH"
    test_images_dir = nnunet_raw / dataset_name / "imagesTs"
    # Include checkpoint in output directory name to avoid conflicts
    checkpoint_suffix = checkpoint.replace('.pth', '').replace('checkpoint_', '')
    output_dir = nnunet_results / dataset_name / f"{trainer}__{plans}__{configuration}" / f"fold_{fold}" / f"test_{checkpoint_suffix}"
    
    if skip_if_exists and output_dir.exists() and len(list(output_dir.glob("*.nii.gz"))) > 0:
        print(f"✓ Predictions already exist in {output_dir} (using checkpoint: {checkpoint})")
        return output_dir
    
    print(f"Running prediction on test set...")
    print(f"  Dataset: {dataset_name}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Test images: {test_images_dir}")
    print(f"  Output: {output_dir}")
    
    if not test_images_dir.exists():
        raise FileNotFoundError(f"Test images directory not found: {test_images_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables
    env = os.environ.copy()
    env['nnUNet_raw'] = str(nnunet_raw)
    env['nnUNet_preprocessed'] = str(Path(os.environ.get('nnUNet_preprocessed', '/opt/data/private/data_sum/nnUNet/preprocessed')))
    env['nnUNet_results'] = str(nnunet_results)
    env['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
    
    # Run prediction
    cmd = [
        'nnUNetv2_predict',
        '-d', str(dataset_id),
        '-i', str(test_images_dir),
        '-o', str(output_dir),
        '-f', str(fold),
        '-c', configuration,
        '-tr', trainer,
        '-p', plans,
        '-device', device,
        '-chk', checkpoint,  # Specify checkpoint
        '-npp', '2',
        '-nps', '2'
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=False)
    
    if result.returncode != 0:
        raise RuntimeError(f"Prediction failed with return code {result.returncode}")
    
    print(f"✓ Prediction completed")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Evaluate Step 5 LDH model on test set')
    parser.add_argument('--fold', type=int, default=0, help='Fold number (default: 0)')
    parser.add_argument('--trainer', type=str, default='nnUNetTrainer_LDH', 
                       help='Trainer name (default: nnUNetTrainer_LDH)')
    parser.add_argument('--configuration', type=str, default='3d_fullres',
                       help='Configuration (default: 3d_fullres)')
    parser.add_argument('--plans', type=str, default='nnUNetPlans_small',
                       help='Plans name (default: nnUNetPlans_small)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (default: cuda)')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory for evaluation results (optional)')
    parser.add_argument('--skip-prediction', action='store_true',
                       help='Skip prediction if already done')
    parser.add_argument('--dataset-id', type=int, default=105,
                       help='Dataset ID (default: 105)')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_best.pth',
                       help='Checkpoint name to use (default: checkpoint_best.pth). Options: checkpoint_best.pth, checkpoint_final.pth')
    
    args = parser.parse_args()
    
    # Get paths
    totalspineseg, nnunet_raw, nnunet_preprocessed, nnunet_results = get_env_paths()
    
    dataset_id = args.dataset_id
    dataset_name = f"Dataset{dataset_id:03d}_TotalSpineSeg_LDH"
    
    print("=" * 80)
    print("Step 5 LDH Model Evaluation")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Fold: {args.fold}")
    print(f"Trainer: {args.trainer}")
    print(f"Configuration: {args.configuration}")
    print(f"Plans: {args.plans}")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 80)
    
    # Run prediction if needed
    pred_output_dir = run_prediction(
        dataset_id=dataset_id,
        fold=args.fold,
        trainer=args.trainer,
        configuration=args.configuration,
        plans=args.plans,
        device=args.device,
        nnunet_raw=nnunet_raw,
        nnunet_results=nnunet_results,
        checkpoint=args.checkpoint,
        skip_if_exists=args.skip_prediction
    )
    
    # Get test labels directory
    test_labels_dir = nnunet_raw / dataset_name / "labelsTs"
    
    if not test_labels_dir.exists():
        raise FileNotFoundError(f"Test labels directory not found: {test_labels_dir}")
    
    # Find all test cases
    test_label_files = sorted(test_labels_dir.glob("*.nii.gz"))
    
    if len(test_label_files) == 0:
        raise ValueError(f"No test label files found in {test_labels_dir}")
    
    print(f"\nFound {len(test_label_files)} test cases")
    print("Evaluating...")
    
    # Evaluate each case
    all_results = []
    failed_cases = []
    
    for label_file in test_label_files:
        case_name = label_file.stem.replace('.nii', '').replace('_0000', '').replace('_0001', '')
        
        # Find corresponding prediction
        pred_file = pred_output_dir / label_file.name
        
        if not pred_file.exists():
            print(f"⚠ Warning: Prediction not found for {case_name}: {pred_file}")
            failed_cases.append(case_name)
            continue
        
        try:
            results = evaluate_case(pred_file, label_file)
            results['case_name'] = case_name
            all_results.append(results)
            print(f"  ✓ {case_name}: Dice={results['dice']:.4f}, IoU={results['iou']:.4f}, "
                  f"Precision={results['precision']:.4f}, Recall={results['recall']:.4f}")
        except Exception as e:
            print(f"  ✗ Error evaluating {case_name}: {e}")
            failed_cases.append(case_name)
            import traceback
            traceback.print_exc()
    
    if len(all_results) == 0:
        raise ValueError("No cases were successfully evaluated!")
    
    # Compute summary statistics
    df = pd.DataFrame(all_results)
    
    # Summary metrics (excluding case_name)
    metric_cols = [c for c in df.columns if c != 'case_name']
    summary = df[metric_cols].describe()
    
    # Add mean and std
    summary.loc['mean'] = df[metric_cols].mean()
    summary.loc['std'] = df[metric_cols].std()
    
    # Print summary
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"Total cases evaluated: {len(all_results)}")
    if failed_cases:
        print(f"Failed cases: {len(failed_cases)}")
        print(f"  {', '.join(failed_cases)}")
    
    print("\nKey Metrics (Mean ± Std):")
    print(f"  Dice Score:        {df['dice'].mean():.4f} ± {df['dice'].std():.4f}")
    print(f"  IoU:               {df['iou'].mean():.4f} ± {df['iou'].std():.4f}")
    print(f"  Precision:         {df['precision'].mean():.4f} ± {df['precision'].std():.4f}")
    print(f"  Recall:            {df['recall'].mean():.4f} ± {df['recall'].std():.4f}")
    print(f"  F1 Score:          {df['f1'].mean():.4f} ± {df['f1'].std():.4f}")
    
    if not df['hd95'].isna().all():
        hd95_mean = df['hd95'].mean()
        hd95_std = df['hd95'].std()
        print(f"  Hausdorff Distance (95%): {hd95_mean:.2f} ± {hd95_std:.2f} mm")
    
    if not df['asd'].isna().all():
        asd_mean = df['asd'].mean()
        asd_std = df['asd'].std()
        print(f"  Average Surface Distance: {asd_mean:.2f} ± {asd_std:.2f} mm")
    
    print("=" * 80)
    
    # Save results
    if args.output_dir is None:
        output_dir = totalspineseg / "results" / "step5_evaluation"
    else:
        output_dir = args.output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint suffix for filenames
    checkpoint_suffix = args.checkpoint.replace('.pth', '').replace('checkpoint_', '')
    
    # Save detailed results CSV
    csv_path = output_dir / f"step5_evaluation_fold{args.fold}_{checkpoint_suffix}_detailed.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Detailed results saved to: {csv_path}")
    
    # Save summary CSV
    summary_path = output_dir / f"step5_evaluation_fold{args.fold}_{checkpoint_suffix}_summary.csv"
    summary.to_csv(summary_path)
    print(f"✓ Summary statistics saved to: {summary_path}")
    
    # Save summary JSON
    summary_json = {
        'dataset': dataset_name,
        'fold': args.fold,
        'trainer': args.trainer,
        'configuration': args.configuration,
        'plans': args.plans,
        'checkpoint': args.checkpoint,
        'total_cases': len(all_results),
        'failed_cases': failed_cases,
        'metrics': {
            'dice': {'mean': float(df['dice'].mean()), 'std': float(df['dice'].std())},
            'iou': {'mean': float(df['iou'].mean()), 'std': float(df['iou'].std())},
            'precision': {'mean': float(df['precision'].mean()), 'std': float(df['precision'].std())},
            'recall': {'mean': float(df['recall'].mean()), 'std': float(df['recall'].std())},
            'f1': {'mean': float(df['f1'].mean()), 'std': float(df['f1'].std())},
        }
    }
    
    if not df['hd95'].isna().all():
        summary_json['metrics']['hd95'] = {
            'mean': float(df['hd95'].mean()),
            'std': float(df['hd95'].std())
        }
    
    if not df['asd'].isna().all():
        summary_json['metrics']['asd'] = {
            'mean': float(df['asd'].mean()),
            'std': float(df['asd'].std())
        }
    
    json_path = output_dir / f"step5_evaluation_fold{args.fold}_{checkpoint_suffix}_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary_json, f, indent=2)
    print(f"✓ Summary JSON saved to: {json_path}")
    
    print("\n" + "=" * 80)
    print("Evaluation completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()

