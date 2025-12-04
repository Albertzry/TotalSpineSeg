#!/usr/bin/env python3
"""
Prepare Dataset 105 for LDH Specialized Training

This script prepares Dataset 105 which contains only LDH samples for specialized
LDH (Lumbar Disc Herniation) training. The preparation process:

1. Extract LDH samples (sub-LDH*) from augmented Dataset 100
2. Run Step 1 model inference on LDH images to get spine structure predictions
3. Merge Step 1 predictions with ground truth LDH labels
4. Create the final Dataset 105 in nnUNet format

Usage:
    python scripts/prepare_dataset_105.py

Environment Variables:
    TOTALSPINESEG: Path to TotalSpineSeg repository
    TOTALSPINESEG_DATA: Path to TotalSpineSeg data folder
    TOTALSPINESEG_JOBS: Number of CPU workers (default: 12)

Requirements:
    - Dataset 100 (augmented) must be prepared
    - Step 1 model (Dataset 101) must be trained
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional

import numpy as np
import nibabel as nib

# Fix for PyTorch 2.6+ weights_only issue when loading old checkpoints
try:
    import torch
    import torch.serialization
    # Add numpy scalar to safe globals for loading old checkpoints
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
except Exception:
    pass


def get_env_paths() -> Tuple[Path, Path, int]:
    """Get paths from environment variables"""
    totalspineseg = Path(os.environ.get('TOTALSPINESEG', 'totalspineseg')).resolve()
    totalspineseg_data = Path(os.environ.get('TOTALSPINESEG_DATA', 'data')).resolve()
    # 限制最大线程数为 6
    jobs = min(int(os.environ.get('TOTALSPINESEG_JOBS', '6')), 6)
    
    return totalspineseg, totalspineseg_data, jobs


def find_ldh_samples(dataset_path: Path) -> List[str]:
    """
    Find all LDH samples (sub-LDH*) in the dataset
    
    Args:
        dataset_path: Path to source dataset (Dataset100)
    
    Returns:
        List of sample IDs (without _0000.nii.gz suffix)
    """
    images_tr = dataset_path / 'imagesTr'
    ldh_samples = []
    
    for f in images_tr.glob('sub-LDH*_0000.nii.gz'):
        sample_id = f.name.replace('_0000.nii.gz', '')
        ldh_samples.append(sample_id)
    
    # Also check for test samples
    images_ts = dataset_path / 'imagesTs'
    if images_ts.exists():
        for f in images_ts.glob('sub-LDH*_0000.nii.gz'):
            sample_id = f.name.replace('_0000.nii.gz', '')
            if sample_id not in ldh_samples:
                ldh_samples.append(sample_id)
    
    return sorted(ldh_samples)


def copy_ldh_images(src_dataset: Path, dst_dataset: Path, 
                    ldh_samples: List[str], jobs: int = 12) -> None:
    """
    Copy LDH image files to Dataset 105
    
    Args:
        src_dataset: Source dataset path (Dataset100)
        dst_dataset: Destination dataset path (Dataset105)
        ldh_samples: List of LDH sample IDs
        jobs: Number of parallel workers
    """
    dst_images = dst_dataset / 'imagesTr'
    dst_images.mkdir(parents=True, exist_ok=True)
    
    def copy_sample(sample_id: str) -> Optional[str]:
        # Try training set first
        src_file = src_dataset / 'imagesTr' / f'{sample_id}_0000.nii.gz'
        if not src_file.exists():
            # Try test set
            src_file = src_dataset / 'imagesTs' / f'{sample_id}_0000.nii.gz'
        
        if src_file.exists():
            dst_file = dst_images / f'{sample_id}_0000.nii.gz'
            shutil.copy2(src_file, dst_file)
            return sample_id
        return None
    
    print(f"Copying {len(ldh_samples)} LDH images...")
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(copy_sample, s): s for s in ldh_samples}
        copied = 0
        for future in as_completed(futures):
            result = future.result()
            if result:
                copied += 1
        print(f"Copied {copied}/{len(ldh_samples)} images")


def copy_ldh_labels(src_dataset: Path, dst_dataset: Path,
                    ldh_samples: List[str], jobs: int = 12) -> None:
    """
    Copy LDH ground truth label files to Dataset 105
    
    These labels will be merged with Step 1 predictions later
    
    Args:
        src_dataset: Source dataset path (Dataset100)
        dst_dataset: Destination dataset path (Dataset105)
        ldh_samples: List of LDH sample IDs
        jobs: Number of parallel workers
    """
    dst_labels = dst_dataset / 'labelsGT'  # GT labels (to be merged)
    dst_labels.mkdir(parents=True, exist_ok=True)
    
    def copy_sample(sample_id: str) -> Optional[str]:
        # Try training set first
        src_file = src_dataset / 'labelsTr' / f'{sample_id}.nii.gz'
        if not src_file.exists():
            # Try test set
            src_file = src_dataset / 'labelsTs' / f'{sample_id}.nii.gz'
        
        if src_file.exists():
            dst_file = dst_labels / f'{sample_id}.nii.gz'
            shutil.copy2(src_file, dst_file)
            return sample_id
        return None
    
    print(f"Copying {len(ldh_samples)} LDH ground truth labels...")
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(copy_sample, s): s for s in ldh_samples}
        copied = 0
        for future in as_completed(futures):
            result = future.result()
            if result:
                copied += 1
        print(f"Copied {copied}/{len(ldh_samples)} GT labels")


def run_step1_inference(dst_dataset: Path, nnunet_results: Path,
                        nnunet_raw: Path, nnunet_preprocessed: Path,
                        jobs: int = 12) -> None:
    """
    Run Step 1 model inference on LDH images to get spine structure predictions
    
    Args:
        dst_dataset: Dataset 105 path
        nnunet_results: nnUNet results path
        nnunet_raw: nnUNet raw data path
        nnunet_preprocessed: nnUNet preprocessed data path
        jobs: Number of parallel workers
    """
    input_dir = dst_dataset / 'imagesTr'
    output_dir = dst_dataset / 'labelsStep1'  # Step 1 predictions
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find Step 1 model
    step1_model_path = nnunet_results / 'Dataset101_TotalSpineSeg_step1'
    
    # Check if model exists
    model_configs = list(step1_model_path.glob('*__*__*/fold_*/checkpoint_final.pth'))
    if not model_configs:
        print("Warning: Step 1 model not found. Please train Dataset 101 first.")
        print(f"Expected path: {step1_model_path}")
        print("Skipping Step 1 inference...")
        return
    
    # Get trainer, planner, config from model path
    model_dir = model_configs[0].parent.parent
    trainer_config = model_dir.name  # e.g., nnUNetTrainer_DASegOrd0_NoMirroring__nnUNetPlans_small__3d_fullres
    parts = trainer_config.split('__')
    trainer = parts[0]
    plans = parts[1] if len(parts) > 1 else 'nnUNetPlans'
    config = parts[2] if len(parts) > 2 else '3d_fullres'
    
    fold = model_configs[0].parent.name.replace('fold_', '')
    
    print(f"Running Step 1 inference on LDH images...")
    print(f"  Model: {trainer_config}")
    print(f"  Fold: {fold}")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    
    # Prepare input files (Step 1 uses single channel)
    input_step1 = dst_dataset / 'imagesTr_step1'
    input_step1.mkdir(parents=True, exist_ok=True)
    
    # Copy and rename files (remove _0001 channel if exists)
    input_count = 0
    for f in input_dir.glob('*_0000.nii.gz'):
        dst_file = input_step1 / f.name
        if not dst_file.exists():
            shutil.copy2(f, dst_file)
        input_count += 1
    
    print(f"  Prepared {input_count} input files")
    
    # Set nnUNet environment variables (copy current env to preserve CUDA_VISIBLE_DEVICES)
    env = os.environ.copy()
    env['nnUNet_raw'] = str(nnunet_raw)
    env['nnUNet_preprocessed'] = str(nnunet_preprocessed)
    env['nnUNet_results'] = str(nnunet_results)
    # Fix for PyTorch 2.6+ weights_only issue
    env['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
    
    # Check for GPU availability
    import torch
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
    print(f"  CUDA_VISIBLE_DEVICES: {cuda_devices}")
    
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("  Warning: No GPU available, using CPU (this will be slow)")
    
    # Run nnUNet predict
    cmd = [
        'nnUNetv2_predict',
        '-d', '101',
        '-i', str(input_step1),
        '-o', str(output_dir),
        '-f', fold,
        '-c', config,
        '-tr', trainer,
        '-p', plans,
        '-device', device,
        '-npp', str(min(jobs, 2)),  # Limit preprocessing workers
        '-nps', str(min(jobs, 2)),  # Limit segmentation workers
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print(f"  nnUNet_raw: {env['nnUNet_raw']}")
    print(f"  nnUNet_results: {env['nnUNet_results']}")
    print("")
    print("=" * 40)
    print("nnUNet inference output:")
    print("=" * 40)
    
    try:
        # Don't capture output so user can see progress
        result = subprocess.run(cmd, check=True, env=env)
        print("=" * 40)
        print("Step 1 inference completed successfully")
        
        # Check output files
        output_files = list(output_dir.glob('*.nii.gz'))
        print(f"  Generated {len(output_files)} prediction files")
        
        if output_files:
            print(f"  Sample output: {output_files[0].name}")
    except subprocess.CalledProcessError as e:
        print("=" * 40)
        print(f"Error running Step 1 inference: {e}")
        print("Please ensure Step 1 model is trained and environment is set up correctly")
    
    # Clean up temp input dir
    if input_step1.exists():
        shutil.rmtree(input_step1)


def create_ldh_only_labels(dst_dataset: Path, jobs: int = 12) -> None:
    """
    Create LDH-only binary labels.
    
    Output labels: 0=background, 1=LDH
    
    LDH comes from ground truth (original label 101).
    Step 1 predictions are used as second INPUT channel, not merged into labels.
    
    Args:
        dst_dataset: Dataset 105 path
        jobs: Number of parallel workers
    """
    gt_labels_dir = dst_dataset / 'labelsGT'
    output_labels_dir = dst_dataset / 'labelsTr'
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    def process_sample(sample_id: str) -> Optional[str]:
        gt_file = gt_labels_dir / f'{sample_id}.nii.gz'
        output_file = output_labels_dir / f'{sample_id}.nii.gz'
        
        if not gt_file.exists():
            print(f"Warning: GT label not found for {sample_id}")
            return None
        
        gt_nii = nib.load(gt_file)
        gt_data = gt_nii.get_fdata().astype(np.int16)
        
        # Create binary label: 0=background, 1=LDH
        output_data = np.zeros_like(gt_data, dtype=np.int16)
        
        # LDH from ground truth (label 101 -> 1)
        ldh_mask = gt_data == 101
        ldh_count = ldh_mask.sum()
        output_data[ldh_mask] = 1  # LDH = 1
        
        if ldh_count > 0:
            print(f"  {sample_id}: {ldh_count} LDH voxels")
        else:
            print(f"  Warning: {sample_id} has no LDH label!")
        
        # Save binary label
        output_nii = nib.Nifti1Image(output_data, gt_nii.affine, gt_nii.header)
        nib.save(output_nii, output_file)
        
        return sample_id
    
    samples = [f.name.replace('.nii.gz', '') for f in gt_labels_dir.glob('*.nii.gz')]
    
    print(f"Creating LDH-only binary labels for {len(samples)} samples...")
    print("Output format: 0=background, 1=LDH")
    print("-" * 60)
    
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(process_sample, s): s for s in samples}
        processed = 0
        for future in as_completed(futures):
            result = future.result()
            if result:
                processed += 1
    
    print("-" * 60)
    print(f"Created {processed}/{len(samples)} binary labels")


def create_second_channel_from_step1(dst_dataset: Path, jobs: int = 12) -> None:
    """
    Create the second input channel from Step 1 predictions.
    
    The second channel contains Step 1 spine structure predictions,
    providing anatomical context for LDH detection.
    
    Step 1 output format (labels 1-9):
    - 1-5: disc labels (used for anatomical attention)
    - 6-7: vertebrae
    - 8: canal
    - 9: cord
    
    Args:
        dst_dataset: Dataset 105 path
        jobs: Number of parallel workers
    """
    step1_dir = dst_dataset / 'labelsStep1'
    images_dir = dst_dataset / 'imagesTr'
    
    if not step1_dir.exists():
        print("Warning: Step 1 predictions not found. Run Step 1 inference first.")
        return
    
    def create_channel(sample_id: str) -> Optional[str]:
        step1_file = step1_dir / f'{sample_id}.nii.gz'
        ch0_file = images_dir / f'{sample_id}_0000.nii.gz'
        ch1_file = images_dir / f'{sample_id}_0001.nii.gz'
        
        if not step1_file.exists():
            print(f"  Warning: No Step 1 prediction for {sample_id}")
            # Create empty channel if no Step 1 prediction
            if ch0_file.exists():
                ref_nii = nib.load(ch0_file)
                ch1_data = np.zeros(ref_nii.shape, dtype=np.float32)
                ch1_nii = nib.Nifti1Image(ch1_data, ref_nii.affine, ref_nii.header)
                nib.save(ch1_nii, ch1_file)
            return None
        
        if not ch0_file.exists():
            print(f"  Warning: No image for {sample_id}")
            return None
        
        # Load Step 1 prediction
        step1_nii = nib.load(step1_file)
        step1_data = step1_nii.get_fdata().astype(np.float32)
        
        # Use Step 1 prediction directly as second channel
        # This preserves all anatomical information (disc, vertebrae, canal, cord)
        ch1_nii = nib.Nifti1Image(step1_data, step1_nii.affine, step1_nii.header)
        nib.save(ch1_nii, ch1_file)
        
        return sample_id
    
    # Get sample list from images
    samples = [f.name.replace('_0000.nii.gz', '') for f in images_dir.glob('*_0000.nii.gz')]
    
    print(f"Creating second channel (Step 1 predictions) for {len(samples)} samples...")
    print("Second channel contains: disc (1-5), vertebrae (6-7), canal (8), cord (9)")
    print("-" * 60)
    
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(create_channel, s): s for s in samples}
        created = 0
        for future in as_completed(futures):
            result = future.result()
            if result:
                created += 1
    
    print("-" * 60)
    print(f"Created second channel for {created}/{len(samples)} samples")


def create_test_split(dst_dataset: Path, test_ratio: float = 0.1) -> None:
    """
    Create test split from training data
    
    Args:
        dst_dataset: Dataset 105 path
        test_ratio: Ratio of data to use for testing
    """
    images_tr = dst_dataset / 'imagesTr'
    labels_tr = dst_dataset / 'labelsTr'
    images_ts = dst_dataset / 'imagesTs'
    labels_ts = dst_dataset / 'labelsTs'
    
    images_ts.mkdir(parents=True, exist_ok=True)
    labels_ts.mkdir(parents=True, exist_ok=True)
    
    # Get all samples
    samples = [f.name.replace('_0000.nii.gz', '') for f in images_tr.glob('*_0000.nii.gz')]
    
    if not samples:
        print("No samples found for test split")
        return
    
    # Random split
    np.random.seed(42)
    np.random.shuffle(samples)
    n_test = max(1, int(len(samples) * test_ratio))
    test_samples = samples[:n_test]
    
    print(f"Moving {n_test} samples to test set...")
    
    for sample_id in test_samples:
        # Move images
        for suffix in ['_0000.nii.gz', '_0001.nii.gz']:
            src = images_tr / f'{sample_id}{suffix}'
            dst = images_ts / f'{sample_id}{suffix}'
            if src.exists():
                shutil.move(str(src), str(dst))
        
        # Move labels
        src = labels_tr / f'{sample_id}.nii.gz'
        dst = labels_ts / f'{sample_id}.nii.gz'
        if src.exists():
            shutil.move(str(src), str(dst))
    
    print(f"Test set created with {n_test} samples")


def create_dataset_json(dst_dataset: Path, resources_path: Path) -> None:
    """
    Create dataset.json for Dataset 105
    
    Args:
        dst_dataset: Dataset 105 path
        resources_path: Path to resources folder
    """
    # Load template
    template_path = resources_path / 'datasets' / 'dataset_step5_ldh.json'
    with open(template_path, 'r') as f:
        dataset_json = json.load(f)
    
    # Count training samples
    labels_tr = dst_dataset / 'labelsTr'
    num_training = len(list(labels_tr.glob('*.nii.gz')))
    dataset_json['numTraining'] = num_training
    
    # Save dataset.json
    output_path = dst_dataset / 'dataset.json'
    with open(output_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)
    
    print(f"Created dataset.json with {num_training} training samples")


def main():
    parser = argparse.ArgumentParser(description='Prepare Dataset 105 for LDH training')
    parser.add_argument('--skip-inference', action='store_true',
                        help='Skip Step 1 inference (use if predictions already exist)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='Ratio of data for testing (default: 0.1)')
    args = parser.parse_args()
    
    # Get paths
    totalspineseg, totalspineseg_data, jobs = get_env_paths()
    
    nnunet_raw = totalspineseg_data / 'nnUNet' / 'raw'
    nnunet_preprocessed = totalspineseg_data / 'nnUNet' / 'preprocessed'
    nnunet_results = totalspineseg_data / 'nnUNet' / 'results'
    resources = totalspineseg / 'totalspineseg' / 'resources'
    
    # Source and destination datasets
    src_dataset = nnunet_raw / 'Dataset100_TotalSpineSeg_Aug'
    dst_dataset = nnunet_raw / 'Dataset105_TotalSpineSeg_LDH'
    
    # Check if source dataset exists
    if not src_dataset.exists():
        print(f"Error: Source dataset not found: {src_dataset}")
        print("Please run prepare_datasets.sh first to create Dataset 100")
        sys.exit(1)
    
    print("=" * 60)
    print("Preparing Dataset 105 for LDH Binary Segmentation")
    print("=" * 60)
    print(f"Source: {src_dataset}")
    print(f"Destination: {dst_dataset}")
    print(f"Workers: {jobs}")
    print("")
    print("Output Format:")
    print("  - Labels: 0=background, 1=LDH (binary)")
    print("  - Input Channel 0: MRI image")
    print("  - Input Channel 1: Step 1 predictions (anatomical context)")
    print("=" * 60)
    
    # Create destination directory
    dst_dataset.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Find LDH samples
    print("\nStep 1: Finding LDH samples...")
    ldh_samples = find_ldh_samples(src_dataset)
    print(f"Found {len(ldh_samples)} LDH samples")
    
    if not ldh_samples:
        print("Error: No LDH samples found in source dataset")
        print("LDH samples should have naming pattern: sub-LDH*")
        sys.exit(1)
    
    # Step 2: Copy LDH images
    print("\nStep 2: Copying LDH images...")
    copy_ldh_images(src_dataset, dst_dataset, ldh_samples, jobs)
    
    # Step 3: Copy LDH ground truth labels
    print("\nStep 3: Copying LDH ground truth labels...")
    copy_ldh_labels(src_dataset, dst_dataset, ldh_samples, jobs)
    
    # Step 4: Run Step 1 inference
    if not args.skip_inference:
        print("\nStep 4: Running Step 1 inference on LDH images...")
        run_step1_inference(dst_dataset, nnunet_results, nnunet_raw, nnunet_preprocessed, jobs)
    else:
        print("\nStep 4: Skipping Step 1 inference (--skip-inference flag)")
    
    # Step 5: Create LDH-only binary labels
    print("\nStep 5: Creating LDH-only binary labels...")
    create_ldh_only_labels(dst_dataset, jobs)
    
    # Step 6: Create second input channel from Step 1 predictions
    print("\nStep 6: Creating second input channel from Step 1 predictions...")
    create_second_channel_from_step1(dst_dataset, jobs)
    
    # Step 7: Create test split
    print("\nStep 7: Creating test split...")
    create_test_split(dst_dataset, args.test_ratio)
    
    # Step 8: Create dataset.json
    print("\nStep 8: Creating dataset.json...")
    create_dataset_json(dst_dataset, resources)
    
    # Clean up temporary directories
    print("\nCleaning up temporary directories...")
    for tmp_dir in ['labelsGT', 'labelsStep1']:
        tmp_path = dst_dataset / tmp_dir
        if tmp_path.exists():
            # Keep these for debugging, but could remove in production
            # shutil.rmtree(tmp_path)
            print(f"  Keeping {tmp_dir} for reference")
    
    print("\n" + "=" * 60)
    print("Dataset 105 preparation completed!")
    print("=" * 60)
    print(f"\nDataset location: {dst_dataset}")
    print(f"Training samples: {len(list((dst_dataset / 'labelsTr').glob('*.nii.gz')))}")
    print(f"Test samples: {len(list((dst_dataset / 'labelsTs').glob('*.nii.gz')))}")
    print("\nNext steps:")
    print("  1. Run training: bash scripts/train.sh 105")
    print("  2. Or with custom trainer: CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh 105 0 nnUNetTrainer_LDH")


if __name__ == '__main__':
    main()

