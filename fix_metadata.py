#!/usr/bin/env python3
"""
Fix metadata mismatches between _0000.nii.gz and _0001.nii.gz files for nnUNet Dataset102.
This script ensures that both channels have identical origin, spacing, and direction.
"""

import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import argparse
import multiprocessing as mp
from functools import partial

def _fix_metadata_wrapper(img_0000_path, overwrite=True):
    """
    Wrapper function for multiprocessing.
    Returns: (status, error_message or None)
    """
    img_0001_path = img_0000_path.parent / img_0000_path.name.replace('_0000.nii.gz', '_0001.nii.gz')
    
    if not img_0001_path.exists():
        return ('missing', img_0000_path.name)
    
    try:
        fix_metadata(img_0000_path, img_0001_path, overwrite=overwrite)
        return ('success', None)
    except Exception as e:
        return ('error', f"{img_0001_path.name}: {str(e)}")

def fix_metadata(image_0000_path, image_0001_path, overwrite=True):
    """
    Fix metadata of _0001 file to match _0000 file exactly.
    
    Parameters
    ----------
    image_0000_path : Path
        Path to the _0000.nii.gz file (reference)
    image_0001_path : Path
        Path to the _0001.nii.gz file (to be fixed)
    overwrite : bool
        Whether to overwrite the existing _0001 file
    """
    # Load both images
    img_0000 = nib.load(image_0000_path)
    img_0001 = nib.load(image_0001_path)
    
    # Get data from _0001 (the mask data we want to keep)
    data_0001 = np.asanyarray(img_0001.dataobj).astype(np.uint8)
    
    # Get the reference affine and header from _0000
    ref_affine = img_0000.affine.copy()
    ref_header = img_0000.header.copy()
    
    # Create new _0001 image with _0000's metadata
    fixed_img_0001 = nib.Nifti1Image(data_0001, ref_affine, ref_header)
    
    # Ensure proper data type
    fixed_img_0001.set_data_dtype(np.uint8)
    
    # Explicitly copy qform and sform codes to ensure exact match
    qform_code = int(img_0000.header['qform_code'])
    sform_code = int(img_0000.header['sform_code'])
    
    fixed_img_0001.set_qform(ref_affine, code=qform_code)
    fixed_img_0001.set_sform(ref_affine, code=sform_code)
    
    # Save the fixed image
    if overwrite:
        nib.save(fixed_img_0001, image_0001_path)
        return True
    else:
        backup_path = image_0001_path.parent / (image_0001_path.stem.replace('.nii', '_backup.nii') + '.gz')
        nib.save(img_0001, backup_path)
        nib.save(fixed_img_0001, image_0001_path)
        return True
    
    return False

def main():
    parser = argparse.ArgumentParser(
        description='Fix metadata mismatches in nnUNet Dataset102 multi-channel images'
    )
    parser.add_argument(
        '--dataset-path', '-d', type=str, required=True,
        help='Path to Dataset102_TotalSpineSeg_step2 folder'
    )
    parser.add_argument(
        '--no-backup', action='store_true',
        help='Do not create backups (overwrites directly)'
    )
    parser.add_argument(
        '--max-workers', '-w', type=int, default=mp.cpu_count(),
        help=f'Number of parallel workers (default: {mp.cpu_count()})'
    )
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    overwrite = args.no_backup
    max_workers = args.max_workers
    
    # Check if dataset path exists
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return
    
    # Process imagesTr folder
    images_tr_path = dataset_path / 'imagesTr'
    if not images_tr_path.exists():
        print(f"Error: imagesTr folder not found: {images_tr_path}")
        return
    
    # Find all _0000.nii.gz files
    image_0000_files = sorted(images_tr_path.glob('*_0000.nii.gz'))
    
    if len(image_0000_files) == 0:
        print(f"No _0000.nii.gz files found in {images_tr_path}")
        return
    
    print(f"Found {len(image_0000_files)} image pairs to process")
    print(f"Workers: {max_workers}")
    print(f"Backup: {'No' if overwrite else 'Yes'}")
    print("")
    
    # Process training images with multiprocessing
    print("Processing training set...")
    results = process_map(
        partial(_fix_metadata_wrapper, overwrite=overwrite),
        image_0000_files,
        max_workers=max_workers,
        chunksize=1,
        desc="Fixing metadata"
    )
    
    # Count results
    fixed_count = sum(1 for status, _ in results if status == 'success')
    error_count = sum(1 for status, _ in results if status == 'error')
    missing_count = sum(1 for status, _ in results if status == 'missing')
    
    # Print errors if any
    for status, msg in results:
        if status == 'error':
            print(f"\nError: {msg}")
    
    # Also process imagesTs if it exists
    images_ts_path = dataset_path / 'imagesTs'
    if images_ts_path.exists():
        image_0000_files_ts = sorted(images_ts_path.glob('*_0000.nii.gz'))
        
        if len(image_0000_files_ts) > 0:
            print(f"\nProcessing test set: {len(image_0000_files_ts)} image pairs")
            
            results_ts = process_map(
                partial(_fix_metadata_wrapper, overwrite=overwrite),
                image_0000_files_ts,
                max_workers=max_workers,
                chunksize=1,
                desc="Fixing test metadata"
            )
            
            # Count test results
            fixed_count += sum(1 for status, _ in results_ts if status == 'success')
            error_count += sum(1 for status, _ in results_ts if status == 'error')
            missing_count += sum(1 for status, _ in results_ts if status == 'missing')
            
            # Print errors if any
            for status, msg in results_ts:
                if status == 'error':
                    print(f"\nError: {msg}")
    
    print("\n" + "="*50)
    print(f"Summary:")
    print(f"  Fixed: {fixed_count}")
    print(f"  Missing _0001 files: {missing_count}")
    print(f"  Errors: {error_count}")
    print("="*50)
    
    if fixed_count > 0:
        print("\nMetadata has been fixed. You can now run:")
        print(f"  nnUNetv2_extract_fingerprint -d 102 -np 8")

if __name__ == '__main__':
    main()

