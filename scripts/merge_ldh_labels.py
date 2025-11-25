import os
import glob
import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import subprocess

# Configuration
# TOTALSPINESEG_DATA should be set in environment or passed via train.sh
if 'TOTALSPINESEG_DATA' not in os.environ:
    # Fallback or error
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
FOLD = 0 # Assume fold 0 for inference

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Identify LDH files
    ldh_images = sorted(glob.glob(str(DATASET_LDH_IMAGES / "sub-LDH*_0000.nii.gz")))
    print(f"Found {len(ldh_images)} LDH images to process.")

    if not ldh_images:
        print("No LDH images found. Check paths.")
        return

    # Temporary output for Step 1 predictions
    temp_pred_dir = Path("temp_step1_preds_ldh")
    os.makedirs(temp_pred_dir, exist_ok=True)
    
    # 2. Run Step 1 Inference on LDH images using CLI
    print("Running Step 1 inference on LDH images via nnUNet CLI...")
    
    # Create temp input dir with symlinks
    temp_input_dir = Path("temp_ldh_input")
    if temp_input_dir.exists():
        import shutil
        shutil.rmtree(temp_input_dir)
    os.makedirs(temp_input_dir)
    
    for img in ldh_images:
        dst = temp_input_dir / os.path.basename(img)
        if not dst.exists():
            os.symlink(img, dst)
        
    # Construct CLI command
    # nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities
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
    
    # 3. Merge Labels
    print("Merging Step 1 predictions with Ground Truth LDH labels...")
    
    # Ensure output dir exists
    os.makedirs(DATASET_102_LABELS_TR, exist_ok=True)
    
    for img_path in tqdm(ldh_images):
        fname = os.path.basename(img_path)
        # Predictor outputs file with same name as input (including _0000 if present? No, usually removes it or keeps base)
        # nnUNetv2_predict behavior: if input is case_0000.nii.gz, output is case.nii.gz
        
        base_name = fname.replace("_0000.nii.gz", "")
        label_name = base_name + ".nii.gz"
        
        # Prediction Path
        pred_path = temp_pred_dir / label_name
        
        # GT Path (Dataset99 labels)
        # Dataset99 label name: sub-LDH..._T2w.nii.gz (without _0000)
        # The image file was sub-LDH..._T2w_0000.nii.gz
        gt_path = DATASET_LDH_LABELS / label_name 
        
        if not pred_path.exists():
            print(f"Warning: Prediction not found for {label_name} at {pred_path}")
            continue
            
        if not gt_path.exists():
            print(f"Warning: GT label not found for {label_name} at {gt_path}")
            continue
            
        # Load
        try:
            pred_nii = nib.load(pred_path)
            pred_data = pred_nii.get_fdata().astype(np.int32)
            
            gt_nii = nib.load(gt_path)
            gt_data = gt_nii.get_fdata().astype(np.int32)
        except Exception as e:
            print(f"Error loading files for {label_name}: {e}")
            continue
        
        # Merge Logic
        # Background: 0
        # Step 1 Predictions (Spine): Keep as is (1-9)
        # GT LDH (101): Map to 12
        
        final_data = pred_data.copy()
        
        # Map GT LDH (101) to 12
        # Note: If GT has other labels, they might be overwritten or ignored based on this logic.
        # Assuming data-ldh only has 101.
        ldh_mask = (gt_data == 101)
        final_data[ldh_mask] = 12
        
        # Save to Dataset 102 Labels
        out_nii = nib.Nifti1Image(final_data, pred_nii.affine, pred_nii.header)
        nib.save(out_nii, DATASET_102_LABELS_TR / label_name)

    print("Merge complete.")
    
    # Cleanup
    # import shutil
    # shutil.rmtree(temp_input_dir)
    # shutil.rmtree(temp_pred_dir)

if __name__ == "__main__":
    main()
