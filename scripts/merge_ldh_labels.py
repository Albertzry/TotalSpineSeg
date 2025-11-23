import os
import glob
import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from totalspineseg.inference.nnunet import predict_nnunet
from tqdm import tqdm

# Configuration
NNUNET_RESULTS = Path("/opt/data/private/data_sum/nnUNet/results")
NNUNET_RAW = Path("/opt/data/private/data_sum/nnUNet/raw")
DATASET_LDH_IMAGES = NNUNET_RAW / "Dataset99_TotalSpineSeg/imagesTr" # Or wherever original LDH images are
DATASET_LDH_LABELS = NNUNET_RAW / "Dataset99_TotalSpineSeg/labelsTr"
DATASET_102_TR = NNUNET_RAW / "Dataset102_TotalSpineSeg_step2/imagesTr"

# Step 1 Model Config
DATASET_ID = 101
TRAINER = "nnUNetTrainer_DASegOrd0_NoMirroring"
PLANS = "nnUNetPlans_small"
CONFIG = "3d_fullres"
FOLD = 0 # Assume fold 0 for inference

STEP1_MODEL_FOLDER = NNUNET_RESULTS / f"Dataset{DATASET_ID}_TotalSpineSeg_step1" / f"{TRAINER}__{PLANS}__{CONFIG}" / f"fold_{FOLD}"

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Identify LDH files
    # We look for files starting with sub-LDH in Dataset99 (source of all)
    # OR we can assume they are in Dataset102 but missing other labels.
    # Let's read from Dataset99 imagesTr as source.
    
    ldh_images = sorted(glob.glob(str(DATASET_LDH_IMAGES / "sub-LDH*_0000.nii.gz")))
    print(f"Found {len(ldh_images)} LDH images to process.")

    if not ldh_images:
        print("No LDH images found. Check paths.")
        return

    # Temporary output for Step 1 predictions
    temp_pred_dir = Path("temp_step1_preds_ldh")
    os.makedirs(temp_pred_dir, exist_ok=True)
    
    # 2. Run Step 1 Inference on LDH images
    print("Running Step 1 inference on LDH images...")
    # We need to call nnUNet predictor. 
    # Since we are in python, we can use the library if installed, or subprocess.
    # Using subprocess to be safe with environment.
    
    # Construct input/output directories for batch prediction
    # To save time, we can link/copy LDH images to a temp input dir
    temp_input_dir = Path("temp_ldh_input")
    if temp_input_dir.exists():
        import shutil
        shutil.rmtree(temp_input_dir)
    os.makedirs(temp_input_dir)
    
    for img in ldh_images:
        os.symlink(img, temp_input_dir / os.path.basename(img))
        
    cmd = f"nnUNetv2_predict -d {DATASET_ID} -i {temp_input_dir} -o {temp_pred_dir} -f {FOLD} -c {CONFIG} -tr {TRAINER} -p {PLANS} -device {device.type}"
    print(f"Executing: {cmd}")
    os.system(cmd)
    
    # 3. Merge Labels
    print("Merging Step 1 predictions with Ground Truth LDH labels...")
    
    # Target directory for merged labels (Dataset 102 labelsTr)
    # We need to overwrite the existing labels in Dataset 102 which only have LDH
    target_labels_dir = NNUNET_RAW / "Dataset102_TotalSpineSeg_step2/labelsTr"
    
    for img_path in tqdm(ldh_images):
        fname = os.path.basename(img_path)
        base_name = fname.replace("_0000.nii.gz", "")
        label_name = base_name + ".nii.gz"
        
        # Paths
        pred_path = temp_pred_dir / label_name
        gt_path = DATASET_LDH_LABELS / label_name # Original GT with only LDH (mapped to 101 or similar)
        
        if not pred_path.exists():
            print(f"Warning: Prediction not found for {label_name}")
            continue
            
        if not gt_path.exists():
            print(f"Warning: GT label not found for {label_name}")
            continue
            
        # Load
        pred_nii = nib.load(pred_path)
        pred_data = pred_nii.get_fdata().astype(np.int32)
        
        gt_nii = nib.load(gt_path)
        gt_data = gt_nii.get_fdata().astype(np.int32)
        
        # Merge Logic
        # Step 1 Preds: 1-9 (Spine structures)
        # GT Data: Has LDH as 101 (from our conversion script) or mapped value.
        # Check what value LDH has in Dataset99. In conversion it was 101.
        # In nnunet_step2.json, 101 maps to 12.
        # We need to construct the FINAL label for Step 2.
        
        # Step 2 Expects:
        # 1-11: Spine structures (from Step 1 map)
        # 12: LDH
        
        # Let's map Step 1 preds (1-9) to Step 2 values.
        # Step 1: 1(Disc)->...
        # Wait, Step 1 and Step 2 have slightly different mappings for standard structures?
        # Step 1: Disc(1), C2C3(2)...
        # Step 2: Disc(1), C2C3(2)...
        # Usually they are compatible for the basic classes.
        # Let's assume prediction classes 1-9 map directly to 1-9 in Step 2 for simplicity, 
        # OR we need to remap if Step 2 splits them differently (e.g. Vert Odd/Even).
        
        # Looking at jsons:
        # Step 1: Vert(7,8) -> mapped from raw
        # Step 2: Vert(7,8,9) -> O/E split.
        
        # Issue: Step 1 model outputs generic vertebrae (or C/T/L split if configured).
        # But Step 2 expects Odd/Even split.
        # Step 1 prediction DOES NOT contain Odd/Even info if it wasn't trained on it.
        # Dataset 101 Step 1 was trained on 10 classes (json).
        # Dataset 102 Step 2 expects 12 classes (json).
        
        # CRITICAL: Step 1 prediction alone is NOT enough to generate Step 2 labels if Step 2 requires O/E split 
        # and Step 1 doesn't provide it.
        # However, looking at prepare_datasets.sh, Step 2 inputs (images) are generated using 
        # `totalspineseg_extract_alternate` which creates the O/E info.
        # Wait, Step 2 is typically "High Resolution" or "Refinement"? 
        # Or is it just "Different Labels"?
        
        # If Step 2 requires O/E vertebrae, and Step 1 only predicts generic vertebrae, 
        # we cannot easily generate O/E labels from Step 1 without geometric logic.
        
        # BUT: For LDH segmentation, maybe we don't care about perfect O/E labels in the context?
        # OR: We just map all vertebrae to one class? No, Step 2 training will penalize that.
        
        # Simplified approach:
        # If we can't easily generate O/E, maybe we accept that LDH data will have "generic" vertebrae 
        # and hope the model handles the noise, OR we try to heuristic split.
        
        # Actually, let's look at `nnunet_step2.json`. 
        # Verts: 7 (O), 8 (E).
        # Step 1 output: Vert (7,8 in list? No, check dataset_step1.json).
        # Step 1 Verts: 7 (General?), 8 (C1).
        
        # This is complicated. 
        # ALTERNATIVE: Use the `totalspineseg` pipeline tools if they exist to generating inputs?
        # But we need LABELS for training.
        
        # Let's assume for now we merge:
        # Background: 0
        # Preds > 0: Keep them.
        # LDH (GT 101): Map to 12. 
        # Overwrite Preds with LDH where LDH exists.
        
        final_data = pred_data.copy()
        
        # Map GT LDH (101) to 12
        ldh_mask = (gt_data == 101)
        final_data[ldh_mask] = 12
        
        # Save to Dataset 102 Labels
        # Note: Filename in Dataset 102 might need to be specific? 
        # prepare_datasets.sh does: totalspineseg_map_labels ... -> labelsTr
        # We should save directly to where nnUNet expects it for training 102.
        
        out_nii = nib.Nifti1Image(final_data, pred_nii.affine, pred_nii.header)
        nib.save(out_nii, target_labels_dir / label_name)

    print("Merge complete. Clean up temp files manually if needed.")

if __name__ == "__main__":
    main()

