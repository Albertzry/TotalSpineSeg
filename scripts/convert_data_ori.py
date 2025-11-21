import os
import glob
import shutil
import re
import nibabel as nib
import numpy as np
from pathlib import Path

# Paths
DATA_ORI = Path("/opt/data/private/data_sum/bids/data_ori")
BIDS_OUT = Path("/opt/data/private/data_sum/bids/data-ldh")

# Ensure output dir exists
if BIDS_OUT.exists():
    shutil.rmtree(BIDS_OUT)
os.makedirs(BIDS_OUT, exist_ok=True)

images_dir = DATA_ORI / "images"
labels_dir = DATA_ORI / "labels"

image_files = sorted(glob.glob(str(images_dir / "*.nii.gz")))

print(f"Found {len(image_files)} images.")

for idx, img_path in enumerate(image_files):
    img_path = Path(img_path)
    fname = img_path.name
    
    # Find corresponding label
    lbl_path = labels_dir / fname
    if not lbl_path.exists():
        print(f"Label not found for {fname}, skipping.")
        continue
        
    # Extract Number and Date from filename
    # Format example: "100倪全金 20130610.nii.gz" -> 100, 20130610
    match = re.match(r"^(\d+).*?(\d+)\.nii\.gz$", fname)
    if match:
        number_part = match.group(1)
        date_part = match.group(2)
        # Construct ID: Number_Date
        new_id = f"{number_part}_{date_part}"
    else:
        print(f"Warning: Could not parse filename {fname}, using default naming.")
        new_id = f"ldh{idx+1:03d}"

    sub_id = f"sub-{new_id}"
    
    # Create BIDS directories
    sub_dir = BIDS_OUT / sub_id
    anat_dir = sub_dir / "anat"
    labels_iso_dir = BIDS_OUT / "derivatives" / "labels_iso" / sub_id / "anat"
    
    os.makedirs(anat_dir, exist_ok=True)
    os.makedirs(labels_iso_dir, exist_ok=True)
    
    # Copy/Process Image
    # Note: Adding _T2w suffix to match typical BIDS T2w anatomy
    dst_img = anat_dir / f"{sub_id}_T2w.nii.gz"
    shutil.copy(img_path, dst_img)
    
    # Process Label
    img = nib.load(lbl_path)
    data = img.get_fdata()
    
    # Remap 1 -> 101
    data = np.round(data).astype(np.int32)
    if 1 in data:
        data[data == 1] = 101
    
    new_img = nib.Nifti1Image(data, img.affine, img.header)
    # Include _T2w in label name to match image, and _label-spine_dseg suffix
    dst_lbl = labels_iso_dir / f"{sub_id}_T2w_label-spine_dseg.nii.gz"
    nib.save(new_img, dst_lbl)
    
    if (idx + 1) % 50 == 0:
        print(f"Processed {idx + 1} subjects")

print("Conversion complete.")
