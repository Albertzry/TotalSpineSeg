#!/bin/bash

# This script prepares custom datasets for the TotalSpineSeg model in nnUNetv2 structure.
# The script expects IMAGE_DIR and LABEL_DIR as the first two positional arguments.
# The script also accepts DATASET as the third positional argument to specify which dataset to prepare.
# It can be either 101, 102, 103 or all. If all is specified, it will prepare all datasets (101, 102, 103).
# By default, it will prepare datasets 101 and 102.
# The script also accepts -noaug parameter to not generate augmentations.

# The script expects the following environment variables to be set:
#   TOTALSPINESEG: The path to the TotalSpineSeg repository.
#   TOTALSPINESEG_DATA: The path to the TotalSpineSeg data folder.
#   TOTALSPINESEG_JOBS: The number of CPU cores to use. Default is the number of CPU cores available.
#
# IMPORTANT: Make sure to activate the conda environment before running this script:
#   conda activate tss

# BASH SETTINGS
# ======================================================================================================================

# Uncomment for full verbose
# set -v

# Immediately exit if error
set -e

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# SCRIPT STARTS HERE
# ======================================================================================================================

# Check if image and label directories are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: IMAGE_DIR and LABEL_DIR are required."
    echo "Usage: $0 IMAGE_DIR LABEL_DIR [DATASET] [-noaug]"
    echo "  IMAGE_DIR: Path to folder containing input images (.nii.gz or .nii files)"
    echo "  LABEL_DIR: Path to folder containing label/segmentation files (.nii.gz or .nii files)"
    echo "  DATASET: Optional. Can be 101, 102, 103, or all. Default is 101 102."
    echo "  -noaug: Optional. Skip data augmentation generation."
    exit 1
fi

IMAGE_DIR="$(realpath "$1")"
LABEL_DIR="$(realpath "$2")"

# Check if directories exist
if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: Image directory does not exist: $IMAGE_DIR"
    exit 1
fi

if [ ! -d "$LABEL_DIR" ]; then
    echo "Error: Label directory does not exist: $LABEL_DIR"
    exit 1
fi

# Set the datasets to work with - default is 101 102
if [[ -z $3 || $3 == 101 || $3 == all || $3 == -* ]]; then PREP_101=1; else PREP_101=0; fi
if [[ -z $3 || $3 == 102 || $3 == all || $3 == -* ]]; then PREP_102=1; else PREP_102=0; fi
if [[ $3 == 103 || $3 == all ]]; then PREP_103=1; else PREP_103=0; fi

# Set the augmentations to generate - default is to generate augmentations
if [[ $3 == -noaug || $4 == -noaug ]]; then NOAUG=1; else NOAUG=0; fi

# set TOTALSPINESEG and TOTALSPINESEG_DATA if not set
TOTALSPINESEG="$(realpath "${TOTALSPINESEG:-totalspineseg}")"
TOTALSPINESEG_DATA="$(realpath "${TOTALSPINESEG_DATA:-data}")"

# Set the path to the resources folder
resources="$TOTALSPINESEG"/totalspineseg/resources

# Get the number of CPUs
CORES=${SLURM_JOB_CPUS_PER_NODE:-$(lscpu -p | egrep -v '^#' | wc -l)}

# Set the number of jobs
JOBS=${TOTALSPINESEG_JOBS:-$CORES}

# Note: This script assumes the conda environment is already activated
# Make sure to run: conda activate tss (or your environment name) before running this script

# Set nnunet params
nnUNet_raw="$TOTALSPINESEG_DATA"/nnUNet/raw

SRC_DATASET=Dataset99_TotalSpineSeg

echo "=========================================="
echo "Preparing custom datasets from:"
echo "  Image directory: $IMAGE_DIR"
echo "  Label directory: $LABEL_DIR"
echo "=========================================="

echo "Make nnUNet raw folders"
mkdir -p "$nnUNet_raw"/$SRC_DATASET/imagesTr
mkdir -p "$nnUNet_raw"/$SRC_DATASET/labelsTr

# Create a temporary directory for renaming
TEMP_RENAME_DIR=$(mktemp -d)
trap "rm -rf $TEMP_RENAME_DIR" EXIT

echo "Rename files to remove Chinese characters"
# Function to sanitize filename: remove Chinese characters, keep numbers and dates
sanitize_filename() {
    local filename="$1"
    local basename_file=$(basename "$filename" .nii.gz)
    basename_file=$(basename "$basename_file" .nii)
    
    # Extract numbers from filename
    local numbers=$(echo "$basename_file" | grep -oE '[0-9]+' | tr '\n' '_' | sed 's/_$//')
    
    # If we found numbers, use them
    if [ -n "$numbers" ] && [ "$numbers" != "" ]; then
        local sanitized="case_${numbers}"
    else
        # If no numbers found, use md5sum to create a unique identifier
        sanitized="case_$(echo "$basename_file" | md5sum | cut -c1-12)"
    fi
    
    # Clean up: remove multiple consecutive underscores and trailing/leading underscores
    sanitized=$(echo "$sanitized" | sed 's/__*/_/g' | sed 's/^_\|_$//g')
    
    echo "$sanitized"
}

# Create mapping file to track original to new names
MAPPING_FILE="$nnUNet_raw"/$SRC_DATASET/filename_mapping.txt
echo "# Original filename -> New filename" > "$MAPPING_FILE"
echo "# Format: original_name|new_name" >> "$MAPPING_FILE"

# First pass: collect all image and label files and create mapping
echo "  Processing images and creating name mapping..."
shopt -s nullglob

# Process all files and create unique mappings
counter=1
processed_names=()

# Function to get sanitized name from mapping file or create new one
get_sanitized_name() {
    local original_name="$1"
    local sanitized
    
    # Check if already in mapping file
    if grep -q "^${original_name}|" "$MAPPING_FILE" 2>/dev/null; then
        sanitized=$(grep "^${original_name}|" "$MAPPING_FILE" | cut -d'|' -f2)
    else
        # Create new sanitized name
        sanitized=$(sanitize_filename "$original_name")
        
        # Ensure uniqueness
        while printf '%s\n' "${processed_names[@]}" | grep -q "^${sanitized}$"; do
            sanitized="${sanitized}_${counter}"
            counter=$((counter + 1))
        done
        
        processed_names+=("$sanitized")
        echo "${original_name}|${sanitized}" >> "$MAPPING_FILE"
    fi
    
    echo "$sanitized"
}

# Process images first
for img in "$IMAGE_DIR"/*.nii.gz "$IMAGE_DIR"/*.nii; do
    [ -f "$img" ] || continue
    basename_img=$(basename "$img" .nii.gz)
    basename_img=$(basename "$basename_img" .nii)
    get_sanitized_name "$basename_img" > /dev/null
done

# Process labels (they should have same base names as images)
for lbl in "$LABEL_DIR"/*.nii.gz "$LABEL_DIR"/*.nii; do
    [ -f "$lbl" ] || continue
    basename_lbl=$(basename "$lbl" .nii.gz)
    basename_lbl=$(basename "$basename_lbl" .nii)
    get_sanitized_name "$basename_lbl" > /dev/null
done

shopt -u nullglob

mapping_count=$(grep -v '^#' "$MAPPING_FILE" | grep -c '|' || echo "0")
echo "  Created mapping for $mapping_count unique files"

echo "Copy images and labels into the nnUNet dataset folder with renamed filenames"
# Copy images - add _0000 suffix for nnUNet format
echo "Copying images..."
count_images=0
shopt -s nullglob
# Handle .nii.gz files
for img in "$IMAGE_DIR"/*.nii.gz; do
    if [ -f "$img" ]; then
        basename_img=$(basename "$img" .nii.gz)
        sanitized_name=$(get_sanitized_name "$basename_img")
        if cp "$img" "$nnUNet_raw"/$SRC_DATASET/imagesTr/${sanitized_name}_0000.nii.gz 2>/dev/null; then
            count_images=$((count_images + 1))
        else
            echo "  Warning: Failed to copy $img" >&2
        fi
    fi
done
# Handle .nii files (but not .nii.gz)
for img in "$IMAGE_DIR"/*.nii; do
    if [ -f "$img" ] && [[ "$img" != *.nii.gz ]]; then
        basename_img=$(basename "$img" .nii)
        sanitized_name=$(get_sanitized_name "$basename_img")
        # Convert .nii to .nii.gz
        if gzip -c "$img" > "$nnUNet_raw"/$SRC_DATASET/imagesTr/${sanitized_name}_0000.nii.gz 2>/dev/null; then
            count_images=$((count_images + 1))
        else
            echo "  Warning: Failed to convert and copy $img" >&2
        fi
    fi
done
shopt -u nullglob
echo "  Copied $count_images images"

# Copy labels
echo "Copying labels..."
count_labels=0
shopt -s nullglob
# Handle .nii.gz files
for lbl in "$LABEL_DIR"/*.nii.gz; do
    if [ -f "$lbl" ]; then
        basename_lbl=$(basename "$lbl" .nii.gz)
        sanitized_name=$(get_sanitized_name "$basename_lbl")
        if cp "$lbl" "$nnUNet_raw"/$SRC_DATASET/labelsTr/${sanitized_name}.nii.gz 2>/dev/null; then
            count_labels=$((count_labels + 1))
        else
            echo "  Warning: Failed to copy $lbl" >&2
        fi
    fi
done
# Handle .nii files (but not .nii.gz)
for lbl in "$LABEL_DIR"/*.nii; do
    if [ -f "$lbl" ] && [[ "$lbl" != *.nii.gz ]]; then
        basename_lbl=$(basename "$lbl" .nii)
        sanitized_name=$(get_sanitized_name "$basename_lbl")
        # Convert .nii to .nii.gz
        if gzip -c "$lbl" > "$nnUNet_raw"/$SRC_DATASET/labelsTr/${sanitized_name}.nii.gz 2>/dev/null; then
            count_labels=$((count_labels + 1))
        else
            echo "  Warning: Failed to convert and copy $lbl" >&2
        fi
    fi
done
shopt -u nullglob
echo "  Copied $count_labels labels"

echo "Remove images without segmentation and segmentation without images"
# Match images and labels by base name (without _0000 and extensions)
for f in "$nnUNet_raw"/$SRC_DATASET/imagesTr/*.nii.gz; do
    if [ -f "$f" ]; then
        basename_img=$(basename "$f" _0000.nii.gz)
        if [ ! -f "$nnUNet_raw"/$SRC_DATASET/labelsTr/${basename_img}.nii.gz ]; then
            echo "  Removing image without matching label: $(basename "$f")"
            rm "$f"
        fi
    fi
done

for f in "$nnUNet_raw"/$SRC_DATASET/labelsTr/*.nii.gz; do
    if [ -f "$f" ]; then
        basename_lbl=$(basename "$f" .nii.gz)
        if [ ! -f "$nnUNet_raw"/$SRC_DATASET/imagesTr/${basename_lbl}_0000.nii.gz ]; then
            echo "  Removing label without matching image: $(basename "$f")"
            rm "$f"
        fi
    fi
done

echo "Convert 4D images to 3D"
totalspineseg_average4d -i "$nnUNet_raw"/$SRC_DATASET/imagesTr -o "$nnUNet_raw"/$SRC_DATASET/imagesTr -r -w $JOBS

echo "Transform images to canonical space"
totalspineseg_reorient_canonical -i "$nnUNet_raw"/$SRC_DATASET/imagesTr -o "$nnUNet_raw"/$SRC_DATASET/imagesTr -r -w $JOBS

echo "Resample images to 1x1x1mm"
totalspineseg_resample -i "$nnUNet_raw"/$SRC_DATASET/imagesTr -o "$nnUNet_raw"/$SRC_DATASET/imagesTr -r -w $JOBS

echo "Transform labels to images space"
totalspineseg_transform_seg2image -i "$nnUNet_raw"/$SRC_DATASET/imagesTr -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/$SRC_DATASET/labelsTr -r -w $JOBS

echo "Map binary labels (0,1) to TotalSpineSeg label system"
echo "  Mapping label 1 (lumbar IVD) to label 95 (L4-L5 disc) for step1 training"
# Create a mapping file: label 1 -> 95 (L4-L5 lumbar disc)
# This will map to output label 1 (disc) in step1
BINARY_TO_TSS_MAP="$nnUNet_raw"/$SRC_DATASET/binary_to_tss_map.json
cat > "$BINARY_TO_TSS_MAP" << 'EOFMAP'
{
    "1": 95
}
EOFMAP

# Apply the mapping: label 1 -> 95 (L4-L5 disc)
totalspineseg_map_labels -m "$BINARY_TO_TSS_MAP" -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/$SRC_DATASET/labelsTr -r -w $JOBS
echo "  Label mapping completed: 1 -> 95 (L4-L5 lumbar disc, maps to disc label 1 in step1)"

echo "Making test folders and moving 10% of the data to test folders"
mkdir -p "$nnUNet_raw"/$SRC_DATASET/imagesTs
mkdir -p "$nnUNet_raw"/$SRC_DATASET/labelsTs

# Move 10% of the data to test folders (simplified version for custom data)
files=($(for f in "$nnUNet_raw"/$SRC_DATASET/labelsTr/*.nii.gz; do basename "${f/.nii.gz/}"; done))
if [ ${#files[@]} -gt 0 ]; then
    files_shuf=($(shuf -e "${files[@]}"))
    files_10p=(${files_shuf[@]:0:$((${#files_shuf[@]} * 10 / 100))})
    for f in ${files_10p[@]}; do
        mv "$nnUNet_raw"/$SRC_DATASET/imagesTr/${f}_0000.nii.gz "$nnUNet_raw"/$SRC_DATASET/imagesTs/ 2>/dev/null || true
        mv "$nnUNet_raw"/$SRC_DATASET/labelsTr/${f}.nii.gz "$nnUNet_raw"/$SRC_DATASET/labelsTs/ 2>/dev/null || true
    done
    echo "  Moved ${#files_10p[@]} files to test set"
    
    # Apply label mapping to test set
    if [ -d "$nnUNet_raw"/$SRC_DATASET/labelsTs ] && [ "$(ls -A "$nnUNet_raw"/$SRC_DATASET/labelsTs 2>/dev/null)" ]; then
        totalspineseg_map_labels -m "$BINARY_TO_TSS_MAP" -s "$nnUNet_raw"/$SRC_DATASET/labelsTs -o "$nnUNet_raw"/$SRC_DATASET/labelsTs -r -w $JOBS
    fi
fi

if [ $NOAUG -eq 0 ]; then
    echo "Generate augmentations"
    # For custom data with only label 95 (lumbar IVD), use only that label class for augmentation
    # This avoids processing non-existent label classes which can cause slowdowns
    # Limit workers to avoid resource contention (max 16 workers)
    AUG_JOBS=$(( JOBS > 16 ? 16 : JOBS ))
    echo "  Using $AUG_JOBS workers for augmentation (reduced from $JOBS to avoid resource contention)"
    totalspineseg_augment -i "$nnUNet_raw"/$SRC_DATASET/imagesTr -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/Dataset100_TotalSpineSeg_Aug/imagesTr -g "$nnUNet_raw"/Dataset100_TotalSpineSeg_Aug/labelsTr --labels2image --seg-classes 95 -r -w $AUG_JOBS
    totalspineseg_cpdir "$nnUNet_raw"/$SRC_DATASET "$nnUNet_raw"/Dataset100_TotalSpineSeg_Aug -p "*Ts/*.nii.gz" -r -w $JOBS
    totalspineseg_transform_seg2image -i "$nnUNet_raw"/Dataset100_TotalSpineSeg_Aug/imagesTr -s "$nnUNet_raw"/Dataset100_TotalSpineSeg_Aug/labelsTr -o "$nnUNet_raw"/Dataset100_TotalSpineSeg_Aug/labelsTr -r -w $JOBS
    SRC_DATASET=Dataset100_TotalSpineSeg_Aug
fi

if [ $PREP_101 -eq 1 ]; then
    echo "Generate nnUNet dataset 101 (step 1)"
    totalspineseg_cpdir "$nnUNet_raw"/$SRC_DATASET "$nnUNet_raw"/Dataset101_TotalSpineSeg_step1 -p "imagesT*/*.nii.gz" -r -w $JOBS
    totalspineseg_map_labels -m "$resources"/labels_maps/nnunet_step1.json -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/Dataset101_TotalSpineSeg_step1/labelsTr -r -w $JOBS
    totalspineseg_map_labels -m "$resources"/labels_maps/nnunet_step1.json -s "$nnUNet_raw"/$SRC_DATASET/labelsTs -o "$nnUNet_raw"/Dataset101_TotalSpineSeg_step1/labelsTs -r -w $JOBS
    # Copy the dataset.json file and update the number of training samples
    jq --arg numTraining "$(ls "$nnUNet_raw"/Dataset101_TotalSpineSeg_step1/labelsTr | wc -l)" '.numTraining = ($numTraining|tonumber)' "$resources"/datasets/dataset_step1.json > "$nnUNet_raw"/Dataset101_TotalSpineSeg_step1/dataset.json
fi

if [ $PREP_102 -eq 1 ]; then
    echo "Generate nnUNet dataset 102 (step 2)"
    totalspineseg_cpdir "$nnUNet_raw"/$SRC_DATASET "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2 -p "imagesT*/*.nii.gz" -r -w $JOBS
    # This make a copy of the labelsTr then later we will map the labels so the odds and evens IVDs are switched
    totalspineseg_cpdir "$nnUNet_raw"/$SRC_DATASET "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2 -p "imagesTr/*.nii.gz" -t "_0000.nii.gz:_o2e_0000.nii.gz" -r -w $JOBS
    # This will map the labels to the second input channel
    totalspineseg_extract_alternate -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/imagesTr --labels 63-100 --prioratize-labels 63 65 67 72 74 76 78 80 82 92 94 --output-seg-suffix _0001 -r -w $JOBS -r
    # This will map the labels to the extra images second input channel so the odd and even IVDs are switched
    totalspineseg_extract_alternate -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/imagesTr --labels 63-100 --prioratize-labels 64 66 71 73 75 77 79 81 91 93 95 --output-seg-suffix _o2e_0001 -r -w $JOBS -r
    # This will map the labels to the second input channel for the test set
    totalspineseg_extract_alternate -s "$nnUNet_raw"/$SRC_DATASET/labelsTs -o "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/imagesTs --labels 63-100 --prioratize-labels 63 65 67 72 74 76 78 80 82 92 94 --output-seg-suffix _0001 -r -w $JOBS -r
    totalspineseg_map_labels -m "$resources"/labels_maps/nnunet_step2.json -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/labelsTr -r -w $JOBS
    # This will map the extra images labels so the odd and even IVDs are switched
    totalspineseg_map_labels -m "$resources"/labels_maps/nnunet_step2_o2e.json -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/labelsTr --output-seg-suffix _o2e -r -w $JOBS
    totalspineseg_map_labels -m "$resources"/labels_maps/nnunet_step2.json -s "$nnUNet_raw"/$SRC_DATASET/labelsTs -o "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/labelsTs -r -w $JOBS
    # Copy the dataset.json file and update the number of training samples
    jq --arg numTraining "$(ls "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/labelsTr | wc -l)" '.numTraining = ($numTraining|tonumber)' "$resources"/datasets/dataset_step2.json > "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/dataset.json
fi

if [ $PREP_103 -eq 1 ]; then
    echo "Generate nnUNet dataset 103 (full)"
    totalspineseg_cpdir "$nnUNet_raw"/$SRC_DATASET "$nnUNet_raw"/Dataset103_TotalSpineSeg_full -p "imagesT*/*.nii.gz" -r -w $JOBS
    totalspineseg_map_labels -m "$resources"/labels_maps/nnunet_full.json -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/Dataset103_TotalSpineSeg_full/labelsTr -r -w $JOBS
    totalspineseg_map_labels -m "$resources"/labels_maps/nnunet_full.json -s "$nnUNet_raw"/$SRC_DATASET/labelsTs -o "$nnUNet_raw"/Dataset103_TotalSpineSeg_full/labelsTs -r -w $JOBS
    # Copy the dataset.json file and update the number of training samples
    jq --arg numTraining "$(ls "$nnUNet_raw"/Dataset103_TotalSpineSeg_full/labelsTr | wc -l)" '.numTraining = ($numTraining|tonumber)' "$resources"/datasets/dataset_full.json > "$nnUNet_raw"/Dataset103_TotalSpineSeg_full/dataset.json
fi

echo "=========================================="
echo "Dataset preparation completed!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Label mapping: 1 (lumbar IVD) -> 95 (L4-L5 disc)"
echo "  - In step1, label 95 maps to output label 1 (disc)"
echo "  - Dataset 101 is ready for step1 training"
if [ $PREP_101 -eq 1 ]; then
    echo "  - Dataset 101: $(ls "$nnUNet_raw"/Dataset101_TotalSpineSeg_step1/labelsTr 2>/dev/null | wc -l) training samples"
fi
if [ $PREP_102 -eq 1 ]; then
    echo "  - Dataset 102: $(ls "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/labelsTr 2>/dev/null | wc -l) training samples"
fi
if [ $PREP_103 -eq 1 ]; then
    echo "  - Dataset 103: $(ls "$nnUNet_raw"/Dataset103_TotalSpineSeg_full/labelsTr 2>/dev/null | wc -l) training samples"
fi

