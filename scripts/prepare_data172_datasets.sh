#!/bin/bash

# This script prepares data_172 datasets for the TotalSpineSeg model in nnUNetv2 structure.
# The script accepts DATASET as the first positional argument to specify which dataset to prepare.
# It can be either 101, 102, 103 or all. If all is specified, it will prepare all datasets (101, 102, 103).
# By default, it will prepare datasets 101 and 102.
# The script also accepts -noaug parameter to not generate augmentations.
#
# Usage:
#   bash prepare_data172_datasets.sh [DATASET] [-noaug]
#
# Examples:
#   bash prepare_data172_datasets.sh              # Prepare 101 and 102 (default)
#   bash prepare_data172_datasets.sh 101          # Prepare only 101
#   bash prepare_data172_datasets.sh 102          # Prepare only 102
#   bash prepare_data172_datasets.sh all          # Prepare all (101, 102, 103)
#   bash prepare_data172_datasets.sh 101 -noaug   # Prepare 101 without augmentation
#
# The script expects the following environment variables to be set:
#   TOTALSPINESEG: The path to the TotalSpineSeg repository.
#   TOTALSPINESEG_DATA: The path to the data_172 folder (containing MR/ and Mask/ subdirectories).
#   TOTALSPINESEG_JOBS: The number of CPU cores to use. Default is the number of CPU cores available.
#                       NOTE: Will be automatically capped at 12 to prevent resource exhaustion.
#
# Expected data structure:
#   $TOTALSPINESEG_DATA/
#   ├── MR/         (containing Case*.nii.gz files)
#   ├── Mask/       (containing mask_case*.nii.gz files)
#   └── nnUNet/     (will be created by the script)
#       ├── raw/
#       ├── preprocessed/
#       └── results/
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

# Set the datasets to work with - default is 101 102
if [[ -z $1 || $1 == 101 || $1 == all || $1 == -* ]]; then PREP_101=1; else PREP_101=0; fi
if [[ -z $1 || $1 == 102 || $1 == all || $1 == -* ]]; then PREP_102=1; else PREP_102=0; fi
if [[ $1 == 103 || $1 == all ]]; then PREP_103=1; else PREP_103=0; fi

# Set the augmentations to generate - default is to generate augmentations
if [[ $1 == -noaug || $2 == -noaug ]]; then NOAUG=1; else NOAUG=0; fi

# set TOTALSPINESEG and TOTALSPINESEG_DATA if not set
TOTALSPINESEG="$(realpath "${TOTALSPINESEG:-totalspineseg}")"
TOTALSPINESEG_DATA="$(realpath "${TOTALSPINESEG_DATA:-data}")"

# Set the paths - TOTALSPINESEG_DATA directly contains MR/ and Mask/ subdirectories
DATA_DIR="$TOTALSPINESEG_DATA"
IMAGE_DIR="$DATA_DIR/MR"
LABEL_DIR="$DATA_DIR/Mask"

# Check if directories exist
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory does not exist: $DATA_DIR"
    echo ""
    echo "Current TOTALSPINESEG_DATA: $TOTALSPINESEG_DATA"
    echo ""
    echo "Please ensure TOTALSPINESEG_DATA is set correctly:"
    echo "  export TOTALSPINESEG_DATA=\"/path/to/data_172\""
    exit 1
fi

if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: MR directory does not exist: $IMAGE_DIR"
    echo ""
    echo "Please ensure your data folder has the correct structure:"
    echo "  $DATA_DIR/"
    echo "  ├── MR/         (containing Case*.nii.gz files)"
    echo "  └── Mask/       (containing mask_case*.nii.gz files)"
    exit 1
fi

if [ ! -d "$LABEL_DIR" ]; then
    echo "Error: Mask directory does not exist: $LABEL_DIR"
    echo ""
    echo "Please ensure your data folder has the correct structure:"
    echo "  $DATA_DIR/"
    echo "  ├── MR/         (containing Case*.nii.gz files)"
    echo "  └── Mask/       (containing mask_case*.nii.gz files)"
    exit 1
fi

# Set the path to the resources folder
resources="$TOTALSPINESEG"/totalspineseg/resources

# Maximum parallel jobs limit
# Limiting to 12 to prevent resource exhaustion and ensure stable processing
MAX_PARALLEL_JOBS=12

# Get the number of CPUs
CORES=${SLURM_JOB_CPUS_PER_NODE:-$(lscpu -p | egrep -v '^#' | wc -l)}

# Set the number of jobs (max 12)
JOBS=${TOTALSPINESEG_JOBS:-$CORES}
JOBS=$(( JOBS > MAX_PARALLEL_JOBS ? MAX_PARALLEL_JOBS : JOBS ))
JOBS=$(( JOBS < 1 ? 1 : JOBS ))

# Note: This script assumes the conda environment is already activated
# Make sure to run: conda activate tss (or your environment name) before running this script

# Set nnunet params
nnUNet_raw="$TOTALSPINESEG_DATA"/nnUNet/raw

SRC_DATASET=Dataset99_TotalSpineSeg

echo ""
echo "=========================================="
echo "Preparing data_172 datasets"
echo "=========================================="
echo "Environment:"
echo "  TOTALSPINESEG: $TOTALSPINESEG"
echo "  TOTALSPINESEG_DATA: $TOTALSPINESEG_DATA"
echo ""
echo "Resource Allocation:"
echo "  Available CPU cores: ${CORES}"
echo "  MAX_PARALLEL_JOBS limit: ${MAX_PARALLEL_JOBS} (enforced)"
echo "  JOBS (workers): ${JOBS}"
echo ""
echo "Data directories:"
echo "  Data directory: $DATA_DIR"
echo "  Image directory: $IMAGE_DIR"
echo "  Label directory: $LABEL_DIR"
echo ""
echo "Datasets to prepare:"
if [ $PREP_101 -eq 1 ]; then echo "  - Dataset 101 (step1)"; fi
if [ $PREP_102 -eq 1 ]; then echo "  - Dataset 102 (step2)"; fi
if [ $PREP_103 -eq 1 ]; then echo "  - Dataset 103 (full)"; fi
if [ $NOAUG -eq 1 ]; then echo "  - Augmentation: DISABLED"; else echo "  - Augmentation: ENABLED"; fi
echo "=========================================="
echo ""

echo "Make nnUNet raw folders"
mkdir -p "$nnUNet_raw"/$SRC_DATASET/imagesTr
mkdir -p "$nnUNet_raw"/$SRC_DATASET/labelsTr

echo "Copy images and labels into the nnUNet dataset folder"
echo "  Matching files: Case*.nii.gz <-> mask_case*.nii.gz"

# Function to extract case number from filename
get_case_number() {
    local filename="$1"
    # Extract number from Case10.nii.gz or mask_case10.nii.gz
    local case_num=$(echo "$filename" | sed -E 's/^(Case|mask_case)([0-9]+)\.nii\.gz$/\2/')
    echo "$case_num"
}

# Copy images - add _0000 suffix for nnUNet format
echo "Copying images..."
count_images=0
shopt -s nullglob
for img in "$IMAGE_DIR"/Case*.nii.gz "$IMAGE_DIR"/Case*.nii; do
    if [ -f "$img" ]; then
        basename_img=$(basename "$img" .nii.gz)
        basename_img=$(basename "$basename_img" .nii)
        # Convert Case10 to case_10 format for nnUNet
        case_num=$(get_case_number "$(basename "$img")")
        if [ -n "$case_num" ]; then
            cp "$img" "$nnUNet_raw"/$SRC_DATASET/imagesTr/case_${case_num}_0000.nii.gz 2>/dev/null || \
            gzip -c "$img" > "$nnUNet_raw"/$SRC_DATASET/imagesTr/case_${case_num}_0000.nii.gz 2>/dev/null
            if [ -f "$nnUNet_raw"/$SRC_DATASET/imagesTr/case_${case_num}_0000.nii.gz ]; then
                count_images=$((count_images + 1))
            fi
        fi
    fi
done
shopt -u nullglob
echo "  Copied $count_images images"

# Copy labels
echo "Copying labels..."
count_labels=0
shopt -s nullglob
for lbl in "$LABEL_DIR"/mask_case*.nii.gz "$LABEL_DIR"/mask_case*.nii; do
    if [ -f "$lbl" ]; then
        basename_lbl=$(basename "$lbl" .nii.gz)
        basename_lbl=$(basename "$basename_lbl" .nii)
        # Convert mask_case10 to case_10 format for nnUNet
        case_num=$(get_case_number "$(basename "$lbl")")
        if [ -n "$case_num" ]; then
            cp "$lbl" "$nnUNet_raw"/$SRC_DATASET/labelsTr/case_${case_num}.nii.gz 2>/dev/null || \
            gzip -c "$lbl" > "$nnUNet_raw"/$SRC_DATASET/labelsTr/case_${case_num}.nii.gz 2>/dev/null
            if [ -f "$nnUNet_raw"/$SRC_DATASET/labelsTr/case_${case_num}.nii.gz ]; then
                count_labels=$((count_labels + 1))
            fi
        fi
    fi
done
shopt -u nullglob
echo "  Copied $count_labels labels"

echo "Remove images without segmentation and segmentation without images"
# Match images and labels by case number
for f in "$nnUNet_raw"/$SRC_DATASET/imagesTr/*.nii.gz; do
    if [ -f "$f" ]; then
        # Extract case number from case_10_0000.nii.gz
        case_num=$(basename "$f" | sed -E 's/case_([0-9]+)_0000\.nii\.gz/\1/')
        if [ ! -f "$nnUNet_raw"/$SRC_DATASET/labelsTr/case_${case_num}.nii.gz ]; then
            echo "  Removing image without matching label: $(basename "$f")"
            rm "$f"
        fi
    fi
done

for f in "$nnUNet_raw"/$SRC_DATASET/labelsTr/*.nii.gz; do
    if [ -f "$f" ]; then
        # Extract case number from case_10.nii.gz
        case_num=$(basename "$f" | sed -E 's/case_([0-9]+)\.nii\.gz/\1/')
        if [ ! -f "$nnUNet_raw"/$SRC_DATASET/imagesTr/case_${case_num}_0000.nii.gz ]; then
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

echo "Map labels to TotalSpineSeg label system"
echo "  Mapping data_172 labels (0-19) to TotalSpineSeg labels"
echo "  Vertebral Bodies: S(1->50), L5(2->45), L4(3->44), L3(4->43), L2(5->42), L1(6->41), T12(7->32), T11(8->31), T10(9->30), T9(10->29)"
echo "  Intervertebral Discs: L5/S(11->100), L4/L5(12->95), L3/L4(13->94), L2/L3(14->93), L1/L2(15->92), T12/L1(16->91), T11/T12(17->82), T10/T11(18->81), T9/T10(19->80)"
LABEL_MAP_FILE="$nnUNet_raw"/$SRC_DATASET/data172_to_tss_map.json
cat > "$LABEL_MAP_FILE" << 'EOFMAP'
{
    "0": 0,
    "1": 50,
    "2": 45,
    "3": 44,
    "4": 43,
    "5": 42,
    "6": 41,
    "7": 32,
    "8": 31,
    "9": 30,
    "10": 29,
    "11": 100,
    "12": 95,
    "13": 94,
    "14": 93,
    "15": 92,
    "16": 91,
    "17": 82,
    "18": 81,
    "19": 80
}
EOFMAP
echo "  Created label mapping file: $LABEL_MAP_FILE"

# Apply the mapping
totalspineseg_map_labels -m "$LABEL_MAP_FILE" -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/$SRC_DATASET/labelsTr -r -w $JOBS

echo "Making test folders and moving 10% of the data to test folders"
mkdir -p "$nnUNet_raw"/$SRC_DATASET/imagesTs
mkdir -p "$nnUNet_raw"/$SRC_DATASET/labelsTs

# Move 10% of the data to test folders
files=($(for f in "$nnUNet_raw"/$SRC_DATASET/labelsTr/*.nii.gz; do basename "${f/.nii.gz/}"; done))
if [ ${#files[@]} -gt 0 ]; then
    files_shuf=($(shuf -e "${files[@]}"))
    files_10p=(${files_shuf[@]:0:$((${#files_shuf[@]} * 10 / 100))})
    for f in ${files_10p[@]}; do
        mv "$nnUNet_raw"/$SRC_DATASET/imagesTr/${f}_0000.nii.gz "$nnUNet_raw"/$SRC_DATASET/imagesTs/ 2>/dev/null || true
        mv "$nnUNet_raw"/$SRC_DATASET/labelsTr/${f}.nii.gz "$nnUNet_raw"/$SRC_DATASET/labelsTs/ 2>/dev/null || true
    done
    echo "  Moved ${#files_10p[@]} files to test set"
fi

if [ $NOAUG -eq 0 ]; then
    echo "Generate augmentations"
    # Limit workers to avoid resource contention (max 12 workers, same as MAX_PARALLEL_JOBS)
    AUG_JOBS=$(( JOBS > MAX_PARALLEL_JOBS ? MAX_PARALLEL_JOBS : JOBS ))
    echo "  Using $AUG_JOBS workers for augmentation (max: $MAX_PARALLEL_JOBS)"
    # seg-classes: all intervertebral disc labels (80, 81, 82, 91, 92, 93, 94, 95, 100)
    totalspineseg_augment -i "$nnUNet_raw"/$SRC_DATASET/imagesTr -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/Dataset100_TotalSpineSeg_Aug/imagesTr -g "$nnUNet_raw"/Dataset100_TotalSpineSeg_Aug/labelsTr --labels2image --seg-classes 80 81 82 91 92 93 94 95 100 -r -w $AUG_JOBS
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
    echo "  Dataset 102 uses a two-channel input approach for intervertebral disc segmentation"
    echo "  Channel 0: Original image"
    echo "  Channel 1: Alternate IVD masks (odd/even alternating pattern)"
    
    # Step 1: Copy all images and test data
    echo "  Step 1: Copying images and test data..."
    totalspineseg_cpdir "$nnUNet_raw"/$SRC_DATASET "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2 -p "imagesT*/*.nii.gz" -r -w $JOBS
    
    # Step 2: Create duplicate images with _o2e suffix (for odd-to-even swap augmentation)
    echo "  Step 2: Creating odd-to-even swapped copies..."
    totalspineseg_cpdir "$nnUNet_raw"/$SRC_DATASET "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2 -p "imagesTr/*.nii.gz" -t "_0000.nii.gz:_o2e_0000.nii.gz" -r -w $JOBS
    
    # Step 3: Extract alternate IVD labels as second channel (_0001) for training
    # For data172: IVDs are 80,81,82,91,92,93,94,95,100
    # Prioritize even IVDs: 80,82,92,94,100 (T9/T10, T11/T12, L1/L2, L3/L4, L5/S)
    echo "  Step 3: Extracting alternate IVD masks (even priority: 80,82,92,94,100)..."
    totalspineseg_extract_alternate -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/imagesTr --labels 63-100 --prioratize-labels 80 82 92 94 100 --output-seg-suffix _0001 -r -w $JOBS
    
    # Step 4: Extract alternate IVD labels for odd-to-even swapped images (_o2e_0001)
    # Prioritize odd IVDs: 81,91,93,95 (T10/T11, T12/L1, L2/L3, L4/L5)
    echo "  Step 4: Extracting alternate IVD masks for swapped images (odd priority: 81,91,93,95)..."
    totalspineseg_extract_alternate -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/imagesTr --labels 63-100 --prioratize-labels 81 91 93 95 --output-seg-suffix _o2e_0001 -r -w $JOBS
    
    # Step 5: Extract alternate IVD labels as second channel for test set
    echo "  Step 5: Extracting alternate IVD masks for test set..."
    totalspineseg_extract_alternate -s "$nnUNet_raw"/$SRC_DATASET/labelsTs -o "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/imagesTs --labels 63-100 --prioratize-labels 80 82 92 94 100 --output-seg-suffix _0001 -r -w $JOBS
    
    # Step 6: Map training labels to step2 format
    echo "  Step 6: Mapping training labels to step2 format..."
    totalspineseg_map_labels -m "$resources"/labels_maps/nnunet_step2.json -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/labelsTr -r -w $JOBS
    
    # Step 7: Map odd-to-even swapped labels
    echo "  Step 7: Mapping odd-to-even swapped labels..."
    totalspineseg_map_labels -m "$resources"/labels_maps/nnunet_step2_o2e.json -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/labelsTr --output-seg-suffix _o2e -r -w $JOBS
    
    # Step 8: Map test labels to step2 format
    echo "  Step 8: Mapping test labels to step2 format..."
    totalspineseg_map_labels -m "$resources"/labels_maps/nnunet_step2.json -s "$nnUNet_raw"/$SRC_DATASET/labelsTs -o "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/labelsTs -r -w $JOBS
    
    # Step 9: Create dataset.json with correct number of training samples
    echo "  Step 9: Creating dataset.json..."
    jq --arg numTraining "$(ls "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/labelsTr | wc -l)" '.numTraining = ($numTraining|tonumber)' "$resources"/datasets/dataset_step2.json > "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/dataset.json
    
    echo "  Dataset 102 preparation completed!"
fi

if [ $PREP_103 -eq 1 ]; then
    echo "Generate nnUNet dataset 103 (full)"
    totalspineseg_cpdir "$nnUNet_raw"/$SRC_DATASET "$nnUNet_raw"/Dataset103_TotalSpineSeg_full -p "imagesT*/*.nii.gz" -r -w $JOBS
    totalspineseg_map_labels -m "$resources"/labels_maps/nnunet_full.json -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/Dataset103_TotalSpineSeg_full/labelsTr -r -w $JOBS
    totalspineseg_map_labels -m "$resources"/labels_maps/nnunet_full.json -s "$nnUNet_raw"/$SRC_DATASET/labelsTs -o "$nnUNet_raw"/Dataset103_TotalSpineSeg_full/labelsTs -r -w $JOBS
    # Copy the dataset.json file and update the number of training samples
    jq --arg numTraining "$(ls "$nnUNet_raw"/Dataset103_TotalSpineSeg_full/labelsTr | wc -l)" '.numTraining = ($numTraining|tonumber)' "$resources"/datasets/dataset_full.json > "$nnUNet_raw"/Dataset103_TotalSpineSeg_full/dataset.json
fi

echo ""
echo "=========================================="
echo "Dataset preparation completed!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  File matching: Case*.nii.gz <-> mask_case*.nii.gz"
echo "  Label mapping: data_172 labels (0-19) -> TotalSpineSeg labels"
echo ""
if [ $PREP_101 -eq 1 ]; then
    count_101=$(ls "$nnUNet_raw"/Dataset101_TotalSpineSeg_step1/labelsTr 2>/dev/null | wc -l)
    echo "  Dataset 101 (step1): $count_101 training samples"
fi
if [ $PREP_102 -eq 1 ]; then
    count_102=$(ls "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/labelsTr 2>/dev/null | wc -l)
    echo "  Dataset 102 (step2): $count_102 training samples"
fi
if [ $PREP_103 -eq 1 ]; then
    count_103=$(ls "$nnUNet_raw"/Dataset103_TotalSpineSeg_full/labelsTr 2>/dev/null | wc -l)
    echo "  Dataset 103 (full): $count_103 training samples"
fi
echo ""

