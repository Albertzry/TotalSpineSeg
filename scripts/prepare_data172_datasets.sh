#!/bin/bash

# This script prepares data_172 datasets for the TotalSpineSeg model in nnUNetv2 structure.
# The script expects DATA_DIR as the first positional argument (should contain MR/ and Mask/ subdirectories).
# The script also accepts DATASET as the second positional argument to specify which dataset to prepare.
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

# Check if data directory is provided
if [ -z "$1" ]; then
    echo "Error: DATA_DIR is required."
    echo "Usage: $0 DATA_DIR [DATASET] [-noaug]"
    echo "  DATA_DIR: Path to folder containing MR/ and Mask/ subdirectories"
    echo "  DATASET: Optional. Can be 101, 102, 103, or all. Default is 101 102."
    echo "  -noaug: Optional. Skip data augmentation generation."
    exit 1
fi

DATA_DIR="$(realpath "$1")"
IMAGE_DIR="$DATA_DIR/MR"
LABEL_DIR="$DATA_DIR/Mask"

# Check if directories exist
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory does not exist: $DATA_DIR"
    exit 1
fi

if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: MR directory does not exist: $IMAGE_DIR"
    exit 1
fi

if [ ! -d "$LABEL_DIR" ]; then
    echo "Error: Mask directory does not exist: $LABEL_DIR"
    exit 1
fi

# Set the datasets to work with - default is 101 102
if [[ -z $2 || $2 == 101 || $2 == all || $2 == -* ]]; then PREP_101=1; else PREP_101=0; fi
if [[ -z $2 || $2 == 102 || $2 == all || $2 == -* ]]; then PREP_102=1; else PREP_102=0; fi
if [[ $2 == 103 || $2 == all ]]; then PREP_103=1; else PREP_103=0; fi

# Set the augmentations to generate - default is to generate augmentations
if [[ $2 == -noaug || $3 == -noaug ]]; then NOAUG=1; else NOAUG=0; fi

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
echo "Preparing data_172 datasets from:"
echo "  Data directory: $DATA_DIR"
echo "  Image directory: $IMAGE_DIR"
echo "  Label directory: $LABEL_DIR"
echo "=========================================="

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
    # Limit workers to avoid resource contention (max 16 workers)
    AUG_JOBS=$(( JOBS > 16 ? 16 : JOBS ))
    echo "  Using $AUG_JOBS workers for augmentation"
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
    totalspineseg_cpdir "$nnUNet_raw"/$SRC_DATASET "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2 -p "imagesT*/*.nii.gz" -r -w $JOBS
    totalspineseg_cpdir "$nnUNet_raw"/$SRC_DATASET "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2 -p "imagesTr/*.nii.gz" -t "_0000.nii.gz:_o2e_0000.nii.gz" -r -w $JOBS
    totalspineseg_extract_alternate -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/imagesTr --labels 63-100 --prioratize-labels 63 65 67 72 74 76 78 80 82 92 94 --output-seg-suffix _0001 -r -w $JOBS -r
    totalspineseg_extract_alternate -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/imagesTr --labels 63-100 --prioratize-labels 64 66 71 73 75 77 79 81 91 93 95 --output-seg-suffix _o2e_0001 -r -w $JOBS -r
    totalspineseg_extract_alternate -s "$nnUNet_raw"/$SRC_DATASET/labelsTs -o "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/imagesTs --labels 63-100 --prioratize-labels 63 65 67 72 74 76 78 80 82 92 94 --output-seg-suffix _0001 -r -w $JOBS -r
    totalspineseg_map_labels -m "$resources"/labels_maps/nnunet_step2.json -s "$nnUNet_raw"/$SRC_DATASET/labelsTr -o "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/labelsTr -r -w $JOBS
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
echo "  - File matching: Case*.nii.gz <-> mask_case*.nii.gz"
echo "  - Label mapping: data_172 labels (0-19) -> TotalSpineSeg labels"
echo "    * Vertebral Bodies: S(50), L5(45), L4(44), L3(43), L2(42), L1(41), T12(32), T11(31), T10(30), T9(29)"
echo "    * Intervertebral Discs: L5/S(100), L4/L5(95), L3/L4(94), L2/L3(93), L1/L2(92), T12/L1(91), T11/T12(82), T10/T11(81), T9/T10(80)"
if [ $PREP_101 -eq 1 ]; then
    echo "  - Dataset 101: $(ls "$nnUNet_raw"/Dataset101_TotalSpineSeg_step1/labelsTr 2>/dev/null | wc -l) training samples"
fi
if [ $PREP_102 -eq 1 ]; then
    echo "  - Dataset 102: $(ls "$nnUNet_raw"/Dataset102_TotalSpineSeg_step2/labelsTr 2>/dev/null | wc -l) training samples"
fi
if [ $PREP_103 -eq 1 ]; then
    echo "  - Dataset 103: $(ls "$nnUNet_raw"/Dataset103_TotalSpineSeg_full/labelsTr 2>/dev/null | wc -l) training samples"
fi

