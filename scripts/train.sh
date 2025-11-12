#!/bin/bash

# This script trains the TotalSpineSeg nnUNet models.
# It accepts optional parameters DATASET and FOLD.
# By default, it trains the models for datasets 101 and 102 with fold 0.
#
# Usage:
#   bash train.sh [DATASET] [FOLD] [nnUNetTrainer] [nnUNetPlanner] [nnUNetPlans]
#
# Examples:
#   bash train.sh              # Train datasets 101 and 102 with fold 0 (default)
#   bash train.sh 101 0        # Train dataset 101 with fold 0
#   bash train.sh "101 102" 0  # Train both datasets 101 and 102 with fold 0
#   bash train.sh all 0        # Train all datasets (101, 102, 103) with fold 0
#
# The script expects the following environment variables to be set:
#   TOTALSPINESEG: The path to the TotalSpineSeg repository.
#   TOTALSPINESEG_DATA: The path to the TotalSpineSeg data folder.
#   TOTALSPINESEG_JOBS: The number of CPU cores to use. Default is the number of CPU cores available.
#                       NOTE: Will be automatically capped at 12 to prevent resource exhaustion.
#   TOTALSPINESEG_JOBSNN: The number of jobs to use for the nnUNet training/processing.
#                         Default is min(JOBS, memory_GB/8).
#                         NOTE: Will be automatically capped at 12 to prevent resource exhaustion.
#   TOTALSPINESEG_DEVICE: The device to use. Default is "cuda" if available, otherwise "cpu".

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
DATASETS=${1:-101 102}
if [ "$DATASETS" == all ]; then DATASETS=(101 102 103); fi

# Set the fold to work with - default is 0
FOLD=${2:-0}

# set TOTALSPINESEG and TOTALSPINESEG_DATA if not set
TOTALSPINESEG="$(realpath "${TOTALSPINESEG:-totalspineseg}")"
TOTALSPINESEG_DATA="$(realpath "${TOTALSPINESEG_DATA:-data}")"

# Maximum parallel jobs limit
# Limiting to 12 to prevent resource exhaustion and ensure stable training
MAX_PARALLEL_JOBS=12

# Get the number of CPUs
CORES=${SLURM_JOB_CPUS_PER_NODE:-$(lscpu -p | egrep -v '^#' | wc -l)}

# Get memory in GB
MEMGB=$(awk '/MemTotal/ {print int($2/1024/1024)}' /proc/meminfo)

# Set the number of jobs (max 12)
JOBS=${TOTALSPINESEG_JOBS:-$CORES}
JOBS=$(( JOBS > MAX_PARALLEL_JOBS ? MAX_PARALLEL_JOBS : JOBS ))
JOBS=$(( JOBS < 1 ? 1 : JOBS ))

# Set the number of jobs for the nnUNet based on memory availability
# Rule: Approximately 8GB RAM per job
JOBSNN=$(( JOBS < $((MEMGB / 8)) ? JOBS : $((MEMGB / 8)) ))
JOBSNN=$(( JOBSNN < 1 ? 1 : JOBSNN ))
JOBSNN=$(( JOBSNN > MAX_PARALLEL_JOBS ? MAX_PARALLEL_JOBS : JOBSNN ))

# Allow override via environment variable, but still enforce maximum limit
JOBSNN=${TOTALSPINESEG_JOBSNN:-$JOBSNN}
JOBSNN=$(( JOBSNN > MAX_PARALLEL_JOBS ? MAX_PARALLEL_JOBS : JOBSNN ))
JOBSNN=$(( JOBSNN < 1 ? 1 : JOBSNN ))

# Set the device to cpu if cuda is not available
DEVICE=${TOTALSPINESEG_DEVICE:-$(python3 -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")}

# Apply torch compatibility patch before importing nnUNet
# This fixes the OptimizedModule import error
python3 -c "import sys; sys.path.insert(0, '$(realpath "$TOTALSPINESEG")'); from totalspineseg.utils import torch_compat" 2>/dev/null || true

# Set nnunet params
export nnUNet_def_n_proc=$JOBSNN
export nnUNet_n_proc_DA=$JOBSNN
export nnUNet_raw="$TOTALSPINESEG_DATA"/nnUNet/raw
export nnUNet_preprocessed="$TOTALSPINESEG_DATA"/nnUNet/preprocessed
export nnUNet_results="$TOTALSPINESEG_DATA"/nnUNet/results
export nnUNet_exports="$TOTALSPINESEG_DATA"/nnUNet/exports


nnUNetTrainer=${3:-nnUNetTrainer_DASegOrd0_NoMirroring}
nnUNetPlanner=${4:-ExperimentPlanner}
# Note on nnUNetPlans_small configuration:
# To train with a small patch size, verify that the nnUNetPlans_small.json file 
# in $nnUNet_preprocessed/Dataset10[1,2]_TotalSpineSeg_step[1,2] matches the version provided in the release.
# Make any necessary updates to this file before starting the training process.
nnUNetPlans=${5:-nnUNetPlans_small}
configuration=3d_fullres
data_identifier=${nnUNetPlans}_${configuration}

echo ""
echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "Paths:"
echo "  nnUNet_raw=${nnUNet_raw}"
echo "  nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "  nnUNet_results=${nnUNet_results}"
echo "  nnUNet_exports=${nnUNet_exports}"
echo ""
echo "Model Settings:"
echo "  nnUNetTrainer=${nnUNetTrainer}"
echo "  nnUNetPlanner=${nnUNetPlanner}"
echo "  nnUNetPlans=${nnUNetPlans}"
echo "  configuration=${configuration}"
echo "  data_identifier=${data_identifier}"
echo ""
echo "Resource Allocation:"
echo "  Available CPU cores: ${CORES}"
echo "  Available memory: ${MEMGB} GB"
echo "  MAX_PARALLEL_JOBS limit: ${MAX_PARALLEL_JOBS} (enforced)"
echo "  JOBS (general): ${JOBS}"
echo "  JOBSNN (nnUNet training): ${JOBSNN}"
echo "  DEVICE: ${DEVICE}"
echo ""
echo "Training Targets:"
echo "  DATASETS: ${DATASETS[@]}"
echo "  FOLD: ${FOLD}"
echo "=========================================="
echo ""

for d in ${DATASETS[@]}; do
    # Get the dataset name
    d_name=$(basename "$(ls -d "$nnUNet_raw"/Dataset${d}_*)")

    if [ ! -f "$nnUNet_preprocessed"/$d_name/dataset_fingerprint.json ]; then
        echo "Extracting fingerprint dataset $d_name (using $JOBSNN workers, max: $MAX_PARALLEL_JOBS)"
        # --verify_dataset_integrity not working in nnunetv2==2.4.2
        # https://github.com/MIC-DKFZ/nnUNet/issues/2144
        # But nnUNetTrainer_DASegOrd0_NoMirroring not working in nnunetv2==2.5.1
        # https://github.com/MIC-DKFZ/nnUNet/issues/2480
        nnUNetv2_extract_fingerprint -d $d -np $JOBSNN #--verify_dataset_integrity
    fi

    if [ ! -f "$nnUNet_preprocessed"/$d_name/${nnUNetPlans}.json ]; then
        echo "Planning dataset $d_name"
        nnUNetv2_plan_experiment -d $d -pl $nnUNetPlanner -overwrite_plans_name $nnUNetPlans
    fi

    # If already preprocess do not preprocess again
    if [[ ! -d "$nnUNet_preprocessed"/$d_name/$data_identifier || ! $(find "$nnUNet_raw"/$d_name/labelsTr -name "*.nii.gz" | wc -l) -eq $(find "$nnUNet_preprocessed"/$d_name/$data_identifier -name "*.npz" | wc -l) || ! $(find "$nnUNet_raw"/$d_name/labelsTr -name "*.nii.gz" | wc -l) -eq $(find "$nnUNet_preprocessed"/$d_name/$data_identifier -name "*.pkl" | wc -l) ]]; then
        echo "Preprocessing dataset $d_name (using $JOBSNN workers, max: $MAX_PARALLEL_JOBS)"
        nnUNetv2_preprocess -d $d -plans_name $nnUNetPlans -c $configuration -np $JOBSNN
    fi

    echo "Training dataset $d_name fold $FOLD"
    # if already decompressed do not decompress again
    if [ $(find "$nnUNet_preprocessed"/$d_name/$data_identifier -name "*.npy" | wc -l) -eq $(( 2 * $(find "$nnUNet_preprocessed"/$d_name/$data_identifier -name "*.npz" | wc -l))) ]; then DECOMPRESSED="--use_compressed"; else DECOMPRESSED=""; fi
    nnUNetv2_train $d $configuration $FOLD -tr $nnUNetTrainer -p $nnUNetPlans --c -device $DEVICE $DECOMPRESSED

    echo "Export the model for dataset $d_name in "$nnUNet_exports""
    mkdir -p "$nnUNet_exports"
    mkdir -p "$nnUNet_results"/$d_name/ensembles
    nnUNetv2_export_model_to_zip -d $d -o "$nnUNet_exports"/${d_name}__${nnUNetTrainer}__${nnUNetPlans}__${configuration}__fold_$FOLD.zip -c $configuration -f $FOLD -tr $nnUNetTrainer -p $nnUNetPlans

    echo "Testing nnUNet model for dataset $d_name (using $JOBSNN workers, max: $MAX_PARALLEL_JOBS)"
    mkdir -p "$nnUNet_results"/$d_name/${nnUNetTrainer}__${nnUNetPlans}__${configuration}/fold_${FOLD}/test
    # -npp: number of processes for preprocessing, -nps: number of processes for segmentation
    nnUNetv2_predict -d $d -i "$nnUNet_raw"/$d_name/imagesTs -o "$nnUNet_results"/$d_name/${nnUNetTrainer}__${nnUNetPlans}__${configuration}/fold_${FOLD}/test -f $FOLD -c $configuration -tr $nnUNetTrainer -p $nnUNetPlans -npp $JOBSNN -nps $JOBSNN
    # Evaluate with controlled parallelism
    nnUNetv2_evaluate_folder "$nnUNet_raw"/$d_name/labelsTs "$nnUNet_results"/$d_name/${nnUNetTrainer}__${nnUNetPlans}__${configuration}/fold_${FOLD}/test -djfile "$nnUNet_results"/$d_name/${nnUNetTrainer}__${nnUNetPlans}__${configuration}/dataset.json -pfile "$nnUNet_results"/$d_name/${nnUNetTrainer}__${nnUNetPlans}__${configuration}/plans.json -np $JOBSNN

    p="$(realpath .)"
    cd "$nnUNet_results"
    zip "$nnUNet_exports"/${d_name}__${nnUNetTrainer}__${nnUNetPlans}__${configuration}__fold_$FOLD.zip $d_name/${nnUNetTrainer}__${nnUNetPlans}__${configuration}/fold_${FOLD}/test/summary.json
    cd "$p"

    echo "Export nnUNet dataset list for dataset $d_name"
    cd "$nnUNet_raw"/$d_name
    ls */ > "$nnUNet_results"/$d_name/dataset.txt
    cd "$nnUNet_results"
    zip "$nnUNet_exports"/${d_name}__${nnUNetTrainer}__${nnUNetPlans}__${configuration}__fold_$FOLD.zip $d_name/dataset.txt
    cd "$nnUNet_preprocessed"
    zip "$nnUNet_exports"/${d_name}__${nnUNetTrainer}__${nnUNetPlans}__${configuration}__fold_$FOLD.zip $d_name/splits_final.json
    cd "$p"

done
