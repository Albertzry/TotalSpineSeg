#!/bin/bash

# This script get the datasets require to train the model from:
#   git@github.com:OpenNeuroDatasets/ds005616.git (replaces whole-spine)
#   git@data.neuro.polymtl.ca:datasets/spider-challenge-2023.git (skip, manual download)
#   git@github.com:spine-generic/data-multi-subject.git
#   git@github.com:spine-generic/data-single-subject.git

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

# set TOTALSPINESEG and TOTALSPINESEG_DATA if not set
TOTALSPINESEG="$(realpath "${TOTALSPINESEG:-totalspineseg}")"
TOTALSPINESEG_DATA="$(realpath "${TOTALSPINESEG_DATA:-data}")"

# Set the paths to the BIDS data folders
bids="$TOTALSPINESEG_DATA"/bids

# Make sure $TOTALSPINESEG_DATA/bids exists and enter it
mkdir -p "$bids"
CURR_DIR="$(realpath .)"
cd "$bids"

datasets=(
    git@github.com:OpenNeuroDatasets/ds005616.git
    # git@data.neuro.polymtl.ca:datasets/spider-challenge-2023.git
    git@github.com:spine-generic/data-multi-subject.git
    git@github.com:spine-generic/data-single-subject.git
)

# Loop over datasets and download them
for ds in ${datasets[@]}; do
    dsn=$(basename $ds .git)

    # Skip spider-challenge-2023 (manual download)
    if [ "$dsn" = "spider-challenge-2023" ]; then
        echo "Skipping $dsn (marked for manual download)"
        continue
    fi

    # Check if directory already exists
    if [ -d "$dsn" ]; then
        echo "Directory $dsn already exists. Skipping clone, continuing with existing directory..."
        cd $dsn
        # Update the repository to get latest changes
        echo "Updating repository..."
        git pull || echo "Warning: git pull failed (may not be a git repository or network issue)"
    else
        # Clone the dataset from the specified repository
        git clone $ds
        # Enter the dataset directory
        cd $dsn
    fi

    # Remove all files and folders not in this formats:
    #   .*
    #   sub-*/anat/sub-*_{T1,T2,T2star,MTS}.nii.gz
    #   derivatives/labels_iso/sub-*/anat/sub-*_{T1w,T2w,T2star,MTS}_space-resampled_{label-spine_dseg,label-SC_seg,label-canal_seg}.nii.gz
    find . ! -path '.' ! -path './.*' \
        ! -regex '^\./sub-[^/]*\(/anat\(/sub-[^/]*_\(T1w\|T2w\|T2star\|MTS\)\.nii\.gz\)?\)?$' \
        ! -regex '^\./derivatives\(/labels_iso\(/sub-[^/]*\(/anat\(/sub-[^/]*_\(T1w\|T2w\|T2star\|MTS\)_space-resampled_\(label-spine_dseg\|label-SC_seg\|label-canal_seg\)\.nii\.gz\)?\)?\)?\)?$' \
        -delete

    # Download the necessary files from git-annex (if git-annex is installed)
    if command -v git-annex >/dev/null 2>&1 || git annex version >/dev/null 2>&1; then
        echo "Downloading files from git-annex..."
        # Check if this is a git-annex repository
        if [ -d ".git/annex" ] || git config --get remote.origin.annex-ignore 2>/dev/null; then
            echo "This repository uses git-annex. Downloading actual files..."
            # Force re-download of all files (in case some are missing or corrupted)
            git annex get --all || echo "Warning: git-annex get --all failed, trying without --all..."
            git annex get || echo "Warning: git-annex get failed"
            # Verify files are downloaded
            echo "Verifying git-annex files..."
            git annex fsck --fast 2>/dev/null || echo "Note: git-annex fsck not available or not needed"
        else
            echo "This repository does not use git-annex, skipping git-annex get"
        fi
    else
        echo "Warning: git-annex is not installed. Skipping git-annex get."
        echo "If this repository uses git-annex, please install it:"
        echo "  conda install -c conda-forge git-annex -y"
        echo "  or: apt-get install git-annex -y"
    fi

    # Move back to the parent directory to process the next dataset
    cd ..
done

# Return to the original working directory
cd "$CURR_DIR"
