#!/usr/bin/env python3
"""
Stage B: ROI fine segmentation using nnUNet (Dataset 107).

NOTICE:
This script is DEPRECATED in the "B2: nnUNet ROI" scheme.
The actual training logic has been moved to the standard nnUNetv2 pipeline.

Please run the full pipeline using:
    bash scripts/train.sh 105 0

This will automatically:
1. Train Stage A (detection) using scripts/train_ldh_stage_a.py
2. Train Stage B (segmentation) using 'nnUNetv2_train 107 ...'
"""
import sys

def main():
    print("="*60)
    print("NOTICE: Stage B is now trained using nnUNet (Dataset 107).")
    print("Please run 'bash scripts/train.sh 105 0' to execute the full pipeline.")
    print("This script is no longer used for training.")
    print("="*60)
    sys.exit(0)

if __name__ == "__main__":
    main()
