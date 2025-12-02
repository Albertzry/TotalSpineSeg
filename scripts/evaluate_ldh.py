import os
import glob
import numpy as np
import nibabel as nib
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def compute_dice(pred, gt, label):
    """Compute Dice score for a specific label."""
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    
    if np.sum(gt_mask) == 0:
        return np.nan # No GT for this label
        
    intersection = np.logical_and(pred_mask, gt_mask)
    dice = 2.0 * intersection.sum() / (pred_mask.sum() + gt_mask.sum())
    return dice

def evaluate_ldh(data_root, dataset_id=102, fold=0):
    """
    Evaluate LDH (Label 12 in prediction, mapped from 101 in original GT)
    We will look at the validation predictions from nnUNet results folder.
    """
    
    # Paths
    nnunet_results = Path(data_root) / "nnUNet/results"
    
    # Find dataset folder (handle variations in name)
    dataset_candidates = list(nnunet_results.glob(f"Dataset{dataset_id}_*"))
    if not dataset_candidates:
        print(f"Dataset {dataset_id} not found in {nnunet_results}")
        return
    dataset_name = dataset_candidates[0].name
    
    # Assume default trainer/planner for now, or find the one with validation results
    # Path: results/Dataset.../Trainer.../fold_X/validation
    trainer_dir = list((nnunet_results / dataset_name).glob("nnUNetTrainer*"))[0] # Pick first trainer
    validation_dir = trainer_dir / f"fold_{fold}" / "validation"
    
    if not validation_dir.exists():
        print(f"Validation folder not found: {validation_dir}")
        print("Make sure training is complete and validation predictions are saved.")
        return

    print(f"Evaluating LDH in: {validation_dir}")
    
    # Ground Truth labels
    # NOTE: nnUNet validation stores predictions. We need to compare against GT.
    # Where is GT? Usually in nnUNet_raw/Dataset.../labelsTr or preprocessed.
    # For validation files, we can use the filenames to find them in raw labelsTr.
    
    raw_labels_dir = Path(data_root) / "nnUNet/raw" / dataset_name / "labelsTr"
    
    pred_files = sorted(list(validation_dir.glob("*.nii.gz")))
    
    # Filter for LDH subjects if identifiable (sub-LDH...)
    # Or just evaluate all if they are mixed
    ldh_pred_files = [f for f in pred_files if "sub-LDH" in f.name]
    
    if not ldh_pred_files:
        print("No sub-LDH* files found in validation set. Maybe this fold didn't include any LDH samples?")
        return

    print(f"Found {len(ldh_pred_files)} LDH subjects in validation set.")
    
    scores = []
    
    for pred_path in tqdm(ldh_pred_files, desc="Calculating Dice"):
        fname = pred_path.name
        gt_path = raw_labels_dir / fname
        
        if not gt_path.exists():
            print(f"Missing GT for {fname}")
            continue
            
        # Load
        pred = nib.load(pred_path).get_fdata().astype(int)
        gt = nib.load(gt_path).get_fdata().astype(int)
        
        # Calculate Dice for Label 12 (LDH)
        # Note: In Dataset 102, LDH is label 12.
        # Ensure your GT also has 12 (it should if it came from raw labelsTr after merge)
        
        dice = compute_dice(pred, gt, 12)
        
        if not np.isnan(dice):
            scores.append({'Case': fname, 'Dice_LDH': dice})
            
    if not scores:
        print("No valid scores computed.")
        return
        
    df = pd.DataFrame(scores)
    mean_dice = df['Dice_LDH'].mean()
    std_dice = df['Dice_LDH'].std()
    
    print("\n" + "="*30)
    print(f"LDH Evaluation Results (Label 12)")
    print("="*30)
    print(f"Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"Median Dice: {df['Dice_LDH'].median():.4f}")
    print(f"Min: {df['Dice_LDH'].min():.4f}")
    print(f"Max: {df['Dice_LDH'].max():.4f}")
    print("="*30)
    
    # Save to CSV
    out_csv = f"LDH_Evaluation_Dataset{dataset_id}_Fold{fold}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Detailed scores saved to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/opt/data/private/data_sum', help='Path to data_sum')
    parser.add_argument('--dataset', type=int, default=102, help='Dataset ID')
    parser.add_argument('--fold', type=int, default=0, help='Fold number')
    args = parser.parse_args()
    
    evaluate_ldh(args.data_root, args.dataset, args.fold)

