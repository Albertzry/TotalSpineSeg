#!/usr/bin/env python3
"""
Stage B: ROI fine segmentation with surface/distance-aware supervision.

Model outputs:
  - LDH mask logits
  - SDM regression

Loss (REQUIRED):
  Total = FocalTversky(α=0.7,β=0.3,γ=0.75) + 0.5*BoundaryLoss + 0.2*L1(SDM)
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import time

# Add repository root to Python path for imports
_script_dir = Path(__file__).parent.resolve()
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from totalspineseg.ldh_twostage.data import StageBDataset
from totalspineseg.ldh_twostage.losses import stage_b_total_loss
from totalspineseg.ldh_twostage.metrics import dice
from totalspineseg.ldh_twostage.models import SmallUNet3D


def _save_training_curves(history: dict, out_ckpt: Path) -> Path:
    """
    Save Stage B training curves to a single PNG, overwritten each call.

    Style requirements (user):
      - val_loss: thin red
      - train_loss: thin blue
      - dice: thin green
      - all in a single figure
    """
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    plot_path = out_ckpt.parent / f"{out_ckpt.stem}_training_curves.png"

    epochs = list(range(len(history["train_loss"])))
    lw = 0.6  # very thin

    fig, ax_loss = plt.subplots(1, 1, figsize=(10, 5))
    ax_dice = ax_loss.twinx()

    # Losses on left axis
    ax_loss.plot(
        epochs,
        history["train_loss"],
        color="b",
        linestyle="-",
        linewidth=lw,
        label="train_loss",
    )
    ax_loss.plot(
        epochs,
        history["val_loss"],
        color="r",
        linestyle="-",
        linewidth=lw,
        label="val_loss",
    )
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)

    # Dice on right axis
    ax_dice.plot(
        epochs,
        history["val_dice"],
        color="g",
        linestyle="-",
        linewidth=lw,
        label="dice",
    )
    ax_dice.set_ylabel("Dice")
    ax_dice.set_ylim(0, 1.05)

    # Combined legend
    handles1, labels1 = ax_loss.get_legend_handles_labels()
    handles2, labels2 = ax_dice.get_legend_handles_labels()
    ax_loss.legend(handles1 + handles2, labels1 + labels2, loc="best")

    ax_loss.set_title("Stage B: train/val loss + dice")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _load_train_sample_ids(rois_dir: Path) -> set[str]:
    """
    Load training sample IDs from Dataset105/imagesTr to avoid data leakage.
    
    Returns:
        Set of sample IDs that belong to the training split.
    """
    # rois_dir is typically: .../Dataset105_TotalSpineSeg_LDH/ldh_twostage/stageB_rois
    # dataset root is: .../Dataset105_TotalSpineSeg_LDH
    dataset_root = rois_dir.parent.parent
    images_tr = dataset_root / "imagesTr"
    
    if not images_tr.exists():
        raise FileNotFoundError(
            f"imagesTr directory not found at {images_tr}. "
            "Ensure prepare_dataset_105.py has been run and created train/test split."
        )
    
    train_ids = set()
    for f in images_tr.glob("*_0000.nii.gz"):
        sample_id = f.name.replace("_0000.nii.gz", "")
        train_ids.add(sample_id)
    
    if not train_ids:
        raise ValueError(f"No training samples found in {images_tr}")
    
    return train_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rois-dir", type=Path, required=True, help="Directory with StageB ROI .npz patches (pos + optional neg)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=min(8, (os.cpu_count() or 8)))
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument(
        "--use-negatives",
        action="store_true",
        default=False,
        help="Scheme D: include negative StageB ROIs so the model learns to output empty masks.",
    )
    ap.add_argument(
        "--neg-ratio",
        type=float,
        default=0.5,
        help="Scheme D: target fraction of negatives in training sampler (0-1). Only used with --use-negatives.",
    )
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True, help="Output checkpoint path (.pth)")
    args = ap.parse_args()

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Stage B init: rois_dir={args.rois_dir}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Stage B loading train split...")
    
    # Load train sample IDs to prevent data leakage
    train_sample_ids = _load_train_sample_ids(args.rois_dir)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Stage B train split: {len(train_sample_ids)} samples (from imagesTr)")
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Stage B indexing ROIs (use_negatives={bool(args.use_negatives)})...")
    ds = StageBDataset(args.rois_dir, filter_sample_ids=train_sample_ids, only_positive=not bool(args.use_negatives))
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Stage B ROIs indexed: n={len(ds)} (train split only)")
    n_val = max(1, int(len(ds) * args.val_ratio))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    dl_kwargs = {
        "num_workers": int(args.num_workers),
        "pin_memory": device.type == "cuda",
        "persistent_workers": int(args.num_workers) > 0,
    }
    if int(args.num_workers) > 0:
        dl_kwargs["prefetch_factor"] = int(args.prefetch_factor)

    # Scheme D: sampler to control pos/neg ratio (otherwise negatives may dominate).
    sampler = None
    if bool(args.use_negatives):
        neg_ratio = float(args.neg_ratio)
        if not (0.0 < neg_ratio < 1.0):
            raise ValueError("--neg-ratio must be in (0,1) when --use-negatives is set")
        from torch.utils.data import WeightedRandomSampler

        base = train_ds.dataset  # StageBDataset
        idxs = list(map(int, train_ds.indices))
        has = np.array([int(base.records[i].has_ldh) for i in idxs], dtype=np.int64)
        n_pos = int((has == 1).sum())
        n_neg = int((has == 0).sum())
        if n_pos == 0 or n_neg == 0:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Stage B warning: n_pos={n_pos}, n_neg={n_neg}; disabling sampler.")
            sampler = None
        else:
            w_pos = (1.0 - neg_ratio) / float(n_pos)
            w_neg = neg_ratio / float(n_neg)
            weights = torch.from_numpy(np.where(has == 1, w_pos, w_neg).astype(np.float32))
            sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, **dl_kwargs)
    model = SmallUNet3D(in_channels=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Metrics tracking for plotting
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
    }

    best_dice = -1.0
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Stage B training started - {args.epochs} epochs")
    training_start = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch{epoch}")

        model.train()
        train_losses = []
        for x, mask_gt, sdm_gt, _has_ldh in train_loader:
            x = x.to(device)
            mask_gt = mask_gt.to(device)
            sdm_gt = sdm_gt.to(device)
            mask_logits, sdm_pred = model(x)
            loss = stage_b_total_loss(mask_logits, sdm_pred, mask_gt, sdm_gt)
            train_losses.append(loss.item())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # validation: val_loss + dice
        model.eval()
        dices = []
        val_losses = []
        n_pos = 0
        with torch.no_grad():
            for x, mask_gt, sdm_gt, has_ldh in val_loader:
                x = x.to(device)
                mask_gt = mask_gt.to(device)
                sdm_gt = sdm_gt.to(device)
                has_ldh = has_ldh.to(device)
                mask_logits, sdm_pred = model(x)
                vloss = stage_b_total_loss(mask_logits, sdm_pred, mask_gt, sdm_gt)
                val_losses.append(float(vloss.item()))
                # Dice is meaningful on positives; negatives are handled via loss (FP suppression).
                if int(has_ldh.item()) == 1 and float(mask_gt.sum().item()) > 0.0:
                    n_pos += 1
                    pred = (torch.sigmoid(mask_logits) >= 0.5).float().cpu().numpy()[0, 0]
                    gt = mask_gt.detach().cpu().numpy()[0, 0]
                    dices.append(dice(pred, gt))

        mean_dice = float(np.nanmean(np.array(dices, dtype=np.float32))) if dices else float("nan")
        mean_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0
        mean_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
        
        # Record metrics
        history['train_loss'].append(mean_train_loss)
        history['val_loss'].append(mean_val_loss)
        history['val_dice'].append(mean_dice)
        
        # Check if current is best
        is_best_dice = mean_dice > best_dice
        best_marker = ""
        if is_best_dice:
            best_marker = f" ★ NEW BEST DICE! (prev: {best_dice:.3f})"
        
        print(
            f"Epoch{epoch} | train_loss={mean_train_loss:.4f} | val_loss={mean_val_loss:.4f} | "
            f"val dice={mean_dice:.3f} (pos={n_pos}){best_marker}"
        )
        epoch_time = time.time() - epoch_start
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch{epoch} time={epoch_time:.1f}s")

        # Save best model based on Dice (primary metric for segmentation)
        if is_best_dice:
            best_dice = mean_dice
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(), 
                "epoch": epoch, 
                "val_dice": best_dice,
                "val_loss": mean_val_loss,
            }, args.out)

        # Update plot every epoch (overwrite)
        plot_path = _save_training_curves(history, args.out)

    training_time = time.time() - training_start
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Stage B training completed")
    print(f"Best val Dice: {best_dice:.3f} | Total training time: {training_time/60:.1f} min")

    # Ensure final plot exists (already updated each epoch, but keep for completeness)
    plot_path = _save_training_curves(history, args.out)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training curves saved to: {plot_path}")


if __name__ == "__main__":
    main()


