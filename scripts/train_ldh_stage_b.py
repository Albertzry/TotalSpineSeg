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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rois-dir", type=Path, required=True, help="Directory with StageB ROI .npz patches (positives)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=min(8, (os.cpu_count() or 8)))
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True, help="Output checkpoint path (.pth)")
    args = ap.parse_args()

    ds = StageBDataset(args.rois_dir)
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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **dl_kwargs)
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
        for x, mask_gt, sdm_gt in train_loader:
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
        with torch.no_grad():
            for x, mask_gt, sdm_gt in val_loader:
                x = x.to(device)
                mask_gt = mask_gt.to(device)
                sdm_gt = sdm_gt.to(device)
                mask_logits, sdm_pred = model(x)
                vloss = stage_b_total_loss(mask_logits, sdm_pred, mask_gt, sdm_gt)
                val_losses.append(float(vloss.item()))
                pred = (torch.sigmoid(mask_logits) >= 0.5).float().cpu().numpy()[0, 0]
                gt = mask_gt.detach().cpu().numpy()[0, 0]
                dices.append(dice(pred, gt))

        mean_dice = float(np.nanmean(np.array(dices, dtype=np.float32)))
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
            f"val dice={mean_dice:.3f}{best_marker}"
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


