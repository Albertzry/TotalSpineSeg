#!/usr/bin/env python3
"""
Stage A: disc-level LDH detection.

Input channels:
  - image
  - disc_mask
  - disc_index (voxel-wise, normalized)

Output:
  - has_LDH (binary)

Loss:
  - Focal loss (gamma=2), recall-oriented.
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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from totalspineseg.ldh_twostage.data import StageADataset
from totalspineseg.ldh_twostage.losses import focal_loss_with_logits
from totalspineseg.ldh_twostage.metrics import DetectionReport
from totalspineseg.ldh_twostage.models import StageADetectorV2


def _save_training_curves(history: dict, out_ckpt: Path) -> Path:
    """
    Save Stage A training curves to a single PNG, overwritten each call.

    Style requirements (user):
      - loss: thin green solid
      - precision/recall: thin red/blue dashed
      - f1: thin purple solid
    """
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    plot_path = out_ckpt.parent / f"{out_ckpt.stem}_training_curves.png"

    epochs = list(range(len(history["train_loss"])))
    lw = 0.6  # very thin

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Training Loss (green solid, thin)
    axes[0].plot(
        epochs,
        history["train_loss"],
        color="g",
        linestyle="-",
        linewidth=lw,
        label="train_loss",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Focal Loss")
    axes[0].set_title("Stage A: Training Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot 2: Validation Metrics
    # precision/recall: red/blue dashed, thin; f1: purple solid, thin
    axes[1].plot(
        epochs,
        history["val_precision"],
        color="r",
        linestyle="--",
        linewidth=lw,
        label="precision",
    )
    axes[1].plot(
        epochs,
        history["val_recall"],
        color="b",
        linestyle="--",
        linewidth=lw,
        label="recall",
    )
    axes[1].plot(
        epochs,
        history["val_f1"],
        color="purple",
        linestyle="-",
        linewidth=lw,
        label="f1",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Stage A: Val Precision/Recall/F1")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _load_train_sample_ids(patches_dir: Path) -> set[str]:
    """
    Load training sample IDs from Dataset105/imagesTr to avoid data leakage.
    
    Returns:
        Set of sample IDs that belong to the training split.
    """
    # patches_dir is typically: .../Dataset105_TotalSpineSeg_LDH/ldh_twostage/stageA_patches
    # dataset root is: .../Dataset105_TotalSpineSeg_LDH
    dataset_root = patches_dir.parent.parent
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
    ap.add_argument(
        "--patches-dir",
        type=Path,
        default=None,
        help="Directory with StageA .npz patches. Auto-resolves from $TOTALSPINESEG_DATA if not provided.",
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8, increased for 24G GPU)")
    ap.add_argument("--num-workers", type=int, default=min(8, (os.cpu_count() or 8)))
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output checkpoint path (.pth). Auto-resolves from $TOTALSPINESEG_DATA if not provided.",
    )
    ap.add_argument("--fold", type=int, default=0, help="Fold number (default: 0)")
    # Stage A now defaults to (and only supports) v2 in codebase
    ap.add_argument(
        "--arch",
        type=str,
        default="v2",
        choices=["v2"],
        help="Stage A detector architecture (v2 residual CNN).",
    )
    ap.add_argument("--base", type=int, default=32, help="(v2) base feature channels (default: 32).")
    ap.add_argument(
        "--blocks",
        type=int,
        nargs=3,
        default=[2, 2, 2],
        help="(v2) number of residual blocks per stage, e.g. 2 2 2 (default).",
    )
    ap.add_argument(
        "--norm",
        type=str,
        default="instance",
        choices=["instance", "group"],
        help="(v2) normalization type (default: instance).",
    )
    ap.add_argument("--no-se", action="store_true", default=False, help="(v2) disable SE attention.")
    ap.add_argument("--dropout", type=float, default=0.10, help="(v2) dropout probability (default: 0.10).")
    args = ap.parse_args()

    # Auto-resolve paths from environment variables
    totalspineseg_data = Path(os.environ.get("TOTALSPINESEG_DATA", ""))
    if args.patches_dir is None:
        if not totalspineseg_data:
            raise SystemExit(
                "Missing --patches-dir. Provide it explicitly or set $TOTALSPINESEG_DATA environment variable."
            )
        d105_candidates = sorted((totalspineseg_data / "nnUNet" / "raw").glob("Dataset105_*"))
        if not d105_candidates:
            raise SystemExit(
                f"未找到 Dataset105_* 目录在 {totalspineseg_data/'nnUNet'/'raw'}。\n"
                "请先运行: python scripts/prepare_dataset_105.py"
            )
        args.patches_dir = d105_candidates[0] / "ldh_twostage" / "stageA_patches"
    
    if args.out is None:
        if not totalspineseg_data:
            raise SystemExit(
                "Missing --out. Provide it explicitly or set $TOTALSPINESEG_DATA environment variable."
            )
        d105_candidates = sorted((totalspineseg_data / "nnUNet" / "results").glob("Dataset105_*"))
        if not d105_candidates:
            d105_candidates = sorted((totalspineseg_data / "nnUNet" / "raw").glob("Dataset105_*"))
        if not d105_candidates:
            raise SystemExit(
                f"未找到 Dataset105_* 目录。\n"
                "请先运行: python scripts/prepare_dataset_105.py"
            )
        ckpt_dir = totalspineseg_data / "nnUNet" / "results" / d105_candidates[0].name / "ldh_twostage" / "checkpoints"
        args.out = ckpt_dir / f"ldh_stageA_fold_{args.fold}.pth"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Stage A init: patches_dir={args.patches_dir}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Stage A loading train split...")
    
    # Load train sample IDs to prevent data leakage
    train_sample_ids = _load_train_sample_ids(args.patches_dir)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Stage A train split: {len(train_sample_ids)} samples (from imagesTr)")
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Stage A indexing patches (reading has_ldh only)...")
    ds = StageADataset(args.patches_dir, filter_sample_ids=train_sample_ids)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Stage A patches indexed: n={len(ds)} (train split only)")
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
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **dl_kwargs)

    model = StageADetectorV2(
        in_channels=3,
        base=int(args.base),
        blocks=(int(args.blocks[0]), int(args.blocks[1]), int(args.blocks[2])),
        norm=str(args.norm),
        use_se=not bool(args.no_se),
        dropout=float(args.dropout),
    ).to(device)
    model_meta = {
        "arch": "v2",
        "model_kwargs": {
            "in_channels": 3,
            "base": int(args.base),
            "blocks": (int(args.blocks[0]), int(args.blocks[1]), int(args.blocks[2])),
            "norm": str(args.norm),
            "use_se": not bool(args.no_se),
            "dropout": float(args.dropout),
        },
    }
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Metrics tracking for plotting
    history = {
        'train_loss': [],
        'val_recall': [],
        'val_precision': [],
        'val_f1': []
    }

    best_recall = -1.0
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Stage A training started - {args.epochs} epochs")
    training_start = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch{epoch}")

        model.train()
        train_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = focal_loss_with_logits(logits, y, gamma=2.0)
            train_losses.append(loss.item())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        rep = DetectionReport()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                pred = (torch.sigmoid(logits) >= 0.5).long()
                gt = y.long()
                rep.tp += int(((pred == 1) & (gt == 1)).sum().item())
                rep.fp += int(((pred == 1) & (gt == 0)).sum().item())
                rep.tn += int(((pred == 0) & (gt == 0)).sum().item())
                rep.fn += int(((pred == 0) & (gt == 1)).sum().item())

        # Record metrics
        mean_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0
        history['train_loss'].append(mean_train_loss)
        history['val_recall'].append(rep.recall)
        history['val_precision'].append(rep.precision)
        history['val_f1'].append(rep.f1)

        is_best = rep.recall > best_recall
        best_marker = " ★ NEW BEST!" if is_best else ""
        
        print(
            f"Epoch{epoch} | train_loss={mean_train_loss:.4f} | "
            f"val recall={rep.recall:.3f} precision={rep.precision:.3f} f1={rep.f1:.3f}{best_marker}"
        )
        epoch_time = time.time() - epoch_start
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch{epoch} time={epoch_time:.1f}s")

        if is_best:
            best_recall = rep.recall
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_recall": best_recall,
                    **model_meta,
                },
                args.out,
            )

        # Update plot every epoch (overwrite)
        plot_path = _save_training_curves(history, args.out)

    training_time = time.time() - training_start
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Stage A training completed")
    print(f"Best val recall: {best_recall:.3f} | Total training time: {training_time/60:.1f} min")
    # Ensure final plot exists (already updated each epoch, but keep for completeness)
    plot_path = _save_training_curves(history, args.out)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training curves saved to: {plot_path}")


if __name__ == "__main__":
    main()


