#!/usr/bin/env python3
"""
Evaluation for LDH two-stage pipeline.

Reports (REQUIRED):
  - Disc-level LDH recall (Stage A, aggregated per disc)
  - Lesion-wise detection rate (same aggregation, recall-focused)
  - Dice (Stage B, secondary)
  - Average surface distance (Stage B)
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

# Add repository root to Python path for imports
_script_dir = Path(__file__).parent.resolve()
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import torch
from tqdm import tqdm

from totalspineseg.ldh_twostage.losses import focal_loss_with_logits
from totalspineseg.ldh_twostage.metrics import DetectionReport, average_surface_distance, dice
from totalspineseg.ldh_twostage.models import SmallUNet3D, StageADetector


def load_npz(path: Path):
    with np.load(str(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _default_data_root() -> Path | None:
    base = os.environ.get("TOTALSPINESEG_DATA")
    if not base:
        return None
    return Path(base) / "nnUNet" / "raw" / "Dataset105_TotalSpineSeg_LDH" / "ldh_twostage"


def _resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    data_root = args.data_root
    if data_root is None:
        data_root = _default_data_root()

    if args.stagea_patches_dir is None or args.stageb_rois_dir is None:
        if data_root is None:
            raise SystemExit(
                "Missing data dirs. Provide --data-root (or set $TOTALSPINESEG_DATA) or pass "
                "--stagea-patches-dir/--stageb-rois-dir explicitly."
            )
        if args.stagea_patches_dir is None:
            args.stagea_patches_dir = Path(data_root) / "stageA_patches"
        if args.stageb_rois_dir is None:
            args.stageb_rois_dir = Path(data_root) / "stageB_rois"

    if args.ckpt_dir is not None:
        ckpt_dir = Path(args.ckpt_dir)
        if args.ckpt_stagea is None:
            args.ckpt_stagea = ckpt_dir / "ldh_stageA_fold_0.pth"
        if args.ckpt_stageb is None:
            args.ckpt_stageb = ckpt_dir / "ldh_stageB_fold_0.pth"

    missing = [
        name
        for name, value in (
            ("--stagea-patches-dir", args.stagea_patches_dir),
            ("--stageb-rois-dir", args.stageb_rois_dir),
            ("--ckpt-stagea", args.ckpt_stagea),
            ("--ckpt-stageb", args.ckpt_stageb),
        )
        if value is None
    ]
    if missing:
        raise SystemExit(f"Missing required arguments: {', '.join(missing)}")

    args.stagea_patches_dir = Path(args.stagea_patches_dir)
    args.stageb_rois_dir = Path(args.stageb_rois_dir)
    args.ckpt_stagea = Path(args.ckpt_stagea)
    args.ckpt_stageb = Path(args.ckpt_stageb)
    return args


def _dataset_root_from_ldh_twostage_root(ldh_twostage_root: Path) -> Path:
    # ldh_twostage root is: .../nnUNet/raw/Dataset105_TotalSpineSeg_LDH/ldh_twostage
    # dataset root is parent folder
    return Path(ldh_twostage_root).parent


def _load_split_ids(dataset_root: Path, split: str) -> set[str] | None:
    split = str(split).lower()
    if split in ("all", "any"):
        return None
    if split not in ("train", "test"):
        raise SystemExit(f"Invalid --split {split!r}. Use: all|train|test")

    if split == "train":
        images_dir = dataset_root / "imagesTr"
    else:
        images_dir = dataset_root / "imagesTs"

    ids = set()
    for p in images_dir.glob("*_0000.nii.gz"):
        ids.add(p.name.replace("_0000.nii.gz", ""))
    if not ids:
        raise SystemExit(f"No cases found for split={split} under {images_dir}")
    return ids


def _filter_files_by_split(files: list[Path], split_ids: set[str] | None) -> list[Path]:
    """Filter files by sample_id prefix if split_ids is provided."""
    if split_ids is None:
        return files
    
    filtered = []
    for f in files:
        # File format: {sample_id}_disc{disc_label}_patch{idx}.npz or similar
        # Extract sample_id from filename (everything before first underscore + disc)
        fname = f.stem
        # Find sample_id: typically "sub-LDH{num}_{date}" format
        for sid in split_ids:
            if fname.startswith(sid + "_"):
                filtered.append(f)
                break
    return filtered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Root of ldh_twostage data dir containing stageA_patches/ and stageB_rois/. "
            "If omitted, uses $TOTALSPINESEG_DATA/nnUNet/raw/Dataset105_TotalSpineSeg_LDH/ldh_twostage when available."
        ),
    )
    ap.add_argument("--stagea-patches-dir", type=Path, default=None)
    ap.add_argument("--stageb-rois-dir", type=Path, default=None)
    ap.add_argument(
        "--ckpt-dir",
        type=Path,
        default=None,
        help="Directory containing ldh_stageA_fold_0.pth and ldh_stageB_fold_0.pth.",
    )
    ap.add_argument("--ckpt-stagea", type=Path, default=None)
    ap.add_argument("--ckpt-stageb", type=Path, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["all", "train", "test"],
        help=(
            "Which Dataset105 split to evaluate. Uses Dataset105/imagesTr or imagesTs to filter by sample_id. "
            "Note: this only filters the exported stageA_patches/stageB_rois; it assumes they were generated from Dataset105." 
        ),
    )
    args = _resolve_paths(ap.parse_args())

    device = torch.device(args.device)

    print("LDH two-stage evaluation")
    print(f"  device: {device}")
    print(f"  thresh: {args.thresh}")
    print(f"  stageA_patches_dir: {args.stagea_patches_dir}")
    print(f"  stageB_rois_dir:    {args.stageb_rois_dir}")
    print(f"  ckpt_stagea:        {args.ckpt_stagea}")
    print(f"  ckpt_stageb:        {args.ckpt_stageb}")
    print(f"  split:              {args.split}")

    dataset_root = _dataset_root_from_ldh_twostage_root(args.stagea_patches_dir.parent)
    split_ids = _load_split_ids(dataset_root, args.split)
    if split_ids is not None:
        print(f"  dataset_root:       {dataset_root}")
        print(f"  split cases:        {len(split_ids)}")

    # ---------------- Stage A (disc-level) ----------------
    det = StageADetector(in_channels=3).to(device)
    det.load_state_dict(torch.load(args.ckpt_stagea, map_location="cpu")["model"])
    det.eval()

    # aggregate per disc: key=(sample_id, disc_label)
    probs_by_disc = defaultdict(list)
    gt_by_disc = {}

    stagea_files_all = sorted(args.stagea_patches_dir.glob("*.npz"))
    stagea_files = _filter_files_by_split(stagea_files_all, split_ids)
    print(f"Stage A: evaluating {len(stagea_files)} patch files (filtered from {len(stagea_files_all)} total)...")
    for p in tqdm(stagea_files, desc="Stage A patches", unit="patch"):
        d = load_npz(p)
        sid = str(d.get("sample_id", b"unknown").astype("S").tobytes().decode(errors="ignore")) if "sample_id" in d else "unknown"
        disc_label = int(d.get("disc_label", -1))
        has_ldh = int(d.get("has_ldh", 0))

        x = np.stack(
            [d["image"].astype(np.float32), d["disc_mask"].astype(np.float32), d["disc_index"].astype(np.float32)],
            axis=0,
        )[None]
        with torch.no_grad():
            prob = torch.sigmoid(det(torch.from_numpy(x).to(device))).item()
        probs_by_disc[(sid, disc_label)].append(prob)
        gt_by_disc[(sid, disc_label)] = has_ldh

    rep = DetectionReport()
    for k, probs in probs_by_disc.items():
        pmax = float(np.max(np.array(probs, dtype=np.float32)))
        pred = 1 if pmax >= args.thresh else 0
        gt = int(gt_by_disc.get(k, 0))
        if pred == 1 and gt == 1:
            rep.tp += 1
        elif pred == 1 and gt == 0:
            rep.fp += 1
        elif pred == 0 and gt == 0:
            rep.tn += 1
        else:
            rep.fn += 1

    print("Stage A (disc-level detection):")
    print(f"  disc-level recall: {rep.recall:.3f}")
    print(f"  precision (secondary): {rep.precision:.3f}")
    print(f"  f1 score: {rep.f1:.3f}")
    print(f"  counts: tp={rep.tp} fp={rep.fp} tn={rep.tn} fn={rep.fn}")
    print(f"  lesion-wise detection rate (proxy): {rep.recall:.3f}")

    # ---------------- Stage B (ROI segmentation) ----------------
    seg = SmallUNet3D(in_channels=3).to(device)
    seg.load_state_dict(torch.load(args.ckpt_stageb, map_location="cpu")["model"])
    seg.eval()

    dices = []
    asds = []
    stageb_files_all = sorted(args.stageb_rois_dir.glob("*.npz"))
    stageb_files = _filter_files_by_split(stageb_files_all, split_ids)
    n_pos = 0
    print(f"Stage B: evaluating {len(stageb_files)} ROI files (filtered from {len(stageb_files_all)} total, positives only for metrics)...")
    for p in tqdm(stageb_files, desc="Stage B ROIs", unit="roi"):
        d = load_npz(p)
        sid = str(d.get("sample_id", b"unknown").astype("S").tobytes().decode(errors="ignore")) if "sample_id" in d else "unknown"
        if int(d.get("has_ldh", 0)) != 1:
            continue
        n_pos += 1
        x = np.stack(
            [d["image"].astype(np.float32), d["disc_mask"].astype(np.float32), d["disc_index"].astype(np.float32)],
            axis=0,
        )[None]
        gt = d["ldh_mask"].astype(np.float32)
        with torch.no_grad():
            mask_logits, _sdm_pred = seg(torch.from_numpy(x).to(device))
            pred = (torch.sigmoid(mask_logits) >= args.thresh).float().cpu().numpy()[0, 0]
        dices.append(dice(pred, gt))
        asds.append(average_surface_distance(pred, gt))

    print("Stage B (ROI fine segmentation, positives only):")
    print(f"  positives evaluated: {n_pos}")
    print(f"  Dice (secondary): {float(np.nanmean(np.array(dices))):.3f}")
    print(f"  ASD: {float(np.nanmean(np.array(asds))):.3f}")


if __name__ == "__main__":
    main()


