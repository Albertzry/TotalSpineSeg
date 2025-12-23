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
import tempfile
import shutil
import nibabel as nib
from tqdm import tqdm

from totalspineseg.ldh_twostage.losses import focal_loss_with_logits
from totalspineseg.ldh_twostage.metrics import DetectionReport, average_surface_distance, dice
from totalspineseg.ldh_twostage.models import StageADetectorV2
from totalspineseg.utils.predict_nnunet import predict_nnunet


def load_npz(path: Path):
    with np.load(str(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _load_stagea_detector(ckpt_path: Path, device: "torch.device") -> torch.nn.Module:
    """
    Load Stage A detector checkpoint with backward compatibility.
    Supports:
      - legacy ckpt: {"model": state_dict, ...}
      - new ckpt: {"model": state_dict, "arch": "v1|v2", "model_kwargs": {...}}
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("model", ckpt)
    arch = str(ckpt.get("arch", "v1")).lower()
    kwargs = ckpt.get("model_kwargs", None)

    if arch not in {"v2"}:
        raise SystemExit(
            f"StageA checkpoint arch={arch!r} 已不再支持（旧 StageADetector 已从代码中移除）。\n"
            "请用新的 scripts/train_ldh_stage_a.py 重新训练 StageA（arch=v2），或换用带 arch=v2 的 checkpoint。"
        )

    if isinstance(kwargs, dict):
        det = StageADetectorV2(**kwargs).to(device)
    else:
        det = StageADetectorV2(in_channels=3).to(device)
    det.load_state_dict(state, strict=True)
    det.eval()
    return det


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

    # Resolve Stage A checkpoint path
    if args.ckpt_stagea is None:
        if args.ckpt_dir is not None:
            ckpt_dir = Path(args.ckpt_dir)
            args.ckpt_stagea = ckpt_dir / f"ldh_stageA_fold_{args.fold_stagea}.pth"
        else:
            # Auto-resolve from $TOTALSPINESEG_DATA
            totalspineseg_data = Path(os.environ.get("TOTALSPINESEG_DATA", ""))
            if totalspineseg_data:
                # Look for Dataset105_* in nnUNet/results
                candidates = sorted((totalspineseg_data / "nnUNet" / "results").glob("Dataset105_*"))
                if candidates:
                    d105_name = candidates[0].name
                    ckpt_dir = totalspineseg_data / "nnUNet" / "results" / d105_name / "ldh_twostage" / "checkpoints"
                    args.ckpt_stagea = ckpt_dir / f"ldh_stageA_fold_{args.fold_stagea}.pth"

    # Resolve Stage B model folder (nnUNet Dataset 107)
    if args.model_folder_stageb is None:
        totalspineseg_data = Path(os.environ.get("TOTALSPINESEG_DATA", ""))
        if not totalspineseg_data:
            raise SystemExit(
                "无法自动解析 Stage B 模型路径。请提供 --model-folder-stageb 或设置 TOTALSPINESEG_DATA。"
            )
        # Look for Dataset107_* in nnUNet/results
        candidates = sorted((totalspineseg_data / "nnUNet" / "results").glob("Dataset107_*"))
        if not candidates:
            raise SystemExit(
                f"未找到 Dataset107_* 目录在 {totalspineseg_data/'nnUNet'/'results'}。\n"
                "请确认已训练 Stage B (Dataset 107) 或手动指定 --model-folder-stageb。"
            )
        d107_name = candidates[0].name
        # Model folder: Dataset107_*/nnUNetTrainer__nnUNetPlans__3d_fullres/
        model_folder = totalspineseg_data / "nnUNet" / "results" / d107_name / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        if not model_folder.exists():
            raise SystemExit(
                f"未找到 Stage B 模型文件夹：{model_folder}\n"
                "请确认已训练 Stage B (Dataset 107) 或手动指定 --model-folder-stageb。"
            )
        args.model_folder_stageb = model_folder
    args.model_folder_stageb = Path(args.model_folder_stageb)

    missing = [
        name
        for name, value in (
            ("--stagea-patches-dir", args.stagea_patches_dir),
            ("--stageb-rois-dir", args.stageb_rois_dir),
            ("--ckpt-stagea", args.ckpt_stagea),
        )
        if value is None
    ]
    if missing:
        raise SystemExit(f"Missing required arguments: {', '.join(missing)}")

    args.stagea_patches_dir = Path(args.stagea_patches_dir)
    args.stageb_rois_dir = Path(args.stageb_rois_dir)
    args.ckpt_stagea = Path(args.ckpt_stagea)
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
    ap.add_argument(
        "--ckpt-stagea",
        type=Path,
        default=None,
        help="Stage A checkpoint (.pth). If omitted, auto-resolves from --ckpt-dir or $TOTALSPINESEG_DATA.",
    )
    ap.add_argument(
        "--fold-stagea",
        type=int,
        default=0,
        help="Fold number for Stage A checkpoint (default: 0).",
    )
    ap.add_argument("--ckpt-stageb", type=Path, default=None, help="DEPRECATED: Stage B now uses nnUNet (Dataset 107). Use --model-folder-stageb instead.")
    ap.add_argument(
        "--model-folder-stageb",
        type=Path,
        default=None,
        help="nnUNet model folder for Stage B (Dataset 107). Default: auto-resolve from $TOTALSPINESEG_DATA/nnUNet/results/Dataset107_*/nnUNetTrainer__nnUNetPlans__3d_fullres/",
    )
    ap.add_argument("--fold-stageb", type=int, default=0, help="Fold number for Stage B nnUNet model (default: 0).")
    ap.add_argument("--checkpoint-stageb", type=str, default="checkpoint_best.pth", help="Checkpoint name for Stage B (default: checkpoint_best.pth).")
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
    print(f"  model_folder_stageb: {args.model_folder_stageb}")
    print(f"  fold_stageb:        {args.fold_stageb}")
    print(f"  checkpoint_stageb:  {args.checkpoint_stageb}")
    print(f"  split:              {args.split}")

    dataset_root = _dataset_root_from_ldh_twostage_root(args.stagea_patches_dir.parent)
    split_ids = _load_split_ids(dataset_root, args.split)
    if split_ids is not None:
        print(f"  dataset_root:       {dataset_root}")
        print(f"  split cases:        {len(split_ids)}")

    # ---------------- Stage A (disc-level) ----------------
    det = _load_stagea_detector(args.ckpt_stagea, device)

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

    # ---------------- Stage B (ROI segmentation using nnUNet) ----------------
    dices = []
    asds = []
    stageb_files_all = sorted(args.stageb_rois_dir.glob("*.npz"))
    stageb_files = _filter_files_by_split(stageb_files_all, split_ids)
    n_pos = 0
    
    # Filter positive ROIs only
    positive_files = []
    for p in stageb_files:
        d = load_npz(p)
        if int(d.get("has_ldh", 0)) == 1:
            positive_files.append(p)
    
    print(f"Stage B: evaluating {len(positive_files)} positive ROI files (filtered from {len(stageb_files_all)} total)...")
    
    if len(positive_files) == 0:
        print("Stage B: No positive ROIs found. Skipping Stage B evaluation.")
    else:
        # Create temporary directories for nnUNet inference
        with tempfile.TemporaryDirectory(prefix="eval_ldh_stageb_") as tmpdir:
            tmp_images = Path(tmpdir) / "images"
            tmp_output = Path(tmpdir) / "output"
            tmp_images.mkdir(parents=True, exist_ok=True)
            tmp_output.mkdir(parents=True, exist_ok=True)
            
            # Prepare 3-channel NIfTI files for each ROI
            roi_id_to_gt = {}
            for p in tqdm(positive_files, desc="Preparing ROIs for nnUNet", unit="roi"):
                d = load_npz(p)
                sid = str(d.get("sample_id", b"unknown").astype("S").tobytes().decode(errors="ignore")) if "sample_id" in d else "unknown"
                disc_label = int(d.get("disc_label", -1))
                roi_id = f"{sid}_disc{disc_label}"
                
                # Save 3-channel input
                affine = np.eye(4)
                nib.save(nib.Nifti1Image(d["image"].astype(np.float32), affine), tmp_images / f"{roi_id}_0000.nii.gz")
                nib.save(nib.Nifti1Image(d["disc_mask"].astype(np.float32), affine), tmp_images / f"{roi_id}_0001.nii.gz")
                nib.save(nib.Nifti1Image(d["disc_index"].astype(np.float32), affine), tmp_images / f"{roi_id}_0002.nii.gz")
                
                # Store GT mask
                roi_id_to_gt[roi_id] = d["ldh_mask"].astype(np.float32)
            
            # Run nnUNet inference
            print("Running nnUNet inference for Stage B...")
            predict_nnunet(
                model_folder=args.model_folder_stageb,
                images_dir=tmp_images,
                output_dir=tmp_output,
                device=device,
                folds=(args.fold_stageb,),
                checkpoint=args.checkpoint_stageb,
                npp=1,
                nps=1,
                disable_tta=False,
                verbose=False,
                disable_progress_bar=False,
            )
            
            # Load predictions and compute metrics
            for roi_id, gt in tqdm(roi_id_to_gt.items(), desc="Computing metrics", unit="roi"):
                pred_path = tmp_output / f"{roi_id}.nii.gz"
                if not pred_path.exists():
                    continue
                n_pos += 1
                pred_nii = nib.load(str(pred_path))
                pred = np.asanyarray(pred_nii.dataobj).astype(np.float32)
                # nnUNet outputs class labels (0=background, 1=LDH), convert to binary mask
                pred = (pred >= 0.5).astype(np.float32)
                dices.append(dice(pred, gt))
                asds.append(average_surface_distance(pred, gt))

    print("Stage B (ROI fine segmentation, positives only):")
    print(f"  positives evaluated: {n_pos}")
    if n_pos > 0:
        print(f"  Dice (secondary): {float(np.nanmean(np.array(dices))):.3f}")
        print(f"  ASD: {float(np.nanmean(np.array(asds))):.3f}")
    else:
        print("  No positive ROIs evaluated.")


if __name__ == "__main__":
    main()


