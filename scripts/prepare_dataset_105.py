#!/usr/bin/env python3
"""
Prepare Dataset 105 for LDH Specialized Training

This script prepares Dataset 105 which contains only LDH samples for specialized
LDH (Lumbar Disc Herniation) training. The preparation process:

1. Build Dataset105 directly from raw BIDS-like folders:
   - images: /opt/data/private/data_sum/bids/data_ori/images/*.nii.gz
   - labels: /opt/data/private/data_sum/bids/data_ori/labels/*.nii.gz
   The raw filenames may contain Chinese characters; output names are sanitized to keep only
   numeric ID + date (digits), e.g. "100倪全金 20130610.nii.gz" -> "sub-LDH100_20130610".
2. Preprocess (similar to scripts/prepare_datasets.sh):
   - Convert 4D to 3D (average over last dim if needed)
   - Reorient to canonical space
   - Resample images to 1x1x1mm
   - Transform labels to image space
3. Optional lightweight augmentation (default small n; can disable) to increase data diversity
   before Step2 inference.
4. Two-stage LDH pipeline:
   - Run Step2 inference and postprocess to full anatomical labels (labelsStep2Full)
   - Export per-disc samples with mandatory 4-class patch sampling (Stage A) + ROI patches (Stage B)

Usage:
    python scripts/prepare_dataset_105.py

Environment Variables:
    TOTALSPINESEG: Path to TotalSpineSeg repository
    TOTALSPINESEG_DATA: Path to TotalSpineSeg data folder
    TOTALSPINESEG_JOBS: Number of CPU workers (default: 12)

Requirements:
    - Step 2 model (Dataset 102) must be trained (for disc index prior)
"""

import os
import sys
import json
import shutil
import argparse
import re
import subprocess
import multiprocessing as mp
import zlib
import itertools
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm

# IMPORTANT: This script runs nnUNet inference on CUDA, then runs multiprocessing-heavy CPU steps.
# On Linux, the default multiprocessing start method is "fork", which can deadlock after CUDA was initialized.
# Force "spawn" to avoid "CUDA + fork" deadlocks (safe but slightly slower).
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # start method already set (e.g., when running under certain launchers)
    pass

# Add repository root to Python path for imports
_script_dir = Path(__file__).parent.resolve()
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import nibabel as nib

# Two-stage LDH pipeline utilities
from totalspineseg.utils.average4d import average4d_mp
from totalspineseg.utils.reorient_canonical import reorient_canonical_mp
from totalspineseg.utils.resample import resample_mp
from totalspineseg.utils.largest_component import largest_component_mp
from totalspineseg.utils.iterative_label import iterative_label_mp
from totalspineseg.utils.fill_canal import fill_canal_mp
from totalspineseg.utils.transform_seg2image import transform_seg2image_mp
from totalspineseg.utils.crop_image2seg import crop_image2seg_mp
from totalspineseg.utils.extract_alternate import extract_alternate_mp
from totalspineseg.utils.predict_nnunet import predict_nnunet

from totalspineseg.ldh_twostage.disc_index import (
    DiscIndexSpec,
    make_disc_index_map_from_step2_full_labels,
)
from totalspineseg.ldh_twostage.distance_maps import signed_distance_map
from totalspineseg.ldh_twostage.sampling import crop_patch_zyx, sample_four_class_centers

# Fix for PyTorch 2.6+ weights_only issue when loading old checkpoints
try:
    import torch
    import torch.serialization
    # Add numpy scalar to safe globals for loading old checkpoints
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
except Exception:
    pass


def get_env_paths() -> Tuple[Path, Path, int]:
    """Get paths from environment variables"""
    totalspineseg = Path(os.environ.get('TOTALSPINESEG', 'totalspineseg')).resolve()
    totalspineseg_data = Path(os.environ.get('TOTALSPINESEG_DATA', 'data')).resolve()
    # 限制最大线程数为 6
    jobs = min(int(os.environ.get('TOTALSPINESEG_JOBS', '6')), 6)
    
    return totalspineseg, totalspineseg_data, jobs


def _run_transform_seg2image_subprocess(
    images_path: Path,
    segs_path: Path,
    output_segs_path: Path,
    *,
    prefix: str = "",
    image_suffix: str = "_0000",
    seg_suffix: str = "",
    output_seg_suffix: str = "",
    interpolation: str = "nearest",
    overwrite: bool = True,
    max_workers: int = 2,
    quiet: bool = False,
) -> None:
    """
    Run transform_seg2image in a fresh Python process.

    Why: avoid Linux "CUDA initialized + fork multiprocessing" deadlocks in the main process.
    This subprocess is CPU-only and can safely use multiprocessing with fork/spawn.
    """
    cmd = [
        sys.executable,
        "-m",
        "totalspineseg.utils.transform_seg2image",
        "--images-dir",
        str(images_path),
        "--segs-dir",
        str(segs_path),
        "--output-segs-dir",
        str(output_segs_path),
        "--prefix",
        str(prefix),
        "--image-suffix",
        str(image_suffix),
        "--seg-suffix",
        str(seg_suffix),
        "--output-seg-suffix",
        str(output_seg_suffix),
        "--interpolation",
        str(interpolation),
        "--max-workers",
        str(int(max_workers)),
    ]
    if overwrite:
        cmd.append("--overwrite")
    if quiet:
        cmd.append("--quiet")

    env = os.environ.copy()
    # Ensure repo root is importable for `python -m totalspineseg...`
    env["PYTHONPATH"] = f"{_repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}"
    subprocess.run(cmd, check=True, env=env)


def _extract_patches_for_case_mp(
    img_path_str: str,
    ldh_labels_dir_str: str,
    step2_full_dir_str: str,
    disc_index_dir_str: str,
    out_a_str: str,
    out_b_str: str,
    nnunet_roi_dir_str: str,
    stagea_patch_size: tuple[int, int, int],
    stageb_roi_size: tuple[int, int, int],
    disc_labels: tuple[int, ...],
    seed: int,
    stageb_neg_per_disc: int,
    stageb_hardneg_per_posdisc: int,
) -> tuple[int, int]:
    """
    Worker for multi-process patch extraction (one case).
    Returns: (num_stagea_patches_written, num_stageb_rois_written)
    """
    img_path = Path(img_path_str)
    sid = img_path.name.replace("_0000.nii.gz", "")

    ldh_labels_dir = Path(ldh_labels_dir_str)
    step2_full_dir = Path(step2_full_dir_str)
    disc_index_dir = Path(disc_index_dir_str)
    out_a = Path(out_a_str)
    out_b = Path(out_b_str)
    
    # Dataset 107 paths (if provided)
    images_tr_107 = None
    labels_tr_107 = None
    if nnunet_roi_dir_str and nnunet_roi_dir_str != "None":
        nnunet_roi_dir = Path(nnunet_roi_dir_str)
        images_tr_107 = nnunet_roi_dir / "imagesTr"
        labels_tr_107 = nnunet_roi_dir / "labelsTr"

    seg_step2_path = step2_full_dir / f"{sid}.nii.gz"
    ldh_path = ldh_labels_dir / f"{sid}.nii.gz"
    if (not seg_step2_path.exists()) or (not ldh_path.exists()):
        return 0, 0

    # Stable per-case RNG seed (avoid Python hash randomization)
    sid_seed = zlib.crc32(sid.encode("utf-8")) & 0xFFFFFFFF
    rng = np.random.RandomState((int(seed) + int(sid_seed)) % (2**31 - 1))

    img = np.asanyarray(nib.load(img_path).dataobj).astype(np.float32)
    step2 = np.asanyarray(nib.load(seg_step2_path).dataobj).astype(np.int32)
    ldh = np.asanyarray(nib.load(ldh_path).dataobj).astype(np.uint8)

    spec = DiscIndexSpec.default_lumbar()
    disc_index_nii = make_disc_index_map_from_step2_full_labels(nib.load(seg_step2_path), spec=spec)
    # Save per-case disc index map (Step0 required output)
    disc_index_dir.mkdir(parents=True, exist_ok=True)
    nib.save(disc_index_nii, disc_index_dir / f"{sid}.nii.gz")
    disc_index_map = np.asanyarray(disc_index_nii.dataobj).astype(np.float32)

    # local import (worker safety + slightly faster process start for main module)
    from scipy.ndimage import binary_dilation

    n_a = 0
    n_b = 0

    for disc_label in disc_labels:
        disc_mask = (step2 == int(disc_label)).astype(np.uint8)
        if disc_mask.sum() == 0:
            continue

        # Disc region for training: slightly dilated to include boundary context
        disc_region = binary_dilation(disc_mask.astype(bool), structure=np.ones((7, 7, 7), dtype=bool)).astype(np.uint8)

        has_ldh = int(((ldh > 0) & (disc_region > 0)).any())
        ldh_in_disc = ((ldh > 0) & (disc_region > 0)).astype(np.uint8)

        centers = sample_four_class_centers(disc_region, ldh_in_disc, rng=rng)
        for ptype, ps in centers.items():
            img_p, _sl, _c = crop_patch_zyx(img, ps.center_zyx, stagea_patch_size, pad_value=0.0)
            disc_p, _, _ = crop_patch_zyx(disc_region.astype(np.float32), ps.center_zyx, stagea_patch_size, pad_value=0.0)
            idx_p, _, _ = crop_patch_zyx(disc_index_map.astype(np.float32), ps.center_zyx, stagea_patch_size, pad_value=0.0)
            ldh_p, _, _ = crop_patch_zyx(ldh.astype(np.float32), ps.center_zyx, stagea_patch_size, pad_value=0.0)

            out_name = f"{sid}__disc{disc_label}__{ptype.value}.npz"
            np.savez_compressed(
                out_a / out_name,
                image=img_p.astype(np.float32),
                disc_mask=disc_p.astype(np.float32),
                disc_index=idx_p.astype(np.float32),
                ldh_mask=ldh_p.astype(np.float32),
                has_ldh=np.int8(has_ldh),
                disc_label=np.int16(disc_label),
                sample_id=np.array(sid, dtype="S"),
                patch_type=np.array(ptype.value, dtype="S"),
            )
            n_a += 1

        # ---------------- StageB ROI ----------------
        # IMPORTANT (Scheme D):
        #   We export BOTH positives and "hard negative" ROIs, so StageB learns to output empty masks
        #   when the ROI is wrong or when the disc has no LDH.
        #
        # ROI center rules:
        #   - Positive disc: 1 positive ROI centered at GT LDH centroid (fallback disc centroid)
        #                   + `stageb_hardneg_per_posdisc` negative ROIs sampled from disc_region \ ldh
        #   - Negative disc: `stageb_neg_per_disc` negative ROIs sampled from disc_region
        #
        # Negative ROI labels:
        #   - ldh_mask all zeros
        #   - sdm all zeros (signed_distance_map handles empty mask -> zeros; this is stable)
        #
        # Why it helps:
        #   StageB otherwise only sees positives during training and may hallucinate positives at inference.
        try:
            from scipy.ndimage import binary_erosion
            from totalspineseg.ldh_twostage.distance_maps import boundary_band
        except Exception:
            binary_erosion = None
            boundary_band = None

        def _centroid_zyx(mask_u8: np.ndarray) -> tuple[int, int, int]:
            coords = np.array(np.nonzero(mask_u8))
            if coords.size == 0:
                return (mask_u8.shape[0] // 2, mask_u8.shape[1] // 2, mask_u8.shape[2] // 2)
            cz, cy, cx = coords.mean(axis=1)
            return (int(round(cz)), int(round(cy)), int(round(cx)))

        def _sample_centers_from_mask(mask_u8: np.ndarray, k: int) -> list[tuple[int, int, int]]:
            k = int(max(0, k))
            if k == 0:
                return []
            coords = np.array(np.nonzero(mask_u8 > 0))
            if coords.size == 0:
                return [_centroid_zyx(mask_u8)]
            n = coords.shape[1]
            take = int(min(k, n))
            idx = rng.choice(n, size=take, replace=False)
            out: list[tuple[int, int, int]] = []
            for j in np.atleast_1d(idx):
                z, y, x = coords[:, int(j)].tolist()
                out.append((int(z), int(y), int(x)))
            return out

        if has_ldh == 1:
            coords = np.array(np.nonzero(ldh_in_disc))
            if coords.size > 0:
                cz, cy, cx = coords.mean(axis=1)
                center = (int(round(cz)), int(round(cy)), int(round(cx)))
            else:
                # fallback: disc centroid
                center = _centroid_zyx(disc_region)

            img_roi, _, _ = crop_patch_zyx(img, center, stageb_roi_size, pad_value=0.0)
            disc_roi, _, _ = crop_patch_zyx(disc_region.astype(np.float32), center, stageb_roi_size, pad_value=0.0)
            idx_roi, _, _ = crop_patch_zyx(disc_index_map.astype(np.float32), center, stageb_roi_size, pad_value=0.0)
            ldh_roi, _, _ = crop_patch_zyx(ldh.astype(np.uint8), center, stageb_roi_size, pad_value=0.0)
            sdm = signed_distance_map(ldh_roi.astype(np.uint8))

            out_name = f"{sid}__disc{disc_label}__roi.npz"
            np.savez_compressed(
                out_b / out_name,
                image=img_roi.astype(np.float32),
                disc_mask=disc_roi.astype(np.float32),
                disc_index=idx_roi.astype(np.float32),
                ldh_mask=ldh_roi.astype(np.float32),
                sdm=sdm.astype(np.float32),
                has_ldh=np.int8(1),
                disc_label=np.int16(disc_label),
                sample_id=np.array(sid, dtype="S"),
                patch_type=np.array("roi", dtype="S"),
            )
            n_b += 1

            # Save as nnUNet Dataset 107 sample (only positives for now, or include negatives?)
            # Usually segmentation models are trained on positives or a mix.
            # Here we save the positive ROI.
            if images_tr_107 is not None and labels_tr_107 is not None:
                roi_id = f"{sid}_disc{disc_label}"
                # Use identity affine since these are cropped patches
                affine = np.eye(4)
                
                # Channel 0: Image
                nib.save(nib.Nifti1Image(img_roi.astype(np.float32), affine), images_tr_107 / f"{roi_id}_0000.nii.gz")
                # Channel 1: Disc Mask
                nib.save(nib.Nifti1Image(disc_roi.astype(np.float32), affine), images_tr_107 / f"{roi_id}_0001.nii.gz")
                # Channel 2: Disc Index
                nib.save(nib.Nifti1Image(idx_roi.astype(np.float32), affine), images_tr_107 / f"{roi_id}_0002.nii.gz")
                
                # Label: LDH Mask
                nib.save(nib.Nifti1Image(ldh_roi.astype(np.uint8), affine), labels_tr_107 / f"{roi_id}.nii.gz")

            # Hard negatives for positive discs: sample centers from disc excluding LDH
            if int(stageb_hardneg_per_posdisc) > 0:
                neg_mask = ((disc_region > 0) & (ldh_in_disc == 0)).astype(np.uint8)
                # Prefer boundary/interior sampling if available, otherwise fall back to neg_mask random
                candidate = neg_mask
                if boundary_band is not None and binary_erosion is not None:
                    try:
                        bnd = boundary_band(disc_region.astype(np.uint8), radius=2)
                        interior = binary_erosion(disc_region.astype(bool), structure=np.ones((3, 3, 3), dtype=bool)).astype(np.uint8)
                        candidate = ((bnd > 0) | (interior > 0)).astype(np.uint8)
                        candidate = (candidate & (ldh_in_disc == 0)).astype(np.uint8)
                        if candidate.sum() == 0:
                            candidate = neg_mask
                    except Exception:
                        candidate = neg_mask

                neg_centers = _sample_centers_from_mask(candidate, int(stageb_hardneg_per_posdisc))
                for j, cneg in enumerate(neg_centers):
                    img_roi_n, _, _ = crop_patch_zyx(img, cneg, stageb_roi_size, pad_value=0.0)
                    disc_roi_n, _, _ = crop_patch_zyx(disc_region.astype(np.float32), cneg, stageb_roi_size, pad_value=0.0)
                    idx_roi_n, _, _ = crop_patch_zyx(disc_index_map.astype(np.float32), cneg, stageb_roi_size, pad_value=0.0)
                    zero = np.zeros(stageb_roi_size, dtype=np.float32)
                    out_name_n = f"{sid}__disc{disc_label}__roi_neg{j}.npz"
                    np.savez_compressed(
                        out_b / out_name_n,
                        image=img_roi_n.astype(np.float32),
                        disc_mask=disc_roi_n.astype(np.float32),
                        disc_index=idx_roi_n.astype(np.float32),
                        ldh_mask=zero,
                        sdm=zero,
                        has_ldh=np.int8(0),
                        disc_label=np.int16(disc_label),
                        sample_id=np.array(sid, dtype="S"),
                        patch_type=np.array("roi_neg", dtype="S"),
                    )
                    n_b += 1
        else:
            # Negatives for negative discs
            if int(stageb_neg_per_disc) > 0:
                candidate = disc_region.astype(np.uint8)
                if boundary_band is not None and binary_erosion is not None:
                    try:
                        bnd = boundary_band(disc_region.astype(np.uint8), radius=2)
                        interior = binary_erosion(disc_region.astype(bool), structure=np.ones((3, 3, 3), dtype=bool)).astype(np.uint8)
                        candidate = ((bnd > 0) | (interior > 0)).astype(np.uint8)
                        if candidate.sum() == 0:
                            candidate = disc_region.astype(np.uint8)
                    except Exception:
                        candidate = disc_region.astype(np.uint8)

                neg_centers = _sample_centers_from_mask(candidate, int(stageb_neg_per_disc))
                for j, cneg in enumerate(neg_centers):
                    img_roi_n, _, _ = crop_patch_zyx(img, cneg, stageb_roi_size, pad_value=0.0)
                    disc_roi_n, _, _ = crop_patch_zyx(disc_region.astype(np.float32), cneg, stageb_roi_size, pad_value=0.0)
                    idx_roi_n, _, _ = crop_patch_zyx(disc_index_map.astype(np.float32), cneg, stageb_roi_size, pad_value=0.0)
                    zero = np.zeros(stageb_roi_size, dtype=np.float32)
                    out_name_n = f"{sid}__disc{disc_label}__roi_neg{j}.npz"
                    np.savez_compressed(
                        out_b / out_name_n,
                        image=img_roi_n.astype(np.float32),
                        disc_mask=disc_roi_n.astype(np.float32),
                        disc_index=idx_roi_n.astype(np.float32),
                        ldh_mask=zero,
                        sdm=zero,
                        has_ldh=np.int8(0),
                        disc_label=np.int16(disc_label),
                        sample_id=np.array(sid, dtype="S"),
                        patch_type=np.array("roi_neg", dtype="S"),
                    )
                    n_b += 1

    return n_a, n_b


_DATA_ORI_STEM_RE = re.compile(r"^\s*(\d+)\s*.*?(\d{8,14})\s*$")

def _case_id_from_data_ori_filename(filename: str) -> str:
    """
    Convert a raw filename (may contain Chinese characters) into a sanitized case id.

    Example:
      "100倪全金 20130610.nii.gz" -> "sub-LDH100_20130610"
      "122 杨依骏20120924.nii.gz" -> "sub-LDH122_20120924"
    """
    stem = filename
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    elif stem.endswith(".nii"):
        stem = stem[:-4]
    m = _DATA_ORI_STEM_RE.match(stem)
    if not m:
        raise ValueError(f"Cannot parse numeric id + date from filename stem: {stem!r}")
    num, date = m.group(1), m.group(2)
    return f"sub-LDH{num}_{date}"


def _load_and_binarize_label(label_path: Path) -> np.ndarray:
    """
    Load a label and convert it to a binary uint8 mask.

    - If the label is already {0,1}, keep it.
    - If it contains label 101 (legacy LDH), convert (==101) to 1.
    - Otherwise, treat any non-zero as 1.
    """
    nii = nib.load(str(label_path))
    data = np.asanyarray(nii.dataobj)
    uniq = np.unique(data)

    # Fast path: already binary
    if uniq.size <= 2 and set(uniq.tolist()).issubset({0, 1}):
        return data.astype(np.uint8)

    # Legacy: LDH encoded as 101
    if (data == 101).any():
        return (data == 101).astype(np.uint8)

    # Fallback: any non-zero is foreground
    return (data != 0).astype(np.uint8)


def build_dataset105_from_data_ori(
    data_ori_root: Path,
    dst_dataset: Path,
    jobs: int,
    mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    augmentations_per_image: int = 2,
    no_aug: bool = False,
    aug_profile: str = "ldh_light",
    label_smooth: bool = True,
):
    """
    Build nnUNet-style Dataset105 directly from raw BIDS-like input folders.

    Expected input:
      - {data_ori_root}/images/*.nii.gz
      - {data_ori_root}/labels/*.nii.gz

    Output (in nnUNet raw Dataset105 folder):
      - imagesTr/{case_id}_0000.nii.gz
      - labelsTr/{case_id}.nii.gz   (binary 0/1)
    """
    images_src = data_ori_root / "images"
    labels_src = data_ori_root / "labels"
    if not images_src.exists() or not labels_src.exists():
        raise FileNotFoundError(f"Expected folders not found under {data_ori_root} (need images/ and labels/)")

    img_files = sorted(images_src.glob("*.nii.gz"))
    lbl_files = sorted(labels_src.glob("*.nii.gz"))
    if not img_files:
        raise RuntimeError(f"No images found in {images_src}")
    if not lbl_files:
        raise RuntimeError(f"No labels found in {labels_src}")

    img_basenames = {p.name for p in img_files}
    common = [p for p in lbl_files if p.name in img_basenames]
    if len(common) != len(img_files) or len(common) != len(lbl_files):
        raise RuntimeError(
            "Images/labels are not 1:1 matched by basename. "
            f"n_images={len(img_files)} n_labels={len(lbl_files)} n_paired={len(common)}"
        )

    dst_images_tr = dst_dataset / "imagesTr"
    dst_labels_tr = dst_dataset / "labelsTr"
    dst_images_tr.mkdir(parents=True, exist_ok=True)
    dst_labels_tr.mkdir(parents=True, exist_ok=True)
    
    mapping = {}
    print(f"Copying and renaming {len(img_files)} raw cases into {dst_dataset} ...")
    for img_path in tqdm(img_files, desc="Copying raw images/labels", unit="case"):
        case_id = _case_id_from_data_ori_filename(img_path.name)
        lbl_path = labels_src / img_path.name

        out_img = dst_images_tr / f"{case_id}_0000.nii.gz"
        out_lbl = dst_labels_tr / f"{case_id}.nii.gz"

        # Copy image as-is
        shutil.copy2(img_path, out_img)

        # Load + binarize labels to make downstream consistent
        lbl_nii = nib.load(str(lbl_path))
        lbl_bin = _load_and_binarize_label(lbl_path)
        out_lbl_nii = nib.Nifti1Image(lbl_bin.astype(np.uint8), lbl_nii.affine, lbl_nii.header)
        out_lbl_nii.set_data_dtype(np.uint8)
        out_lbl_nii.set_qform(out_lbl_nii.affine)
        out_lbl_nii.set_sform(out_lbl_nii.affine)
        nib.save(out_lbl_nii, out_lbl)

        mapping[case_id] = {"image": img_path.name, "label": lbl_path.name}

    # Persist the mapping for traceability/debugging
    with open(dst_dataset / "source_name_map.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    # Preprocessing steps (align with scripts/prepare_datasets.sh)
    print("\nPreprocessing images (average4d -> canonical -> resample 1mm) ...")
    average4d_mp(
        images_path=dst_images_tr,
        output_images_path=dst_images_tr,
        overwrite=True,
        max_workers=jobs,
        quiet=False,
    )
    reorient_canonical_mp(
        images_path=dst_images_tr,
        output_images_path=dst_images_tr,
        overwrite=True,
        max_workers=jobs,
        quiet=False,
    )

    # Resampling can be memory-hungry; keep it conservative
    safe_workers = min(2, jobs)
    resample_mp(
        images_path=dst_images_tr,
        output_images_path=dst_images_tr,
        mm=mm,
        overwrite=True,
        max_workers=safe_workers,
        quiet=False,
    )

    print("\nTransforming labels to image space ...")
    transform_seg2image_mp(
        images_path=dst_images_tr,
        segs_path=dst_labels_tr,
        output_segs_path=dst_labels_tr,
        interpolation="nearest",
        overwrite=True,
        max_workers=safe_workers,
        quiet=False,
    )

    # Optional augmentation (lightweight default)
    if not no_aug and augmentations_per_image > 0:
        if aug_profile not in {"ldh_light", "default"}:
            raise ValueError(f"Unknown aug_profile={aug_profile!r}. Expected 'ldh_light' or 'default'.")

        if aug_profile == "default":
            # Keep legacy behavior available for debugging, but not recommended for LDH.
            from totalspineseg.utils.augment import augment_mp
            print(f"\nGenerating augmentations with DEFAULT profile (n={augmentations_per_image} per image) ...")
            augment_mp(
                images_path=dst_images_tr,
                segs_path=dst_labels_tr,
                output_images_path=dst_images_tr,
                output_segs_path=dst_labels_tr,
                prefix="",               # work on all
                image_suffix="_0000",
                seg_suffix="",
                output_image_suffix="_0000",
                output_seg_suffix="",
                augmentations_per_image=augmentations_per_image,
                labels2image=False,
                seg_classes=[1],
                overwrite=False,
                max_workers=safe_workers,
                quiet=False,
            )
        else:
            print(
                f"\nGenerating augmentations with LDH_LIGHT profile "
                f"(n={augmentations_per_image} per image, label_smooth={label_smooth}) ..."
            )
            augment_ldh_light_mp(
                images_path=dst_images_tr,
                segs_path=dst_labels_tr,
                output_images_path=dst_images_tr,
                output_segs_path=dst_labels_tr,
                augmentations_per_image=augmentations_per_image,
                max_workers=safe_workers,
                quiet=False,
                label_smooth=label_smooth,
            )


def augment_ldh_light_mp(
    images_path: Path,
    segs_path: Path,
    output_images_path: Path,
    output_segs_path: Path,
    augmentations_per_image: int = 2,
    max_workers: int = 2,
    quiet: bool = False,
    label_smooth: bool = True,
):
    """
    LDH-specific lightweight augmentation to avoid amplifying jagged label boundaries.

    Strategy:
      - Small spatial transform only: mild affine (rotation/scale/translation), no elastic / no strong anisotropy.
      - Mild intensity transform: gamma + small Gaussian noise + small blur.
      - Optional tiny morphological closing on the output label to reduce 1-voxel "staircase" artifacts.
    """
    from tqdm.contrib.concurrent import process_map
    from functools import partial

    images_path = Path(images_path)
    segs_path = Path(segs_path)
    output_images_path = Path(output_images_path)
    output_segs_path = Path(output_segs_path)

    img_list = sorted(images_path.glob("*_0000.nii.gz"))
    seg_list = [segs_path / p.name.replace("_0000.nii.gz", ".nii.gz") for p in img_list]

    process_map(
        partial(
            _augment_ldh_light_one,
            augmentations_per_image=augmentations_per_image,
            output_images_path=output_images_path,
            output_segs_path=output_segs_path,
            label_smooth=label_smooth,
        ),
        img_list,
        seg_list,
        max_workers=max_workers,
        chunksize=1,
        disable=quiet,
        desc="Augmenting (LDH_LIGHT)",
        unit="case",
    )


def _augment_ldh_light_one(
    image_path: Path,
    seg_path: Path,
    augmentations_per_image: int,
    output_images_path: Path,
    output_segs_path: Path,
    label_smooth: bool = True,
):
    import nibabel as nib
    import numpy as np
    import torchio as tio
    import scipy.ndimage as ndi

    image_path = Path(image_path)
    seg_path = Path(seg_path)
    if not seg_path.is_file():
        print(f"Error: {seg_path}, Segmentation file not found")
        return
    
    image = nib.load(str(image_path))
    seg = nib.load(str(seg_path))

    # Use float32 for torchio transforms; keep original dtype when saving.
    image_data = np.asanyarray(image.dataobj).astype(np.float32)
    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    # LDH-friendly transform params:
    # - small rotation to reduce overfitting to staircase edges
    # - very mild scaling & translation to stay on-disc
    # - minimal blur/noise to match acquisition variability without smearing micro-structures
    xform = tio.Compose(
        [
            tio.RandomAffine(
                scales=(0.97, 1.03),
                degrees=5,
                translation=4,
                image_interpolation="linear",
            ),
            tio.RandomGamma(log_gamma=(-0.10, 0.10)),
            tio.RandomNoise(std=(0.0, 0.02)),
            tio.RandomBlur(std=(0.0, 0.6)),
        ]
    )

    base = image_path.name.replace("_0000.nii.gz", "")

    for i in range(int(augmentations_per_image)):
        out_img = output_images_path / f"{base}_a{i}_0000.nii.gz"
        out_seg = output_segs_path / f"{base}_a{i}.nii.gz"

        # If outputs exist, skip to keep the script resumable
        if out_img.exists() or out_seg.exists():
            continue

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image_data[None, ...], affine=image.affine),
            seg=tio.LabelMap(tensor=seg_data[None, ...], affine=seg.affine),
        )
        out = xform(subject)
        out_img_data = out.image.data.numpy()[0, ...].astype(np.float32)
        out_seg_data = out.seg.data.numpy()[0, ...].round().astype(np.uint8)
    
        # Optional minimal smoothing for jagged boundaries
        if label_smooth:
            # Small closing to remove 1-voxel "stairs"/holes; keep effect minimal
            st = ndi.generate_binary_structure(rank=3, connectivity=1)
            out_seg_data = ndi.binary_closing(out_seg_data.astype(bool), structure=st, iterations=1).astype(np.uint8)

        out_img_nii = nib.Nifti1Image(out_img_data.astype(np.float32), out.image.affine, image.header)
        out_img_nii.set_data_dtype(np.float32)
        out_img_nii.set_qform(out_img_nii.affine)
        out_img_nii.set_sform(out_img_nii.affine)
        out_img.parent.mkdir(parents=True, exist_ok=True)
        nib.save(out_img_nii, str(out_img))

        out_seg_nii = nib.Nifti1Image(out_seg_data.astype(np.uint8), out.seg.affine, seg.header)
        out_seg_nii.set_data_dtype(np.uint8)
        out_seg_nii.set_qform(out_seg_nii.affine)
        out_seg_nii.set_sform(out_seg_nii.affine)
        out_seg.parent.mkdir(parents=True, exist_ok=True)
        nib.save(out_seg_nii, str(out_seg))


def run_step2_inference_and_postprocess(
    dst_dataset: Path,
    nnunet_results: Path,
    jobs: int = 12,
    device: str = "cuda",
) -> None:
    """
    Run Step2 inference (Dataset102) and post-process to full anatomical labels (C1..L5, disc levels 63..100).

    Outputs:
      - dst_dataset/labelsStep2Full/*.nii.gz  (in original image space)
    """
    import torch
    images_dir = dst_dataset / "imagesTr"
    step1_raw = dst_dataset / "step1_raw"
    step1_out = dst_dataset / "step1_output"
    step2_input = dst_dataset / "step2_input"
    step2_raw = dst_dataset / "step2_raw"
    step2_out_cropped = dst_dataset / "step2_output_cropped"
    step2_full = dst_dataset / "labelsStep2Full"
    step2_full.mkdir(parents=True, exist_ok=True)

    # ---------- Step1 predict (needed to build Step2 input channel) ----------
    print("[1/5] Running Step1 inference...")
    def _pick_checkpoint(dataset_dir: Path):
        finals = list(dataset_dir.glob("*__*__*/fold_*/checkpoint_final.pth"))
        if finals:
            return finals[0], "checkpoint_final.pth"
        bests = list(dataset_dir.glob("*__*__*/fold_*/checkpoint_best.pth"))
        if bests:
            return bests[0], "checkpoint_best.pth"
        raise RuntimeError(f"nnUNet checkpoint not found under {dataset_dir}")

    step1_dataset = nnunet_results / "Dataset101_TotalSpineSeg_step1"
    ckpt_path1, ckpt_name1 = _pick_checkpoint(step1_dataset)
    model_dir = ckpt_path1.parent.parent
    fold = ckpt_path1.parent.name.replace("fold_", "")

    step1_raw.mkdir(parents=True, exist_ok=True)
    predict_nnunet(
        model_folder=model_dir,
        images_dir=images_dir,
        output_dir=step1_raw,
        folds=str(fold),
        save_probabilities=False,
        checkpoint=ckpt_name1,
        npp=min(6, jobs),
        nps=min(6, jobs),
        device=torch.device(device) if isinstance(device, str) else device,
    )
    # Largest component + iterative label (disc levels)
    print("[2/5] Post-processing Step1 (largest component + iterative labeling)...")
    print("  - Extracting largest component...")
    largest_component_mp(step1_raw, step1_out, binarize=True, dilate=5, overwrite=True, max_workers=jobs, quiet=False)
    print("  - Iterative labeling (disc levels 63-100)...")
    iterative_label_mp(
        step1_out,
        step1_out,
        selected_disc_landmarks=[2, 5, 3, 4],
        disc_labels=[1, 2, 3, 4, 5],
        disc_landmark_labels=[2, 3, 4, 5],
        disc_landmark_output_labels=[63, 71, 91, 100],
        canal_labels=[8],
        canal_output_label=2,
        cord_labels=[9],
        cord_output_label=1,
        sacrum_labels=[6],
        sacrum_output_label=50,
        map_input_dict={7: 11},
        overwrite=True,
        max_workers=jobs,
        quiet=False,
    )
    print("  - Filling canal...")
    fill_canal_mp(step1_out, step1_out, canal_label=2, cord_label=1, largest_canal=True, largest_cord=True,
                  overwrite=True, max_workers=jobs, quiet=False)
        
    # ---------- Validate Step1 postprocessing results ----------
    # Some cases may fail Step1 labeling (missing disc/canal/landmarks). Those cases cannot build Step2 input
    # reliably. We explicitly filter them out to avoid cascading failures.
    print("[3/5] Validating Step1 results and filtering valid cases...")
    valid_cases = []
    invalid_cases = []
    for f in sorted(step1_out.glob("*.nii.gz")):
        sid = f.name.replace(".nii.gz", "")
        try:
            seg = np.asanyarray(nib.load(f).dataobj).astype(np.int32)
            has_any_disc_level = np.isin(seg, list(range(63, 101))).any()
            has_canal = (seg == 2).any()
            if has_any_disc_level and has_canal:
                valid_cases.append(sid)
            else:
                invalid_cases.append(sid)
        except Exception:
            invalid_cases.append(sid)

    # Also make sure we have corresponding images
    valid_cases = [sid for sid in valid_cases if (images_dir / f"{sid}_0000.nii.gz").exists()]

    if invalid_cases:
        print(f"  ⚠ Step1 labeling failed or incomplete for {len(invalid_cases)} cases; they will be skipped for Step2.")
    print(f"  ✓ {len(valid_cases)} valid cases will proceed to Step2 inference")
    if not valid_cases:
        raise RuntimeError("[Two-stage] No valid cases left after Step1 labeling. Cannot proceed to Step2 inference.")

    # ---------- Build Step2 input (crop + odd-disc channel) ----------
    print(f"[4/5] Building Step2 input (cropping + odd-disc channel) for {len(valid_cases)} valid cases...")
    step2_input.mkdir(parents=True, exist_ok=True)
    # copy _0000 images
    for sid in valid_cases:
        f = images_dir / f"{sid}_0000.nii.gz"
        shutil.copy2(f, step2_input / f.name)

    print("  - Cropping images to segmentation regions...")
    crop_image2seg_mp(step2_input, step1_out, step2_input, margin=10, overwrite=True, max_workers=jobs, quiet=False)
    # NOTE: transform_seg2image uses TorchIO resampling and can be memory-heavy; keep worker count conservative
    step2_input_workers = max(1, min(2, jobs))
    print(f"  - Transforming segmentations to image space... (max_workers={step2_input_workers})", flush=True)
    # Run in a subprocess to avoid potential deadlocks when multiprocessing after CUDA was initialized
    _run_transform_seg2image_subprocess(
        images_path=step2_input,
        segs_path=step1_out,
        output_segs_path=step2_input,
        output_seg_suffix="_0001",
        overwrite=True,
        max_workers=step2_input_workers,
        quiet=False,
    )
    print("  - Extracting odd-disc channel (every other disc)...")
    extract_alternate_mp(
        step2_input,
        step2_input,
        seg_suffix="_0001",
        output_seg_suffix="_0001",
        labels=list(range(63, 101)),
        overwrite=True,
        max_workers=step2_input_workers,
        quiet=False,
    )

    # ---------- Step2 predict ----------
    print("[5/5] Running Step2 inference and post-processing to full labels...")
    step2_dataset = nnunet_results / "Dataset102_TotalSpineSeg_step2"
    ckpt_path2, ckpt_name2 = _pick_checkpoint(step2_dataset)
    model_dir2 = ckpt_path2.parent.parent
    fold2 = ckpt_path2.parent.name.replace("fold_", "")

    step2_raw.mkdir(parents=True, exist_ok=True)
    predict_nnunet(
        model_folder=model_dir2,
        images_dir=step2_input,
        output_dir=step2_raw,
        folds=str(fold2),
        save_probabilities=False,
        checkpoint=ckpt_name2,
        npp=min(6, jobs),
        nps=min(6, jobs),
        device=torch.device(device) if isinstance(device, str) else device,
    )

    # Post-process in cropped space
    print("  - Post-processing Step2: largest component...")
    largest_component_mp(step2_raw, step2_out_cropped, binarize=True, dilate=5, overwrite=True, max_workers=jobs, quiet=False)
    print("  - Post-processing Step2: iterative labeling (C1-L5, disc 63-100)...")
    iterative_label_mp(
        step2_out_cropped,
        step2_out_cropped,
        selected_disc_landmarks=[2, 5, 3, 4],
        disc_labels=[1, 2, 3, 4, 5],
        disc_landmark_labels=[2, 3, 4, 5],
        disc_landmark_output_labels=[63, 71, 91, 100],
        vertebrae_labels=[7, 8, 9],
        vertebrae_landmark_output_labels=[13, 21, 41, 50],
        vertebrae_extra_labels=[6],
        canal_labels=[10],
        canal_output_label=2,
        cord_labels=[11],
        cord_output_label=1,
        sacrum_labels=[9],
        sacrum_output_label=50,
        overwrite=True,
        max_workers=jobs,
        quiet=False,
    )
    print("  - Post-processing Step2: filling canal...")
    fill_canal_mp(step2_out_cropped, step2_out_cropped, canal_label=2, cord_label=1, largest_canal=True, largest_cord=True,
                  overwrite=True, max_workers=jobs, quiet=False)

    # Transform back to original image space
    print("  - Transforming Step2 results back to original image space...")
    _run_transform_seg2image_subprocess(
        images_path=images_dir,
        segs_path=step2_out_cropped,
        output_segs_path=step2_full,
        overwrite=True,
        max_workers=jobs,
        quiet=False,
    )


def export_disc_patches_twostage(
    dst_dataset: Path,
    nnunet_roi_dir: Optional[Path] = None,
    stagea_patch_size: tuple[int, int, int] = (96, 96, 96),
    stageb_roi_size: tuple[int, int, int] = (48, 48, 48),
    stageb_neg_per_disc: int = 1,
    stageb_hardneg_per_posdisc: int = 1,
    disc_labels: tuple[int, ...] = (91, 92, 93, 94, 95, 100),
    seed: int = 42,
    max_workers: int = 4,
):
    """
    Create per-disc patches with mandatory 4-class sampling (StageA) and ROI patches for StageB.

    Output:
      - dst_dataset/ldh_twostage/stageA_patches/*.npz
      - dst_dataset/ldh_twostage/stageB_rois/*.npz
      - (Optional) nnunet_roi_dir/imagesTr/*.nii.gz etc. for Dataset 107

    Scheme D (recommended):
      Export StageB ROIs for BOTH positives and negatives so StageB learns to output empty masks.
      - Positive disc: 1 positive ROI (center at GT LDH centroid) + `stageb_hardneg_per_posdisc` negative ROIs
      - Negative disc: `stageb_neg_per_disc` negative ROIs
    """
    images_dir = dst_dataset / "imagesTr"
    ldh_labels_dir = dst_dataset / "labelsTr"
    step2_full_dir = dst_dataset / "labelsStep2Full"
    disc_index_dir = dst_dataset / "disc_index_maps"
    disc_index_dir.mkdir(parents=True, exist_ok=True)

    out_root = dst_dataset / "ldh_twostage"
    out_a = out_root / "stageA_patches"
    out_b = out_root / "stageB_rois"
    out_a.mkdir(parents=True, exist_ok=True)
    out_b.mkdir(parents=True, exist_ok=True)
    
    nnunet_roi_dir_str = str(nnunet_roi_dir) if nnunet_roi_dir else "None"
    if nnunet_roi_dir:
        (nnunet_roi_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
        (nnunet_roi_dir / "labelsTr").mkdir(parents=True, exist_ok=True)

    img_paths = sorted(images_dir.glob("*_0000.nii.gz"))
    print(f"Processing {len(img_paths)} cases for patch extraction...")

    # Multi-process across cases (keeps a tqdm progress bar, same style as earlier steps).
    # Use conservative worker count to avoid disk thrash / memory spikes.
    max_workers = int(max(1, max_workers))
    try:
        from tqdm.contrib.concurrent import process_map
    except Exception:
        process_map = None
        
    if process_map is None or max_workers == 1:
        for img_path in tqdm(img_paths, desc="Extracting disc patches", unit="case"):
            _extract_patches_for_case_mp(
                str(img_path),
                str(ldh_labels_dir),
                str(step2_full_dir),
                str(disc_index_dir),
                str(out_a),
                str(out_b),
                nnunet_roi_dir_str,
                stagea_patch_size,
                stageb_roi_size,
                disc_labels,
                seed,
                int(stageb_neg_per_disc),
                int(stageb_hardneg_per_posdisc),
            )
    else:
        # With mp.set_start_method('spawn') set at module import, this avoids fork-after-CUDA deadlocks.
        results = process_map(
            _extract_patches_for_case_mp,
            [str(p) for p in img_paths],
            itertools.repeat(str(ldh_labels_dir)),
            itertools.repeat(str(step2_full_dir)),
            itertools.repeat(str(disc_index_dir)),
            itertools.repeat(str(out_a)),
            itertools.repeat(str(out_b)),
            itertools.repeat(nnunet_roi_dir_str),
            itertools.repeat(stagea_patch_size),
            itertools.repeat(stageb_roi_size),
            itertools.repeat(disc_labels),
            itertools.repeat(seed),
            itertools.repeat(int(stageb_neg_per_disc),),
            itertools.repeat(int(stageb_hardneg_per_posdisc)),
            max_workers=max_workers,
            chunksize=1,
            desc="Extracting disc patches",
            unit="case",
        )
        # Force evaluation to surface any worker exceptions early
        _ = list(results)
    
    # Print summary
    stagea_patches = list(out_a.glob("*.npz"))
    stageb_rois = list(out_b.glob("*.npz"))
    print(f"\n✓ Patch extraction completed:")
    print(f"  - StageA patches: {len(stagea_patches)}")
    print(f"  - StageB ROIs: {len(stageb_rois)}")
    
    if nnunet_roi_dir:
        rois_107 = list((nnunet_roi_dir / "labelsTr").glob("*.nii.gz"))
        print(f"  - Dataset 107 ROIs: {len(rois_107)}")


def create_test_split(dst_dataset: Path, test_ratio: float = 0.1) -> None:
    """
    Create test split from training data
    
    Args:
        dst_dataset: Dataset 105 path
        test_ratio: Ratio of data to use for testing
    """
    images_tr = dst_dataset / 'imagesTr'
    labels_tr = dst_dataset / 'labelsTr'
    images_ts = dst_dataset / 'imagesTs'
    labels_ts = dst_dataset / 'labelsTs'
    
    images_ts.mkdir(parents=True, exist_ok=True)
    labels_ts.mkdir(parents=True, exist_ok=True)
    
    # Get all samples
    samples = [f.name.replace('_0000.nii.gz', '') for f in images_tr.glob('*_0000.nii.gz')]
    
    if not samples:
        print("No samples found for test split")
        return
    
    # Random split
    np.random.seed(42)
    np.random.shuffle(samples)
    n_test = max(1, int(len(samples) * test_ratio))
    test_samples = samples[:n_test]
    
    print(f"Moving {n_test} samples to test set...")
    
    for sample_id in test_samples:
        # Move images
        for suffix in ['_0000.nii.gz', '_0001.nii.gz']:
            src = images_tr / f'{sample_id}{suffix}'
            dst = images_ts / f'{sample_id}{suffix}'
            if src.exists():
                shutil.move(str(src), str(dst))
        
        # Move labels
        src = labels_tr / f'{sample_id}.nii.gz'
        dst = labels_ts / f'{sample_id}.nii.gz'
        if src.exists():
            shutil.move(str(src), str(dst))
    
    print(f"Test set created with {n_test} samples")


def create_dataset_json(dst_dataset: Path, resources_path: Path) -> None:
    """
    Create dataset.json for Dataset 105
    
    Args:
        dst_dataset: Dataset 105 path
        resources_path: Path to resources folder
    """
    # Load template
    template_path = resources_path / 'datasets' / 'dataset_step5_ldh.json'
    with open(template_path, 'r') as f:
        dataset_json = json.load(f)
    
    # Count training samples
    labels_tr = dst_dataset / 'labelsTr'
    num_training = len(list(labels_tr.glob('*.nii.gz')))
    dataset_json['numTraining'] = num_training
    
    # Save dataset.json
    output_path = dst_dataset / 'dataset.json'
    with open(output_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)
    
    print(f"Created dataset.json with {num_training} training samples")


def create_dataset107_json(dst_dataset: Path) -> None:
    """
    Create dataset.json for Dataset 107 (LDH ROI)
    """
    labels_tr = dst_dataset / 'labelsTr'
    num_training = len(list(labels_tr.glob('*.nii.gz')))
    
    dataset_json = {
        "channel_names": {
            "0": "MRI",
            "1": "DiscMask",
            "2": "DiscIndex"
        },
        "labels": {
            "background": 0,
            "LDH": 1
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz"
    }
    
    output_path = dst_dataset / 'dataset.json'
    with open(output_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)
    
    print(f"Created Dataset 107 dataset.json with {num_training} training samples")


def main():
    parser = argparse.ArgumentParser(description='Prepare Dataset 105 for LDH training')
    parser.add_argument(
        '--data-ori-root', type=str, default='/opt/data/private/data_sum/bids/data_ori',
        help='Raw input root that contains images/ and labels/ (default: /opt/data/private/data_sum/bids/data_ori)'
    )
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='Ratio of data for testing (default: 0.1)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for nnUNet inference (cuda/cpu)')
    parser.add_argument('--stagea-patch', type=int, default=96,
                        help='StageA patch size (cube), default 96')
    parser.add_argument('--stageb-roi', type=int, default=64,
                        help='StageB ROI size (cube), default 64')
    parser.add_argument('--stageb-neg-per-disc', type=int, default=1,
                        help='Scheme D: number of negative StageB ROIs per negative disc (default: 1)')
    parser.add_argument('--stageb-hardneg-per-posdisc', type=int, default=1,
                        help='Scheme D: number of hard-negative StageB ROIs per positive disc (default: 1)')
    parser.add_argument('--no-aug', action='store_true', default=False,
                        help='Disable raw-level augmentation before Step2 inference')
    parser.add_argument('--augmentations-per-image', type=int, default=2,
                        help='Number of augmentations per image (default: 2)')
    parser.add_argument('--aug-profile', type=str, default='ldh_light', choices=['ldh_light', 'default'],
                        help='Augmentation profile (default: ldh_light). Use default only for debugging.')
    parser.add_argument('--no-label-smooth', action='store_true', default=False,
                        help='Disable minimal label smoothing during augmentation (default: enabled)')
    parser.add_argument('--mm', type=float, nargs='+', default=[1.0],
                        help='Target voxel size in mm, provide 1 or 3 numbers (default: 1.0)')
    parser.add_argument('--patch-workers', type=int, default=None,
                        help='Workers for patch extraction (default: min(4, TOTALSPINESEG_JOBS)).')
    args = parser.parse_args()
    
    # Get paths
    totalspineseg, totalspineseg_data, jobs = get_env_paths()
    
    nnunet_raw = totalspineseg_data / 'nnUNet' / 'raw'
    nnunet_preprocessed = totalspineseg_data / 'nnUNet' / 'preprocessed'
    nnunet_results = totalspineseg_data / 'nnUNet' / 'results'
    resources = totalspineseg / 'totalspineseg' / 'resources'
    
    dst_dataset = nnunet_raw / 'Dataset105_TotalSpineSeg_LDH'
    dst_dataset_107 = nnunet_raw / 'Dataset107_LDH_ROI'
    
    mm = tuple(args.mm if len(args.mm) == 3 else [args.mm[0]] * 3)
    
    print("=" * 60)
    print("Preparing Dataset 105 for LDH Binary Segmentation")
    print("=" * 60)
    print(f"Raw source: {args.data_ori_root}")
    print(f"Destination: {dst_dataset}")
    print(f"Destination (ROI): {dst_dataset_107}")
    print(f"Workers: {jobs}")
    print(f"Resample mm: {mm}")
    print(f"Augmentations per image: {0 if args.no_aug else args.augmentations_per_image}")
    print(f"Aug profile: {args.aug_profile}")
    print(f"Aug label smooth: {False if args.no_aug else (not args.no_label_smooth)}")
    _patch_workers = int(args.patch_workers) if args.patch_workers is not None else int(min(4, jobs))
    print(f"Patch workers: {_patch_workers}")
    print("")
    print("Output Format:")
    print("  - Labels: 0=background, 1=LDH (binary)")
    print("  - Input Channel 0: MRI image")
    print("  - Disc-level samples: per-disc patches + ROI patches")
    print("  - Additional priors: disc_index_maps (derived from Step2 full labels)")
    print("=" * 60)
    
    # Create destination directory
    dst_dataset.mkdir(parents=True, exist_ok=True)
    dst_dataset_107.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Build Dataset105 from raw data_ori (copy/rename + preprocessing + optional augmentation)
    print("\nStep 1: Building Dataset105 from raw data_ori ...")
    build_dataset105_from_data_ori(
        data_ori_root=Path(args.data_ori_root),
        dst_dataset=dst_dataset,
        jobs=jobs,
        mm=mm,
        augmentations_per_image=int(args.augmentations_per_image),
        no_aug=bool(args.no_aug),
        aug_profile=str(args.aug_profile),
        label_smooth=not bool(args.no_label_smooth),
    )

    print("\n" + "=" * 60)
    print("Step 2: Two-stage Pipeline - Step2 Inference & Patch Extraction")
    print("=" * 60)
    run_step2_inference_and_postprocess(dst_dataset, nnunet_results, jobs=jobs, device=args.device)
    print("\nExporting per-disc patches with mandatory 4-class sampling + ROI patches...")
    export_disc_patches_twostage(
        dst_dataset,
        nnunet_roi_dir=dst_dataset_107,
        stagea_patch_size=(args.stagea_patch, args.stagea_patch, args.stagea_patch),
        stageb_roi_size=(args.stageb_roi, args.stageb_roi, args.stageb_roi),
        stageb_neg_per_disc=int(args.stageb_neg_per_disc),
        stageb_hardneg_per_posdisc=int(args.stageb_hardneg_per_posdisc),
        max_workers=_patch_workers,
    )
    
    # Step 3: Create test split
    print("\nStep 3: Creating test split...")
    create_test_split(dst_dataset, args.test_ratio)
    
    # Step 4: Create dataset.json
    print("\nStep 4: Creating dataset.json...")
    create_dataset_json(dst_dataset, resources)
    create_dataset107_json(dst_dataset_107)
    
    # Clean up / keep intermediate directories
    print("\nCleaning up temporary directories...")
    for tmp_dir in ['step1_raw', 'step1_output', 'step2_input', 'step2_raw', 'step2_output_cropped', 'labelsStep2Full', 'disc_index_maps', 'ldh_twostage']:
        tmp_path = dst_dataset / tmp_dir
        if tmp_path.exists():
            # Keep these for debugging, but could remove in production
            # shutil.rmtree(tmp_path)
            print(f"  Keeping {tmp_dir} for reference")
    
    print("\n" + "=" * 60)
    print("Dataset 105 preparation completed!")
    print("=" * 60)
    print(f"\nDataset location: {dst_dataset}")
    print(f"Dataset 107 (ROI) location: {dst_dataset_107}")
    print(f"Training samples: {len(list((dst_dataset / 'labelsTr').glob('*.nii.gz')))}")
    print(f"Test samples: {len(list((dst_dataset / 'labelsTs').glob('*.nii.gz')))}")
    print("\nNext steps:")
    print(f"  1. Train (two-stage): bash scripts/train.sh 105 0")
    print(f"     - expects: {dst_dataset/'ldh_twostage/stageA_patches'} and {dst_dataset/'ldh_twostage/stageB_rois'}")


if __name__ == '__main__':
    main()

