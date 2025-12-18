from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@dataclass(frozen=True)
class PatchRecord:
    path: Path
    has_ldh: int


def index_patches(
    patches_dir: Path, 
    *, 
    desc: str | None = None,
    filter_sample_ids: set[str] | None = None
) -> List[PatchRecord]:
    """
    Index patches from directory, optionally filtering by sample_id.
    
    Args:
        patches_dir: Directory containing .npz patches
        desc: Progress bar description
        filter_sample_ids: If provided, only include patches whose filename starts with one of these sample IDs
    
    Returns:
        List of PatchRecord objects
    """
    patches = sorted(patches_dir.glob("*.npz"))
    
    # Pre-filter files by sample_id if requested
    if filter_sample_ids is not None:
        filtered_patches = []
        for p in patches:
            fname = p.stem
            # Check if filename starts with any of the allowed sample IDs
            # Format: {sample_id}_disc{disc_label}_...
            for sid in filter_sample_ids:
                if fname.startswith(sid + "_"):
                    filtered_patches.append(p)
                    break
        patches = filtered_patches
    
    records: List[PatchRecord] = []
    it = tqdm(patches, desc=(desc or f"Indexing {patches_dir.name}"), unit="patch", leave=False)
    for p in it:
        # Fast path: only read the label field needed for indexing.
        # (Avoid decompressing/loading all arrays in the .npz.)
        try:
            with np.load(str(p), allow_pickle=False) as z:
                if "has_ldh" in z.files:
                    has = int(z["has_ldh"])
                else:
                    has = 0
        except Exception:
            # Corrupted file or partial write: treat as negative and keep going
            has = 0
        records.append(PatchRecord(p, has))
    return records


class StageADataset(Dataset):
    """
    Stage A: disc-level detection.
    Each item is one patch with mandatory sampling type already applied in preparation.
    """

    def __init__(self, patches_dir: Path, filter_sample_ids: set[str] | None = None):
        """
        Initialize Stage A dataset.
        
        Args:
            patches_dir: Directory containing .npz patches
            filter_sample_ids: If provided, only load patches from these sample IDs (prevents data leakage)
        """
        self.records = index_patches(
            patches_dir, 
            desc="Indexing StageA patches",
            filter_sample_ids=filter_sample_ids
        )
        if len(self.records) == 0:
            raise ValueError(f"no .npz found in {patches_dir}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        d = _load_npz(rec.path)
        # (Z,Y,X) -> (C,Z,Y,X)
        img = d["image"].astype(np.float32)[None]
        disc = d["disc_mask"].astype(np.float32)[None]
        disc_idx = d["disc_index"].astype(np.float32)[None]
        x = np.concatenate([img, disc, disc_idx], axis=0)
        y = np.float32(rec.has_ldh)
        return torch.from_numpy(x), torch.tensor(y)


class StageBDataset(Dataset):
    """
    Stage B: ROI fine segmentation (positives + optional hard negatives).
    Expects each .npz to already be the ROI patch with GT mask (+ optional SDM).
    """

    def __init__(
        self,
        rois_dir: Path,
        filter_sample_ids: set[str] | None = None,
        *,
        only_positive: bool = True,
    ):
        """
        Initialize Stage B dataset.
        
        Args:
            rois_dir: Directory containing .npz ROI patches
            filter_sample_ids: If provided, only load patches from these sample IDs (prevents data leakage)
            only_positive: If True (default), keep only has_ldh==1 ROIs (legacy behavior).
                           If False, include negatives as well (recommended for Scheme D).
        """
        records = index_patches(
            rois_dir, 
            desc="Indexing StageB rois",
            filter_sample_ids=filter_sample_ids
        )
        if bool(only_positive):
            self.records = [r for r in records if r.has_ldh == 1]
            if len(self.records) == 0:
                raise ValueError(f"no positive ROIs found in {rois_dir}")
        else:
            self.records = records
            if len(self.records) == 0:
                raise ValueError(f"no ROIs found in {rois_dir}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        d = _load_npz(rec.path)
        img = d["image"].astype(np.float32)[None]
        disc = d["disc_mask"].astype(np.float32)[None]
        disc_idx = d["disc_index"].astype(np.float32)[None]
        x = np.concatenate([img, disc, disc_idx], axis=0)
        mask = d["ldh_mask"].astype(np.float32)[None]
        sdm = d["sdm"].astype(np.float32)[None]
        y = np.int64(rec.has_ldh)
        return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(sdm), torch.tensor(y)


