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


def index_patches(patches_dir: Path, *, desc: str | None = None) -> List[PatchRecord]:
    patches = sorted(patches_dir.glob("*.npz"))
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

    def __init__(self, patches_dir: Path):
        self.records = index_patches(patches_dir, desc="Indexing StageA patches")
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
    Stage B: ROI fine segmentation (positives only).
    Expects each .npz to already be the ROI patch with GT mask (+ optional SDM).
    """

    def __init__(self, rois_dir: Path):
        records = index_patches(rois_dir, desc="Indexing StageB rois")
        self.records = [r for r in records if r.has_ldh == 1]
        if len(self.records) == 0:
            raise ValueError(f"no positive ROIs found in {rois_dir}")

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
        return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(sdm)


