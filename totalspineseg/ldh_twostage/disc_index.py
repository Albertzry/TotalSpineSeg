from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import nibabel as nib


@dataclass(frozen=True)
class DiscIndexSpec:
    """
    Disc-level hard prior specification.

    We encode lumbar disc levels into an index that matches clinical naming:
      - L1/L2 -> 1
      - L2/L3 -> 2
      - L3/L4 -> 3
      - L4/L5 -> 4
      - L5/S  -> 5

    If T12/L1 is present it is mapped to 0 (often not used for LDH).

    The voxel-wise map stores either:
      - a normalized float in [0, 1] (recommended), or
      - an integer index (if normalize=False).
    """

    # Full-label disc codes as produced by Step2 post-processing (`iterative_label`)
    disc_label_to_index: Dict[int, int]
    max_index: int

    @staticmethod
    def default_lumbar() -> "DiscIndexSpec":
        # Step2 post-processing outputs disc labels like: 91(T12-L1), 92(L1-L2), ..., 95(L4-L5), 100(L5-S)
        mapping = {91: 0, 92: 1, 93: 2, 94: 3, 95: 4, 100: 5}
        return DiscIndexSpec(disc_label_to_index=mapping, max_index=5)


def make_disc_index_map_from_step2_full_labels(
    step2_full_labels_nii: nib.Nifti1Image,
    spec: DiscIndexSpec | None = None,
    normalize: bool = True,
    background_value: float = 0.0,
) -> nib.Nifti1Image:
    """
    Create a voxel-wise disc-index prior map from Step2 *post-processed* full labels.

    Expected input: `step2_full_labels_nii` contains disc labels (e.g. 91,92,93,94,95,100).

    Returns:
        NIfTI with float32 data; background is 0 by default; disc voxels store normalized index in [0,1].
    """
    if spec is None:
        spec = DiscIndexSpec.default_lumbar()

    seg = np.asanyarray(step2_full_labels_nii.dataobj).astype(np.int32)
    out = np.full(seg.shape, background_value, dtype=np.float32)

    denom = float(spec.max_index) if normalize and spec.max_index > 0 else 1.0
    for disc_label, idx in spec.disc_label_to_index.items():
        mask = seg == int(disc_label)
        if not np.any(mask):
            continue
        val = float(idx) / denom if normalize else float(idx)
        out[mask] = np.float32(val)

    return nib.Nifti1Image(out, step2_full_labels_nii.affine, step2_full_labels_nii.header)


def disc_mask_from_step2_full_labels(
    step2_full_labels_nii: nib.Nifti1Image,
    disc_labels: Iterable[int],
) -> nib.Nifti1Image:
    seg = np.asanyarray(step2_full_labels_nii.dataobj).astype(np.int32)
    disc_labels = list(map(int, disc_labels))
    mask = np.isin(seg, disc_labels).astype(np.uint8)
    return nib.Nifti1Image(mask, step2_full_labels_nii.affine, step2_full_labels_nii.header)


def get_disc_bbox(mask: np.ndarray, margin: int = 0) -> Tuple[slice, slice, slice]:
    """Return a tight 3D bbox (z,y,x) as slices with optional margin."""
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3D, got shape={mask.shape}")
    coords = np.array(np.nonzero(mask))
    if coords.size == 0:
        raise ValueError("empty mask; cannot compute bbox")
    z0, y0, x0 = coords.min(axis=1)
    z1, y1, x1 = coords.max(axis=1) + 1
    z0 = max(0, z0 - margin)
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    z1 = min(mask.shape[0], z1 + margin)
    y1 = min(mask.shape[1], y1 + margin)
    x1 = min(mask.shape[2], x1 + margin)
    return slice(z0, z1), slice(y0, y1), slice(x0, x1)


