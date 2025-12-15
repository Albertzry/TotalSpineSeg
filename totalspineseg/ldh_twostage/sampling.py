from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from .distance_maps import boundary_band


class PatchType(str, Enum):
    LDH_CENTER = "ldh_center"
    LDH_BOUNDARY = "ldh_boundary"
    DISC_BOUNDARY_HARD_NEG = "disc_boundary_hard_negative"
    DISC_INTERIOR_NEG = "disc_interior_negative"


@dataclass(frozen=True)
class PatchSample:
    center_zyx: Tuple[int, int, int]
    patch_type: PatchType


def _choose_one(mask: np.ndarray, rng: np.random.RandomState) -> Optional[Tuple[int, int, int]]:
    coords = np.array(np.nonzero(mask))
    if coords.size == 0:
        return None
    k = int(rng.randint(0, coords.shape[1]))
    z, y, x = coords[:, k].tolist()
    return int(z), int(y), int(x)


def _centroid(mask: np.ndarray) -> Optional[Tuple[int, int, int]]:
    coords = np.array(np.nonzero(mask))
    if coords.size == 0:
        return None
    mean = coords.mean(axis=1)
    return int(round(mean[0])), int(round(mean[1])), int(round(mean[2]))


def sample_four_class_centers(
    disc_mask: np.ndarray,
    ldh_mask: np.ndarray,
    rng: np.random.RandomState,
    disc_boundary_radius: int = 2,
    ldh_boundary_radius: int = 1,
    ldh_exclusion_radius: int = 2,
) -> Dict[PatchType, PatchSample]:
    """
    Mandatory 4-class patch sampling (no random sampling of classes).

    Returns one center per class. If a class has no valid voxels, it falls back
    to disc centroid to keep the pipeline robust.
    """
    disc = disc_mask.astype(bool)
    ldh = ldh_mask.astype(bool)

    disc_ctr = _centroid(disc) or (disc.shape[0] // 2, disc.shape[1] // 2, disc.shape[2] // 2)

    # LDH center (prefer centroid)
    ldh_ctr = _centroid(ldh) or disc_ctr

    # LDH boundary
    ldh_b = boundary_band(ldh.astype(np.uint8), radius=ldh_boundary_radius).astype(bool)
    ldh_b_center = _choose_one(ldh_b, rng) or ldh_ctr

    # Disc boundary hard negative (disc boundary but excluding LDH vicinity)
    disc_b = boundary_band(disc.astype(np.uint8), radius=disc_boundary_radius).astype(bool)
    if ldh.any():
        struct = np.ones((2 * ldh_exclusion_radius + 1,) * 3, dtype=bool)
        ldh_dil = binary_dilation(ldh, structure=struct)
        hard_neg = np.logical_and(disc_b, np.logical_not(ldh_dil))
    else:
        hard_neg = disc_b
    hard_neg_center = _choose_one(hard_neg, rng) or disc_ctr

    # Disc interior negative (eroded disc region excluding LDH)
    struct_in = np.ones((3, 3, 3), dtype=bool)
    disc_in = binary_erosion(disc, structure=struct_in)
    interior_neg = np.logical_and(disc_in, np.logical_not(ldh))
    interior_center = _choose_one(interior_neg, rng) or disc_ctr

    return {
        PatchType.LDH_CENTER: PatchSample(ldh_ctr, PatchType.LDH_CENTER),
        PatchType.LDH_BOUNDARY: PatchSample(ldh_b_center, PatchType.LDH_BOUNDARY),
        PatchType.DISC_BOUNDARY_HARD_NEG: PatchSample(hard_neg_center, PatchType.DISC_BOUNDARY_HARD_NEG),
        PatchType.DISC_INTERIOR_NEG: PatchSample(interior_center, PatchType.DISC_INTERIOR_NEG),
    }


def crop_patch_zyx(
    vol: np.ndarray,
    center_zyx: Tuple[int, int, int],
    patch_size_zyx: Tuple[int, int, int],
    pad_value: float = 0.0,
) -> Tuple[np.ndarray, Tuple[slice, slice, slice], Tuple[int, int, int]]:
    """
    Crop a patch centered at center_zyx from a 3D volume. Pads if needed.

    Returns:
      patch (Z,Y,X), the slices used on the *padded* volume, and center in patch coords.
    """
    if vol.ndim != 3:
        raise ValueError(f"vol must be 3D, got shape={vol.shape}")
    cz, cy, cx = map(int, center_zyx)
    pz, py, px = map(int, patch_size_zyx)
    hz, hy, hx = pz // 2, py // 2, px // 2

    z0, z1 = cz - hz, cz - hz + pz
    y0, y1 = cy - hy, cy - hy + py
    x0, x1 = cx - hx, cx - hx + px

    pad_before = (max(0, -z0), max(0, -y0), max(0, -x0))
    pad_after = (max(0, z1 - vol.shape[0]), max(0, y1 - vol.shape[1]), max(0, x1 - vol.shape[2]))
    if any(pad_before) or any(pad_after):
        vol_p = np.pad(
            vol,
            pad_width=((pad_before[0], pad_after[0]), (pad_before[1], pad_after[1]), (pad_before[2], pad_after[2])),
            mode="constant",
            constant_values=pad_value,
        )
        cz += pad_before[0]
        cy += pad_before[1]
        cx += pad_before[2]
        z0, z1 = cz - hz, cz - hz + pz
        y0, y1 = cy - hy, cy - hy + py
        x0, x1 = cx - hx, cx - hx + px
    else:
        vol_p = vol

    sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
    patch = vol_p[sl]
    center_in_patch = (hz, hy, hx)
    return patch, sl, center_in_patch


