from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt


def signed_distance_map(binary_mask: np.ndarray) -> np.ndarray:
    """
    Compute Signed Distance Map (SDM) for a binary mask.

    Convention:
      - Inside foreground (mask==1): negative distances
      - Outside foreground (mask==0): positive distances
      - Boundary: ~0

    SDM = dist_outside - dist_inside
    """
    if binary_mask.ndim != 3:
        raise ValueError(f"binary_mask must be 3D, got shape={binary_mask.shape}")
    fg = (binary_mask > 0).astype(np.uint8)
    if fg.sum() == 0:
        # all background: positive distances to "empty" are not meaningful; return zeros for stability
        return np.zeros_like(binary_mask, dtype=np.float32)
    if fg.sum() == fg.size:
        # all foreground: similarly return zeros
        return np.zeros_like(binary_mask, dtype=np.float32)

    dist_out = distance_transform_edt(1 - fg)  # distance to nearest fg for background voxels
    dist_in = distance_transform_edt(fg)       # distance to nearest background for fg voxels
    sdm = dist_out - dist_in
    return sdm.astype(np.float32)


def boundary_band(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    Simple boundary extraction as XOR between dilation and erosion.
    Used only for sampling (not the differentiable boundary loss).
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    fg = mask.astype(bool)
    if fg.sum() == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    struct = np.ones((2 * radius + 1,) * 3, dtype=bool)
    dil = binary_dilation(fg, structure=struct)
    ero = binary_erosion(fg, structure=struct)
    band = np.logical_and(dil, np.logical_not(ero))
    return band.astype(np.uint8)


