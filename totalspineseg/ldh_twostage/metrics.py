from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt


def dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = (pred & gt).sum()
    return float((2 * inter + eps) / (pred.sum() + gt.sum() + eps))


def surface_distances(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute surface-to-surface distances from pred->gt and gt->pred.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 or gt.sum() == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    # surface = boundary voxels (6-connectivity approximation via erosion)
    from scipy.ndimage import binary_erosion

    struct = np.ones((3, 3, 3), dtype=bool)
    pred_surf = pred & (~binary_erosion(pred, structure=struct))
    gt_surf = gt & (~binary_erosion(gt, structure=struct))

    dt_gt = distance_transform_edt(~gt_surf)
    dt_pred = distance_transform_edt(~pred_surf)

    d_pred_to_gt = dt_gt[pred_surf]
    d_gt_to_pred = dt_pred[gt_surf]
    return d_pred_to_gt.astype(np.float32), d_gt_to_pred.astype(np.float32)


def average_surface_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    d1, d2 = surface_distances(pred, gt)
    if d1.size == 0 and d2.size == 0:
        return float("nan")
    if d1.size == 0:
        return float(d2.mean())
    if d2.size == 0:
        return float(d1.mean())
    return float((d1.mean() + d2.mean()) / 2.0)


@dataclass
class DetectionReport:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return float(self.tp / denom) if denom > 0 else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return float(self.tp / denom) if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        denom = p + r
        return float((2.0 * p * r) / denom) if denom > 0 else 0.0


