"""
calculate.py
============
Calculate LDH and sagittal alignment parameters from spinal MRI (.nii.gz) + Step2 multi-class segmentation + LDH segmentation masks.
Outputs:
  1) JSON report (all values in physical units: mm or degrees)
  2) PNG visualization snapshots in preview/ folder for each calculated parameter (with measurement lines, angle annotations, mask contours)

Dependencies: SimpleITK, NumPy, SciPy, Matplotlib (OpenCV optional)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import SimpleITK as sitk

from scipy import ndimage
from scipy.spatial import cKDTree

import matplotlib

# Use non-interactive backend for server/headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


# =========================
# User-configurable label dictionary (MUST edit based on your Step2 output IDs)
# =========================
LABEL_MAP: Dict[str, Any] = {
    # Based on tss_map.json - TotalSpineSeg label mapping
    "discs": {
        "C2-C3": 63, "C3-C4": 64, "C4-C5": 65, "C5-C6": 66, "C6-C7": 67,
        "C7-T1": 71,
        "T1-T2": 72, "T2-T3": 73, "T3-T4": 74, "T4-T5": 75, "T5-T6": 76,
        "T6-T7": 77, "T7-T8": 78, "T8-T9": 79, "T9-T10": 80, "T10-T11": 81,
        "T11-T12": 82,
        "T12-L1": 91, "L1-L2": 92, "L2-L3": 93, "L3-L4": 94, "L4-L5": 95,
        "L5-S": 100,
    },
    "vertebrae": {
        "C1": 11, "C2": 12, "C3": 13, "C4": 14, "C5": 15, "C6": 16, "C7": 17,
        "T1": 21, "T2": 22, "T3": 23, "T4": 24, "T5": 25, "T6": 26,
        "T7": 27, "T8": 28, "T9": 29, "T10": 30, "T11": 31, "T12": 32,
        "L1": 41, "L2": 42, "L3": 43, "L4": 44, "L5": 45, "L6": 46, "L7": 47,
        "S": 50, "sacrum": 50, "S1": 50,  # All names for sacrum
    },
    # Spinal canal / CSF: label 2
    "spinal_canal": 2,
    "CSF": 2,
    "SC": 1,  # Spinal cord (if different from canal)
}


# =========================
# Level 1: Helper Functions
# =========================


@dataclass(frozen=True)
class NiftiVolume:
    """SimpleITK volume container with consistent numpy array order."""

    img: sitk.Image
    arr_zyx: np.ndarray  # shape (Z, Y, X)
    spacing_xyz: Tuple[float, float, float]  # (sx, sy, sz) in mm


def load_nifti(path: str) -> NiftiVolume:
    """
    Load NIfTI file and return:
      - arr_zyx: numpy array with order (Z, Y, X) = (SI, AP, LR) indexing
      - spacing_xyz: (sx, sy, sz) in mm
    """
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    # Note: We don't flip here - flipping will be done in save_visualization to match medical image convention
    spacing = img.GetSpacing()  # (x, y, z)
    return NiftiVolume(img=img, arr_zyx=arr, spacing_xyz=(float(spacing[0]), float(spacing[1]), float(spacing[2])))


def resample_to_reference(
    moving_img: sitk.Image,
    reference_img: sitk.Image,
    interpolator: int = sitk.sitkLinear,
) -> sitk.Image:
    """
    Resample moving image to match reference image's space (origin, spacing, direction, size).
    Preserves direction matrix to avoid flipping issues.
    
    Parameters:
    -----------
    moving_img : sitk.Image
        Image to be resampled
    reference_img : sitk.Image
        Reference image (target space)
    interpolator : int
        Interpolation method (default: sitkLinear for images, use sitkNearestNeighbor for masks)
    
    Returns:
    --------
    sitk.Image
        Resampled image in reference space
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    # Use identity transform - SimpleITK will handle the spatial transformation
    # based on the reference image's geometry
    resampler.SetTransform(sitk.Transform())
    # Ensure output has same direction as reference
    resampler.SetOutputDirection(reference_img.GetDirection())
    resampler.SetOutputOrigin(reference_img.GetOrigin())
    resampler.SetOutputSpacing(reference_img.GetSpacing())
    return resampler.Execute(moving_img)


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _standardize_previews_from_raw_preview(output_dir: str) -> None:
    """Move + rename legacy preview images from output_dir/raw/preview into output_dir/previews.

    This enforces a strict flat directory convention:
      - result/previews/vertebrae/
      - result/previews/discs/
      - result/previews/global/

    Files are matched and renamed using the provided Type_Level_Metric rule.
    After moving, empty directories under raw/preview are removed.
    """

    out = Path(output_dir)
    src_root = out / "raw" / "preview"
    if not src_root.exists() or not src_root.is_dir():
        return

    dst_previews = out / "previews"
    dst_vertebrae = dst_previews / "vertebrae"
    dst_discs = dst_previews / "discs"
    dst_global = dst_previews / "global"
    dst_vertebrae.mkdir(parents=True, exist_ok=True)
    dst_discs.mkdir(parents=True, exist_ok=True)
    dst_global.mkdir(parents=True, exist_ok=True)

    global_map = {
        "angle_LL.png": (dst_global, "global_cobb_ll.png"),
        "angle_SS.png": (dst_global, "global_cobb_ss.png"),
        "angle_LSA.png": (dst_global, "global_cobb_lsa.png"),
        "ldh_PD_PA_PAR_PLR.png": (dst_global, "global_herniation_summary.png"),
        "agl_discs.png": (dst_global, "global_intensity_agl.png"),
    }

    rx_vh = re.compile(r"^vh_?(?P<level>[A-Za-z0-9]+)\\.png$", re.IGNORECASE)
    rx_vert_ap = re.compile(r"^vertebra_ap_(?P<level>[A-Za-z0-9]+)\\.png$", re.IGNORECASE)
    rx_dia = re.compile(r"^dia_(?P<level>[A-Za-z0-9\-]+)\\.png$", re.IGNORECASE)
    rx_disc_metrics = re.compile(r"^disc_metrics_(?P<level>[A-Za-z0-9\-]+)\\.png$", re.IGNORECASE)

    for f in src_root.rglob("*.png"):
        if not f.is_file():
            continue

        name = f.name
        dst_dir: Optional[Path] = None
        dst_name: Optional[str] = None

        if name in global_map:
            dst_dir, dst_name = global_map[name]
        else:
            m = rx_vh.match(name)
            if m:
                level = m.group("level").upper()
                dst_dir, dst_name = (dst_vertebrae, f"vert_{level}_vh.png")
            else:
                m = rx_vert_ap.match(name)
                if m:
                    level = m.group("level").upper()
                    dst_dir, dst_name = (dst_vertebrae, f"vert_{level}_ap.png")
                else:
                    m = rx_dia.match(name)
                    if m:
                        level = m.group("level").upper()
                        dst_dir, dst_name = (dst_discs, f"disc_{level}_dia.png")
                    else:
                        m = rx_disc_metrics.match(name)
                        if m:
                            level = m.group("level").upper()
                            dst_dir, dst_name = (dst_discs, f"disc_{level}_dm.png")

        if dst_dir is None or dst_name is None:
            continue

        dst_path = dst_dir / dst_name
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists():
            dst_path.unlink()
        shutil.move(str(f), str(dst_path))

    # Cleanup: remove empty directories under src_root (and parent raw if empty)
    for root, dirs, files in os.walk(str(src_root), topdown=False):
        if not dirs and not files:
            try:
                os.rmdir(root)
            except OSError:
                pass
    raw_dir = out / "raw"
    if raw_dir.exists() and raw_dir.is_dir():
        try:
            raw_dir.rmdir()
        except OSError:
            pass


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, (np.floating, np.integer)):
            x = x.item()
        if isinstance(x, (float, int)) and (math.isfinite(float(x))):
            return float(x)
    except Exception:
        return None
    return None


def get_mask_contour(mask_2d: np.ndarray) -> np.ndarray:
    """
    Return contour points of 2D mask as (N, 2) array, coordinates are (x, y) pixel coordinates.
    - Prefer OpenCV findContours; otherwise fallback to morphological boundary.
    """
    m = (mask_2d > 0).astype(np.uint8)
    if m.max() == 0:
        return np.zeros((0, 2), dtype=np.float32)

    if cv2 is not None:
        # OpenCV contour coordinates are (x, y)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return np.zeros((0, 2), dtype=np.float32)
        pts = np.concatenate(contours, axis=0).reshape(-1, 2).astype(np.float32)
        return pts

    # fallback: mask boundary = mask - erosion(mask)
    er = ndimage.binary_erosion(m.astype(bool))
    edge = (m.astype(bool) & (~er))
    ys, xs = np.where(edge)
    return np.stack([xs, ys], axis=1).astype(np.float32)


def _get_line_intersection(line1: Tuple[Tuple[float, float], Tuple[float, float]], line2: Tuple[Tuple[float, float], Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """
    Calculate intersection point of two lines.
    Each line is defined by two points: ((x0, y0), (x1, y1))
    Returns (x, y) intersection point or None if lines are parallel.
    """
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    
    # Line 1: y = m1*x + b1
    # Line 2: y = m2*x + b2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None  # Lines are parallel
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    return (float(x), float(y))


def _draw_angle_arc(ax: plt.Axes, center: Tuple[float, float], angle_start: float, angle_end: float, radius: float, color: str = "yellow", linewidth: float = 2.0) -> None:
    """
    Draw an arc indicating the angle between two lines.
    
    Parameters:
    -----------
    ax : matplotlib Axes
    center : (x, y) center point of the arc
    angle_start : start angle in degrees
    angle_end : end angle in degrees
    radius : radius of the arc
    color : arc color
    linewidth : arc line width
    """
    from matplotlib.patches import Arc
    arc = Arc(
        center,
        width=2 * radius,
        height=2 * radius,
        angle=0,
        theta1=angle_start,
        theta2=angle_end,
        color=color,
        linewidth=linewidth,
    )
    ax.add_patch(arc)


def save_visualization(
    save_path: str,
    image_2d: np.ndarray,
    title: str,
    overlays: Iterable[Tuple[str, Dict[str, Any]]],
    cmap: str = "gray",
) -> None:
    """
    Save 2D snapshot with overlays (mask overlays/lines/points/text).
    Completely matches infer_ldh.py approach:
    - Normalize using min-max (not percentile)
    - Apply flipud + rot90(k=1) transformations
    - Transform overlay coordinates accordingly

    overlays: iterable of (kind, kwargs)
      kind == "mask": {"mask": 2D bool array, "color": "r" or RGB tuple, "alpha": 0.55}
      kind == "line": {"p0": (x0,y0), "p1": (x1,y1), "color": "y", "lw": 2, "label": "...", "linestyle": "-" (solid) or "--" (dashed)}
      kind == "scatter": {"pts": (N,2), "color":"c", "s": 10, "label": "..."}
      kind == "text": {"xy": (x,y), "text": "...", "color":"w", "fontsize": 14}
      kind == "bbox": {"xywh": (x,y,w,h), "color":"g", "lw": 2}
      kind == "arc": {"center": (x,y), "angle_start": deg, "angle_end": deg, "radius": float, "color": "y", "lw": 2}
    """
    _ensure_dir(os.path.dirname(save_path))
    
    # Normalize image to 0-255 range using min-max (same as infer_ldh.py)
    # infer_ldh.py: denom = (np.max(slice_img) - np.min(slice_img)) or 1.0
    #               slice_img_u8 = (255 * (slice_img - np.min(slice_img)) / denom).astype(np.uint8)
    img_min = float(np.min(image_2d))
    img_max = float(np.max(image_2d))
    denom = (img_max - img_min) or 1.0
    img_u8 = (255 * (image_2d - img_min) / denom).astype(np.uint8)
    
    # Convert to RGB (three channels)
    rgb = np.repeat(img_u8[:, :, None], 3, axis=2).astype(np.float32)
    
    # Store original shape
    h_img_orig, w_img_orig = rgb.shape[:2]
    
    # Apply 180 degree rotation to rgb image
    rgb = np.rot90(rgb, k=2)  # Rotate 180 degrees
    
    # Apply mask overlays with same transformation
    for kind, kw in overlays:
        if kind == "mask":
            mask = kw.get("mask")
            if mask is None or mask.size == 0 or not mask.any():
                continue
            if mask.shape != (h_img_orig, w_img_orig):
                continue
            
            # Apply 180 degree rotation to mask (same as rgb)
            mask_transformed = np.rot90(mask, k=2)  # Rotate 180 degrees
            
            # Check shape match
            if mask_transformed.shape != rgb.shape[:2]:
                continue
            
            # Get color
            color_str = kw.get("color", "r")
            if isinstance(color_str, str):
                # Map color names to RGB
                color_map = {
                    "r": (255, 0, 0),      # red
                    "g": (0, 255, 0),      # green
                    "b": (0, 0, 255),      # blue
                    "y": (255, 255, 0),    # yellow
                    "c": (0, 255, 255),    # cyan
                    "m": (255, 0, 255),    # magenta
                    "lime": (0, 255, 0),   # lime green
                    "green": (0, 255, 0),  # green
                    "orange": (255, 165, 0), # orange
                    "red": (255, 0, 0),    # red
                    "blue": (0, 0, 255),   # blue
                    "yellow": (255, 255, 0), # yellow
                    "cyan": (0, 255, 255), # cyan
                    "magenta": (255, 0, 255), # magenta
                }
                overlay_color = np.array(color_map.get(color_str.lower(), (255, 0, 0)), dtype=np.float32)
            else:
                overlay_color = np.array(color_str, dtype=np.float32)
            
            alpha = float(kw.get("alpha", 0.55))
            # Blend: rgb[mask] = rgb[mask] * (1 - alpha) + overlay_color * alpha
            # Apply to transformed mask (after all transformations)
            rgb[mask_transformed] = rgb[mask_transformed] * (1.0 - alpha) + overlay_color * alpha
    
    # After transformations, image shape is (h_img_orig, w_img_orig)
    h_img_final, w_img_final = rgb.shape[:2]
    
    # Create figure with fixed size (original style)
    fig = plt.figure(figsize=(7, 7), dpi=160)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(np.clip(rgb, 0, 255).astype(np.uint8), origin='upper')
    ax.set_title(title, fontsize=16)
    ax.axis("off")

    # Add other overlays (lines, text, etc.)
    # Coordinate transformation after all transformations (flipud + rot90(k=1) + rot90(k=-1) + fliplr):
    # Verified with test: (x=50, y=30) -> (x=149, y=69) for (h=100, w=200)
    # Final transformation: (x, y) -> (w_orig - 1 - x, h_orig - 1 - y)
    for kind, kw in overlays:
        if kind == "mask":
            continue  # Already processed
        elif kind == "line":
            p0 = kw["p0"]
            p1 = kw["p1"]
            # Transform coordinates: (x, y) -> (w_orig - 1 - x, h_orig - 1 - y)
            p0_transformed = (w_img_orig - 1 - p0[0], h_img_orig - 1 - p0[1])
            p1_transformed = (w_img_orig - 1 - p1[0], h_img_orig - 1 - p1[1])
            linestyle = kw.get("linestyle", "-")
            ax.plot([p0_transformed[0], p1_transformed[0]], [p0_transformed[1], p1_transformed[1]], color=kw.get("color", "y"), linewidth=kw.get("lw", 2), linestyle=linestyle)
            if kw.get("label"):
                # Place label offset from the line to avoid overlap
                mx = (p0_transformed[0] + p1_transformed[0]) * 0.5
                my = (p0_transformed[1] + p1_transformed[1]) * 0.5
                # Perpendicular offset (8 px away from line)
                dx = p1_transformed[0] - p0_transformed[0]
                dy = p1_transformed[1] - p0_transformed[1]
                length = max(np.sqrt(dx * dx + dy * dy), 1e-9)
                # Normal vector (perpendicular)
                nx, ny = -dy / length, dx / length
                offset_px = 8.0
                ax.text(
                    mx + nx * offset_px,
                    my + ny * offset_px,
                    str(kw["label"]),
                    color=kw.get("color", "y"),
                    fontsize=kw.get("fontsize", 12),
                    ha="center", va="center",
                    bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1),
                )
        elif kind == "scatter":
            pts = kw.get("pts")
            if pts is None or len(pts) == 0:
                continue
            # Transform coordinates for 180 degree rotation: (x, y) -> (w - 1 - x, h - 1 - y)
            pts_transformed = pts.copy()
            pts_transformed[:, 0] = w_img_orig - 1 - pts[:, 0]
            pts_transformed[:, 1] = h_img_orig - 1 - pts[:, 1]
            ax.scatter(pts_transformed[:, 0], pts_transformed[:, 1], c=kw.get("color", "c"), s=kw.get("s", 10))
        elif kind == "text":
            xy = kw["xy"]
            # Transform coordinates for 180 degree rotation: (x, y) -> (w - 1 - x, h - 1 - y)
            xy_transformed = (w_img_orig - 1 - xy[0], h_img_orig - 1 - xy[1])
            ax.text(
                xy_transformed[0],
                xy_transformed[1],
                str(kw["text"]),
                color=kw.get("color", "w"),
                fontsize=kw.get("fontsize", 13),
                ha=kw.get("ha", "left"), va=kw.get("va", "center"),
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1),
            )
        elif kind == "bbox":
            x, y, w, h = kw["xywh"]
            # Transform bbox for 180 degree rotation
            # Calculate corners and transform them
            x0, y0 = x, y
            x1, y1 = x + w, y + h
            # Transform corners: (x, y) -> (w - 1 - x, h - 1 - y)
            p0_t = (w_img_orig - 1 - x0, h_img_orig - 1 - y0)
            p1_t = (w_img_orig - 1 - x1, h_img_orig - 1 - y1)
            # New bbox
            x_new = min(p0_t[0], p1_t[0])
            y_new = min(p0_t[1], p1_t[1])
            w_new = abs(p1_t[0] - p0_t[0])
            h_new = abs(p1_t[1] - p0_t[1])
            rect = plt.Rectangle((x_new, y_new), w_new, h_new, fill=False, edgecolor=kw.get("color", "g"), linewidth=kw.get("lw", 2))
            ax.add_patch(rect)
        elif kind == "arc":
            center = kw["center"]
            # Transform coordinates for 180 degree rotation: (x, y) -> (w - 1 - x, h - 1 - y)
            center_transformed = (w_img_orig - 1 - center[0], h_img_orig - 1 - center[1])
            _draw_angle_arc(
                ax,
                center_transformed,
                kw["angle_start"],
                kw["angle_end"],
                kw.get("radius", 20.0),
                kw.get("color", "yellow"),
                kw.get("lw", 2.0),
            )

    fig.tight_layout(pad=0.1)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _spacing_zyx(spacing_xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
    sx, sy, sz = spacing_xyz
    return (sz, sy, sx)  # arr order is (z,y,x)


def _normalize_intensity_percentile(arr: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    a = arr.astype(np.float32)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a, dtype=np.float32)
    lo = np.percentile(a[finite], p_lo)
    hi = np.percentile(a[finite], p_hi)
    if hi <= lo + 1e-6:
        return np.clip(a - lo, 0.0, 1.0)
    out = (a - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _center_of_mass_idx(mask_zyx: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """
    Calculate center of mass in pixel indices (Z, Y, X).
    Returns (z_com, y_com, x_com) or None if mask is empty.
    Handles both 2D and 3D arrays.
    """
    m = (mask_zyx > 0).astype(np.uint8)
    if m.sum() == 0:
        return None
    com = ndimage.center_of_mass(m)
    # scipy.ndimage.center_of_mass returns a tuple with length matching array dimensions
    # For 2D arrays, it returns (z, y), for 3D it returns (z, y, x)
    if len(com) == 2:
        # 2D array (Z, Y): return (z, y, 0) assuming x=0
        return (float(com[0]), float(com[1]), 0.0)
    elif len(com) == 3:
        # 3D array (Z, Y, X): return (z, y, x)
        return (float(com[0]), float(com[1]), float(com[2]))
    else:
        # Unexpected dimension
        return None


def _mid_sagittal_x_index(step2_zyx: np.ndarray, canal_label: int) -> int:
    canal = (step2_zyx == canal_label).astype(np.uint8)
    com = _center_of_mass_idx(canal)
    if com is not None:
        return int(round(com[2]))
    # fallback: geometric center
    return int(step2_zyx.shape[2] // 2)


def _axial_z_index_max_area(mask_zyx: np.ndarray) -> Optional[int]:
    m = (mask_zyx > 0).astype(np.uint8)
    if m.sum() == 0:
        return None
    areas = m.reshape(m.shape[0], -1).sum(axis=1)
    z = int(np.argmax(areas))
    return z


def _crop_around_mask(mask_2d: np.ndarray, pad: int = 20) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask_2d > 0)
    if len(xs) == 0:
        return None
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad, mask_2d.shape[1] - 1)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad, mask_2d.shape[0] - 1)
    return (x0, y0, x1, y1)


def _apply_crop(img: np.ndarray, crop: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    if crop is None:
        return img
    # Validate crop tuple has 4 elements
    if not isinstance(crop, (tuple, list)) or len(crop) < 4:
        return img
    try:
        x0, y0, x1, y1 = crop
        # Ensure indices are within bounds
        x0 = max(0, min(int(x0), img.shape[1] - 1))
        y0 = max(0, min(int(y0), img.shape[0] - 1))
        x1 = max(x0, min(int(x1), img.shape[1] - 1))
        y1 = max(y0, min(int(y1), img.shape[0] - 1))
        return img[y0 : y1 + 1, x0 : x1 + 1]
    except (ValueError, TypeError, IndexError):
        return img


def _safe_unpack_crop(crop: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int]]:
    """
    Safely unpack crop tuple to get (x0, y0).
    Returns None if crop is invalid.
    """
    if crop is None:
        return None
    if not isinstance(crop, (tuple, list)) or len(crop) < 4:
        return None
    try:
        x0, y0, _, _ = crop
        return (int(x0), int(y0))
    except (ValueError, TypeError, IndexError):
        return None


def _shift_points(pts: np.ndarray, crop: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    if crop is None or pts.size == 0:
        return pts
    crop_xy = _safe_unpack_crop(crop)
    if crop_xy is None:
        return pts
    x0, y0 = crop_xy
    out = pts.copy()
    out[:, 0] -= float(x0)
    out[:, 1] -= float(y0)
    return out


def _pick_anterior_posterior_from_canal(
    body_mask_2d_zy: np.ndarray, canal_mask_2d_zy: Optional[np.ndarray]
) -> Tuple[int, int]:
    """
    On sagittal (Z,Y) plane, return:
      - anterior_y_idx: end that is farther from spinal canal
      - posterior_y_idx: end that is closer to spinal canal

    Since NIfTI orientation is not standardized, use the mean Y position of spinal canal as "posterior" reference (canal is usually posterior to vertebral body).
    """
    ys = np.where(body_mask_2d_zy > 0)[1]
    if ys.size == 0:
        return (0, 0)
    y_min = int(ys.min())
    y_max = int(ys.max())

    if canal_mask_2d_zy is None or canal_mask_2d_zy.sum() == 0:
        # fallback: assume anterior=minY by index order (only as last resort)
        return (y_min, y_max)

    canal_y = np.where(canal_mask_2d_zy > 0)[1]
    if canal_y.size == 0:
        return (y_min, y_max)
    y_c = float(np.mean(canal_y))
    # posterior end should be closer to canal y_c
    if abs(y_min - y_c) < abs(y_max - y_c):
        posterior = y_min
        anterior = y_max
    else:
        posterior = y_max
        anterior = y_min
    return (anterior, posterior)


def _get_canal_axis_direction(canal_zy: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    通过canal找到轴向方向（Z方向）。
    找到canal的上下两个端点，返回轴向方向向量。
    
    Parameters:
    -----------
    canal_zy : 2D binary mask of canal in ZY plane
    
    Returns:
    --------
    axis_direction : (dz, dy) 归一化的方向向量，或者None如果失败
    """
    if canal_zy.sum() == 0:
        return None
    
    # 找到canal的所有点
    zs, ys = np.where(canal_zy > 0)
    if zs.size < 2:
        return None
    
    # 找到Z方向的最小和最大值（上下端点）
    z_min_idx = np.argmin(zs)
    z_max_idx = np.argmax(zs)
    
    z_min = float(zs[z_min_idx])
    z_max = float(zs[z_max_idx])
    y_min = float(ys[z_min_idx])
    y_max = float(ys[z_max_idx])
    
    # 计算方向向量（从下到上）
    dz = z_max - z_min
    dy = y_max - y_min
    
    # 归一化
    norm = np.sqrt(dz**2 + dy**2)
    if norm < 1e-6:
        # 如果canal几乎是水平的，返回垂直方向
        return (1.0, 0.0)
    
    return (float(dz / norm), float(dy / norm))


def _find_vertebra_rectangle_corners(body_zy: np.ndarray) -> Optional[np.ndarray]:
    """
    找到椎体矩形的四个顶点，确保所有点都在椎体mask内部。
    
    方法：
    1. 找到椎体的轮廓
    2. 找到轮廓上四个方向（上、下、左、右）的最远点
    3. 如果外接矩形的角点不在mask内，则使用轮廓上最近的点
    
    Parameters:
    -----------
    body_zy : 2D binary mask of vertebral body in ZY plane
    
    Returns:
    --------
    corners : np.ndarray of shape (4, 2) with four corner points in (y, z) format, or None if failed
    """
    if body_zy.sum() == 0:
        return None
    
    # 获取所有mask内的点坐标
    zs, ys = np.where(body_zy > 0)
    if zs.size == 0:
        return None
    
    # 计算mask的边界框和中心
    z_min, z_max = float(zs.min()), float(zs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    z_center = (z_min + z_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    
    if cv2 is None:
        # Fallback: 使用边界框的四个角，但需要确保在mask内
        # 找到四个象限中最接近边界的点
        corners = []
        
        # 左上角 (y < y_center, z < z_center)
        mask_ul = (ys < y_center) & (zs < z_center)
        if mask_ul.any():
            # 找到距离左上角最远的点（在mask内）
            dists_ul = (ys[mask_ul] - y_min)**2 + (zs[mask_ul] - z_min)**2
            idx_ul = np.argmax(dists_ul)
            corners.append([float(ys[mask_ul][idx_ul]), float(zs[mask_ul][idx_ul])])
        else:
            corners.append([y_min, z_min])
        
        # 右上角 (y >= y_center, z < z_center)
        mask_ur = (ys >= y_center) & (zs < z_center)
        if mask_ur.any():
            dists_ur = (ys[mask_ur] - y_max)**2 + (zs[mask_ur] - z_min)**2
            idx_ur = np.argmax(dists_ur)
            corners.append([float(ys[mask_ur][idx_ur]), float(zs[mask_ur][idx_ur])])
        else:
            corners.append([y_max, z_min])
        
        # 右下角 (y >= y_center, z >= z_center)
        mask_lr = (ys >= y_center) & (zs >= z_center)
        if mask_lr.any():
            dists_lr = (ys[mask_lr] - y_max)**2 + (zs[mask_lr] - z_max)**2
            idx_lr = np.argmax(dists_lr)
            corners.append([float(ys[mask_lr][idx_lr]), float(zs[mask_lr][idx_lr])])
        else:
            corners.append([y_max, z_max])
        
        # 左下角 (y < y_center, z >= z_center)
        mask_ll = (ys < y_center) & (zs >= z_center)
        if mask_ll.any():
            dists_ll = (ys[mask_ll] - y_min)**2 + (zs[mask_ll] - z_max)**2
            idx_ll = np.argmax(dists_ll)
            corners.append([float(ys[mask_ll][idx_ll]), float(zs[mask_ll][idx_ll])])
        else:
            corners.append([y_min, z_max])
        
        return np.array(corners, dtype=np.float32)
    
    # 转换为uint8
    body_uint8 = (body_zy > 0).astype(np.uint8) * 255
    
    # 找到轮廓
    contours, _ = cv2.findContours(body_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    
    # 使用最大轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 找到最小外接矩形作为参考
    rect = cv2.minAreaRect(largest_contour)
    box_ref = cv2.boxPoints(rect)  # 返回4个点，格式为 (x, y) 其中x是列（Y），y是行（Z）
    box_ref = box_ref.astype(np.float32)
    
    # 获取轮廓上的所有点（格式为 (y, z)）
    # OpenCV返回的轮廓点格式为 (x, y)，其中x是列（Y坐标），y是行（Z坐标）
    contour_points = largest_contour.reshape(-1, 2).astype(np.float32)
    if contour_points.shape[0] == 0:
        return None
    
    # 轮廓点已经是 (y, z) 格式（x是列Y，y是行Z）
    contour_yz = contour_points.copy()
    
    # 方法：找到轮廓上四个角落的点
    # 使用加权方法，同时考虑y和z方向，找到真正在角落位置的点
    corners = []
    
    # 左上角：找到轮廓上y坐标最小且z坐标最小的点（角落点）
    # 使用加权评分：同时考虑y和z方向的距离，找到最接近角落的点
    mask_ul = (contour_yz[:, 0] < y_center) & (contour_yz[:, 1] < z_center)
    if mask_ul.any():
        contour_ul = contour_yz[mask_ul]
        if contour_ul.shape[0] > 0:
            # 计算到左上角(y_min, z_min)的距离
            # 使用加权评分，强调同时接近两个边界（真正的角落点）
            # 方法：使用 y_dist * z_dist，这样只有当两个距离都小时，评分才小
            y_dist = contour_ul[:, 0] - y_min  # y方向到最小值的距离
            z_dist = contour_ul[:, 1] - z_min  # z方向到最小值的距离
            # 使用乘积来强调角落：只有当y和z都接近边界时，评分才小
            # 添加小的正则项避免数值问题
            scores_ul = y_dist * z_dist + 0.1 * (y_dist + z_dist)
            idx_ul = np.argmin(scores_ul)  # 找评分最小的点（最接近角落）
            corners.append([float(contour_ul[idx_ul, 0]), float(contour_ul[idx_ul, 1])])
        else:
            # Fallback: 在mask内找到最接近左上角的点
            mask_ul_mask = (ys < y_center) & (zs < z_center)
            if mask_ul_mask.any():
                y_dist = ys[mask_ul_mask] - y_min
                z_dist = zs[mask_ul_mask] - z_min
                scores = y_dist * z_dist + 0.1 * (y_dist + z_dist)
                idx = np.argmin(scores)
                corners.append([float(ys[mask_ul_mask][idx]), float(zs[mask_ul_mask][idx])])
            else:
                corners.append([y_min, z_min])
    else:
        # 如果没有点在这个象限，在mask内找到最接近左上角的点
        mask_ul_mask = (ys < y_center) & (zs < z_center)
        if mask_ul_mask.any():
            y_dist = ys[mask_ul_mask] - y_min
            z_dist = zs[mask_ul_mask] - z_min
            scores = y_dist * z_dist + 0.1 * (y_dist + z_dist)
            idx = np.argmin(scores)
            corners.append([float(ys[mask_ul_mask][idx]), float(zs[mask_ul_mask][idx])])
        else:
            corners.append([y_min, z_min])
    
    # 右上角：找到轮廓上y坐标最大且z坐标最小的点
    mask_ur = (contour_yz[:, 0] >= y_center) & (contour_yz[:, 1] < z_center)
    if mask_ur.any():
        contour_ur = contour_yz[mask_ur]
        if contour_ur.shape[0] > 0:
            y_dist = y_max - contour_ur[:, 0]  # y方向到最大值的距离
            z_dist = contour_ur[:, 1] - z_min  # z方向到最小值的距离
            scores_ur = y_dist * z_dist + 0.1 * (y_dist + z_dist)
            idx_ur = np.argmin(scores_ur)
            corners.append([float(contour_ur[idx_ur, 0]), float(contour_ur[idx_ur, 1])])
        else:
            mask_ur_mask = (ys >= y_center) & (zs < z_center)
            if mask_ur_mask.any():
                y_dist = y_max - ys[mask_ur_mask]
                z_dist = zs[mask_ur_mask] - z_min
                scores = y_dist * z_dist + 0.1 * (y_dist + z_dist)
                idx = np.argmin(scores)
                corners.append([float(ys[mask_ur_mask][idx]), float(zs[mask_ur_mask][idx])])
            else:
                corners.append([y_max, z_min])
    else:
        mask_ur_mask = (ys >= y_center) & (zs < z_center)
        if mask_ur_mask.any():
            y_dist = y_max - ys[mask_ur_mask]
            z_dist = zs[mask_ur_mask] - z_min
            scores = y_dist * z_dist + 0.1 * (y_dist + z_dist)
            idx = np.argmin(scores)
            corners.append([float(ys[mask_ur_mask][idx]), float(zs[mask_ur_mask][idx])])
        else:
            corners.append([y_max, z_min])
    
    # 右下角：找到轮廓上y坐标最大且z坐标最大的点
    mask_lr = (contour_yz[:, 0] >= y_center) & (contour_yz[:, 1] >= z_center)
    if mask_lr.any():
        contour_lr = contour_yz[mask_lr]
        if contour_lr.shape[0] > 0:
            y_dist = y_max - contour_lr[:, 0]  # y方向到最大值的距离
            z_dist = z_max - contour_lr[:, 1]  # z方向到最大值的距离
            scores_lr = y_dist * z_dist + 0.1 * (y_dist + z_dist)
            idx_lr = np.argmin(scores_lr)
            corners.append([float(contour_lr[idx_lr, 0]), float(contour_lr[idx_lr, 1])])
        else:
            mask_lr_mask = (ys >= y_center) & (zs >= z_center)
            if mask_lr_mask.any():
                y_dist = y_max - ys[mask_lr_mask]
                z_dist = z_max - zs[mask_lr_mask]
                scores = y_dist * z_dist + 0.1 * (y_dist + z_dist)
                idx = np.argmin(scores)
                corners.append([float(ys[mask_lr_mask][idx]), float(zs[mask_lr_mask][idx])])
            else:
                corners.append([y_max, z_max])
    else:
        mask_lr_mask = (ys >= y_center) & (zs >= z_center)
        if mask_lr_mask.any():
            y_dist = y_max - ys[mask_lr_mask]
            z_dist = z_max - zs[mask_lr_mask]
            scores = y_dist * z_dist + 0.1 * (y_dist + z_dist)
            idx = np.argmin(scores)
            corners.append([float(ys[mask_lr_mask][idx]), float(zs[mask_lr_mask][idx])])
        else:
            corners.append([y_max, z_max])
    
    # 左下角：找到轮廓上y坐标最小且z坐标最大的点
    mask_ll = (contour_yz[:, 0] < y_center) & (contour_yz[:, 1] >= z_center)
    if mask_ll.any():
        contour_ll = contour_yz[mask_ll]
        if contour_ll.shape[0] > 0:
            y_dist = contour_ll[:, 0] - y_min  # y方向到最小值的距离
            z_dist = z_max - contour_ll[:, 1]  # z方向到最大值的距离
            scores_ll = y_dist * z_dist + 0.1 * (y_dist + z_dist)
            idx_ll = np.argmin(scores_ll)
            corners.append([float(contour_ll[idx_ll, 0]), float(contour_ll[idx_ll, 1])])
        else:
            mask_ll_mask = (ys < y_center) & (zs >= z_center)
            if mask_ll_mask.any():
                y_dist = ys[mask_ll_mask] - y_min
                z_dist = z_max - zs[mask_ll_mask]
                scores = y_dist * z_dist + 0.1 * (y_dist + z_dist)
                idx = np.argmin(scores)
                corners.append([float(ys[mask_ll_mask][idx]), float(zs[mask_ll_mask][idx])])
            else:
                corners.append([y_min, z_max])
    else:
        mask_ll_mask = (ys < y_center) & (zs >= z_center)
        if mask_ll_mask.any():
            y_dist = ys[mask_ll_mask] - y_min
            z_dist = z_max - zs[mask_ll_mask]
            scores = y_dist * z_dist + 0.1 * (y_dist + z_dist)
            idx = np.argmin(scores)
            corners.append([float(ys[mask_ll_mask][idx]), float(zs[mask_ll_mask][idx])])
        else:
            corners.append([y_min, z_max])
    
    corners_array = np.array(corners, dtype=np.float32)
    
    # 最终验证：确保所有点都在mask内
    final_corners = []
    for corner in corners_array:
        corner_y, corner_z = corner[0], corner[1]
        corner_y_int = int(round(corner_y))
        corner_z_int = int(round(corner_z))
        
        # 检查点是否在mask内
        if (0 <= corner_z_int < body_zy.shape[0] and 
            0 <= corner_y_int < body_zy.shape[1] and 
            body_zy[corner_z_int, corner_y_int] > 0):
            final_corners.append([corner_y, corner_z])
        else:
            # 在mask内找到最近的点
            distances_to_mask = np.sqrt((ys - corner_y)**2 + (zs - corner_z)**2)
            nearest_mask_idx = np.argmin(distances_to_mask)
            final_corners.append([float(ys[nearest_mask_idx]), float(zs[nearest_mask_idx])])
    
    return np.array(final_corners, dtype=np.float32)


def _group_corners_by_axis(corners: np.ndarray, axis_direction: Tuple[float, float]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    根据轴向方向将四个顶点分成两组（上方两个点和下方两个点）。
    
    Parameters:
    -----------
    corners : np.ndarray of shape (4, 2) with corner points in (y, z) format
    axis_direction : (dz, dy) 归一化的轴向方向向量
    
    Returns:
    --------
    (upper_corners, lower_corners) : 每组两个点，或者None如果失败
    """
    if corners.shape[0] != 4:
        return None
    
    # 计算每个点在轴向方向上的投影
    # 使用Z坐标作为主要方向（因为轴向主要是Z方向）
    z_coords = corners[:, 1]  # Z坐标在第二列
    
    # 找到Z坐标的中位数，用于分组
    z_median = np.median(z_coords)
    
    # 分组：Z坐标小于中位数的为上方，大于中位数的为下方
    upper_mask = z_coords < z_median
    lower_mask = z_coords >= z_median
    
    # 如果分组后每组不是2个点，使用更简单的方法：按Z坐标排序
    if upper_mask.sum() != 2 or lower_mask.sum() != 2:
        # 按Z坐标排序
        sorted_indices = np.argsort(z_coords)
        upper_corners = corners[sorted_indices[:2]]
        lower_corners = corners[sorted_indices[2:]]
    else:
        upper_corners = corners[upper_mask]
        lower_corners = corners[lower_mask]
    
    return (upper_corners, lower_corners)


def _identify_anterior_posterior_lines(
    upper_corners: np.ndarray,
    lower_corners: np.ndarray,
    canal_zy: np.ndarray
) -> Tuple[Optional[Tuple[Tuple[float, float], Tuple[float, float]]], Optional[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    """
    根据与canal的距离判断哪条线段是anterior（前向），哪条是posterior（后向）。
    
    Parameters:
    -----------
    upper_corners : np.ndarray of shape (2, 2) with upper corner points in (y, z) format
    lower_corners : np.ndarray of shape (2, 2) with lower corner points in (y, z) format
    canal_zy : 2D binary mask of canal in ZY plane
    
    Returns:
    --------
    (anterior_line, posterior_line) : 每条线由两个点组成 ((y1, z1), (y2, z2))
    """
    if upper_corners.shape[0] != 2 or lower_corners.shape[0] != 2:
        return (None, None)
    
    # 找到canal的中心点作为参考
    if canal_zy.sum() == 0:
        # 如果没有canal，使用默认方法：假设左侧是anterior
        # 连接Y坐标较小的两个点作为anterior
        upper_y_sorted = upper_corners[np.argsort(upper_corners[:, 0])]
        lower_y_sorted = lower_corners[np.argsort(lower_corners[:, 0])]
        anterior_line = (
            (float(upper_y_sorted[0, 0]), float(upper_y_sorted[0, 1])),
            (float(lower_y_sorted[0, 0]), float(lower_y_sorted[0, 1]))
        )
        posterior_line = (
            (float(upper_y_sorted[1, 0]), float(upper_y_sorted[1, 1])),
            (float(lower_y_sorted[1, 0]), float(lower_y_sorted[1, 1]))
        )
        return (anterior_line, posterior_line)
    
    # 找到canal的中心点
    canal_zs, canal_ys = np.where(canal_zy > 0)
    canal_center_y = float(canal_ys.mean())
    canal_center_z = float(canal_zs.mean())
    canal_center = np.array([canal_center_y, canal_center_z])
    
    # 构建两条可能的线段
    # 需要正确配对：根据Y坐标配对（左侧点配对，右侧点配对）
    # 或者根据到canal的距离配对
    
    # 方法1：根据Y坐标配对（左侧的两个点配对，右侧的两个点配对）
    upper_sorted_by_y = upper_corners[np.argsort(upper_corners[:, 0])]  # 按Y坐标排序
    lower_sorted_by_y = lower_corners[np.argsort(lower_corners[:, 0])]  # 按Y坐标排序
    
    # 线段1：连接左侧的两个点（Y坐标较小的）
    line1_upper = upper_sorted_by_y[0]
    line1_lower = lower_sorted_by_y[0]
    line1_mid = (line1_upper + line1_lower) / 2.0
    
    # 线段2：连接右侧的两个点（Y坐标较大的）
    line2_upper = upper_sorted_by_y[1]
    line2_lower = lower_sorted_by_y[1]
    line2_mid = (line2_upper + line2_lower) / 2.0
    
    # 计算每条线段中点到canal中心的距离
    dist1 = np.linalg.norm(line1_mid - canal_center)
    dist2 = np.linalg.norm(line2_mid - canal_center)
    
    # 距离canal更近的是posterior，更远的是anterior
    if dist1 < dist2:
        posterior_line = (
            (float(line1_upper[0]), float(line1_upper[1])),
            (float(line1_lower[0]), float(line1_lower[1]))
        )
        anterior_line = (
            (float(line2_upper[0]), float(line2_upper[1])),
            (float(line2_lower[0]), float(line2_lower[1]))
        )
    else:
        posterior_line = (
            (float(line2_upper[0]), float(line2_upper[1])),
            (float(line2_lower[0]), float(line2_lower[1]))
        )
        anterior_line = (
            (float(line1_upper[0]), float(line1_upper[1])),
            (float(line1_lower[0]), float(line1_lower[1]))
        )
    
    return (anterior_line, posterior_line)


def _line_angle_deg_from_points(p0: Tuple[float, float], p1: Tuple[float, float], spacing_u: float, spacing_v: float) -> float:
    """
    p0/p1: (u, v) pixel coordinates (u horizontal, v vertical)
    Return angle of line segment relative to horizontal line (u-axis) in degrees, range (-90, 90]
    """
    du = (p1[0] - p0[0]) * spacing_u
    dv = (p1[1] - p0[1]) * spacing_v
    if abs(du) < 1e-9 and abs(dv) < 1e-9:
        return 0.0
    return float(np.degrees(np.arctan2(dv, du)))


# =========================
# Level 2: Parameter Functions
# =========================


def calc_vertebral_height_anterior_corrected(
    mri_zyx: np.ndarray,
    step2_zyx: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    vertebra_label: int,
    canal_label: int,
    mid_sag_x: int,
    save_path: Optional[str] = None,
    name: str = "",
) -> Dict[str, Any]:
    """
    Vertebral height (anterior and posterior): New method using four corner points.
    
    Method:
    1. Find canal axis direction to determine the axial (Z) direction
    2. Find four corner points of the vertebral body rectangle
    3. Group corners by axis direction (upper two and lower two)
    4. Connect points along axis to form two lines
    5. Identify anterior (far from canal) and posterior (close to canal) lines
    """
    sz, sy, sx = _spacing_zyx(spacing_xyz)

    body = (step2_zyx == vertebra_label).astype(np.uint8)
    canal = (step2_zyx == canal_label).astype(np.uint8)
    body_zy = body[:, :, mid_sag_x]
    canal_zy = canal[:, :, mid_sag_x]

    if body_zy.sum() == 0:
        return {"anterior_mm": None, "posterior_mm": None, "units": "mm", "status": "missing_label"}
    
    # Step 1: Get canal axis direction (for reference, though we mainly use Z direction)
    canal_axis = _get_canal_axis_direction(canal_zy)
    
    # Step 2: Isolate vertebral body to remove posterior elements
    body_body_only = isolate_vertebral_body(body_zy)
    if body_body_only.sum() == 0:
        # Fallback to original body if isolation fails
        body_body_only = body_zy
    
    # Step 3: Find four corner points of the vertebral body rectangle
    corners = _find_vertebra_rectangle_corners(body_body_only)
    if corners is None:
        return {"anterior_mm": None, "posterior_mm": None, "units": "mm", "status": "failed_to_find_corners"}
    
    # Step 4: Group corners by axis direction (upper two and lower two)
    axis_direction = canal_axis if canal_axis is not None else (1.0, 0.0)  # Default to vertical
    corner_groups = _group_corners_by_axis(corners, axis_direction)
    if corner_groups is None:
        return {"anterior_mm": None, "posterior_mm": None, "units": "mm", "status": "failed_to_group_corners"}
    
    upper_corners, lower_corners = corner_groups
    
    # Step 5: Identify anterior and posterior lines
    anterior_line, posterior_line = _identify_anterior_posterior_lines(
        upper_corners, lower_corners, canal_zy
    )
    
    if anterior_line is None or posterior_line is None:
        return {"anterior_mm": None, "posterior_mm": None, "units": "mm", "status": "failed_to_identify_lines"}
    
    # Step 6: Calculate heights
    # Anterior height
    ant_p0 = np.array([anterior_line[0][0], anterior_line[0][1]])  # (y, z)
    ant_p1 = np.array([anterior_line[1][0], anterior_line[1][1]])  # (y, z)
    ant_vec = ant_p1 - ant_p0
    ant_dy_mm = abs(ant_vec[0]) * sy  # Y direction (anterior-posterior)
    ant_dz_mm = abs(ant_vec[1]) * sz  # Z direction (superior-inferior)
    anterior_height_mm = float(np.sqrt(ant_dy_mm**2 + ant_dz_mm**2))
    
    # Posterior height
    post_p0 = np.array([posterior_line[0][0], posterior_line[0][1]])  # (y, z)
    post_p1 = np.array([posterior_line[1][0], posterior_line[1][1]])  # (y, z)
    post_vec = post_p1 - post_p0
    post_dy_mm = abs(post_vec[0]) * sy  # Y direction
    post_dz_mm = abs(post_vec[1]) * sz  # Z direction
    posterior_height_mm = float(np.sqrt(post_dy_mm**2 + post_dz_mm**2))
    
    # Step 7: Visualization
    if save_path is not None:
        img_zy = _normalize_intensity_percentile(mri_zyx[:, :, mid_sag_x])
        crop = _crop_around_mask(body_body_only, pad=30)
        
        if crop is not None:
            img_c = _apply_crop(img_zy, crop)
            body_c = _apply_crop(body_body_only, crop)
            overlays = [
                ("mask", {"mask": body_c > 0, "color": "lime", "alpha": 0.55}),
            ]
            
            crop_xy = _safe_unpack_crop(crop)
            if crop_xy is not None:
                x0, y0 = crop_xy
                
                # Draw anterior line (cyan) - far from canal
                overlays.append(("line", {
                    "p0": (float(anterior_line[0][0] - x0), float(anterior_line[0][1] - y0)),
                    "p1": (float(anterior_line[1][0] - x0), float(anterior_line[1][1] - y0)),
                    "color": "cyan",
                    "lw": 3,
                    "label": f"A {anterior_height_mm:.1f}",
                }))
                
                # Draw posterior line (yellow) - close to canal
                overlays.append(("line", {
                    "p0": (float(posterior_line[0][0] - x0), float(posterior_line[0][1] - y0)),
                    "p1": (float(posterior_line[1][0] - x0), float(posterior_line[1][1] - y0)),
                    "color": "yellow",
                    "lw": 3,
                    "label": f"P {posterior_height_mm:.1f}",
                }))
                
                # Draw corner points
                corners_crop = corners.copy()
                corners_crop[:, 0] -= x0  # Y coordinate
                corners_crop[:, 1] -= y0  # Z coordinate
                overlays.append(("scatter", {
                    "pts": corners_crop,
                    "color": "red",
                    "s": 80
                }))
            
            save_visualization(save_path, img_c, f"Vertebral Height {name}", overlays)
        else:
            # Fallback: save full image
            overlays = [
                ("mask", {"mask": body_body_only > 0, "color": "lime", "alpha": 0.55}),
                ("line", {
                    "p0": (float(anterior_line[0][0]), float(anterior_line[0][1])),
                    "p1": (float(anterior_line[1][0]), float(anterior_line[1][1])),
                    "color": "cyan",
                    "lw": 3,
                    "label": f"A {anterior_height_mm:.1f}",
                }),
                ("line", {
                    "p0": (float(posterior_line[0][0]), float(posterior_line[0][1])),
                    "p1": (float(posterior_line[1][0]), float(posterior_line[1][1])),
                    "color": "yellow",
                    "lw": 3,
                    "label": f"P {posterior_height_mm:.1f}",
                }),
                ("scatter", {
                    "pts": corners,
                    "color": "red",
                    "s": 80
                }),
            ]
            save_visualization(save_path, img_zy, f"Vertebral Height {name}", overlays)
    
    return {
        "anterior_mm": _safe_float(anterior_height_mm),
        "posterior_mm": _safe_float(posterior_height_mm),
        "units": "mm",
        "status": "ok",
        "method": "four_corner_points",
    }


# Keep old function for backward compatibility
def calc_vertebral_height(
    mri_zyx: np.ndarray,
    step2_zyx: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    vertebra_label: int,
    canal_label: int,
    mid_sag_x: int,
    save_path: Optional[str] = None,
    name: str = "",
) -> Dict[str, Any]:
    """Legacy function - redirects to corrected version."""
    return calc_vertebral_height_anterior_corrected(
        mri_zyx, step2_zyx, spacing_xyz, vertebra_label, canal_label, mid_sag_x, save_path, name
    )


def calc_vertebral_width_axial(
    mri_zyx: np.ndarray,
    step2_zyx: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    vertebra_label: int,
    canal_label: int,
    mid_sag_x: int,
    save_path: Optional[str] = None,
    name: str = "",
) -> Dict[str, Any]:
    """
    Vertebral Body midline AP diameter (axial plane).

    Measures *only* the vertebral body AP diameter on the axial (transverse)
    plane at the geometric Z-center of the vertebra.  The measurement line
    runs along the mid-sagittal column (``mid_sag_x``) from the **anterior
    margin** of the vertebral body to the **posterior wall** of the vertebral
    body.  The posterior wall is defined as the anterior edge of the spinal
    canal on the same axial slice, thereby excluding pedicles, laminae, and
    the spinous process.

    Parameters
    ----------
    mri_zyx : np.ndarray
        3-D MRI intensity volume (Z, Y, X).
    step2_zyx : np.ndarray
        3-D segmentation label volume (Z, Y, X).
    spacing_xyz : (float, float, float)
        Voxel spacing in mm (X, Y, Z).
    vertebra_label : int
        Label ID of the target vertebra in ``step2_zyx``.
    canal_label : int
        Label ID of the spinal canal in ``step2_zyx``.
    mid_sag_x : int
        X-index of the mid-sagittal slice.
    save_path : str or None
        If given, save a visualisation PNG at this path.
    name : str
        Human-readable vertebra name for the title.

    Returns
    -------
    dict
        ``ap_diameter_mm``, ``z_index``, ``units``, ``status``, ``method``.
    """
    sx, sy, sz = spacing_xyz  # For axial: (X, Y, Z) spacing

    body = (step2_zyx == vertebra_label).astype(np.uint8)
    canal = (step2_zyx == canal_label).astype(np.uint8)
    if body.sum() == 0:
        return {"ap_diameter_mm": None, "units": "mm", "status": "missing_label"}

    # Find geometric center (Z-axis center) of vertebra
    com = _center_of_mass_idx(body)
    if com is None:
        return {"ap_diameter_mm": None, "units": "mm", "status": "failed"}

    z_center = int(round(com[0]))

    # ---------- Find a usable axial slice ----------
    body_axial = body[z_center, :, :]       # Shape: (Y, X)
    canal_axial = canal[z_center, :, :]     # Shape: (Y, X)
    mri_axial = mri_zyx[z_center, :, :]     # Shape: (Y, X)

    if body_axial.sum() == 0:
        # Try nearby slices if center slice is empty
        for offset in [1, 2, -1, -2]:
            z_try = z_center + offset
            if 0 <= z_try < body.shape[0]:
                body_axial = body[z_try, :, :]
                canal_axial = canal[z_try, :, :]
                mri_axial = mri_zyx[z_try, :, :]
                if body_axial.sum() > 0:
                    z_center = z_try
                    break
        if body_axial.sum() == 0:
            return {"ap_diameter_mm": None, "units": "mm", "status": "no_axial_slice", "z_index": None}

    # ---------- Determine midline column ----------
    # Use provided mid_sag_x. If it does not intersect the vertebra on this
    # axial slice, fall back to the mean X of the vertebra mask.
    ys_all, xs_all = np.where(body_axial > 0)
    if ys_all.size == 0:
        return {"ap_diameter_mm": None, "units": "mm", "status": "failed", "z_index": z_center}

    midline_x = mid_sag_x
    ys_at_midline = np.where(body_axial[:, midline_x] > 0)[0] if 0 <= midline_x < body_axial.shape[1] else np.array([])
    if ys_at_midline.size == 0:
        # Fallback: use the mean X of the vertebra mask on this slice
        midline_x = int(round(float(xs_all.mean())))
        ys_at_midline = np.where(body_axial[:, midline_x] > 0)[0]
    if ys_at_midline.size == 0:
        return {"ap_diameter_mm": None, "units": "mm", "status": "failed", "z_index": z_center}

    # Full vertebra extent along the midline column
    vert_y_min = int(ys_at_midline.min())
    vert_y_max = int(ys_at_midline.max())

    # ---------- Use canal to find posterior wall of vertebral body ----------
    # The posterior wall of the vertebral body is defined as the anterior edge
    # of the spinal canal.  We search along a narrow band of columns around
    # the midline to be robust against small mis-alignments.
    search_half_width = 3  # pixels on each side of midline_x
    x_lo = max(0, midline_x - search_half_width)
    x_hi = min(body_axial.shape[1], midline_x + search_half_width + 1)
    canal_band = canal_axial[:, x_lo:x_hi]

    posterior_wall_y: Optional[int] = None

    if canal_band.sum() > 0:
        canal_ys_band = np.where(canal_band > 0)[0]
        # Determine AP direction: which end of the vertebra is closer to the
        # canal?  That end is the posterior side.
        canal_mean_y = float(canal_ys_band.mean())
        vert_mid_y = (vert_y_min + vert_y_max) / 2.0

        if canal_mean_y < vert_mid_y:
            # Canal is on the low-Y side  →  low Y is posterior
            # Posterior wall = max Y of canal band (its edge closest to vertebral body)
            posterior_wall_y = int(canal_ys_band.max())
            anterior_y = vert_y_max
        else:
            # Canal is on the high-Y side  →  high Y is posterior
            # Posterior wall = min Y of canal band (its edge closest to vertebral body)
            posterior_wall_y = int(canal_ys_band.min())
            anterior_y = vert_y_min
    else:
        # Canal not found on this axial slice – try the mid-sagittal (ZY)
        # plane as a fallback to at least get the A/P direction right, then
        # use half the vertebra extent as a rough body estimate.
        body_zy = body[:, :, mid_sag_x]
        canal_zy = canal[:, :, mid_sag_x]
        ant_y, post_y = _pick_anterior_posterior_from_canal(body_zy, canal_zy)
        if ant_y == post_y == 0:
            # Absolute fallback: full extent
            anterior_y = vert_y_min
            posterior_wall_y = vert_y_max
        else:
            # Use the direction hint but constrain to this slice
            if abs(ant_y - vert_y_min) < abs(ant_y - vert_y_max):
                anterior_y = vert_y_min
                posterior_wall_y = vert_y_max
            else:
                anterior_y = vert_y_max
                posterior_wall_y = vert_y_min

    # Sanity: if posterior_wall_y ended up beyond / equal to anterior_y,
    # clamp to the vertebra extent in the correct direction.
    if posterior_wall_y is None:
        posterior_wall_y = vert_y_max if anterior_y == vert_y_min else vert_y_min

    ap_pixels = abs(anterior_y - posterior_wall_y)
    ap_mm = float(ap_pixels * sy)

    # Ensure the line endpoints are ordered (line_y_start < line_y_end)
    line_y_start = min(anterior_y, posterior_wall_y)
    line_y_end = max(anterior_y, posterior_wall_y)

    # ---------- Visualisation ----------
    if save_path is not None:
        img_axial = _normalize_intensity_percentile(mri_axial)
        # Use a combined mask for the crop region so the canal is visible too
        combined_mask = np.clip(body_axial.astype(np.int16) + canal_axial.astype(np.int16), 0, 1).astype(np.uint8)
        crop = _crop_around_mask(combined_mask, pad=50)
        img_c = _apply_crop(img_axial, crop)
        body_c = _apply_crop(body_axial, crop)
        canal_c = _apply_crop(canal_axial, crop)

        overlays = [
            ("mask", {"mask": body_c > 0, "color": "lime", "alpha": 0.45}),
            ("mask", {"mask": canal_c > 0, "color": "cyan", "alpha": 0.45}),
        ]

        if crop is not None:
            crop_xy = _safe_unpack_crop(crop)
            if crop_xy is not None:
                x0, y0 = crop_xy
                # Draw measurement line from anterior to posterior wall
                overlays.append(
                    (
                        "line",
                        {
                            "p0": (float(midline_x - x0), float(line_y_start - y0)),
                            "p1": (float(midline_x - x0), float(line_y_end - y0)),
                            "color": "yellow",
                            "lw": 2,
                            "label": f"AP body {ap_mm:.1f}mm",
                        },
                    )
                )
        save_visualization(save_path, img_c, f"Vertebral Body AP Diameter (Axial) {name}", overlays)

    return {
        "ap_diameter_mm": _safe_float(ap_mm),
        "z_index": int(z_center),
        "units": "mm",
        "status": "ok",
        "method": "axial_midline_body_only_canal_boundary",
    }


# Keep old function for backward compatibility
def calc_disc_height_corrected(
    mri_zyx: np.ndarray,
    step2_zyx: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    disc_label: int,
    mid_sag_x: int,
    canal_label: int,
    upper_vh_avg_mm: Optional[float],
    lower_vh_avg_mm: Optional[float],
    save_path: Optional[str] = None,
    name: str = "",
) -> Dict[str, Any]:
    """
    Disc height (DH): "Linear Regression" method.
    Uses np.polyfit to fit a line through disc mask pixels, calculates rotation angle,
    mathematically rotates points to horizontal, measures height at 3 locations,
    then maps points back to original space using inverse rotation.
    HDR: DH / disc AP diameter (sagittal)
    DHI: DH / mean(upper_VH_avg, lower_VH_avg)
    """
    sz, sy, _sx = _spacing_zyx(spacing_xyz)
    disc = (step2_zyx == disc_label).astype(np.uint8)
    canal = (step2_zyx == canal_label).astype(np.uint8)
    disc_zy = disc[:, :, mid_sag_x]
    canal_zy = canal[:, :, mid_sag_x]

    if disc_zy.sum() == 0:
        return {"dh_mm": None, "hdr": None, "dhi": None, "units": {"dh": "mm"}, "status": "missing_label"}

    # Step 1: Extract Mask Points
    # In ZY space: disc_zy is (Z, Y) where Z is row (vertical), Y is column (horizontal)
    # np.where returns (z_coords, y_coords) where z is row index, y is column index
    z_coords, y_coords = np.where(disc_zy > 0)
    
    if len(z_coords) < 10:
        # Fallback to simple method
        if z_coords.size == 0:
            return {"dh_mm": None, "hdr": None, "dhi": None, "units": {"dh": "mm"}, "status": "missing_label"}
        z_min, z_max = int(z_coords.min()), int(z_coords.max())
        dh_mm = float((z_max - z_min) * sz)
        
        if save_path is not None:
            img_zy = _normalize_intensity_percentile(mri_zyx[:, :, mid_sag_x])
            crop = _crop_around_mask(disc_zy, pad=30)
            if crop is not None:
                img_c = _apply_crop(img_zy, crop)
                disc_c = _apply_crop(disc_zy, crop)
                overlays = [("mask", {"mask": disc_c > 0, "color": "lime", "alpha": 0.55})]
                crop_xy = _safe_unpack_crop(crop)
                if crop_xy is not None:
                    x0, y0 = crop_xy
                    y_mid = int(round(float(y_coords.mean())))
                    overlays.append(("line", {
                        "p0": (float(y_mid - x0), float(z_min - y0)),
                        "p1": (float(y_mid - x0), float(z_max - y0)),
                        "color": "yellow",
                        "lw": 2,
                        "label": f"{dh_mm:.1f}",
                    }))
                save_visualization(save_path, img_c, f"Disc Height (Fallback) {name}", overlays)
        
        return {"dh_mm": _safe_float(dh_mm), "hdr": None, "dhi": None, "units": {"dh": "mm"}, "status": "insufficient_points"}
    
    # Convert to (x, y) where x is Y (column, horizontal) and y is Z (row, vertical)
    # This matches the standard image coordinate system
    x_coords = y_coords.astype(np.float64)  # Y axis (horizontal, column)
    y_coords_float = z_coords.astype(np.float64)  # Z axis (vertical, row)
    
    # Step 2: Fit Line (y = kx + b) to get orientation
    # Fit: y_coords_float = k * x_coords + b
    k, b = np.polyfit(x_coords, y_coords_float, 1)
    theta = np.arctan(k)  # Orientation in radians
    
    # Step 3: Calculate Centroid
    cx = float(np.mean(x_coords))
    cy = float(np.mean(y_coords_float))
    
    # Step 4: Rotate Points to be Horizontal (Mathematical Flattener)
    # Rotation by -theta around centroid
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    
    # Shift to origin
    x_shifted = x_coords - cx
    y_shifted = y_coords_float - cy
    
    # Rotate: x_rot = x_shifted * cos(-theta) - y_shifted * sin(-theta)
    #         y_rot = x_shifted * sin(-theta) + y_shifted * cos(-theta)
    x_rot = x_shifted * cos_t - y_shifted * sin_t
    y_rot = x_shifted * sin_t + y_shifted * cos_t
    
    # Step 5: Measure at 25%, 50%, 75% along the ROTATED X-axis
    min_xr = float(np.min(x_rot))
    max_xr = float(np.max(x_rot))
    width_r = max_xr - min_xr
    
    measure_locs = [0.25, 0.50, 0.75]
    heights_mm = []
    lines_visual = []  # Store pairs of (pt1, pt2) in original coords
    
    # Inverse rotation parameters (for mapping back)
    cos_inv = np.cos(theta)
    sin_inv = np.sin(theta)
    
    def inverse_map(xr: float, yr: float) -> Tuple[float, float]:
        """Map rotated coordinates back to original space."""
        # x = xr * cos(theta) - yr * sin(theta) + cx
        # y = xr * sin(theta) + yr * cos(theta) + cy
        xo = xr * cos_inv - yr * sin_inv + cx
        yo = xr * sin_inv + yr * cos_inv + cy
        return (xo, yo)
    
    tolerance = 1.0  # 1 pixel width
    
    for loc in measure_locs:
        target_x = min_xr + width_r * loc
        
        # Find points within tolerance of this X-location
        indices = np.where(np.abs(x_rot - target_x) < tolerance)[0]
        
        if len(indices) == 0:
            continue
        
        # In rotated space, height is simply Y range
        current_ys = y_rot[indices]
        y_min_r = float(np.min(current_ys))
        y_max_r = float(np.max(current_ys))
        
        # Map back to Original Space for visualization and robust distance calc
        pt_top = inverse_map(target_x, y_min_r)
        pt_bottom = inverse_map(target_x, y_max_r)
        
        # Euclidean distance in physical space
        # Note: pt_top and pt_bottom are (x, y) where x is Y (column), y is Z (row)
        # Spacing: sz is Z spacing, sy is Y spacing
        dx_mm = (pt_bottom[0] - pt_top[0]) * sy
        dy_mm = (pt_bottom[1] - pt_top[1]) * sz
        dist = float(np.sqrt(dx_mm**2 + dy_mm**2))
        
        heights_mm.append(dist)
        lines_visual.append((pt_top, pt_bottom, dist))
    
    # Average Height
    if heights_mm:
        dh_mm = float(np.mean(heights_mm))
    else:
        # Fallback to bounding box height in rotated space
        y_min_r = float(np.min(y_rot))
        y_max_r = float(np.max(y_rot))
        pt_top = inverse_map(min_xr, y_min_r)
        pt_bottom = inverse_map(min_xr, y_max_r)
        dx_mm = (pt_bottom[0] - pt_top[0]) * sy
        dy_mm = (pt_bottom[1] - pt_top[1]) * sz
        dh_mm = float(np.sqrt(dx_mm**2 + dy_mm**2))
    
    # disc AP diameter (sagittal): take Y range near middle Z slice of disc mask
    zs_all = np.where(disc_zy > 0)[0]
    z_mid = int(round(float(zs_all.mean())))
    band = 2
    z0 = max(z_mid - band, 0)
    z1 = min(z_mid + band, disc_zy.shape[0] - 1)
    ys2 = np.where(disc_zy[z0 : z1 + 1, :] > 0)[1]
    ap_mm = None
    if ys2.size > 0:
        ap_mm = float((int(ys2.max()) - int(ys2.min())) * sy)

    hdr = None if ap_mm is None or ap_mm <= 1e-6 else float(dh_mm / ap_mm)

    dhi = None
    if upper_vh_avg_mm is not None and lower_vh_avg_mm is not None:
        denom = float((upper_vh_avg_mm + lower_vh_avg_mm) / 2.0)
        if denom > 1e-6:
            dhi = float(dh_mm / denom)
    elif upper_vh_avg_mm is not None:
        denom = float(upper_vh_avg_mm)
        if denom > 1e-6:
            dhi = float(dh_mm / denom)
    elif lower_vh_avg_mm is not None:
        denom = float(lower_vh_avg_mm)
        if denom > 1e-6:
            dhi = float(dh_mm / denom)

    # Step 6: Visualization
    if save_path is not None:
        img_zy = _normalize_intensity_percentile(mri_zyx[:, :, mid_sag_x])
        crop = _crop_around_mask(disc_zy, pad=30)
        img_c = _apply_crop(img_zy, crop)
        disc_c = _apply_crop(disc_zy, crop)
        overlays = [("mask", {"mask": disc_c > 0, "color": "lime", "alpha": 0.55})]

        if crop is not None:
            crop_xy = _safe_unpack_crop(crop)
            if crop_xy is not None:
                x0, y0 = crop_xy
                
                # Draw measurement lines (mathematically guaranteed to be perpendicular to disc axis)
                labels = ["P", "M", "A"]  # 0.25位置=后(P), 0.50位置=中(M), 0.75位置=前(A)
                for i, (pt_top, pt_bottom, height_mm) in enumerate(lines_visual):
                    # Convert to crop coordinates
                    # pt_top and pt_bottom are (x, y) where x is Y (column), y is Z (row)
                    p_top_crop = (float(pt_top[0] - x0), float(pt_top[1] - y0))
                    p_bot_crop = (float(pt_bottom[0] - x0), float(pt_bottom[1] - y0))
                    
                    lbl = labels[i] if i < len(labels) else f"L{i+1}"
                    overlays.append(
                        (
                            "line",
                            {
                                "p0": p_top_crop,
                                "p1": p_bot_crop,
                                "color": "yellow",
                                "lw": 2,
                                "label": f"{lbl} {height_mm:.1f}",
                            },
                        )
                    )

        save_visualization(save_path, img_c, f"Disc Height (Linear Regression) {name}", overlays)

    # Prepare scan_lines info for return
    scan_lines = []
    labels = ["P", "M", "A"]  # 0.25位置=后(P), 0.50位置=中(M), 0.75位置=前(A)
    for i, (pt_top, pt_bottom, height_mm) in enumerate(lines_visual):
        label = labels[i] if i < len(labels) else f"L{i+1}"
        scan_lines.append({
            "label": label,
            "height_mm": height_mm,
        })

    return {
        "dh_mm": _safe_float(dh_mm),
        "disc_ap_diameter_mm": _safe_float(ap_mm),
        "hdr": _safe_float(hdr),
        "dhi": _safe_float(dhi),
        "units": {"dh": "mm", "disc_ap_diameter": "mm"},
        "status": "ok",
        "method": "linear_regression_mathematical_rotation",
        "scan_line_heights_mm": {sl["label"]: _safe_float(sl["height_mm"]) for sl in scan_lines} if scan_lines else None,
        "formula": {"hdr": "HDR = DH / Disc_AP_Diameter", "dhi": "DHI = DH / mean(Upper_VH_avg, Lower_VH_avg)"},
    }


# Keep old function for backward compatibility
def calc_disc_height_and_hdr_dhi(
    mri_zyx: np.ndarray,
    step2_zyx: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    disc_label: int,
    mid_sag_x: int,
    canal_label: int,
    upper_vh_avg_mm: Optional[float],
    lower_vh_avg_mm: Optional[float],
    save_path: Optional[str] = None,
    name: str = "",
) -> Dict[str, Any]:
    """Legacy function - redirects to corrected version."""
    return calc_disc_height_corrected(
        mri_zyx, step2_zyx, spacing_xyz, disc_label, mid_sag_x, canal_label,
        upper_vh_avg_mm, lower_vh_avg_mm, save_path, name
    )


def extract_vertebral_body(mask: np.ndarray) -> np.ndarray:
    """
    Extract ONLY the anterior vertebral body (cylinder-like shape) from a binary mask,
    removing posterior elements (spinous/transverse processes).
    
    Algorithm (Robust Geometric Cut):
    1. Find bounding box of the mask
    2. Define cut region: keep leftmost 60-70% of the mask's width
    3. Apply vertical cut at cutoff_x
    4. Post-cut cleanup: keep only the largest connected component
    
    Parameters:
    -----------
    mask : 2D binary mask array (uint8, 0 or 255, or boolean)
    
    Returns:
    --------
    body_mask : 2D binary mask containing only the vertebral body (anterior part)
    """
    if cv2 is None:
        # Fallback: assume left half is the body
        h, w = mask.shape
        body_mask = np.zeros_like(mask)
        body_mask[:, :w//2] = mask[:, :w//2]
        return body_mask
    
    # Convert to uint8 if needed
    if mask.dtype != np.uint8:
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
    else:
        mask_uint8 = mask.copy()
    
    if mask_uint8.sum() == 0:
        return np.zeros_like(mask_uint8)
    
    # Step 1: Find Bounding Box
    # Use cv2.boundingRect on the mask to get (x, y, w, h)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.zeros_like(mask_uint8)
    
    # Get the bounding box of the largest contour (or all contours combined)
    x, y, w, h = cv2.boundingRect(mask_uint8)
    
    if w == 0 or h == 0:
        return np.zeros_like(mask_uint8)
    
    # Step 2: Define Cut Region
    # Keep the leftmost 60-70% of the mask's total width
    # Calculate the cutoff X-coordinate: cutoff_x = int(x + 0.70 * w)
    cutoff_x = int(x + 0.70 * w)
    
    # Create a new blank mask of the same size
    cut_mask = np.zeros_like(mask_uint8)
    
    # Copy the original mask's pixels to cut_mask, but only for columns where pixel_x < cutoff_x
    # All pixels to the right of cutoff_x must be 0
    cut_mask[:, :cutoff_x] = mask_uint8[:, :cutoff_x]
    
    # Step 3: Post-Cut Cleanup
    # The vertical cut might leave small, disconnected fragments
    # Perform connected components analysis on the cut_mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cut_mask, connectivity=8)
    
    if num_labels < 2:  # Only background (label 0)
        return np.zeros_like(mask_uint8)
    
    # Identify and keep only the largest connected component
    # This is guaranteed to be the main vertebral body
    component_sizes = []
    for i in range(1, num_labels):  # Skip label 0 (background)
        area = stats[i, cv2.CC_STAT_AREA]
        component_sizes.append((area, i))
    
    if not component_sizes:
        return np.zeros_like(mask_uint8)
    
    # Sort by area (largest first)
    component_sizes.sort(key=lambda x: x[0], reverse=True)
    
    # Get the largest component
    largest_label = component_sizes[0][1]
    body_mask = (labels == largest_label).astype(np.uint8) * 255
    
    # Step 4: Output
    # Return the clean, geometrically cut body_mask
    return body_mask


def isolate_vertebral_body(mask: np.ndarray) -> np.ndarray:
    """Isolate vertebral body region from a 2D binary vertebra mask.

    Logic: Robust Fallback Mechanism
    1. Input Check: If mask is empty or None, return None.
    2. Attempt 1 (The Hard Cut):
       * Try the existing logic: Distance Transform -> Threshold -> Vertical Cut (max_loc[0] + radius * 1.2).
       * Check result: If the resulting mask has non-zero pixels, return it.
    3. Fallback (Safe Mode):
       * If Attempt 1 results in an empty mask (or fails), catch the error.
       * Action: Simply return the Largest Connected Component of the original mask.
       * Reason: It's better to calculate on a slightly imperfect mask (with attachments) than to return nothing.

    Parameters
    ----------
    mask:
        2D binary mask (uint8 or bool). Non-zero treated as foreground.

    Returns
    -------
    np.ndarray
        2D uint8 mask with values {0,255}.
    """
    # 1) Input Check: If mask is empty or None, return None
    m = (mask > 0).astype(np.uint8)
    if m.ndim != 2 or m.max() == 0:
        return np.zeros_like(m, dtype=np.uint8)

    if cv2 is None:
        # OpenCV is optional in this repo; fall back to the legacy heuristic.
        return extract_vertebral_body(m)

    # Ensure OpenCV-compatible binary (0/255)
    m255 = (m * 255).astype(np.uint8)

    # 2) Attempt 1 (The Hard Cut): Try the existing logic
    try:
        # Distance transform
        dist_map = cv2.distanceTransform(m255, cv2.DIST_L2, 5)
        max_val, _, max_loc, _ = cv2.minMaxLoc(dist_map)
        if max_val <= 0 or max_loc is None:
            raise ValueError("Distance transform failed")

        # The Hard Cut: Determine cutoff X coordinate
        cutoff_x = int(max_loc[0] + max_val * 1.2)
        cutoff_x = max(0, min(cutoff_x, m255.shape[1]))

        # Action: Create a new mask. Copy pixels from the original mask only where x < cutoff_x.
        result = np.zeros_like(m255)
        result[:, :cutoff_x] = m255[:, :cutoff_x]

        # Check result: If the resulting mask has non-zero pixels, return it
        if result.sum() > 0:
            # For L1-L5, select the Left-Most (Anterior) component, not necessarily the largest
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((result > 0).astype(np.uint8), connectivity=8)
            if num_labels >= 2:
                # Find the component with the minimum X coordinate (left-most/anterior)
                left_most_label = None
                left_most_x = float('inf')
                
                for label_id in range(1, num_labels):  # Skip label 0 (background)
                    left_x = float(stats[label_id, cv2.CC_STAT_LEFT])
                    area = int(stats[label_id, cv2.CC_STAT_AREA])
                    
                    # Only consider components with reasonable area (> 10 pixels to ignore noise)
                    if area < 10:
                        continue
                    
                    # Select the component with Minimum X coordinate (Left-Most/Anterior)
                    if left_x < left_most_x:
                        left_most_x = left_x
                        left_most_label = label_id
                
                if left_most_label is not None:
                    return (labels == left_most_label).astype(np.uint8) * 255
                else:
                    # Fallback within attempt 1: use largest component
                    areas = stats[1:, cv2.CC_STAT_AREA]
                    if len(areas) > 0:
                        largest = int(1 + np.argmax(areas))
                        return (labels == largest).astype(np.uint8) * 255
    except Exception:
        pass  # Fall through to Fallback (Safe Mode)

    # 3) Fallback (Safe Mode): If Attempt 1 results in an empty mask (or fails)
    # Simply return the Largest Connected Component of the original mask
    try:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m255, connectivity=8)
        if num_labels >= 2:
            areas = stats[1:, cv2.CC_STAT_AREA]
            if len(areas) > 0:
                largest = int(1 + np.argmax(areas))
                return (labels == largest).astype(np.uint8) * 255
    except Exception:
        pass

    # Final fallback: return original mask (converted to uint8)
    return m255


def _line_to_points(line: Tuple[float, float, float, float], img_width: int, img_height: int) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Convert line parameters to two points for drawing.
    
    Parameters:
    -----------
    line : (vx, vy, x0, y0) from cv2.fitLine or (k, b, None, None) from np.polyfit
    img_width : Image width
    img_height : Image height
    
    Returns:
    --------
    ((x1, y1), (x2, y2)) : Two points defining the line, or None if invalid
    """
    if line is None or len(line) < 2:
        return None
    
    if len(line) == 4 and line[2] is not None:
        # cv2.fitLine format: (vx, vy, x0, y0)
        vx, vy, x0, y0 = line[0], line[1], line[2], line[3]
        
        if abs(vx) < 1e-9 and abs(vy) < 1e-9:
            return None
        
        # Normalize direction vector
        norm = np.sqrt(vx * vx + vy * vy)
        if norm < 1e-9:
            return None
        vx, vy = vx / norm, vy / norm
        
        # Extend line to image boundaries
        # Find intersection with image edges
        t_values = []
        # Left edge (x=0)
        if abs(vx) > 1e-9:
            t = (0 - x0) / vx
            y = y0 + t * vy
            if 0 <= y < img_height:
                t_values.append((0, y))
        # Right edge (x=img_width-1)
        if abs(vx) > 1e-9:
            t = (img_width - 1 - x0) / vx
            y = y0 + t * vy
            if 0 <= y < img_height:
                t_values.append((img_width - 1, y))
        # Top edge (y=0)
        if abs(vy) > 1e-9:
            t = (0 - y0) / vy
            x = x0 + t * vx
            if 0 <= x < img_width:
                t_values.append((x, 0))
        # Bottom edge (y=img_height-1)
        if abs(vy) > 1e-9:
            t = (img_height - 1 - y0) / vy
            x = x0 + t * vx
            if 0 <= x < img_width:
                t_values.append((x, img_height - 1))
        
        if len(t_values) >= 2:
            return (t_values[0], t_values[1])
        elif len(t_values) == 1:
            # Use point on line at distance from x0, y0
            p1 = t_values[0]
            p2 = (x0 + vx * img_width, y0 + vy * img_height)
            return (p1, p2)
        else:
            return None
    else:
        # np.polyfit format: (k, b, None, None)
        k, b = line[0], line[1]
        
        # y = k*x + b
        # Find intersections with image boundaries
        x1, y1 = 0.0, b
        x2, y2 = float(img_width - 1), k * (img_width - 1) + b
        
        # Clamp to image boundaries
        if y1 < 0:
            y1 = 0.0
            x1 = (y1 - b) / k if abs(k) > 1e-9 else 0.0
        elif y1 >= img_height:
            y1 = float(img_height - 1)
            x1 = (y1 - b) / k if abs(k) > 1e-9 else 0.0
        
        if y2 < 0:
            y2 = 0.0
            x2 = (y2 - b) / k if abs(k) > 1e-9 else float(img_width - 1)
        elif y2 >= img_height:
            y2 = float(img_height - 1)
            x2 = (y2 - b) / k if abs(k) > 1e-9 else float(img_width - 1)
        
        x1 = max(0, min(img_width - 1, x1))
        x2 = max(0, min(img_width - 1, x2))
        
        return ((x1, y1), (x2, y2))


def _get_line_intersection(line1: Tuple[float, float, float, float], line2: Tuple[float, float, float, float], img_width: int, img_height: int) -> Optional[Tuple[float, float]]:
    """
    Calculate intersection point of two lines.
    
    Parameters:
    -----------
    line1, line2 : Line parameters from fit_endplate_line
    img_width, img_height : Image dimensions
    
    Returns:
    --------
    (x, y) : Intersection point, or None if lines are parallel
    """
    if line1 is None or line2 is None:
        return None
    
    # Convert to slope-intercept form
    def line_to_slope_intercept(line):
        if len(line) == 4 and line[2] is not None:
            vx, vy, x0, y0 = line[0], line[1], line[2], line[3]
            if abs(vx) < 1e-9:
                return None, None, x0  # Vertical line at x = x0
            k = vy / vx
            b = y0 - k * x0
            return k, b, None
        else:
            k, b = line[0], line[1]
            return k, b, None
    
    k1, b1, x_vert1 = line_to_slope_intercept(line1)
    k2, b2, x_vert2 = line_to_slope_intercept(line2)
    
    # Handle vertical lines
    if x_vert1 is not None:
        if x_vert2 is not None:
            return None  # Both vertical, parallel
        # Line1 is vertical at x = x_vert1
        y = k2 * x_vert1 + b2
        return (x_vert1, y)
    
    if x_vert2 is not None:
        # Line2 is vertical at x = x_vert2
        y = k1 * x_vert2 + b1
        return (x_vert2, y)
    
    # Both are non-vertical
    if abs(k1 - k2) < 1e-9:
        return None  # Parallel lines
    
    # Intersection: k1*x + b1 = k2*x + b2
    x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1
    
    return (x, y)


def draw_line_and_angle(
    image: np.ndarray,
    line1: Tuple[float, float, float, float],
    line2: Optional[Tuple[float, float, float, float]],
    text: str,
    color1: Tuple[int, int, int] = (0, 255, 0),
    color2: Tuple[int, int, int] = (255, 0, 0),
    angle_color: Tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """
    Draw two lines, their intersection, angle arc, and text label on an image.
    
    Parameters:
    -----------
    image : Input image (will be converted to BGR if grayscale)
    line1 : First line parameters from fit_endplate_line
    line2 : Second line parameters (None for horizontal reference)
    text : Text label to display
    color1, color2 : BGR colors for lines
    angle_color : BGR color for angle arc
    
    Returns:
    --------
    vis_image : Visualization image with overlays
    """
    if cv2 is None:
        return image
    
    h, w = image.shape[:2]
    
    # Convert to BGR if grayscale
    if len(image.shape) == 2:
        vis_img = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    else:
        vis_img = image.copy()
    
    # Draw line1
    pts1 = _line_to_points(line1, w, h)
    if pts1 is not None:
        p1, p2 = pts1
        # CRITICAL: cv2.line uses (x, y) = (col, row) format
        # p1[0] = x (column), p1[1] = y (row)
        # Use round() to avoid truncation errors
        cv2.line(vis_img, (int(round(p1[0])), int(round(p1[1]))), (int(round(p2[0])), int(round(p2[1]))), color1, 2)
    
    # Draw line2 (if provided)
    if line2 is not None:
        pts2 = _line_to_points(line2, w, h)
        if pts2 is not None:
            p1, p2 = pts2
            # CRITICAL: cv2.line uses (x, y) = (col, row) format
            cv2.line(vis_img, (int(round(p1[0])), int(round(p1[1]))), (int(round(p2[0])), int(round(p2[1]))), color2, 2, cv2.LINE_AA)
    else:
        # Draw horizontal reference line
        cy = h // 2
        cv2.line(vis_img, (0, cy), (w, cy), color2, 2, cv2.LINE_AA)
        pts2 = None
    
    # Calculate intersection and draw angle arc
    if line2 is not None:
        intersection = _get_line_intersection(line1, line2, w, h)
        if intersection is not None:
            ix, iy = intersection[0], intersection[1]
            
            # Draw intersection point
            cv2.circle(vis_img, (int(ix), int(iy)), 5, (255, 255, 0), -1)
            
            # Calculate angle between lines
            if pts1 is not None and pts2 is not None:
                # Get direction vectors from intersection point
                dx1 = pts1[1][0] - ix
                dy1 = pts1[1][1] - iy
                dx2 = pts2[1][0] - ix
                dy2 = pts2[1][1] - iy
                
                # Normalize
                norm1 = np.sqrt(dx1*dx1 + dy1*dy1)
                norm2 = np.sqrt(dx2*dx2 + dy2*dy2)
                if norm1 > 1e-9 and norm2 > 1e-9:
                    dx1, dy1 = dx1/norm1, dy1/norm1
                    dx2, dy2 = dx2/norm2, dy2/norm2
                    
                    angle1 = np.degrees(np.arctan2(dy1, dx1))
                    angle2 = np.degrees(np.arctan2(dy2, dx2))
                    
                    # Draw arc (cv2.ellipse uses 0-360 degrees, with 0 at 3 o'clock, clockwise)
                    radius = 30
                    start_angle = int(angle1)
                    end_angle = int(angle2)
                    
                    # Ensure proper angle range
                    if end_angle < start_angle:
                        start_angle, end_angle = end_angle, start_angle
                    
                    cv2.ellipse(vis_img, (int(ix), int(iy)), (radius, radius), 0, start_angle, end_angle, angle_color, 2)
    
    # Add text label
    cv2.putText(vis_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    
    return vis_img


def get_s1_superior_line(mask_s: np.ndarray, mask_l5: np.ndarray, mask_l5s1_disc: Optional[np.ndarray] = None, mask_spinal_canal: Optional[np.ndarray] = None, fit_func=None, debug_roi=None) -> Optional[Tuple[float, float, float, float]]:
    """
    Get S1 superior endplate line by finding two points in S1:
    - Point 1: Leftmost candidate point (closest to L5-S1 disc, within 20% tolerance)
    - Point 2: Rightmost candidate point (closest to L5-S1 disc, within 20% tolerance)
    Then connect these two points to form the S1 superior endplate line covering the full width.
    
    CRITICAL: 在S1内部取两个点，然后形成一条线。
    第一个点：候选点中x坐标最小的点（最左侧），同时离L5-S1椎间盘距离最近（20%容差内）
    第二个点：候选点中x坐标最大的点（最右侧），同时离L5-S1椎间盘距离最近（20%容差内）
    这样可以确保线覆盖整个椎骨的宽度
    
    Parameters:
    -----------
    mask_s : 2D binary mask of S1/Sacrum (sagittal slice)
    mask_l5 : 2D binary mask of L5 (sagittal slice, used as fallback reference)
    mask_l5s1_disc : 2D binary mask of L5-S1 disc (sagittal slice, REQUIRED for accurate extraction)
    mask_spinal_canal : 2D binary mask of spinal canal (label 2, REQUIRED for accurate extraction)
    fit_func : Not used (kept for compatibility)
    debug_roi : Optional dict to store ROI info for visualization
    
    Returns:
    --------
    (vx, vy, x0, y0) : Line parameters from cv2.fitLine, or None if failed
    Also stores point1 and point2 in debug_roi if provided
    """
    if mask_s.sum() == 0:
        return None
    
    if cv2 is None:
        return None

    # CRITICAL: Both L5-S1 disc and spinal canal are REQUIRED for accurate extraction
    if mask_l5s1_disc is None or mask_l5s1_disc.sum() == 0:
        # Fallback: use L5 position if disc is not available
        if mask_l5 is None or mask_l5.sum() == 0:
            return _fallback_s1_horizontal_line(((mask_s > 0) * 255).astype(np.uint8))
        mask_l5s1_disc = mask_l5  # Use L5 as fallback
    
    if mask_spinal_canal is None or mask_spinal_canal.sum() == 0:
        # Cannot proceed without spinal canal
        return _fallback_s1_horizontal_line(((mask_s > 0) * 255).astype(np.uint8))
    
    # Ensure binary masks
    ms = ((mask_s > 0) * 255).astype(np.uint8)
    ml5s1 = ((mask_l5s1_disc > 0) * 255).astype(np.uint8)
    mcanal = ((mask_spinal_canal > 0) * 255).astype(np.uint8)
    
    # Step 1: Get spinal canal boundary (contour)
    canal_contours, _ = cv2.findContours(mcanal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not canal_contours:
        return _fallback_s1_horizontal_line(ms)
    
    # Get the largest contour (main canal)
    canal_contour = max(canal_contours, key=cv2.contourArea)
    canal_boundary_pts = canal_contour.reshape(-1, 2).astype(np.float32)  # Shape: (N, 2) where each point is (x, y)
    
    if len(canal_boundary_pts) == 0:
        return _fallback_s1_horizontal_line(ms)
    
    # Step 2: Get L5-S1 disc region (for distance calculation)
    disc_y_coords, disc_x_coords = np.where(ml5s1 > 0)
    if len(disc_x_coords) == 0:
        return _fallback_s1_horizontal_line(ms)
    
    # Step 3: Get all points inside S1 mask
    s1_y_coords, s1_x_coords = np.where(ms > 0)
    if len(s1_x_coords) == 0:
        return _fallback_s1_horizontal_line(ms)
    
    s1_points = np.stack([s1_x_coords.astype(np.float32), s1_y_coords.astype(np.float32)], axis=1)  # Shape: (N, 2) as (x, y)
    
    # Step 4: Calculate distances for each S1 point
    # 4.1: Distance to spinal canal boundary
    # Use cKDTree for efficient nearest neighbor search
    from scipy.spatial import cKDTree
    canal_tree = cKDTree(canal_boundary_pts)
    distances_to_canal, _ = canal_tree.query(s1_points, k=1)
    
    # 4.2: Distance to L5-S1 disc (use centroid for simplicity, or could use distance transform)
    disc_centroid_x = float(disc_x_coords.mean())
    disc_centroid_y = float(disc_y_coords.mean())
    disc_centroid = np.array([[disc_centroid_x, disc_centroid_y]], dtype=np.float32)
    
    # Calculate distance to disc centroid for each S1 point
    distances_to_disc = np.sqrt(np.sum((s1_points - disc_centroid) ** 2, axis=1))
    
    # Step 5: Find two points
    # CRITICAL: First prioritize distance to L5-S1 disc (closest), then consider canal distance
    # Strategy: 
    # 1. Find points closest to disc (within 20% tolerance of minimum distance)
    # 2. Among these candidates:
    #    - Point 1: Select the leftmost point (minimum x-coordinate)
    #    - Point 2: Select the rightmost point (maximum x-coordinate)
    #    This ensures the line covers the full width of the vertebra
    
    # Find the minimum distance to disc
    min_disc_distance = float(np.min(distances_to_disc))
    
    # Find all points that are close to the minimum distance (within 20% tolerance)
    # This gives us candidates that are all "closest to disc"
    tolerance = max(min_disc_distance * 0.20, 1.0)  # 20% tolerance, but at least 1 pixel
    candidates_mask = distances_to_disc <= (min_disc_distance + tolerance)
    candidate_indices = np.where(candidates_mask)[0]
    
    if len(candidate_indices) == 0:
        # Fallback: use all points if no candidates found
        candidate_indices = np.arange(len(s1_points))
    
    # Among candidates (closest to disc), find points to maximize coverage
    candidate_distances_to_canal = distances_to_canal[candidate_indices]
    candidate_points = s1_points[candidate_indices]  # All candidate points
    
    # Strategy: Find points that maximize x-coordinate span (leftmost and rightmost)
    # while still considering canal distance to ensure anatomical correctness
    candidate_x_coords = candidate_points[:, 0]
    
    # Find the leftmost and rightmost candidate points
    # This ensures the line covers the full width of the vertebra
    leftmost_idx_in_candidates = int(np.argmin(candidate_x_coords))
    rightmost_idx_in_candidates = int(np.argmax(candidate_x_coords))
    
    idx_point1 = candidate_indices[leftmost_idx_in_candidates]
    idx_point2 = candidate_indices[rightmost_idx_in_candidates]
    point1 = s1_points[idx_point1]  # (x, y) - leftmost point
    point2 = s1_points[idx_point2]  # (x, y) - rightmost point
    
    # Step 6: Store points in debug_roi if provided
    if debug_roi is not None:
        debug_roi['point1'] = (float(point1[0]), float(point1[1]))
        debug_roi['point2'] = (float(point2[0]), float(point2[1]))
        # Also store bounding box
        x_coords = [point1[0], point2[0]]
        y_coords = [point1[1], point2[1]]
        debug_roi['x'] = int(min(x_coords))
        debug_roi['y'] = int(min(y_coords))
        debug_roi['w'] = int(max(x_coords) - min(x_coords))
        debug_roi['h'] = int(max(y_coords) - min(y_coords))
    
    # Step 7: Fit line through the two points
    # Convert to cv2 format: (N, 1, 2)
    pts_for_line = np.array([point1, point2], dtype=np.float32).reshape(-1, 1, 2)
    
    try:
        # Use cv2.fitLine to get line parameters
        line = cv2.fitLine(pts_for_line, cv2.DIST_L2, 0, 0.01, 0.01)
        if line is not None and len(line) >= 4:
            vx, vy, x0, y0 = line[0].item(), line[1].item(), line[2].item(), line[3].item()
            return (float(vx), float(vy), float(x0), float(y0))
    except Exception:
        pass
    
    # Fallback: Calculate line directly from two points
    # Line through point1 and point2: y = k*x + b
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    
    if abs(dx) < 1e-9:
        # Vertical line
        vx, vy = 0.0, 1.0
        x0, y0 = float(point1[0]), float(point1[1])
    else:
        # Normalize direction vector
        norm = np.sqrt(dx * dx + dy * dy)
        vx, vy = float(dx / norm), float(dy / norm)
        # Point on line (use midpoint)
        x0, y0 = float((point1[0] + point2[0]) / 2.0), float((point1[1] + point2[1]) / 2.0)
    
    return (vx, vy, x0, y0)
    
    # Step 1: Extract disc boundary (inferior edge) - bottom contour of disc
    disc_contours, _ = cv2.findContours(ml5s1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not disc_contours:
        return _fallback_s1_horizontal_line(ms)
    
    # Get the largest contour (main disc)
    disc_contour = max(disc_contours, key=cv2.contourArea)
    
    # Extract disc inferior edge points (bottom part of contour)
    # Find points with maximum Y values (bottom edge)
    disc_pts = disc_contour.reshape(-1, 2).astype(np.float32)  # Shape: (N, 2) where each point is (x, y)
    if len(disc_pts) == 0:
        return _fallback_s1_horizontal_line(ms)
    
    # Get disc's Y range and focus on bottom 30% of the contour
    disc_y_coords = disc_pts[:, 1]
    disc_y_min = float(disc_y_coords.min())
    disc_y_max = float(disc_y_coords.max())
    disc_y_threshold = disc_y_min + 0.7 * (disc_y_max - disc_y_min)  # Bottom 30%
    
    # Filter to get inferior edge points
    disc_inferior_mask = disc_pts[:, 1] >= disc_y_threshold
    disc_inferior_pts = disc_pts[disc_inferior_mask]
    
    if len(disc_inferior_pts) < 3:
        # Fallback: use all disc contour points
        disc_inferior_pts = disc_pts
    
    # Step 2: Extract S1 boundary in the region adjacent to disc
    # CRITICAL: S1 is the entire sacrum label, so we need to constrain to the region near disc
    
    # Get disc X and Y range to define ROI
    disc_x_min = float(disc_inferior_pts[:, 0].min())
    disc_x_max = float(disc_inferior_pts[:, 0].max())
    disc_y_min = float(disc_inferior_pts[:, 1].min())
    disc_y_max = float(disc_inferior_pts[:, 1].max())
    
    # Define ROI: X range from disc (with padding), Y range from disc inferior edge downward
    roi_x_start = max(0, int(disc_x_min) - 5)
    roi_x_end = min(ms.shape[1] - 1, int(disc_x_max) + 5)
    roi_y_start = max(0, int(disc_y_min) - 5)  # Start slightly above disc
    roi_y_end = min(ms.shape[0] - 1, int(disc_y_max) + 20)  # Extend 20 pixels below disc
    
    # Create ROI mask for S1
    s1_roi = np.zeros_like(ms)
    s1_roi[roi_y_start:roi_y_end+1, roi_x_start:roi_x_end+1] = ms[roi_y_start:roi_y_end+1, roi_x_start:roi_x_end+1]
    
    # Extract S1 main body, ignoring attachments (附件)
    # Use connected components to find the largest component (main body)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(s1_roi, connectivity=8)
    
    if num_labels < 2:  # Only background (label 0)
        return _fallback_s1_horizontal_line(ms)
    
    # Find the largest component (main body)
    component_areas = stats[1:, cv2.CC_STAT_AREA]  # Skip label 0 (background)
    largest_label = int(np.argmax(component_areas)) + 1  # +1 because we skipped label 0
    
    # Create mask with only the main body
    s1_body_roi = np.zeros_like(s1_roi)
    s1_body_roi[labels == largest_label] = 255
    
    # Optional: Apply morphological opening to remove small attachments
    # This helps remove small protrusions while keeping the main body
    kernel_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    s1_body_roi = cv2.morphologyEx(s1_body_roi, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Extract S1 boundary in ROI region (main body only)
    s1_contours, _ = cv2.findContours(s1_body_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not s1_contours:
        return _fallback_s1_horizontal_line(ms)
    
    # Get the largest contour in ROI (should be the main body)
    s1_contour = max(s1_contours, key=cv2.contourArea)
    
    # Extract S1 superior edge points in ROI (top part relative to disc)
    s1_pts = s1_contour.reshape(-1, 2).astype(np.float32)  # Shape: (N, 2)
    if len(s1_pts) == 0:
        return _fallback_s1_horizontal_line(ms)
    
    # Filter points: only keep points that are near disc inferior edge (within 15 pixels vertically)
    # and in the top part of the ROI (above disc_y_max + 10)
    s1_superior_pts = []
    for pt in s1_pts:
        x, y = pt[0], pt[1]
        # Point should be near disc in X direction
        if disc_x_min - 5 <= x <= disc_x_max + 5:
            # Point should be near disc inferior edge in Y direction (within 15 pixels)
            if disc_y_max - 5 <= y <= disc_y_max + 15:
                s1_superior_pts.append(pt)
    
    s1_superior_pts = np.array(s1_superior_pts, dtype=np.float32) if len(s1_superior_pts) > 0 else s1_pts
    
    if len(s1_superior_pts) < 3:
        # Fallback: use all S1 points in ROI that are below disc
        s1_below_disc = []
        for pt in s1_pts:
            if disc_y_max <= pt[1] <= disc_y_max + 20:
                s1_below_disc.append(pt)
        s1_superior_pts = np.array(s1_below_disc, dtype=np.float32) if len(s1_below_disc) > 0 else s1_pts
    
    # Step 3: Find interface points by matching X coordinates
    # Strategy: For each X coordinate, find the closest pair of points (disc inferior and S1 superior)
    # This handles cases where boundaries don't directly touch but are close
    
    # Get X range that covers both disc and S1
    all_x_coords = np.concatenate([disc_inferior_pts[:, 0], s1_superior_pts[:, 0]])
    x_min = float(np.min(all_x_coords))
    x_max = float(np.max(all_x_coords))
    
    # Sample X coordinates at regular intervals (every 1-2 pixels)
    x_step = 1.0  # Sample every pixel
    x_samples = np.arange(x_min, x_max + x_step, x_step)
    
    interface_points = []
    max_vertical_distance = 15.0  # Maximum vertical distance (Y difference) to consider points as adjacent
    
    for x in x_samples:
        # Find disc inferior points near this X
        disc_x_distances = np.abs(disc_inferior_pts[:, 0] - x)
        disc_near_mask = disc_x_distances <= 2.0  # Within 2 pixels in X direction
        disc_near_pts = disc_inferior_pts[disc_near_mask]
        
        # Find S1 superior points near this X
        s1_x_distances = np.abs(s1_superior_pts[:, 0] - x)
        s1_near_mask = s1_x_distances <= 2.0  # Within 2 pixels in X direction
        s1_near_pts = s1_superior_pts[s1_near_mask]
        
        if len(disc_near_pts) == 0 or len(s1_near_pts) == 0:
            continue
        
        # For this X, find the closest pair (minimum vertical distance)
        # Get Y coordinates
        disc_y = disc_near_pts[:, 1]  # Disc inferior Y (larger values = lower in image)
        s1_y = s1_near_pts[:, 1]      # S1 superior Y (smaller values = higher in image)
        
        # Find the pair with minimum vertical distance
        min_dist = float('inf')
        best_disc_y = None
        best_s1_y = None
        
        for d_y in disc_y:
            for s_y in s1_y:
                # Vertical distance (Y difference)
                vertical_dist = abs(d_y - s_y)
                # Also check that S1 is below disc (s1_y should be >= disc_y, but allow some tolerance)
                if vertical_dist < min_dist and vertical_dist <= max_vertical_distance:
                    # Additional check: S1 should be at or below disc (with tolerance)
                    if s_y >= d_y - 3:  # Allow 3 pixel tolerance for S1 being slightly above disc
                        min_dist = vertical_dist
                        best_disc_y = d_y
                        best_s1_y = s_y
        
        if best_disc_y is not None and best_s1_y is not None:
            # Use the midpoint or S1 point as interface point
            # Since S1 superior edge is what we want, use S1 point or midpoint
            interface_y = (best_disc_y + best_s1_y) / 2.0
            interface_points.append([float(x), float(interface_y)])
    
    if len(interface_points) < 5:
        # Fallback: try with larger X tolerance
        interface_points = []
        for x in x_samples:
            disc_x_distances = np.abs(disc_inferior_pts[:, 0] - x)
            disc_near_mask = disc_x_distances <= 5.0
            disc_near_pts = disc_inferior_pts[disc_near_mask]
            
            s1_x_distances = np.abs(s1_superior_pts[:, 0] - x)
            s1_near_mask = s1_x_distances <= 5.0
            s1_near_pts = s1_superior_pts[s1_near_mask]
            
            if len(disc_near_pts) == 0 or len(s1_near_pts) == 0:
                continue
            
            disc_y = disc_near_pts[:, 1]
            s1_y = s1_near_pts[:, 1]
            
            min_dist = float('inf')
            best_disc_y = None
            best_s1_y = None
            
            for d_y in disc_y:
                for s_y in s1_y:
                    vertical_dist = abs(d_y - s_y)
                    if vertical_dist < min_dist and vertical_dist <= max_vertical_distance * 1.5:
                        if s_y >= d_y - 5:
                            min_dist = vertical_dist
                            best_disc_y = d_y
                            best_s1_y = s_y
            
            if best_disc_y is not None and best_s1_y is not None:
                interface_y = (best_disc_y + best_s1_y) / 2.0
                interface_points.append([float(x), float(interface_y)])
    
    if len(interface_points) < 3:
        return _fallback_s1_horizontal_line(ms)
    
    # Remove duplicates (points that are very close to each other)
    interface_points = np.array(interface_points, dtype=np.float32)
    if len(interface_points) > 0:
        # Simple deduplication: keep only points that are at least 1 pixel apart
        unique_points = []
        for pt in interface_points:
            is_duplicate = False
            for existing_pt in unique_points:
                if np.sqrt(np.sum((pt - existing_pt) ** 2)) < 1.0:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_points.append(pt)
        interface_points = np.array(unique_points, dtype=np.float32)
    
    if len(interface_points) < 3:
        return _fallback_s1_horizontal_line(ms)
    
    # Step 4: Filter out points from attachments, keep only main body points
    # Strategy: Remove outliers that are likely from attachments using multiple criteria
    
    interface_points = np.array(interface_points, dtype=np.float32)
    
    # 4.1: Filter by X coordinate - main body should be in the central X range
    x_coords = interface_points[:, 0]
    x_median = float(np.median(x_coords))
    x_mad = float(np.median(np.abs(x_coords - x_median)))  # Median Absolute Deviation
    
    if x_mad > 0:
        # Keep points within 3 * MAD from median (main body region)
        x_mask = np.abs(x_coords - x_median) <= 3.0 * x_mad
        interface_points = interface_points[x_mask]
    
    if len(interface_points) < 3:
        return _fallback_s1_horizontal_line(ms)
    
    # 4.2: Filter by Y coordinate consistency - main body should have consistent Y values
    y_coords = interface_points[:, 1]
    y_median = float(np.median(y_coords))
    y_mad = float(np.median(np.abs(y_coords - y_median)))
    
    if y_mad > 0:
        # Keep points within 2.5 * MAD from median (consistent with main body)
        y_mask = np.abs(y_coords - y_median) <= 2.5 * y_mad
        interface_points = interface_points[y_mask]
    
    if len(interface_points) < 3:
        return _fallback_s1_horizontal_line(ms)
    
    # 4.3: Preliminary line fitting and distance-based filtering
    # Fit a preliminary line to identify main body points
    pts_prelim = interface_points.reshape(-1, 1, 2).astype(np.float32)
    try:
        prelim_line = cv2.fitLine(pts_prelim, cv2.DIST_HUBER, 0, 0.01, 0.01)
        if prelim_line is not None and len(prelim_line) >= 4:
            vx, vy, x0, y0 = prelim_line[0].item(), prelim_line[1].item(), prelim_line[2].item(), prelim_line[3].item()
            
            # Calculate distance from each point to the preliminary line
            # Line equation: (x - x0) * vy - (y - y0) * vx = 0
            # Distance = |(x - x0) * vy - (y - y0) * vx| / sqrt(vx^2 + vy^2)
            line_norm = np.sqrt(vx * vx + vy * vy)
            if line_norm > 1e-9:
                distances = []
                for pt in interface_points:
                    x, y = pt[0], pt[1]
                    dist = abs((x - x0) * vy - (y - y0) * vx) / line_norm
                    distances.append(dist)
                
                distances = np.array(distances)
                dist_median = float(np.median(distances))
                dist_mad = float(np.median(np.abs(distances - dist_median)))
                
                if dist_mad > 0:
                    # Keep points within 2.5 * MAD from median distance (main body points)
                    dist_mask = np.abs(distances - dist_median) <= 2.5 * dist_mad
                    interface_points = interface_points[dist_mask]
    except Exception:
        pass  # If preliminary fitting fails, use all points
    
    if len(interface_points) < 3:
        return _fallback_s1_horizontal_line(ms)
    
    # Store ROI info for debugging
    if debug_roi is not None:
        if len(interface_points) > 0:
            x_coords = interface_points[:, 0]
            y_coords = interface_points[:, 1]
            debug_roi['x'] = int(x_coords.min())
            debug_roi['y'] = int(y_coords.min())
            debug_roi['w'] = int(x_coords.max() - x_coords.min())
            debug_roi['h'] = int(y_coords.max() - y_coords.min())
        else:
            # Use the ROI region we defined
            debug_roi['x'] = roi_x_start
            debug_roi['y'] = roi_y_start
            debug_roi['w'] = roi_x_end - roi_x_start
            debug_roi['h'] = roi_y_end - roi_y_start
    
    # Step 5: Final Robust Line Fitting using DIST_HUBER (on filtered main body points)
    pts_cv = interface_points.reshape(-1, 1, 2).astype(np.float32)
    
    try:
        # DIST_HUBER is more robust to outliers
        line = cv2.fitLine(pts_cv, cv2.DIST_HUBER, 0, 0.01, 0.01)
        if line is not None and len(line) >= 4:
            vx, vy, x0, y0 = line[0].item(), line[1].item(), line[2].item(), line[3].item()
            return (float(vx), float(vy), float(x0), float(y0))
    except Exception:
        pass
    
    # Fallback to DIST_L2 if DIST_HUBER fails
    try:
        line = cv2.fitLine(pts_cv, cv2.DIST_L2, 0, 0.01, 0.01)
        if line is not None and len(line) >= 4:
            vx, vy, x0, y0 = line[0].item(), line[1].item(), line[2].item(), line[3].item()
            return (float(vx), float(vy), float(x0), float(y0))
    except Exception:
        pass
    
    # Final fallback
    return _fallback_s1_horizontal_line(ms)


def _fallback_s1_horizontal_line(mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """
    Fallback mechanism for S1: Use top-most edge pixels of raw mask to draw a horizontal line.
    
    Parameters:
    -----------
    mask : 2D binary mask
    
    Returns:
    --------
    (0.0, y_center, None, None) : Horizontal line (slope=0) passing through top-most pixel
    """
    y_coords, x_coords = np.where(mask > 0)
    if len(y_coords) == 0:
        return None
    
    # Find top-most pixel
    min_y_idx = np.argmin(y_coords)
    y_center = float(y_coords[min_y_idx])
    
    # Return horizontal line (slope = 0) passing through top-most pixel
    # Format: (k, b, None, None) where k=0 (horizontal), b=y_center
    return (0.0, float(y_center), None, None)


def calculate_angle_between_lines(line1: Tuple[float, float, float, float], line2: Tuple[float, float, float, float]) -> Optional[float]:
    """
    Calculate the ACUTE angle between two lines (their extended lines intersection angle).
    
    This function calculates the angle between two lines by:
    1. Getting direction vectors for each line
    2. Calculating the angle between the direction vectors
    3. Ensuring the result is an acute angle (0-90 degrees)
    
    Parameters:
    -----------
    line1, line2 : Line parameters from fit_endplate_line
                   Format: (vx, vy, x0, y0) from cv2.fitLine, or (k, b, None, None) from np.polyfit
                   Special case: (inf, x, None, None) for vertical line at x
    
    Returns:
    --------
    angle_deg : Acute angle in degrees (0-90), or None if calculation fails
    """
    # Convert both lines to direction vectors
    def line_to_direction_vector(line):
        if len(line) == 4 and line[2] is not None:
            # cv2.fitLine format: (vx, vy, x0, y0)
            # Handle both NumPy arrays and tuples
            vx = line[0].item() if hasattr(line[0], 'item') else line[0]
            vy = line[1].item() if hasattr(line[1], 'item') else line[1]
            x0 = line[2].item() if hasattr(line[2], 'item') else line[2]
            y0 = line[3].item() if hasattr(line[3], 'item') else line[3]
            # Normalize direction vector
            norm = np.sqrt(vx*vx + vy*vy)
            if norm < 1e-9:
                return None
            return (float(vx/norm), float(vy/norm))
        elif len(line) >= 2:
            # np.polyfit format: (k, b, None, None)
            k_val = line[0].item() if hasattr(line[0], 'item') else line[0]
            k = float(k_val)
            # Handle vertical line (k = inf)
            if k == float('inf'):
                # Vertical line: direction vector is (0, 1) or (0, -1), we use (0, 1)
                return (0.0, 1.0)
            # Direction vector for line y = k*x + b is (1, k) normalized
            norm = np.sqrt(1 + k*k)
            if norm < 1e-9:
                return None
            return (float(1/norm), float(k/norm))
        return None
    
    dir1 = line_to_direction_vector(line1)
    dir2 = line_to_direction_vector(line2)
    
    if dir1 is None or dir2 is None:
        return None
    
    dx1, dy1 = dir1
    dx2, dy2 = dir2
    
    # Calculate angle between two direction vectors using dot product
    # cos(θ) = (v1 · v2) / (|v1| * |v2|)
    # Since vectors are normalized, |v1| = |v2| = 1
    dot_product = dx1 * dx2 + dy1 * dy2
    
    # Clamp dot product to [-1, 1] to avoid numerical errors
    dot_product = max(-1.0, min(1.0, dot_product))
    
    # Calculate angle in radians
    angle_rad = np.arccos(dot_product)
    
    # Convert to degrees
    angle_deg = float(np.degrees(angle_rad))
    
    # Ensure we return the acute angle (0-90 degrees)
    # If angle > 90, use the supplement (180 - angle)
    if angle_deg > 90.0:
        angle_deg = 180.0 - angle_deg
    
    return angle_deg


def calculate_angle_to_horizontal(line: Tuple[float, float, float, float]) -> Optional[float]:
    """
    Calculate the angle between a line and the horizontal vector (1, 0).
    
    Parameters:
    -----------
    line : Line parameters from fit_endplate_line
    
    Returns:
    --------
    angle_deg : Angle in degrees, or None if calculation fails
    """
    # Convert line to slope-intercept format
    if len(line) == 4 and line[2] is not None:
        # Handle both NumPy arrays and tuples
        vx = line[0].item() if hasattr(line[0], 'item') else line[0]
        vy = line[1].item() if hasattr(line[1], 'item') else line[1]
        x0 = line[2].item() if hasattr(line[2], 'item') else line[2]
        y0 = line[3].item() if hasattr(line[3], 'item') else line[3]
        if abs(vx) < 1e-9:
            return 90.0  # Vertical line
        k = float(vy / vx)
    elif len(line) >= 2:
        k_val = line[0].item() if hasattr(line[0], 'item') else line[0]
        k = float(k_val)
    else:
        return None
    
    # Angle with horizontal: arctan(k)
    angle_rad = np.arctan(k)
    return float(abs(np.degrees(angle_rad)))


def weighted_line_fit(points: np.ndarray, weights: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """
    使用加权最小二乘法拟合直线。
    
    参数:
    ------
    points : np.ndarray, shape (N, 2)
        点集，每行为 (x, y)
    weights : np.ndarray, shape (N,)
        每个点的权重
    
    返回:
    ------
    (vx, vy, x0, y0) : 直线参数，与cv2.fitLine格式兼容
        如果拟合失败，返回None
    """
    if len(points) < 2:
        return None
    
    if len(weights) != len(points):
        return None
    
    # 归一化权重
    weights = weights / np.sum(weights) * len(weights)
    
    x = points[:, 0]
    y = points[:, 1]
    
    # 加权最小二乘法拟合 y = k*x + b
    # 最小化 sum(w_i * (y_i - k*x_i - b)^2)
    w_sum = np.sum(weights)
    w_x = np.sum(weights * x)
    w_y = np.sum(weights * y)
    w_xx = np.sum(weights * x * x)
    w_xy = np.sum(weights * x * y)
    
    # 求解线性方程组
    # w_sum * b + w_x * k = w_y
    # w_x * b + w_xx * k = w_xy
    det = w_sum * w_xx - w_x * w_x
    
    if abs(det) < 1e-9:
        # 垂直线或退化情况
        return None
    
    k = (w_sum * w_xy - w_x * w_y) / det
    b = (w_xx * w_y - w_x * w_xy) / det
    
    # 转换为cv2.fitLine格式 (vx, vy, x0, y0)
    # 方向向量归一化
    vx = 1.0 / np.sqrt(1.0 + k * k)
    vy = k * vx
    
    # 选择一个点作为参考点（使用加权中心）
    x0 = w_x / w_sum
    y0 = w_y / w_sum
    
    return (float(vx), float(vy), float(x0), float(y0))


def extract_endplate_line_by_disc_and_canal(
    vertebra_mask: np.ndarray,
    adjacent_disc_mask: np.ndarray,
    spinal_canal_mask: np.ndarray,
    debug_roi: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Extract endplate line using disc and canal distance method.
    
    Core logic:
    1. Find candidate points: Uniformly sample points across canal distance range
       - Divide the canal distance range into bins (10-20 bins, adaptive)
       - For each bin, select the point closest to the adjacent disc
       - This ensures candidate points are evenly distributed across the vertebra width
    2. Use all candidate points to fit a line, then extend it to full vertebra width
    3. The line covers the full width of the vertebra from x_min to x_max
    
    Parameters:
    -----------
    vertebra_mask : 2D binary mask of vertebra (sagittal slice)
    adjacent_disc_mask : 2D binary mask of adjacent disc (sagittal slice, REQUIRED)
    spinal_canal_mask : 2D binary mask of spinal canal (label 2, REQUIRED)
    
    Returns:
    --------
    (vx, vy, x0, y0) : Line parameters from cv2.fitLine, or None if failed
    """
    if cv2 is None:
        return None
    
    if vertebra_mask.sum() == 0:
        return None
    
    if adjacent_disc_mask is None or adjacent_disc_mask.sum() == 0:
        return None
    
    if spinal_canal_mask is None or spinal_canal_mask.sum() == 0:
        return None
    
    # Ensure binary masks
    m_vert = ((vertebra_mask > 0) * 255).astype(np.uint8)
    m_disc = ((adjacent_disc_mask > 0) * 255).astype(np.uint8)
    m_canal = ((spinal_canal_mask > 0) * 255).astype(np.uint8)
    
    # Step 1: Get spinal canal boundary (contour)
    canal_contours, _ = cv2.findContours(m_canal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not canal_contours:
        return None
    
    # Get the largest contour (main canal)
    canal_contour = max(canal_contours, key=cv2.contourArea)
    canal_boundary_pts = canal_contour.reshape(-1, 2).astype(np.float32)  # Shape: (N, 2) where each point is (x, y)
    
    if len(canal_boundary_pts) == 0:
        return None
    
    # Step 2: Get adjacent disc region (for distance calculation)
    disc_y_coords, disc_x_coords = np.where(m_disc > 0)
    if len(disc_x_coords) == 0:
        return None
    
    # Step 3: Get all points inside vertebra mask
    vert_y_coords, vert_x_coords = np.where(m_vert > 0)
    if len(vert_x_coords) == 0:
        return None
    
    vert_points = np.stack([vert_x_coords.astype(np.float32), vert_y_coords.astype(np.float32)], axis=1)  # Shape: (N, 2) as (x, y)
    
    # Step 4: Calculate distances for each vertebra point
    # 4.1: Distance to spinal canal boundary
    from scipy.spatial import cKDTree
    canal_tree = cKDTree(canal_boundary_pts)
    distances_to_canal, _ = canal_tree.query(vert_points, k=1)
    
    # 4.2: Distance to adjacent disc (use centroid for simplicity)
    disc_centroid_x = float(disc_x_coords.mean())
    disc_centroid_y = float(disc_y_coords.mean())
    disc_centroid = np.array([[disc_centroid_x, disc_centroid_y]], dtype=np.float32)
    
    # Calculate distance to disc centroid for each vertebra point
    distances_to_disc = np.sqrt(np.sum((vert_points - disc_centroid) ** 2, axis=1))
    
    # Step 5: Find candidate points based on canal distance bins
    # 新策略：根据到canal的距离，均匀产生候选点
    # 1. 将到canal的距离范围分成若干区间
    # 2. 在每个区间内，选择距离相邻椎间盘最近的点
    # 这样可以确保候选点在到canal的不同距离上都有均匀分布
    
    min_canal_dist = float(np.min(distances_to_canal))
    max_canal_dist = float(np.max(distances_to_canal))
    canal_dist_range = max_canal_dist - min_canal_dist
    
    candidate_indices = []
    
    if canal_dist_range > 1e-6:
        # 将距离范围分成若干区间（建议10-20个区间，根据椎体大小自适应）
        # 区间数量：至少10个，最多20个，根据距离范围自适应
        num_bins = max(10, min(20, int(canal_dist_range / 2.0)))  # 每个区间约2像素宽
        
        # 创建距离区间
        bin_edges = np.linspace(min_canal_dist, max_canal_dist, num_bins + 1)
        
        # 对每个区间，找到距离椎间盘最近的点
        for i in range(num_bins):
            bin_min = bin_edges[i]
            bin_max = bin_edges[i + 1]
            
            # 找到在当前距离区间内的点
            in_bin_mask = (distances_to_canal >= bin_min) & (distances_to_canal < bin_max)
            # 最后一个区间包含最大值
            if i == num_bins - 1:
                in_bin_mask = (distances_to_canal >= bin_min) & (distances_to_canal <= bin_max)
            
            bin_indices = np.where(in_bin_mask)[0]
            
            if len(bin_indices) > 0:
                # 在该区间内，选择距离椎间盘最近的点
                bin_disc_distances = distances_to_disc[bin_indices]
                min_disc_idx_in_bin = bin_indices[np.argmin(bin_disc_distances)]
                candidate_indices.append(min_disc_idx_in_bin)
        
        candidate_indices = np.array(candidate_indices, dtype=np.int64)
    else:
        # 如果距离范围太小（所有点到canal的距离几乎相同），使用旧策略
        min_disc_distance = float(np.min(distances_to_disc))
        tolerance = max(min_disc_distance * 0.10, 1.0)  # 10% tolerance, but at least 1 pixel
        candidates_mask = distances_to_disc <= (min_disc_distance + tolerance)
        candidate_indices = np.where(candidates_mask)[0]
    
    if len(candidate_indices) == 0:
        # Fallback: use all points if no candidates found
        candidate_indices = np.arange(len(vert_points))
    
    # Step 6: Use all candidate points to fit a line, then extend it to full vertebra width
    candidate_points = vert_points[candidate_indices]  # All candidate points
    candidate_distances_to_canal = distances_to_canal[candidate_indices]  # Distances to canal for candidate points
    
    if len(candidate_points) < 2:
        # Fallback: use all vertebra points if too few candidates
        candidate_points = vert_points
        candidate_indices = np.arange(len(vert_points))
        candidate_distances_to_canal = distances_to_canal
    
    # Get the x-coordinate range of the entire vertebra mask
    vert_x_min = float(vert_x_coords.min())
    vert_x_max = float(vert_x_coords.max())
    
    # Step 7: Calculate weights based on distance to canal
    # 加权策略：强调椎体边缘的点（前缘和后缘），降低中间部分的影响
    # 原理：距离canal最近的点（后缘）和距离canal最远的点（前缘）更能代表椎体上下缘的真实位置
    # 算法：使用二次函数，中间距离的点权重低，两端（最小和最大距离）的点权重高
    if len(candidate_distances_to_canal) > 0:
        min_dist = float(np.min(candidate_distances_to_canal))
        max_dist = float(np.max(candidate_distances_to_canal))
        dist_range = max_dist - min_dist
        
        if dist_range > 1e-6:
            # 归一化距离到 [0, 1]
            # normalized_dist = 0: 距离canal最近的点（后缘）
            # normalized_dist = 1: 距离canal最远的点（前缘）
            # normalized_dist = 0.5: 距离canal中间位置的点
            normalized_dist = (candidate_distances_to_canal - min_dist) / dist_range
            
            # 计算权重：使用二次函数 weight = 4 * (normalized_dist - 0.5)^2 + 0.1
            # - 当 normalized_dist = 0.5 时，权重 = 0.1（最低，中间位置）
            # - 当 normalized_dist = 0 或 1 时，权重 = 1.0 + 0.1 = 1.1（最高，边缘位置）
            # 这样前缘和后缘的点权重高，中间的点权重低
            weights = 4.0 * (normalized_dist - 0.5) ** 2
            # 添加基础权重0.1，避免中间权重为0，确保所有点都有最小权重
            weights = weights + 0.1
            # 权重范围：应该在 [0.1, 1.1] 之间
            # 确保权重为float32类型
            weights = weights.astype(np.float32)
        else:
            # 所有距离相同（dist_range <= 1e-6），使用均匀权重
            weights = np.ones(len(candidate_distances_to_canal), dtype=np.float32)
    else:
        # 没有canal距离信息，使用均匀权重
        weights = np.ones(len(candidate_points), dtype=np.float32)
    
    # Step 8: Fit line using weighted linear fitting
    try:
        line = weighted_line_fit(candidate_points, weights)
        
        if line is not None and len(line) >= 4:
            vx, vy, x0_fit, y0_fit = line[0], line[1], line[2], line[3]
            
            # Extend the line to cover the full width of the vertebra
            # The line equation: (x, y) = (x0, y0) + t * (vx, vy)
            # We want to find t values that give us x = vert_x_min and x = vert_x_max
            
            if abs(vx) > 1e-9:  # Line is not vertical
                # Calculate t for x_min and x_max
                t_min = (vert_x_min - x0_fit) / vx
                t_max = (vert_x_max - x0_fit) / vx
                
                # Get the corresponding y coordinates
                y_min = y0_fit + t_min * vy
                y_max = y0_fit + t_max * vy
                
                # Use the midpoint of the extended line as the reference point
                x0 = (vert_x_min + vert_x_max) / 2.0
                y0 = (y_min + y_max) / 2.0
                
                # Store extended endpoints in debug_roi if provided
                if debug_roi is not None:
                    debug_roi['point1'] = (float(vert_x_min), float(y_min))
                    debug_roi['point2'] = (float(vert_x_max), float(y_max))
                    debug_roi['x'] = int(vert_x_min)
                    debug_roi['y'] = int(min(y_min, y_max))
                    debug_roi['w'] = int(vert_x_max - vert_x_min)
                    debug_roi['h'] = int(abs(y_max - y_min))
                
                return (float(vx), float(vy), float(x0), float(y0))
            else:
                # Vertical line - use original fit point
                return (float(vx), float(vy), float(x0_fit), float(y0_fit))
    except Exception:
        pass
    
    # Fallback: Use leftmost and rightmost candidate points
    candidate_x_coords = candidate_points[:, 0]
    leftmost_idx = int(np.argmin(candidate_x_coords))
    rightmost_idx = int(np.argmax(candidate_x_coords))
    
    point1 = candidate_points[leftmost_idx]
    point2 = candidate_points[rightmost_idx]
    
    # Extend line to full vertebra width
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    
    if abs(dx) < 1e-9:
        # Vertical line
        vx, vy = 0.0, 1.0
        x0, y0 = float(point1[0]), float(point1[1])
    else:
        # Normalize direction vector
        norm = np.sqrt(dx * dx + dy * dy)
        vx, vy = float(dx / norm), float(dy / norm)
        
        # Extend to full width
        # Line: y = y1 + (x - x1) * (dy/dx)
        y_at_xmin = point1[1] + (vert_x_min - point1[0]) * (dy / dx)
        y_at_xmax = point1[1] + (vert_x_max - point1[0]) * (dy / dx)
        
        # Use midpoint
        x0 = (vert_x_min + vert_x_max) / 2.0
        y0 = (y_at_xmin + y_at_xmax) / 2.0
        
        if debug_roi is not None:
            debug_roi['point1'] = (float(vert_x_min), float(y_at_xmin))
            debug_roi['point2'] = (float(vert_x_max), float(y_at_xmax))
            debug_roi['x'] = int(vert_x_min)
            debug_roi['y'] = int(min(y_at_xmin, y_at_xmax))
            debug_roi['w'] = int(vert_x_max - vert_x_min)
            debug_roi['h'] = int(abs(y_at_xmax - y_at_xmin))
    
    return (vx, vy, x0, y0)


def fit_endplate_line(body_mask: np.ndarray, mode: str = 'superior') -> Optional[Tuple[float, float, float, float]]:
    """
    Fit a line to the superior (top) or inferior (bottom) surface of the isolated vertebral body.
    
    Logic: Robust Fallback Mechanism
    1. Input Check: If body_mask is empty/None, return None.
    2. Point Collection:
       * Try to collect edge points (min Y or max Y per column).
       * Check: If fewer than 2 points are collected (cannot fit line), Do Not Fail.
    3. Fallback:
       * If generic fitting fails, find the absolute top-most (or bottom-most) pixel of the mask.
       * Return a horizontal line passing through that pixel (Slope = 0).
       * Reason: A flat line is a better approximation than crashing.
    
    Parameters:
    -----------
    body_mask : 2D binary mask of isolated vertebral body
    mode : 'superior' (top) or 'inferior' (bottom)
    
    Returns:
    --------
    (vx, vy, x0, y0) : line vector and point from cv2.fitLine, or
    (k, b, None, None) : slope and intercept from np.polyfit (if cv2.fitLine fails)
    Returns horizontal line (slope=0) as fallback if insufficient points
    """
    # 1) Input Check: If body_mask is empty/None, return None
    if body_mask.sum() == 0:
        return None
    
    # Extract all non-zero pixel coordinates
    y_coords, x_coords = np.where(body_mask > 0)
    if len(y_coords) == 0:
        return None
    
    # 2) Point Collection: Try to collect edge points (min Y or max Y per column)
    # Get unique x columns
    unique_x = np.unique(x_coords)
    
    # Step 1: Edge Point Extraction (Column-wise Scan)
    edge_points = []
    
    for x in unique_x:
        # Find all y values at this x
        y_at_x = y_coords[x_coords == x]
        
        if len(y_at_x) == 0:
            continue
        
        if mode == 'superior':
            # Select point with Minimum Y (Topmost pixel)
            y_selected = float(np.min(y_at_x))
        elif mode == 'inferior':
            # Select point with Maximum Y (Bottommost pixel)
            y_selected = float(np.max(y_at_x))
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'superior' or 'inferior'")
        
        edge_points.append([float(x), y_selected])
    
    # Check: If fewer than 2 points are collected (cannot fit line), Do Not Fail - use Fallback
    if len(edge_points) < 2:
        # 3) Fallback: Find the absolute top-most (or bottom-most) pixel of the mask
        if mode == 'superior':
            min_y_idx = np.argmin(y_coords)
            y_center = float(y_coords[min_y_idx])
        else:  # inferior
            max_y_idx = np.argmax(y_coords)
            y_center = float(y_coords[max_y_idx])
        
        # Return a horizontal line passing through that pixel (Slope = 0)
        # Format: (k, b, None, None) where k=0 (horizontal), b=y_center
        return (0.0, float(y_center), None, None)
    
    edge_points = np.array(edge_points, dtype=np.float32)
    
    # Try to fit line with at least 2 points (prefer 5+ for better fit)
    if len(edge_points) >= 5:
        # Step 2: Linear Regression
        # Try cv2.fitLine first (more robust)
        if cv2 is not None:
            try:
                line = cv2.fitLine(edge_points, cv2.DIST_L2, 0, 0.01, 0.01)
                if line is not None and len(line) >= 4:
                    vx, vy, x0, y0 = line[0].item(), line[1].item(), line[2].item(), line[3].item()
                    return (float(vx), float(vy), float(x0), float(y0))
            except Exception:
                pass  # Fall back to polyfit
        
        # Fallback: Use np.polyfit
        try:
            x_vals = edge_points[:, 0]
            y_vals = edge_points[:, 1]
            k, b = np.polyfit(x_vals, y_vals, 1)
            return (float(k), float(b), None, None)  # Return as (k, b, None, None) for compatibility
        except Exception:
            pass  # Fall through to horizontal line fallback
    
    # 3) Fallback: If generic fitting fails, return horizontal line
    # Find the absolute top-most (or bottom-most) pixel of the mask
    if mode == 'superior':
        min_y_idx = np.argmin(y_coords)
        y_center = float(y_coords[min_y_idx])
    else:  # inferior
        max_y_idx = np.argmax(y_coords)
        y_center = float(y_coords[max_y_idx])
    
    # Return a horizontal line passing through that pixel (Slope = 0)
    # Format: (k, b, None, None) where k=0 (horizontal), b=y_center
    return (0.0, float(y_center), None, None)


def extract_superior_endplate_points(mask: np.ndarray, roi_top_percent: float = 0.2, anterior_focus: bool = True) -> Optional[np.ndarray]:
    """
    Extract upper boundary points (superior endplate) from a vertebral mask.
    
    Algorithm:
    1. For each x-coordinate, find the pixel with minimum y-coordinate (top edge)
    2. Restrict to top ROI (top 20% of bounding box)
    3. Optionally prioritize anterior (left-most) portion
    
    Parameters:
    -----------
    mask : 2D binary mask array
    roi_top_percent : Fraction of bounding box height to consider (default 0.2 = top 20%)
    anterior_focus : If True, prioritize left-most (anterior) portion of top edge
    
    Returns:
    --------
    edge_points : Nx2 array of (x, y) coordinates, or None if insufficient points
    """
    # Extract all mask coordinates
    y_coords, x_coords = np.where(mask > 0)
    if len(y_coords) == 0:
        return None
    
    # Get bounding box
    y_min, y_max = float(y_coords.min()), float(y_coords.max())
    x_min, x_max = float(x_coords.min()), float(x_coords.max())
    
    # ROI restriction: top 20% of bounding box height
    roi_y_max = y_min + roi_top_percent * (y_max - y_min)
    
    # Extract upper boundary: for each x, find minimum y (topmost pixel)
    edge_points = []
    x_range = np.arange(int(x_min), int(x_max) + 1)
    
    for x in x_range:
        # Find all y coordinates at this x
        mask_at_x = mask[:, int(x)]
        y_indices = np.where(mask_at_x > 0)[0]
        
        if len(y_indices) > 0:
            y_top = float(y_indices.min())  # Topmost pixel at this x
            
            # Only include if within ROI (top 20%)
            if y_top <= roi_y_max:
                edge_points.append([float(x), y_top])
    
    if len(edge_points) < 5:
        return None
    
    edge_points = np.array(edge_points, dtype=np.float32)
    
    # Anterior focus: if enabled, prioritize left portion
    if anterior_focus and len(edge_points) > 10:
        # Sort by x coordinate
        edge_points = edge_points[edge_points[:, 0].argsort()]
        # Keep left 60% of points (anterior portion)
        n_keep = int(len(edge_points) * 0.6)
        edge_points = edge_points[:n_keep]
    
    return edge_points


def fit_line_to_superior_endplate(mask: np.ndarray, roi_top_percent: float = 0.2, anterior_focus: bool = True) -> Optional[Tuple[float, float]]:
    """
    Fit a line to the superior endplate using upper boundary points only.
    
    Parameters:
    -----------
    mask : 2D binary mask array
    roi_top_percent : Fraction of bounding box height to consider (default 0.2 = top 20%)
    anterior_focus : If True, prioritize left-most (anterior) portion
    
    Returns:
    --------
    (k, b) : tuple of slope and intercept for line y = k*x + b, or None if insufficient points
    """
    edge_points = extract_superior_endplate_points(mask, roi_top_percent, anterior_focus)
    if edge_points is None or len(edge_points) < 5:
        return None
    
    x_coords = edge_points[:, 0]
    y_coords = edge_points[:, 1]
    
    # Fit Line: y = k*x + b
    k, b = np.polyfit(x_coords, y_coords, 1)
    return (float(k), float(b))


def fit_line_to_edge(mask: np.ndarray, edge_type: str = 'top') -> Optional[Tuple[float, float]]:
    """
    Helper to get strictly the top or bottom edge pixels and fit a line.
    
    Parameters:
    -----------
    mask : 2D binary mask array
    edge_type : 'top' or 'bottom'
    
    Returns:
    --------
    (k, b) : tuple of slope and intercept for line y = k*x + b, or None if insufficient points
    """
    # For superior endplate (top), use the new specialized function
    if edge_type == 'top':
        return fit_line_to_superior_endplate(mask, roi_top_percent=0.2, anterior_focus=True)
    
    # For inferior endplate (bottom), use the original method
    # Extract coordinates: y is row (Z), x is column (Y)
    y, x = np.where(mask > 0)
    if len(y) == 0:
        return None
    
    # Determine Y-threshold to isolate the plate
    y_min, y_max = float(np.min(y)), float(np.max(y))
    height = y_max - y_min
    
    # bottom: Keep points in the bottom 15-20%
    threshold = y_max - 0.2 * height
    indices = np.where(y >= threshold)[0]
    
    if len(indices) == 0:
        return None
    
    edge_y = y[indices]
    edge_x = x[indices]
    
    # Fit Line: y = k*x + b
    if len(edge_x) < 5:
        return None  # Too small
    
    k, b = np.polyfit(edge_x, edge_y, 1)
    return (float(k), float(b))


def extract_inferior_endplate_points(mask: np.ndarray, roi_bottom_percent: float = 0.2) -> Optional[np.ndarray]:
    """
    Extract lower boundary points (inferior endplate) from a vertebral mask.
    
    Algorithm:
    1. For each x-coordinate, find the pixel with maximum y-coordinate (bottom edge)
    2. Restrict to bottom ROI (bottom 20% of bounding box)
    
    Parameters:
    -----------
    mask : 2D binary mask array
    roi_bottom_percent : Fraction of bounding box height to consider (default 0.2 = bottom 20%)
    
    Returns:
    --------
    edge_points : Nx2 array of (x, y) coordinates, or None if insufficient points
    """
    # Extract all mask coordinates
    y_coords, x_coords = np.where(mask > 0)
    if len(y_coords) == 0:
        return None
    
    # Get bounding box
    y_min, y_max = float(y_coords.min()), float(y_coords.max())
    x_min, x_max = float(x_coords.min()), float(x_coords.max())
    
    # ROI restriction: bottom 20% of bounding box height
    roi_y_min = y_max - roi_bottom_percent * (y_max - y_min)
    
    # Extract lower boundary: for each x, find maximum y (bottommost pixel)
    edge_points = []
    x_range = np.arange(int(x_min), int(x_max) + 1)
    
    for x in x_range:
        # Find all y coordinates at this x
        mask_at_x = mask[:, int(x)]
        y_indices = np.where(mask_at_x > 0)[0]
        
        if len(y_indices) > 0:
            y_bottom = float(y_indices.max())  # Bottommost pixel at this x
            
            # Only include if within ROI (bottom 20%)
            if y_bottom >= roi_y_min:
                edge_points.append([float(x), y_bottom])
    
    if len(edge_points) < 5:
        return None
    
    edge_points = np.array(edge_points, dtype=np.float32)
    return edge_points


def fit_line_to_inferior_endplate(
    mask: np.ndarray, 
    roi_bottom_percent: float = 0.2,
    adjacent_disc_mask: Optional[np.ndarray] = None,
    spinal_canal_mask: Optional[np.ndarray] = None,
) -> Optional[Tuple[float, float]]:
    """
    Fit a line to the inferior endplate using disc and canal distance method.
    
    Core logic:
    1. Find candidate points: vertebra points closest to adjacent disc (within 5% tolerance)
    2. Use all candidate points to fit a line, then extend it to full vertebra width
    3. The line covers the full width of the vertebra from x_min to x_max
    
    If disc or canal mask is not provided, falls back to old method using lower boundary points.
    
    Parameters:
    -----------
    mask : 2D binary mask array (should be isolated vertebral body)
    roi_bottom_percent : Fraction of bounding box height to consider (default 0.2 = bottom 20%, used in fallback)
    adjacent_disc_mask : 2D binary mask of adjacent disc (OPTIONAL, if provided uses new method)
    spinal_canal_mask : 2D binary mask of spinal canal (OPTIONAL, if provided uses new method)
    
    Returns:
    --------
    (k, b) : tuple of slope and intercept for line y = k*x + b, or None if insufficient points
    """
    # Try new method first if disc and canal masks are provided
    if adjacent_disc_mask is not None and adjacent_disc_mask.sum() > 0 and \
       spinal_canal_mask is not None and spinal_canal_mask.sum() > 0:
        line_result = extract_endplate_line_by_disc_and_canal(mask, adjacent_disc_mask, spinal_canal_mask, debug_roi=None)
        if line_result is not None:
            # Convert cv2.fitLine format (vx, vy, x0, y0) to (k, b)
            # Handle both NumPy arrays and tuples
            vx = line_result[0].item() if hasattr(line_result[0], 'item') else line_result[0]
            vy = line_result[1].item() if hasattr(line_result[1], 'item') else line_result[1]
            x0 = line_result[2].item() if hasattr(line_result[2], 'item') else line_result[2]
            y0 = line_result[3].item() if hasattr(line_result[3], 'item') else line_result[3]
            if abs(vx) < 1e-9:
                # Vertical line - cannot represent as y = k*x + b
                return None
            k = float(vy / vx)
            b = float(y0 - k * x0)
            return (k, b)
    
    # Fallback to old method using lower boundary points
    # First try using the fit_endplate_line function
    result = fit_endplate_line(mask, mode='inferior')
    if result is not None:
        # Convert cv2.fitLine format to (k, b) if needed
        if len(result) == 4 and result[2] is not None:
            # Handle both NumPy arrays and tuples
            vx = result[0].item() if hasattr(result[0], 'item') else result[0]
            vy = result[1].item() if hasattr(result[1], 'item') else result[1]
            x0 = result[2].item() if hasattr(result[2], 'item') else result[2]
            y0 = result[3].item() if hasattr(result[3], 'item') else result[3]
            if abs(vx) > 1e-9:
                k = float(vy / vx)
                b = float(y0 - k * x0)
                return (k, b)
        elif len(result) >= 2:
            k_val = result[0].item() if hasattr(result[0], 'item') else result[0]
            b_val = result[1].item() if hasattr(result[1], 'item') else result[1]
            k, b = float(k_val), float(b_val)
            return (k, b)
    
    # Fallback to original method
    edge_points = extract_inferior_endplate_points(mask, roi_bottom_percent)
    if edge_points is None or len(edge_points) < 5:
        return None
    
    x_coords = edge_points[:, 0]
    y_coords = edge_points[:, 1]
    
    # Fit Line: y = k*x + b
    k, b = np.polyfit(x_coords, y_coords, 1)
    return (float(k), float(b))


def calculate_lumbosacral_angle(
    l5_mask: np.ndarray,
    s1_mask: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    mask_l5s1_disc: Optional[np.ndarray] = None,
    mask_spinal_canal: Optional[np.ndarray] = None,
) -> Optional[float]:
    """
    Calculate Lumbosacral Angle (LSA) using correct anatomical ROI.
    
    Definition: LSA is the angle between the Inferior Endplate of L5 and the Superior Endplate of S1.
    
    Algorithm:
    1. L5: Isolate vertebral body, extract lower boundary points (max y for each x), fit line
    2. S1: Use specialized get_s1_superior_line to extract frustum top surface (圆台上缘线)
    3. Calculate intersection angle between the two lines
    
    Parameters:
    -----------
    l5_mask : 2D binary mask for L5 vertebra (sagittal slice)
    s1_mask : 2D binary mask for S1 (sacrum, sagittal slice)
    spacing_xyz : (x, y, z) spacing in mm
    mask_l5s1_disc : 2D binary mask for L5-S1 disc (sagittal slice, OPTIONAL but recommended)
    
    Returns:
    --------
    angle_deg : LSA angle in degrees, or None if calculation fails
    """
    if cv2 is None:
        return None
    
    # Step 1: Process L5 - Isolate vertebral body and extract inferior endplate
    l5_body = isolate_vertebral_body(l5_mask)
    if l5_body.sum() == 0:
        return None
    
    # Extract L5 inferior endplate using disc and canal method
    # For L5 inferior, use L5-S1 disc as adjacent disc
    l5_fit = fit_line_to_inferior_endplate(
        l5_body, 
        roi_bottom_percent=0.2,
        adjacent_disc_mask=mask_l5s1_disc,
        spinal_canal_mask=mask_spinal_canal,
    )
    if l5_fit is None:
        return None
    
    # Step 2: Process S1 - Use specialized get_s1_superior_line (same as SS)
    # CRITICAL: Do NOT use fit_line_to_superior_endplate for S1 - use get_s1_superior_line instead
    # Use L5 mask, L5-S1 disc mask, and spinal canal mask for guidance
    line_s1 = get_s1_superior_line(s1_mask, mask_l5=l5_mask, mask_l5s1_disc=mask_l5s1_disc, mask_spinal_canal=mask_spinal_canal)
    if line_s1 is None:
        return None
    
    # Convert line parameters to slope-intercept format (k, b)
    if len(line_s1) == 4 and line_s1[2] is not None:
        # cv2.fitLine format: (vx, vy, x0, y0)
        vx, vy, x0, y0 = line_s1
        if abs(vx) < 1e-9:
            # Vertical line - cannot represent as y = k*x + b
            return None
        k_s1 = float(vy / vx)
        b_s1 = float(y0 - k_s1 * x0)
    else:
        # np.polyfit format: (k, b, None, None)
        k_s1, b_s1 = float(line_s1[0]), float(line_s1[1])
    
    k_l5, b_l5 = l5_fit  # L5: y = k_l5*x + b_l5
    # k_s1, b_s1 already extracted above from get_s1_superior_line
    
    # Step 3: Calculate angle between the two lines
    # Convert to line format for calculate_angle_between_lines
    # L5 line: (k_l5, b_l5, None, None)
    # S1 line: (k_s1, b_s1, None, None)
    line_l5 = (k_l5, b_l5, None, None)
    line_s1 = (k_s1, b_s1, None, None)
    
    angle_deg = calculate_angle_between_lines(line_l5, line_s1)
    
    return angle_deg


def calculate_lumbar_lordosis_angle(
    l1_mask: np.ndarray,
    l5_mask: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    mask_l1_disc: Optional[np.ndarray] = None,  # T12-L1 disc (above L1)
    mask_l5_disc: Optional[np.ndarray] = None,  # L5-S1 disc (below L5)
    mask_spinal_canal: Optional[np.ndarray] = None,
) -> Optional[float]:
    """
    Calculate Lumbar Lordosis (LL) angle using Cobb method.
    Uses the same endplate extraction method as visualization for consistency.
    
    Pre-processing:
    - Isolates vertebral body from posterior elements
    - Assumes anterior (vertebral body) is on the left side of the image
    
    Line Fitting (same as visualization):
    - L1: Uses extract_endplate_line_by_disc_and_canal if disc and canal masks are provided,
          otherwise falls back to fit_endplate_line
    - L5: Uses extract_endplate_line_by_disc_and_canal if disc and canal masks are provided,
          otherwise falls back to fit_endplate_line
    
    Calculation:
    - Computes intersection angle (Cobb angle) between the two lines
    
    Parameters:
    -----------
    l1_mask : 2D binary mask for L1 vertebra (sagittal slice)
    l5_mask : 2D binary mask for L5 vertebra (sagittal slice)
    spacing_xyz : (x, y, z) spacing in mm
    mask_l1_disc : 2D binary mask for T12-L1 disc (optional, for L1 superior endplate)
    mask_l5_disc : 2D binary mask for L5-S1 disc (optional, for L5 inferior endplate)
    mask_spinal_canal : 2D binary mask for spinal canal (optional)
    
    Returns:
    --------
    angle_deg : LL angle in degrees, or None if calculation fails
    """
    if cv2 is None:
        return None
    
    # Step 1: Pre-processing - Isolate vertebral bodies
    l1_body = isolate_vertebral_body(l1_mask)
    l5_body = isolate_vertebral_body(l5_mask)
    
    if l1_body.sum() == 0 or l5_body.sum() == 0:
        return None
    
    # Step 2: Extract L1 superior endplate using same method as visualization
    line_l1 = None
    
    # Try new method first if disc and canal masks are provided
    if mask_l1_disc is not None and mask_l1_disc.sum() > 0 and \
       mask_spinal_canal is not None and mask_spinal_canal.sum() > 0:
        line_l1 = extract_endplate_line_by_disc_and_canal(
            l1_body, mask_l1_disc, mask_spinal_canal, debug_roi=None
        )
    
    # Fallback to old method if new method failed
    if line_l1 is None:
        l1_fit_old = fit_endplate_line(l1_body, mode='superior')
        if l1_fit_old is not None:
            # Convert to cv2 format if needed
            if len(l1_fit_old) == 4 and l1_fit_old[2] is not None:
                line_l1 = l1_fit_old
            else:
                # Convert (k, b) to (vx, vy, x0, y0)
                k, b = l1_fit_old[0], l1_fit_old[1]
                if abs(k) < 1e9:
                    vx, vy = 1.0, float(k)
                    norm = np.sqrt(vx * vx + vy * vy)
                    vx, vy = vx / norm, vy / norm
                    # Use a point on the line
                    l1_ys, l1_xs = np.where(l1_body > 0)
                    if l1_xs.size > 0:
                        x0 = float(l1_xs.mean())
                        y0 = k * x0 + b
                        line_l1 = (vx, vy, x0, y0)
    
    if line_l1 is None:
        return None
    
    # Step 3: Extract L5 inferior endplate using same method as visualization
    line_l5 = None
    
    # Try new method first if disc and canal masks are provided
    if mask_l5_disc is not None and mask_l5_disc.sum() > 0 and \
       mask_spinal_canal is not None and mask_spinal_canal.sum() > 0:
        line_l5 = extract_endplate_line_by_disc_and_canal(
            l5_body, mask_l5_disc, mask_spinal_canal, debug_roi=None
        )
    
    # Fallback to old method if new method failed
    if line_l5 is None:
        l5_fit_old = fit_endplate_line(l5_body, mode='inferior')
        if l5_fit_old is not None:
            # Convert to cv2 format if needed
            if len(l5_fit_old) == 4 and l5_fit_old[2] is not None:
                line_l5 = l5_fit_old
            else:
                # Convert (k, b) to (vx, vy, x0, y0)
                k, b = l5_fit_old[0], l5_fit_old[1]
                if abs(k) < 1e9:
                    vx, vy = 1.0, float(k)
                    norm = np.sqrt(vx * vx + vy * vy)
                    vx, vy = vx / norm, vy / norm
                    # Use a point on the line
                    l5_ys, l5_xs = np.where(l5_body > 0)
                    if l5_xs.size > 0:
                        x0 = float(l5_xs.mean())
                        y0 = k * x0 + b
                        line_l5 = (vx, vy, x0, y0)
    
    if line_l5 is None:
        return None
    
    # Step 4: Convert lines to slope-intercept format for angle calculation
    # L1 line
    if len(line_l1) == 4 and line_l1[2] is not None:
        # cv2.fitLine format: (vx, vy, x0, y0)
        vx1, vy1, x01, y01 = line_l1
        if abs(vx1) < 1e-9:
            return None  # Vertical line, cannot calculate angle
        k1 = float(vy1 / vx1)
        b1 = float(y01 - k1 * x01)
    else:
        # np.polyfit format: (k, b, None, None)
        k1, b1 = float(line_l1[0]), float(line_l1[1])
    
    # L5 line
    if len(line_l5) == 4 and line_l5[2] is not None:
        # cv2.fitLine format: (vx, vy, x0, y0)
        vx5, vy5, x05, y05 = line_l5
        if abs(vx5) < 1e-9:
            return None  # Vertical line, cannot calculate angle
        k2 = float(vy5 / vx5)
        b2 = float(y05 - k2 * x05)
    else:
        # np.polyfit format: (k, b, None, None)
        k2, b2 = float(line_l5[0]), float(line_l5[1])
    
    # Step 5: Calculate angle between the two lines
    # Convert to line format for calculate_angle_between_lines
    line_l1_formatted = (k1, b1, None, None)
    line_l5_formatted = (k2, b2, None, None)
    
    angle_deg = calculate_angle_between_lines(line_l1_formatted, line_l5_formatted)
    
    return angle_deg


def _fit_endplate_line_zy(mask_zy: np.ndarray, which: str) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Fit endplate line on sagittal (Z,Y) mask.
    Return two endpoints (u=Y, v=Z) in pixel coordinates: ((y0,z0),(y1,z1))
    which: "superior" | "inferior"
    """
    if mask_zy.sum() == 0:
        return None
    ys = np.arange(mask_zy.shape[1], dtype=np.int32)
    pts_y = []
    pts_z = []
    for y in ys:
        col = mask_zy[:, y]
        zs = np.where(col > 0)[0]
        if zs.size == 0:
            continue
        if which == "superior":
            z = int(zs.min())
        elif which == "inferior":
            z = int(zs.max())
        else:
            raise ValueError(f"invalid which={which!r}")
        pts_y.append(float(y))
        pts_z.append(float(z))
    if len(pts_y) < 20:
        return None
    yv = np.array(pts_y, dtype=np.float32)
    zv = np.array(pts_z, dtype=np.float32)
    # linear fit z = a*y + b
    a, b = np.polyfit(yv, zv, deg=1)
    y0 = float(np.min(yv))
    y1 = float(np.max(yv))
    z0 = float(a * y0 + b)
    z1 = float(a * y1 + b)
    return ((y0, z0), (y1, z1))


def clip_line_to_mask(line_params: Tuple[float, float], mask: np.ndarray, x_range: Optional[Tuple[float, float]] = None) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    将线段裁剪到mask内部。
    
    参数:
    ------
    line_params : (k, b) 直线参数 y = k*x + b
    mask : 2D binary mask
    x_range : Optional (x_min, x_max) 初始X范围
    
    返回:
    ------
    ((x0, y0), (x1, y1)) : 裁剪后的线段端点，如果线段完全在mask外则返回None
    """
    k, b = line_params
    h, w = mask.shape[:2]
    
    # 获取mask的X范围
    mask_ys, mask_xs = np.where(mask > 0)
    if mask_xs.size == 0:
        return None
    
    mask_x_min, mask_x_max = float(mask_xs.min()), float(mask_xs.max())
    
    # 如果提供了x_range，则使用它和mask范围的交集
    if x_range is not None:
        x_min = max(x_range[0], mask_x_min)
        x_max = min(x_range[1], mask_x_max)
    else:
        x_min, x_max = mask_x_min, mask_x_max
    
    if x_min >= x_max:
        return None
    
    # 在X范围内采样点，找到在mask内的点
    valid_points = []
    
    # 采样X值（使用足够细的步长以确保找到所有有效点）
    num_samples = max(200, int((x_max - x_min) * 2))
    x_samples = np.linspace(x_min, x_max, num_samples)
    
    for x in x_samples:
        y = k * x + b
        y_int = int(round(y))
        x_int = int(round(x))
        
        # 检查点是否在图像范围内且在mask内
        if 0 <= y_int < h and 0 <= x_int < w:
            if mask[y_int, x_int] > 0:
                valid_points.append((float(x), float(y)))
    
    if len(valid_points) == 0:
        return None
    
    # 找到最左和最右的有效点
    valid_points = np.array(valid_points)
    x_coords = valid_points[:, 0]
    
    # 找到最左和最右的点
    leftmost_idx = np.argmin(x_coords)
    rightmost_idx = np.argmax(x_coords)
    
    p0 = (float(valid_points[leftmost_idx][0]), float(valid_points[leftmost_idx][1]))
    p1 = (float(valid_points[rightmost_idx][0]), float(valid_points[rightmost_idx][1]))
    
    return (p0, p1)


def draw_cobb_visualization(
    slice_img: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    line1_params: Tuple[float, float],  # (k1, b1) for y = k1*x + b1
    line2_params: Tuple[float, float],  # (k2, b2) for y = k2*x + b2
    intersection_pt: Optional[Tuple[float, float]],
    angle_deg: Optional[float],
    title: str,
    save_path: str,
    color1: str = "yellow",
    color2: str = "cyan",
    line1_x_range: Optional[Tuple[float, float]] = None,  # Optional X range for line1
    line2_x_range: Optional[Tuple[float, float]] = None,  # Optional X range for line2
) -> None:
    """
    Universal Cobb Angle visualizer with proper extension lines and fixed-radius arc.
    
    Parameters:
    -----------
    slice_img : 2D image array
    mask1, mask2 : 2D masks for the two vertebrae
    line1_params, line2_params : (k, b) for line equations y = k*x + b
    intersection_pt : (x, y) intersection point or None
    angle_deg : calculated angle in degrees
    title : plot title
    save_path : output path
    color1, color2 : colors for the two lines
    line1_x_range : Optional (x_min, x_max) for line1 - if provided, use this instead of mask bounds
    line2_x_range : Optional (x_min, x_max) for line2 - if provided, use this instead of mask bounds
    """
    h, w = slice_img.shape[:2]
    k1, b1 = line1_params
    k2, b2 = line2_params
    
    # Get X range for solid lines
    # If custom X range is provided, use it; otherwise use mask bounds
    if line1_x_range is not None:
        m1_x_min, m1_x_max = line1_x_range
    else:
        m1_ys, m1_xs = np.where(mask1 > 0)
        if m1_xs.size == 0:
            return
        m1_x_min, m1_x_max = float(m1_xs.min()), float(m1_xs.max())
    
    if line2_x_range is not None:
        m2_x_min, m2_x_max = line2_x_range
    else:
        m2_ys, m2_xs = np.where(mask2 > 0)
        if m2_xs.size == 0:
            return
        m2_x_min, m2_x_max = float(m2_xs.min()), float(m2_xs.max())
    
    # Solid line segments: clip to mask boundaries
    # Use clip_line_to_mask to ensure lines are only drawn within mask regions
    m1_clipped = clip_line_to_mask((k1, b1), mask1, line1_x_range)
    if m1_clipped is None:
        # Fallback: use original method if clipping fails
        m1_y0 = k1 * m1_x_min + b1
        m1_y1 = k1 * m1_x_max + b1
        m1_solid_p0 = (m1_x_min, m1_y0)
        m1_solid_p1 = (m1_x_max, m1_y1)
    else:
        m1_solid_p0, m1_solid_p1 = m1_clipped
    
    m2_clipped = clip_line_to_mask((k2, b2), mask2, line2_x_range)
    if m2_clipped is None:
        # Fallback: use original method if clipping fails
        m2_y0 = k2 * m2_x_min + b2
        m2_y1 = k2 * m2_x_max + b2
        m2_solid_p0 = (m2_x_min, m2_y0)
        m2_solid_p1 = (m2_x_max, m2_y1)
    else:
        m2_solid_p0, m2_solid_p1 = m2_clipped
    
    # Calculate centroids for extension direction (use mask bounds for centroid)
    m1_ys, m1_xs = np.where(mask1 > 0)
    m2_ys, m2_xs = np.where(mask2 > 0)
    if m1_xs.size == 0 or m2_xs.size == 0:
        return
    m1_centroid = (float(m1_xs.mean()), float(m1_ys.mean()))
    m2_centroid = (float(m2_xs.mean()), float(m2_ys.mean()))
    
    # Calculate centroids for extension direction
    m1_centroid = (float(m1_xs.mean()), float(m1_ys.mean()))
    m2_centroid = (float(m2_xs.mean()), float(m2_ys.mean()))
    
    overlays = []
    
    # Draw masks
    if mask1.sum() > 0:
        overlays.append(("mask", {"mask": mask1 > 0, "color": color1, "alpha": 0.3}))
    if mask2.sum() > 0:
        overlays.append(("mask", {"mask": mask2 > 0, "color": color2, "alpha": 0.3}))
    
    # Draw solid lines on endplates (thinner)
    overlays.append(("line", {"p0": m1_solid_p0, "p1": m1_solid_p1, "color": color1, "lw": 1.5, "linestyle": "-"}))
    overlays.append(("line", {"p0": m2_solid_p0, "p1": m2_solid_p1, "color": color2, "lw": 1.5, "linestyle": "-"}))
    
    # Add angle value label between the two lines
    if angle_deg is not None:
        # Place text near the midpoint between the two centroids, offset to the right
        label_x = max(m1_centroid[0], m2_centroid[0]) + 10
        label_y = (m1_centroid[1] + m2_centroid[1]) * 0.5
        overlays.append(("text", {
            "xy": (label_x, label_y),
            "text": f"{angle_deg:.1f}°",
            "color": "white",
            "fontsize": 13,
        }))
    
    save_visualization(save_path, slice_img, title, overlays)


def visualize_LL_L1_L5(
    slice_img: np.ndarray,
    l1_mask: np.ndarray,
    l5_mask: np.ndarray,
    angle_deg: Optional[float],
    title: str,
    save_path: str,
    mask_l1_disc: Optional[np.ndarray] = None,  # T12-L1 disc (above L1, label 91)
    mask_l5_disc: Optional[np.ndarray] = None,  # L5-S1 disc (below L5)
    mask_spinal_canal: Optional[np.ndarray] = None,
) -> None:
    """
    Visualize Lumbar Lordosis (LL): Angle between L1 Superior Endplate (Top) and L5 Inferior Endplate (Bottom).
    Uses new method with disc and canal for accurate endplate extraction.
    """
    if cv2 is None:
        return
    
    # Step 1: Isolate vertebral bodies
    l1_body = isolate_vertebral_body(l1_mask)
    l5_body = isolate_vertebral_body(l5_mask)
    
    if l1_body.sum() == 0 or l5_body.sum() == 0:
        return
    
    # Step 2: Extract L1 superior endplate using new method
    debug_roi_l1 = {}
    line_l1 = None
    
    if mask_l1_disc is not None and mask_l1_disc.sum() > 0 and \
       mask_spinal_canal is not None and mask_spinal_canal.sum() > 0:
        line_l1 = extract_endplate_line_by_disc_and_canal(
            l1_body, mask_l1_disc, mask_spinal_canal, debug_roi=debug_roi_l1
        )
    
    # Fallback to old method if new method failed
    if line_l1 is None:
        l1_fit_old = fit_endplate_line(l1_body, mode='superior')
        if l1_fit_old is not None:
            # Convert to cv2 format if needed
            if len(l1_fit_old) == 4 and l1_fit_old[2] is not None:
                line_l1 = l1_fit_old
            else:
                # Convert (k, b) to (vx, vy, x0, y0)
                k, b = l1_fit_old[0], l1_fit_old[1]
                if abs(k) < 1e9:
                    vx, vy = 1.0, float(k)
                    norm = np.sqrt(vx * vx + vy * vy)
                    vx, vy = vx / norm, vy / norm
                    # Use a point on the line
                    l1_ys, l1_xs = np.where(l1_body > 0)
                    if l1_xs.size > 0:
                        x0 = float(l1_xs.mean())
                        y0 = k * x0 + b
                        line_l1 = (vx, vy, x0, y0)
    
    if line_l1 is None:
        return
    
    # Step 3: Extract L5 inferior endplate using new method
    debug_roi_l5 = {}
    line_l5 = None
    
    if mask_l5_disc is not None and mask_l5_disc.sum() > 0 and \
       mask_spinal_canal is not None and mask_spinal_canal.sum() > 0:
        line_l5 = extract_endplate_line_by_disc_and_canal(
            l5_body, mask_l5_disc, mask_spinal_canal, debug_roi=debug_roi_l5
        )
    
    # Fallback to old method if new method failed
    if line_l5 is None:
        l5_fit_old = fit_endplate_line(l5_body, mode='inferior')
        if l5_fit_old is not None:
            # Convert to cv2 format if needed
            if len(l5_fit_old) == 4 and l5_fit_old[2] is not None:
                line_l5 = l5_fit_old
            else:
                # Convert (k, b) to (vx, vy, x0, y0)
                k, b = l5_fit_old[0], l5_fit_old[1]
                if abs(k) < 1e9:
                    vx, vy = 1.0, float(k)
                    norm = np.sqrt(vx * vx + vy * vy)
                    vx, vy = vx / norm, vy / norm
                    # Use a point on the line
                    l5_ys, l5_xs = np.where(l5_body > 0)
                    if l5_xs.size > 0:
                        x0 = float(l5_xs.mean())
                        y0 = k * x0 + b
                        line_l5 = (vx, vy, x0, y0)
    
    if line_l5 is None:
        return
    
    # Step 4: Convert lines to slope-intercept format for visualization
    # L1 line
    if len(line_l1) == 4 and line_l1[2] is not None:
        vx1, vy1, x01, y01 = line_l1
        if abs(vx1) < 1e-9:
            return
        k1 = float(vy1 / vx1)
        b1 = float(y01 - k1 * x01)
    else:
        k1, b1 = float(line_l1[0]), float(line_l1[1])
    
    # L5 line
    if len(line_l5) == 4 and line_l5[2] is not None:
        vx5, vy5, x05, y05 = line_l5
        if abs(vx5) < 1e-9:
            return
        k2 = float(vy5 / vx5)
        b2 = float(y05 - k2 * x05)
    else:
        k2, b2 = float(line_l5[0]), float(line_l5[1])
    
    # Step 5: Get X ranges for lines
    # L1 X range
    l1_ys, l1_xs = np.where(l1_body > 0)
    if l1_xs.size == 0:
        return
    l1_x_min, l1_x_max = float(l1_xs.min()), float(l1_xs.max())
    
    # L5 X range
    l5_ys, l5_xs = np.where(l5_body > 0)
    if l5_xs.size == 0:
        return
    l5_x_min, l5_x_max = float(l5_xs.min()), float(l5_xs.max())
    
    # Step 6: Compute intersection (not used for drawing, but for reference)
    if abs(k1 - k2) < 1e-9:
        intersection = None
    else:
        x_int = (b2 - b1) / (k1 - k2)
        y_int = k1 * x_int + b1
        intersection = (float(x_int), float(y_int))
    
    # Step 7: Use universal Cobb visualizer (already modified to only show two line segments)
    draw_cobb_visualization(
        slice_img, l1_body, l5_body,
        (k1, b1), (k2, b2), intersection, angle_deg,
        title, save_path, color1="yellow", color2="cyan",
        line1_x_range=(l1_x_min, l1_x_max),  # L1 superior endplate X range
        line2_x_range=(l5_x_min, l5_x_max),  # L5 inferior endplate X range
    )


def visualize_LSA(
    slice_img: np.ndarray,
    l5_mask: np.ndarray,
    s1_mask: np.ndarray,
    angle_deg: Optional[float],
    title: str,
    save_path: str,
    mask_l5s1_disc: Optional[np.ndarray] = None,
    mask_spinal_canal: Optional[np.ndarray] = None,
) -> None:
    """
    Visualize Lumbosacral Angle (LSA): Angle between L5 Inferior Endplate and S1 Superior Endplate.
    Uses correct anatomical ROI: L5 body isolation + S1 frustum top surface (圆台上缘线).
    """
    if cv2 is None:
        return
    
    # Step 1: Process L5 - Isolate vertebral body
    l5_body = isolate_vertebral_body(l5_mask)
    if l5_body.sum() == 0:
        return
    
    # Extract L5 inferior endplate line using disc and canal method
    # For L5 inferior, use L5-S1 disc as adjacent disc
    debug_roi_l5 = {}  # Store point1 and point2
    l5_fit = None
    
    # Try new method first if disc and canal masks are provided
    if mask_l5s1_disc is not None and mask_l5s1_disc.sum() > 0 and \
       mask_spinal_canal is not None and mask_spinal_canal.sum() > 0:
        line_result = extract_endplate_line_by_disc_and_canal(l5_body, mask_l5s1_disc, mask_spinal_canal, debug_roi=debug_roi_l5)
        if line_result is not None:
            # Convert cv2.fitLine format (vx, vy, x0, y0) to (k, b)
            vx, vy, x0, y0 = line_result
            if abs(vx) > 1e-9:
                k = float(vy / vx)
                b = float(y0 - k * x0)
                l5_fit = (k, b)
    
    # Fallback to old method if new method failed
    if l5_fit is None:
        l5_fit = fit_line_to_inferior_endplate(l5_body, roi_bottom_percent=0.2)
    
    if l5_fit is None:
        return
    
    # Get L5 inferior endplate actual X range
    # If using new method, get X range from the two points
    if 'point1' in debug_roi_l5 and 'point2' in debug_roi_l5:
        point1 = debug_roi_l5['point1']
        point2 = debug_roi_l5['point2']
        l5_x_min = float(min(point1[0], point2[0]))
        l5_x_max = float(max(point1[0], point2[0]))
    else:
        # Fallback: use L5 body X range
        l5_ys, l5_xs = np.where(l5_body > 0)
        if l5_xs.size == 0:
            return
        l5_x_min, l5_x_max = float(l5_xs.min()), float(l5_xs.max())
    
    # Step 2: Process S1 - Use specialized get_s1_superior_line (same as SS)
    # CRITICAL: Do NOT use fit_line_to_superior_endplate for S1 - use get_s1_superior_line instead
    line_s1 = get_s1_superior_line(s1_mask, mask_l5=l5_mask, mask_l5s1_disc=mask_l5s1_disc, mask_spinal_canal=mask_spinal_canal)
    if line_s1 is None:
        return
    
    # Convert line_s1 to slope-intercept format (k, b)
    if len(line_s1) == 4 and line_s1[2] is not None:
        vx, vy, x0, y0 = line_s1
        if abs(vx) < 1e-9:
            return
        k_s1 = float(vy / vx)
        b_s1 = float(y0 - k_s1 * x0)
    else:
        k_s1, b_s1 = float(line_s1[0]), float(line_s1[1])
    
    # Get S1 superior endplate actual X range - ensure endpoints are within S1 mask
    # CRITICAL: Use interface points X range (from get_s1_superior_line), but clip to S1 mask bounds
    # We need to reconstruct the X range from the line and S1 mask
    
    # Get S1 mask bounds
    s1_ys_all, s1_xs_all = np.where(s1_mask > 0)
    if s1_xs_all.size == 0:
        return
    
    s1_mask_x_min = float(s1_xs_all.min())
    s1_mask_x_max = float(s1_xs_all.max())
    
    # Get initial X range from disc (if available) - this is where interface points should be
    s1_x_min_init, s1_x_max_init = l5_x_min, l5_x_max
    if mask_l5s1_disc is not None and mask_l5s1_disc.sum() > 0:
        disc_ys, disc_xs = np.where(mask_l5s1_disc > 0)
        if disc_xs.size > 0:
            s1_x_min_init = float(disc_xs.min())
            s1_x_max_init = float(disc_xs.max())
    
    # CRITICAL: Find valid X range where line points are within S1 mask
    # For each X in the candidate range, check if the corresponding Y on the line is within S1 mask
    # Use S1 mask as a lookup table for fast checking
    h_img, w_img = slice_img.shape[:2]
    s1_mask_binary = (s1_mask > 0)
    
    valid_x_coords = []
    x_candidate_min = max(s1_mask_x_min, s1_x_min_init - 5)
    x_candidate_max = min(s1_mask_x_max, s1_x_max_init + 5)
    
    for x_test in np.arange(x_candidate_min, x_candidate_max + 1, 0.5):
        y_test = k_s1 * x_test + b_s1
        # Check if this point is within image bounds
        if 0 <= y_test < h_img and 0 <= x_test < w_img:
            # Check if point is within S1 mask (strict check)
            y_int = int(round(y_test))
            x_int = int(round(x_test))
            if 0 <= y_int < h_img and 0 <= x_int < w_img:
                if s1_mask_binary[y_int, x_int]:
                    valid_x_coords.append(x_test)
                else:
                    # Also check nearby points (within 2 pixels) in case of rounding
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            y_check = y_int + dy
                            x_check = x_int + dx
                            if 0 <= y_check < h_img and 0 <= x_check < w_img:
                                if s1_mask_binary[y_check, x_check]:
                                    valid_x_coords.append(x_test)
                                    break
                        else:
                            continue
                        break
    
    if len(valid_x_coords) > 0:
        s1_x_min = float(min(valid_x_coords))
        s1_x_max = float(max(valid_x_coords))
    else:
        # Fallback: use intersection of disc X range and S1 mask X range
        s1_x_min = max(s1_x_min_init, s1_mask_x_min)
        s1_x_max = min(s1_x_max_init, s1_mask_x_max)
        
        # Additional validation: ensure endpoints are reasonable
        if s1_x_min >= s1_x_max:
            # If intersection is empty, use S1 mask range but centered on disc
            center = (s1_x_min_init + s1_x_max_init) / 2.0
            width = min(s1_x_max_init - s1_x_min_init, s1_mask_x_max - s1_mask_x_min) / 2.0
            s1_x_min = max(s1_mask_x_min, center - width)
            s1_x_max = min(s1_mask_x_max, center + width)
        
        # Final validation: ensure endpoints are within S1 mask
        y_min_check = k_s1 * s1_x_min + b_s1
        y_max_check = k_s1 * s1_x_max + b_s1
        y_min_int = int(round(y_min_check))
        y_max_int = int(round(y_max_check))
        x_min_int = int(round(s1_x_min))
        x_max_int = int(round(s1_x_max))
        
        # If endpoints are not in S1, clip to S1 bounds
        if not (0 <= y_min_int < h_img and 0 <= x_min_int < w_img and s1_mask_binary[y_min_int, x_min_int]):
            # Find the leftmost X where line is in S1
            for x_search in np.arange(s1_mask_x_min, s1_x_max_init + 1, 0.5):
                y_search = k_s1 * x_search + b_s1
                y_search_int = int(round(y_search))
                x_search_int = int(round(x_search))
                if 0 <= y_search_int < h_img and 0 <= x_search_int < w_img:
                    if s1_mask_binary[y_search_int, x_search_int]:
                        s1_x_min = float(x_search)
                        break
        
        if not (0 <= y_max_int < h_img and 0 <= x_max_int < w_img and s1_mask_binary[y_max_int, x_max_int]):
            # Find the rightmost X where line is in S1
            for x_search in np.arange(s1_x_max_init, s1_mask_x_min - 1, -0.5):
                y_search = k_s1 * x_search + b_s1
                y_search_int = int(round(y_search))
                x_search_int = int(round(x_search))
                if 0 <= y_search_int < h_img and 0 <= x_search_int < w_img:
                    if s1_mask_binary[y_search_int, x_search_int]:
                        s1_x_max = float(x_search)
                        break
    
    k_l5, b_l5 = l5_fit
    
    # Compute intersection
    if abs(k_l5 - k_s1) < 1e-9:
        intersection = None
    else:
        x_int = (b_s1 - b_l5) / (k_l5 - k_s1)
        y_int = k_l5 * x_int + b_l5
        intersection = (float(x_int), float(y_int))
    
    # Use universal Cobb visualizer with isolated L5 body and full S1 mask
    # Pass actual X ranges for accurate line drawing
    draw_cobb_visualization(
        slice_img, l5_body, s1_mask,
        (k_l5, b_l5), (k_s1, b_s1), intersection, angle_deg,
        title, save_path, color1="yellow", color2="cyan",
        line1_x_range=(l5_x_min, l5_x_max),  # L5 inferior endplate X range
        line2_x_range=(s1_x_min, s1_x_max),  # S1 superior endplate X range
    )


def visualize_SS_S1(
    slice_img: np.ndarray,
    s1_mask: np.ndarray,
    l5_mask: Optional[np.ndarray],
    angle_deg: Optional[float],
    title: str,
    save_path: str,
    mask_l5s1_disc: Optional[np.ndarray] = None,
    mask_spinal_canal: Optional[np.ndarray] = None,
) -> None:
    """
    Visualize Sacral Slope (SS): Angle between S1 Superior Endplate (Top) and Horizontal Reference Line.
    Horizontal line starts from Posterior-Superior Corner of S1 and extends backwards.
    Uses specialized Frustum Top Surface method to extract S1 superior endplate.
    """
    h, w = slice_img.shape[:2]
    
    # Step 1: Extract S1 Superior Endplate Line using specialized method
    # CRITICAL: Use get_s1_superior_line with spinal canal mask
    debug_roi_s1 = {}  # Store point1 and point2
    line_s1 = get_s1_superior_line(s1_mask, mask_l5=l5_mask, mask_l5s1_disc=mask_l5s1_disc, mask_spinal_canal=mask_spinal_canal, debug_roi=debug_roi_s1) if cv2 is not None else None
    if line_s1 is None:
        return
    
    # Get point1 (farthest from canal, closest to disc) - this is the anchor point for horizontal line
    if 'point1' not in debug_roi_s1:
        return
    
    point1 = debug_roi_s1['point1']  # (x, y)
    anchor_x = float(point1[0])
    anchor_y = float(point1[1])
    
    # Convert line_s1 to slope-intercept format (k, b)
    if len(line_s1) == 4 and line_s1[2] is not None:
        # cv2.fitLine format: (vx, vy, x0, y0)
        vx, vy, x0, y0 = line_s1
        if abs(vx) < 1e-9:
            return
        k = float(vy / vx)
        b = float(y0 - k * x0)
    else:
        # np.polyfit format: (k, b, None, None)
        k, b = float(line_s1[0]), float(line_s1[1])
    
    # Step 3: Get S1 superior endplate X range - ensure endpoints are within S1 mask
    # CRITICAL: Find X range where line points are within S1 mask
    s1_ys, s1_xs = np.where(s1_mask > 0)  # ys are rows (Z), xs are columns (Y)
    if s1_xs.size == 0:
        return
    s1_mask_x_min = float(s1_xs.min())
    s1_mask_x_max = float(s1_xs.max())
    
    # Get initial X range from disc (if available)
    s1_x_min_init, s1_x_max_init = s1_mask_x_min, s1_mask_x_max
    if mask_l5s1_disc is not None and mask_l5s1_disc.sum() > 0:
        disc_ys, disc_xs = np.where(mask_l5s1_disc > 0)
        if disc_xs.size > 0:
            s1_x_min_init = float(disc_xs.min())
            s1_x_max_init = float(disc_xs.max())
    
    # Find valid X range where line points are within S1 mask (strict check)
    s1_mask_binary = (s1_mask > 0)
    valid_x_coords = []
    x_candidate_min = max(s1_mask_x_min, s1_x_min_init - 5)
    x_candidate_max = min(s1_mask_x_max, s1_x_max_init + 5)
    
    for x_test in np.arange(x_candidate_min, x_candidate_max + 1, 0.5):
        y_test = k * x_test + b
        # Check if point is within image bounds
        if 0 <= y_test < h and 0 <= x_test < w:
            # Check if point is within S1 mask (strict check)
            y_int = int(round(y_test))
            x_int = int(round(x_test))
            if 0 <= y_int < h and 0 <= x_int < w:
                if s1_mask_binary[y_int, x_int]:
                    valid_x_coords.append(x_test)
                else:
                    # Also check nearby points (within 2 pixels) in case of rounding
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            y_check = y_int + dy
                            x_check = x_int + dx
                            if 0 <= y_check < h and 0 <= x_check < w:
                                if s1_mask_binary[y_check, x_check]:
                                    valid_x_coords.append(x_test)
                                    break
                        else:
                            continue
                        break
    
    if len(valid_x_coords) > 0:
        s1_x_min_all = float(min(valid_x_coords))
        s1_x_max_all = float(max(valid_x_coords))
    else:
        # Fallback: use intersection of disc X range and S1 mask X range
        s1_x_min_all = max(s1_x_min_init, s1_mask_x_min)
        s1_x_max_all = min(s1_x_max_init, s1_mask_x_max)
        
        # Additional validation
        if s1_x_min_all >= s1_x_max_all:
            center = (s1_x_min_init + s1_x_max_init) / 2.0
            width = min(s1_x_max_init - s1_x_min_init, s1_mask_x_max - s1_mask_x_min) / 2.0
            s1_x_min_all = max(s1_mask_x_min, center - width)
            s1_x_max_all = min(s1_mask_x_max, center + width)
        
        # Final validation: ensure endpoints are within S1 mask
        y_min_check = k * s1_x_min_all + b
        y_max_check = k * s1_x_max_all + b
        y_min_int = int(round(y_min_check))
        y_max_int = int(round(y_max_check))
        x_min_int = int(round(s1_x_min_all))
        x_max_int = int(round(s1_x_max_all))
        
        # If endpoints are not in S1, clip to S1 bounds
        if not (0 <= y_min_int < h and 0 <= x_min_int < w and s1_mask_binary[y_min_int, x_min_int]):
            # Find the leftmost X where line is in S1
            for x_search in np.arange(s1_mask_x_min, s1_x_max_init + 1, 0.5):
                y_search = k * x_search + b
                y_search_int = int(round(y_search))
                x_search_int = int(round(x_search))
                if 0 <= y_search_int < h and 0 <= x_search_int < w:
                    if s1_mask_binary[y_search_int, x_search_int]:
                        s1_x_min_all = float(x_search)
                        break
        
        if not (0 <= y_max_int < h and 0 <= x_max_int < w and s1_mask_binary[y_max_int, x_max_int]):
            # Find the rightmost X where line is in S1
            for x_search in np.arange(s1_x_max_init, s1_mask_x_min - 1, -0.5):
                y_search = k * x_search + b
                y_search_int = int(round(y_search))
                x_search_int = int(round(x_search))
                if 0 <= y_search_int < h and 0 <= x_search_int < w:
                    if s1_mask_binary[y_search_int, x_search_int]:
                        s1_x_max_all = float(x_search)
                        break
    
    # Solid line segment: within S1 superior endplate X range (ensured to be within S1 mask)
    s1_y0 = k * s1_x_min_all + b
    s1_y1 = k * s1_x_max_all + b
    s1_solid_p0 = (s1_x_min_all, s1_y0)
    s1_solid_p1 = (s1_x_max_all, s1_y1)
    
    # Step 4: Draw horizontal line passing through the rightmost endpoint of S1 line
    # The rightmost endpoint is s1_solid_p1 (larger x coordinate)
    rightmost_point = s1_solid_p1  # (x, y) - this is the rightmost point
    horizontal_y = rightmost_point[1]  # y coordinate of rightmost point
    
    # Step 5: Prepare overlays
    overlays = []
    
    # Draw S1 mask
    if s1_mask.sum() > 0:
        overlays.append(("mask", {"mask": s1_mask > 0, "color": "cyan", "alpha": 0.3}))
    
    # Draw solid S1 endplate line (thinner)
    overlays.append(("line", {"p0": s1_solid_p0, "p1": s1_solid_p1, "color": "cyan", "lw": 1.5, "linestyle": "-"}))
    
    # Draw horizontal reference line (dashed, thinner) - passing through rightmost point
    horizontal_p0 = (max(0.0, rightmost_point[0] - w * 0.3), horizontal_y)  # Extend backwards (left)
    horizontal_p1 = (float(w), horizontal_y)  # Extend forwards (right)
    overlays.append(("line", {"p0": horizontal_p0, "p1": horizontal_p1, "color": "yellow", "lw": 1.5, "linestyle": "--"}))
    
    # Draw angle arc (acute angle only, thinner) - centered at rightmost point
    if angle_deg is not None:
        # Calculate S1 line direction vector
        dx = s1_x_max_all - s1_x_min_all
        dy = s1_y1 - s1_y0
        
        # Calculate angle of S1 line relative to horizontal (in degrees)
        # atan2 returns angle in radians, convert to degrees
        # Note: atan2(dy, dx) gives angle from positive x-axis (0 degrees = right, 90 degrees = up)
        angle_s1_rad = np.arctan2(dy, dx)
        angle_s1 = np.degrees(angle_s1_rad)
        
        # Calculate the acute angle between S1 line and horizontal (0-90 degrees)
        # The acute angle is always between 0 and 90 degrees
        angle_diff = abs(angle_s1)
        if angle_diff > 90.0:
            acute_angle = 180.0 - angle_diff
        else:
            acute_angle = angle_diff
        
        # Determine arc start and end angles
        # Always draw from horizontal (0 degrees) to show the acute angle
        if angle_s1 >= 0:
            if angle_s1 <= 90.0:
                # S1 line is between 0 and 90 degrees, draw from 0 to angle_s1
                angle_start = 0.0
                angle_end = angle_s1
            else:
                # S1 line is > 90 degrees, draw from (180 - acute_angle) to 180
                angle_start = 180.0 - acute_angle
                angle_end = 180.0
        else:
            if angle_s1 >= -90.0:
                # S1 line is between -90 and 0 degrees, draw from angle_s1 to 0
                angle_start = angle_s1
                angle_end = 0.0
            else:
                # S1 line is < -90 degrees, draw from -180 to (-180 + acute_angle)
                angle_start = -180.0
                angle_end = -180.0 + acute_angle
        
        # Ensure arc span is exactly the acute angle (0-90 degrees)
        arc_span = abs(angle_end - angle_start)
        if arc_span > 90.0:
            # If span is too large, adjust to show only acute angle
            if angle_s1 >= 0:
                angle_start = 0.0
                angle_end = acute_angle
            else:
                angle_start = -acute_angle
                angle_end = 0.0
        
        radius = 30.0  # Fixed radius in pixels
        
        # Use rightmost point as arc center
        arc_center = rightmost_point
        
        overlays.append(("arc", {
            "center": arc_center,
            "angle_start": angle_start,
            "angle_end": angle_end,
            "radius": radius,
            "color": "red",
            "lw": 1.5,
        }))
        
        # Add angle value text, offset from the arc
        overlays.append(("text", {
            "xy": (arc_center[0] + radius + 10, arc_center[1]),
            "text": f"{angle_deg:.1f}°",
            "color": "white",
            "fontsize": 13,
        }))
    
    save_visualization(save_path, slice_img, title, overlays)


def visualize_DIA(
    slice_img: np.ndarray,
    disc_mask: np.ndarray,
    angle_deg: float,
    save_path: str,
    name: str = "",
    line_upper: Optional[Tuple[float, float, float, float]] = None,
    line_lower: Optional[Tuple[float, float, float, float]] = None,
    upper_mask: Optional[np.ndarray] = None,
    lower_mask: Optional[np.ndarray] = None,
) -> None:
    """
    Visualize Disc Inclination Angle (DIA): Angle between upper vertebra inferior endplate and lower vertebra superior endplate.
    
    Parameters:
    -----------
    slice_img : 2D image array (Z, Y) in sagittal view
    disc_mask : 2D mask for the disc (will be shown in green)
    angle_deg : calculated angle in degrees
    save_path : output path
    name : disc name (e.g., "L4-L5")
    line_upper : Upper endplate line parameters (vx, vy, x0, y0)
    line_lower : Lower endplate line parameters (vx, vy, x0, y0)
    upper_mask : Upper vertebra mask (will be shown in red)
    lower_mask : Lower vertebra mask (will be shown in red)
    """
    if line_upper is None or line_lower is None:
        return
    
    h, w = slice_img.shape[:2]
    
    # Get disc region bounds to limit line drawing
    disc_ys, disc_xs = np.where(disc_mask > 0)
    if len(disc_xs) > 0:
        disc_x_min = float(disc_xs.min())
        disc_x_max = float(disc_xs.max())
        # Extend a bit beyond disc bounds for better visualization
        margin = max((disc_x_max - disc_x_min) * 0.3, 20.0)
        line_x_min = max(0, disc_x_min - margin)
        line_x_max = min(w - 1, disc_x_max + margin)
    else:
        line_x_min = 0
        line_x_max = w - 1
    
    # Prepare overlays
    overlays = []
    
    # Draw upper vertebra mask in red (exclude disc region to avoid overlap)
    if upper_mask is not None and upper_mask.sum() > 0:
        upper_mask_excluding_disc = (upper_mask > 0) & (disc_mask == 0)
        if upper_mask_excluding_disc.sum() > 0:
            overlays.append(("mask", {"mask": upper_mask_excluding_disc, "color": "red", "alpha": 0.3}))
    
    # Draw lower vertebra mask in red (exclude disc region to avoid overlap)
    if lower_mask is not None and lower_mask.sum() > 0:
        lower_mask_excluding_disc = (lower_mask > 0) & (disc_mask == 0)
        if lower_mask_excluding_disc.sum() > 0:
            overlays.append(("mask", {"mask": lower_mask_excluding_disc, "color": "red", "alpha": 0.3}))
    
    # Draw disc mask in green (draw last to ensure it's visible)
    if disc_mask.sum() > 0:
        overlays.append(("mask", {"mask": disc_mask > 0, "color": "green", "alpha": 0.4}))
    
    # Draw upper endplate line (thin, cyan) - clipped to upper vertebra mask
    vx_upper, vy_upper, x0_upper, y0_upper = line_upper
    if abs(vx_upper) > 1e-9:
        # Convert (vx, vy, x0, y0) to (k, b) format: y = k*x + b
        k_upper = vy_upper / vx_upper
        b_upper = y0_upper - k_upper * x0_upper
        
        # Clip line to upper vertebra mask (use isolated body if available)
        clipped_upper = None
        if upper_mask is not None and upper_mask.sum() > 0:
            # Use isolated vertebral body for clipping (same as used for line extraction)
            upper_body = isolate_vertebral_body(upper_mask)
            if upper_body.sum() > 0:
                clipped_upper = clip_line_to_mask((k_upper, b_upper), upper_body)
            else:
                # Fallback to original mask if isolation fails
                clipped_upper = clip_line_to_mask((k_upper, b_upper), upper_mask)
        
        if clipped_upper is not None:
            p0_upper, p1_upper = clipped_upper
            overlays.append(("line", {"p0": p0_upper, "p1": p1_upper, "color": "cyan", "lw": 1, "linestyle": "-"}))
        else:
            # Fallback: use disc region bounds
            t1 = (line_x_min - x0_upper) / vx_upper
            t2 = (line_x_max - x0_upper) / vx_upper
            y1_upper = y0_upper + t1 * vy_upper
            y2_upper = y0_upper + t2 * vy_upper
            # Clamp y values to image bounds
            y1_upper = max(0, min(h - 1, y1_upper))
            y2_upper = max(0, min(h - 1, y2_upper))
            overlays.append(("line", {"p0": (line_x_min, y1_upper), "p1": (line_x_max, y2_upper), "color": "cyan", "lw": 1, "linestyle": "-"}))
    else:
        # Vertical line - clip to upper vertebra mask
        if upper_mask is not None and upper_mask.sum() > 0:
            # For vertical line, find y range within mask at x = x0_upper
            x_int = int(round(x0_upper))
            if 0 <= x_int < w:
                mask_ys, mask_xs = np.where(upper_mask > 0)
                # Find points where x is close to x0_upper
                mask_points = np.column_stack([mask_xs, mask_ys])
                close_points = mask_points[np.abs(mask_points[:, 0] - x0_upper) < 1]
                if close_points.size > 0:
                    y_min = float(close_points[:, 1].min())
                    y_max = float(close_points[:, 1].max())
                    overlays.append(("line", {"p0": (x0_upper, y_min), "p1": (x0_upper, y_max), "color": "cyan", "lw": 1, "linestyle": "-"}))
                else:
                    # Fallback for vertical line
                    if 0 <= x0_upper < w:
                        overlays.append(("line", {"p0": (x0_upper, 0), "p1": (x0_upper, h - 1), "color": "cyan", "lw": 1, "linestyle": "-"}))
            else:
                # Fallback for vertical line
                if 0 <= x0_upper < w:
                    overlays.append(("line", {"p0": (x0_upper, 0), "p1": (x0_upper, h - 1), "color": "cyan", "lw": 1, "linestyle": "-"}))
        else:
            # Fallback for vertical line
            if 0 <= x0_upper < w:
                overlays.append(("line", {"p0": (x0_upper, 0), "p1": (x0_upper, h - 1), "color": "cyan", "lw": 1, "linestyle": "-"}))
    
    # Draw lower endplate line (thin, magenta) - clipped to lower vertebra mask's x-range
    vx_lower, vy_lower, x0_lower, y0_lower = line_lower
    if abs(vx_lower) > 1e-9:
        # Convert (vx, vy, x0, y0) to (k, b) format: y = k*x + b
        k_lower = vy_lower / vx_lower
        b_lower = y0_lower - k_lower * x0_lower
        
        # Clip line to lower vertebra mask x-range directly (no isolate_vertebral_body for S)
        clipped_lower = None
        if lower_mask is not None and lower_mask.sum() > 0:
            clipped_lower = clip_line_to_mask((k_lower, b_lower), lower_mask)
        
        if clipped_lower is not None:
            p0_lower, p1_lower = clipped_lower
            overlays.append(("line", {"p0": p0_lower, "p1": p1_lower, "color": "magenta", "lw": 1, "linestyle": "-"}))
        else:
            # Fallback: use disc region bounds
            t1 = (line_x_min - x0_lower) / vx_lower
            t2 = (line_x_max - x0_lower) / vx_lower
            y1_lower = y0_lower + t1 * vy_lower
            y2_lower = y0_lower + t2 * vy_lower
            # Clamp y values to image bounds
            y1_lower = max(0, min(h - 1, y1_lower))
            y2_lower = max(0, min(h - 1, y2_lower))
            overlays.append(("line", {"p0": (line_x_min, y1_lower), "p1": (line_x_max, y2_lower), "color": "magenta", "lw": 1, "linestyle": "-"}))
    else:
        # Vertical line - clip to lower vertebra mask
        if lower_mask is not None and lower_mask.sum() > 0:
            # For vertical line, find y range within mask at x = x0_lower
            x_int = int(round(x0_lower))
            if 0 <= x_int < w:
                mask_ys, mask_xs = np.where(lower_mask > 0)
                # Find points where x is close to x0_lower
                mask_points = np.column_stack([mask_xs, mask_ys])
                close_points = mask_points[np.abs(mask_points[:, 0] - x0_lower) < 1]
                if close_points.size > 0:
                    y_min = float(close_points[:, 1].min())
                    y_max = float(close_points[:, 1].max())
                    overlays.append(("line", {"p0": (x0_lower, y_min), "p1": (x0_lower, y_max), "color": "magenta", "lw": 1, "linestyle": "-"}))
                else:
                    # Fallback for vertical line
                    if 0 <= x0_lower < w:
                        overlays.append(("line", {"p0": (x0_lower, 0), "p1": (x0_lower, h - 1), "color": "magenta", "lw": 1, "linestyle": "-"}))
            else:
                # Fallback for vertical line
                if 0 <= x0_lower < w:
                    overlays.append(("line", {"p0": (x0_lower, 0), "p1": (x0_lower, h - 1), "color": "magenta", "lw": 1, "linestyle": "-"}))
        else:
            # Fallback for vertical line
            if 0 <= x0_lower < w:
                overlays.append(("line", {"p0": (x0_lower, 0), "p1": (x0_lower, h - 1), "color": "magenta", "lw": 1, "linestyle": "-"}))
    
    # Add angle value label near the disc region
    disc_ys, disc_xs = np.where(disc_mask > 0)
    if disc_xs.size > 0:
        label_x = float(disc_xs.max()) + 10
        label_y = float(disc_ys.mean())
        overlays.append(("text", {
            "xy": (label_x, label_y),
            "text": f"{angle_deg:.1f}°",
            "color": "white",
            "fontsize": 13,
        }))
    
    save_visualization(save_path, slice_img, f"Disc Inclination Angle {name}", overlays)


def calc_cobb_angles(
    mri_zyx: np.ndarray,
    step2_zyx: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    mid_sag_x: int,
    label_L1: int,
    label_L5: int,
    label_S1: int,
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculate:
      - LL: Cobb angle between L1 superior endplate and S1 superior endplate
      - SS: Angle between S1 superior endplate and horizontal line
      - LSA: Angle between L5 inferior endplate and S1 superior endplate
    """
    _sz, sy, _sx = _spacing_zyx(spacing_xyz)
    sz = spacing_xyz[2]  # for readability
    # 2D plane: u=Y, v=Z; spacing_u=sy, spacing_v=sz
    spacing_u = float(spacing_xyz[1])
    spacing_v = float(spacing_xyz[2])

    img_zy = _normalize_intensity_percentile(mri_zyx[:, :, mid_sag_x])

    def _angle_for(label: int, which: str) -> Tuple[Optional[float], Optional[Tuple[Tuple[float, float], Tuple[float, float]]]]:
        m = (step2_zyx == label).astype(np.uint8)[:, :, mid_sag_x]
        line = _fit_endplate_line_zy(m, which)
        if line is None:
            return (None, None)
        # Validate line format: should be ((y0, z0), (y1, z1))
        try:
            if not isinstance(line, (tuple, list)) or len(line) < 2:
                return (None, None)
            pt0, pt1 = line[0], line[1]
            if not isinstance(pt0, (tuple, list)) or len(pt0) < 2:
                return (None, None)
            if not isinstance(pt1, (tuple, list)) or len(pt1) < 2:
                return (None, None)
            (y0, z0), (y1, z1) = line
            theta = _line_angle_deg_from_points((y0, z0), (y1, z1), spacing_u=spacing_u, spacing_v=spacing_v)
            return (theta, line)
        except (ValueError, TypeError, IndexError) as e:
            return (None, None)

    # CRITICAL: Use specialized methods for S1, not generic vertebral body methods
    # L1 and L5: Use vertebral body isolation and endplate fitting
    theta_L1_sup, line_L1_sup = _angle_for(label_L1, "superior")
    theta_L5_inf, line_L5_inf = _angle_for(label_L5, "inferior")
    
    # S1: Use specialized get_s1_superior_line (Frustum Top Surface method)
    mask_S1 = (step2_zyx == label_S1).astype(np.uint8)[:, :, mid_sag_x]
    mask_L5_ref = (step2_zyx == label_L5).astype(np.uint8)[:, :, mid_sag_x] # Needed for S1 guidance
    # Get L5-S disc mask (label 100) for better frustum top surface extraction
    label_L5S1_disc = int(LABEL_MAP["discs"]["L5-S"])
    mask_L5S1_disc = (step2_zyx == label_L5S1_disc).astype(np.uint8)[:, :, mid_sag_x]
    # Get spinal canal mask (label 2) for S1 superior endplate extraction
    mask_spinal_canal = (step2_zyx == LABEL_MAP["spinal_canal"]).astype(np.uint8)[:, :, mid_sag_x]
    debug_roi_s1 = {}  # Store point1 and point2
    line_s1 = get_s1_superior_line(mask_S1, mask_l5=mask_L5_ref, mask_l5s1_disc=mask_L5S1_disc, mask_spinal_canal=mask_spinal_canal, debug_roi=debug_roi_s1) if cv2 is not None else None
    theta_S1_sup = None
    line_S1_sup = None
    if line_s1 is not None:
        # Convert line_s1 to angle
        # line_s1 format: (vx, vy, x0, y0) or (k, b, None, None)
        if len(line_s1) == 4 and line_s1[2] is not None:
            vx, vy, x0, y0 = line_s1
            if abs(vx) > 1e-9:
                # Calculate angle from direction vector
                theta_S1_sup = float(np.degrees(np.arctan2(vy, vx)))
                # Create two points on the line for visualization
                h, w = mask_S1.shape
                p1 = (max(0, min(w-1, int(x0 - vx * w))), max(0, min(h-1, int(y0 - vy * h))))
                p2 = (max(0, min(w-1, int(x0 + vx * w))), max(0, min(h-1, int(y0 + vy * h))))
                line_S1_sup = (p1, p2)
        else:
            # np.polyfit format: (k, b, None, None)
            k, b = line_s1[0], line_s1[1]
            theta_S1_sup = float(np.degrees(np.arctan(k)))
            # Create two points on the line for visualization
            h, w = mask_S1.shape
            x1, x2 = 0, w-1
            y1, y2 = k * x1 + b, k * x2 + b
            y1 = max(0, min(h-1, int(y1)))
            y2 = max(0, min(h-1, int(y2)))
            line_S1_sup = ((x1, y1), (x2, y2))

    LL = None
    SS = None
    LSA = None
    
    # Calculate LL: L1 Superior vs L5 Inferior (NOT S1)
    # Use line intersection angle, not angle difference
    if line_L1_sup is not None and line_L5_inf is not None:
        # Convert point pairs to line format (k, b, None, None)
        # line format: ((y0, z0), (y1, z1))
        (y0_l1, z0_l1), (y1_l1, z1_l1) = line_L1_sup
        (y0_l5, z0_l5), (y1_l5, z1_l5) = line_L5_inf
        
        # Calculate slope and intercept for L1 line: z = k*y + b
        if abs(y1_l1 - y0_l1) > 1e-9:
            k_l1 = (z1_l1 - z0_l1) / (y1_l1 - y0_l1)
            b_l1 = z0_l1 - k_l1 * y0_l1
            line_l1 = (k_l1, b_l1, None, None)
        else:
            # Vertical line - use calculate_angle_between_lines with special handling
            # For vertical line, we'll use a very large slope
            line_l1 = (float('inf'), y0_l1, None, None)
        
        # Calculate slope and intercept for L5 line: z = k*y + b
        if abs(y1_l5 - y0_l5) > 1e-9:
            k_l5 = (z1_l5 - z0_l5) / (y1_l5 - y0_l5)
            b_l5 = z0_l5 - k_l5 * y0_l5
            line_l5 = (k_l5, b_l5, None, None)
        else:
            # Vertical line
            line_l5 = (float('inf'), y0_l5, None, None)
        
        # Calculate acute angle between the two lines
        LL = calculate_angle_between_lines(line_l1, line_l5)
    
    # Calculate SS: Angle between S1 Superior Endplate and Horizontal line
    # SS is the ACUTE angle between the S1 line and a horizontal line
    if line_s1 is not None:
        # Create a horizontal line: (k=0, b=y, None, None) where y is any point on S1 line
        # For simplicity, use y=0 for horizontal line
        horizontal_line = (0.0, 0.0, None, None)
        
        # Calculate angle between S1 line and horizontal line
        SS = calculate_angle_between_lines(line_s1, horizontal_line)
    
    # Calculate LSA: L5 Inferior vs S1 Superior (using specialized S1 method)
    mask_L5_for_lsa = (step2_zyx == label_L5).astype(np.uint8)[:, :, mid_sag_x]
    mask_S1_for_lsa = (step2_zyx == label_S1).astype(np.uint8)[:, :, mid_sag_x]
    mask_L5S1_disc_for_lsa = (step2_zyx == label_L5S1_disc).astype(np.uint8)[:, :, mid_sag_x]
    if cv2 is not None:
        mask_spinal_canal_for_lsa = (step2_zyx == LABEL_MAP["spinal_canal"]).astype(np.uint8)[:, :, mid_sag_x]
        lsa_new = calculate_lumbosacral_angle(mask_L5_for_lsa, mask_S1_for_lsa, spacing_xyz, mask_l5s1_disc=mask_L5S1_disc_for_lsa, mask_spinal_canal=mask_spinal_canal_for_lsa)
        if lsa_new is not None:
            LSA = lsa_new  # Use the specialized calculation
    elif theta_L5_inf is not None and theta_S1_sup is not None:
        # Fallback to old method
        LSA = float(abs(theta_L5_inf - theta_S1_sup))

    out = {
        "lumbar_lordosis_LL_deg": _safe_float(LL),
        "sacral_slope_SS_deg": _safe_float(SS),
        "lumbosacral_angle_LSA_deg": _safe_float(LSA),
        "units": "deg",
        "status": "ok",
        "endplate_line_angles_deg": {"L1_sup": _safe_float(theta_L1_sup), "S1_sup": _safe_float(theta_S1_sup), "L5_inf": _safe_float(theta_L5_inf)},
    }

    if save_dir is not None:
        _ensure_dir(save_dir)
        
        # Get masks for visualization
        mask_L1 = (step2_zyx == label_L1).astype(np.uint8)[:, :, mid_sag_x]
        mask_S1 = (step2_zyx == label_S1).astype(np.uint8)[:, :, mid_sag_x]
        mask_L5 = (step2_zyx == label_L5).astype(np.uint8)[:, :, mid_sag_x]
        
        # Get disc masks for LL calculation (same as visualization)
        # L1 superior endplate uses T12-L1 disc (label 91, above L1)
        mask_T12L1_disc = (step2_zyx == LABEL_MAP["discs"]["T12-L1"]).astype(np.uint8)[:, :, mid_sag_x] if "T12-L1" in LABEL_MAP.get("discs", {}) else None
        mask_L5S1_disc = (step2_zyx == LABEL_MAP["discs"]["L5-S"]).astype(np.uint8)[:, :, mid_sag_x] if "L5-S" in LABEL_MAP.get("discs", {}) else None
        mask_spinal_canal = (step2_zyx == LABEL_MAP["spinal_canal"]).astype(np.uint8)[:, :, mid_sag_x] if "spinal_canal" in LABEL_MAP else None
        
        # Calculate LL using new method (with vertebral body isolation and same method as visualization)
        if cv2 is not None:
            ll_new = calculate_lumbar_lordosis_angle(
                mask_L1, 
                mask_L5, 
                spacing_xyz,
                mask_l1_disc=mask_T12L1_disc,  # T12-L1 disc for L1 superior endplate
                mask_l5_disc=mask_L5S1_disc,    # L5-S1 disc for L5 inferior endplate
                mask_spinal_canal=mask_spinal_canal,
            )
            if ll_new is not None:
                LL = ll_new  # Use the new calculation if available
                out["lumbar_lordosis_LL_deg"] = _safe_float(LL)
        
        # Visualize Lumbar Lordosis (LL): L1 Top and L5 Bottom
        if LL is not None:
            
            visualize_LL_L1_L5(
                slice_img=img_zy,
                l1_mask=mask_L1,
                l5_mask=mask_L5,
                angle_deg=LL,
                title="Lumbar Lordosis (LL)",
                save_path=os.path.join(save_dir, "global_cobb_ll.png"),
                mask_l1_disc=mask_T12L1_disc,  # T12-L1 disc for L1 superior endplate
                mask_l5_disc=mask_L5S1_disc,
                mask_spinal_canal=mask_spinal_canal,
            )
        
        # Visualize Sacral Slope (SS): S1 Top and Horizontal
        if SS is not None:
            visualize_SS_S1(
                slice_img=img_zy,
                s1_mask=mask_S1,
                l5_mask=mask_L5, # Pass L5 mask
                angle_deg=SS,
                title="Sacral Slope (SS)",
                save_path=os.path.join(save_dir, "global_cobb_ss.png"),
                mask_l5s1_disc=mask_L5S1_disc,
                mask_spinal_canal=mask_spinal_canal,
            )
        
        # Visualize Lumbosacral Angle (LSA) - using new method
        if LSA is not None:
            visualize_LSA(
                slice_img=img_zy,
                l5_mask=mask_L5,
                s1_mask=mask_S1,
                angle_deg=LSA,
                title="Lumbosacral Angle (LSA)",
                save_path=os.path.join(save_dir, "global_cobb_lsa.png"),
                mask_l5s1_disc=mask_L5S1_disc,
                mask_spinal_canal=mask_spinal_canal,
            )

    return out


def calc_disc_inclination_angle(
    mri_zyx: np.ndarray,
    step2_zyx: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    disc_label: int,
    mid_sag_x: int,
    save_path: Optional[str] = None,
    name: str = "",
    upper_vertebra_label: Optional[int] = None,
    lower_vertebra_label: Optional[int] = None,
    spinal_canal_label: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Disc inclination angle (DIA): Angle between upper vertebra inferior endplate and lower vertebra superior endplate.
    Uses extract_endplate_line_by_disc_and_canal to extract endplate lines.
    """
    if cv2 is None:
        return {"dia_deg": None, "units": "deg", "status": "opencv_not_available"}
    
    if upper_vertebra_label is None or lower_vertebra_label is None or spinal_canal_label is None:
        return {"dia_deg": None, "units": "deg", "status": "missing_required_labels", "error": "upper_vertebra_label, lower_vertebra_label, and spinal_canal_label are required"}
    
    disc_zy = (step2_zyx == disc_label).astype(np.uint8)[:, :, mid_sag_x]
    if disc_zy.sum() == 0:
        return {"dia_deg": None, "units": "deg", "status": "missing_label"}
    
    try:
        # Get masks
        upper_mask = (step2_zyx == upper_vertebra_label).astype(np.uint8)[:, :, mid_sag_x]
        lower_mask = (step2_zyx == lower_vertebra_label).astype(np.uint8)[:, :, mid_sag_x]
        canal_mask = (step2_zyx == spinal_canal_label).astype(np.uint8)[:, :, mid_sag_x]
        
        if upper_mask.sum() == 0:
            return {"dia_deg": None, "units": "deg", "status": "missing_upper_vertebra_mask"}
        if lower_mask.sum() == 0:
            return {"dia_deg": None, "units": "deg", "status": "missing_lower_vertebra_mask"}
        if canal_mask.sum() == 0:
            return {"dia_deg": None, "units": "deg", "status": "missing_canal_mask"}
        
        # Isolate vertebral bodies
        upper_body = isolate_vertebral_body(upper_mask)
        if upper_body.sum() == 0:
            return {"dia_deg": None, "units": "deg", "status": "failed_to_isolate_upper_body"}
        
        # Extract upper vertebra inferior endplate
        line_upper = extract_endplate_line_by_disc_and_canal(
            upper_body, disc_zy, canal_mask, debug_roi=None
        )
        
        if line_upper is None:
            return {"dia_deg": None, "units": "deg", "status": "failed_to_extract_upper_endplate"}
        
        # Extract lower vertebra superior endplate
        # Special handling for S1
        if lower_vertebra_label == LABEL_MAP.get("vertebrae", {}).get("S", 50) or \
           lower_vertebra_label == LABEL_MAP.get("vertebrae", {}).get("S1", 50):
            # For S1, use get_s1_superior_line
            l5_mask = None
            if upper_vertebra_label == LABEL_MAP.get("vertebrae", {}).get("L5", 45):
                l5_mask = upper_mask
            line_lower = get_s1_superior_line(
                lower_mask, 
                mask_l5=l5_mask,
                mask_l5s1_disc=disc_zy,
                mask_spinal_canal=canal_mask
            )
        else:
            lower_body = isolate_vertebral_body(lower_mask)
            if lower_body.sum() == 0:
                return {"dia_deg": None, "units": "deg", "status": "failed_to_isolate_lower_body"}
            line_lower = extract_endplate_line_by_disc_and_canal(
                lower_body, disc_zy, canal_mask, debug_roi=None
            )
        
        if line_lower is None:
            return {"dia_deg": None, "units": "deg", "status": "failed_to_extract_lower_endplate"}
        
        # Calculate angle between two lines
        dia_angle = calculate_angle_between_lines(line_upper, line_lower)
        if dia_angle is None:
            return {"dia_deg": None, "units": "deg", "status": "failed_to_calculate_angle"}
        
        if save_path is not None:
            visualize_DIA(
                mri_zyx[:, :, mid_sag_x],
                disc_zy,
                dia_angle,
                save_path,
                name=name,
                line_upper=line_upper,
                line_lower=line_lower,
                upper_mask=upper_mask,
                lower_mask=lower_mask,
            )
        
        return {"dia_deg": _safe_float(dia_angle), "units": "deg", "status": "ok", "method": "angle between upper inferior and lower superior endplates"}
    except Exception as e:
        return {"dia_deg": None, "units": "deg", "status": "error", "error": str(e)}


def calc_ldh_parameters(
    mri_zyx: np.ndarray,
    step2_zyx: np.ndarray,
    ldh_zyx: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    disc_labels: Dict[str, int],
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    LDH parameters (axial, slice with maximum LDH area):
      - PD: protrusion distance (mm)
      - PA: protrusion area (mm^2)
      - PAR: PA / disc_area
      - PLR: PD / disc_AP_diameter

    disc selection: Choose disc label with maximum 3D overlap with LDH; if no overlap, choose disc with z COM closest to maximum LDH slice.
    """
    sx, sy, _sz = spacing_xyz
    z_max = _axial_z_index_max_area(ldh_zyx)
    if z_max is None:
        return {
            "axial_z_index_max_ldh": None,
            "pd_mm": None,
            "pa_mm2": None,
            "par": None,
            "plr": None,
            "units": {"pd": "mm", "pa": "mm^2"},
            "status": "missing_ldh",
        }

    ldh_slice = (ldh_zyx[z_max, :, :] > 0).astype(np.uint8)  # (Y,X)

    # select disc
    best_name = None
    best_label = None
    best_overlap = -1
    for dn, dl in disc_labels.items():
        dmask = (step2_zyx == int(dl)).astype(np.uint8)
        overlap = int(((dmask > 0) & (ldh_zyx > 0)).sum())
        if overlap > best_overlap:
            best_overlap = overlap
            best_name = dn
            best_label = int(dl)

    if best_label is None or best_name is None:
        return {"axial_z_index_max_ldh": int(z_max), "status": "no_disc_labels"}

    if best_overlap <= 0:
        # fallback: use disc with closest z COM
        z_target = float(z_max)
        best_dist = float("inf")
        for dn, dl in disc_labels.items():
            dmask = (step2_zyx == int(dl)).astype(np.uint8)
            com = _center_of_mass_idx(dmask)
            if com is None:
                continue
            dist = abs(float(com[0]) - z_target)
            if dist < best_dist:
                best_dist = dist
                best_name = dn
                best_label = int(dl)

    disc_slice = (step2_zyx[z_max, :, :] == best_label).astype(np.uint8)  # (Y,X)

    # area
    pa_mm2 = float(ldh_slice.sum() * sx * sy)
    disc_area_mm2 = float(disc_slice.sum() * sx * sy) if disc_slice.sum() > 0 else None
    par = None if disc_area_mm2 is None or disc_area_mm2 <= 1e-6 else float(pa_mm2 / disc_area_mm2)

    # disc AP diameter (axial, Y direction is AP)
    disc_ap_mm = None
    if disc_slice.sum() > 0:
        ys = np.where(disc_slice > 0)[0]
        disc_ap_mm = float((int(ys.max()) - int(ys.min())) * sy) if ys.size > 0 else None

    # PD: Find the longest vertical line in LDH by scanning horizontally (X direction)
    # Then calculate distance from this line to disc boundary
    pd_mm = None
    longest_line_x = None
    longest_line_y_min = None
    longest_line_y_max = None
    longest_line_length = 0
    
    contour = get_mask_contour(disc_slice)
    if ldh_slice.sum() > 0 and contour.shape[0] > 0:
        # Scan horizontally (X direction) to find longest vertical line (Y direction)
        for x in range(ldh_slice.shape[1]):
            # Get all y coordinates where LDH exists at this x position
            y_coords = np.where(ldh_slice[:, x] > 0)[0]
            if y_coords.size == 0:
                continue
            
            # Find continuous segments in y direction
            # Sort y coordinates and find gaps
            y_sorted = np.sort(y_coords)
            if y_sorted.size == 1:
                segment_length = 1
                y_min = y_max = y_sorted[0]
            else:
                # Find the longest continuous segment
                gaps = np.diff(y_sorted) > 1
                if np.any(gaps):
                    # Multiple segments, find the longest one
                    gap_indices = np.where(gaps)[0]
                    segment_starts = np.concatenate([[0], gap_indices + 1])
                    segment_ends = np.concatenate([gap_indices + 1, [y_sorted.size]])
                    segment_lengths = segment_ends - segment_starts
                    longest_seg_idx = np.argmax(segment_lengths)
                    y_min = y_sorted[segment_starts[longest_seg_idx]]
                    y_max = y_sorted[segment_ends[longest_seg_idx] - 1]
                    segment_length = segment_lengths[longest_seg_idx]
                else:
                    # Single continuous segment
                    y_min = y_sorted[0]
                    y_max = y_sorted[-1]
                    segment_length = y_max - y_min + 1
            
            # Update if this is the longest line found so far
            if segment_length > longest_line_length:
                longest_line_length = segment_length
                longest_line_x = x
                longest_line_y_min = y_min
                longest_line_y_max = y_max
        
        # Calculate distance from the longest vertical line to disc boundary
        if longest_line_x is not None and longest_line_y_min is not None and longest_line_y_max is not None:
            # contour is (x,y); convert to physical (x*sx, y*sy)
            boundary_xy_mm = np.stack([contour[:, 0] * sx, contour[:, 1] * sy], axis=1)
            tree = cKDTree(boundary_xy_mm)
            
            # Sample points along the vertical line (use midpoint and endpoints)
            y_mid = (longest_line_y_min + longest_line_y_max) / 2.0
            line_points_yx = [
                (longest_line_y_min, longest_line_x),  # top point
                (y_mid, longest_line_x),  # midpoint
                (longest_line_y_max, longest_line_x),  # bottom point
            ]
            
            # Convert to physical coordinates and find maximum distance
            max_dist = 0.0
            farthest_point_xy = None
            nearest_boundary_xy = None
            
            for y, x in line_points_yx:
                point_xy_mm = np.array([float(x) * sx, float(y) * sy], dtype=np.float32)
                dist, idx = tree.query(point_xy_mm, k=1)
                if dist > max_dist:
                    max_dist = float(dist)
                    farthest_point_xy = point_xy_mm
                    nearest_boundary_xy = boundary_xy_mm[int(idx)]
            
            if max_dist > 0:
                pd_mm = max_dist
                # For visualization: store the vertical line endpoints
                ldh_tip_xy = None  # Not used for line visualization
                nearest_xy = nearest_boundary_xy
                # Store line coordinates for visualization
                line_x_px = longest_line_x
                line_y_min_px = longest_line_y_min
                line_y_max_px = longest_line_y_max
            else:
                ldh_tip_xy = None
                nearest_xy = None
                line_x_px = None
                line_y_min_px = None
                line_y_max_px = None
        else:
            ldh_tip_xy = None
            nearest_xy = None
            line_x_px = None
            line_y_min_px = None
            line_y_max_px = None
    else:
        ldh_tip_xy = None
        nearest_xy = None
        line_x_px = None
        line_y_min_px = None
        line_y_max_px = None

    plr = None
    if pd_mm is not None and disc_ap_mm is not None and disc_ap_mm > 1e-6:
        plr = float(pd_mm / disc_ap_mm)

    out = {
        "axial_z_index_max_ldh": int(z_max),
        "disc_level": best_name,
        "disc_label_id": int(best_label),
        "pd_mm": _safe_float(pd_mm),
        "pa_mm2": _safe_float(pa_mm2),
        "disc_area_mm2": _safe_float(disc_area_mm2),
        "par": _safe_float(par),
        "disc_ap_diameter_mm": _safe_float(disc_ap_mm),
        "plr": _safe_float(plr),
        "units": {"pd": "mm", "pa": "mm^2", "disc_area": "mm^2", "disc_ap_diameter": "mm"},
        "status": "ok",
    }

    if save_dir is not None:
        _ensure_dir(save_dir)
        img_yx = _normalize_intensity_percentile(mri_zyx[z_max, :, :])  # axial (Y,X)
        # crop around disc + ldh
        union = ((disc_slice > 0) | (ldh_slice > 0)).astype(np.uint8)
        crop = _crop_around_mask(union, pad=50)  # Increased padding to show more surrounding area
        img_c = _apply_crop(img_yx, crop)
        disc_c = _apply_crop(disc_slice, crop)
        ldh_c = _apply_crop(ldh_slice, crop)
        overlays = [
            ("mask", {"mask": disc_c > 0, "color": "lime", "alpha": 0.55}),
            ("mask", {"mask": ldh_c > 0, "color": "r", "alpha": 0.55}),
        ]
        # Draw the longest vertical line in LDH
        if crop is not None and line_x_px is not None and line_y_min_px is not None and line_y_max_px is not None:
            crop_xy = _safe_unpack_crop(crop)
            if crop_xy is not None:
                x0, y0 = crop_xy
                # Convert to cropped coordinates
                line_x_cropped = float(line_x_px - x0)
                line_y_min_cropped = float(line_y_min_px - y0)
                line_y_max_cropped = float(line_y_max_px - y0)
                # Draw vertical line
                overlays.append(
                    (
                        "line",
                        {
                            "p0": (line_x_cropped, line_y_min_cropped),
                            "p1": (line_x_cropped, line_y_max_cropped),
                            "color": "yellow",
                            "lw": 2,
                            "label": f"PD {pd_mm:.1f}" if pd_mm is not None else None,
                        },
                    )
                )
        save_visualization(
            os.path.join(save_dir, "global_herniation_summary.png"),
            img_c,
            "LDH Measurements (Axial max area)",
            overlays,
        )

    return out


def calc_average_gray_level_in_discs(
    mri_zyx: np.ndarray,
    step2_zyx: np.ndarray,
    disc_labels: Dict[str, int],
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    AGL: Average gray level within disc mask (0-1 normalized, percentile clipping).
    """
    norm = _normalize_intensity_percentile(mri_zyx)
    out_by = {}
    for dn, dl in disc_labels.items():
        dmask = (step2_zyx == int(dl))
        if dmask.sum() == 0:
            out_by[dn] = {"agl_norm": None, "status": "missing_label"}
            continue
        val = float(norm[dmask].mean())
        out_by[dn] = {"agl_norm": _safe_float(val), "status": "ok", "units": "normalized_0_1"}

    # AGL visualization: as required "each parameter needs png", create overview chart (text labels for each disc)
    if save_dir is not None:
        _ensure_dir(save_dir)
        fig = plt.figure(figsize=(7, 4), dpi=160)
        ax = fig.add_subplot(1, 1, 1)
        names = list(out_by.keys())
        vals = [out_by[n]["agl_norm"] if out_by[n]["agl_norm"] is not None else np.nan for n in names]
        ax.bar(np.arange(len(names)), vals)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("AGL (normalized 0-1)")
        ax.set_title("Average Gray Level in Discs")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "global_intensity_agl.png"), bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    return {"status": "ok", "by_disc": out_by, "method": "percentile(1,99) -> [0,1] then mean within disc mask"}


# =========================
# Level 3: Orchestrator
# =========================


def generate_clinical_report(mri_path: str, step2_path: str, ldh_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Main entry: Load data, calculate all parameters, output JSON + preview PNG.
    """
    _ensure_dir(output_dir)
    preview_dir = os.path.join(output_dir, "previews")
    preview_vertebrae = os.path.join(preview_dir, "vertebrae")
    preview_discs = os.path.join(preview_dir, "discs")
    preview_global = os.path.join(preview_dir, "global")
    _ensure_dir(preview_vertebrae)
    _ensure_dir(preview_discs)
    _ensure_dir(preview_global)

    mri = load_nifti(mri_path)
    step2 = load_nifti(step2_path)
    ldh = load_nifti(ldh_path)

    # Resample MRI and LDH to step2's space (1mm iso space)
    # step2 is in 1mm iso space from infer_ldh.py output
    # Check if resampling is needed (shape or spacing mismatch)
    needs_mri_resample = (
        mri.arr_zyx.shape != step2.arr_zyx.shape 
        or mri.spacing_xyz != step2.spacing_xyz
        or mri.img.GetDirection() != step2.img.GetDirection()
    )
    needs_ldh_resample = (
        ldh.arr_zyx.shape != step2.arr_zyx.shape 
        or ldh.spacing_xyz != step2.spacing_xyz
        or ldh.img.GetDirection() != step2.img.GetDirection()
    )
    
    if needs_mri_resample:
        mri_resampled = resample_to_reference(mri.img, step2.img, interpolator=sitk.sitkLinear)
        mri_arr = sitk.GetArrayFromImage(mri_resampled).astype(np.float32)
        # Note: Flipping will be done in save_visualization to match medical image convention
        mri = NiftiVolume(img=mri_resampled, arr_zyx=mri_arr, spacing_xyz=step2.spacing_xyz)
    
    if needs_ldh_resample:
        ldh_resampled = resample_to_reference(ldh.img, step2.img, interpolator=sitk.sitkNearestNeighbor)
        ldh_arr = sitk.GetArrayFromImage(ldh_resampled).astype(np.float32)
        # Note: Flipping will be done in save_visualization to match medical image convention
        ldh = NiftiVolume(img=ldh_resampled, arr_zyx=ldh_arr, spacing_xyz=step2.spacing_xyz)
    
    # Verify all are now in the same space
    if mri.arr_zyx.shape != step2.arr_zyx.shape:
        raise ValueError(f"After resampling, MRI shape {mri.arr_zyx.shape} != step2 mask shape {step2.arr_zyx.shape}")
    if mri.arr_zyx.shape != ldh.arr_zyx.shape:
        raise ValueError(f"After resampling, MRI shape {mri.arr_zyx.shape} != ldh mask shape {ldh.arr_zyx.shape}")

    # slice selection
    canal_label = int(LABEL_MAP["spinal_canal"])
    mid_sag_x = _mid_sagittal_x_index(step2.arr_zyx, canal_label=canal_label)
    axial_ldh_z = _axial_z_index_max_area(ldh.arr_zyx)

    vertebrae: Dict[str, int] = dict(LABEL_MAP.get("vertebrae", {}))
    discs: Dict[str, int] = dict(LABEL_MAP.get("discs", {}))

    report: Dict[str, Any] = {
        "inputs": {"mri_path": mri_path, "step2_mask_path": step2_path, "ldh_mask_path": ldh_path},
        "spacing_mm": {"x": mri.spacing_xyz[0], "y": mri.spacing_xyz[1], "z": mri.spacing_xyz[2]},
        "selected_slices": {
            "mid_sagittal_x_index": int(mid_sag_x),
            "axial_z_index_max_ldh": (int(axial_ldh_z) if axial_ldh_z is not None else None),
        },
        "geometry": {"vertebral_height": {}, "vertebral_ap_diameter": {}, "disc_metrics": {}},
        "angles": {},
        "herniation": {},
        "intensity": {},
        "notes": [
            "All distances are converted to mm using SimpleITK spacing; angles are in degrees.",
            "Mid-sagittal slice is estimated from spinal canal mask centroid x index; if canal is missing, use volume center slice.",
            "Since NIfTI orientation may not be standardized, anterior/posterior direction is heuristically estimated from 'canal relative position'.",
            "DHI formula used here: DHI = DH / mean(upper_VH_avg, lower_VH_avg) (modify in function if strict clinical standard is needed).",
        ],
    }

    # -------- Geometry: Vertebrae (VH + AP diameter) --------
    # CRITICAL: Only process L1-L5, skip S1 (Sacrum) as it uses different measurement methods
    vh_avg_by_v = {}
    for vn, vid in vertebrae.items():
        # Skip S1 - it's not a vertebra and uses specialized S1 extraction methods
        if vn.upper() == 'S' or vn.upper() == 'S1' or "SACRUM" in vn.upper():
            vh_avg_by_v[vn] = None
            report["geometry"]["vertebral_height"][vn] = {"anterior_mm": None, "posterior_mm": None, "units": "mm", "status": "skipped_s1"}
            report["geometry"]["vertebral_ap_diameter"][vn] = {"ap_diameter_mm": None, "units": "mm", "status": "skipped_s1"}
            continue
        
        vh = calc_vertebral_height(
            mri_zyx=mri.arr_zyx,
            step2_zyx=step2.arr_zyx.astype(np.int32),
            spacing_xyz=mri.spacing_xyz,
            vertebra_label=int(vid),
            canal_label=canal_label,
            mid_sag_x=mid_sag_x,
            save_path=os.path.join(preview_vertebrae, f"vert_{vn}_vh.png"),
            name=vn,
        )
        report["geometry"]["vertebral_height"][vn] = vh
        # average vertebral height (for DHI)
        if vh.get("anterior_mm") is not None and vh.get("posterior_mm") is not None:
            vh_avg_by_v[vn] = float((vh["anterior_mm"] + vh["posterior_mm"]) / 2.0)
        else:
            vh_avg_by_v[vn] = None

        apd = calc_vertebral_width_axial(
            mri_zyx=mri.arr_zyx,
            step2_zyx=step2.arr_zyx.astype(np.int32),
            spacing_xyz=mri.spacing_xyz,
            vertebra_label=int(vid),
            canal_label=canal_label,
            mid_sag_x=mid_sag_x,
            save_path=os.path.join(preview_vertebrae, f"vert_{vn}_ap.png"),
            name=vn,
        )
        report["geometry"]["vertebral_ap_diameter"][vn] = apd

    # -------- Geometry: Discs (DH, DHI, HDR) --------
    # upper/lower vertebra mapping (for DHI)
    disc_to_upper_lower = {
        # Cervical discs
        "C2-C3": ("C2", "C3"),
        "C3-C4": ("C3", "C4"),
        "C4-C5": ("C4", "C5"),
        "C5-C6": ("C5", "C6"),
        "C6-C7": ("C6", "C7"),
        "C7-T1": ("C7", "T1"),
        # Thoracic discs
        "T1-T2": ("T1", "T2"),
        "T2-T3": ("T2", "T3"),
        "T3-T4": ("T3", "T4"),
        "T4-T5": ("T4", "T5"),
        "T5-T6": ("T5", "T6"),
        "T6-T7": ("T6", "T7"),
        "T7-T8": ("T7", "T8"),
        "T8-T9": ("T8", "T9"),
        "T9-T10": ("T9", "T10"),
        "T10-T11": ("T10", "T11"),
        "T11-T12": ("T11", "T12"),
        "T12-L1": ("T12", "L1"),
        # Lumbar discs
        "L1-L2": ("L1", "L2"),
        "L2-L3": ("L2", "L3"),
        "L3-L4": ("L3", "L4"),
        "L4-L5": ("L4", "L5"),
        "L5-S": ("L5", "S"),  # S = Sacrum
    }

    for dn, did in discs.items():
        up, low = disc_to_upper_lower.get(dn, (None, None))
        up_vh = vh_avg_by_v.get(up) if up is not None else None
        low_vh = vh_avg_by_v.get(low) if low is not None else None
        disc_metrics = calc_disc_height_and_hdr_dhi(
            mri_zyx=mri.arr_zyx,
            step2_zyx=step2.arr_zyx.astype(np.int32),
            spacing_xyz=mri.spacing_xyz,
            disc_label=int(did),
            mid_sag_x=mid_sag_x,
            canal_label=canal_label,
            upper_vh_avg_mm=_safe_float(up_vh),
            lower_vh_avg_mm=_safe_float(low_vh),
            save_path=os.path.join(preview_discs, f"disc_{dn}_dm.png"),
            name=dn,
        )
        report["geometry"]["disc_metrics"][dn] = disc_metrics

        # Get upper and lower vertebra labels for new DIA calculation method
        up_label = None
        low_label = None
        if up is not None and up in vertebrae:
            up_label = int(vertebrae[up])
        if low is not None and low in vertebrae:
            low_label = int(vertebrae[low])
        
        dia = calc_disc_inclination_angle(
            mri_zyx=mri.arr_zyx,
            step2_zyx=step2.arr_zyx.astype(np.int32),
            spacing_xyz=mri.spacing_xyz,
            disc_label=int(did),
            mid_sag_x=mid_sag_x,
            save_path=os.path.join(preview_discs, f"disc_{dn}_dia.png"),
            name=dn,
            upper_vertebra_label=up_label,
            lower_vertebra_label=low_label,
            spinal_canal_label=canal_label,
        )
        report.setdefault("angles", {}).setdefault("disc_inclination_angle_DIA", {})[dn] = dia

    # -------- Angles: LL/SS/LSA --------
    if all(k in vertebrae for k in ("L1", "L5", "S")):  # S = Sacrum
        cobb_results = calc_cobb_angles(
            mri_zyx=mri.arr_zyx,
            step2_zyx=step2.arr_zyx.astype(np.int32),
            spacing_xyz=mri.spacing_xyz,
            mid_sag_x=mid_sag_x,
            label_L1=int(vertebrae["L1"]),
            label_L5=int(vertebrae["L5"]),
            label_S1=int(vertebrae["S"]),  # S = Sacrum (label 50)
            save_dir=preview_global,
        )
        report["angles"].update(cobb_results)
        
        # Extract global metrics: LL, SS, LSA
        ll_value = cobb_results.get("lumbar_lordosis_LL_deg")
        ss_value = cobb_results.get("sacral_slope_SS_deg")
        lsa_value = cobb_results.get("lumbosacral_angle_LSA_deg")
        
        # Determine status based on available values
        status = "ok"
        if ll_value is None and ss_value is None and lsa_value is None:
            status = "failed"
        elif ll_value is None or ss_value is None or lsa_value is None:
            status = "partial"
        
        # Build global_metrics object with rounded values (2 decimal places)
        def _round_angle(val):
            if val is None:
                return None
            try:
                return round(float(val), 2)
            except (ValueError, TypeError):
                return None
        
        report["global_metrics"] = {
            "ll_deg": _round_angle(ll_value),
            "ss_deg": _round_angle(ss_value),
            "lsa_deg": _round_angle(lsa_value),
            "units": "degrees",
            "status": status
        }
    else:
        report["angles"]["status"] = "missing_required_vertebra_labels_for_LL_SS_LSA"
        # Set global_metrics to failed status when required vertebrae are missing
        report["global_metrics"] = {
            "ll_deg": None,
            "ss_deg": None,
            "lsa_deg": None,
            "units": "degrees",
            "status": "failed"
        }

    # -------- Herniation (LDH) --------
    report["herniation"] = calc_ldh_parameters(
        mri_zyx=mri.arr_zyx,
        step2_zyx=step2.arr_zyx.astype(np.int32),
        ldh_zyx=(ldh.arr_zyx > 0).astype(np.uint8),
        spacing_xyz=mri.spacing_xyz,
        disc_labels=discs,
        save_dir=preview_global,
    )

    # -------- Intensity: AGL --------
    report["intensity"]["average_gray_level_AGL"] = calc_average_gray_level_in_discs(
        mri_zyx=mri.arr_zyx,
        step2_zyx=step2.arr_zyx.astype(np.int32),
        disc_labels=discs,
        save_dir=preview_global,
    )

    # Extract herniation metrics (pd, pa, par, plr) from calc_ldh_parameters result
    # and add to global_metrics
    herniation_data = report.get("herniation", {})
    if isinstance(herniation_data, dict) and herniation_data.get("status") == "ok":
        # Extract values and round to 2 decimal places
        pd_val = herniation_data.get("pd_mm")
        pa_val = herniation_data.get("pa_mm2")
        par_val = herniation_data.get("par")
        plr_val = herniation_data.get("plr")
        
        # Round to 2 decimal places if not None
        def _round_value(val):
            if val is None:
                return None
            try:
                return round(float(val), 2)
            except (ValueError, TypeError):
                return None
        
        # Update global_metrics
        if report.get("global_metrics") is None:
            report["global_metrics"] = {}
        
        report["global_metrics"]["pd_mm"] = _round_value(pd_val)
        report["global_metrics"]["pa_mm2"] = _round_value(pa_val)
        report["global_metrics"]["par"] = _round_value(par_val)
        report["global_metrics"]["plr"] = _round_value(plr_val)

    def _pivot_entity_based(old_report: Dict[str, Any]) -> Dict[str, Any]:
        vertebrae_out: Dict[str, Dict[str, Any]] = {}
        discs_out: Dict[str, Dict[str, Any]] = {}

        vh_by_v = old_report.get("geometry", {}).get("vertebral_height", {})
        ap_by_v = old_report.get("geometry", {}).get("vertebral_ap_diameter", {})
        for level, vh in (vh_by_v or {}).items():
            if not isinstance(vh, dict) or vh.get("status") != "ok":
                continue
            vertebrae_out.setdefault(str(level), {"level": str(level)})["vh"] = vh
        for level, ap in (ap_by_v or {}).items():
            if not isinstance(ap, dict) or ap.get("status") != "ok":
                continue
            vertebrae_out.setdefault(str(level), {"level": str(level)})["ap"] = ap

        disc_metrics_by_d = old_report.get("geometry", {}).get("disc_metrics", {})
        for level, dm in (disc_metrics_by_d or {}).items():
            if not isinstance(dm, dict) or dm.get("status") != "ok":
                continue
            discs_out.setdefault(str(level), {"level": str(level)})["dm"] = dm

        dia_by_d = (
            old_report.get("angles", {})
            .get("disc_inclination_angle_DIA", {})
        )
        for level, dia in (dia_by_d or {}).items():
            if not isinstance(dia, dict) or dia.get("status") != "ok":
                continue
            discs_out.setdefault(str(level), {"level": str(level)})["dia"] = dia

        ldh_by_disc = old_report.get("herniation", {}).get("by_disc", {})
        for level, ldh_item in (ldh_by_disc or {}).items():
            if not isinstance(ldh_item, dict) or ldh_item.get("status") != "ok":
                continue
            discs_out.setdefault(str(level), {"level": str(level)})["ldh"] = ldh_item

        entity_report: Dict[str, Any] = {
            "inputs": old_report.get("inputs", {}),
            "spacing_mm": old_report.get("spacing_mm", {}),
            "selected_slices": old_report.get("selected_slices", {}),
            "previews": {
                "vertebrae_dir": "previews/vertebrae",
                "discs_dir": "previews/discs",
                "global_dir": "previews/global",
            },
            "vertebrae": sorted(vertebrae_out.values(), key=lambda x: x.get("level", "")),
            "discs": sorted(discs_out.values(), key=lambda x: x.get("level", "")),
            "notes": old_report.get("notes", []),
            "global_metrics": old_report.get("global_metrics", {}),
        }

        for v in entity_report["vertebrae"]:
            level = v.get("level")
            if isinstance(level, str):
                v["previews"] = {
                    "vh": f"previews/vertebrae/vert_{level}_vh.png",
                    "ap": f"previews/vertebrae/vert_{level}_ap.png",
                }
        for d in entity_report["discs"]:
            level = d.get("level")
            if isinstance(level, str):
                d["previews"] = {
                    "dm": f"previews/discs/disc_{level}_dm.png",
                    "dia": f"previews/discs/disc_{level}_dia.png",
                }
        return entity_report

    report = _pivot_entity_based(report)

    # Save JSON
    json_path = os.path.join(output_dir, "report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Legacy compatibility: if any producer wrote into raw/preview, flatten & rename into previews/*
    _standardize_previews_from_raw_preview(output_dir)

    return report


def _find_matching_files(input_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Find raw images in input directory and step2_output, ldh_output files under infer_output.
    Returns: {case_id: {"mri": path, "step2": path, "ldh": path}}
    
    Parameters:
    -----------
    input_dir : str
        Input directory path containing raw images and infer_output folder
    """
    input_path = Path(input_dir)
    infer_output = input_path / "infer_output"
    step2_dir = infer_output / "step2_output"
    ldh_dir = infer_output / "ldh_output"
    
    if not infer_output.exists():
        raise SystemExit(f"infer_output folder not found: {infer_output}")
    if not step2_dir.exists():
        raise SystemExit(f"step2_output folder not found: {step2_dir}")
    if not ldh_dir.exists():
        raise SystemExit(f"ldh_output folder not found: {ldh_dir}")
    
    # Find raw images (.nii.gz files, exclude infer_output directory)
    raw_images = {}
    for img_path in input_path.glob("*.nii.gz"):
        if img_path.parent == input_path:  # ensure in input directory root
            case_id = img_path.stem.replace(".nii", "")  # remove .nii.gz suffix
            raw_images[case_id] = str(img_path)
    
    if not raw_images:
        raise SystemExit(f"No raw .nii.gz images found in input directory: {input_path}")
    
    # Match step2 and ldh files
    matches = {}
    for case_id, mri_path in raw_images.items():
        step2_path = step2_dir / f"{case_id}.nii.gz"
        ldh_path = ldh_dir / f"{case_id}.nii.gz"
        
        if not step2_path.exists():
            continue
        if not ldh_path.exists():
            continue
        
        matches[case_id] = {
            "mri": mri_path,
            "step2": str(step2_path),
            "ldh": str(ldh_path),
        }
    
    if not matches:
        raise SystemExit(f"No matching file pairs found (mri + step2 + ldh)")
    
    return matches


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Calculate LDH & spinal alignment clinical parameters with visualizations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # 方式1: 使用显式参数（推荐）
  python calculate.py --input-dir /path/to/input --output-dir /path/to/output
  
  # 方式2: 使用parent_dir（向后兼容）
  python calculate.py /path/to/parent_dir

Directory structure (方式1 - 显式参数):
  input_dir/
    ├── case1.nii.gz          (raw MRI)
    ├── case2.nii.gz
    └── infer_output/
        ├── step2_output/
        │   ├── case1.nii.gz
        │   └── case2.nii.gz
        └── ldh_output/
            ├── case1.nii.gz
            └── case2.nii.gz

Directory structure (方式2 - parent_dir):
  parent_dir/
    ├── case1.nii.gz          (raw MRI)
    ├── case2.nii.gz
    └── infer_output/
        ├── step2_output/
        │   ├── case1.nii.gz
        │   └── case2.nii.gz
        └── ldh_output/
            ├── case1.nii.gz
            └── case2.nii.gz

Output will be saved to: output_dir/ (方式1) or parent_dir/clinical_report/ (方式2)
        """,
    )
    ap.add_argument(
        "parent_dir",
        type=str,
        nargs="?",
        default=None,
        help="Parent directory path (contains raw images and infer_output folder). 如果提供了--input-dir，此参数将被忽略。",
    )
    ap.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="输入文件夹路径（包含原始MRI图像和infer_output文件夹）",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出文件夹路径（用于保存计算结果）",
    )
    ap.add_argument(
        "--out-dir-name",
        type=str,
        default="result",
        help="输出目录名称（仅在未指定--output-dir时使用，默认: result）",
    )
    return ap


def main() -> None:
    args = _build_argparser().parse_args()
    
    # 确定输入目录：优先使用 --input-dir，否则使用 parent_dir
    if args.input_dir:
        input_dir = Path(args.input_dir).resolve()
    elif args.parent_dir:
        input_dir = Path(args.parent_dir).resolve()
    else:
        raise SystemExit("必须提供 --input-dir 或 parent_dir 参数")
    
    if not input_dir.is_dir():
        raise SystemExit(f"输入目录不存在: {input_dir}")
    
    # 确定输出目录：优先使用 --output-dir，否则基于 parent_dir 生成
    if args.output_dir:
        output_base = Path(args.output_dir).resolve()
    elif args.parent_dir:
        output_base = Path(args.parent_dir).resolve() / args.out_dir_name
    else:
        # 如果只提供了 --input-dir 而没有 --output-dir，在输入目录下创建输出目录
        output_base = input_dir / args.out_dir_name
    
    # 确保输出目录的父目录存在
    output_base.parent.mkdir(parents=True, exist_ok=True)
    
    # Find matching files
    matches = _find_matching_files(str(input_dir))
    
    # Output directory - clean existing folder before creating new one
    if output_base.exists():
        shutil.rmtree(output_base)
    _ensure_dir(str(output_base))

    multi_case = len(matches) > 1

    # Process each case with progress bar
    for case_id, paths in tqdm(matches.items(), desc="Processing cases", total=len(matches)):
        try:
            # Multi-case safety: write each case into its own subfolder to avoid
            # collisions while preserving the standardized previews/* convention.
            report_dir = (output_base / case_id) if multi_case else output_base
            _ensure_dir(str(report_dir))
            report = generate_clinical_report(
                mri_path=paths["mri"],
                step2_path=paths["step2"],
                ldh_path=paths["ldh"],
                output_dir=str(report_dir),
            )
        except Exception as e:
            # Only print errors, not warnings
            print(f"Error: Failed to process case {case_id}: {e}")


if __name__ == "__main__":
    main()


