#!/usr/bin/env python3
"""
LDH two-stage inference (Stage A detection + Stage B ROI segmentation) for unknown MRI volumes.

输入：
  - 仅支持“文件夹”形式输入（里面可以放 1 个或多个 .nii.gz / .nii）
  - 最简用法：`python scripts/infer_ldh.py /path/to/input_dir`（checkpoint 默认从 $TOTALSPINESEG_DATA 下解析）
输出（在输入文件夹下创建一个新的输出目录，结构参考 totalspineseg 推理输出）：
  - step1_raw/, step1_output/, step2_input/, step2_raw/, step2_output/, preview/, ...
  - ldh_stagea/            (每例 disc-level 概率与中心点 JSON)
  - ldh_stageb_iso/        (每例 LDH mask，1mm/iso 空间)
  - ldh_output/            (每例 LDH mask，回到原始输入空间)
  - preview/ldh/           (LDH 红色 overlay 预览图)

注意：
  - 为匹配 Dataset105 的训练预处理，这里强制使用 output_iso=True（1mm LPI canonical）。
  - step1/step2 的 preview 图片由 totalspineseg 原生逻辑生成；LDH preview 由本脚本生成（红色）。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import multiprocessing as mp
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Add repository root to Python path for imports (so `python scripts/infer_ldh.py ...` works)
_script_dir = Path(__file__).parent.resolve()
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# IMPORTANT: this script runs nnUNet inference on CUDA, then runs multiprocessing-heavy CPU steps
# (including preview generation). On Linux, the default multiprocessing method is "fork", which
# can deadlock after CUDA was initialized. Force a safer start method.
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # start method already set
    pass


@dataclass(frozen=True)
class DiscDecision:
    disc_label: int
    prob: float
    positive: bool
    best_center_zyx: Tuple[int, int, int]
    best_center_rank: int


def _list_input_images(input_dir: Path) -> List[Path]:
    # 递归找 nii/nii.gz，扁平化输出（与 totalspineseg inference 的 flat 行为保持一致）。
    exts = {".nii", ".gz"}
    files: List[Path] = []
    for p in sorted(input_dir.rglob("*")):
        if not p.is_file():
            continue
        # accept .nii or .nii.gz
        if p.name.endswith(".nii") or p.name.endswith(".nii.gz"):
            files.append(p)
    return files


def _stem_nii(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def _prepare_original_raw(
    input_dir: Path,
    original_raw_dir: Path,
    *,
    overwrite: bool,
) -> Dict[str, Path]:
    """
    将输入文件夹里的原始影像复制/压缩到 original_raw_dir，并统一命名为 <case_id>_0000.nii.gz
    返回：case_id -> original source path（用于追踪）
    """
    original_raw_dir.mkdir(parents=True, exist_ok=True)
    files = _list_input_images(input_dir)
    if not files:
        raise SystemExit(f"在输入文件夹里没有找到 .nii/.nii.gz：{input_dir}")

    mapping: Dict[str, Path] = {}
    seen = set()
    for src in files:
        case_id = _stem_nii(src)
        if case_id in seen:
            raise SystemExit(
                f"检测到重复文件名（扁平化后会冲突）：{case_id!r}。\n"
                f"请确保输入目录下每个病例文件名唯一（不含扩展名）。"
            )
        seen.add(case_id)
        mapping[case_id] = src

        dst = original_raw_dir / f"{case_id}_0000.nii.gz"
        if dst.exists() and not overwrite:
            continue

        if src.name.endswith(".nii.gz"):
            shutil.copy2(src, dst)
        else:
            # .nii -> .nii.gz
            try:
                import nibabel as nib  # lazy import so --help works without deps
            except Exception as e:
                raise SystemExit(
                    "需要 nibabel 来处理 .nii 输入（压缩为 .nii.gz）。请先安装依赖：\n"
                    "  pip install -e .\n"
                    f"原始错误: {type(e).__name__}: {e}"
                )
            img = nib.load(str(src))
            nib.save(img, str(dst))

    return mapping


def _mkdir_clean(path: Path, *, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def _move_preview_by_stage(preview_dir: Path) -> None:
    """
    将 totalspineseg 原生 preview/*.jpg 按 stage 整理到子文件夹：
      preview/input, preview/step1, preview/step2, preview/loc, preview/other
    """
    if not preview_dir.exists():
        return
    stage_dirs = {
        "input": preview_dir / "input",
        "step1": preview_dir / "step1",
        "step2": preview_dir / "step2",
        "loc": preview_dir / "loc",
        "other": preview_dir / "other",
        "ldh": preview_dir / "ldh",  # 由本脚本直接写入
    }
    for d in stage_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    for p in list(preview_dir.glob("*.jpg")):
        name = p.name
        if "_step1" in name:
            dst_dir = stage_dirs["step1"]
        elif "_step2" in name:
            dst_dir = stage_dirs["step2"]
        elif "_loc" in name:
            dst_dir = stage_dirs["loc"]
        elif "_input" in name:
            dst_dir = stage_dirs["input"]
        else:
            dst_dir = stage_dirs["other"]
        dst = dst_dir / p.name
        dst.exists() and dst.unlink()
        p.rename(dst)


def _choose_k_coords(mask: "np.ndarray", rng: "np.random.RandomState", k: int) -> List[Tuple[int, int, int]]:
    import numpy as np

    coords = np.array(np.nonzero(mask))
    if coords.size == 0:
        return []
    n = coords.shape[1]
    k = int(min(max(k, 0), n))
    if k == 0:
        return []
    idx = rng.choice(n, size=k, replace=False)
    out: List[Tuple[int, int, int]] = []
    for j in np.atleast_1d(idx):
        z, y, x = coords[:, int(j)].tolist()
        out.append((int(z), int(y), int(x)))
    return out


def _centroid(mask: "np.ndarray") -> Tuple[int, int, int]:
    import numpy as np

    coords = np.array(np.nonzero(mask))
    if coords.size == 0:
        return mask.shape[0] // 2, mask.shape[1] // 2, mask.shape[2] // 2
    mean = coords.mean(axis=1)
    return int(round(mean[0])), int(round(mean[1])), int(round(mean[2]))


def _crop_with_padding(
    vol: "np.ndarray",
    center_zyx: Tuple[int, int, int],
    size_zyx: Tuple[int, int, int],
    pad_value: float = 0.0,
) -> Tuple["np.ndarray", Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
    import numpy as np

    cz, cy, cx = map(int, center_zyx)
    pz, py, px = map(int, size_zyx)
    hz, hy, hx = pz // 2, py // 2, px // 2

    z0, z1 = cz - hz, cz - hz + pz
    y0, y1 = cy - hy, cy - hy + py
    x0, x1 = cx - hx, cx - hx + px

    z0c, z1c = max(0, z0), min(vol.shape[0], z1)
    y0c, y1c = max(0, y0), min(vol.shape[1], y1)
    x0c, x1c = max(0, x0), min(vol.shape[2], x1)

    patch = vol[z0c:z1c, y0c:y1c, x0c:x1c]

    pad_before = (max(0, -z0), max(0, -y0), max(0, -x0))
    pad_after = (max(0, z1 - vol.shape[0]), max(0, y1 - vol.shape[1]), max(0, x1 - vol.shape[2]))
    if any(pad_before) or any(pad_after):
        patch = np.pad(
            patch,
            pad_width=((pad_before[0], pad_after[0]), (pad_before[1], pad_after[1]), (pad_before[2], pad_after[2])),
            mode="constant",
            constant_values=float(pad_value),
        )
    if patch.shape != (pz, py, px):
        raise RuntimeError(f"Patch shape mismatch: got {patch.shape}, expected {(pz, py, px)}")
    return patch, (z0c, y0c, x0c), (z1c, y1c, x1c), pad_before


def _uncrop_remove_padding(
    patch: "np.ndarray",
    start_zyx: Tuple[int, int, int],
    end_zyx: Tuple[int, int, int],
    pad_before_zyx: Tuple[int, int, int],
) -> "np.ndarray":
    z0c, y0c, x0c = start_zyx
    z1c, y1c, x1c = end_zyx
    pbz, pby, pbx = pad_before_zyx
    sz = z1c - z0c
    sy = y1c - y0c
    sx = x1c - x0c
    return patch[pbz : pbz + sz, pby : pby + sy, pbx : pbx + sx]


def _disc_region_from_step2(step2_full: "np.ndarray", disc_label: int) -> "np.ndarray":
    from scipy.ndimage import binary_dilation
    import numpy as np

    disc_mask = (step2_full == int(disc_label))
    if not disc_mask.any():
        return disc_mask.astype(np.uint8)
    return binary_dilation(disc_mask, structure=np.ones((7, 7, 7), dtype=bool)).astype(np.uint8)


def _sample_candidate_centers(
    disc_region: "np.ndarray",
    rng: "np.random.RandomState",
    n_boundary: int,
    n_interior: int,
) -> List[Tuple[int, int, int]]:
    import numpy as np
    from scipy.ndimage import binary_erosion

    from totalspineseg.ldh_twostage.distance_maps import boundary_band

    disc = disc_region.astype(bool)
    if not disc.any():
        return []
    bnd = boundary_band(disc_region.astype(np.uint8), radius=2).astype(bool)
    interior = binary_erosion(disc, structure=np.ones((3, 3, 3), dtype=bool))

    centers: List[Tuple[int, int, int]] = []
    centers.extend(_choose_k_coords(bnd, rng, int(n_boundary)))
    centers.extend(_choose_k_coords(interior, rng, int(n_interior)))

    # 去重（保持顺序）
    seen = set()
    uniq: List[Tuple[int, int, int]] = []
    for c in centers:
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)
    if not uniq:
        uniq = [_centroid(disc)]
    return uniq


def _save_ldh_preview_red(
    image_path: Path,
    seg_path: Path,
    out_jpg: Path,
    *,
    orient: str = "sag",
    sliceloc: float = 0.5,
    alpha: float = 0.55,
) -> None:
    """
    生成 LDH 预览图：灰度 MRI + 红色 overlay（seg>0）。
    逻辑尽量贴近 totalspineseg/utils/preview_jpg.py（canonical + 1mm + 取单张切片）。
    """
    import numpy as np
    import torchio as tio
    from PIL import Image

    image = tio.ScalarImage(image_path)
    image = tio.ToCanonical()(image)
    image = tio.Resample((1, 1, 1))(image)
    img = image.data.squeeze().numpy().astype(np.float64)

    axis = {"sag": 0, "cor": 1, "ax": 2}[orient]
    slice_index = int(float(sliceloc) * img.shape[axis])
    slice_img = img.take(slice_index, axis=axis)
    # normalize 0-255
    denom = (np.max(slice_img) - np.min(slice_img)) or 1.0
    slice_img_u8 = (255 * (slice_img - np.min(slice_img)) / denom).astype(np.uint8)
    rgb = np.repeat(slice_img_u8[:, :, None], 3, axis=2).astype(np.float32)

    # rotate/flip same as preview_jpg
    rgb = np.flipud(rgb)
    rgb = np.rot90(rgb, k=1)

    seg = tio.LabelMap(seg_path)
    seg = tio.ToCanonical()(seg)
    seg = tio.Resample(image)(seg)
    seg_data = seg.data.squeeze().numpy().round().astype(np.uint8)
    slice_seg = seg_data.take(slice_index, axis=axis)
    slice_seg = np.flipud(slice_seg)
    slice_seg = np.rot90(slice_seg, k=1)

    mask = slice_seg > 0
    if mask.any():
        red = np.zeros_like(rgb)
        red[..., 0] = 255.0
        a = float(alpha)
        rgb[mask] = rgb[mask] * (1.0 - a) + red[mask] * a

    out = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_jpg, quality=95)


def _resolve_dataset105_name(totalspineseg_data: Path) -> str:
    """
    Resolve Dataset105 folder name like "Dataset105_TotalSpineSeg_LDH".
    Prefer nnUNet/results (checkpoints are stored there), fallback to nnUNet/raw.
    """
    base = Path(totalspineseg_data)
    candidates = sorted((base / "nnUNet" / "results").glob("Dataset105_*"))
    if not candidates:
        candidates = sorted((base / "nnUNet" / "raw").glob("Dataset105_*"))
    if not candidates:
        raise SystemExit(
            "无法自动定位 Dataset105_* 目录。\n"
            "请确认已设置环境变量 TOTALSPINESEG_DATA，且其中存在 nnUNet/results/Dataset105_*。\n"
            f"当前 TOTALSPINESEG_DATA={base}"
        )
    return candidates[0].name


def _default_ckpt_paths(*, totalspineseg_data: Path, fold: int) -> tuple[Path, Path]:
    d105 = _resolve_dataset105_name(Path(totalspineseg_data))
    ckpt_dir = Path(totalspineseg_data) / "nnUNet" / "results" / d105 / "ldh_twostage" / "checkpoints"
    ckpt_a = ckpt_dir / f"ldh_stageA_fold_{int(fold)}.pth"
    ckpt_b = ckpt_dir / f"ldh_stageB_fold_{int(fold)}.pth"
    return ckpt_a, ckpt_b


def main() -> None:
    ap = argparse.ArgumentParser()
    # 最简用法：python scripts/infer_ldh.py <input_dir>
    ap.add_argument("input_dir_pos", type=Path, nargs="?", help="未知 MRI 输入文件夹（支持单个或多个病例）。")
    ap.add_argument("--input-dir", type=Path, default=None, help="同 positional input_dir；如同时提供，以该值为准。")
    ap.add_argument("--data-dir", type=Path, default=None, help="TotalSpineSeg data dir（默认读 $TOTALSPINESEG_DATA）。")
    ap.add_argument("--fold", type=int, default=0, help="使用哪一个 fold 的 checkpoint（默认 0）。")
    ap.add_argument("--ckpt-stagea", type=Path, default=None, help="Stage A checkpoint (.pth)，可覆盖默认路径。")
    ap.add_argument("--ckpt-stageb", type=Path, default=None, help="Stage B checkpoint (.pth)，可覆盖默认路径。")
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda/cpu（默认自动：优先 $TOTALSPINESEG_DEVICE；否则有 GPU 用 cuda，否则 cpu）。",
    )
    ap.add_argument("--out-name", type=str, default=None, help="输出目录名（默认自动带时间戳）。")
    ap.add_argument("--overwrite", action="store_true", default=False, help="允许覆盖已存在输出目录。")
    ap.add_argument("--no-init", action="store_true", help="不自动下载 nnUNet 权重。")
    ap.add_argument("--max-workers", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--max-workers-nnunet", type=int, default=1)
    ap.add_argument("--thresh-a", type=float, default=0.5)
    ap.add_argument("--thresh-b", type=float, default=0.5)
    ap.add_argument(
        "--no-clip-to-disc",
        action="store_true",
        default=False,
        help="默认会把 StageB 的预测硬限制在椎间盘区域内（抑制假阳性）。加上该参数可关闭此约束用于对比。",
    )
    ap.add_argument("--patch-a", type=int, default=96)
    ap.add_argument("--roi-b", type=int, default=48)
    # Default: restrict to the two most common lumbar LDH levels (L4/L5 and L5/S1)
    # - L4/L5 -> disc label 95
    # - L5/S  -> disc label 100
    # You can override this to evaluate other levels, e.g. --disc-labels 92 93 94 95 100
    ap.add_argument("--disc-labels", type=int, nargs="+", default=[95, 100])
    ap.add_argument("--n-boundary", type=int, default=6)
    ap.add_argument("--n-interior", type=int, default=6)
    ap.add_argument("--topk-b", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preview-orient", type=str, default="sag", choices=["sag", "cor", "ax"])
    ap.add_argument("--preview-sliceloc", type=float, default=0.5)
    ap.add_argument("--preview-alpha", type=float, default=0.55)
    args = ap.parse_args()

    input_dir = args.input_dir if args.input_dir is not None else args.input_dir_pos
    if input_dir is None:
        raise SystemExit("请提供输入目录：python scripts/infer_ldh.py /path/to/input_dir")
    input_dir = Path(input_dir).resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"输入必须是文件夹：{input_dir}")

    # Lazy imports for runtime dependencies (so --help works even if env missing deps)
    try:
        import numpy as np
        import torch
        import nibabel as nib
    except Exception as e:
        raise SystemExit(
            "运行推理需要安装依赖（numpy/torch/nibabel 等）。建议：\n"
            "  pip install -e .\n"
            "如需 nnUNet：\n"
            "  pip install -e '.[nnunetv2]'\n"
            f"原始错误: {type(e).__name__}: {e}"
        )

    from totalspineseg.init_inference import init_inference
    from totalspineseg.inference import inference as tss_inference
    from totalspineseg.ldh_twostage.disc_index import DiscIndexSpec, make_disc_index_map_from_step2_full_labels
    from totalspineseg.ldh_twostage.models import SmallUNet3D, StageADetector
    from totalspineseg.utils.transform_seg2image import transform_seg2image_mp
    from totalspineseg.utils.utils import ZIP_URLS

    # output directory under input_dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = args.out_name or f"infer_ldh_output_{ts}"
    out_dir = (input_dir / out_name).resolve()
    if out_dir.exists() and not args.overwrite:
        raise SystemExit(f"输出目录已存在：{out_dir}（如需覆盖请加 --overwrite）")
    _mkdir_clean(out_dir, overwrite=bool(args.overwrite))

    # Keep a copy of originals (named as <case>_0000.nii.gz) for resampling back
    original_raw_dir = out_dir / "original_raw"
    _prepare_original_raw(input_dir, original_raw_dir, overwrite=bool(args.overwrite))

    # TotalSpineSeg data dir
    if args.data_dir is not None:
        data_dir = Path(args.data_dir).resolve()
    elif "TOTALSPINESEG_DATA" in os.environ:
        data_dir = Path(os.environ["TOTALSPINESEG_DATA"]).resolve()
    else:
        data_dir = (Path(__file__).parent.parent / "data").resolve()

    # Resolve device default
    if args.device is None:
        env_dev = os.environ.get("TOTALSPINESEG_DEVICE", "").strip().lower()
        if env_dev in {"cuda", "cpu"}:
            dev_str = env_dev
        else:
            dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev_str = str(args.device).strip().lower()
        if dev_str not in {"cuda", "cpu"}:
            raise SystemExit(f"--device 仅支持 cuda/cpu，当前={args.device!r}")
    device = torch.device(dev_str)

    # Resolve checkpoints (default: under $TOTALSPINESEG_DATA/nnUNet/results/Dataset105_*/ldh_twostage/checkpoints/)
    ckpt_stagea = args.ckpt_stagea
    ckpt_stageb = args.ckpt_stageb
    if ckpt_stagea is None or ckpt_stageb is None:
        ckpt_a_def, ckpt_b_def = _default_ckpt_paths(totalspineseg_data=data_dir, fold=int(args.fold))
        if ckpt_stagea is None:
            ckpt_stagea = ckpt_a_def
        if ckpt_stageb is None:
            ckpt_stageb = ckpt_b_def
    ckpt_stagea = Path(ckpt_stagea).resolve()
    ckpt_stageb = Path(ckpt_stageb).resolve()
    if not ckpt_stagea.is_file() or not ckpt_stageb.is_file():
        raise SystemExit(
            "未找到 StageA/StageB checkpoint（自动解析失败）。\n"
            f"  期望 StageA: {ckpt_stagea}\n"
            f"  期望 StageB: {ckpt_stageb}\n"
            "请确认：\n"
            "  - 已设置 TOTALSPINESEG_DATA，且训练输出位于 nnUNet/results/Dataset105_*/ldh_twostage/checkpoints/\n"
            "或显式指定：\n"
            "  --ckpt-stagea /path/to/ldh_stageA_fold_0.pth --ckpt-stageb /path/to/ldh_stageB_fold_0.pth\n"
        )

    # Step1+Step2 inference (creates preview + step folders similar to original)
    default_release = list(ZIP_URLS.values())[0].split("/")[-2]
    if not args.no_init:
        init_inference(data_path=data_dir, dict_urls=ZIP_URLS, quiet=False)

    tss_inference(
        input_path=input_dir,
        output_path=out_dir,
        data_path=data_dir,
        default_release=default_release,
        output_iso=True,
        loc_path=None,
        suffix=[""],
        loc_suffix="",
        step1_only=False,
        keep_only=[""],  # keep all (so step1/step2 previews are generated)
        max_workers=int(args.max_workers),
        max_workers_nnunet=int(args.max_workers_nnunet),
        device=device,
        quiet=False,
    )

    # Re-organize preview images by stage
    _move_preview_by_stage(out_dir / "preview")

    # LDH output folders
    stagea_dir = out_dir / "ldh_stagea"
    stageb_iso_dir = out_dir / "ldh_stageb_iso"
    ldh_out_dir = out_dir / "ldh_output"
    preview_ldh_dir = out_dir / "preview" / "ldh"
    stagea_dir.mkdir(parents=True, exist_ok=True)
    stageb_iso_dir.mkdir(parents=True, exist_ok=True)
    ldh_out_dir.mkdir(parents=True, exist_ok=True)
    preview_ldh_dir.mkdir(parents=True, exist_ok=True)

    # Load Stage A/B models
    det = StageADetector(in_channels=3).to(device)
    det.load_state_dict(torch.load(ckpt_stagea, map_location="cpu")["model"])
    det.eval()

    seg = SmallUNet3D(in_channels=3).to(device)
    seg.load_state_dict(torch.load(ckpt_stageb, map_location="cpu")["model"])
    seg.eval()

    patch_a = (int(args.patch_a), int(args.patch_a), int(args.patch_a))
    roi_b = (int(args.roi_b), int(args.roi_b), int(args.roi_b))

    rng = np.random.RandomState(int(args.seed))

    # Iterate cases from preprocessed iso inputs
    iso_images = sorted((out_dir / "input").glob("*_0000.nii.gz"))
    if not iso_images:
        raise SystemExit(f"未在推理输出中找到 input/*_0000.nii.gz：{out_dir/'input'}")

    summary: Dict[str, object] = {
        "input_dir": str(input_dir),
        "output_dir": str(out_dir),
        "ckpt_stagea": str(ckpt_stagea),
        "ckpt_stageb": str(ckpt_stageb),
        "thresh_a": float(args.thresh_a),
        "thresh_b": float(args.thresh_b),
        "disc_labels": list(map(int, args.disc_labels)),
        "cases": [],
    }

    for img_path in iso_images:
        case_id = img_path.name.replace("_0000.nii.gz", "")
        step2_path = out_dir / "step2_output" / f"{case_id}.nii.gz"
        if not step2_path.exists():
            # If step2 missing, skip
            continue

        img_nii = nib.load(str(img_path))
        img = np.asanyarray(img_nii.dataobj).astype(np.float32)
        step2_nii = nib.load(str(step2_path))
        step2 = np.asanyarray(step2_nii.dataobj).astype(np.int32)

        spec = DiscIndexSpec.default_lumbar()
        disc_index_nii = make_disc_index_map_from_step2_full_labels(step2_nii, spec=spec, normalize=True)
        disc_index_map = np.asanyarray(disc_index_nii.dataobj).astype(np.float32)

        ldh_pred_iso = np.zeros(step2.shape, dtype=np.uint8)
        decisions: List[DiscDecision] = []
        debug_top_centers: Dict[str, List[Dict[str, object]]] = {}

        for disc_label in list(map(int, args.disc_labels)):
            disc_region = _disc_region_from_step2(step2, disc_label)
            if disc_region.sum() == 0:
                decisions.append(
                    DiscDecision(
                        disc_label=int(disc_label),
                        prob=0.0,
                        positive=False,
                        best_center_zyx=(0, 0, 0),
                        best_center_rank=-1,
                    )
                )
                continue

            centers = _sample_candidate_centers(
                disc_region,
                rng=rng,
                n_boundary=int(args.n_boundary),
                n_interior=int(args.n_interior),
            )

            # Stage A over candidate centers
            probs: List[Tuple[Tuple[int, int, int], float]] = []
            best_p = -1.0
            best_center = centers[0]
            best_rank = 0

            with torch.no_grad():
                for i, c in enumerate(centers):
                    img_p, _, _, _ = _crop_with_padding(img, c, patch_a, pad_value=0.0)
                    disc_p, _, _, _ = _crop_with_padding(disc_region.astype(np.float32), c, patch_a, pad_value=0.0)
                    idx_p, _, _, _ = _crop_with_padding(disc_index_map.astype(np.float32), c, patch_a, pad_value=0.0)
                    x = np.stack([img_p, disc_p, idx_p], axis=0)[None]
                    logits = det(torch.from_numpy(x).to(device))
                    p = float(torch.sigmoid(logits).item())
                    probs.append((c, p))
                    if p > best_p:
                        best_p = p
                        best_center = c
                        best_rank = i

            probs_sorted = sorted(probs, key=lambda t: t[1], reverse=True)
            debug_top_centers[str(disc_label)] = [
                {"center_zyx": [int(c[0]), int(c[1]), int(c[2])], "prob": float(p)} for c, p in probs_sorted[:12]
            ]

            positive = bool(best_p >= float(args.thresh_a))
            decisions.append(
                DiscDecision(
                    disc_label=int(disc_label),
                    prob=float(best_p),
                    positive=positive,
                    best_center_zyx=tuple(map(int, best_center)),
                    best_center_rank=int(best_rank),
                )
            )
            if not positive:
                continue

            # Stage B on top-K centers (default 1)
            with torch.no_grad():
                for k in range(int(max(1, args.topk_b))):
                    if k >= len(probs_sorted):
                        break
                    center_k = probs_sorted[k][0]
                    img_roi, start, end, pad_before = _crop_with_padding(img, center_k, roi_b, pad_value=0.0)
                    disc_roi, _, _, _ = _crop_with_padding(disc_region.astype(np.float32), center_k, roi_b, pad_value=0.0)
                    idx_roi, _, _, _ = _crop_with_padding(disc_index_map.astype(np.float32), center_k, roi_b, pad_value=0.0)
                    x = np.stack([img_roi, disc_roi, idx_roi], axis=0)[None]
                    mask_logits, _sdm = seg(torch.from_numpy(x).to(device))
                    roi_pred = (torch.sigmoid(mask_logits) >= float(args.thresh_b)).float().cpu().numpy()[0, 0].astype(np.uint8)

                    roi_unpadded = _uncrop_remove_padding(roi_pred, start, end, pad_before)
                    # Hard constraint: clip prediction to disc region (strong anatomy prior).
                    # This suppresses spurious positives that spill outside the disc ROI when running on full spine volumes.
                    if not bool(args.no_clip_to_disc):
                        disc_roi_u8 = (disc_roi >= 0.5).astype(np.uint8)
                        disc_roi_unpadded = _uncrop_remove_padding(disc_roi_u8, start, end, pad_before)
                        roi_unpadded = (roi_unpadded & disc_roi_unpadded).astype(np.uint8)
                    z0c, y0c, x0c = start
                    z1c, y1c, x1c = end
                    ldh_pred_iso[z0c:z1c, y0c:y1c, x0c:x1c] = np.maximum(
                        ldh_pred_iso[z0c:z1c, y0c:y1c, x0c:x1c],
                        roi_unpadded,
                    )

        # Save per-case outputs
        iso_mask_path = stageb_iso_dir / f"{case_id}.nii.gz"
        iso_nii = nib.Nifti1Image(ldh_pred_iso.astype(np.uint8), img_nii.affine, img_nii.header)
        iso_nii.set_data_dtype(np.uint8)
        iso_nii.set_qform(iso_nii.affine)
        iso_nii.set_sform(iso_nii.affine)
        nib.save(iso_nii, str(iso_mask_path))

        stagea_report_path = stagea_dir / f"{case_id}.json"
        stagea_report = {
            "case_id": case_id,
            "thresh_a": float(args.thresh_a),
            "disc_labels": list(map(int, args.disc_labels)),
            "decisions": [asdict(d) for d in decisions],
            "debug_top_centers": debug_top_centers,
        }
        with open(stagea_report_path, "w", encoding="utf-8") as f:
            json.dump(stagea_report, f, ensure_ascii=False, indent=2)

        # LDH preview (red overlay) in iso space (same slice logic as preview_jpg)
        out_jpg = preview_ldh_dir / f"{case_id}_{args.preview_orient}_{args.preview_sliceloc}_ldh.jpg"
        _save_ldh_preview_red(
            image_path=img_path,
            seg_path=iso_mask_path,
            out_jpg=out_jpg,
            orient=str(args.preview_orient),
            sliceloc=float(args.preview_sliceloc),
            alpha=float(args.preview_alpha),
        )

        summary["cases"].append(
            {
                "case_id": case_id,
                "image_iso": str(img_path),
                "step2_iso": str(step2_path),
                "ldh_iso": str(iso_mask_path),
                "stagea_json": str(stagea_report_path),
                "preview_ldh": str(out_jpg),
            }
        )

    # Resample iso LDH masks back to original input space (batch)
    transform_seg2image_mp(
        images_path=original_raw_dir,
        segs_path=stageb_iso_dir,
        output_segs_path=ldh_out_dir,
        prefix="",
        image_suffix="_0000",
        seg_suffix="",
        output_seg_suffix="",
        interpolation="nearest",
        overwrite=True,
        max_workers=1,
        quiet=False,
    )

    # Final summary json
    with open(out_dir / "ldh_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()


