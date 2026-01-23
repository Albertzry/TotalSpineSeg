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
import tempfile
import multiprocessing as mp
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from tqdm import tqdm

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


def _default_ckpt_paths(*, totalspineseg_data: Path, fold: int) -> tuple[Path, Path | None]:
    d105 = _resolve_dataset105_name(Path(totalspineseg_data))
    ckpt_dir = Path(totalspineseg_data) / "nnUNet" / "results" / d105 / "ldh_twostage" / "checkpoints"
    ckpt_a = ckpt_dir / f"ldh_stageA_fold_{int(fold)}.pth"
    # Stage B now uses nnUNet, not a .pth checkpoint
    ckpt_b = None
    return ckpt_a, ckpt_b


def _default_model_folder_stageb(*, totalspineseg_data: Path) -> Path:
    """Resolve default Stage B nnUNet model folder (Dataset 107)."""
    candidates = sorted((totalspineseg_data / "nnUNet" / "results").glob("Dataset107_*"))
    if not candidates:
        raise SystemExit(
            f"未找到 Dataset107_* 目录在 {totalspineseg_data/'nnUNet'/'results'}。\n"
            "请确认已训练 Stage B (Dataset 107) 或手动指定 --model-folder-stageb。"
        )
    d107_name = candidates[0].name
    model_folder = totalspineseg_data / "nnUNet" / "results" / d107_name / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    if not model_folder.exists():
        raise SystemExit(
            f"未找到 Stage B 模型文件夹：{model_folder}\n"
            "请确认已训练 Stage B (Dataset 107) 或手动指定 --model-folder-stageb。"
        )
    return model_folder


def main() -> None:
    ap = argparse.ArgumentParser(
        description="LDH two-stage inference (Stage A detection + Stage B ROI segmentation)",
        epilog="""
使用示例：
  # 方式1: 使用显式参数（推荐）
  python scripts/infer_ldh.py --input-dir /path/to/input --output-dir /path/to/output --data-dir /path/to/TotalSpineSegData
  
  # 方式2: 使用位置参数（向后兼容）
  python scripts/infer_ldh.py /path/to/input_dir
  
  # 方式3: 只指定输入，输出在输入目录下创建
  python scripts/infer_ldh.py --input-dir /path/to/input --data-dir /path/to/TotalSpineSegData

参数优先级：
  - --input-dir 优先于位置参数 input_dir_pos
  - --output-dir 优先于 --out-name（如果未指定 --output-dir，则在输入目录下创建名为 --out-name 的文件夹）
  - --data-dir 优先于环境变量 TOTALSPINESEG_DATA
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # 最简用法：python scripts/infer_ldh.py <input_dir>
    ap.add_argument("input_dir_pos", type=Path, nargs="?", help="未知 MRI 输入文件夹（支持单个或多个病例）。如果提供了--input-dir，此参数将被忽略。")
    ap.add_argument("--input-dir", type=Path, default=None, help="输入文件夹路径（包含 .nii.gz 或 .nii 图像文件）。优先于位置参数。")
    ap.add_argument("--output-dir", type=Path, default=None, help="输出文件夹路径（用于保存推理结果）。如果未指定，将在输入目录下创建名为 --out-name 的文件夹（默认：infer_output）。")
    ap.add_argument("--data-dir", type=Path, default=None, help="TotalSpineSeg 数据目录路径（用于存储模型权重和 nnUNet 数据）。优先于环境变量 $TOTALSPINESEG_DATA。")
    ap.add_argument("--fold", type=int, default=0, help="使用哪一个 fold 的 checkpoint（默认 0）。")
    ap.add_argument("--ckpt-stagea", type=Path, default=None, help="Stage A checkpoint (.pth)，可覆盖默认路径。")
    ap.add_argument("--ckpt-stageb", type=Path, default=None, help="DEPRECATED: Stage B 现在使用 nnUNet (Dataset 107)。请使用 --model-folder-stageb。")
    ap.add_argument(
        "--model-folder-stageb",
        type=Path,
        default=None,
        help="Stage B 的 nnUNet 模型文件夹（Dataset 107）。默认：从 $TOTALSPINESEG_DATA/nnUNet/results/Dataset107_*/nnUNetTrainer__nnUNetPlans__3d_fullres/ 自动解析。",
    )
    ap.add_argument("--checkpoint-stageb", type=str, default="checkpoint_best.pth", help="Stage B 的 checkpoint 名称（默认：checkpoint_best.pth）。")
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda/cpu（默认自动：优先 $TOTALSPINESEG_DEVICE；否则有 GPU 用 cuda，否则 cpu）。",
    )
    ap.add_argument("--out-name", type=str, default=None, help="输出目录名（默认：infer_output）。")
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
    ap.add_argument("--roi-b", type=int, default=64, help="Stage B ROI size (cube), default 64 (should match Dataset107 ROI size)")
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

    # 确定输入目录：优先使用 --input-dir，否则使用位置参数
    input_dir = args.input_dir if args.input_dir is not None else args.input_dir_pos
    if input_dir is None:
        raise SystemExit("请提供输入目录：python scripts/infer_ldh.py --input-dir /path/to/input_dir 或 python scripts/infer_ldh.py /path/to/input_dir")
    input_dir = Path(input_dir).resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"输入必须是文件夹：{input_dir}")

    # 确定输出目录：优先使用 --output-dir，否则在输入目录下创建
    if args.output_dir is not None:
        out_dir = Path(args.output_dir).resolve()
    else:
        out_name = args.out_name or "infer_output"
        out_dir = (input_dir / out_name).resolve()
    
    # 确保输出目录的父目录存在
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    
    if out_dir.exists() and not args.overwrite:
        raise SystemExit(f"输出目录已存在：{out_dir}（如需覆盖请加 --overwrite）")
    _mkdir_clean(out_dir, overwrite=bool(args.overwrite))

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
    from totalspineseg.ldh_twostage.models import StageADetectorV2
    from totalspineseg.utils.predict_nnunet import predict_nnunet
    from totalspineseg.utils.transform_seg2image import transform_seg2image_mp
    from totalspineseg.utils.utils import ZIP_URLS

    # Keep a copy of originals (named as <case>_0000.nii.gz) for resampling back
    original_raw_dir = out_dir / "original_raw"
    _prepare_original_raw(input_dir, original_raw_dir, overwrite=bool(args.overwrite))

    # TotalSpineSeg data dir：优先使用 --data-dir，否则使用环境变量，最后回退到默认路径
    if args.data_dir is not None:
        data_dir = Path(args.data_dir).resolve()
    elif "TOTALSPINESEG_DATA" in os.environ:
        data_dir = Path(os.environ["TOTALSPINESEG_DATA"]).resolve()
    else:
        data_dir = (Path(__file__).parent.parent / "data").resolve()
    
    if not data_dir.exists():
        raise SystemExit(
            f"TotalSpineSeg 数据目录不存在：{data_dir}\n"
            "请使用以下方式之一指定数据目录：\n"
            "  1. 使用 --data-dir 参数：--data-dir /path/to/TotalSpineSegData\n"
            "  2. 设置环境变量：export TOTALSPINESEG_DATA=/path/to/TotalSpineSegData"
        )

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

    # Resolve Stage A checkpoint
    ckpt_stagea = args.ckpt_stagea
    if ckpt_stagea is None:
        ckpt_a_def, _ = _default_ckpt_paths(totalspineseg_data=data_dir, fold=int(args.fold))
        ckpt_stagea = ckpt_a_def
    ckpt_stagea = Path(ckpt_stagea).resolve()
    if not ckpt_stagea.is_file():
        raise SystemExit(
            "未找到 StageA checkpoint（自动解析失败）。\n"
            f"  期望 StageA: {ckpt_stagea}\n"
            "请确认：\n"
            "  - 已设置 TOTALSPINESEG_DATA，且训练输出位于 nnUNet/results/Dataset105_*/ldh_twostage/checkpoints/\n"
            "或显式指定：\n"
            "  --ckpt-stagea /path/to/ldh_stageA_fold_0.pth\n"
        )
    
    # Resolve Stage B model folder (nnUNet Dataset 107)
    model_folder_stageb = args.model_folder_stageb
    if model_folder_stageb is None:
        model_folder_stageb = _default_model_folder_stageb(totalspineseg_data=data_dir)
    model_folder_stageb = Path(model_folder_stageb).resolve()
    if not model_folder_stageb.is_dir():
        raise SystemExit(
            "未找到 StageB 模型文件夹（自动解析失败）。\n"
            f"  期望 StageB: {model_folder_stageb}\n"
            "请确认：\n"
            "  - 已设置 TOTALSPINESEG_DATA，且训练输出位于 nnUNet/results/Dataset107_*/nnUNetTrainer__nnUNetPlans__3d_fullres/\n"
            "或显式指定：\n"
            "  --model-folder-stageb /path/to/nnUNetTrainer__nnUNetPlans__3d_fullres\n"
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

    # Load Stage A model
    det = StageADetectorV2(in_channels=3).to(device)
    ckpt = torch.load(ckpt_stagea, map_location="cpu")
    state = ckpt.get("model", ckpt)
    det.load_state_dict(state, strict=True)
    det.eval()
    
    # Stage B uses nnUNet, will be initialized during inference

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
        "model_folder_stageb": str(model_folder_stageb),
        "checkpoint_stageb": str(args.checkpoint_stageb),
        "thresh_a": float(args.thresh_a),
        "thresh_b": float(args.thresh_b),
        "disc_labels": list(map(int, args.disc_labels)),
        "cases": [],
    }

    # Stage A: Run detection for all cases and collect positive ROIs for Stage B
    case_data: Dict[str, Dict] = {}
    all_rois: List[Dict] = []  # List of ROI metadata for batch Stage B inference
    
    for img_path in iso_images:
        case_id = img_path.name.replace("_0000.nii.gz", "")
        step2_path = out_dir / "step2_output" / f"{case_id}.nii.gz"
        if not step2_path.exists():
            continue

        img_nii = nib.load(str(img_path))
        img = np.asanyarray(img_nii.dataobj).astype(np.float32)
        step2_nii = nib.load(str(step2_path))
        step2 = np.asanyarray(step2_nii.dataobj).astype(np.int32)

        spec = DiscIndexSpec.default_lumbar()
        disc_index_nii = make_disc_index_map_from_step2_full_labels(step2_nii, spec=spec, normalize=True)
        disc_index_map = np.asanyarray(disc_index_nii.dataobj).astype(np.float32)

        decisions: List[DiscDecision] = []
        debug_top_centers: Dict[str, List[Dict[str, object]]] = {}
        case_rois: List[Dict] = []  # ROIs for this case

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

            # Collect ROIs for Stage B (top-K centers)
            for k in range(int(max(1, args.topk_b))):
                if k >= len(probs_sorted):
                    break
                center_k = probs_sorted[k][0]
                img_roi, start, end, pad_before = _crop_with_padding(img, center_k, roi_b, pad_value=0.0)
                disc_roi, _, _, _ = _crop_with_padding(disc_region.astype(np.float32), center_k, roi_b, pad_value=0.0)
                idx_roi, _, _, _ = _crop_with_padding(disc_index_map.astype(np.float32), center_k, roi_b, pad_value=0.0)
                
                roi_id = f"{case_id}_disc{disc_label}_k{k}"
                case_rois.append({
                    "roi_id": roi_id,
                    "case_id": case_id,
                    "disc_label": disc_label,
                    "center": center_k,
                    "start": start,
                    "end": end,
                    "pad_before": pad_before,
                    "img_roi": img_roi,
                    "disc_roi": disc_roi,
                    "idx_roi": idx_roi,
                })
                all_rois.append(case_rois[-1])

        case_data[case_id] = {
            "img_nii": img_nii,
            "img": img,
            "step2": step2,
            "decisions": decisions,
            "debug_top_centers": debug_top_centers,
            "rois": case_rois,
        }

    # Stage B: Batch nnUNet inference for all collected ROIs
    if len(all_rois) > 0:
        with tempfile.TemporaryDirectory(prefix="infer_ldh_stageb_") as tmpdir:
            tmp_images = Path(tmpdir) / "images"
            tmp_output = Path(tmpdir) / "output"
            tmp_images.mkdir(parents=True, exist_ok=True)
            tmp_output.mkdir(parents=True, exist_ok=True)
            
            # Prepare 3-channel NIfTI files
            print(f"准备 {len(all_rois)} 个 ROI 用于 Stage B 推理...")
            for roi_info in tqdm(all_rois, desc="准备 ROI", unit="roi"):
                roi_id = roi_info["roi_id"]
                affine = np.eye(4)
                nib.save(nib.Nifti1Image(roi_info["img_roi"].astype(np.float32), affine), tmp_images / f"{roi_id}_0000.nii.gz")
                nib.save(nib.Nifti1Image(roi_info["disc_roi"].astype(np.float32), affine), tmp_images / f"{roi_id}_0001.nii.gz")
                nib.save(nib.Nifti1Image(roi_info["idx_roi"].astype(np.float32), affine), tmp_images / f"{roi_id}_0002.nii.gz")
            
            # Run nnUNet inference
            print("运行 Stage B nnUNet 推理...")
            predict_nnunet(
                model_folder=str(model_folder_stageb),
                images_dir=str(tmp_images),
                output_dir=str(tmp_output),
                device=device,
                folds=(args.fold,),
                checkpoint=args.checkpoint_stageb,
                npp=1,
                nps=1,
                disable_tta=False,
                verbose=False,
                disable_progress_bar=False,
            )
            
            # Load predictions and merge back to full images
            print("合并 Stage B 预测结果...")
            for roi_info in tqdm(all_rois, desc="合并结果", unit="roi"):
                roi_id = roi_info["roi_id"]
                case_id = roi_info["case_id"]
                pred_path = tmp_output / f"{roi_id}.nii.gz"
                if not pred_path.exists():
                    continue
                
                pred_nii = nib.load(str(pred_path))
                pred = np.asanyarray(pred_nii.dataobj).astype(np.float32)
                # nnUNet outputs class labels (0=background, 1=LDH), convert to binary mask
                roi_pred = (pred >= float(args.thresh_b)).astype(np.uint8)
                
                # Unpad and merge
                start = roi_info["start"]
                end = roi_info["end"]
                pad_before = roi_info["pad_before"]
                roi_unpadded = _uncrop_remove_padding(roi_pred, start, end, pad_before)
                
                # Hard constraint: clip prediction to disc region
                if not bool(args.no_clip_to_disc):
                    disc_roi_u8 = (roi_info["disc_roi"] >= 0.5).astype(np.uint8)
                    disc_roi_unpadded = _uncrop_remove_padding(disc_roi_u8, start, end, pad_before)
                    roi_unpadded = (roi_unpadded & disc_roi_unpadded).astype(np.uint8)
                
                # Merge into case's full prediction
                if case_id not in case_data:
                    continue
                z0c, y0c, x0c = start
                z1c, y1c, x1c = end
                if "ldh_pred_iso" not in case_data[case_id]:
                    case_data[case_id]["ldh_pred_iso"] = np.zeros(case_data[case_id]["step2"].shape, dtype=np.uint8)
                case_data[case_id]["ldh_pred_iso"][z0c:z1c, y0c:y1c, x0c:x1c] = np.maximum(
                    case_data[case_id]["ldh_pred_iso"][z0c:z1c, y0c:y1c, x0c:x1c],
                    roi_unpadded,
                )

    # Save per-case outputs
    for case_id, data in case_data.items():
        ldh_pred_iso = data.get("ldh_pred_iso", np.zeros(data["step2"].shape, dtype=np.uint8))
        iso_mask_path = stageb_iso_dir / f"{case_id}.nii.gz"
        iso_nii = nib.Nifti1Image(ldh_pred_iso.astype(np.uint8), data["img_nii"].affine, data["img_nii"].header)
        iso_nii.set_data_dtype(np.uint8)
        iso_nii.set_qform(iso_nii.affine)
        iso_nii.set_sform(iso_nii.affine)
        nib.save(iso_nii, str(iso_mask_path))

        stagea_report_path = stagea_dir / f"{case_id}.json"
        stagea_report = {
            "case_id": case_id,
            "thresh_a": float(args.thresh_a),
            "disc_labels": list(map(int, args.disc_labels)),
            "decisions": [asdict(d) for d in data["decisions"]],
            "debug_top_centers": data["debug_top_centers"],
        }
        with open(stagea_report_path, "w", encoding="utf-8") as f:
            json.dump(stagea_report, f, ensure_ascii=False, indent=2)

        # LDH preview (red overlay) in iso space (same slice logic as preview_jpg)
        img_path = out_dir / "input" / f"{case_id}_0000.nii.gz"
        out_jpg = preview_ldh_dir / f"{case_id}_{args.preview_orient}_{args.preview_sliceloc}_ldh.jpg"
        _save_ldh_preview_red(
            image_path=img_path,
            seg_path=iso_mask_path,
            out_jpg=out_jpg,
            orient=str(args.preview_orient),
            sliceloc=float(args.preview_sliceloc),
            alpha=float(args.preview_alpha),
        )

        step2_path = out_dir / "step2_output" / f"{case_id}.nii.gz"
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


