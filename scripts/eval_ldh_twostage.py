#!/usr/bin/env python3
"""
Evaluation for LDH two-stage pipeline.

Reports (REQUIRED):
  - Disc-level LDH recall (Stage A, aggregated per disc)
  - Lesion-wise detection rate (same aggregation, recall-focused)
  - Dice (Stage B, secondary)
  - Average surface distance (Stage B)
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# Add repository root to Python path for imports
_script_dir = Path(__file__).parent.resolve()
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import torch

from totalspineseg.ldh_twostage.losses import focal_loss_with_logits
from totalspineseg.ldh_twostage.metrics import DetectionReport, average_surface_distance, dice
from totalspineseg.ldh_twostage.models import SmallUNet3D, StageADetector


def load_npz(path: Path):
    with np.load(str(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stagea-patches-dir", type=Path, required=True)
    ap.add_argument("--stageb-rois-dir", type=Path, required=True)
    ap.add_argument("--ckpt-stagea", type=Path, required=True)
    ap.add_argument("--ckpt-stageb", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device(args.device)

    # ---------------- Stage A (disc-level) ----------------
    det = StageADetector(in_channels=3).to(device)
    det.load_state_dict(torch.load(args.ckpt_stagea, map_location="cpu")["model"])
    det.eval()

    # aggregate per disc: key=(sample_id, disc_label)
    probs_by_disc = defaultdict(list)
    gt_by_disc = {}

    for p in sorted(args.stagea_patches_dir.glob("*.npz")):
        d = load_npz(p)
        sid = str(d.get("sample_id", b"unknown").astype("S").tobytes().decode(errors="ignore")) if "sample_id" in d else "unknown"
        disc_label = int(d.get("disc_label", -1))
        has_ldh = int(d.get("has_ldh", 0))

        x = np.stack(
            [d["image"].astype(np.float32), d["disc_mask"].astype(np.float32), d["disc_index"].astype(np.float32)],
            axis=0,
        )[None]
        with torch.no_grad():
            prob = torch.sigmoid(det(torch.from_numpy(x).to(device))).item()
        probs_by_disc[(sid, disc_label)].append(prob)
        gt_by_disc[(sid, disc_label)] = has_ldh

    rep = DetectionReport()
    for k, probs in probs_by_disc.items():
        pmax = float(np.max(np.array(probs, dtype=np.float32)))
        pred = 1 if pmax >= args.thresh else 0
        gt = int(gt_by_disc.get(k, 0))
        if pred == 1 and gt == 1:
            rep.tp += 1
        elif pred == 1 and gt == 0:
            rep.fp += 1
        elif pred == 0 and gt == 0:
            rep.tn += 1
        else:
            rep.fn += 1

    print("Stage A (disc-level detection):")
    print(f"  disc-level recall: {rep.recall:.3f}")
    print(f"  precision (secondary): {rep.precision:.3f}")
    print(f"  counts: tp={rep.tp} fp={rep.fp} tn={rep.tn} fn={rep.fn}")
    print(f"  lesion-wise detection rate (proxy): {rep.recall:.3f}")

    # ---------------- Stage B (ROI segmentation) ----------------
    seg = SmallUNet3D(in_channels=3).to(device)
    seg.load_state_dict(torch.load(args.ckpt_stageb, map_location="cpu")["model"])
    seg.eval()

    dices = []
    asds = []
    for p in sorted(args.stageb_rois_dir.glob("*.npz")):
        d = load_npz(p)
        if int(d.get("has_ldh", 0)) != 1:
            continue
        x = np.stack(
            [d["image"].astype(np.float32), d["disc_mask"].astype(np.float32), d["disc_index"].astype(np.float32)],
            axis=0,
        )[None]
        gt = d["ldh_mask"].astype(np.float32)
        with torch.no_grad():
            mask_logits, _sdm_pred = seg(torch.from_numpy(x).to(device))
            pred = (torch.sigmoid(mask_logits) >= args.thresh).float().cpu().numpy()[0, 0]
        dices.append(dice(pred, gt))
        asds.append(average_surface_distance(pred, gt))

    print("Stage B (ROI fine segmentation, positives only):")
    print(f"  Dice (secondary): {float(np.nanmean(np.array(dices))):.3f}")
    print(f"  ASD: {float(np.nanmean(np.array(asds))):.3f}")


if __name__ == "__main__":
    main()


