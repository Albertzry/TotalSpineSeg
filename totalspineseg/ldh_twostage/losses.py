from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Binary focal loss on logits.
    targets: float tensor in {0,1}
    """
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, prob, 1 - prob)
    loss = ((1 - pt) ** gamma) * bce
    if alpha is not None:
        a = torch.where(targets > 0.5, torch.tensor(alpha, device=logits.device), torch.tensor(1 - alpha, device=logits.device))
        loss = a * loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def focal_tversky_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.7,
    beta: float = 0.3,
    gamma: float = 0.75,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """
    probs: sigmoid output in [0,1]
    targets: {0,1}
    """
    probs = probs.float()
    targets = targets.float()
    tp = (probs * targets).sum()
    fp = (probs * (1 - targets)).sum()
    fn = ((1 - probs) * targets).sum()
    ti = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return torch.pow(1 - ti, gamma)


def boundary_loss_from_sdm(
    probs: torch.Tensor,
    sdm: torch.Tensor,
) -> torch.Tensor:
    """
    Boundary loss as in Kervadec et al. using a signed distance map.
    sdm convention: negative inside, positive outside.
    """
    probs = probs.float()
    sdm = sdm.float()
    return (probs * sdm).mean()


@dataclass(frozen=True)
class StageBLossWeights:
    boundary: float = 0.5
    sdm_l1: float = 0.2


def stage_b_total_loss(
    mask_logits: torch.Tensor,
    sdm_pred: torch.Tensor,
    mask_gt: torch.Tensor,
    sdm_gt: torch.Tensor,
    weights: StageBLossWeights = StageBLossWeights(),
) -> torch.Tensor:
    probs = torch.sigmoid(mask_logits)
    ft = focal_tversky_loss(probs, mask_gt, alpha=0.7, beta=0.3, gamma=0.75)
    bnd = boundary_loss_from_sdm(probs, sdm_gt)
    l1 = F.l1_loss(sdm_pred, sdm_gt)
    return ft + weights.boundary * bnd + weights.sdm_l1 * l1


