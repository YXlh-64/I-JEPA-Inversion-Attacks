import torch
import torch.nn.functional as F


def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """Total variation regularizer like in the notebook."""
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w


def reconstruction_loss(outputs: torch.Tensor, targets: torch.Tensor, tv_weight: float = 0.0) -> torch.Tensor:
    """L1 reconstruction with optional TV penalty to mirror notebook behavior."""
    l1 = F.l1_loss(outputs, targets)
    if tv_weight > 0:
        return l1 + tv_weight * total_variation_loss(outputs)
    return l1

