import torch
from typing import Dict

try:
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except Exception:
        from torchmetrics.image import LearnedPerceptualImagePatchSimilarity  # older versions
except Exception:
    PeakSignalNoiseRatio = None
    StructuralSimilarityIndexMeasure = None
    LearnedPerceptualImagePatchSimilarity = None


class ImageMetrics:
    def __init__(self, device=None):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.psnr = PeakSignalNoiseRatio().to(self.device) if PeakSignalNoiseRatio else None
        self.ssim = StructuralSimilarityIndexMeasure().to(self.device) if StructuralSimilarityIndexMeasure else None
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(self.device) if LearnedPerceptualImagePatchSimilarity else None

    @torch.no_grad()
    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        out = {}
        if self.psnr:
            out['psnr'] = float(self.psnr(pred, target).item())
        if self.ssim:
            out['ssim'] = float(self.ssim(pred, target).item())
        if self.lpips:
            out['lpips'] = float(self.lpips(pred, target).item())
        return out

