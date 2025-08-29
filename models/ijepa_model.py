import torch
import torch.nn as nn
import math
from typing import List
from PIL import Image
from transformers import AutoProcessor, AutoModel


class IJepaTokens(nn.Module):
    """Returns spatial tokens (B, 16, 16, 1280) for a batch of PIL images."""
    def __init__(self, model_id="facebook/ijepa_vith14_1k", device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.backbone = AutoModel.from_pretrained(
            model_id, torch_dtype="auto", attn_implementation="sdpa"
        ).to(self.device).eval()

    @torch.no_grad()
    def forward(self, pil_list: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=pil_list, return_tensors="pt").to(self.device)
        outputs = self.backbone(**inputs)
        # outputs.last_hidden_state: (B, N, 1280), expect N=256 => 16x16
        h = outputs.last_hidden_state
        B, N, C = h.shape
        S = int(math.sqrt(N))
        h = h.reshape(B, S, S, C)  # (B, 16, 16, 1280)
        return h