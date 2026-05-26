import torch
import torch.nn as nn
from typing import Optional
import warnings

class TrainOnlyEmbeddingModulation(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mode: str = "scale",  # "scale" | "bias"
        scale: float = 3.0,
        bias_scale: float = 1.0,
        learnable_bias: bool = False,
        bias_init: str = "zeros",  # "zeros" | "normal" | "unit" | "ones"
    ):
        super().__init__()
        assert mode in ("scale", "bias"), "mode must be 'scale' or 'bias'"
        self.mode = mode
        self.scale = float(scale)
        self.bias_scale = float(bias_scale)
        self._warned_zero_bias = False

        if mode == "bias":
            if learnable_bias:
                if bias_init == "normal":
                    bias_vec = torch.randn(hidden_size) * 0.02
                elif bias_init == "unit":
                    v = torch.randn(hidden_size)
                    v = v / (v.norm(p=2) + 1e-12)
                    bias_vec = v
                elif bias_init == "ones":
                    bias_vec = torch.ones(hidden_size)
                else:  # "zeros"
                    bias_vec = torch.zeros(hidden_size)
                self.bias_vec = nn.Parameter(bias_vec)
            else:
                if bias_init == "normal":
                    vec = torch.randn(hidden_size) * 0.02
                elif bias_init == "unit":
                    v = torch.randn(hidden_size)
                    v = v / (v.norm(p=2) + 1e-12)
                    vec = v
                elif bias_init == "ones":
                    vec = torch.ones(hidden_size)
                else:  # "zeros"
                    vec = torch.zeros(hidden_size)
                self.register_buffer("bias_vec", vec, persistent=True)
        else:
            self.register_parameter("bias_vec", None)

    @torch.no_grad()
    def _sanitize_mask(self, mask: torch.Tensor, target_shape, device, dtype) -> torch.Tensor:
        # Convert mask to float and expand to [B, T, 1]; validate shape
        if mask.dtype != torch.float32 and mask.dtype != torch.float16 and mask.dtype != torch.bfloat16:
            mask = mask.to(torch.float32)
        if mask.dim() != 2:
            raise ValueError("sensitive_mask must be 2D [batch, seq_len]")
        b, t = target_shape[:2]
        if mask.size(0) != b or mask.size(1) != t:
            raise ValueError(f"sensitive_mask shape mismatch: got {mask.shape}, expected {(b, t)}")
        mask = mask.unsqueeze(-1).to(device=device, dtype=dtype)  # [B, T, 1]
        return mask

    def forward(
        self,
        emb: torch.Tensor,            # [B, T, H]
        sensitive_mask: Optional[torch.Tensor] = None,
        *,
        training: bool,
    ) -> torch.Tensor:
        if not training or sensitive_mask is None:
            return emb
        mask = self._sanitize_mask(sensitive_mask, emb.shape, emb.device, emb.dtype)
        if self.mode == "scale":
            return emb * (1.0 + (self.scale - 1.0) * mask)
        else:  # bias
            # Optional warning when bias vector is zero but bias_scale > 0
            if (not self._warned_zero_bias) and self.bias_scale > 0:
                vec = self.bias_vec.data if isinstance(self.bias_vec, nn.Parameter) else self.bias_vec
                if torch.count_nonzero(vec).item() == 0:
                    warnings.warn(
                        "TrainOnlyEmbeddingModulation: bias_vec is all zeros while bias_scale>0; bias-mode will be no-op.",
                        RuntimeWarning,
                    )
                    self._warned_zero_bias = True
            bias_vec = self.bias_vec.to(device=emb.device, dtype=emb.dtype)  # [H]
            return emb + self.bias_scale * mask * bias_vec.view(1, 1, -1)