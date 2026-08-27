from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ExperimentConfig


TARGET_MODULES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.out_proj",
    "fc1",
    "fc2",
)


def orthogonal_factors(
    out_features: int, in_features: int, rank: int, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create inexpensive orthonormal factors for the P-E-Q parameterization."""

    p, _ = torch.linalg.qr(torch.randn(out_features, rank, dtype=torch.float32))
    q, _ = torch.linalg.qr(torch.randn(in_features, rank, dtype=torch.float32))
    return p.to(dtype), q.T.to(dtype)


class SVDLoRALinear(nn.Module):
    """Frozen linear layer plus a masked P-E-Q low-rank update."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.max_rank = min(rank, self.in_features, self.out_features)
        self.scaling = alpha / max(self.max_rank, 1)
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        self.bias = None
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)

        p_init, q_init = orthogonal_factors(
            self.out_features, self.in_features, self.max_rank, base.weight.dtype
        )

        self.lora_P = nn.Parameter(p_init.to(base.weight.dtype))
        # Zero E preserves the pretrained model at initialization.
        self.lora_E = nn.Parameter(torch.zeros(self.max_rank, dtype=base.weight.dtype))
        self.lora_Q = nn.Parameter(q_init.to(base.weight.dtype))
        self.register_buffer("rank_mask", torch.ones(self.max_rank, dtype=base.weight.dtype))

    def set_rank(self, rank: int) -> None:
        rank = int(np.clip(rank, 0, self.max_rank))
        self.rank_mask.zero_()
        self.rank_mask[:rank] = 1.0

    @property
    def effective_rank(self) -> int:
        return int(self.rank_mask.sum().item())

    @property
    def transmitted_parameters(self) -> int:
        return self.effective_rank * (self.in_features + self.out_features + 1)

    def orthogonal_regularization(self) -> torch.Tensor:
        rank = self.effective_rank
        p = self.lora_P[:, :rank]
        q = self.lora_Q[:rank, :]
        eye = torch.eye(rank, device=p.device, dtype=p.dtype)
        return (p.T @ p - eye).square().sum() + (q @ q.T - eye).square().sum()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = F.linear(inputs, self.weight, self.bias)
        masked_e = self.lora_E * self.rank_mask
        update = (self.dropout(inputs) @ self.lora_Q.T) * masked_e
        return base + self.scaling * (update @ self.lora_P.T)


def add_svd_lora_adapters(
    model: nn.Module, config: ExperimentConfig
) -> List[Tuple[str, SVDLoRALinear]]:
    model.requires_grad_(False)
    selected = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and any(name.endswith(target) for target in TARGET_MODULES)
    ]
    adapters: List[Tuple[str, SVDLoRALinear]] = []
    for name, original in selected:
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        adapter = SVDLoRALinear(
            original,
            config.max_rank,
            config.lora_alpha,
            config.lora_dropout,
        )
        setattr(parent, child_name, adapter)
        adapters.append((name, adapter))

    for head_name in ("score", "classifier"):
        head = getattr(model, head_name, None)
        if head is not None:
            head.requires_grad_(True)
    if not adapters:
        raise RuntimeError(f"No target modules found. Expected suffixes: {TARGET_MODULES}")
    return adapters
