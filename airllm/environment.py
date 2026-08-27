from __future__ import annotations

import math
import random
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .adapters import SVDLoRALinear
from .config import ExperimentConfig, parse_number_list
from .data import batch_linguistic_statistics


class WirelessChannel:
    def __init__(self, config: ExperimentConfig) -> None:
        self.snr_values = parse_number_list(config.snr_db_values, float)
        self.bandwidth_values = parse_number_list(config.bandwidth_hz_values, float)
        self.latency_s = config.latency_s
        self.parameter_bits = config.parameter_bits
        self.snr_db = self.snr_values[0]
        self.bandwidth_hz = self.bandwidth_values[0]

    def sample(self) -> None:
        self.snr_db = random.choice(self.snr_values)
        self.bandwidth_hz = random.choice(self.bandwidth_values)

    @property
    def capacity_bps(self) -> float:
        return self.bandwidth_hz * math.log2(1.0 + 10.0 ** (self.snr_db / 10.0))

    def normalized_cost(self, parameter_count: int) -> float:
        budget_bits = max(self.capacity_bps * self.latency_s, 1.0)
        return parameter_count * self.parameter_bits / budget_bits

    def normalized_state(self) -> Tuple[float, float]:
        snr_min, snr_max = min(self.snr_values), max(self.snr_values)
        bw_min, bw_max = min(self.bandwidth_values), max(self.bandwidth_values)
        snr = 0.5 if snr_min == snr_max else (self.snr_db - snr_min) / (snr_max - snr_min)
        bandwidth = 0.5 if bw_min == bw_max else (self.bandwidth_hz - bw_min) / (bw_max - bw_min)
        return float(snr), float(bandwidth)


class AirLLMEnvironment:
    """MDP implementing the paper's state, constraint, and reward."""

    def __init__(
        self,
        model: nn.Module,
        adapters: List[Tuple[str, SVDLoRALinear]],
        train_loader: DataLoader,
        tokenizer,
        model_optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: ExperimentConfig,
    ) -> None:
        self.model = model
        self.adapters = adapters
        self.loader = train_loader
        self.iterator: Iterator = iter(train_loader)
        self.tokenizer = tokenizer
        self.optimizer = model_optimizer
        self.device = device
        self.config = config
        self.channel = WirelessChannel(config)
        self.current_entropy = 0.0
        self.current_oov = 0.0
        self.current_step = 0
        self.last_batch: Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor]] = None

    @property
    def action_dim(self) -> int:
        return len(self.adapters)

    @property
    def state_dim(self) -> int:
        return 4 + self.action_dim

    def _next_batch(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
        self.last_batch = batch
        return batch

    def state(self) -> np.ndarray:
        snr, bandwidth = self.channel.normalized_state()
        ranks = [layer.effective_rank / layer.max_rank for _, layer in self.adapters]
        return np.asarray(
            [snr, bandwidth, self.current_entropy, self.current_oov, *ranks],
            dtype=np.float32,
        )

    def reset(self) -> np.ndarray:
        for _, layer in self.adapters:
            layer.set_rank(max(self.config.min_rank, layer.max_rank // 2))
        self.channel.sample()
        inputs, _ = self._next_batch()
        self.current_entropy, self.current_oov = batch_linguistic_statistics(
            inputs["input_ids"], inputs["attention_mask"], self.tokenizer
        )
        self.current_step = 0
        return self.state()

    def _apply_ranks(self, ranks: Sequence[int]) -> None:
        if len(ranks) != self.action_dim:
            raise ValueError(f"Expected {self.action_dim} ranks, received {len(ranks)}")
        for rank, (_, layer) in zip(ranks, self.adapters):
            layer.set_rank(int(np.clip(rank, self.config.min_rank, layer.max_rank)))

    def step(self, ranks: Sequence[int]) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        self._apply_ranks(ranks)
        transmitted = sum(layer.transmitted_parameters for _, layer in self.adapters)
        communication_cost = self.channel.normalized_cost(transmitted)
        constraint_violated = communication_cost > 1.0

        if constraint_violated:
            task_loss_value = self.config.constraint_penalty * communication_cost
            reward = -task_loss_value - self.config.lambda_comm * communication_cost
        else:
            inputs, labels = self._next_batch()
            self.current_entropy, self.current_oov = batch_linguistic_statistics(
                inputs["input_ids"], inputs["attention_mask"], self.tokenizer
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            labels = labels.to(self.device)
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.config.amp and self.device.type == "cuda",
            ):
                logits = self.model(**inputs).logits
                task_loss = F.cross_entropy(logits, labels)
                orthogonal = torch.stack(
                    [layer.orthogonal_regularization() for _, layer in self.adapters]
                ).mean()
                training_loss = task_loss + self.config.orthogonal_weight * orthogonal
            training_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            task_loss_value = float(task_loss.detach().cpu())
            reward = -task_loss_value - self.config.lambda_comm * communication_cost

        self.current_step += 1
        done = self.current_step >= self.config.env_steps
        used_snr_db = self.channel.snr_db
        used_bandwidth_hz = self.channel.bandwidth_hz
        self.channel.sample()
        info = {
            "task_loss": task_loss_value,
            "communication_cost": communication_cost,
            "transmitted_parameters": float(transmitted),
            "snr_db": used_snr_db,
            "bandwidth_hz": used_bandwidth_hz,
            "constraint_violated": float(constraint_violated),
        }
        return self.state(), float(reward), done, info
