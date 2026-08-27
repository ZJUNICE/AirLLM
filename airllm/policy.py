from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .config import ExperimentConfig
from .environment import AirLLMEnvironment


class CoarsePPOActorCritic(nn.Module):
    def __init__(self, state_dim: int, coarse_dim: int, hidden: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
            nn.Mish(),
        )
        self.mean = nn.Linear(hidden, coarse_dim)
        self.log_std = nn.Parameter(torch.full((coarse_dim,), -0.5))
        self.value = nn.Linear(hidden, 1)

    def distribution_and_value(self, states: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        features = self.encoder(states)
        mean = self.mean(features)
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std), self.value(features).squeeze(-1)

    def sample(self, states: torch.Tensor):
        distribution, value = self.distribution_and_value(states)
        raw_action = distribution.rsample()
        coarse = torch.tanh(raw_action)
        log_prob = distribution.log_prob(raw_action).sum(-1)
        return raw_action, coarse, log_prob, value


def sinusoidal_embedding(timesteps: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    exponent = -math.log(10000.0) * torch.arange(half, device=timesteps.device) / max(half - 1, 1)
    angles = timesteps.float().unsqueeze(-1) * exponent.exp().unsqueeze(0)
    embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
    return F.pad(embedding, (0, dimension - embedding.shape[-1]))


class ConditionalDenoiser(nn.Module):
    def __init__(self, action_dim: int, coarse_dim: int, hidden: int, time_dim: int = 64) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.time_projection = nn.Sequential(nn.Linear(time_dim, hidden), nn.Mish())
        self.input_projection = nn.Linear(action_dim + coarse_dim, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden), nn.Mish(), nn.Linear(hidden, hidden)),
            nn.Sequential(nn.LayerNorm(hidden), nn.Mish(), nn.Linear(hidden, hidden)),
        ])
        self.output = nn.Sequential(nn.LayerNorm(hidden), nn.Mish(), nn.Linear(hidden, action_dim))

    def forward(self, noisy_rank: torch.Tensor, timestep: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        time = self.time_projection(sinusoidal_embedding(timestep, self.time_dim))
        hidden = self.input_projection(torch.cat([noisy_rank, coarse], dim=-1)) + time
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output(hidden)


class ConditionalDDIM(nn.Module):
    def __init__(self, action_dim: int, config: ExperimentConfig) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.config = config
        self.denoiser = ConditionalDenoiser(action_dim, config.coarse_dim, config.diffusion_hidden)
        betas = torch.linspace(
            math.sqrt(config.beta_start),
            math.sqrt(config.beta_end),
            config.diffusion_train_steps,
        ).square()
        self.register_buffer("alpha_bar", torch.cumprod(1.0 - betas, dim=0))

    def q_sample(self, clean: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bar[timestep].unsqueeze(-1)
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise

    def training_loss(self, clean: torch.Tensor, coarse: torch.Tensor, rewards: torch.Tensor):
        batch = clean.shape[0]
        timestep = torch.randint(0, len(self.alpha_bar), (batch,), device=clean.device)
        noise = torch.randn_like(clean)
        noisy = self.q_sample(clean, timestep, noise)
        keep_condition = (
            torch.rand(batch, 1, device=clean.device) >= self.config.condition_dropout
        ).to(clean.dtype)
        prediction = self.denoiser(noisy, timestep, coarse * keep_condition)
        per_sample_mse = (prediction - noise).square().mean(dim=-1)
        standardized_reward = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
        sample_weight = (1.0 + self.config.reward_weight * standardized_reward).clamp(0.1, 10.0)
        loss = (sample_weight.detach() * per_sample_mse).mean()
        return loss, {"ddim_mse": float(loss.detach().cpu())}

    @torch.no_grad()
    def sample(self, coarse: torch.Tensor) -> torch.Tensor:
        batch = coarse.shape[0]
        sample = torch.randn(batch, self.action_dim, device=coarse.device)
        schedule = torch.linspace(
            len(self.alpha_bar) - 1,
            0,
            self.config.diffusion_inference_steps,
            device=coarse.device,
        ).long()
        for index, timestep_scalar in enumerate(schedule):
            timestep = timestep_scalar.expand(batch)
            conditional = self.denoiser(sample, timestep, coarse)
            unconditional = self.denoiser(sample, timestep, torch.zeros_like(coarse))
            predicted_noise = unconditional + self.config.cfg_scale * (conditional - unconditional)
            alpha_bar_t = self.alpha_bar[timestep_scalar]
            clean = (
                sample - (1.0 - alpha_bar_t).sqrt() * predicted_noise
            ) / alpha_bar_t.sqrt()
            clean = clean.clamp(-1.0, 1.0)
            if index == len(schedule) - 1:
                sample = clean
            else:
                alpha_bar_previous = self.alpha_bar[schedule[index + 1]]
                sample = (
                    alpha_bar_previous.sqrt() * clean
                    + (1.0 - alpha_bar_previous).sqrt() * predicted_noise
                )
        return sample.clamp(-1.0, 1.0)

    def decode_ranks(self, normalized: torch.Tensor) -> torch.Tensor:
        span = self.config.max_rank - self.config.min_rank
        ranks = torch.floor((normalized.clamp(-1.0, 1.0) + 1.0) * 0.5 * span)
        return (ranks + self.config.min_rank).clamp(
            self.config.min_rank, self.config.max_rank
        ).long()

    def encode_ranks(self, ranks: torch.Tensor) -> torch.Tensor:
        span = max(self.config.max_rank - self.config.min_rank, 1)
        return 2.0 * (ranks.float() - self.config.min_rank) / span - 1.0


@dataclass
class Transition:
    state: torch.Tensor
    raw_action: torch.Tensor
    coarse: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    reward: float
    done: bool
    ranks: torch.Tensor


class RankReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.data: Deque[Tuple[torch.Tensor, torch.Tensor, float]] = deque(maxlen=capacity)

    def add(self, coarse: torch.Tensor, normalized_ranks: torch.Tensor, reward: float) -> None:
        self.data.append((coarse.detach().cpu(), normalized_ranks.detach().cpu(), float(reward)))

    def __len__(self) -> int:
        return len(self.data)

    def sample(self, batch_size: int, device: torch.device):
        chosen = random.sample(self.data, min(batch_size, len(self.data)))
        coarse, ranks, rewards = zip(*chosen)
        return (
            torch.stack(coarse).to(device),
            torch.stack(ranks).to(device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
        )


def generalized_advantage_estimation(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), device=rewards.device)
    next_value = last_value
    for index in reversed(range(len(rewards))):
        not_done = 1.0 - dones[index]
        delta = rewards[index] + gamma * next_value * not_done - values[index]
        gae = delta + gamma * gae_lambda * not_done * gae
        advantages[index] = gae
        next_value = values[index]
    return advantages, advantages + values


def collect_episode(
    env: AirLLMEnvironment,
    ppo: CoarsePPOActorCritic,
    diffusion: ConditionalDDIM,
    replay: RankReplayBuffer,
    device: torch.device,
    start_step: int = 0,
    step_callback: Optional[Callable[[int, Sequence[Transition]], bool]] = None,
):
    transitions: List[Transition] = []
    diagnostics: List[Dict[str, float]] = []
    state = env.reset()
    done = False
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            raw_action, coarse, log_prob, value = ppo.sample(state_tensor)
            normalized_ranks = diffusion.sample(coarse)
            ranks = diffusion.decode_ranks(normalized_ranks)
        next_state, reward, done, info = env.step(ranks.squeeze(0).cpu().tolist())
        transitions.append(Transition(
            state_tensor.squeeze(0).cpu(), raw_action.squeeze(0).cpu(),
            coarse.squeeze(0).cpu(), log_prob.squeeze(0).cpu(),
            value.squeeze(0).cpu(), reward, done, ranks.squeeze(0).cpu(),
        ))
        replay.add(coarse.squeeze(0), diffusion.encode_ranks(ranks).squeeze(0), reward)
        diagnostics.append(info)
        state = next_state
        training_step = start_step + len(transitions)
        if step_callback is not None and step_callback(training_step, transitions):
            done = True
    return transitions, diagnostics, state


def update_ppo(
    ppo: CoarsePPOActorCritic,
    optimizer: torch.optim.Optimizer,
    transitions: Sequence[Transition],
    final_state: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, float]:
    states = torch.stack([item.state for item in transitions]).to(device)
    raw_actions = torch.stack([item.raw_action for item in transitions]).to(device)
    old_log_probs = torch.stack([item.log_prob for item in transitions]).to(device)
    values = torch.stack([item.value for item in transitions]).to(device)
    rewards = torch.tensor([item.reward for item in transitions], dtype=torch.float32, device=device)
    dones = torch.tensor([item.done for item in transitions], dtype=torch.float32, device=device)
    with torch.no_grad():
        final = torch.tensor(final_state, dtype=torch.float32, device=device).unsqueeze(0)
        _, last_value = ppo.distribution_and_value(final)
        advantages, returns = generalized_advantage_estimation(
            rewards, values, dones, last_value.squeeze(0), config.gamma, config.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)

    metrics: Dict[str, float] = {}
    indices = np.arange(len(transitions))
    for _ in range(config.ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), config.ppo_batch_size):
            batch = torch.as_tensor(indices[start:start + config.ppo_batch_size], device=device)
            distribution, new_values = ppo.distribution_and_value(states[batch])
            new_log_probs = distribution.log_prob(raw_actions[batch]).sum(-1)
            entropy = distribution.entropy().sum(-1).mean()
            ratio = (new_log_probs - old_log_probs[batch]).exp()
            surrogate_1 = ratio * advantages[batch]
            surrogate_2 = ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantages[batch]
            policy_loss = -torch.minimum(surrogate_1, surrogate_2).mean()
            value_loss = F.mse_loss(new_values, returns[batch])
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(ppo.parameters(), config.max_grad_norm)
            optimizer.step()
            metrics = {
                "ppo_loss": float(loss.detach().cpu()),
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
            }
    return metrics


def update_diffusion(
    diffusion: ConditionalDDIM,
    optimizer: torch.optim.Optimizer,
    replay: RankReplayBuffer,
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, float]:
    if len(replay) < config.replay_warmup:
        return {"ddim_mse": float("nan")}
    metrics: Dict[str, float] = {}
    diffusion.train()
    for _ in range(config.diffusion_updates):
        coarse, ranks, rewards = replay.sample(config.diffusion_batch_size, device)
        loss, metrics = diffusion.training_loss(ranks, coarse, rewards)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(diffusion.parameters(), config.max_grad_norm)
        optimizer.step()
    return metrics
