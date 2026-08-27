from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ExperimentConfig:
    model_name: str = "facebook/opt-1.3b"
    train_file: str = "data/train-00000-of-00001.parquet"
    validation_file: str = "data/validation-00000-of-00001.parquet"
    test_file: str = "data/test-00000-of-00001.parquet"
    output_dir: str = "outputs/opt-1.3b"
    seed: int = 42
    batch_size: int = 32
    max_length: int = 128
    num_workers: int = 0
    max_rank: int = 64
    min_rank: int = 1
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    model_lr: float = 5e-5
    ppo_lr: float = 1e-4
    diffusion_lr: float = 5e-5
    weight_decay: float = 1e-5
    lambda_comm: float = 0.1
    orthogonal_weight: float = 0.01
    constraint_penalty: float = 10.0
    snr_db_values: str = "-5,0,5,10,15"
    bandwidth_hz_values: str = "100000000"
    latency_s: float = 1.0
    parameter_bits: int = 32
    episodes: int = 1000
    env_steps: int = 15
    eval_every_steps: int = 100
    ppo_epochs: int = 4
    ppo_batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    coarse_dim: int = 16
    policy_hidden: int = 256
    diffusion_hidden: int = 512
    diffusion_train_steps: int = 1000
    diffusion_inference_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    cfg_scale: float = 1.5
    condition_dropout: float = 0.1
    diffusion_updates: int = 4
    diffusion_batch_size: int = 64
    reward_weight: float = 0.1
    replay_size: int = 10000
    replay_warmup: int = 128
    early_stopping_patience: int = 5
    amp: bool = True
    device: str = "auto"


def parse_number_list(value: str, cast=float) -> List[float]:
    values = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated number")
    return values


def parse_config() -> ExperimentConfig:
    """Load the experiment JSON and allow only essential runtime overrides."""

    parser = argparse.ArgumentParser(description="Train AirLLM with hierarchical PPO-DDIM.")
    parser.add_argument("--config", default="configs/paper_rank8_snr10.json")
    parser.add_argument("--output-dir", help="Override the output directory.")
    parser.add_argument("--device", help="Override device, for example cuda or cpu.")
    args = parser.parse_args()

    payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    unknown = set(payload) - set(ExperimentConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    if args.output_dir:
        payload["output_dir"] = args.output_dir
    if args.device:
        payload["device"] = args.device
    return ExperimentConfig(**payload)
