from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .adapters import add_svd_lora_adapters
from .config import ExperimentConfig
from .data import DynamicPaddingCollator, SST2ParquetDataset
from .environment import AirLLMEnvironment
from .policy import (
    CoarsePPOActorCritic,
    ConditionalDDIM,
    RankReplayBuffer,
    collect_episode,
    update_diffusion,
    update_ppo,
)


LOGGER = logging.getLogger("airllm")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def build_loaders(config: ExperimentConfig, tokenizer):
    collator = DynamicPaddingCollator(tokenizer)
    common = dict(
        batch_size=config.batch_size,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train = SST2ParquetDataset(config.train_file, tokenizer, config.max_length)
    validation = SST2ParquetDataset(config.validation_file, tokenizer, config.max_length)
    test = SST2ParquetDataset(config.test_file, tokenizer, config.max_length)
    return (
        DataLoader(train, shuffle=True, **common),
        DataLoader(validation, shuffle=False, **common),
        DataLoader(test, shuffle=False, **common),
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs = {key: value.to(device) for key, value in inputs.items()}
        labels = labels.to(device)
        labelled = labels >= 0
        if labelled.any():
            predictions = model(**inputs).logits.argmax(dim=-1)
            correct += int((predictions[labelled] == labels[labelled]).sum())
            total += int(labelled.sum())
    return correct / total if total else 0.0


def append_csv(path: Path, fieldnames: Sequence[str], row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def adapter_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    keep = ("lora_P", "lora_E", "lora_Q", "rank_mask", "score", "classifier")
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if any(token in key for token in keep)
    }


def train(config: ExperimentConfig) -> None:
    set_seed(config.seed)
    device = resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, num_labels=2
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    adapters = add_svd_lora_adapters(model, config)
    model.to(device)
    train_loader, validation_loader, test_loader = build_loaders(config, tokenizer)

    model_optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.model_lr,
        weight_decay=config.weight_decay,
    )
    environment = AirLLMEnvironment(
        model, adapters, train_loader, tokenizer, model_optimizer, device, config
    )
    ppo = CoarsePPOActorCritic(
        environment.state_dim, config.coarse_dim, config.policy_hidden
    ).to(device)
    diffusion = ConditionalDDIM(environment.action_dim, config).to(device)
    ppo_optimizer = torch.optim.Adam(ppo.parameters(), lr=config.ppo_lr)
    diffusion_optimizer = torch.optim.Adam(diffusion.parameters(), lr=config.diffusion_lr)
    replay = RankReplayBuffer(config.replay_size)

    metrics_path = output / "metrics.csv"
    best_validation = -1.0
    stale_evaluations = 0
    training_step = 0
    total_training_steps = config.episodes * config.env_steps
    stop_requested = False

    def evaluate_at_step(step: int, current_transitions) -> bool:
        nonlocal best_validation, stale_evaluations, stop_requested
        should_evaluate = (
            step % config.eval_every_steps == 0 or step == total_training_steps
        )
        if not should_evaluate:
            return False

        validation_accuracy = evaluate(model, validation_loader, device)
        LOGGER.info(
            "training_step=%d validation_accuracy=%.4f", step, validation_accuracy
        )
        if validation_accuracy > best_validation:
            best_validation = validation_accuracy
            stale_evaluations = 0
            torch.save(
                {
                    "ppo": ppo.state_dict(),
                    "diffusion": diffusion.state_dict(),
                    "adapters": adapter_state_dict(model),
                    "adapter_names": [name for name, _ in adapters],
                    "validation_accuracy": validation_accuracy,
                    "training_step": step,
                    "config": asdict(config),
                },
                output / "best.pt",
            )
        else:
            stale_evaluations += 1

        mean_reward_at_evaluation = float(
            np.mean([item.reward for item in current_transitions])
        )
        reward_threshold = min(-config.lambda_comm, -0.5)
        stop_requested = (
            mean_reward_at_evaluation >= reward_threshold
            and stale_evaluations >= config.early_stopping_patience
        )
        if stop_requested:
            LOGGER.info("early_stopping training_step=%d", step)
        return stop_requested

    for episode in range(1, config.episodes + 1):
        diffusion.eval()
        transitions, diagnostics, final_state = collect_episode(
            environment,
            ppo,
            diffusion,
            replay,
            device,
            start_step=training_step,
            step_callback=evaluate_at_step,
        )
        training_step += len(transitions)
        ppo_metrics = update_ppo(
            ppo, ppo_optimizer, transitions, final_state, config, device
        )
        diffusion_metrics = update_diffusion(
            diffusion, diffusion_optimizer, replay, config, device
        )

        mean_reward = float(np.mean([item.reward for item in transitions]))
        mean_task_loss = float(np.mean([item["task_loss"] for item in diagnostics]))
        mean_comm = float(np.mean([item["communication_cost"] for item in diagnostics]))
        row = {
            "episode": episode,
            "training_step": training_step,
            "reward": mean_reward,
            "task_loss": mean_task_loss,
            "communication_cost": mean_comm,
            **ppo_metrics,
            **diffusion_metrics,
        }
        append_csv(metrics_path, list(row), row)
        LOGGER.info(
            "episode=%d training_step=%d reward=%.4f task_loss=%.4f comm=%.4f replay=%d",
            episode,
            training_step,
            mean_reward,
            mean_task_loss,
            mean_comm,
            len(replay),
        )

        if stop_requested:
            break

    best_checkpoint = output / "best.pt"
    if best_checkpoint.is_file():
        payload = torch.load(best_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(payload["adapters"], strict=False)
    test_accuracy = evaluate(model, test_loader, device)
    summary = {
        "best_validation_accuracy": best_validation,
        "test_accuracy": test_accuracy,
        "training_steps_completed": training_step,
        "final_ranks": {name: layer.effective_rank for name, layer in adapters},
    }
    if device.type == "cuda":
        summary["peak_gpu_memory_gib"] = round(
            torch.cuda.max_memory_allocated(device) / 2**30, 3
        )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    LOGGER.info("test_accuracy=%.4f", test_accuracy)
