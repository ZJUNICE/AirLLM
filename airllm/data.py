from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SST2ParquetDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int) -> None:
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Dataset not found: {file_path}. Run prepare_sst2.py first."
            )
        frame = pd.read_parquet(file_path)
        missing = {"sentence", "label"} - set(frame.columns)
        if missing:
            raise ValueError(f"{file_path} is missing columns: {sorted(missing)}")
        self.sentences = frame["sentence"].astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, index: int) -> Tuple[Dict[str, List[int]], int]:
        encoded = self.tokenizer(
            self.sentences[index],
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        return encoded, self.labels[index]


class DynamicPaddingCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch):
        encoded, labels = zip(*batch)
        padded = self.tokenizer.pad(list(encoded), padding=True, return_tensors="pt")
        return dict(padded), torch.tensor(labels, dtype=torch.long)


def batch_linguistic_statistics(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer
) -> Tuple[float, float]:
    """Return normalized token entropy and tokenizer OOV rate."""

    valid = attention_mask.bool()
    special_ids = set(tokenizer.all_special_ids)
    tokens = [
        int(token)
        for token in input_ids[valid].detach().cpu().tolist()
        if int(token) not in special_ids
    ]
    if not tokens:
        return 0.0, 0.0
    counts = Counter(tokens)
    total = float(len(tokens))
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    max_entropy = max(math.log2(max(len(counts), 2)), 1.0)
    normalized_entropy = float(np.clip(entropy / max_entropy, 0.0, 1.0))
    unk_id = tokenizer.unk_token_id
    oov_rate = 0.0 if unk_id is None else sum(token == unk_id for token in tokens) / total
    return normalized_entropy, float(oov_rate)
