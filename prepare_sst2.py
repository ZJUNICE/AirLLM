"""Download GLUE/SST-2 and export the three splits as parquet files."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("glue", "sst2")
    # GLUE hides labels for the official test split.  Reserve a deterministic
    # part of the training set for model selection and use the labelled
    # official validation split as the local test split.
    split = dataset["train"].train_test_split(
        test_size=args.validation_fraction, seed=args.seed, shuffle=True
    )
    frames = {
        "train": split["train"].to_pandas(),
        "validation": split["test"].to_pandas(),
        "test": dataset["validation"].to_pandas(),
    }
    for name, frame in frames.items():
        frame.to_parquet(output / f"{name}-00000-of-00001.parquet", index=False)


if __name__ == "__main__":
    main()
