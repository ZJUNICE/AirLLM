"""Command-line entry point for AirLLM training."""

import logging

from airllm.config import parse_config
from airllm.trainer import train


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    train(parse_config())


if __name__ == "__main__":
    main()
