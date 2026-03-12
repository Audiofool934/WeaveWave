#!/usr/bin/env python3
# Copyright (c) 2023-2024 WeaveWave Team.
# All rights reserved.

"""High-level training runner with data preparation and GPU selection."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from weavewave.core.logging import setup_logging

logger = setup_logging(name=__name__)


def _ensure_directories() -> None:
    """Validate that config, output, and data directories exist."""
    if not Path("./config").exists():
        raise FileNotFoundError("Config directory not found — are you in the project root?")
    Path("./outputs").mkdir(exist_ok=True)
    Path("./data").mkdir(exist_ok=True)


def prepare_data(
    *,
    dummy: bool = False,
    dummy_samples: int = 100,
    source_dir: str | None = None,
    force_rebuild: bool = False,
) -> None:
    """Run the dataset preparation script.

    Args:
        dummy: Create a synthetic dummy dataset for testing.
        dummy_samples: Number of dummy samples to generate.
        source_dir: Path to a real dataset source directory.
        force_rebuild: Rebuild the dataset even if it already exists.
    """
    dataset_dir = Path("./data/multimodal_music_dataset")
    if dataset_dir.exists() and not force_rebuild:
        sample_count = sum(1 for _ in dataset_dir.glob("*/*.json"))
        if sample_count > 0:
            logger.info("Dataset already exists with %d samples", sample_count)
            return

    cmd = [
        sys.executable,
        "-m",
        "weavewave.data.prepare_dataset",
        "--output_dir",
        "./data/multimodal_music_dataset",
    ]

    if dummy or source_dir is None:
        cmd.extend(["--create_dummy", "--dummy_samples", str(dummy_samples)])
    else:
        cmd.extend(["--source_dir", source_dir])

    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    logger.info("Data preparation complete")


def run_training(gpu_ids: list[int] | None = None) -> None:
    """Launch the training script, optionally pinning specific GPUs.

    Args:
        gpu_ids: List of CUDA device IDs to use.
    """
    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        logger.info("Using GPU(s): %s", os.environ["CUDA_VISIBLE_DEVICES"])

    cmd = [sys.executable, "-m", "weavewave.training.train"]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    logger.info("Training complete")


def main() -> None:
    """Entry point for ``weavewave-train`` (runner mode)."""
    parser = argparse.ArgumentParser(description="WeaveWave training runner")

    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--dummy_data", action="store_true", help="Use a synthetic dummy dataset"
    )
    data_group.add_argument(
        "--dummy_samples", type=int, default=100, help="Number of dummy samples"
    )
    data_group.add_argument(
        "--source_data", type=str, default=None, help="Real dataset source directory"
    )
    data_group.add_argument(
        "--force_rebuild_dataset",
        action="store_true",
        help="Rebuild dataset even if it exists",
    )

    train_group = parser.add_argument_group("training")
    train_group.add_argument(
        "--gpu", type=int, nargs="+", default=None, help="GPU device IDs (e.g. 0 1)"
    )

    args = parser.parse_args()

    logger.info("=== WeaveWave Training Runner ===")

    _ensure_directories()

    prepare_data(
        dummy=args.dummy_data,
        dummy_samples=args.dummy_samples,
        source_dir=args.source_data,
        force_rebuild=args.force_rebuild_dataset,
    )

    run_training(gpu_ids=args.gpu)
    logger.info("=== Training pipeline finished ===")


if __name__ == "__main__":
    main()
