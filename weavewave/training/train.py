#!/usr/bin/env python3
# Copyright (c) 2023-2024 WeaveWave Team.
# All rights reserved.

"""Training script for the MusicGen-Style model based on AudioCraft."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

from weavewave.core.logging import setup_logging

logger = setup_logging(name=__name__)


def setup_environment(
    output_dir: str = "./outputs",
    team: str = "default",
) -> None:
    """Prepare the AudioCraft training environment.

    Args:
        output_dir: Directory for model checkpoints and logs.
        team: AudioCraft team identifier.
    """
    os.environ.setdefault("AUDIOCRAFT_TEAM", team)
    os.environ.setdefault("DORA_DIR_ROOT", output_dir)

    ref_dir = Path(tempfile.mkdtemp())
    os.environ["AUDIOCRAFT_REFERENCE_DIR"] = str(ref_dir)

    Path(output_dir).mkdir(exist_ok=True)

    if not torch.cuda.is_available():
        logger.warning("CUDA is not available — training will be slow")
    else:
        logger.info("Detected %d GPU(s)", torch.cuda.device_count())


def train_musicgen_style(
    config_path: str = "config/musicgen_style_32khz.yaml",
    data_dir: str = "./data/multimodal_music_dataset",
) -> None:
    """Launch MusicGen-Style training from a Hydra config.

    Args:
        config_path: Path to the YAML configuration file.
        data_dir: Root of the dataset directory (train/valid/test splits).
    """
    logger.info("Preparing MusicGen-Style training")

    data_root = Path(data_dir)
    for split in ("train", "valid", "test"):
        (data_root / split).mkdir(parents=True, exist_ok=True)

    cfg_file = Path(config_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info("Loading config: %s", config_path)
    cfg = OmegaConf.load(config_path)

    logger.info("Importing AudioCraft training module")
    from audiocraft.train import main as train_main  # noqa: E402

    logger.info("Starting training")
    train_main(cfg)


def main() -> None:
    """Entry point for ``weavewave-train``."""
    logger.info("=== WeaveWave MusicGen-Style Training ===")
    setup_environment()
    train_musicgen_style()
    logger.info("=== Training script finished ===")


if __name__ == "__main__":
    main()
