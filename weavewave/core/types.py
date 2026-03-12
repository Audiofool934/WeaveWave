"""Shared dataclass definitions used across WeaveWave."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class GenerationConfig:
    """Parameters that control MusicGen audio generation."""

    duration: float = 10.0
    use_sampling: bool = True
    top_k: int = 250
    top_p: float = 0.0
    temperature: float = 1.0
    cfg_coef: float = 3.0
    cfg_coef_beta: float = 5.0


@dataclass
class MLLMServerConfig:
    """Configuration for the MLLM FastAPI server."""

    model_name: str = field(
        default_factory=lambda: os.getenv("WEAVEWAVE_MLLM_MODEL", "google/gemma-3-12b-it")
    )
    device: str = field(default_factory=lambda: os.getenv("WEAVEWAVE_MLLM_DEVICE", "cuda"))
    max_num_frames: int = 32
    max_new_tokens: int = 200
    host: str = "0.0.0.0"
    port: int = 8001
