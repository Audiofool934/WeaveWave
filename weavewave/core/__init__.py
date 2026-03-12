"""Core abstractions for WeaveWave.

Heavy dependencies (audiocraft, torch) are imported lazily to allow lightweight
modules (config, logging, types) to be used without GPU libraries installed.
"""

from .config import AppConfig, PromptConfig
from .logging import setup_logging
from .mllm_client import MLLMClient, MLLMClientError
from .types import GenerationConfig, MLLMServerConfig

__all__ = [
    "AppConfig",
    "GenerationConfig",
    "MLLMClient",
    "MLLMClientError",
    "MLLMServerConfig",
    "MusicGenerationError",
    "MusicGenerator",
    "PromptConfig",
    "setup_logging",
]


def __getattr__(name: str):
    if name in ("MusicGenerationError", "MusicGenerator"):
        from .music_generator import MusicGenerationError, MusicGenerator

        globals()["MusicGenerationError"] = MusicGenerationError
        globals()["MusicGenerator"] = MusicGenerator
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
