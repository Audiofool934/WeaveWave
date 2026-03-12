"""Centralized configuration for the WeaveWave demo application."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_MUSIC_MODELS: list[str] = [
    "facebook/musicgen-melody",
    "facebook/musicgen-medium",
    "facebook/musicgen-small",
    "facebook/musicgen-large",
    "facebook/musicgen-melody-large",
    "facebook/musicgen-stereo-small",
    "facebook/musicgen-stereo-medium",
    "facebook/musicgen-stereo-melody",
    "facebook/musicgen-stereo-large",
    "facebook/musicgen-stereo-melody-large",
]


DEFAULT_PROMPTS: dict[str, str] = {
    "text": "Compose a short, vivid music description based on the user prompt.",
    "image": (
        "Compose a concise music description that captures the mood, colors, and motion "
        "suggested by the image together with the user prompt."
    ),
    "video": (
        "Compose a concise music description that matches the pacing and tone of the video "
        "while incorporating the user prompt."
    ),
}


@dataclass
class PromptConfig:
    """Holds prompt templates for different multimodal tasks."""

    prompts: dict[str, str] = field(default_factory=lambda: DEFAULT_PROMPTS.copy())

    def get(self, task: str) -> str:
        """Return the prompt template for *task*, or an empty string."""
        return self.prompts.get(task.lower(), "")


@dataclass
class AppConfig:
    """Centralized configuration for the WeaveWave demo.

    Parameters can be overridden via environment variables so that ports, model
    selections, and prompt templates remain flexible across deployments.
    """

    prompts: PromptConfig = field(default_factory=PromptConfig)
    available_music_models: list[str] = field(default_factory=lambda: DEFAULT_MUSIC_MODELS.copy())
    mllm_api_url: str = field(
        default_factory=lambda: os.getenv("WEAVEWAVE_MLLM_URL", "http://127.0.0.1:8001")
    )
    default_music_model: str = field(
        default_factory=lambda: os.getenv("WEAVEWAVE_DEFAULT_MUSIC_MODEL", DEFAULT_MUSIC_MODELS[-1])
    )
    diffusion_decoder_label: str = "MultiBand_Diffusion"
    mllm_timeout: int = 60

    @staticmethod
    def _normalise_media_type(media_type: str) -> str:
        return (media_type or "").lower()

    def task_from_media_type(self, media_type: str) -> str:
        """Map a UI media-type string to a canonical task name."""
        media = self._normalise_media_type(media_type)
        if media == "image":
            return "image"
        if media == "video":
            return "video"
        return "text"

    def endpoint_for_task(self, task: str) -> str:
        """Return the MLLM API endpoint path for the given task."""
        task = task.lower()
        if task == "image":
            return "/describe_image/"
        if task == "video":
            return "/describe_video/"
        return "/describe_text/"

    def compose_prompt(self, media_type: str, user_prompt: str) -> str:
        """Build the prompt string sent to the MLLM service."""
        prompt = (user_prompt or "").strip()
        if prompt:
            return prompt
        task = self.task_from_media_type(media_type)
        return self.prompts.get(task)
