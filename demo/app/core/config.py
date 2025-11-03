import os
from dataclasses import dataclass, field
from typing import Dict, List


DEFAULT_MUSIC_MODELS: List[str] = [
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


DEFAULT_PROMPTS: Dict[str, str] = {
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

    prompts: Dict[str, str] = field(default_factory=lambda: DEFAULT_PROMPTS.copy())

    def get(self, task: str) -> str:
        return self.prompts.get(task.lower(), "")


@dataclass
class AppConfig:
    """
    Centralised configuration for the WeaveWave demo.

    The parameters can be overridden via environment variables so that ports, model
    selections, and prompt templates remain flexible when swapping components.
    """

    prompts: PromptConfig = field(default_factory=PromptConfig)
    available_music_models: List[str] = field(
        default_factory=lambda: DEFAULT_MUSIC_MODELS.copy()
    )
    mllm_api_url: str = field(
        default_factory=lambda: os.getenv("WEAVEWAVE_MLLM_URL", "http://127.0.0.1:8001")
    )
    default_music_model: str = field(
        default_factory=lambda: os.getenv(
            "WEAVEWAVE_DEFAULT_MUSIC_MODEL", DEFAULT_MUSIC_MODELS[-1]
        )
    )
    diffusion_decoder_label: str = "MultiBand_Diffusion"
    mllm_timeout: int = 60

    @staticmethod
    def _normalise_media_type(media_type: str) -> str:
        return (media_type or "").lower()

    def task_from_media_type(self, media_type: str) -> str:
        media = self._normalise_media_type(media_type)
        if media == "image":
            return "image"
        if media == "video":
            return "video"
        return "text"

    def endpoint_for_task(self, task: str) -> str:
        task = task.lower()
        if task == "image":
            return "/describe_image/"
        if task == "video":
            return "/describe_video/"
        return "/describe_text/"

    def compose_prompt(self, media_type: str, user_prompt: str) -> str:
        prompt = (user_prompt or "").strip()
        if prompt:
            return prompt
        task = self.task_from_media_type(media_type)
        return self.prompts.get(task)

