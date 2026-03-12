"""HTTP client for the multimodal large language model service."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import requests

from .config import AppConfig


class MLLMClientError(RuntimeError):
    """Raised when the MLLM client fails to obtain a description."""


@dataclass
class MLLMClient:
    """Thin HTTP client around the MLLM service.

    Args:
        config: Application configuration providing URLs and timeouts.
    """

    config: AppConfig

    def __post_init__(self) -> None:
        self._base_url = self.config.mllm_api_url.rstrip("/")
        self._session = requests.Session()
        self._session.trust_env = False  # bypass proxy for localhost

    def describe(
        self,
        media_type: str | None,
        media_path: str | None,
        user_prompt: str,
    ) -> str:
        """Request a music description for the given media input.

        Args:
            media_type: ``'image'``, ``'video'``, or ``None``/``'text'``.
            media_path: Local file path for image/video inputs.
            user_prompt: Prompt typed by the user in the UI.

        Returns:
            A music-description string produced by the MLLM service.

        Raises:
            MLLMClientError: On network errors, missing media, or bad responses.
        """
        task = self.config.task_from_media_type(media_type or "")
        endpoint = self.config.endpoint_for_task(task)
        prompt = self.config.compose_prompt(media_type or "", user_prompt)

        payload: dict[str, str] = {"user_prompt": prompt}
        if task in {"image", "video"}:
            if not media_path:
                raise MLLMClientError(f"Media path is required for {task} description requests.")
            payload[task] = self._encode_media(media_path)

        try:
            response = self._session.post(
                f"{self._base_url}{endpoint}",
                json=payload,
                timeout=self.config.mllm_timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise MLLMClientError(f"MLLM request failed: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise MLLMClientError("MLLM response was not valid JSON.") from exc

        description = data.get("description")
        if not description:
            raise MLLMClientError("MLLM response did not include a description.")
        return str(description).strip()

    @staticmethod
    def _encode_media(media_path: str) -> str:
        """Read a local file and return its base64-encoded content."""
        path = Path(media_path)
        if not path.exists():
            raise MLLMClientError(f"Media file not found: {media_path}")
        try:
            return base64.b64encode(path.read_bytes()).decode("utf-8")
        except OSError as exc:
            raise MLLMClientError(f"Failed to read media file: {media_path}") from exc
