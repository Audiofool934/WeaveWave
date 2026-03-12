"""MusicGen wrapper with optional MultiBand Diffusion post-processing."""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import numpy as np
import torch
from audiocraft.data.audio_utils import convert_audio
from audiocraft.models import MultiBandDiffusion, MusicGen
from audiocraft.models.encodec import InterleaveStereoCompressionModel
from einops import rearrange

MelodyInput = tuple[int, np.ndarray]


class MusicGenerationError(RuntimeError):
    """Raised when music generation fails."""


@dataclass
class MusicGenerator:
    """Wrapper around MusicGen + optional MultiBand Diffusion decoding.

    Args:
        device: Torch device string.  Auto-detected when *None*.
    """

    device: str | None = None

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model: MusicGen | None = None
        self._model_version: str | None = None
        self._diffusion: MultiBandDiffusion | None = None

    def generate(
        self,
        *,
        model_version: str,
        description: str,
        duration: float,
        top_k: int,
        top_p: float,
        temperature: float,
        cfg_coef: float,
        melody: MelodyInput | None,
        use_diffusion: bool,
    ) -> torch.Tensor:
        """Generate an audio tensor from a text description.

        Args:
            model_version: HuggingFace model identifier for MusicGen.
            description: Text description to condition generation.
            duration: Desired audio length in seconds.
            top_k: Top-k sampling parameter.
            top_p: Nucleus sampling probability.
            temperature: Sampling temperature.
            cfg_coef: Classifier-free guidance coefficient.
            melody: Optional ``(sample_rate, ndarray)`` tuple for chroma conditioning.
            use_diffusion: Whether to apply MultiBand Diffusion post-processing.

        Returns:
            Audio tensor of shape ``(batch, channels, samples)``.
        """
        model = self._load_model(model_version)
        model.set_generation_params(
            duration=duration,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            cfg_coef=cfg_coef,
        )

        melody_tensor = None
        if melody is not None:
            melody_tensor = self._prepare_melody(model, melody, duration)

        try:
            if melody_tensor is not None:
                output = model.generate_with_chroma(
                    descriptions=[description],
                    melody_wavs=[melody_tensor],
                    melody_sample_rate=model.sample_rate,
                    progress=True,
                    return_tokens=use_diffusion,
                )
            else:
                output = model.generate(
                    descriptions=[description],
                    progress=True,
                    return_tokens=use_diffusion,
                )
        except RuntimeError as exc:
            raise MusicGenerationError(f"Music generation failed: {exc}") from exc
        finally:
            if melody_tensor is not None:
                del melody_tensor

        audio_batch = self._post_process_output(model, output, use_diffusion)
        audio_batch = audio_batch.detach().cpu().float()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio_batch

    @property
    def sample_rate(self) -> int:
        """Return the sample rate of the currently loaded model."""
        if self._model is None:
            raise MusicGenerationError("MusicGen model is not loaded.")
        return int(self._model.sample_rate)

    def _load_model(self, model_version: str) -> MusicGen:
        if self._model is not None and self._model_version == model_version:
            return self._model

        if self._model is not None:
            del self._model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._model = MusicGen.get_pretrained(model_version, device=self.device)
        self._model_version = model_version
        return self._model

    def _prepare_melody(
        self,
        model: MusicGen,
        melody: MelodyInput,
        duration: float,
    ) -> torch.Tensor:
        try:
            sample_rate, data = melody
        except (TypeError, ValueError) as exc:
            raise MusicGenerationError("Invalid melody input format.") from exc

        melody_tensor = torch.from_numpy(data).to(model.device).float().t()
        if melody_tensor.dim() == 1:
            melody_tensor = melody_tensor.unsqueeze(0)

        max_length = int(sample_rate * duration)
        melody_tensor = melody_tensor[..., :max_length]

        try:
            result: torch.Tensor = convert_audio(
                melody_tensor,
                sample_rate,
                model.sample_rate,
                model.audio_channels,
            )
            return result
        except RuntimeError as exc:
            raise MusicGenerationError(f"Melody preprocessing failed: {exc}") from exc

    def _post_process_output(
        self,
        model: MusicGen,
        output: tp.Any,
        use_diffusion: bool,
    ) -> torch.Tensor:
        if not use_diffusion:
            return output[0]  # type: ignore[no-any-return]

        diffusion = self._load_diffusion()
        audio_batch, tokens = output

        if isinstance(model.compression_model, InterleaveStereoCompressionModel):
            left, right = model.compression_model.get_left_right_codes(tokens)
            tokens = torch.cat([left, right])

        diffusion_audio = diffusion.tokens_to_wav(tokens)
        if isinstance(model.compression_model, InterleaveStereoCompressionModel):
            diffusion_audio = rearrange(diffusion_audio, "(s b) c t -> b (s c) t", s=2)

        return torch.cat([audio_batch, diffusion_audio], dim=0)  # type: ignore[no-any-return]

    def _load_diffusion(self) -> MultiBandDiffusion:
        if self._diffusion is None:
            self._diffusion = MultiBandDiffusion.get_mbd_musicgen(device=self.device)
        return self._diffusion
