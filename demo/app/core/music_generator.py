from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from einops import rearrange

from audiocraft.data.audio_utils import convert_audio
from audiocraft.models import MusicGen, MultiBandDiffusion
from audiocraft.models.encodec import InterleaveStereoCompressionModel


MelodyInput = Tuple[int, np.ndarray]


class MusicGenerationError(RuntimeError):
    """Raised when music generation fails."""


@dataclass
class MusicGenerator:
    """Wrapper around MusicGen + optional MultiBand Diffusion decoding."""

    device: Optional[str] = None

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model: Optional[MusicGen] = None
        self._model_version: Optional[str] = None
        self._diffusion: Optional[MultiBandDiffusion] = None

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
        melody: Optional[MelodyInput],
        use_diffusion: bool,
    ) -> torch.Tensor:
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
        if self._model is None:
            raise MusicGenerationError("MusicGen model is not loaded.")
        return self._model.sample_rate

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
            return convert_audio(
                melody_tensor,
                sample_rate,
                model.sample_rate,
                model.audio_channels,
            )
        except RuntimeError as exc:
            raise MusicGenerationError(f"Melody preprocessing failed: {exc}") from exc

    def _post_process_output(
        self,
        model: MusicGen,
        output,
        use_diffusion: bool,
    ) -> torch.Tensor:
        if not use_diffusion:
            return output[0]

        diffusion = self._load_diffusion()
        audio_batch, tokens = output

        if isinstance(model.compression_model, InterleaveStereoCompressionModel):
            left, right = model.compression_model.get_left_right_codes(tokens)
            tokens = torch.cat([left, right])

        diffusion_audio = diffusion.tokens_to_wav(tokens)
        if isinstance(model.compression_model, InterleaveStereoCompressionModel):
            diffusion_audio = rearrange(
                diffusion_audio, "(s b) c t -> b (s c) t", s=2
            )

        return torch.cat([audio_batch, diffusion_audio], dim=0)

    def _load_diffusion(self) -> MultiBandDiffusion:
        if self._diffusion is None:
            self._diffusion = MultiBandDiffusion.get_mbd_musicgen(device=self.device)
        return self._diffusion
