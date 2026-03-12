#!/usr/bin/env python3
# Copyright (c) 2023-2024 WeaveWave Team.
# All rights reserved.

"""Evaluation script for trained MusicGen-Style models."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import torch
import torchaudio

from weavewave.core.logging import setup_logging
from weavewave.core.types import GenerationConfig

logger = setup_logging(name=__name__)


class MusicGenStyleEvaluator:
    """Evaluator for MusicGen-Style models.

    Args:
        model_path: Path to a local model checkpoint or a HuggingFace identifier.
        device: Torch device string.  Auto-detected when *None*.
        generation_config: Parameters controlling the generation process.
    """

    DEFAULT_FALLBACK_MODEL = "facebook/musicgen-style"

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gen_cfg = generation_config or GenerationConfig()

        logger.info("Evaluator initialised — model: %s, device: %s", self.model_path, self.device)

        try:
            from audiocraft.models import MusicGen

            self._MusicGen = MusicGen
        except ImportError as exc:
            raise ImportError("Could not import MusicGen. Ensure audiocraft is installed.") from exc

        self._load_model()

    def _load_model(self) -> None:
        """Load a MusicGen model from a local path or HuggingFace."""
        logger.info("Loading model")
        if self.model_path.exists():
            self.model = self._MusicGen.get_pretrained(str(self.model_path))
            logger.info("Loaded local model from %s", self.model_path)
        else:
            logger.warning(
                "Local path %s not found — falling back to %s",
                self.model_path,
                self.DEFAULT_FALLBACK_MODEL,
            )
            self.model = self._MusicGen.get_pretrained(self.DEFAULT_FALLBACK_MODEL)

        self.model.set_generation_params(
            duration=self.gen_cfg.duration,
            use_sampling=self.gen_cfg.use_sampling,
            top_k=self.gen_cfg.top_k,
            top_p=self.gen_cfg.top_p,
            cfg_coef=self.gen_cfg.cfg_coef,
            cfg_coef_beta=self.gen_cfg.cfg_coef_beta,
        )

        self.model.set_style_conditioner_params(eval_q=1, excerpt_length=3.0)
        self.model.to(self.device)

    def evaluate_text_to_music(
        self,
        texts: Sequence[str],
        output_dir: str | Path,
    ) -> list[Path]:
        """Generate audio from text descriptions and save to *output_dir*.

        Args:
            texts: Text descriptions to condition generation.
            output_dir: Directory to write generated WAV files.

        Returns:
            List of paths to the saved audio files.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        logger.info("Text-to-music evaluation — %d samples", len(texts))

        with torch.no_grad():
            wav = self.model.generate(list(texts))

        paths: list[Path] = []
        for i, one_wav in enumerate(wav):
            p = out / f"text2music_{i + 1}.wav"
            torchaudio.save(str(p), one_wav.cpu(), self.model.sample_rate)
            logger.info("Saved: %s", p)
            paths.append(p)

        return paths

    def evaluate_style_to_music(
        self,
        audio_paths: Sequence[str | Path],
        output_dir: str | Path,
    ) -> list[Path]:
        """Generate audio conditioned on style (chroma features).

        Args:
            audio_paths: Paths to reference audio files.
            output_dir: Directory to write generated WAV files.

        Returns:
            List of paths to the saved audio files.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        logger.info("Style-to-music evaluation — %d samples", len(audio_paths))

        paths: list[Path] = []
        for i, audio_path in enumerate(audio_paths):
            try:
                melody, sr = torchaudio.load(str(audio_path))
                if melody.dim() == 1:
                    melody = melody.unsqueeze(0)

                with torch.no_grad():
                    wav = self.model.generate_with_chroma(
                        descriptions=[None],
                        melody=melody.to(self.device),
                        melody_sample_rate=sr,
                    )

                p = out / f"style2music_{i + 1}.wav"
                torchaudio.save(str(p), wav[0].cpu(), self.model.sample_rate)
                logger.info("Saved: %s", p)
                paths.append(p)
            except Exception:
                logger.exception("Error processing %s", audio_path)

        return paths

    def evaluate_style_and_text_to_music(
        self,
        texts: Sequence[str],
        audio_paths: Sequence[str | Path],
        output_dir: str | Path,
    ) -> list[Path]:
        """Generate audio conditioned on both style and text.

        Args:
            texts: Text descriptions (must match *audio_paths* in length).
            audio_paths: Paths to reference audio files.
            output_dir: Directory to write generated WAV files.

        Returns:
            List of paths to the saved audio files.

        Raises:
            ValueError: If *texts* and *audio_paths* differ in length.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if len(texts) != len(audio_paths):
            raise ValueError(
                f"texts ({len(texts)}) and audio_paths ({len(audio_paths)}) must have equal length"
            )

        logger.info("Style+text-to-music evaluation — %d samples", len(texts))

        paths: list[Path] = []
        for i, (text, audio_path) in enumerate(zip(texts, audio_paths)):
            try:
                melody, sr = torchaudio.load(str(audio_path))
                if melody.dim() == 1:
                    melody = melody.unsqueeze(0)

                with torch.no_grad():
                    wav = self.model.generate_with_chroma(
                        descriptions=[text],
                        melody=melody.to(self.device),
                        melody_sample_rate=sr,
                    )

                p = out / f"style_and_text2music_{i + 1}.wav"
                torchaudio.save(str(p), wav[0].cpu(), self.model.sample_rate)
                logger.info("Saved: %s", p)
                paths.append(p)
            except Exception:
                logger.exception("Error processing sample %d", i + 1)

        return paths


DEFAULT_TEXTS = [
    "Upbeat electronic dance music with energetic synthesizers",
    "Calm piano melody with gentle strings in the background",
    "Heavy rock with distorted guitars and powerful drums",
    "Jazz fusion with smooth saxophone solo and walking bass line",
    "Orchestral cinematic music with epic brass section",
]


def main() -> None:
    """Entry point for ``weavewave-evaluate``."""
    parser = argparse.ArgumentParser(description="WeaveWave evaluation")

    parser.add_argument(
        "--model_path", type=str, default="./outputs/latest_model", help="Model path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/evaluation",
        help="Evaluation output directory",
    )
    parser.add_argument("--gpu", type=int, default=None, help="GPU device ID")

    eval_group = parser.add_argument_group("evaluation modes")
    eval_group.add_argument("--eval_text2music", action="store_true")
    eval_group.add_argument("--eval_style2music", action="store_true")
    eval_group.add_argument("--eval_style_and_text2music", action="store_true")

    data_group = parser.add_argument_group("evaluation data")
    data_group.add_argument(
        "--text_file", type=str, default=None, help="File with one description per line"
    )
    data_group.add_argument("--audio_dir", type=str, default=None, help="Directory of audio files")

    args = parser.parse_args()
    logger.info("=== WeaveWave Evaluation ===")

    device: str | None = None
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = "cuda"
        logger.info("Using GPU %d", args.gpu)

    evaluator = MusicGenStyleEvaluator(args.model_path, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load text descriptions
    if args.text_file:
        texts = [
            line.strip() for line in Path(args.text_file).read_text().splitlines() if line.strip()
        ]
        logger.info("Loaded %d descriptions from %s", len(texts), args.text_file)
    else:
        texts = DEFAULT_TEXTS
        logger.info("Using %d default text descriptions", len(texts))

    # Load audio paths
    audio_paths: list[Path] = []
    if args.audio_dir:
        audio_dir = Path(args.audio_dir)
        if audio_dir.exists():
            audio_paths = sorted(list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3")))
            logger.info("Found %d audio files in %s", len(audio_paths), args.audio_dir)

    needs_audio = args.eval_style2music or args.eval_style_and_text2music
    if not audio_paths and needs_audio:
        raise ValueError("Audio files required for style evaluation — use --audio_dir")

    if args.eval_text2music:
        evaluator.evaluate_text_to_music(texts, output_dir / "text2music")

    if args.eval_style2music and audio_paths:
        evaluator.evaluate_style_to_music(audio_paths, output_dir / "style2music")

    if args.eval_style_and_text2music and audio_paths:
        # Align text count to audio count
        if len(texts) >= len(audio_paths):
            aligned_texts = texts[: len(audio_paths)]
        else:
            aligned_texts = (texts * (len(audio_paths) // len(texts) + 1))[: len(audio_paths)]
        evaluator.evaluate_style_and_text_to_music(
            aligned_texts, audio_paths, output_dir / "style_and_text2music"
        )

    logger.info("=== Evaluation complete ===")


if __name__ == "__main__":
    main()
