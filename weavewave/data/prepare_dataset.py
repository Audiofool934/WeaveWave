#!/usr/bin/env python3
# Copyright (c) 2023-2024 WeaveWave Team.
# All rights reserved.

"""Dataset preparation script for multimodal music generation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from weavewave.core.logging import setup_logging

logger = setup_logging(name=__name__)

EXAMPLE_DESCRIPTIONS = [
    "Upbeat electronic dance music with energetic synthesizers",
    "Calm piano melody with gentle strings in the background",
    "Heavy rock with distorted guitars and powerful drums",
    "Jazz fusion with smooth saxophone solo and walking bass line",
    "Orchestral cinematic music with epic brass section",
    "Ambient soundscape with atmospheric pads and subtle percussion",
    "Funk groove with slap bass and wah-wah guitar",
    "Classical string quartet with emotional violin solo",
    "Hip hop beat with deep bass and trap hi-hats",
    "Acoustic folk with finger-picked guitar and harmonica",
]


class MultimodalMusicDataProcessor:
    """Processor for multimodal music datasets.

    Args:
        output_dir: Root directory for the prepared dataset.
        sample_rate: Target audio sample rate in Hz.
    """

    def __init__(self, output_dir: str, sample_rate: int = 32000) -> None:
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for split in ("train", "valid", "test"):
            (self.output_dir / split).mkdir(exist_ok=True)

        logger.info(
            "Data processor initialised — output: %s, sample_rate: %d Hz",
            self.output_dir,
            self.sample_rate,
        )

        self._torchaudio: object | None = None

    def _get_torchaudio(self):
        """Lazy-import torchaudio."""
        if self._torchaudio is None:
            try:
                import torchaudio

                self._torchaudio = torchaudio
            except ImportError as exc:
                raise ImportError(
                    "torchaudio is required — install with: pip install torchaudio"
                ) from exc
        return self._torchaudio

    def process_audio_file(self, audio_path: str, target_path: str) -> bool:
        """Resample and convert an audio file to mono at the target sample rate.

        Args:
            audio_path: Source audio file.
            target_path: Destination file path.

        Returns:
            *True* on success, *False* on failure.
        """
        torchaudio = self._get_torchaudio()
        try:
            waveform, sr = torchaudio.load(audio_path)

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            torchaudio.save(target_path, waveform, self.sample_rate)
            return True
        except Exception:
            logger.exception("Error processing audio file %s", audio_path)
            return False

    def prepare_dummy_dataset(self, num_samples: int = 10) -> None:
        """Create a synthetic dataset for testing the training pipeline.

        Args:
            num_samples: Total number of dummy samples to generate.
        """
        logger.info("Creating dummy dataset with %d samples", num_samples)

        splits = {
            "train": int(num_samples * 0.7),
            "valid": int(num_samples * 0.15),
            "test": int(num_samples * 0.15),
        }
        splits["train"] += num_samples - sum(splits.values())

        for split, count in splits.items():
            logger.info("Creating %d samples for '%s' split", count, split)
            split_dir = self.output_dir / split

            for i in range(count):
                sample_id = f"{split}_{i + 1:04d}"
                desc = random.choice(EXAMPLE_DESCRIPTIONS)

                metadata = {
                    "id": sample_id,
                    "description": desc,
                    "tags": ["dummy", "sample", split],
                    "duration": round(random.uniform(5.0, 10.0), 2),
                    "audio_features": {
                        "tempo": random.randint(60, 180),
                        "key": random.choice(list("CDEFGAB")),
                        "mode": random.choice(["major", "minor"]),
                    },
                }

                (split_dir / f"{sample_id}.json").write_text(json.dumps(metadata, indent=2))
                (split_dir / f"{sample_id}.txt").write_text(
                    f"Placeholder for dummy audio file.\nDescription: {desc}"
                )

        logger.info("Dummy dataset created")

    def process_real_dataset(
        self,
        source_dir: str,
        split_ratio: list[float] | None = None,
    ) -> None:
        """Process a real dataset from *source_dir*.

        Args:
            source_dir: Source directory containing audio files and metadata.
            split_ratio: Train / valid / test ratios (default ``[0.8, 0.1, 0.1]``).
        """
        if split_ratio is None:
            split_ratio = [0.8, 0.1, 0.1]

        source = Path(source_dir)
        if not source.exists():
            raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

        logger.info(
            "Processing dataset from %s (split: %.0f/%.0f/%.0f)",
            source,
            split_ratio[0] * 100,
            split_ratio[1] * 100,
            split_ratio[2] * 100,
        )
        # Real dataset processing would go here.
        logger.info("Dataset processing complete")


def main() -> None:
    """Entry point for ``weavewave-prepare-data``."""
    parser = argparse.ArgumentParser(description="WeaveWave dataset preparation")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/multimodal_music_dataset",
        help="Output dataset directory",
    )
    parser.add_argument("--sample_rate", type=int, default=32000, help="Target sample rate (Hz)")
    parser.add_argument(
        "--create_dummy", action="store_true", help="Create a dummy dataset for testing"
    )
    parser.add_argument("--dummy_samples", type=int, default=10, help="Number of dummy samples")
    parser.add_argument(
        "--source_dir", type=str, default=None, help="Source directory for real data"
    )
    args = parser.parse_args()

    logger.info("=== WeaveWave Dataset Preparation ===")

    processor = MultimodalMusicDataProcessor(
        output_dir=args.output_dir, sample_rate=args.sample_rate
    )

    if args.create_dummy:
        processor.prepare_dummy_dataset(num_samples=args.dummy_samples)
    elif args.source_dir:
        processor.process_real_dataset(source_dir=args.source_dir)
    else:
        logger.warning("No action specified. Use --create_dummy or --source_dir.")

    logger.info("=== Preparation complete ===")


if __name__ == "__main__":
    main()
