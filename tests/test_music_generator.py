"""Tests for weavewave.core.music_generator (no GPU required)."""

from __future__ import annotations

import pytest

audiocraft = pytest.importorskip("audiocraft", reason="audiocraft not installed")

from weavewave.core.music_generator import MusicGenerationError, MusicGenerator  # noqa: E402


class TestMusicGenerator:
    def test_default_device_is_string(self):
        gen = MusicGenerator()
        assert gen.device in ("cuda", "cpu")

    def test_explicit_device(self):
        gen = MusicGenerator(device="cpu")
        assert gen.device == "cpu"

    def test_sample_rate_before_load_raises(self):
        gen = MusicGenerator(device="cpu")
        with pytest.raises(MusicGenerationError, match="not loaded"):
            _ = gen.sample_rate
