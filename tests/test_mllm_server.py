"""Tests for weavewave.serving.mllm_server (unit-level, no GPU)."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")

from weavewave.core.types import MLLMServerConfig  # noqa: E402
from weavewave.serving.mllm_server import SYSTEM_PROMPTS, MLLMService  # noqa: E402


class TestSystemPrompts:
    def test_all_modalities_present(self):
        assert set(SYSTEM_PROMPTS.keys()) == {"text", "image", "video"}

    def test_prompts_are_nonempty(self):
        for key, prompt in SYSTEM_PROMPTS.items():
            assert len(prompt) > 50, f"Prompt for {key} is suspiciously short"

    def test_prompts_mention_music(self):
        for key, prompt in SYSTEM_PROMPTS.items():
            assert "music" in prompt.lower(), f"Prompt for {key} should mention music"


class TestMLLMServerConfig:
    def test_defaults(self):
        cfg = MLLMServerConfig()
        assert cfg.model_name == "google/gemma-3-12b-it"
        assert cfg.port == 8001
        assert cfg.max_num_frames == 32

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("WEAVEWAVE_MLLM_MODEL", "custom/model")
        cfg = MLLMServerConfig()
        assert cfg.model_name == "custom/model"


class TestMLLMService:
    def test_is_ready_false_before_load(self):
        cfg = MLLMServerConfig()
        svc = MLLMService(cfg)
        assert not svc.is_ready
