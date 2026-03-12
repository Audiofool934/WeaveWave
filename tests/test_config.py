"""Tests for weavewave.core.config."""

from __future__ import annotations

from weavewave.core.config import AppConfig, PromptConfig


class TestPromptConfig:
    def test_get_known_task(self):
        cfg = PromptConfig()
        assert "music description" in cfg.get("text").lower()

    def test_get_unknown_returns_empty(self):
        cfg = PromptConfig()
        assert cfg.get("nonexistent") == ""

    def test_get_is_case_insensitive(self):
        cfg = PromptConfig()
        assert cfg.get("IMAGE") == cfg.get("image")


class TestAppConfig:
    def test_default_mllm_url(self):
        cfg = AppConfig()
        assert cfg.mllm_api_url == "http://127.0.0.1:8001"

    def test_mllm_url_from_env(self, monkeypatch):
        monkeypatch.setenv("WEAVEWAVE_MLLM_URL", "http://custom:9000")
        cfg = AppConfig()
        assert cfg.mllm_api_url == "http://custom:9000"

    def test_default_music_model_from_env(self, monkeypatch):
        monkeypatch.setenv("WEAVEWAVE_DEFAULT_MUSIC_MODEL", "facebook/musicgen-small")
        cfg = AppConfig()
        assert cfg.default_music_model == "facebook/musicgen-small"

    def test_task_from_media_type(self, app_config):
        assert app_config.task_from_media_type("Image") == "image"
        assert app_config.task_from_media_type("Video") == "video"
        assert app_config.task_from_media_type("Text") == "text"
        assert app_config.task_from_media_type("") == "text"
        assert app_config.task_from_media_type("unknown") == "text"

    def test_endpoint_for_task(self, app_config):
        assert app_config.endpoint_for_task("image") == "/describe_image/"
        assert app_config.endpoint_for_task("video") == "/describe_video/"
        assert app_config.endpoint_for_task("text") == "/describe_text/"

    def test_compose_prompt_returns_user_input_when_given(self, app_config):
        result = app_config.compose_prompt("text", "my custom prompt")
        assert result == "my custom prompt"

    def test_compose_prompt_falls_back_to_template(self, app_config):
        result = app_config.compose_prompt("image", "")
        assert "mood" in result.lower() or "music" in result.lower()

    def test_available_models_not_empty(self, app_config):
        assert len(app_config.available_music_models) > 0
