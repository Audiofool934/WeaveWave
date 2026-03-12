"""Tests for weavewave.core.mllm_client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from weavewave.core.config import AppConfig
from weavewave.core.mllm_client import MLLMClient, MLLMClientError


@pytest.fixture
def client():
    cfg = AppConfig()
    return MLLMClient(cfg)


class TestMLLMClient:
    def test_encode_media_file_not_found(self, client):
        with pytest.raises(MLLMClientError, match="not found"):
            client._encode_media("/nonexistent/file.jpg")

    def test_encode_media_success(self, client, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
        result = client._encode_media(str(img))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_describe_raises_without_media_path(self, client):
        with pytest.raises(MLLMClientError, match="Media path is required"):
            client.describe("image", None, "prompt")

    @patch("weavewave.core.mllm_client.requests.Session")
    def test_describe_text_success(self, mock_session_cls):
        mock_response = MagicMock()
        mock_response.json.return_value = {"description": "A calm piano piece"}
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_session_cls.return_value = mock_session

        cfg = AppConfig()
        client = MLLMClient(cfg)
        client._session = mock_session

        result = client.describe("text", None, "something calm")
        assert result == "A calm piano piece"

    @patch("weavewave.core.mllm_client.requests.Session")
    def test_describe_network_error(self, mock_session_cls):
        import requests

        mock_session = MagicMock()
        mock_session.post.side_effect = requests.ConnectionError("refused")
        mock_session_cls.return_value = mock_session

        cfg = AppConfig()
        client = MLLMClient(cfg)
        client._session = mock_session

        with pytest.raises(MLLMClientError, match="MLLM request failed"):
            client.describe("text", None, "prompt")

    @patch("weavewave.core.mllm_client.requests.Session")
    def test_describe_empty_description(self, mock_session_cls):
        mock_response = MagicMock()
        mock_response.json.return_value = {"description": ""}
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_session_cls.return_value = mock_session

        cfg = AppConfig()
        client = MLLMClient(cfg)
        client._session = mock_session

        with pytest.raises(MLLMClientError, match="did not include"):
            client.describe("text", None, "prompt")
