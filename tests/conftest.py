"""Shared fixtures for WeaveWave tests."""

from __future__ import annotations

import pytest

from weavewave.core.config import AppConfig


@pytest.fixture
def app_config() -> AppConfig:
    """Return a fresh ``AppConfig`` with default values."""
    return AppConfig()


@pytest.fixture
def tmp_output(tmp_path):
    """Return a temporary output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out
