"""Tests for weavewave.data.prepare_dataset."""

from __future__ import annotations

import json

from weavewave.data.prepare_dataset import MultimodalMusicDataProcessor


class TestDummyDataset:
    def test_creates_split_directories(self, tmp_output):
        MultimodalMusicDataProcessor(str(tmp_output), sample_rate=16000)
        for split in ("train", "valid", "test"):
            assert (tmp_output / split).is_dir()

    def test_dummy_dataset_sample_count(self, tmp_output):
        proc = MultimodalMusicDataProcessor(str(tmp_output))
        proc.prepare_dummy_dataset(num_samples=20)

        total = sum(1 for _ in tmp_output.rglob("*.json"))
        assert total == 20

    def test_dummy_dataset_split_ratios(self, tmp_output):
        proc = MultimodalMusicDataProcessor(str(tmp_output))
        proc.prepare_dummy_dataset(num_samples=100)

        train_count = len(list((tmp_output / "train").glob("*.json")))
        valid_count = len(list((tmp_output / "valid").glob("*.json")))
        test_count = len(list((tmp_output / "test").glob("*.json")))

        assert train_count + valid_count + test_count == 100
        assert train_count >= 70
        assert valid_count >= 10
        assert test_count >= 10

    def test_metadata_structure(self, tmp_output):
        proc = MultimodalMusicDataProcessor(str(tmp_output))
        proc.prepare_dummy_dataset(num_samples=5)

        json_files = list(tmp_output.rglob("*.json"))
        assert len(json_files) > 0

        data = json.loads(json_files[0].read_text())
        assert "id" in data
        assert "description" in data
        assert "duration" in data
        assert "audio_features" in data
        assert "tempo" in data["audio_features"]
        assert "key" in data["audio_features"]

    def test_real_dataset_nonexistent_source(self, tmp_output):
        proc = MultimodalMusicDataProcessor(str(tmp_output))
        import pytest

        with pytest.raises(FileNotFoundError):
            proc.process_real_dataset("/nonexistent/path")
