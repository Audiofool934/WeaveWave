# CLAUDE.md — WeaveWave

## Project Summary

WeaveWave is a multimodal music generation system that converts text, images, and video into music. It uses a **text-bridging pipeline**: a multimodal LLM (Gemma-3-12b-it) generates rich music descriptions from any input modality, then MusicGen (AudioCraft) synthesizes audio conditioned on those descriptions.

## Quick Commands

```bash
# Install (editable, with dev + demo extras)
pip install -e ".[dev,demo]"
pip install -e repos/audiocraft

# Prepare dataset (dummy for testing)
weavewave-prepare-data --create_dummy --dummy_samples 100

# Train
weavewave-train
# or with options:
weavewave-run-training --dummy_data --dummy_samples 100 --gpu 0 1

# Evaluate
weavewave-evaluate --eval_text2music --model_path ./outputs/latest_model --output_dir ./outputs/evaluation --gpu 0

# Demo app (two terminals)
weavewave-mllm-server    # MLLM backend on :8001
weavewave-demo           # Gradio UI on :7860

# Lint & test
ruff check weavewave/ tests/
ruff format --check weavewave/ tests/
pytest tests/ -v -m "not gpu"
```

## Architecture

```
Input (Text / Image / Video)
  → MLLM Backend (Gemma-3-12b-it, FastAPI :8001)
    → Music description text
  → MusicGen (AudioCraft)
    → Text conditioning (T5-base, frozen)
    → Style conditioning (MERT, 6 codebooks, 5Hz, 3s excerpts)
    → Optional melody conditioning (chroma features)
    → EnCodec 32kHz decoder
    → Optional MultiBand Diffusion post-processing
  → WAV output (32kHz, loudness-normalized)
```

**Package structure:**
- `weavewave/core/` — Config, types, logging, MLLM client, MusicGen wrapper
- `weavewave/training/` — Training script and runner
- `weavewave/evaluation/` — Multi-mode evaluation
- `weavewave/data/` — Dataset preparation
- `weavewave/serving/` — FastAPI MLLM backend + Gradio frontend
- `config/` — Hydra YAML configs (training, model architecture)
- `repos/audiocraft/` — Meta AudioCraft (git submodule)
- `tests/` — Test suite (pytest)
- `demo/app/assets/` — Demo media files (images, videos, audio)

**Config system:** Hydra + OmegaConf. Main config at `config/musicgen_style_32khz.yaml`, model scales at `config/model/musicgen_model.yaml`.

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `WEAVEWAVE_MLLM_URL` | MLLM service endpoint | `http://127.0.0.1:8001` |
| `WEAVEWAVE_DEFAULT_MUSIC_MODEL` | Default MusicGen checkpoint | `facebook/musicgen-stereo-melody-large` |
| `WEAVEWAVE_MLLM_MODEL` | MLLM model identifier | `google/gemma-3-12b-it` |
| `WEAVEWAVE_MLLM_DEVICE` | MLLM device | `cuda` |
| `CUDA_VISIBLE_DEVICES` | GPU selection | — |

## Conventions

- **Python 3.9+**, PyTorch 2.3.1, Transformers >=4.50
- **English comments only** throughout the codebase
- Linting: ruff (configured in `pyproject.toml`)
- Type checking: mypy
- Testing: pytest with `gpu` marker for GPU-dependent tests
- Training config uses 32kHz sample rate, batch size 8, AdamW optimizer, cosine LR schedule
- Demo auto-cleans temp audio files after 1 hour
- Package installed via `pyproject.toml` with entry-point scripts
