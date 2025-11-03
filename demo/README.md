# WeaveWave Demo

This directory hosts the interactive demo for the WeaveWave project. It combines a
multimodal large language model (MLLM) to craft music descriptions and a MusicGen
backend to synthesise audio on demand via a Gradio UI.

## Layout

- `weavewave_app.py` &mdash; Gradio entrypoint that wires the UI to the core modules.
- `mllm_fastapi.py` &mdash; FastAPI server exposing `/describe_*` endpoints backed by MiniCPM-o.
- `core/`
  - `config.py` &mdash; Consolidates default prompts, port/URL settings, and the list of available MusicGen checkpoints. Environment variables such as `WEAVEWAVE_MLLM_URL` or `WEAVEWAVE_DEFAULT_MUSIC_MODEL` can override the defaults.
  - `mllm_client.py` &mdash; Minimal HTTP client that talks to the MLLM service and handles media encoding.
  - `music_generator.py` &mdash; Wrapper around AudioCraft’s MusicGen (and optional MultiBand Diffusion) with helpers to preprocess melody conditioning.
- `theme_wave.py` &mdash; Custom Gradio theme used by the UI.
- `assets/` &mdash; Example media files used by the demo.
- `alt/` &mdash; Earlier iterations of the demo kept for reference.

## Running the demo

1. Launch the MLLM server:
   ```bash
   cd demo/app
   python mllm_fastapi.py
   ```
   Adjust `MODEL_PATH` in `mllm_fastapi.py` if your MiniCPM-o weights live elsewhere.

2. In a second terminal, start the Gradio UI:
   ```bash
   cd demo/app
   python weavewave_app.py --listen 0.0.0.0 --server_port 7860
   ```

3. Open the reported URL in your browser and generate music from text, image, or video
   prompts. The app will call the MLLM to obtain a description and then trigger MusicGen
   locally.

## Configuration notes

- **MLLM Endpoint**: set `WEAVEWAVE_MLLM_URL` before launching the UI if the API runs on a
  different host or port.
- **Model selection**: update `WEAVEWAVE_DEFAULT_MUSIC_MODEL` to change the preselected
  MusicGen checkpoint. Extend `AppConfig.available_music_models` if you want to expose
  custom checkpoints in the dropdown.
- **Prompt templates**: customise `AppConfig.prompts` (text/image/video keys) in
  `core/config.py` to tune description style for different tasks.
- **Flashy dependency**: the AudioCraft stack depends on Meta's `flashy` utilities. The
  requirements file references the official GitHub repo (`flashy @ git+https://...`), so
  make sure outbound network and Git are available during installation.
- **Hydra/OmegaConf versions**: the project now targets modern releases
  (`hydra-core>=1.3.2`, `omegaconf>=2.3.0`). If you add libraries that depend on older
  Hydra (for example, legacy `fairseq`), you may need to pin compatible versions or
  install them in a separate environment.
- **Optional deps**: if you plan to run legacy training utilities that rely on
  `fairseq`, install it manually after setting up the base environment to avoid
  dependency conflicts.
- **Transformers/PyTorch**: The stack targets PyTorch 2.3.1 (with the matching
  TorchVision/Torchaudio releases) to stay compatible with Transformers `>=4.50.0`. If you
  upgrade Transformers further, keep these three PyTorch packages aligned with the same
  release number.

## Extending the pipeline

- For a new MLLM, implement a FastAPI-compatible server and point the UI to it by updating
  `WEAVEWAVE_MLLM_URL`. The `MLLMClient` only requires the same JSON contract.
- To integrate another music generator, create a drop-in replacement for
  `MusicGenerator` that exposes a `generate` method returning a batch of waveforms.
- Additional decoding backends (e.g., diffusion variants) can be introduced by extending
  `AppConfig.diffusion_decoder_label` and adjusting the `MusicGenerator` to return the
  appropriate audio slices.
