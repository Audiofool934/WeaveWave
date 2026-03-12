"""Gradio frontend for the WeaveWave demo."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import typing as tp
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile

import gradio as gr
import torch
from audiocraft.data.audio import audio_write

from weavewave.core.config import AppConfig
from weavewave.core.mllm_client import MLLMClient, MLLMClientError
from weavewave.core.music_generator import MusicGenerationError, MusicGenerator
from weavewave.serving.theme import css, theme

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application state (replaces module-level globals)
# ---------------------------------------------------------------------------
@dataclass
class AppState:
    """Encapsulates all mutable runtime state for the Gradio app."""

    config: AppConfig = field(default_factory=AppConfig)
    mllm_client: MLLMClient = field(init=False)
    music_generator: MusicGenerator = field(default_factory=MusicGenerator)
    interrupting: bool = False

    def __post_init__(self) -> None:
        self.mllm_client = MLLMClient(self.config)


class FileCleaner:
    """Remove temporary audio files after a fixed lifetime."""

    def __init__(self, file_lifetime: float = 3600) -> None:
        self.file_lifetime = file_lifetime
        self.files: list[tuple[float, Path]] = []

    def add(self, path: str | Path) -> None:
        self._cleanup()
        self.files.append((time.time(), Path(path)))

    def _cleanup(self) -> None:
        now = time.time()
        while self.files and now - self.files[0][0] > self.file_lifetime:
            _, path = self.files.pop(0)
            if path.exists():
                try:
                    path.unlink()
                except OSError as exc:
                    logger.warning("Error deleting %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_waveform(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gr.make_waveform(*args, **kwargs)


def _select_media_path(media_type: str, image_path: str, video_path: str) -> str | None:
    if media_type == "Image":
        return image_path or None
    if media_type == "Video":
        return video_path or None
    return None


def _save_audio_batch(
    audio_batch: torch.Tensor,
    sample_rate: int,
    file_cleaner: FileCleaner,
) -> list[str]:
    if audio_batch.numel() == 0:
        raise gr.Error("Music generation returned an empty result.")

    paths: list[str] = []
    for audio in audio_batch:
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as f:
            audio_write(
                f.name,
                audio,
                sample_rate,
                strategy="loudness",
                loudness_headroom_db=16,
                loudness_compressor=True,
                add_suffix=False,
            )
            paths.append(f.name)
            file_cleaner.add(f.name)
    return paths


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------
def _predict_full(
    state: AppState,
    file_cleaner: FileCleaner,
    model_version: str,
    media_type: str,
    image_input: str,
    video_input: str,
    text_prompt: str,
    melody: tp.Any,
    duration: float,
    topk: int,
    topp: float,
    temperature: float,
    cfg_coef: float,
    decoder: str,
    progress: gr.Progress | None = None,
) -> tuple[str, str | None]:
    if progress is None:
        progress = gr.Progress()
    state.interrupting = False
    use_diffusion = decoder == state.config.diffusion_decoder_label
    media_path = _select_media_path(media_type, image_input, video_input)

    # 1. Description
    progress(progress=None, desc="Generating music description...")
    task_type = media_type if media_path else "text"
    description = None

    try:
        description = state.mllm_client.describe(task_type, media_path, text_prompt)
    except MLLMClientError as exc:
        if media_path:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise gr.Error(str(exc)) from exc
        description = state.config.compose_prompt("text", text_prompt)

    # 2. Generate
    progress(progress=None, desc="Running MusicGen...")
    audio_batch = None
    try:
        audio_batch = state.music_generator.generate(
            model_version=model_version,
            description=description,
            duration=float(duration),
            top_k=int(topk),
            top_p=float(topp),
            temperature=float(temperature),
            cfg_coef=float(cfg_coef),
            melody=melody if melody else None,
            use_diffusion=use_diffusion,
        )
    except MusicGenerationError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise gr.Error(f"Unexpected error during music generation: {exc}") from exc

    # 3. Save
    progress(progress=None, desc="Saving results...")
    try:
        audio_paths = _save_audio_batch(
            audio_batch, state.music_generator.sample_rate, file_cleaner
        )
    except Exception as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise gr.Error(f"Failed to save audio: {exc}") from exc
    finally:
        if audio_batch is not None:
            del audio_batch

    if use_diffusion and len(audio_paths) >= 2:
        return audio_paths[0], audio_paths[1]
    return audio_paths[0], None


# ---------------------------------------------------------------------------
# UI builder
# ---------------------------------------------------------------------------
def create_ui(
    state: AppState | None = None,
    launch_kwargs: dict | None = None,
) -> gr.Blocks:
    """Build and launch the Gradio interface.

    Args:
        state: Application state.  A default is created when *None*.
        launch_kwargs: Extra keyword arguments forwarded to ``gr.Blocks.launch``.

    Returns:
        The Gradio ``Blocks`` interface.
    """
    if state is None:
        state = AppState()
    if launch_kwargs is None:
        launch_kwargs = {}

    file_cleaner = FileCleaner()
    wave_theme = theme()

    def interrupt_handler() -> None:
        state.interrupting = True

    def predict_full(
        model_version,
        media_type,
        image_input,
        video_input,
        text_prompt,
        melody,
        duration,
        topk,
        topp,
        temperature,
        cfg_coef,
        decoder,
        progress=None,
    ):
        if progress is None:
            progress = gr.Progress()
        return _predict_full(
            state,
            file_cleaner,
            model_version,
            media_type,
            image_input,
            video_input,
            text_prompt,
            melody,
            duration,
            topk,
            topp,
            temperature,
            cfg_coef,
            decoder,
            progress,
        )

    with gr.Blocks(theme=wave_theme, css=css) as interface:
        gr.Markdown(
            """
            <div style="text-align: center;">
                <h1>WeaveWave</h1>
                <h2>Towards Multimodal Music Generation</h2>
            </div>
            """
        )

        with gr.Row():
            with gr.Column():
                with gr.Group():
                    image_input = gr.Image(
                        value="./assets/WeaveWave.png",
                        label="Input Image",
                        type="filepath",
                        height=320,
                        visible=True,
                    )
                    video_input = gr.Video(
                        value="./assets/example_video_1.mp4",
                        label="Input Video",
                        height=320,
                        visible=False,
                    )
                with gr.Row():
                    media_type = gr.Radio(
                        choices=["Image", "Video", "Text"],
                        value="Image",
                        label="",
                        interactive=True,
                        elem_classes="center-radio compact-radio",
                    )

                def toggle_media(choice: str) -> dict:
                    return {
                        image_input: gr.update(visible=(choice == "Image")),
                        video_input: gr.update(visible=(choice == "Video")),
                    }

                media_type.change(
                    toggle_media,
                    inputs=media_type,
                    outputs=[image_input, video_input],
                )

            with gr.Column():
                text_input = gr.Text(value="Anything you like", label="User Prompt")
                melody_input = gr.Audio(value="./assets/bach.mp3", type="numpy", label="Melody")
                with gr.Row():
                    submit_button = gr.Button("Generate Music", variant="primary")
                    interrupt_button = gr.Button("Interrupt", variant="stop")

        with gr.Row():
            model_version = gr.Dropdown(
                state.config.available_music_models,
                label="MusicGen Model",
                value=state.config.default_music_model,
            )
            duration = gr.Slider(minimum=1, maximum=120, value=10, label="Duration (seconds)")

        with gr.Row():
            topk = gr.Number(label="Top-k", value=250)
            topp = gr.Number(label="Top-p", value=0)
            temperature = gr.Number(label="Temperature", value=1.0)
            cfg_coef = gr.Number(label="Classifier-Free Guidance", value=3.0)
            decoder = gr.Dropdown(
                ["Default", state.config.diffusion_decoder_label],
                label="Decoder",
                value="Default",
                interactive=True,
            )

        with gr.Row():
            output_audio = gr.Audio(label="Generated Music", type="filepath")
            output_audio_mbd = gr.Audio(label=state.config.diffusion_decoder_label, type="filepath")

        submit_button.click(
            predict_full,
            inputs=[
                model_version,
                media_type,
                image_input,
                video_input,
                text_input,
                melody_input,
                duration,
                topk,
                topp,
                temperature,
                cfg_coef,
                decoder,
            ],
            outputs=[output_audio, output_audio_mbd],
        )
        interrupt_button.click(interrupt_handler, [], [])
        if state.interrupting:
            raise gr.Error("Interrupted.")

        gr.Examples(
            examples=[
                [
                    "Image",
                    "./assets/example_image_1.jpg",
                    None,
                    "Acoustic guitar solo. Country and folk music.",
                    None,
                    state.config.default_music_model,
                    10,
                    250,
                    0,
                    1.0,
                    3.0,
                    state.config.diffusion_decoder_label,
                ],
                [
                    "Video",
                    None,
                    "./assets/example_video_1.mp4",
                    "Space Rock, Synthwave, 80s. Electric guitar and Drums.",
                    None,
                    state.config.default_music_model,
                    10,
                    250,
                    0,
                    1.0,
                    3.0,
                    state.config.diffusion_decoder_label,
                ],
                [
                    "Text",
                    None,
                    None,
                    "An 80s driving pop song with heavy drums and synth pads in the background",
                    "./assets/bach.mp3",
                    state.config.default_music_model,
                    10,
                    250,
                    0,
                    1.0,
                    3.0,
                    state.config.diffusion_decoder_label,
                ],
            ],
            inputs=[
                media_type,
                image_input,
                video_input,
                text_input,
                melody_input,
                model_version,
                duration,
                topk,
                topp,
                temperature,
                cfg_coef,
                decoder,
            ],
        )

    interface.queue().launch(**launch_kwargs)
    return interface


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for ``weavewave-demo``."""
    parser = argparse.ArgumentParser(description="WeaveWave Gradio demo")
    parser.add_argument(
        "--listen",
        type=str,
        default="0.0.0.0" if "SPACE_ID" in os.environ else "127.0.0.1",
        help="IP to listen on",
    )
    parser.add_argument("--username", type=str, default="", help="Auth username")
    parser.add_argument("--password", type=str, default="", help="Auth password")
    parser.add_argument("--server_port", type=int, default=0, help="Server port")
    parser.add_argument("--inbrowser", action="store_true", help="Open in browser")
    parser.add_argument("--share", action="store_true", help="Share the Gradio UI")

    args = parser.parse_args()

    launch_kwargs: dict[str, tp.Any] = {"server_name": args.listen}
    if args.username and args.password:
        launch_kwargs["auth"] = (args.username, args.password)
    if args.server_port:
        launch_kwargs["server_port"] = args.server_port
    if args.inbrowser:
        launch_kwargs["inbrowser"] = args.inbrowser
    if args.share:
        launch_kwargs["share"] = args.share
    elif args.listen == "0.0.0.0":
        launch_kwargs["share"] = True

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    create_ui(launch_kwargs=launch_kwargs)


if __name__ == "__main__":
    main()
