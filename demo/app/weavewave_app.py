import argparse
import logging
import os
import sys
import time
import typing as tp
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile

import gradio as gr
import torch

from audiocraft.data.audio import audio_write

from core import (
    AppConfig,
    MLLMClient,
    MLLMClientError,
    MusicGenerationError,
    MusicGenerator,
)
from theme_wave import css, theme

# --- Global State ---
INTERRUPTING = False
app_config = AppConfig()
mlmm_client = MLLMClient(app_config)
music_generator = MusicGenerator()


def interrupt():
    global INTERRUPTING
    INTERRUPTING = True


class FileCleaner:
    def __init__(self, file_lifetime: float = 3600):
        self.file_lifetime = file_lifetime
        self.files = []

    def add(self, path: tp.Union[str, Path]):
        self._cleanup()
        self.files.append((time.time(), Path(path)))

    def _cleanup(self):
        now = time.time()
        for time_added, path in list(self.files):
            if now - time_added > self.file_lifetime:
                if path.exists():
                    try:
                        path.unlink()
                    except Exception as exc:
                        print(f"Error deleting file {path}: {exc}")
                self.files.pop(0)
            else:
                break


file_cleaner = FileCleaner()


def make_waveform(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gr.make_waveform(*args, **kwargs)


def _select_media_path(media_type: str, image_path: str, video_path: str) -> tp.Optional[str]:
    if media_type == "Image":
        return image_path or None
    if media_type == "Video":
        return video_path or None
    return None


def _save_audio_batch(audio_batch) -> tp.List[str]:
    paths: tp.List[str] = []
    if audio_batch.numel() == 0:
        raise gr.Error("Music generation returned an empty result.")

    for idx, audio in enumerate(audio_batch):
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(
                file.name,
                audio,
                music_generator.sample_rate,
                strategy="loudness",
                loudness_headroom_db=16,
                loudness_compressor=True,
                add_suffix=False,
            )
            paths.append(file.name)
            file_cleaner.add(file.name)
    return paths


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
    progress=gr.Progress(),
):
    global INTERRUPTING
    INTERRUPTING = False
    use_diffusion = decoder == app_config.diffusion_decoder_label

    media_path = _select_media_path(media_type, image_input, video_input)

    # 1. Prepare description.
    progress(progress=None, desc="Generating music description...")
    task_type = media_type if media_path else "text"
    description = None
    
    try:
        description = mlmm_client.describe(task_type, media_path, text_prompt)
    except MLLMClientError as exc:
        if media_path:
            # Clean up CUDA cache before raising error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise gr.Error(str(exc))
        description = app_config.compose_prompt("text", text_prompt)

    # 2. Generate music.
    progress(progress=None, desc="Running MusicGen...")
    audio_batch = None
    
    try:
        audio_batch = music_generator.generate(
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
        # Clean up CUDA cache before raising error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise gr.Error(str(exc))
    except Exception as exc:
        # Catch any unexpected errors and clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise gr.Error(f"Unexpected error during music generation: {str(exc)}")

    # 3. Save results.
    progress(progress=None, desc="Saving results...")
    try:
        audio_paths = _save_audio_batch(audio_batch)
    except Exception as exc:
        # Clean up audio batch and CUDA cache
        if audio_batch is not None:
            del audio_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise gr.Error(f"Failed to save audio: {str(exc)}")
    finally:
        # Always clean up audio batch after saving
        if audio_batch is not None:
            del audio_batch

    if use_diffusion and len(audio_paths) >= 2:
        return audio_paths[0], audio_paths[1]
    return audio_paths[0], None


Wave = theme()


def create_ui(launch_kwargs=None):
    """Creates and launches the Gradio UI."""

    if launch_kwargs is None:
        launch_kwargs = {}

    def interrupt_handler():
        interrupt()

    with gr.Blocks(theme=Wave, css=css) as interface:
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

                def toggle_media(choice):
                    return {
                        image_input: gr.update(visible=(choice == "Image")),
                        video_input: gr.update(visible=(choice == "Video")),
                    }

                media_type.change(
                    toggle_media, inputs=media_type, outputs=[image_input, video_input]
                )
            with gr.Column():
                text_input = gr.Text(
                    value="Anything you like",
                    label="User Prompt",
                )
                melody_input = gr.Audio(
                    value="./assets/bach.mp3",
                    type="numpy",
                    label="Melody",
                )
                with gr.Row():
                    submit_button = gr.Button("Generate Music", variant="primary")
                    interrupt_button = gr.Button("Interrupt", variant="stop")
        with gr.Row():
            model_version = gr.Dropdown(
                app_config.available_music_models,
                label="MusicGen Model",
                value=app_config.default_music_model,
            )
            duration = gr.Slider(
                minimum=1, maximum=120, value=10, label="Duration (seconds)"
            )
        with gr.Row():
            topk = gr.Number(label="Top-k", value=250)
            topp = gr.Number(label="Top-p", value=0)
            temperature = gr.Number(label="Temperature", value=1.0)
            cfg_coef = gr.Number(label="Classifier-Free Guidance", value=3.0)
            decoder = gr.Dropdown(
                ["Default", app_config.diffusion_decoder_label],
                label="Decoder",
                value="Default",
                interactive=True,
            )

        with gr.Row():
            output_audio = gr.Audio(label="Generated Music", type="filepath")
            output_audio_mbd = gr.Audio(
                label=app_config.diffusion_decoder_label, type="filepath"
            )

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
        if INTERRUPTING:
            raise gr.Error("Interrupted.")

        gr.Examples(
            examples=[
                [
                    "Image",
                    "./assets/example_image_1.jpg",
                    None,
                    "Acoustic guitar solo. Country and folk music.",
                    None,
                    app_config.default_music_model,
                    10,
                    250,
                    0,
                    1.0,
                    3.0,
                    app_config.diffusion_decoder_label,
                ],
                [
                    "Video",
                    None,
                    "./assets/example_video_1.mp4",
                    "Space Rock, Synthwave, 80s. Electric guitar and Drums.",
                    None,
                    app_config.default_music_model,
                    10,
                    250,
                    0,
                    1.0,
                    3.0,
                    app_config.diffusion_decoder_label,
                ],
                [
                    "Text",
                    None,
                    None,
                    "An 80s driving pop song with heavy drums and synth pads in the background",
                    "./assets/bach.mp3",
                    app_config.default_music_model,
                    10,
                    250,
                    0,
                    1.0,
                    3.0,
                    app_config.diffusion_decoder_label,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--listen",
        type=str,
        default="0.0.0.0" if "SPACE_ID" in os.environ else "127.0.0.1",
        help="IP to listen on",
    )
    parser.add_argument(
        "--username", type=str, default="", help="Username for authentication"
    )
    parser.add_argument(
        "--password", type=str, default="", help="Password for authentication"
    )
    parser.add_argument(
        "--server_port", type=int, default=0, help="Port to run the server on"
    )
    parser.add_argument("--inbrowser", action="store_true", help="Open in browser")
    parser.add_argument("--share", action="store_true", help="Share the Gradio UI")

    args = parser.parse_args()

    launch_kwargs = {}
    launch_kwargs["server_name"] = args.listen
    if args.username and args.password:
        launch_kwargs["auth"] = (args.username, args.password)
    if args.server_port:
        launch_kwargs["server_port"] = args.server_port
    if args.inbrowser:
        launch_kwargs["inbrowser"] = args.inbrowser
    if args.share:
        launch_kwargs["share"] = args.share
    # When listening on 0.0.0.0 and share is not explicitly set, enable share
    elif args.listen == "0.0.0.0":
        launch_kwargs["share"] = True

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    create_ui(launch_kwargs)
