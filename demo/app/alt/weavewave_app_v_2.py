# musicgen_app.py (Main Gradio App - Runs in musicgen-env)

import argparse
import logging
import os
from pathlib import Path
import sys
import time
import typing as tp
import warnings
import base64
from tempfile import NamedTemporaryFile

from einops import rearrange
import torch
import gradio as gr
import requests  #  For making API requests.

from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from audiocraft.models.encodec import InterleaveStereoCompressionModel
from audiocraft.models import MusicGen, MultiBandDiffusion

# --- Configuration (Main App) ---
MLLM_API_URL = (
    "http://localhost:8000"  #  REPLACE with the actual URL of your MLLM API server.
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Global Variables (Main App) ---
MODEL = None
MBD = None
INTERRUPTING = False
USE_DIFFUSION = False


# --- Utility Functions (Main App) ---
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
                    except Exception as e:
                        print(f"Error deleting file {path}: {e}")
                self.files.pop(0)
            else:
                break


file_cleaner = FileCleaner()


def make_waveform(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gr.make_waveform(*args, **kwargs)


# --- Model Loading (Main App) ---


def load_musicgen_model(version="facebook/musicgen-melody"):
    global MODEL
    print(f"Loading MusicGen model: {version}")
    if MODEL is None or MODEL.name != version:
        if MODEL is not None:
            del MODEL
        torch.cuda.empty_cache()
        MODEL = MusicGen.get_pretrained(version, device=DEVICE)


def load_diffusion_model():
    global MBD
    if MBD is None:
        print("Loading diffusion model")
        MBD = MultiBandDiffusion.get_mbd_musicgen().to(DEVICE)


# --- API Client Functions ---


def get_mllm_description(media_path: str, user_prompt: str) -> str:
    """Gets the music description from the MLLM API."""

    try:
        if media_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            # Video
            with open(media_path, "rb") as f:
                video_data = f.read()
            encoded_video = base64.b64encode(video_data).decode("utf-8")
            response = requests.post(
                f"{MLLM_API_URL}/describe_video/",
                json={"video": encoded_video, "user_prompt": user_prompt},
            )
        elif media_path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
            # Image
            with open(media_path, "rb") as f:
                image_data = f.read()
            encoded_image = base64.b64encode(image_data).decode("utf-8")
            response = requests.post(
                f"{MLLM_API_URL}/describe_image/",
                json={"image": encoded_image, "user_prompt": user_prompt},
            )
        else:  # Text-only
            response = requests.post(
                f"{MLLM_API_URL}/describe_text/", json={"user_prompt": user_prompt}
            )

        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx).
        return response.json()["description"]

    except requests.exceptions.RequestException as e:
        raise gr.Error(f"Error communicating with MLLM API: {e}")
    except Exception as e:
        raise gr.Error(f"An unexpected error occurred: {e}")


# --- Music Generation ---


def predict_full(
    model_version,
    media_path,
    text_prompt,
    melody_path,
    duration,
    topk,
    topp,
    temperature,
    cfg_coef,
    progress=gr.Progress(),
):
    global INTERRUPTING, USE_DIFFUSION
    INTERRUPTING = False

    # 1. Get Music Description (using the API client).
    progress(0, desc="Generating music description...")
    if media_path:
        try:
            music_description = get_mllm_description(media_path, text_prompt)
        except Exception as e:
            raise gr.Error(str(e))  # Re-raise for Gradio to handle.
    else:
        music_description = text_prompt

    # 2. Load MusicGen Model (locally).
    progress(0.2, desc="Loading MusicGen model...")
    load_musicgen_model(model_version)

    # 3. Set Generation Parameters (locally).
    MODEL.set_generation_params(
        duration=duration,
        top_k=topk,
        top_p=topp,
        temperature=temperature,
        cfg_coef=cfg_coef,
    )

    # 4. Melody Preprocessing (locally).
    progress(0.4, desc="Processing melody...")
    melody = None
    if melody_path:
        try:
            sr, melody = gr.load(melody_path)
            melody = torch.from_numpy(melody).to(MODEL.device).float()
            if melody.dim() == 1:
                melody = melody.unsqueeze(0)
            if melody.dim() == 2 and melody.shape[0] == 2:
                melody = melody.mean(dim=0, keepdim=True)
            melody = melody[..., : int(sr * duration)]
            melody = convert_audio(melody, sr, MODEL.sample_rate, MODEL.audio_channels)

        except Exception as e:
            raise gr.Error(f"Error processing melody: {e}")

    # 5. Music Generation (locally).
    progress(0.6, desc="Generating music...")
    if USE_DIFFUSION:
        load_diffusion_model()
    try:
        if melody is not None:
            output = MODEL.generate_with_chroma(
                descriptions=[music_description],
                melody_wavs=[melody],
                melody_sample_rate=MODEL.sample_rate,
                progress=True,
                return_tokens=USE_DIFFUSION,
            )
        else:
            output = MODEL.generate(
                descriptions=[music_description],
                progress=True,
                return_tokens=USE_DIFFUSION,
            )
    except RuntimeError as e:
        raise gr.Error("Error while generating: " + str(e))

    if USE_DIFFUSION:
        progress(0.8, desc="Running MultiBandDiffusion...")
        tokens = output[1]
        if isinstance(MODEL.compression_model, InterleaveStereoCompressionModel):
            left, right = MODEL.compression_model.get_left_right_codes(tokens)
            tokens = torch.cat([left, right])
        outputs_diffusion = MBD.tokens_to_wav(tokens)
        if isinstance(MODEL.compression_model, InterleaveStereoCompressionModel):
            assert outputs_diffusion.shape[1] == 1  # output is mono
            outputs_diffusion = rearrange(
                outputs_diffusion, "(s b) c t -> b (s c) t", s=2
            )
        output_audio = torch.cat([output[0], outputs_diffusion], dim=0)
    else:
        output_audio = output[0]

    output_audio = output_audio.detach().cpu().float()

    # 6. Save and Return (locally).
    progress(0.9, desc="Saving and returning...")
    output_video_paths = []
    output_audio_paths = []
    for i, audio in enumerate(output_audio):
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(
                file.name,
                audio,
                MODEL.sample_rate,
                strategy="loudness",
                loudness_headroom_db=16,
                loudness_compressor=True,
                add_suffix=False,
            )
            if USE_DIFFUSION:
                if i == 0:
                    video_path = make_waveform(file.name)
                else:
                    video_path = make_waveform(file.name)
            else:
                video_path = make_waveform(file.name)
            output_video_paths.append(video_path)
            output_audio_paths.append(file.name)
            file_cleaner.add(file.name)
            file_cleaner.add(video_path)

    if USE_DIFFUSION:
        return (
            output_video_paths[0],
            output_audio_paths[0],
            output_video_paths[1],
            output_audio_paths[1],
            music_description,
        )
    return output_video_paths[0], output_audio_paths[0], None, None, music_description


def toggle_audio_src(choice):
    if choice == "mic":
        return gr.update(source="microphone", value=None, label="Microphone")
    else:
        return gr.update(source="upload", value=None, label="File", visible=True)


def toggle_diffusion(choice):
    if choice == "MultiBand_Diffusion":
        return [gr.update(visible=True)] * 2
    else:
        return [gr.update(visible=False)] * 2


# --- Gradio Interface (Main App) ---
from typing import Iterable, Union
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from gradio.themes.utils.colors import Color
import time

ocean1 = Color(
    name="ocean1",
    c50="#6699cc",
    c100="#6699cc",
    c200="#6699cc",
    c300="#6699cc",
    c400="#6699cc",
    c500="#6699cc",
    c600="#6699cc",
    c700="#6699cc",
    c800="#6699cc",
    c900="#6699cc",
    c950="#6699cc",
)
ocean2 = Color(
    name="ocean2",
    c50="#4d79a1",
    c100="#4d79a1",
    c200="#4d79a1",
    c300="#4d79a1",
    c400="#4d79a1",
    c500="#4d79a1",
    c600="#4d79a1",
    c700="#4d79a1",
    c800="#4d79a1",
    c900="#4d79a1",
    c950="#4d79a1",
)
ocean3 = Color(
    name="ocean3",
    c50="#1e334d",
    c100="#1e334d",
    c200="#1e334d",
    c300="#1e334d",
    c400="#1e334d",
    c500="#1e334d",
    c600="#1e334d",
    c700="#1e334d",
    c800="#1e334d",
    c900="#1e334d",
    c950="#1e334d",
)


class WeaveWave(Base):
    def __init__(
        self,
        *,
        primary_hue: Union[colors.Color, str] = ocean1,
        secondary_hue: Union[colors.Color, str] = ocean3,
        neutral_hue: Union[colors.Color, str] = ocean2,
        spacing_size: Union[sizes.Size, str] = sizes.spacing_md,
        radius_size: Union[sizes.Size, str] = sizes.radius_md,
        text_size: Union[sizes.Size, str] = sizes.text_lg,
        font: Union[fonts.Font, str, Iterable[Union[fonts.Font, str]]] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: Union[fonts.Font, str, Iterable[Union[fonts.Font, str]]] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            # Background:  Light blue and white cross-gradient for light mode, light blue and dark blue for dark mode.
            # body_background_fill="repeating-linear-gradient(-45deg, *primary_50, *primary_50 10px, *neutral_50 10px, *neutral_50 20px)",  # Light mode
            body_background_fill="linear-gradient(*primary_500, *secondary_400)",  # Light mode
            body_background_fill_dark="linear-gradient(*primary_800, *secondary_800)",  # Dark mode
            # Primary Buttons: Gradient with a "wave" feel
            button_primary_background_fill="linear-gradient(135deg, *primary_500, *secondary_400)",  # Teal to Cobalt
            button_primary_background_fill_dark="linear-gradient(135deg, *primary_800, *secondary_800)",
            button_primary_background_fill_hover="linear-gradient(135deg, *primary_200, *secondary_300)",  # Lighter on hover
            button_primary_text_color="white",
            button_primary_text_color_dark="white",
            # Sliders: Use the secondary color for a visual pop
            slider_color="*neutral_50",
            slider_color_dark="*neutral_50",
            # Block Styling:
            block_title_text_weight="600",
            block_border_width="1px",  # Thinner border for a cleaner look
            block_border_color="*neutral_200",
            block_border_color_dark="*neutral_50",
            block_shadow="*shadow_drop",  # Less aggressive shadow
            # button_primary_shadow="*shadow_drop",
            # Text:  Use neutral_hue for text on primary backgrounds
            block_title_text_color="#000000",
            block_title_text_color_dark="#000000",
            block_label_text_color="#000000",
            block_label_text_color_dark="#000000",
            input_background_fill = "white",
            input_background_fill_dark = "white",
            body_text_color="#000000",
            body_text_color_dark="#000000",
            input_border_color="#000000",
            input_border_color_dark="#000000",
            input_shadow="none",
            # Other button
            button_secondary_background_fill="*neutral_50",
            button_secondary_background_fill_dark="*neutral_50",
            button_secondary_background_fill_hover="*neutral_50",
            button_secondary_text_color="*neutral_50",
            button_secondary_text_color_dark="*neutral_50",
            # Panel
            panel_background_fill="*neutral_50",
            panel_background_fill_dark="*neutral_50",
        )


weavewave = WeaveWave()


def ui_full(launch_kwargs):
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            <div style="text-align: center;">
                <h1>WeaveWave</h1>
                <p>Multimodal Music Generation</p>
            </div>
            """
        )
        with gr.Row():
            text_input = gr.Text(label="User Prompt (Optional)", interactive=True)
        with gr.Row():
            with gr.Column():
                media_input = gr.File(
                    value="./assets/WeaveWave.png",
                    # label="Input Image(s)/Video", 
                    # file_types=["image", "video"],
                    # visible=True,
                    # interactive=True,
                )
            with gr.Column():
                melody_input = gr.Audio(
                    # sources=["upload", "microphone"],
                    # type="filepath",
                    # label="Melody (Optional)",
                    # interactive=True,
                )
        with gr.Row():
            submit_button = gr.Button("Generate Music", variant="primary")
            _ = gr.Button("Interrupt", variant="primary").click(
                fn=interrupt, queue=False
            )
        with gr.Row():
            description_output = gr.Textbox(
                label="MLLM Generated Description"
            )  # Display description
            # output_video = gr.Video(label="Generated Music (Video)")
            output_audio = gr.Audio(label="Generated Music (WAV)", type="filepath")
            # diffusion_output = gr.Video(label="MultiBand Diffusion Decoder (Video)")
            # audio_diffusion = gr.Audio(
            #     label="MultiBand Diffusion Decoder (wav)", type="filepath"
            # )
        with gr.Row():
            model_version = gr.Radio(
                [
                    "facebook/musicgen-melody",
                    "facebook/musicgen-medium",
                    "facebook/musicgen-small",
                    "facebook/musicgen-large",
                    "facebook/musicgen-melody-large",
                    "facebook/musicgen-stereo-small",
                    "facebook/musicgen-stereo-medium",
                    "facebook/musicgen-stereo-melody",
                    "facebook/musicgen-stereo-large",
                    "facebook/musicgen-stereo-melody-large",
                ],
                label="MusicGen Model",
                value="facebook/musicgen-stereo-melody",
                interactive=True,
            )
        with gr.Row():
            # with gr.Column():
                decoder = gr.Radio(
                    ["Default", "MultiBand_Diffusion"],
                    label="Decoder",
                    value="Default",
                    interactive=True,
                )

            # with gr.Column():
                duration = gr.Slider(
                    minimum=1,
                    maximum=120,
                    value=10,
                    label="Duration (seconds)",
                    interactive=True,
                )
            # with gr.Column():
                topk = gr.Number(label="Top-k", value=250, interactive=True)
                topp = gr.Number(label="Top-p", value=0, interactive=True)
                temperature = gr.Number(
                    label="Temperature", value=1.0, interactive=True
                )
                cfg_coef = gr.Number(
                    label="Classifier-Free Guidance", value=3.0, interactive=True
                )

        submit_button.click(
            toggle_diffusion,
            decoder,
            # [diffusion_output, audio_diffusion],
            queue=False,
            show_progress=False,
        ).then(
            predict_full,
            inputs=[
                model_version,
                media_input,
                text_input,
                melody_input,
                duration,
                topk,
                topp,
                temperature,
                cfg_coef,
            ],
            outputs=[
                # output_video,
                output_audio,
                # diffusion_output,
                # audio_diffusion,
                description_output,
            ],  # Include description output.
        )

        gr.Examples(
            examples=[
                [
                    "./assets/example_image_1.jpg",
                    "Acoustic guitar solo",
                    None,
                    "facebook/musicgen-stereo-medium",
                    "Default",
                    10,
                ],
                [
                    "./assets/example_video_1.mp4",
                    "Energetic dance track",
                    None,
                    "facebook/musicgen-stereo-medium",
                    "Default",
                    10,
                ],
            ],
            inputs=[
                media_input,
                text_input,
                melody_input,
                model_version,
                decoder,
                duration,
            ],
            # outputs=[output_video],
            fn=predict_full,
        )

        interface.queue().launch(**launch_kwargs)


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
    )  # Add server_port argument.
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

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    ui_full(launch_kwargs)
