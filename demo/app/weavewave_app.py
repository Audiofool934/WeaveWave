import argparse
import logging
import os
import sys
import time
import typing as tp
import warnings
import base64
from pathlib import Path
from tempfile import NamedTemporaryFile

from einops import rearrange
import torch
import gradio as gr
import requests

from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from audiocraft.models.encodec import InterleaveStereoCompressionModel
from audiocraft.models import MusicGen, MultiBandDiffusion

from theme_wave import theme, css

# --- Configuration (Main App) ---
MLLM_API_URL = (
    "http://localhost:8000"  #  REPLACE with the actual URL of your MLLM API server.
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Global Variables (Main App) ---
MODEL = None
MBD = None
INTERRUPTING = False
USE_DIFFUSION = False  # Keep this for now, even if unused, for easier switching


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


def load_musicgen_model(version="facebook/musicgen-stereo-melody-large"):
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
        MBD = MultiBandDiffusion.get_mbd_musicgen(device=DEVICE)


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
    global INTERRUPTING, USE_DIFFUSION
    INTERRUPTING = False
    USE_DIFFUSION = decoder == "MultiBand_Diffusion"

    if media_type == "Image":
        media = image_input if image_input else None
    elif media_type == "Video":
        media = video_input if video_input else None
    else:
        media = None

    # 1. Get Music Description (using the API client).
    progress(progress=None, desc="Generating music description...")
    if media:
        try:
            music_description = get_mllm_description(media, text_prompt)
        except Exception as e:
            raise gr.Error(str(e))  # Re-raise for Gradio to handle.
    else:
        music_description = text_prompt

    # 2. Load MusicGen Model (locally).
    progress(progress=None, desc="Loading MusicGen model...")
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
    progress(progress=None, desc="Processing melody...")
    melody_tensor = None  # Use a different variable name
    if melody:
        try:
            sr, melody_tensor = (
                melody[0],
                torch.from_numpy(melody[1]).to(MODEL.device).float().t(),
            )
            if melody_tensor.dim() == 1:
                melody_tensor = melody_tensor[None]
            melody_tensor = melody_tensor[..., : int(sr * duration)]
            melody_tensor = convert_audio(
                melody_tensor, sr, MODEL.sample_rate, MODEL.audio_channels
            )

        except Exception as e:
            raise gr.Error(f"Error processing melody: {e}")

    # 5. Music Generation (locally).
    progress(progress=None, desc="Generating music...")
    if USE_DIFFUSION:
        load_diffusion_model()

    try:
        if melody_tensor is not None:  # Use the new variable
            output = MODEL.generate_with_chroma(
                descriptions=[music_description],
                melody_wavs=[melody_tensor],
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
        progress(progress=None, desc="Running MultiBandDiffusion...")
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
    progress(progress=None, desc="Saving and returning...")
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
            output_audio_paths.append(file.name)
            file_cleaner.add(file.name)

    if USE_DIFFUSION:
        # Return both audios, but make sure to return the correct one first
        result = (
            output_audio_paths[0],  # Original
            output_audio_paths[1],  # MBD
        )
    else:
        result = (
            output_audio_paths[0],
            None,
        )  # Only original audio and description

    del melody_tensor, output, output_audio
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


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
                        choices=["Image", "Video"],
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
                    interrupt_button = gr.Button(
                        "Interrupt", variant="stop"
                    )  # Keep as gr.Button
        with gr.Row():
            model_version = gr.Dropdown(
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
                value="facebook/musicgen-stereo-melody-large",
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
                ["Default", "MultiBand_Diffusion"],
                label="Decoder",
                value="Default",
                interactive=True,
            )

        # with gr.Row():
        #     description_output = gr.Textbox(label="MLLM Generated Description")
        with gr.Row():
            output_audio = gr.Audio(label="Generated Music", type="filepath")
            output_audio_mbd = gr.Audio(
                label="MultiBand Diffusion Decoder", type="filepath"
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
            # outputs=[output_audio, description_output, output_audio_mbd],
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
                    "facebook/musicgen-stereo-melody-large",
                    10,
                    250,
                    0,
                    1.0,
                    3.0,
                    "MultiBand_Diffusion",
                ],
                [
                    "Video",
                    None,
                    "./assets/example_video_1.mp4",
                    "Space Rock, Synthwave, 80s. Electric guitar and Drums.",
                    None,
                    "facebook/musicgen-stereo-melody-large",
                    10,
                    250,
                    0,
                    1.0,
                    3.0,
                    "MultiBand_Diffusion",
                ],
                [
                    None,
                    None,
                    None,
                    "An 80s driving pop song with heavy drums and synth pads in the background",
                    "./assets/bach.mp3",
                    "facebook/musicgen-stereo-melody-large",
                    10,
                    250,
                    0,
                    1.0,
                    3.0,
                    "MultiBand_Diffusion",
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
    create_ui(launch_kwargs)
