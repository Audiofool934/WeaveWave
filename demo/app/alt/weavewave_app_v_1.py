# musicgen_app.py (Main Gradio App - Runs in musicgen-env)

import argparse
import logging
import os
from pathlib import Path
import sys
import time
import typing as tp
import warnings
import subprocess
import json
from tempfile import NamedTemporaryFile

from einops import rearrange
import torch
import gradio as gr

from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from audiocraft.models.encodec import InterleaveStereoCompressionModel
from audiocraft.models import MusicGen, MultiBandDiffusion


# --- Configuration (Main App) ---

# Path to the MLLM subprocess script.  IMPORTANT!
MLLM_SUBPROCESS_SCRIPT = "mllm_subprocess.py"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEVICE_for_MLLM = '1'

# --- Global Variables (Main App) ---

MODEL = None  # MusicGen model
MBD = None  # MultiBand Diffusion
INTERRUPTING = False
USE_DIFFUSION = False


# --- Utility Functions (Main App)---
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
    """Wraps gr.make_waveform and catches warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gr.make_waveform(*args, **kwargs)


# --- Model Loading (Main App) ---
def load_musicgen_model(version="facebook/musicgen-melody"):
    """Loads the MusicGen model (or switches to a different one)."""
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


# --- Subprocess Communication ---


def run_mllm_subprocess(media_path: str, user_prompt: str) -> str:
    """
    Runs the MLLM in a separate process to generate a music description.
    """
    try:
        # Find the Python executable within the mllm-env.
        if "CONDA_PREFIX" in os.environ:
            # Windows
            mllm_env_python = os.path.join(
                os.environ["CONDA_PREFIX"], "..", "Transformers", "python.exe"
            )  # Corrected path
            if not os.path.exists(mllm_env_python):
                # Linux/macOS
                mllm_env_python = os.path.join(
                    os.environ["CONDA_PREFIX"], "..", "Transformers", "bin", "python"
                )  # Corrected path
                if not os.path.exists(mllm_env_python):
                    raise FileNotFoundError(
                        "Could not find Python interpreter in mllm-env."
                    )
        else:
            raise EnvironmentError("CONDA_PREFIX environment variable not found.")

        # Prepare the arguments for the subprocess.
        args = {"media_path": media_path, "user_prompt": user_prompt, "device": DEVICE_for_MLLM}

        cmd = [
            mllm_env_python,  # Use the correct Python interpreter from mllm-env!
            MLLM_SUBPROCESS_SCRIPT,
            "--args_json",
            json.dumps(args),
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            logging.error(f"MLLM subprocess failed:\n{stderr.decode()}")
            raise gr.Error(f"MLLM subprocess failed: {stderr.decode()}")

        result = json.loads(stdout.decode())
        if "error" in result:
            raise gr.Error(f"MLLM subprocess error: {result['error']}")
        return result["description"]  # Return the generated description.

    except Exception as e:
        logging.exception("Error in run_mllm_subprocess")
        raise gr.Error(f"Error running MLLM subprocess: {e}")


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

    # 1. Get Music Description (using the subprocess).
    progress(0, desc="Generating music description...")
    if media_path:
        try:
            music_description = run_mllm_subprocess(media_path, text_prompt)
        except Exception as e:
            raise gr.Error(str(e))  # Re-raise for Gradio
    else:
        music_description = text_prompt  # Use text prompt directly if no media.

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
            sr, melody = gr.load(melody_path, sr=None)
            melody = torch.from_numpy(melody).to(MODEL.device).float()
            if melody.dim() == 1:
                melody = melody.unsqueeze(0)
            if melody.dim() == 2 and melody.shape[0] == 2:  # Stereo to mono
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
            file_cleaner.add(file.name)  # Add audio to cleaner.
            file_cleaner.add(video_path)  # Add video to cleaner

    if USE_DIFFUSION:
        return (
            output_video_paths[0],
            output_audio_paths[0],
            output_video_paths[1],
            output_audio_paths[1],
        )
    return output_video_paths[0], output_audio_paths[0], None, None


def toggle_audio_src(choice):
    """Toggles the source of the melody input (file upload or microphone)."""
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


def ui_full(launch_kwargs):
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            # Multimodal Music Generation
            This demo combines MiniCPM-o-2.6 with MusicGen to generate music.
            """
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    media_input = gr.File(
                        label="Input Image(s)/Video", file_types=["image", "video"]
                    )
                with gr.Row():
                    text_input = gr.Text(
                        label="User Prompt (Optional)", interactive=True
                    )
                with gr.Row():
                    melody_input = gr.Audio(
                        sources=["upload"],
                        type="filepath",
                        label="Melody (Optional)",
                        interactive=True,
                    )
                with gr.Row():
                    submit_button = gr.Button("Generate Music")
                    _ = gr.Button("Interrupt").click(fn=interrupt, queue=False)

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
                    decoder = gr.Radio(
                        ["Default", "MultiBand_Diffusion"],
                        label="Decoder",
                        value="Default",
                        interactive=True,
                    )

                with gr.Row():
                    duration = gr.Slider(
                        minimum=1,
                        maximum=120,
                        value=10,
                        label="Duration (seconds)",
                        interactive=True,
                    )
                with gr.Row():
                    topk = gr.Number(label="Top-k", value=250, interactive=True)
                    topp = gr.Number(label="Top-p", value=0, interactive=True)
                    temperature = gr.Number(
                        label="Temperature", value=1.0, interactive=True
                    )
                    cfg_coef = gr.Number(
                        label="Classifier-Free Guidance", value=3.0, interactive=True
                    )

            with gr.Column():
                output_video = gr.Video(label="Generated Music (Video)")
                output_audio = gr.Audio(label="Generated Music (WAV)", type="filepath")
                diffusion_output = gr.Video(
                    label="MultiBand Diffusion Decoder (Video)"
                )  # Output for MBD video
                audio_diffusion = gr.Audio(
                    label="MultiBand Diffusion Decoder (wav)", type="filepath"
                )  # Output for MBD audio

        submit_button.click(
            toggle_diffusion,
            decoder,
            [diffusion_output, audio_diffusion],
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
            outputs=[output_video, output_audio, diffusion_output, audio_diffusion],
        )

        gr.Examples(
            examples=[
                [
                    "./assets/example_image_1.jpg",
                    "Acoustic guitar solo",
                    None,
                    "facebook/musicgen-stereo-melody",
                    "Default",
                    10,
                ],
                [
                    "./assets/example_video_1.mp4",
                    "Energetic dance track",
                    None,
                    "facebook/musicgen-stereo-melody",
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
            outputs=[output_video],
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

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    ui_full(launch_kwargs)
