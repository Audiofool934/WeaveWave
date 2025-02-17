import argparse
import logging
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
import time
import typing as tp
import warnings

from einops import rearrange
import torch
import gradio as gr
from PIL import Image
from decord import VideoReader, cpu

from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from audiocraft.models.encodec import InterleaveStereoCompressionModel
from audiocraft.models import MusicGen, MultiBandDiffusion

from transformers import AutoModel, AutoTokenizer

# --- Configuration ---
MODEL_NAME = "MiniCPM-o-2.6"
MODEL_PATH = f"/home/chuangyan/WeaveWave/models/{MODEL_NAME}"  # Your model path
# Set the CUDA device. IMPORTANT: Adjust if necessary.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Or "0", "0,1", etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NUM_FRAMES = 32  # Reduced for memory efficiency. Adjust as needed.


# --- Global Variables ---
MODEL = None  # MusicGen model
MBD = None    # MultiBandDiffusion model
MLLM = None  # MiniCPM-o-2.6 model
MLLM_TOKENIZER = None # MiniCPM-o-2.6 tokenizer
INTERRUPTING = False
USE_DIFFUSION = False

# --- Utility Functions ---

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
        warnings.simplefilter('ignore')
        return gr.make_waveform(*args, **kwargs)

# --- Model Loading ---

def load_musicgen_model(version='facebook/musicgen-melody'):
    global MODEL
    print("Loading MusicGen model:", version)
    if MODEL is None or MODEL.name != version:
        if MODEL is not None:
            del MODEL
        torch.cuda.empty_cache()
        MODEL = MusicGen.get_pretrained(version, device=DEVICE)

def load_diffusion_model():
    global MBD
    if MBD is None:
        print("Loading MultiBand Diffusion model")
        MBD = MultiBandDiffusion.get_mbd_musicgen().to(DEVICE)

def load_mllm():
    """Loads the MiniCPM-o-2.6 model and tokenizer."""
    global MLLM, MLLM_TOKENIZER
    if MLLM is None:
        print("Loading MLLM:", MODEL_NAME)
        try:
            MLLM = AutoModel.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True,
                attn_implementation='sdpa',
                torch_dtype=torch.bfloat16,
                init_vision=True,
                init_audio=True,
                init_tts=True,
            ).eval().to(DEVICE)
            MLLM_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
            MLLM.init_tts()  # Initialize TTS components
        except Exception as e:
            raise gr.Error(f"Failed to load MLLM: {e}")


# --- MLLM Interaction ---
def encode_video(video_path):
    """Encodes a video into a sequence of frames (PIL Images)."""
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        sample_fps = round(vr.get_avg_fps() / 1)
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        if len(frame_idx) > MAX_NUM_FRAMES:
            frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype('uint8')) for v in frames]
        print('num frames:', len(frames))
        return frames
    except Exception as e:
        raise gr.Error(f"Error encoding video: {e}")


def generate_music_description(media_path: str, user_prompt: str) -> str:
    """Generates a music description from an image, video, or multiple images."""
    global MLLM, MLLM_TOKENIZER
    if MLLM is None:
        load_mllm()

    try:
        # Determine media type and process accordingly.
        if media_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Video
            media_content = encode_video(media_path)
        elif media_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  # Image
              media_content = [Image.open(media_path).convert('RGB')]
        else:
            raise ValueError("Unsupported file type.  Must be image or video.")


        msgs = [{'role': 'user', 'content': media_content + [f"Generate a detailed description of music suitable for a text-to-music model, based on this media and the following user prompt (if any): {user_prompt} Include genre, instruments, tempo, and mood. Keep it concise (around 100 words or less)."]}]

        # Set decode params (especially for video).
        params = {}
        if isinstance(media_content, list) and isinstance(media_content[0], Image.Image) :  # Check if it's a list of PIL Images
                params["use_image_id"] = False  # Important for handling multiple images/frames
                params["max_slice_nums"] = 2  # Adjust as needed based on your GPU memory


        answer = MLLM.chat(
            msgs=msgs,
            tokenizer=MLLM_TOKENIZER,
            **params
        )
        return answer.strip()

    except FileNotFoundError:
        raise gr.Error(f"File not found: {media_path}")
    except ValueError as ve:
        raise gr.Error(str(ve))  # Re-raise the ValueError
    except Exception as e:
        raise gr.Error(f"Error during MLLM processing: {e}")



# --- Music Generation ---

def predict_full(model_version, media_path, text_prompt, melody, duration, topk, topp, temperature, cfg_coef, progress=gr.Progress()):
    global INTERRUPTING, USE_DIFFUSION
    INTERRUPTING = False

    # 1.  Get Music Description from MLLM
    progress(0, desc="Generating music description...")
    if media_path is None:
      music_description = text_prompt
    else:
      music_description = generate_music_description(media_path, text_prompt)


    # 2. Load MusicGen Model (if not already loaded)
    progress(0.2, desc="Loading MusicGen model...")
    load_musicgen_model(model_version)

    # 3.  Set Generation Parameters
    MODEL.set_generation_params(
        duration=duration,
        top_k=topk,
        top_p=topp,
        temperature=temperature,
        cfg_coef=cfg_coef
    )

    # 4.  Melody Conditioning (Preprocessing)
    progress(0.4, desc="Processing melody...")
    if melody is not None:
        sr, melody = melody[0], torch.from_numpy(melody[1]).to(MODEL.device).float().t()
        if melody.dim() == 1:
            melody = melody[None]
        melody = melody[..., :int(sr * duration)]
        melody = convert_audio(melody, sr, MODEL.sample_rate, MODEL.audio_channels)
    else:
        melody = None

    # 5.  Music Generation
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
                return_tokens = USE_DIFFUSION
            )
        else:
            output = MODEL.generate(descriptions=[music_description], progress=True, return_tokens = USE_DIFFUSION)
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
            outputs_diffusion = rearrange(outputs_diffusion, '(s b) c t -> b (s c) t', s=2)
        output_audio = torch.cat([output[0], outputs_diffusion], dim=0) #both audios if diffusion
    else:
        output_audio = output[0]  # only the em model output
    output_audio = output_audio.detach().cpu().float()

    # 6. Save and Return
    progress(0.9, desc="Saving and returning...")
    output_video_paths = []
    output_audio_paths = []
    for i, audio in enumerate(output_audio):
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(
                file.name, audio, MODEL.sample_rate, strategy="loudness",
                loudness_headroom_db=16, loudness_compressor=True, add_suffix=False
            )
            if USE_DIFFUSION:
                if i == 0: # the non diffusion model
                    output_video_paths.append(make_waveform(file.name))
                    output_audio_paths.append(file.name)
                else:
                    output_video_paths.append(make_waveform(file.name))
                    output_audio_paths.append(file.name)
            else:
                output_video_paths.append(make_waveform(file.name))
                output_audio_paths.append(file.name)
            file_cleaner.add(file.name)
            file_cleaner.add(output_video_paths[-1])  # Add video to cleaner as well
    if USE_DIFFUSION:
        return output_video_paths[0], output_audio_paths[0], output_video_paths[1], output_audio_paths[1]
    return output_video_paths[0], output_audio_paths[0], None, None

def toggle_audio_src(choice):
    if choice == "mic":
        return gr.update(source="microphone", value=None, label="Microphone")
    else:
        return gr.update(source="upload", value=None, label="File")


def toggle_diffusion(choice):
    if choice == "MultiBand_Diffusion":
        return [gr.update(visible=True)] * 2
    else:
        return [gr.update(visible=False)] * 2
# --- Gradio Interface ---

def ui_full(launch_kwargs):
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            # Multimodal Music Generation
            This demo combines MiniCPM-o-2.6 with MusicGen to generate music from images/videos and text prompts.
            """
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    media_input = gr.File(label="Input Image(s)/Video", file_types=["image", "video"]) # Changed to File Input
                with gr.Row():
                    text_input = gr.Text(label="User Prompt (Optional)", interactive=True)
                with gr.Row():
                  melody_input = gr.Audio(sources=["upload"], type="numpy", label="Melody (Optional)", interactive=True)
                with gr.Row():
                    submit_button = gr.Button("Generate Music")
                    _ = gr.Button("Interrupt").click(fn=interrupt, queue=False)  # Interrupt button

                with gr.Row():
                    model_version = gr.Radio([
                        "facebook/musicgen-melody", "facebook/musicgen-medium", "facebook/musicgen-small",
                        "facebook/musicgen-large", "facebook/musicgen-melody-large",
                        "facebook/musicgen-stereo-small", "facebook/musicgen-stereo-medium",
                        "facebook/musicgen-stereo-melody", "facebook/musicgen-stereo-large",
                        "facebook/musicgen-stereo-melody-large"
                    ], label="MusicGen Model", value="facebook/musicgen-stereo-melody", interactive=True)
                with gr.Row():
                    decoder = gr.Radio(["Default", "MultiBand_Diffusion"],
                                       label="Decoder", value="Default", interactive=True)
                with gr.Row():
                    duration = gr.Slider(minimum=1, maximum=120, value=10, label="Duration (seconds)", interactive=True)
                with gr.Row():
                    topk = gr.Number(label="Top-k", value=250, interactive=True)
                    topp = gr.Number(label="Top-p", value=0, interactive=True)
                    temperature = gr.Number(label="Temperature", value=1.0, interactive=True)
                    cfg_coef = gr.Number(label="Classifier-Free Guidance", value=3.0, interactive=True)

            with gr.Column():
                output_video = gr.Video(label="Generated Music (Video)")
                output_audio = gr.Audio(label="Generated Music (WAV)", type='filepath')
                diffusion_output = gr.Video(label="MultiBand Diffusion Decoder")
                audio_diffusion = gr.Audio(label="MultiBand Diffusion Decoder (wav)", type='filepath')

        submit_button.click(
            toggle_diffusion, decoder, [diffusion_output, audio_diffusion], queue=False, show_progress=False
            ).then(
            predict_full,
            inputs=[model_version, media_input, text_input, melody_input, duration, topk, topp, temperature, cfg_coef],  # Changed to media_input
            outputs=[output_video, output_audio, diffusion_output, audio_diffusion]
        )

        gr.Examples(
            examples=[
                ["./assets/example_image_1.jpg", "Acoustic guitar solo", None,"facebook/musicgen-stereo-melody", "Default", 10],
                ["./assets/example_video_1.mp4", "Energetic pop track", None, "facebook/musicgen-stereo-melody", "Default", 10], # Added a video example
            ],
            inputs=[media_input, text_input, melody_input, model_version, decoder, duration], # Changed to media_input
            outputs=[output_video],
            fn=predict_full,
        )
        interface.queue().launch(**launch_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--listen', type=str, default='0.0.0.0' if 'SPACE_ID' in os.environ else '127.0.0.1', help='IP to listen on')
    parser.add_argument('--username', type=str, default='', help='Username for authentication')
    parser.add_argument('--password', type=str, default='', help='Password for authentication')
    parser.add_argument('--server_port', type=int, default=0, help='Port to run the server on')
    parser.add_argument('--inbrowser', action='store_true', help='Open in browser')
    parser.add_argument('--share', action='store_true', help='Share the Gradio UI')
    args = parser.parse_args()

    launch_kwargs = {}
    launch_kwargs['server_name'] = args.listen
    if args.username and args.password:
        launch_kwargs['auth'] = (args.username, args.password)
    if args.server_port:
        launch_kwargs['server_port'] = args.server_port
    if args.inbrowser:
        launch_kwargs['inbrowser'] = args.inbrowser
    if args.share:
        launch_kwargs['share'] = args.share

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    ui_full(launch_kwargs)