# mllm_subprocess.py (MLLM Subprocess - Runs in mllm-env)

import argparse
import logging
import os
from pathlib import Path
import sys
import typing as tp
import warnings
import subprocess
import json
import base64


import torch
import gradio as gr
from PIL import Image
from decord import VideoReader, cpu

from transformers import AutoModel, AutoTokenizer

# --- Configuration (Subprocess) ---
MODEL_NAME = "MiniCPM-o-2_6"
MODEL_PATH = f"/home/chuangyan/WeaveWave/models/{MODEL_NAME}"  # Your model path. CHANGE THIS!
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NUM_FRAMES = 32

# --- Setup Logging (Subprocess) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Global Variables (Subprocess) ---
MLLM = None
MLLM_TOKENIZER = None

# --- Utility Functions (Subprocess) ---

def load_mllm():
    """Loads the MiniCPM-o-2.6 model and tokenizer."""
    global MLLM, MLLM_TOKENIZER
    if MLLM is None:
        logging.info(f"Loading MLLM: {MODEL_NAME}")
        try:
            MLLM = AutoModel.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True,
                attn_implementation='sdpa',  # Or 'flash_attention_2'.
                torch_dtype=torch.bfloat16,
                init_vision=True,
                init_audio=True,
                init_tts=True,
            ).eval().to(DEVICE)  # Move to device (GPU or CPU).
            MLLM_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
            MLLM.init_tts()
        except Exception as e:
            logging.exception("Failed to load MLLM")  # Log the full traceback.
            raise  # Re-raise the exception to be caught in the main part of the script.


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
        logging.info(f'Number of frames: {len(frames)}')
        return frames
    except Exception as e:
        logging.exception(f"Error encoding video: {video_path}")
        raise  # Re-raise


def generate_music_description(media_path: str, user_prompt: str) -> str:
    """Generates a music description from an image or video."""
    global MLLM, MLLM_TOKENIZER
    if MLLM is None:  # Load lazily, only when needed.
        load_mllm()

    try:
        if media_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            media_content = encode_video(media_path)
        elif media_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
              media_content = [Image.open(media_path).convert('RGB')]
        else:
            raise ValueError("Unsupported file type.")


        msgs = [{'role': 'user', 'content': media_content + [f"Generate a detailed description of music: {user_prompt}"]}]

        params = {}
        if isinstance(media_content, list) and isinstance(media_content[0], Image.Image) :
                params["use_image_id"] = False
                params["max_slice_nums"] = 2

        answer = MLLM.chat(
            msgs=msgs,
            tokenizer=MLLM_TOKENIZER,
            **params
        )
        return answer.strip()

    except FileNotFoundError:
        logging.error(f"File not found: {media_path}")
        raise  # Re-raise to be caught in the main part of the script
    except ValueError as ve:
        logging.error(str(ve))  # Log ValueErrors.
        raise
    except Exception as e:
        logging.exception(f"Error during MLLM processing")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLLM subprocess for generating music descriptions.")
    parser.add_argument("--args_json", type=str, required=True, help="JSON string containing MLLM arguments.")
    args = parser.parse_args()

    try:
        gen_args = json.loads(args.args_json)
        media_path = gen_args['media_path']
        user_prompt = gen_args['user_prompt']
        device = gen_args['device']
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # Set the CUDA_VISIBLE_DEVICES environment variable.

        description = generate_music_description(media_path, user_prompt)
        result = {'description': description}  # Return the description.
        print(json.dumps(result))
        sys.stdout.flush()

    except Exception as e:
        logging.exception("Error in MLLM subprocess") # Use logging.exception to get full traceback
        # Return an error message in JSON format.  Consistent with the main app.
        print(json.dumps({'error': str(e)}))
        sys.stdout.flush()
        sys.exit(1)  # Exit with a non-zero code to indicate failure.