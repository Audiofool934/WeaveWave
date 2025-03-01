import io
import logging
import os
from typing import Union, List
from pathlib import Path
from contextlib import asynccontextmanager

import torch
from PIL import Image
from decord import VideoReader, cpu
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer
import base64
from tempfile import NamedTemporaryFile

# --- Configuration ---
MODEL_NAME = "MiniCPM-o-2_6"
MODEL_PATH = f"/home/chuangyan/WeaveWave/models/{MODEL_NAME}"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NUM_FRAMES = 32  # Adjust as needed.

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# --- Load Model ---
MLLM = None
MLLM_TOKENIZER = None

SYSTEM_PROMPT_TXT = "You are a music composer who generates short, concise description of music inspired by text input. When provided with user's text prompt, interpret text's elements and translate them into musical terms. Tailor descriptions to capture the tone, rhythm, genre, and instruments most suited to the atmosphere. Be imaginative yet relatable, offering unique musical ideas grounded in the essence of the text prompt. The description should be in one or two sentences, providing a clear and vivid image of the music that the picture evokes. The description should focus on the music, no need to mention the visual content."
SYSTEM_PROMPT_IMG = "You are a music composer who generates short, concise description of music inspired by visual input. When provided with a image and text prompt, interpret videos' elements such as colors, mood, and content—and translate them into musical terms. Tailor descriptions to capture the tone, rhythm, genre, and instruments most suited to the image's atmosphere. Be imaginative yet relatable, offering unique musical ideas grounded in the essence of the visual stimulus and the text prompt. The description should be in one or two sentences, providing a clear and vivid image of the music that the picture evokes. The description should focus on the music, no need to mention the visual content."
SYSTEM_PROMPT_VID = "You are a music composer who generates short, concise description of music inspired by visual input. When provided with a video and text prompt, interpret videos' elements such as colors, mood, and content—and translate them into musical terms. Tailor descriptions to capture the tone, rhythm, genre, and instruments most suited to the image's atmosphere. Be imaginative yet relatable, offering unique musical ideas grounded in the essence of the visual stimulus and the text prompt. The description should be in one or two sentences, providing a clear and vivid image of the music that the picture evokes. The description should focus on the music, no need to mention the visual content."


def load_mllm():
    """Loads the MiniCPM model and tokenizer."""
    global MLLM, MLLM_TOKENIZER
    logging.info(f"Loading MLLM: {MODEL_NAME}")
    try:
        MLLM = (
            AutoModel.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True,
                attn_implementation="sdpa",  # Or 'flash_attention_2'.
                torch_dtype=torch.bfloat16,
                init_vision=True,
                init_audio=True,
                init_tts=True,
            )
            .eval()
            .to(DEVICE)
        )
        MLLM_TOKENIZER = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True
        )
        MLLM.init_tts()  # Init tts model.
    except Exception as e:
        logging.exception("Failed to load MLLM")
        raise  # Re-raise to have it handled at a higher level.


# --- Helper Functions ---


def encode_video(video_bytes: bytes) -> List[Image.Image]:
    """Encodes a video (provided as bytes) into a list of PIL Images."""

    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    try:
        # Save the video bytes to a temporary file.
        with NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video_file:
            temp_video_file.write(video_bytes)
            temp_video_path = temp_video_file.name

        vr = VideoReader(temp_video_path, ctx=cpu(0))
        sample_fps = round(vr.get_avg_fps() / 1)
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        if len(frame_idx) > MAX_NUM_FRAMES:
            frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype("uint8")) for v in frames]

        os.remove(temp_video_path)
        return frames
    except Exception as e:
        logging.exception("Error encoding video")
        raise HTTPException(status_code=500, detail=f"Error encoding video: {e}")


# --- Pydantic Models ---
class ImageRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image data")
    user_prompt: str = Field("", description="Optional user prompt")


class VideoRequest(BaseModel):
    video: str = Field(..., description="Base64 encoded video data")
    user_prompt: str = Field("", description="Optional user prompt")


class TextRequest(BaseModel):
    user_prompt: str = Field(..., description="User prompt")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_mllm()
        yield
    finally:
        # 清理显存
        global MLLM
        if MLLM is not None:
            MLLM.to("cpu")
            del MLLM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        logging.info("Released CUDA memory")


app = FastAPI(lifespan=lifespan)

# --- API Endpoints ---


@app.post("/describe_image/")
async def describe_image(request: ImageRequest):
    global MLLM, MLLM_TOKENIZER
    try:
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        print(type(image))
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT_IMG},
            {"role": "user", "content": [image, request.user_prompt]},
        ]
        # params = {"use_image_id": False, "max_slice_nums": 2}
        print(type(MLLM))
        answer = MLLM.chat(
            msgs=msgs,
            tokenizer=MLLM_TOKENIZER,
        )
        del image, msgs
        
        logging.info(f"Answer: {answer}")
        
        return {"description": answer.strip()}

    except Exception as e:
        logging.exception("Error in /describe_image/")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/describe_video/")
async def describe_video(request: VideoRequest):
    global MLLM, MLLM_TOKENIZER
    try:
        video_data = base64.b64decode(request.video)
        frames = encode_video(video_data)

        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT_VID},
            {"role": "user", "content": frames + [request.user_prompt]},
        ]
        params = {"use_image_id": False, "max_slice_nums": 2}

        answer = MLLM.chat(msgs=msgs, tokenizer=MLLM_TOKENIZER, **params)
        
        logging.info(f"Answer: {answer}")
        
        del msgs
        torch.cuda.empty_cache()
        return {"description": answer.strip()}
    except Exception as e:
        logging.exception("Error in /describe_video/")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/describe_text/")
async def describe_text(request: TextRequest):
    global MLLM, MLLM_TOKENIZER
    try:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT_TXT},
            {"role": "user", "content": request.user_prompt},
        ]
        answer = MLLM.chat(msgs=msgs, tokenizer=MLLM_TOKENIZER)
        del msgs
        torch.cuda.empty_cache()
        
        logging.info(f"Answer: {answer}")
        
        return {"description": answer.strip()}
    except Exception as e:
        logging.exception("Error in /describe_text/")
        raise HTTPException(status_code=500, detail=str(e))


# --- Run with Uvicorn ---

if __name__ == "__main__":
    # load_mllm()
    import uvicorn

    uvicorn.run("mllm_api_server:app", host="0.0.0.0", port=8000)
