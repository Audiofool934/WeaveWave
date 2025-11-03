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
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
import base64
from tempfile import NamedTemporaryFile

# --- Configuration ---
MODEL_NAME = "google/gemma-3-12b-it"
MODEL_PATH = f"{MODEL_NAME}"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NUM_FRAMES = 32  # Adjust as needed.

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# --- Load Model ---
MLLM = None
MLLM_PROCESSOR = None

SYSTEM_PROMPT_TXT = "You are a music composer who generates short, concise description of music inspired by text input. When provided with user's text prompt, interpret text's elements and translate them into musical terms. Tailor descriptions to capture the tone, rhythm, genre, and instruments most suited to the atmosphere. Be imaginative yet relatable, offering unique musical ideas grounded in the essence of the text prompt. The description should be in one or two sentences, providing a clear and vivid image of the music that the picture evokes. The description should focus on the music, no need to mention the visual content."
SYSTEM_PROMPT_IMG = "You are a music composer who generates short, concise description of music inspired by visual input. When provided with a image and text prompt, interpret videos' elements such as colors, mood, and content—and translate them into musical terms. Tailor descriptions to capture the tone, rhythm, genre, and instruments most suited to the image's atmosphere. Be imaginative yet relatable, offering unique musical ideas grounded in the essence of the visual stimulus and the text prompt. The description should be in one or two sentences, providing a clear and vivid image of the music that the picture evokes. The description should focus on the music, no need to mention the visual content."
SYSTEM_PROMPT_VID = "You are a music composer who generates short, concise description of music inspired by visual input. When provided with a video and text prompt, interpret videos' elements such as colors, mood, and content—and translate them into musical terms. Tailor descriptions to capture the tone, rhythm, genre, and instruments most suited to the image's atmosphere. Be imaginative yet relatable, offering unique musical ideas grounded in the essence of the visual stimulus and the text prompt. The description should be in one or two sentences, providing a clear and vivid image of the music that the picture evokes. The description should focus on the music, no need to mention the visual content."


def load_mllm():
    """Loads the Gemma3 model and processor."""
    global MLLM, MLLM_PROCESSOR
    logging.info(f"Loading MLLM: {MODEL_NAME}")
    try:
        MLLM = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()
        
        MLLM_PROCESSOR = AutoProcessor.from_pretrained(MODEL_PATH)
        
        logging.info("Model and processor loaded successfully")
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
        logging.info("Starting MLLM server, loading model...")
        load_mllm()
        logging.info("MLLM model loaded successfully, server ready")
        yield
    except Exception as e:
        logging.exception("Failed to start MLLM server")
        raise
    finally:
        # 清理显存
        logging.info("Shutting down MLLM server...")
        global MLLM
        if MLLM is not None:
            MLLM.to("cpu")
            del MLLM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        logging.info("Released CUDA memory")


app = FastAPI(lifespan=lifespan)

# --- API Endpoints ---


@app.get("/health")
async def health_check():
    """Health check endpoint to verify server is ready."""
    global MLLM, MLLM_PROCESSOR
    if MLLM is None or MLLM_PROCESSOR is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/describe_image/")
async def describe_image(request: ImageRequest):
    global MLLM, MLLM_PROCESSOR
    image = None
    messages = None
    inputs = None
    generation = None
    
    try:
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Prepare messages in Gemma3 format
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT_IMG}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": request.user_prompt}
                ]
            }
        ]
        
        # Apply chat template and tokenize
        inputs = MLLM_PROCESSOR.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(MLLM.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate response
        with torch.inference_mode():
            generation = MLLM.generate(**inputs, max_new_tokens=200, do_sample=False)
            generation = generation[0][input_len:]
        
        answer = MLLM_PROCESSOR.decode(generation, skip_special_tokens=True)
        
        logging.info(f"Answer: {answer}")
        
        return {"description": answer.strip()}

    except Exception as e:
        logging.exception("Error in /describe_image/")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Always clean up resources
        if image is not None:
            del image
        if messages is not None:
            del messages
        if inputs is not None:
            del inputs
        if generation is not None:
            del generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@app.post("/describe_video/")
async def describe_video(request: VideoRequest):
    global MLLM, MLLM_PROCESSOR
    frames = None
    representative_image = None
    messages = None
    inputs = None
    generation = None
    
    try:
        video_data = base64.b64decode(request.video)
        frames = encode_video(video_data)
        
        # Use first frame as representative image (Gemma3 doesn't support multi-frame video)
        # For better results, could use middle frame or average multiple frames
        representative_image = frames[len(frames) // 2] if frames else frames[0]
        
        # Prepare messages in Gemma3 format
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT_VID}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": representative_image},
                    {"type": "text", "text": request.user_prompt}
                ]
            }
        ]
        
        # Apply chat template and tokenize
        inputs = MLLM_PROCESSOR.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(MLLM.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate response
        with torch.inference_mode():
            generation = MLLM.generate(**inputs, max_new_tokens=200, do_sample=False)
            generation = generation[0][input_len:]
        
        answer = MLLM_PROCESSOR.decode(generation, skip_special_tokens=True)
        
        logging.info(f"Answer: {answer}")
        
        return {"description": answer.strip()}
    except Exception as e:
        logging.exception("Error in /describe_video/")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Always clean up resources
        if frames is not None:
            del frames
        if representative_image is not None:
            del representative_image
        if messages is not None:
            del messages
        if inputs is not None:
            del inputs
        if generation is not None:
            del generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@app.post("/describe_text/")
async def describe_text(request: TextRequest):
    global MLLM, MLLM_PROCESSOR
    messages = None
    inputs = None
    generation = None
    
    try:
        # Prepare messages in Gemma3 format (text-only)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT_TXT}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": request.user_prompt}]
            }
        ]
        
        # Apply chat template and tokenize
        inputs = MLLM_PROCESSOR.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(MLLM.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate response
        with torch.inference_mode():
            generation = MLLM.generate(**inputs, max_new_tokens=200, do_sample=False)
            generation = generation[0][input_len:]
        
        answer = MLLM_PROCESSOR.decode(generation, skip_special_tokens=True)
        
        logging.info(f"Answer: {answer}")
        
        return {"description": answer.strip()}
    except Exception as e:
        logging.exception("Error in /describe_text/")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Always clean up resources
        if messages is not None:
            del messages
        if inputs is not None:
            del inputs
        if generation is not None:
            del generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --- Run with Uvicorn ---

if __name__ == "__main__":
    # load_mllm()
    import uvicorn

    uvicorn.run("mllm_fastapi:app", host="0.0.0.0", port=8001)
