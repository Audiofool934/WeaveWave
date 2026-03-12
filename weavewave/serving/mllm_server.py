"""FastAPI server exposing the multimodal LLM for music description generation.

The three endpoints (``/describe_text/``, ``/describe_image/``, ``/describe_video/``)
share most of their logic.  An :class:`MLLMService` class encapsulates the model and
provides a single ``describe`` method parameterised by modality.
"""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from weavewave.core.types import MLLMServerConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts — they differ only by the input modality description.
# ---------------------------------------------------------------------------
_PROMPT_TEMPLATE = (
    "You are a music composer who generates short, concise description of music "
    "inspired by {modality} input. When provided with {context}, interpret "
    "{elements} and translate them into musical terms. Tailor descriptions to "
    "capture the tone, rhythm, genre, and instruments most suited to the "
    "{atmosphere}'s atmosphere. Be imaginative yet relatable, offering unique "
    "musical ideas grounded in the essence of {grounding}. The description should "
    "be in one or two sentences, providing a clear and vivid image of the music "
    "that the {source} evokes. The description should focus on the music, no need "
    "to mention the visual content."
)

SYSTEM_PROMPTS = {
    "text": _PROMPT_TEMPLATE.format(
        modality="text",
        context="user's text prompt",
        elements="text's elements",
        atmosphere="the text",
        grounding="the text prompt",
        source="picture",
    ),
    "image": _PROMPT_TEMPLATE.format(
        modality="visual",
        context="a image and text prompt",
        elements="videos' elements such as colors, mood, and content—",
        atmosphere="the image",
        grounding="the visual stimulus and the text prompt",
        source="picture",
    ),
    "video": _PROMPT_TEMPLATE.format(
        modality="visual",
        context="a video and text prompt",
        elements="videos' elements such as colors, mood, and content—",
        atmosphere="the image",
        grounding="the visual stimulus and the text prompt",
        source="picture",
    ),
}


# ---------------------------------------------------------------------------
# CUDA cleanup helper
# ---------------------------------------------------------------------------
@contextmanager
def cuda_cleanup():
    """Context manager that empties the CUDA cache on exit."""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------------------------
def encode_video(video_bytes: bytes, max_frames: int = 32) -> list[Image.Image]:
    """Decode video bytes into a list of uniformly-sampled PIL frames.

    Args:
        video_bytes: Raw video file content.
        max_frames: Maximum number of frames to extract.

    Returns:
        List of PIL Images.
    """
    from decord import VideoReader, cpu  # lazy import — heavy dependency

    def _uniform_sample(seq: list, n: int) -> list:
        gap = len(seq) / n
        return [seq[int(i * gap + gap / 2)] for i in range(n)]

    with NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        vr = VideoReader(tmp_path, ctx=cpu(0))
        sample_fps = round(vr.get_avg_fps() / 1)
        frame_idx = list(range(0, len(vr), sample_fps))
        if len(frame_idx) > max_frames:
            frame_idx = _uniform_sample(frame_idx, max_frames)
        frames = vr.get_batch(frame_idx).asnumpy()
        return [Image.fromarray(f.astype("uint8")) for f in frames]
    except Exception as exc:
        logger.exception("Error encoding video")
        raise HTTPException(status_code=500, detail=f"Error encoding video: {exc}") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# MLLM service
# ---------------------------------------------------------------------------
class MLLMService:
    """Wraps model loading, inference, and resource cleanup.

    Args:
        config: Server configuration dataclass.
    """

    def __init__(self, config: MLLMServerConfig) -> None:
        self.config = config
        self.model: Gemma3ForConditionalGeneration | None = None
        self.processor: AutoProcessor | None = None

    def load(self) -> None:
        """Load the model and processor onto the configured device."""
        logger.info("Loading MLLM: %s", self.config.model_name)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.config.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        logger.info("Model loaded successfully")

    def unload(self) -> None:
        """Move model to CPU and release CUDA memory."""
        if self.model is not None:
            self.model.to("cpu")
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Released CUDA memory")

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.processor is not None

    def describe(
        self,
        modality: str,
        user_prompt: str,
        *,
        image: Image.Image | None = None,
    ) -> str:
        """Generate a music description for the given modality.

        Args:
            modality: One of ``'text'``, ``'image'``, ``'video'``.
            user_prompt: User-provided text prompt.
            image: PIL Image for image/video modalities.

        Returns:
            Generated description string.
        """
        assert self.model is not None and self.processor is not None, "Model not loaded"

        system_prompt = SYSTEM_PROMPTS[modality]

        user_content: list[dict] = []
        if image is not None:
            user_content.append({"type": "image", "image": image})
        user_content.append({"type": "text", "text": user_prompt})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]

        with cuda_cleanup():
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                )
                generation = generation[0][input_len:]

            answer: str = self.processor.decode(generation, skip_special_tokens=True)

        logger.info("Generated description: %s", answer.strip())
        return answer.strip()


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------
class ImageRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image data")
    user_prompt: str = Field("", description="Optional user prompt")


class VideoRequest(BaseModel):
    video: str = Field(..., description="Base64-encoded video data")
    user_prompt: str = Field("", description="Optional user prompt")


class TextRequest(BaseModel):
    user_prompt: str = Field(..., description="User prompt")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
def create_app(config: MLLMServerConfig | None = None) -> FastAPI:
    """Build and return the FastAPI application.

    Args:
        config: Server configuration.  Uses defaults when *None*.
    """
    if config is None:
        config = MLLMServerConfig()

    service = MLLMService(config)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        logger.info("Starting MLLM server")
        service.load()
        yield
        logger.info("Shutting down MLLM server")
        service.unload()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health_check():
        if not service.is_ready:
            raise HTTPException(status_code=503, detail="Model not loaded yet")
        return {"status": "ok", "model": config.model_name}

    @app.post("/describe_image/")
    async def describe_image(request: ImageRequest):
        import base64

        try:
            image_data = base64.b64decode(request.image)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            return {"description": service.describe("image", request.user_prompt, image=image)}
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Error in /describe_image/")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/describe_video/")
    async def describe_video(request: VideoRequest):
        import base64

        try:
            video_data = base64.b64decode(request.video)
            frames = encode_video(video_data, max_frames=config.max_num_frames)
            representative = frames[len(frames) // 2] if frames else frames[0]
            return {
                "description": service.describe("video", request.user_prompt, image=representative)
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Error in /describe_video/")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/describe_text/")
    async def describe_text(request: TextRequest):
        try:
            return {"description": service.describe("text", request.user_prompt)}
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Error in /describe_text/")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for ``weavewave-mllm-server``."""
    import uvicorn

    config = MLLMServerConfig()
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
