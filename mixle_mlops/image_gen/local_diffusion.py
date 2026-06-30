"""Local Stable Diffusion adapter — runs a diffusers pipeline directly on the host GPU/MPS/CPU.

No external server needed: the pipeline is loaded once at startup and stays in memory.
Registered in the model registry as ``"sd-local"`` (or whatever name is passed) with ``kind="image"``.
Integrates with the existing BlobStore so generated images are served via ``/v1/files/{id}/content``.

Usage (env-based)::

    MIXLE_DIFFUSION_MODEL=CompVis/stable-diffusion-v1-4  \\
    MIXLE_DIFFUSION_STEPS=20  \\
    uvicorn mixle_mlops.gateway.app:create_app --factory
"""
from __future__ import annotations

import asyncio
import base64
import io
from functools import lru_cache
from typing import Any

from ..core.adapters import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatRequest,
    ModelAdapter,
    ModelInfo,
)
from ..multimodal.store import BlobStore, get_blob_store


def _pick_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


@lru_cache(maxsize=1)
def _load_pipeline(model_id: str, device: str):
    """Load and cache the pipeline — called once, re-used for every request."""
    import torch
    from diffusers import StableDiffusionPipeline

    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    if device == "mps":
        # Recommended for MPS: disable attention slicing (it's already efficient on Apple silicon).
        pipe.safety_checker = None          # remove NSFW checker (slows down, needs CPU fallback)
    elif device == "cuda":
        pipe.enable_attention_slicing()
    return pipe


class LocalDiffusionAdapter(ModelAdapter):
    """Runs a Stable Diffusion pipeline locally via diffusers."""

    kind = "image"

    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        steps: int = 20,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        store: BlobStore | None = None,
    ):
        self._name = name
        self.model_id = model_id
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.width = width
        self.height = height
        self._store = store
        self._device = _pick_device()

    @property
    def name(self) -> str:
        return self._name

    def capabilities(self) -> set[str]:
        return {"image_generation", "local"}

    def info(self) -> ModelInfo:
        return ModelInfo(id=self.name, kind="composite", capabilities=sorted(self.capabilities()))

    @property
    def store(self) -> BlobStore:
        return self._store if self._store is not None else get_blob_store()

    # Refuse chat requests with a helpful redirect.
    async def stream(self, req: ChatRequest):  # type: ignore[override]
        raise NotImplementedError(f"{self._name!r} is an image model; use /v1/images/generations")
        yield  # makes this an async generator

    async def chat(self, req: ChatRequest) -> ChatCompletion:
        from ..core.adapters import ChatChoice, ChatMessage
        note = (f"[{self._name}] is a local image-generation model. "
                f"POST /v1/images/generations with {{model, prompt, n, size}}.")
        return ChatCompletion(
            model=req.model or self._name,
            choices=[ChatChoice(message=ChatMessage(role="assistant", content=note), finish_reason="stop")],
        )

    async def generate(
        self,
        prompt: str,
        *,
        n: int = 1,
        size: str | None = None,
        negative_prompt: str | None = None,
        **opts: Any,
    ) -> list[dict[str, Any]]:
        """Generate ``n`` images, store them in the blob store, return ``[{id, url, b64_json}]``."""
        if not prompt or not str(prompt).strip():
            raise ValueError("prompt must be a non-empty string")
        n = max(1, min(int(n), 4))

        w, h = self.width, self.height
        if size:
            try:
                parts = size.lower().replace("x", "×").split("×")
                if len(parts) == 2:
                    w, h = int(parts[0]), int(parts[1])
            except (ValueError, IndexError):
                pass

        steps = int(opts.get("num_inference_steps", self.steps))
        guidance = float(opts.get("guidance_scale", self.guidance_scale))

        # Run inference in a thread so we don't block the event loop.
        loop = asyncio.get_event_loop()
        images = await loop.run_in_executor(
            None,
            lambda: self._run_pipeline(prompt, n=n, width=w, height=h,
                                       steps=steps, guidance=guidance,
                                       negative_prompt=negative_prompt),
        )

        out: list[dict[str, Any]] = []
        store = self.store
        for pil_img in images:
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            data = buf.getvalue()
            b64 = base64.b64encode(data).decode("ascii")
            record = store.put(data, filename=f"{self._name}.png", content_type="image/png")
            out.append({"id": record.id, "url": record.url, "b64_json": b64})
        return out

    def _run_pipeline(
        self,
        prompt: str,
        *,
        n: int,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        negative_prompt: str | None,
    ):
        pipe = _load_pipeline(self.model_id, self._device)
        kwargs: dict[str, Any] = dict(
            prompt=[prompt] * n,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
        )
        if negative_prompt:
            kwargs["negative_prompt"] = [negative_prompt] * n
        result = pipe(**kwargs)
        return result.images


def load_local_diffusion(
    name: str,
    model_id: str,
    *,
    steps: int = 20,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
) -> LocalDiffusionAdapter:
    return LocalDiffusionAdapter(name, model_id, steps=steps, guidance_scale=guidance_scale,
                                 width=width, height=height)
