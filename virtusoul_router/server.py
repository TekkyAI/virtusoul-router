"""
VirtuSoul Router — FastAPI server with OpenAI-compatible endpoints.

This is the main entry point. It exposes:
  POST /v1/chat/completions  — OpenAI-compatible chat completions with smart routing
  POST /classify             — Classify a query (for debugging/testing)
  POST /retrain              — Retrain the classifier with custom examples
  GET  /health               — Health check
"""

import logging
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .classifier import get_classifier
from .config import RouterConfig, load_config, TIER_NAMES
from .providers import chat_completion

logger = logging.getLogger("virtusoul.server")

# Global config — set during startup
_config: Optional[RouterConfig] = None


def get_config() -> RouterConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


# ── Request / Response models ──

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[list[str] | str] = None
    stream: Optional[bool] = False


class ClassifyRequest(BaseModel):
    messages: list[Message] = Field(default=None)
    text: Optional[str] = None


# ── App ──

def create_app(config: Optional[RouterConfig] = None) -> FastAPI:
    global _config
    if config:
        _config = config
    else:
        _config = load_config()

    app = FastAPI(
        title="VirtuSoul Router",
        version=__version__,
        description="Intelligent LLM router with ML-based query classification",
    )

    # ── Auth middleware ──

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        cfg = get_config()
        if cfg.api_key:
            # Skip auth for health check
            if request.url.path in ("/health", "/"):
                return await call_next(request)
            auth = request.headers.get("authorization", "")
            if not auth.startswith("Bearer ") or auth[7:] != cfg.api_key:
                return JSONResponse({"error": "Invalid or missing API key"}, status_code=401)
        return await call_next(request)

    # ── Routes ──

    @app.get("/")
    async def root():
        return {
            "name": "VirtuSoul Router",
            "version": __version__,
            "status": "running",
            "model": _config.model_name,
            "tiers": {name: f"{t.provider}/{t.model}" for name, t in _config.tiers.items()},
        }

    @app.get("/health")
    async def health():
        classifier = get_classifier()
        return {
            "status": "healthy",
            "version": __version__,
            "classifier_ready": classifier._is_trained,
            "configured_tiers": list(_config.tiers.keys()),
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        cfg = get_config()
        messages_dicts = [m.model_dump() for m in req.messages]

        # Determine which tier to use
        requested_model = req.model
        tier_name = None
        classification_info = None

        if requested_model in TIER_NAMES:
            # Direct tier selection
            tier_name = requested_model
        elif requested_model == cfg.model_name:
            # Smart routing — classify the query
            classifier = get_classifier()
            user_text = " ".join(m.content for m in req.messages if m.role == "user")
            start = time.time()
            result = classifier.classify(user_text)
            elapsed_ms = round((time.time() - start) * 1000, 1)
            tier_name = result.tier
            classification_info = {
                "tier": result.tier,
                "confidence": result.confidence,
                "latency_ms": elapsed_ms,
            }
            logger.info(f"Classified as '{tier_name}' ({result.confidence:.2f}) in {elapsed_ms}ms")
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model '{requested_model}'. Use '{cfg.model_name}' for smart routing or one of: {', '.join(TIER_NAMES)}",
            )

        # Get tier config
        tier_config = cfg.tiers.get(tier_name)
        if not tier_config:
            # Fallback
            tier_config = cfg.tiers.get(cfg.fallback_tier)
            if not tier_config:
                raise HTTPException(status_code=503, detail=f"Tier '{tier_name}' is not configured")
            logger.warning(f"Tier '{tier_name}' not configured, falling back to '{cfg.fallback_tier}'")
            tier_name = cfg.fallback_tier

        # Build kwargs
        kwargs = {}
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        if req.max_tokens is not None:
            kwargs["max_tokens"] = req.max_tokens
        if req.top_p is not None:
            kwargs["top_p"] = req.top_p
        if req.frequency_penalty is not None:
            kwargs["frequency_penalty"] = req.frequency_penalty
        if req.presence_penalty is not None:
            kwargs["presence_penalty"] = req.presence_penalty
        if req.stop is not None:
            kwargs["stop"] = req.stop
        if req.stream:
            kwargs["stream"] = True

        try:
            result = await chat_completion(tier_config, messages_dicts, timeout=cfg.timeout, **kwargs)
        except Exception as e:
            logger.error(f"Provider error ({tier_config.provider}/{tier_config.model}): {e}")
            raise HTTPException(status_code=502, detail=f"Upstream provider error: {str(e)}")

        # Streaming response
        if req.stream:
            return StreamingResponse(result, media_type="text/event-stream")

        # Add routing metadata to non-streaming response
        if classification_info:
            result["_virtusoul"] = {
                "routed_to": f"{tier_config.provider}/{tier_config.model}",
                **classification_info,
            }

        return result

    @app.post("/classify")
    async def classify(req: ClassifyRequest):
        """Classify a query without routing — useful for testing."""
        text = req.text
        if not text and req.messages:
            text = " ".join(m.content for m in req.messages if m.role == "user")
        if not text:
            raise HTTPException(status_code=400, detail="Provide 'text' or 'messages'")

        classifier = get_classifier()
        start = time.time()
        result = classifier.classify(text)
        elapsed_ms = round((time.time() - start) * 1000, 1)

        return {
            "tier": result.tier,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "latency_ms": elapsed_ms,
            "flagged": result.confidence < 0.60,
        }

    @app.post("/retrain")
    async def retrain():
        """Retrain the classifier (uses built-in training data)."""
        classifier = get_classifier()
        result = classifier.train()
        return {"success": True, **result}

    return app
