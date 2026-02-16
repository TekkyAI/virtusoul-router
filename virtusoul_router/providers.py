"""
LLM Provider adapters — thin HTTP layer that replaces LiteLLM.

Most providers are OpenAI-compatible. Only Anthropic and Google need adapters.
"""

import json
import logging
from typing import AsyncIterator, Optional

import httpx

from .config import TierConfig

logger = logging.getLogger("virtusoul.providers")

# Default base URLs per provider
PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "ollama": "http://localhost:11434/v1",
    "mistral": "https://api.mistral.ai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai",
}

ANTHROPIC_API_VERSION = "2023-06-01"


def _get_base_url(tier: TierConfig) -> str:
    """Get the base URL for a provider."""
    if tier.base_url:
        return tier.base_url.rstrip("/")
    return PROVIDER_BASE_URLS.get(tier.provider, tier.base_url or "")


def _build_headers(tier: TierConfig) -> dict:
    """Build auth headers based on provider."""
    if tier.provider == "anthropic":
        return {
            "x-api-key": tier.api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }
    if tier.provider == "google":
        return {"content-type": "application/json"}

    # OpenAI-compatible providers
    headers = {"content-type": "application/json"}
    if tier.api_key:
        headers["authorization"] = f"Bearer {tier.api_key}"
    return headers


def _to_anthropic_request(messages: list, model: str, **kwargs) -> dict:
    """Convert OpenAI-format messages to Anthropic format."""
    system_parts = []
    chat_messages = []

    for msg in messages:
        if msg.get("role") == "system":
            system_parts.append(msg.get("content", ""))
        else:
            chat_messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

    body = {"model": model, "messages": chat_messages, "max_tokens": kwargs.get("max_tokens", 4096)}
    if system_parts:
        body["system"] = "\n".join(system_parts)
    if "temperature" in kwargs and kwargs["temperature"] is not None:
        body["temperature"] = kwargs["temperature"]
    if kwargs.get("stream"):
        body["stream"] = True
    return body


def _from_anthropic_response(data: dict, model: str) -> dict:
    """Convert Anthropic response to OpenAI format."""
    content = ""
    if data.get("content"):
        content = "".join(block.get("text", "") for block in data["content"] if block.get("type") == "text")

    usage = data.get("usage", {})
    return {
        "id": data.get("id", ""),
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": _map_anthropic_stop(data.get("stop_reason", "end_turn")),
        }],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
    }


def _map_anthropic_stop(reason: str) -> str:
    return {"end_turn": "stop", "max_tokens": "length", "stop_sequence": "stop"}.get(reason, "stop")


async def chat_completion(tier: TierConfig, messages: list, timeout: float = 120.0, **kwargs) -> dict:
    """Send a chat completion request to the configured provider. Returns OpenAI-format response."""
    base_url = _get_base_url(tier)
    headers = _build_headers(tier)
    stream = kwargs.get("stream", False)

    if tier.provider == "anthropic":
        url = f"{base_url}/v1/messages"
        body = _to_anthropic_request(messages, tier.model, **kwargs)
    else:
        # OpenAI-compatible
        url = f"{base_url}/chat/completions"
        body = {"model": tier.model, "messages": messages}
        for key in ("temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty", "stop"):
            if key in kwargs and kwargs[key] is not None:
                body[key] = kwargs[key]
        if stream:
            body["stream"] = True

    logger.info(f"→ {tier.provider}/{tier.model} ({url})")

    if stream:
        return await _stream_request(tier, url, headers, body, timeout)

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()

    if tier.provider == "anthropic":
        return _from_anthropic_response(data, tier.model)
    return data


async def _stream_request(tier: TierConfig, url: str, headers: dict, body: dict, timeout: float) -> AsyncIterator:
    """Handle streaming responses. Returns an async iterator of SSE chunks."""

    async def stream_generator():
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, headers=headers, json=body) as resp:
                resp.raise_for_status()
                if tier.provider == "anthropic":
                    async for chunk in _adapt_anthropic_stream(resp, tier.model):
                        yield chunk
                else:
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            yield line + "\n\n"
                        elif line.strip() == "":
                            continue
                    yield "data: [DONE]\n\n"

    return stream_generator()


async def _adapt_anthropic_stream(resp, model: str) -> AsyncIterator[str]:
    """Convert Anthropic SSE stream to OpenAI SSE format."""
    async for line in resp.aiter_lines():
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload.strip() == "[DONE]":
            yield "data: [DONE]\n\n"
            return
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type", "")
        if event_type == "content_block_delta":
            delta = event.get("delta", {})
            text = delta.get("text", "")
            if text:
                chunk = {
                    "id": "",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
        elif event_type == "message_stop":
            chunk = {
                "id": "",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
