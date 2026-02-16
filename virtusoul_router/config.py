"""Configuration loader — reads .env and validates tier setup."""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger("virtusoul.config")

SUPPORTED_PROVIDERS = {
    "openai", "anthropic", "openrouter", "groq", "together",
    "ollama", "mistral", "deepseek", "google", "custom",
}

TIER_NAMES = ["simple", "medium", "complex", "reasoning"]


@dataclass
class TierConfig:
    """Configuration for a single routing tier."""
    name: str
    provider: str
    model: str
    api_key: str = ""
    base_url: Optional[str] = None

    def is_configured(self) -> bool:
        return bool(self.provider and self.model)


@dataclass
class RouterConfig:
    """Full router configuration."""
    host: str = "0.0.0.0"
    port: int = 4000
    model_name: str = "virtusoul-v1"
    api_key: Optional[str] = None
    log_level: str = "INFO"
    timeout: float = 120.0
    tiers: dict = field(default_factory=dict)

    @property
    def fallback_tier(self) -> str:
        return "medium"


def load_config(env_path: Optional[str] = None) -> RouterConfig:
    """Load configuration from .env file and environment variables."""
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()

    config = RouterConfig(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "4000")),
        model_name=os.getenv("MODEL_NAME", "virtusoul-v1"),
        api_key=os.getenv("API_KEY"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        timeout=float(os.getenv("TIMEOUT", "120")),
    )

    # Load tier configurations
    for tier_name in TIER_NAMES:
        prefix = tier_name.upper()
        provider = os.getenv(f"{prefix}_PROVIDER", "")
        model = os.getenv(f"{prefix}_MODEL", "")
        api_key = os.getenv(f"{prefix}_API_KEY", "")
        base_url = os.getenv(f"{prefix}_BASE_URL")

        if provider and model:
            if provider not in SUPPORTED_PROVIDERS:
                logger.warning(f"Unknown provider '{provider}' for {tier_name} tier. Treating as 'custom'.")
                provider = "custom"

            config.tiers[tier_name] = TierConfig(
                name=tier_name,
                provider=provider,
                model=model,
                api_key=api_key,
                base_url=base_url,
            )
            logger.info(f"  ✓ {tier_name}: {provider}/{model}")
        else:
            logger.info(f"  ○ {tier_name}: not configured (will fallback to {config.fallback_tier})")

    if not config.tiers:
        logger.error("No tiers configured! Set at least MEDIUM_PROVIDER and MEDIUM_MODEL in .env")

    return config
