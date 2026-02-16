"""Tests for configuration loading."""

import os
import pytest
from virtusoul_router.config import load_config, TierConfig


class TestConfig:
    def test_default_values(self, monkeypatch):
        # Clear any existing tier env vars
        for tier in ["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"]:
            for suffix in ["PROVIDER", "MODEL", "API_KEY", "BASE_URL"]:
                monkeypatch.delenv(f"{tier}_{suffix}", raising=False)

        config = load_config()
        assert config.host == "0.0.0.0"
        assert config.port == 4000
        assert config.model_name == "virtusoul-v1"

    def test_tier_loading(self, monkeypatch):
        monkeypatch.setenv("SIMPLE_PROVIDER", "openai")
        monkeypatch.setenv("SIMPLE_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("SIMPLE_API_KEY", "sk-test")

        config = load_config()
        assert "simple" in config.tiers
        assert config.tiers["simple"].provider == "openai"
        assert config.tiers["simple"].model == "gpt-4o-mini"

    def test_tier_config_is_configured(self):
        tier = TierConfig(name="test", provider="openai", model="gpt-4o")
        assert tier.is_configured()

        empty = TierConfig(name="test", provider="", model="")
        assert not empty.is_configured()
