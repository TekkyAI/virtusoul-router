"""CLI entry point for VirtuSoul Router."""

import argparse
import logging
import sys

import uvicorn

from . import __version__
from .config import load_config


def main():
    parser = argparse.ArgumentParser(
        prog="virtusoul-router",
        description="Intelligent LLM router with ML-based query classification",
    )
    parser.add_argument("--version", action="version", version=f"virtusoul-router {__version__}")
    parser.add_argument("--host", type=str, default=None, help="Host to bind (default: from .env or 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Port to bind (default: from .env or 4000)")
    parser.add_argument("--env", type=str, default=None, help="Path to .env file")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    # Load config to get defaults and validate
    config = load_config(args.env)
    host = args.host or config.host
    port = args.port or config.port

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"""
╔══════════════════════════════════════════════╗
║          VirtuSoul Router v{__version__}           ║
║   Intelligent LLM Routing — Open Source      ║
╚══════════════════════════════════════════════╝

  Model name:  {config.model_name}
  Endpoint:    http://{host}:{port}/v1/chat/completions
  Tiers:""")

    for name in ["simple", "medium", "complex", "reasoning"]:
        tier = config.tiers.get(name)
        if tier:
            print(f"    {name:12s} → {tier.provider}/{tier.model}")
        else:
            print(f"    {name:12s} → (not configured)")

    if config.api_key:
        print(f"  Auth:        Bearer token required")
    else:
        print(f"  Auth:        None (open access)")

    print(f"""
  Use with any OpenAI SDK:
    base_url = "http://{host}:{port}/v1"
    model    = "{config.model_name}"

  Starting server...
""")

    uvicorn.run(
        "virtusoul_router.server:create_app",
        host=host,
        port=port,
        reload=args.reload,
        factory=True,
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    main()
