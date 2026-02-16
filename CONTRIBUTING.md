# Contributing to VirtuSoul Router

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/VirtuSoul/virtusoul-router.git
cd virtusoul-router

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Copy and configure .env
cp .env.example .env
# Edit .env with your API keys

# Run the server
virtusoul-router --reload
```

## Running Tests

```bash
pytest
```

## Project Structure

```
virtusoul-router/
├── virtusoul_router/
│   ├── __init__.py         # Package version
│   ├── cli.py              # CLI entry point
│   ├── server.py           # FastAPI app and routes
│   ├── classifier.py       # ML classifier (MiniLM + LogReg)
│   ├── providers.py        # LLM provider adapters
│   ├── config.py           # Configuration loader
│   └── training_data.py    # Curated training examples
├── tests/                  # Test suite
├── .env.example            # Example configuration
├── pyproject.toml          # Package metadata
├── Dockerfile              # Container build
└── README.md               # Documentation
```

## Adding Training Data

The classifier improves with more examples. To add training data:

1. Edit `virtusoul_router/training_data.py`
2. Add tuples of `(query_text, tier)` to `TRAINING_DATA`
3. Run `POST /retrain` or restart the server
4. Submit a PR with your additions

Guidelines for training data:
- Each example should clearly belong to one tier
- Aim for diverse phrasing and topics
- Include both short and long queries
- Test accuracy after adding examples

## Adding a Provider

1. Add the provider's default base URL to `PROVIDER_BASE_URLS` in `providers.py`
2. If the provider uses a non-OpenAI format, add adapter functions (like the Anthropic adapter)
3. Add the provider name to `SUPPORTED_PROVIDERS` in `config.py`
4. Update the README provider table
5. Submit a PR

## Code Style

- We use `ruff` for linting
- Keep it simple — this project values simplicity over features
- Type hints are encouraged
- Docstrings for public functions

## Pull Request Process

1. Fork and create a feature branch
2. Make your changes
3. Ensure tests pass
4. Update documentation if needed
5. Submit a PR with a clear description

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
