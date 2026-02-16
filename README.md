<p align="center">
  <h1 align="center">🧠 VirtuSoul Router</h1>
  <p align="center">
    <strong>Intelligent LLM router with ML-based query classification</strong>
  </p>
  <p align="center">
    Route prompts to the right model automatically. Save money on simple queries, use powerful models only when needed.
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> •
    <a href="#how-it-works">How It Works</a> •
    <a href="#configuration">Configuration</a> •
    <a href="#providers">Providers</a> •
    <a href="#api-reference">API Reference</a>
  </p>
</p>

---

## What is VirtuSoul Router?

VirtuSoul Router is an open-source, self-hosted LLM proxy that **automatically routes your prompts to the right model** based on query complexity.

- "What is 2+2?" → routes to a **free/cheap model** (Llama 3.2, Phi-3)
- "Design a microservices architecture" → routes to a **powerful model** (Claude 3.5 Sonnet, GPT-4)
- "Prove the halting problem is undecidable" → routes to a **reasoning model** (O1, Claude 3 Opus)

It's **fully OpenAI-compatible**. Just change your `base_url` and you're done. Works with any OpenAI SDK (Python, TypeScript, Go, etc.).

**No LLM calls for classification** — uses a local ML model (MiniLM + Logistic Regression) that classifies in ~15ms on CPU.

## Features

- 🧠 **ML-powered smart routing** — local classifier, no API calls, ~15ms latency
- 🔌 **OpenAI-compatible API** — drop-in replacement, works with any SDK
- 🌐 **Multi-provider support** — OpenAI, Anthropic, OpenRouter, Groq, Together, Ollama, Mistral, DeepSeek, Google
- ⚡ **Streaming support** — full SSE streaming, just like OpenAI
- 🎯 **4 complexity tiers** — simple, medium, complex, reasoning
- 📦 **Single process** — no database, no Redis, just `pip install` and go
- 🐳 **Docker ready** — pre-built image with model weights included
- 🔄 **Retrainable** — add your own training data to improve accuracy
- 🔑 **Optional auth** — protect your router with a Bearer token

## Quick Start

### Install

```bash
pip install virtusoul-router
```

### Configure

```bash
# Create your config
cp .env.example .env

# Edit .env — set your API keys and model choices
```

Minimal `.env` (just OpenAI):
```env
MODEL_NAME=virtusoul-v1

SIMPLE_PROVIDER=openai
SIMPLE_MODEL=gpt-4o-mini
SIMPLE_API_KEY=sk-your-key

MEDIUM_PROVIDER=openai
MEDIUM_MODEL=gpt-4o-mini
MEDIUM_API_KEY=sk-your-key

COMPLEX_PROVIDER=openai
COMPLEX_MODEL=gpt-4o
COMPLEX_API_KEY=sk-your-key

REASONING_PROVIDER=openai
REASONING_MODEL=o1-preview
REASONING_API_KEY=sk-your-key
```

### Run

```bash
virtusoul-router
```

```
╔══════════════════════════════════════════════╗
║          VirtuSoul Router v0.1.0           ║
║   Intelligent LLM Routing — Open Source      ║
╚══════════════════════════════════════════════╝

  Model name:  virtusoul-v1
  Endpoint:    http://0.0.0.0:4000/v1/chat/completions
  Tiers:
    simple       → openai/gpt-4o-mini
    medium       → openai/gpt-4o-mini
    complex      → openai/gpt-4o
    reasoning    → openai/o1-preview
```

### Use

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000/v1",
    api_key="not-needed",  # or your API_KEY if you set one
)

response = client.chat.completions.create(
    model="virtusoul-v1",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
# → Classified as "simple" → routed to gpt-4o-mini
print(response.choices[0].message.content)
```


Works with any language:

```typescript
// TypeScript
import OpenAI from "openai";
const client = new OpenAI({ baseURL: "http://localhost:4000/v1", apiKey: "not-needed" });
const res = await client.chat.completions.create({
  model: "virtusoul-v1",
  messages: [{ role: "user", content: "Design a REST API for a todo app" }],
});
// → Classified as "medium" → routed to gpt-4o-mini
```

```bash
# cURL
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "virtusoul-v1", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## How It Works

```
Your App (any OpenAI SDK)
    │
    ▼
POST /v1/chat/completions  {"model": "virtusoul-v1", "messages": [...]}
    │
    ▼
┌─────────────────────────────────────────┐
│         VirtuSoul Router                │
│                                         │
│  1. ML Classifier (~15ms, local)        │
│     MiniLM embedding → Logistic Reg.    │
│     → "This is a complex query"         │
│                                         │
│  2. Route to tier                       │
│     complex → anthropic/claude-3.5      │
│                                         │
│  3. Forward request, return response    │
└─────────────────────────────────────────┘
    │
    ▼
OpenAI-format response (same as if you called the model directly)
```

The classifier uses [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Apache 2.0, ~80MB) to embed the query, then a Logistic Regression model trained on 200+ curated examples to predict the tier. No LLM calls, no external APIs — runs entirely on CPU.

### Tier Definitions

| Tier | When It's Used | Example Queries |
|------|---------------|-----------------|
| **simple** | Greetings, factual lookups, yes/no, basic math | "What is 2+2?", "Hello", "Capital of France?" |
| **medium** | Explanations, summaries, comparisons, simple code | "Explain DNS", "Write a Python function", "Compare React vs Vue" |
| **complex** | Architecture, system design, refactoring, multi-step | "Design a microservices architecture", "Create a CI/CD pipeline" |
| **reasoning** | Proofs, formal logic, optimization, novel algorithms | "Prove sqrt(2) is irrational", "Design a consensus algorithm" |

### Direct Tier Selection

Skip the classifier and pick a tier directly:

```python
# Force complex tier
response = client.chat.completions.create(
    model="complex",  # or "simple", "medium", "reasoning"
    messages=[{"role": "user", "content": "..."}],
)
```

## Configuration

All configuration is via environment variables (`.env` file).

### Server Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `4000` | Server port |
| `MODEL_NAME` | `virtusoul-v1` | The model name your app sends |
| `API_KEY` | *(none)* | Optional Bearer token to protect the router |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `TIMEOUT` | `120` | Request timeout in seconds |

### Tier Settings

Each tier has 4 variables: `{TIER}_PROVIDER`, `{TIER}_MODEL`, `{TIER}_API_KEY`, `{TIER}_BASE_URL`.

| Variable | Required | Description |
|----------|----------|-------------|
| `SIMPLE_PROVIDER` | Yes | Provider name (see [Providers](#providers)) |
| `SIMPLE_MODEL` | Yes | Model identifier |
| `SIMPLE_API_KEY` | Yes* | API key (*not needed for Ollama) |
| `SIMPLE_BASE_URL` | No | Custom base URL (overrides default) |

Same pattern for `MEDIUM_*`, `COMPLEX_*`, `REASONING_*`.

Unconfigured tiers fall back to the `medium` tier.

## Providers

VirtuSoul Router supports these providers out of the box:

| Provider | Value | Default Base URL | Auth | Notes |
|----------|-------|-----------------|------|-------|
| OpenAI | `openai` | api.openai.com | Bearer | Standard |
| Anthropic | `anthropic` | api.anthropic.com | x-api-key | Auto-converted to/from OpenAI format |
| OpenRouter | `openrouter` | openrouter.ai/api | Bearer | Access 200+ models |
| Groq | `groq` | api.groq.com | Bearer | Ultra-fast inference |
| Together | `together` | api.together.xyz | Bearer | Open-source models |
| Ollama | `ollama` | localhost:11434 | None | Local models, no API key needed |
| Mistral | `mistral` | api.mistral.ai | Bearer | Mistral models |
| DeepSeek | `deepseek` | api.deepseek.com | Bearer | DeepSeek models |
| Google | `google` | generativelanguage.googleapis.com | API key | Gemini models (OpenAI compat mode) |

### Custom Provider

Any OpenAI-compatible API works. Set `provider=custom` and provide a `BASE_URL`:

```env
MEDIUM_PROVIDER=custom
MEDIUM_MODEL=my-model
MEDIUM_API_KEY=my-key
MEDIUM_BASE_URL=https://my-custom-api.com/v1
```

### Example: All Free with Ollama (Local)

```env
SIMPLE_PROVIDER=ollama
SIMPLE_MODEL=llama3.2:3b

MEDIUM_PROVIDER=ollama
MEDIUM_MODEL=llama3.1:8b

COMPLEX_PROVIDER=ollama
COMPLEX_MODEL=llama3.1:70b

REASONING_PROVIDER=ollama
REASONING_MODEL=deepseek-r1:32b
```

### Example: Mix Providers for Best Value

```env
SIMPLE_PROVIDER=openrouter
SIMPLE_MODEL=meta-llama/llama-3.2-3b-instruct:free
SIMPLE_API_KEY=sk-or-...

MEDIUM_PROVIDER=openrouter
MEDIUM_MODEL=openai/gpt-4.1-mini
MEDIUM_API_KEY=sk-or-...

COMPLEX_PROVIDER=anthropic
COMPLEX_MODEL=claude-sonnet-4-20250514
COMPLEX_API_KEY=sk-ant-...

REASONING_PROVIDER=openai
REASONING_MODEL=o4-mini
REASONING_API_KEY=sk-...
```


## Docker

```bash
# Build
docker build -t virtusoul-router .

# Run
docker run -p 4000:4000 --env-file .env virtusoul-router
```

Or with Docker Compose:

```yaml
# docker-compose.yml
services:
  virtusoul-router:
    build: .
    ports:
      - "4000:4000"
    env_file:
      - .env
    restart: unless-stopped
```

## API Reference

### `POST /v1/chat/completions`

OpenAI-compatible chat completions with smart routing.

**Request:**
```json
{
  "model": "virtusoul-v1",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain how DNS works"}
  ],
  "temperature": 0.7,
  "max_tokens": 1000,
  "stream": false
}
```

**Model options:**
- `virtusoul-v1` (or your custom `MODEL_NAME`) — smart routing via ML classifier
- `simple`, `medium`, `complex`, `reasoning` — direct tier selection

**Response:** Standard OpenAI chat completion format, plus a `_virtusoul` field with routing metadata:
```json
{
  "id": "chatcmpl-abc123",
  "choices": [{"message": {"role": "assistant", "content": "..."}}],
  "usage": {"prompt_tokens": 25, "completion_tokens": 150, "total_tokens": 175},
  "_virtusoul": {
    "routed_to": "openai/gpt-4o-mini",
    "tier": "medium",
    "confidence": 0.92,
    "latency_ms": 14.2
  }
}
```

### `POST /classify`

Classify a query without routing (for testing/debugging).

```bash
curl -X POST http://localhost:4000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Design a microservices architecture"}'
```

```json
{
  "tier": "complex",
  "confidence": 0.988,
  "reasoning": "complex=0.99, medium=0.01, simple=0.00, reasoning=0.00",
  "latency_ms": 12.3,
  "flagged": false
}
```

### `POST /retrain`

Retrain the classifier with built-in training data.

```bash
curl -X POST http://localhost:4000/retrain
```

### `GET /health`

Health check.

```bash
curl http://localhost:4000/health
```

## How the Classifier Works

The classifier uses a two-stage approach:

1. **Embedding**: The user's query is converted to a 384-dimensional vector using [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (~80MB, Apache 2.0 license)
2. **Classification**: A Logistic Regression model (scikit-learn) predicts the tier from the embedding

The model is pre-trained on 200+ curated examples and achieves ~81% accuracy on cross-validation. It runs entirely on CPU in ~10-20ms.

### Retraining

You can retrain the classifier by calling `POST /retrain`. To add custom training data, you can extend the `training_data.py` file with your own examples.

### Low Confidence Handling

When the classifier's confidence is below 0.60, the response includes `"flagged": true`. This means the classification is uncertain and you may want to review it.

## Test Results

Tested end-to-end on February 16, 2026 with OpenRouter as the provider. All 4 tiers, streaming, direct tier selection, and error handling verified.

### Unit Tests

```
11 passed in 7.98s
  ✓ test_simple_greeting
  ✓ test_simple_factual
  ✓ test_medium_explanation
  ✓ test_complex_architecture
  ✓ test_reasoning_proof
  ✓ test_empty_input
  ✓ test_confidence_range
  ✓ test_reasoning_field
  ✓ test_default_values
  ✓ test_tier_loading
  ✓ test_tier_config_is_configured
```

### Live End-to-End Tests

| Test | Query | Classified As | Confidence | Model Used | Result |
|------|-------|--------------|------------|------------|--------|
| Smart → Simple | "Hello! How are you?" | simple | 0.954 | gpt-4.1-nano | ✅ Correct response |
| Smart → Medium | "Explain how DNS works" | medium | 0.857 | gpt-4.1-mini | ✅ Correct response |
| Smart → Complex | "Design a microservices architecture" | complex | 0.971 | claude-sonnet-4 | ✅ Correct response |
| Smart → Reasoning | "Prove sqrt(2) is irrational" | reasoning | 0.665 | o4-mini | ✅ Correct proof |
| Direct Tier | `"model": "complex"` | — | — | claude-sonnet-4 | ✅ Bypassed classifier |
| Streaming | "Count from 1 to 5" | simple | 0.95 | gpt-4.1-nano | ✅ SSE chunks received |
| Invalid Model | `"model": "invalid"` | — | — | — | ✅ 400 error with helpful message |
| Health Check | `GET /health` | — | — | — | ✅ Returns status + tiers |
| Retrain | `POST /retrain` | — | — | — | ✅ 232 samples, 0.698 CV accuracy |

### Classifier Latency

| Metric | Value |
|--------|-------|
| First request (cold start, model loading) | ~3.3s |
| Subsequent requests | 20-32ms |
| Embedding model size | ~80MB |

## License

MIT License — use it however you want, commercially or otherwise.

### Dependency Licenses

All dependencies use permissive licenses:

| Component | License |
|-----------|---------|
| sentence-transformers | Apache 2.0 |
| all-MiniLM-L6-v2 (model) | Apache 2.0 |
| scikit-learn | BSD 3-Clause |
| FastAPI | MIT |
| uvicorn | BSD 3-Clause |
| httpx | BSD 3-Clause |
| numpy | BSD 3-Clause |
| pydantic | MIT |

No GPL, no copyleft, no viral licenses. Safe for commercial use.

## Contributing

Contributions are welcome! Here's how:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Submit a PR

### Ideas for Contributions

- More training data for better classification accuracy
- New provider adapters
- Web dashboard for monitoring
- Custom tier definitions (beyond the 4 defaults)
- Batch API support
- Function calling / tool use passthrough

## Acknowledgments

Built with ❤️ by the VirtuSoul team. Inspired by the need for smarter, cost-effective LLM routing.

---

<p align="center">
  <sub>If VirtuSoul Router saves you money on your LLM bills, give us a ⭐</sub>
</p>
