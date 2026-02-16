FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy source
COPY virtusoul_router/ virtusoul_router/

# Pre-download the embedding model during build (~80MB)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Pre-train the classifier
RUN python -c "from virtusoul_router.classifier import get_classifier; get_classifier().train()"

EXPOSE 4000

CMD ["virtusoul-router"]
