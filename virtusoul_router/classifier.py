"""
ML Query Classifier — MiniLM embeddings + Logistic Regression.

Classifies user prompts into complexity tiers: simple, medium, complex, reasoning.
~80MB model, ~10-20ms inference on CPU. No LLM calls needed.

License: All dependencies are Apache 2.0 or BSD (sentence-transformers, scikit-learn, numpy).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from .training_data import TRAINING_DATA

logger = logging.getLogger("virtusoul.classifier")

MODEL_NAME = "all-MiniLM-L6-v2"
CLASSIFIER_PATH = Path(__file__).parent / "trained_classifier.joblib"
VALID_TIERS = ["simple", "medium", "complex", "reasoning"]


@dataclass
class ClassificationResult:
    tier: str
    confidence: float
    reasoning: str


class Classifier:
    """Fast query classifier using sentence embeddings + logistic regression."""

    def __init__(self):
        self._model: Optional[SentenceTransformer] = None
        self._classifier: Optional[LogisticRegression] = None
        self._is_trained = False

    def _load_model(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {MODEL_NAME} (first request may take a moment)...")
            self._model = SentenceTransformer(MODEL_NAME)
            logger.info("Embedding model loaded ✓")

    def train(self, extra_data: Optional[List[Tuple[str, str]]] = None) -> dict:
        """Train the classifier on built-in data + optional extra examples."""
        self._load_model()

        texts = [t[0] for t in TRAINING_DATA]
        labels = [t[1] for t in TRAINING_DATA]
        base_count = len(texts)

        extra_count = 0
        if extra_data:
            for text, tier in extra_data:
                if tier in VALID_TIERS:
                    texts.append(text)
                    labels.append(tier)
            extra_count = len(extra_data)

        logger.info(f"Training on {base_count} built-in + {extra_count} custom = {len(texts)} total examples")
        embeddings = self._model.encode(texts, show_progress_bar=False)

        self._classifier = LogisticRegression(max_iter=1000, C=10.0, class_weight="balanced")
        self._classifier.fit(embeddings, labels)

        cv_folds = min(5, len(set(labels)))
        scores = cross_val_score(self._classifier, embeddings, labels, cv=cv_folds, scoring="accuracy")

        joblib.dump(self._classifier, CLASSIFIER_PATH)
        self._is_trained = True

        result = {
            "total_samples": len(texts),
            "cv_accuracy": f"{scores.mean():.3f} (+/- {scores.std():.3f})",
        }
        logger.info(f"Training complete: {result}")
        return result

    def load(self) -> bool:
        """Load a previously trained classifier from disk."""
        self._load_model()
        if CLASSIFIER_PATH.exists():
            self._classifier = joblib.load(CLASSIFIER_PATH)
            self._is_trained = True
            logger.info("Loaded trained classifier from disk ✓")
            return True
        return False

    def classify(self, text: str) -> ClassificationResult:
        """Classify a query into a complexity tier."""
        if not text or not text.strip():
            return ClassificationResult(tier="simple", confidence=0.95, reasoning="Empty input")

        if not self._is_trained:
            if not self.load():
                self.train()

        embedding = self._model.encode([text], show_progress_bar=False)
        probs = self._classifier.predict_proba(embedding)[0]
        classes = self._classifier.classes_

        tier_idx = np.argmax(probs)
        tier = classes[tier_idx]
        confidence = float(probs[tier_idx])

        sorted_indices = np.argsort(probs)[::-1]
        prob_str = ", ".join(f"{classes[i]}={probs[i]:.2f}" for i in sorted_indices)
        reasoning = f"{prob_str}"

        return ClassificationResult(tier=tier, confidence=round(confidence, 3), reasoning=reasoning)


# Singleton
_instance: Optional[Classifier] = None


def get_classifier() -> Classifier:
    global _instance
    if _instance is None:
        _instance = Classifier()
    return _instance
