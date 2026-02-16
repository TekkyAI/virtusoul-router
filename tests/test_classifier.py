"""Tests for the ML classifier."""

import pytest
from virtusoul_router.classifier import Classifier


@pytest.fixture(scope="module")
def classifier():
    c = Classifier()
    c.train()
    return c


class TestClassifier:
    def test_simple_greeting(self, classifier):
        result = classifier.classify("Hello")
        assert result.tier == "simple"
        assert result.confidence > 0.5

    def test_simple_factual(self, classifier):
        result = classifier.classify("What is the capital of France?")
        assert result.tier == "simple"

    def test_medium_explanation(self, classifier):
        result = classifier.classify("Explain how DNS works")
        assert result.tier == "medium"

    def test_complex_architecture(self, classifier):
        result = classifier.classify("Design a microservices architecture for an e-commerce platform")
        assert result.tier == "complex"

    def test_reasoning_proof(self, classifier):
        result = classifier.classify("Prove that the square root of 2 is irrational")
        assert result.tier == "reasoning"

    def test_empty_input(self, classifier):
        result = classifier.classify("")
        assert result.tier == "simple"
        assert result.confidence > 0.9

    def test_confidence_range(self, classifier):
        result = classifier.classify("What is 2 + 2?")
        assert 0.0 <= result.confidence <= 1.0

    def test_reasoning_field(self, classifier):
        result = classifier.classify("Hello")
        assert result.reasoning  # non-empty string
