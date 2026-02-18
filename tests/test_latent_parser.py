"""Tests for provider output normalization."""

import pytest

from nanobot.agent.latent_parser import ProviderOutputNormalizer


class TestProviderOutputNormalizer:
    """Test suite for ProviderOutputNormalizer."""

    def test_strip_markdown_with_json_fence(self):
        """Test stripping markdown code fences with json language."""
        text = "```json\n{\"key\": \"value\"}\n```"
        result = ProviderOutputNormalizer.strip_markdown(text)
        assert result == '{"key": "value"}'

    def test_strip_markdown_with_plain_fence(self):
        """Test stripping plain markdown code fences."""
        text = "```\n{\"key\": \"value\"}\n```"
        result = ProviderOutputNormalizer.strip_markdown(text)
        assert result == '{"key": "value"}'

    def test_strip_markdown_no_fence(self):
        """Test that text without fences is unchanged."""
        text = '{"key": "value"}'
        result = ProviderOutputNormalizer.strip_markdown(text)
        assert result == '{"key": "value"}'

    def test_extract_json_object_from_wrapped_text(self):
        """Test extracting JSON from text with extra content."""
        text = 'Here is the result: {"key": "value"} and some trailing text'
        result = ProviderOutputNormalizer.extract_json_object(text)
        assert result == '{"key": "value"}'

    def test_extract_json_object_nested_braces(self):
        """Test extracting JSON with nested objects."""
        text = 'Result: {"outer": {"inner": "value"}} extra'
        result = ProviderOutputNormalizer.extract_json_object(text)
        assert result == '{"outer": {"inner": "value"}}'

    def test_extract_json_object_no_json(self):
        """Test handling text without JSON."""
        text = "No JSON here"
        result = ProviderOutputNormalizer.extract_json_object(text)
        assert result == "No JSON here"

    def test_normalize_hypothesis_standard_keys(self):
        """Test normalizing hypothesis with standard keys."""
        hyp = {
            "intent": "test intent",
            "confidence": 0.8,
            "reasoning": "test reasoning"
        }
        result = ProviderOutputNormalizer.normalize_hypothesis(hyp)
        
        assert result is not None
        assert result["intent"] == "test intent"
        assert result["confidence"] == 0.8
        assert result["reasoning"] == "test reasoning"

    def test_normalize_hypothesis_alternative_keys(self):
        """Test normalizing hypothesis with alternative key names."""
        # Alternative key: hypothesis instead of intent
        hyp = {
            "hypothesis": "test hypothesis",
            "probability": 0.7,
            "rationale": "test rationale"
        }
        result = ProviderOutputNormalizer.normalize_hypothesis(hyp)
        
        assert result is not None
        assert result["intent"] == "test hypothesis"
        assert result["confidence"] == 0.7
        assert result["reasoning"] == "test rationale"

    def test_normalize_hypothesis_ollama_style(self):
        """Test normalizing Ollama-style response."""
        hyp = {
            "description": "user wants help",
            "confidence_score": 0.9,
            "explanation": "based on context"
        }
        result = ProviderOutputNormalizer.normalize_hypothesis(hyp)
        
        assert result is not None
        assert result["intent"] == "user wants help"
        assert result["confidence"] == 0.9
        assert result["reasoning"] == "based on context"

    def test_normalize_hypothesis_missing_confidence(self):
        """Test that missing confidence defaults to 0.5."""
        hyp = {
            "intent": "test",
            "reasoning": "test"
        }
        result = ProviderOutputNormalizer.normalize_hypothesis(hyp)
        
        assert result is not None
        assert result["confidence"] == 0.5

    def test_normalize_hypothesis_missing_reasoning(self):
        """Test that missing reasoning gets default value."""
        hyp = {
            "intent": "test",
            "confidence": 0.8
        }
        result = ProviderOutputNormalizer.normalize_hypothesis(hyp)
        
        assert result is not None
        assert result["reasoning"] == "latent inference"

    def test_normalize_hypothesis_no_intent(self):
        """Test that hypothesis without intent returns None."""
        hyp = {
            "confidence": 0.8,
            "reasoning": "test"
        }
        result = ProviderOutputNormalizer.normalize_hypothesis(hyp)
        
        assert result is None

    def test_normalize_hypothesis_confidence_bounds(self):
        """Test that confidence is clamped to [0, 1]."""
        hyp1 = {"intent": "test", "confidence": 1.5}
        result1 = ProviderOutputNormalizer.normalize_hypothesis(hyp1)
        assert result1["confidence"] == 1.0
        
        hyp2 = {"intent": "test", "confidence": -0.5}
        result2 = ProviderOutputNormalizer.normalize_hypothesis(hyp2)
        assert result2["confidence"] == 0.0

    def test_normalize_superpositional_state_standard(self):
        """Test normalizing standard SuperpositionalState."""
        state = {
            "hypotheses": [
                {"intent": "h1", "confidence": 0.8, "reasoning": "r1"},
                {"intent": "h2", "confidence": 0.6, "reasoning": "r2"}
            ],
            "entropy": 0.5,
            "strategic_direction": "proceed"
        }
        result = ProviderOutputNormalizer.normalize_superpositional_state(state)
        
        assert len(result["hypotheses"]) == 2
        assert result["entropy"] == 0.5
        assert result["strategic_direction"] == "proceed"

    def test_normalize_superpositional_state_alternative_keys(self):
        """Test normalizing with alternative key names."""
        state = {
            "hypotheses": [
                {"hypothesis": "h1", "probability": 0.8, "rationale": "r1"}
            ],
            "ambiguity": 0.7,
            "strategicDirection": "clarify"
        }
        result = ProviderOutputNormalizer.normalize_superpositional_state(state)
        
        assert len(result["hypotheses"]) == 1
        assert result["hypotheses"][0]["intent"] == "h1"
        assert result["entropy"] == 0.7
        assert result["strategic_direction"] == "clarify"

    def test_normalize_superpositional_state_single_hypothesis(self):
        """Test normalizing when response is a single hypothesis."""
        state = {
            "intent": "single hypothesis",
            "confidence": 0.9,
            "reasoning": "only one option"
        }
        result = ProviderOutputNormalizer.normalize_superpositional_state(state)
        
        assert len(result["hypotheses"]) == 1
        assert result["hypotheses"][0]["intent"] == "single hypothesis"

    def test_parse_and_normalize_with_markdown(self):
        """Test full parsing with markdown fences."""
        raw = '''```json
{
    "hypotheses": [
        {"intent": "help", "confidence": 0.8, "reasoning": "context"}
    ],
    "entropy": 0.3,
    "strategic_direction": "proceed"
}
```'''
        result = ProviderOutputNormalizer.parse_and_normalize(raw)
        
        assert len(result["hypotheses"]) == 1
        assert result["hypotheses"][0]["intent"] == "help"
        assert result["entropy"] == 0.3

    def test_parse_and_normalize_with_wrapped_json(self):
        """Test parsing JSON wrapped in text."""
        raw = 'Here is the analysis: {"hypotheses": [{"intent": "test", "confidence": 0.7, "reasoning": "r"}], "entropy": 0.4, "strategic_direction": "go"} end'
        result = ProviderOutputNormalizer.parse_and_normalize(raw)
        
        assert len(result["hypotheses"]) == 1
        assert result["entropy"] == 0.4

    def test_parse_and_normalize_empty_raises(self):
        """Test that empty input raises ValueError."""
        with pytest.raises(ValueError, match="Empty response"):
            ProviderOutputNormalizer.parse_and_normalize("")

    def test_parse_and_normalize_invalid_json_raises(self):
        """Test that invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            ProviderOutputNormalizer.parse_and_normalize("not json at all")

    def test_parse_and_normalize_safe_fallback(self):
        """Test safe version returns fallback on error."""
        result = ProviderOutputNormalizer.parse_and_normalize_safe("")
        
        assert result["hypotheses"] == []
        assert result["entropy"] == 0.0
        assert "parsing error" in result["strategic_direction"]

    def test_parse_and_normalize_safe_handles_malformed(self):
        """Test safe version handles malformed JSON gracefully."""
        result = ProviderOutputNormalizer.parse_and_normalize_safe("{{invalid")
        
        assert result["hypotheses"] == []
        assert result["entropy"] == 0.0

    def test_ollama_common_response_format(self):
        """Test common Ollama response format with extra markdown."""
        raw = '''Sure! Here's the analysis:

```json
{
  "hypotheses": [
    {
      "description": "User wants to create a file",
      "confidence_score": 0.85,
      "explanation": "Keywords indicate file creation"
    }
  ],
  "ambiguity": 0.2,
  "strategy": "Use create_file tool"
}
```

Hope this helps!'''
        result = ProviderOutputNormalizer.parse_and_normalize(raw)
        
        assert len(result["hypotheses"]) == 1
        assert result["hypotheses"][0]["intent"] == "User wants to create a file"
        assert abs(result["hypotheses"][0]["confidence"] - 0.85) < 0.01
        assert result["entropy"] == 0.2

    def test_openai_style_response(self):
        """Test OpenAI-style clean JSON response."""
        raw = '''{
  "hypotheses": [
    {
      "intent": "search for information",
      "confidence": 0.9,
      "reasoning": "explicit search request"
    },
    {
      "intent": "clarify topic",
      "confidence": 0.5,
      "reasoning": "ambiguous phrasing"
    }
  ],
  "entropy": 0.6,
  "strategic_direction": "Execute search with clarification prompt"
}'''
        result = ProviderOutputNormalizer.parse_and_normalize(raw)
        
        assert len(result["hypotheses"]) == 2
        assert result["hypotheses"][0]["intent"] == "search for information"
        assert result["entropy"] == 0.6

    def test_anthropic_style_with_thinking(self):
        """Test response with thinking/explanation wrapper."""
        raw = '''Let me analyze this request.

{
  "hypotheses": [
    {"intent": "code generation", "probability": 0.95, "rationale": "clear intent"}
  ],
  "uncertainty": 0.1,
  "strategicDirection": "Generate code with examples"
}

This is my analysis.'''
        result = ProviderOutputNormalizer.parse_and_normalize(raw)
        
        assert len(result["hypotheses"]) == 1
        assert result["hypotheses"][0]["intent"] == "code generation"
        assert result["entropy"] == 0.1
