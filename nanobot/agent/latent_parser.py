"""Robust parser/normalizer for LLM provider outputs.

Handles:
- Markdown code fences
- Alternative key names
- Malformed JSON
- Empty responses
- Different provider response formats (Ollama, OpenAI, Anthropic)
"""

import json
from json import JSONDecodeError
from typing import Any

from loguru import logger


class ProviderOutputNormalizer:
    """Normalizes raw LLM provider outputs into consistent SuperpositionalState format."""

    # Alternative key mappings for hypothesis fields
    INTENT_KEYS = ["intent", "hypothesis", "description", "interpretation"]
    CONFIDENCE_KEYS = ["confidence", "probability", "confidence_score", "score"]
    REASONING_KEYS = ["reasoning", "rationale", "explanation", "justification"]
    ENTROPY_KEYS = ["entropy", "ambiguity", "uncertainty"]
    STRATEGIC_KEYS = ["strategic_direction", "strategicDirection", "strategy", "direction"]

    @staticmethod
    def strip_markdown(text: str) -> str:
        """Remove markdown code fences from text.

        Handles:
        - ```json ... ```
        - ``` ... ```
        - Mixed whitespace
        """
        if not text:
            return text

        text = text.strip()

        # Remove opening fence (```json or ```)
        if text.startswith("```"):
            # Find first newline after opening fence
            newline_pos = text.find("\n")
            if newline_pos != -1:
                text = text[newline_pos + 1:]
            else:
                # No newline, just strip the backticks
                text = text.lstrip("`").lstrip("json").lstrip()

        # Remove closing fence
        if text.endswith("```"):
            text = text[:-3].rstrip()

        return text.strip()

    @staticmethod
    def extract_json_object(text: str) -> str:
        """Extract the first valid JSON object from text.

        Handles cases where JSON is wrapped in text or has trailing content.
        """
        if not text:
            return text

        # Try to find JSON object boundaries
        start_idx = text.find("{")
        if start_idx == -1:
            return text

        # Find matching closing brace
        brace_count = 0
        for i in range(start_idx, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    return text[start_idx : i + 1]

        # If no matching brace found, return from first { to end
        return text[start_idx:]

    @classmethod
    def normalize_hypothesis(cls, item: dict[str, Any]) -> dict[str, Any] | None:
        """Normalize a single hypothesis dict with alternative key names.

        Args:
            item: Raw hypothesis dict from LLM

        Returns:
            Normalized dict with standard keys, or None if invalid
        """
        if not isinstance(item, dict):
            return None

        normalized = {}

        # Extract intent
        intent = None
        for key in cls.INTENT_KEYS:
            if key in item:
                intent = str(item[key])
                break
        if not intent:
            logger.debug(f"No intent found in hypothesis: {list(item.keys())}")
            return None
        normalized["intent"] = intent

        # Extract confidence (default to 0.5 if missing)
        confidence = 0.5
        for key in cls.CONFIDENCE_KEYS:
            if key in item:
                try:
                    confidence = float(item[key])
                    break
                except (ValueError, TypeError):
                    pass
        normalized["confidence"] = max(0.0, min(1.0, confidence))

        # Extract reasoning (default to generic text if missing)
        reasoning = "latent inference"
        for key in cls.REASONING_KEYS:
            if key in item:
                reasoning = str(item[key])
                break
        normalized["reasoning"] = reasoning

        # Extract required_tools if present
        if "required_tools" in item and isinstance(item["required_tools"], list):
            normalized["required_tools"] = item["required_tools"]

        return normalized

    @classmethod
    def normalize_superpositional_state(cls, parsed: dict[str, Any]) -> dict[str, Any]:
        """Normalize a SuperpositionalState dict with alternative key names.

        Args:
            parsed: Raw parsed JSON from LLM

        Returns:
            Normalized SuperpositionalState dict
        """
        normalized = {
            "hypotheses": [],
            "entropy": 0.0,
            "strategic_direction": "Proceed with standard processing.",
        }

        # Extract hypotheses
        hypotheses_data = None
        if "hypotheses" in parsed:
            hypotheses_data = parsed["hypotheses"]
        elif all(k in parsed for k in ("intent", "confidence")):
            # Single hypothesis format
            hypotheses_data = [parsed]
        elif any(k in parsed for k in cls.INTENT_KEYS):
            # Might be a single hypothesis with alternative keys
            hypotheses_data = [parsed]

        if hypotheses_data and isinstance(hypotheses_data, list):
            for item in hypotheses_data:
                normalized_hyp = cls.normalize_hypothesis(item)
                if normalized_hyp:
                    normalized["hypotheses"].append(normalized_hyp)

        # Extract entropy
        for key in cls.ENTROPY_KEYS:
            if key in parsed:
                try:
                    normalized["entropy"] = float(parsed[key])
                    break
                except (ValueError, TypeError):
                    pass

        # Extract strategic direction
        for key in cls.STRATEGIC_KEYS:
            if key in parsed:
                normalized["strategic_direction"] = str(parsed[key])
                break

        return normalized

    @classmethod
    def parse_and_normalize(cls, raw_output: str) -> dict[str, Any]:
        """Parse and normalize raw LLM output into SuperpositionalState format.

        Args:
            raw_output: Raw text output from LLM provider

        Returns:
            Normalized SuperpositionalState dict with standard keys

        Raises:
            ValueError: If output is empty or cannot be parsed
        """
        if not raw_output or not raw_output.strip():
            raise ValueError("Empty response from LLM")

        # Step 1: Strip markdown fences
        cleaned = cls.strip_markdown(raw_output)

        # Step 2: Extract JSON object
        json_str = cls.extract_json_object(cleaned)

        # Step 3: Parse JSON
        try:
            parsed = json.loads(json_str)
        except JSONDecodeError as e:
            logger.warning(
                f"JSON parse error: {e}. "
                f"Response (truncated): {raw_output[:200]}..."
            )
            raise ValueError(f"Invalid JSON: {e}") from e

        # Step 4: Validate structure
        if not isinstance(parsed, dict):
            raise ValueError(
                f"Expected dict, got {type(parsed).__name__}. "
                f"Response: {str(parsed)[:200]}..."
            )

        # Step 5: Normalize keys and structure
        normalized = cls.normalize_superpositional_state(parsed)

        # Validate we got at least something useful
        if not normalized["hypotheses"]:
            logger.warning(
                f"No valid hypotheses found. "
                f"Original keys: {list(parsed.keys())}"
            )

        return normalized

    @classmethod
    def parse_and_normalize_safe(cls, raw_output: str) -> dict[str, Any]:
        """Safe version that returns fallback instead of raising.

        Args:
            raw_output: Raw text output from LLM provider

        Returns:
            Normalized SuperpositionalState dict, or fallback dict on error
        """
        try:
            return cls.parse_and_normalize(raw_output)
        except Exception as e:
            logger.warning(f"Failed to parse/normalize LLM output: {e}")
            return {
                "hypotheses": [],
                "entropy": 0.0,
                "strategic_direction": "Proceed with standard processing due to parsing error.",
            }
