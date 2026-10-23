"""Model client.

This file defines ``call``, the single function used to send a prompt to
Gemini 3.5 Flash, Gemini 3.5 Flash-Lite, or Qwen3-8B. It selects the
provider, sends the request through LiteLLM, and returns the response
text with token counts. Repeated requests are read from the disk cache.
"""

import logging
import os
import time
from dataclasses import dataclass

import litellm
from dotenv import load_dotenv
from litellm.exceptions import RateLimitError, ServiceUnavailableError

from harness.cache import get as cache_get
from harness.cache import put as cache_put
from harness.pricing import compute_cost

logger = logging.getLogger(__name__)

# Transient provider failures (e.g. Gemini 503 high demand).
MAX_RETRIES = 5
RETRY_BASE_SECONDS = 5.0

# Load API settings from a local .env file when present.
load_dotenv()

# Logical model names mapped to LiteLLM provider identifiers.
MODELS = {
    "flash": "gemini/gemini-3.5-flash",
    "flash-lite": "gemini/gemini-3.5-flash-lite",
    "qwen": "openai",
}


@dataclass
class ModelResponse:
    """Text and token usage for a single model call."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached: bool = False
    cost: float = 0.0


def _build_response(
    text: str, model: str, prompt_tokens: int, completion_tokens: int, cached: bool
) -> ModelResponse:
    """Attach the cost to a response and log the token counts."""
    cost = compute_cost(model, prompt_tokens, completion_tokens)
    logger.info(
        "model=%s tokens_in=%d tokens_out=%d cost=$%.6f cached=%s",
        model,
        prompt_tokens,
        completion_tokens,
        cost,
        cached,
    )
    return ModelResponse(
        text=text,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached=cached,
        cost=cost,
    )


def call(model: str, messages: list[dict]) -> ModelResponse:
    """Call a registered model and return the response text and token counts.

    Args:
        model: One of ``flash``, ``flash-lite``, or ``qwen``.
        messages: Chat messages in OpenAI format.
    """
    if model not in MODELS:
        allowed = ", ".join(MODELS)
        raise ValueError(f"Unknown model {model!r}. Use one of: {allowed}")

    stored = cache_get(model, messages)
    if stored is not None:
        return _build_response(
            stored["text"],
            stored["model"],
            stored["prompt_tokens"],
            stored["completion_tokens"],
            cached=True,
        )

    kwargs = {
        "messages": messages,
        # Fixed so repeated prompts stay deterministic.
        "temperature": 0,
    }

    if model == "qwen":
        api_base = os.environ.get("QWEN_API_BASE", "").strip()
        if not api_base:
            raise RuntimeError(
                "QWEN_API_BASE is not set. Set it to the address of the computer "
                "where Qwen is running."
            )
        model_name = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-8B-AWQ")
        kwargs["model"] = "openai/" + model_name
        kwargs["api_base"] = api_base
        kwargs["api_key"] = os.environ.get("QWEN_API_KEY", "not-needed")
        # Qwen3 thinking mode is disabled for comparable runs.
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    else:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Set it in the environment or in a .env file."
            )
        kwargs["model"] = MODELS[model]
        kwargs["api_key"] = api_key

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = litellm.completion(**kwargs)
            break
        except (ServiceUnavailableError, RateLimitError) as exc:
            last_error = exc
            if attempt >= MAX_RETRIES:
                raise
            wait = RETRY_BASE_SECONDS * (2 ** attempt)
            logger.warning(
                "model=%s attempt=%d/%d failed (%s); retrying in %.0fs",
                model,
                attempt + 1,
                MAX_RETRIES + 1,
                type(exc).__name__,
                wait,
            )
            time.sleep(wait)
    else:
        raise last_error  # pragma: no cover - loop always breaks or raises

    usage = response.usage

    result = _build_response(
        response.choices[0].message.content or "",
        model,
        getattr(usage, "prompt_tokens", 0) or 0,
        getattr(usage, "completion_tokens", 0) or 0,
        cached=False,
    )
    cache_put(
        model,
        messages,
        {
            "text": result.text,
            "model": result.model,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
        },
    )
    return result