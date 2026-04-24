from __future__ import annotations

import random
import time


DEFAULT_LLM_RATE_LIMIT_RETRIES = 8
DEFAULT_LLM_RATE_LIMIT_BASE_DELAY_S = 2.0
DEFAULT_LLM_RATE_LIMIT_MAX_DELAY_S = 60.0


def _coerce_retry_after_seconds(value) -> float | None:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        seconds = float(value)
        return seconds if seconds > 0 else None

    if isinstance(value, str):
        try:
            seconds = float(value.strip())
        except ValueError:
            return None
        return seconds if seconds > 0 else None

    return None


def _extract_retry_after_seconds(exc: Exception) -> float | None:
    for attr_name in ("retry_after", "retry_after_seconds"):
        retry_after = _coerce_retry_after_seconds(getattr(exc, attr_name, None))
        if retry_after is not None:
            return retry_after

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers:
        for header_name in (
            "retry-after",
            "Retry-After",
            "x-ratelimit-reset-requests",
            "x-ratelimit-reset-tokens",
        ):
            retry_after = _coerce_retry_after_seconds(headers.get(header_name))
            if retry_after is not None:
                return retry_after

    return None


def is_rate_limit_error(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) == 429:
        return True

    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) == 429:
        return True

    body = getattr(exc, "body", None)
    if isinstance(body, dict) and body.get("status") == 429:
        return True

    message = str(exc).lower()
    return (
        "too many requests" in message
        or "rate limit" in message
        or "ratelimit" in message
        or "error code: 429" in message
        or "{'status': 429" in message
    )


def invoke_with_rate_limit_retry(
    llm_with_tools,
    messages,
    *,
    max_retries: int = DEFAULT_LLM_RATE_LIMIT_RETRIES,
    base_delay_s: float = DEFAULT_LLM_RATE_LIMIT_BASE_DELAY_S,
    max_delay_s: float = DEFAULT_LLM_RATE_LIMIT_MAX_DELAY_S,
):
    """Invoke the model, retrying the same step on provider 429 responses."""
    retry_count = 0

    while True:
        try:
            return llm_with_tools.invoke(messages)
        except Exception as exc:
            if not is_rate_limit_error(exc):
                raise

            if retry_count >= max_retries:
                raise RuntimeError(
                    f"LLM rate-limit retries exhausted after {max_retries} retries: {exc}"
                ) from exc

            retry_count += 1
            retry_after_s = _extract_retry_after_seconds(exc)
            if retry_after_s is None:
                delay_s = min(max_delay_s, base_delay_s * (2 ** (retry_count - 1)))
            else:
                delay_s = min(max_delay_s, max(base_delay_s, retry_after_s))

            jitter_s = min(1.0, delay_s * 0.1) * random.random()
            wait_s = delay_s + jitter_s
            print(
                "   → Model API rate limited (429). "
                f"Waiting {wait_s:.1f}s before retrying the same step "
                f"[retry {retry_count}/{max_retries}]."
            )
            time.sleep(wait_s)
