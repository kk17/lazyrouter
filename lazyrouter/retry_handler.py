"""Retry and fallback logic for handling model failures"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from .config import ModelConfig

logger = logging.getLogger(__name__)

# Hardcoded defaults - no need for user configuration
INITIAL_RETRY_DELAY = 10.0  # seconds
RETRY_MULTIPLIER = 2.0
MAX_FALLBACK_MODELS = 3  # try up to 3 models before giving up
DEFAULT_RATE_LIMIT_COOLDOWN = 60.0  # seconds to block a model if no reset time is known


def is_rate_limit_error(error: Exception) -> bool:
    """Check if an error is specifically a rate limit (429) error."""
    err_str = str(error).lower()
    return "429" in err_str or ("rate" in err_str and "limit" in err_str)


def _find_response_headers(error: Exception) -> Optional[object]:
    """Walk the exception chain and return the first non-None headers object found.

    LiteLLM's Anthropic 429 handler does not forward the original httpx response
    to the exception it raises, so the headers only appear on the inner
    ``__context__`` exception (typically ``HTTPStatusError``).
    """
    seen: set[int] = set()
    stack: list[BaseException] = [error]
    while stack:
        exc = stack.pop()
        if id(exc) in seen:
            continue
        seen.add(id(exc))
        response = getattr(exc, "response", None)
        if response is not None:
            headers = getattr(response, "headers", None)
            if headers:
                return headers
        if exc.__cause__ is not None:
            stack.append(exc.__cause__)
        if exc.__context__ is not None:
            stack.append(exc.__context__)
    return None


def extract_rate_limit_reset_dt(error: Exception) -> Optional[datetime]:
    """Return the exact reset datetime from provider response headers, or None.

    Walks the full exception chain to find headers (needed for Anthropic where
    LiteLLM drops the response object from the top-level exception).

    Checks headers in order:
      1. ``anthropic-ratelimit-unified-reset`` — Unix timestamp (Anthropic)
      2. ``retry-after`` as an HTTP-date string
      3. ``retry-after`` as integer seconds → computes ``now + seconds``
    Returns None only when no usable hint exists at all.
    """
    headers = _find_response_headers(error)
    if headers is None:
        return None

    # Anthropic unified rate limit reset — Unix timestamp integer
    for header_name in (
        "anthropic-ratelimit-unified-reset",
        "anthropic-ratelimit-unified-5h-reset",
    ):
        reset_str = headers.get(header_name)
        if reset_str:
            try:
                return datetime.fromtimestamp(int(reset_str), tz=timezone.utc)
            except (ValueError, OSError):
                pass

    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after:
        # Try HTTP-date format first
        try:
            float(retry_after)  # raises if it's not a plain number
        except ValueError:
            try:
                from email.utils import parsedate_to_datetime
                return parsedate_to_datetime(retry_after)
            except Exception:
                pass
        else:
            # Plain integer seconds — compute absolute time
            try:
                return datetime.now(timezone.utc) + timedelta(seconds=float(retry_after))
            except (ValueError, OverflowError):
                pass

    return None


def extract_rate_limit_reset_seconds(error: Exception) -> float:
    """Return seconds until the rate limit resets.

    Uses the exact reset datetime when available; falls back to
    DEFAULT_RATE_LIMIT_COOLDOWN when no hint is present.
    """
    reset_dt = extract_rate_limit_reset_dt(error)
    if reset_dt is not None:
        delta = (reset_dt - datetime.now(timezone.utc)).total_seconds()
        return max(delta, 0.0)

    headers = _find_response_headers(error)
    if headers is not None:
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after:
            try:
                return max(float(retry_after), 0.0)
            except ValueError:
                pass

    return DEFAULT_RATE_LIMIT_COOLDOWN


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable (rate limit, temporary failure, etc.)"""
    err_str = str(error).lower()

    # Rate limit errors
    if "429" in err_str or ("rate" in err_str and "limit" in err_str):
        return True

    # Service unavailable
    if "503" in err_str or "service unavailable" in err_str:
        return True

    # Temporary/transient errors
    if "502" in err_str or "bad gateway" in err_str:
        return True
    if "504" in err_str or "gateway timeout" in err_str:
        return True

    # Connection errors (often transient)
    if "connection" in err_str and ("reset" in err_str or "refused" in err_str):
        return True

    # Timeout errors
    if "timeout" in err_str or "timed out" in err_str:
        return True

    # Overloaded errors (Anthropic)
    if "overloaded" in err_str:
        return True

    return False


def get_model_elo(model_config: ModelConfig) -> int:
    """Get a representative ELO for a model (average of coding and writing)"""
    coding = model_config.coding_elo or 0
    writing = model_config.writing_elo or 0
    if coding and writing:
        return (coding + writing) // 2
    return coding or writing or 0


def select_fallback_models(
    failed_model: str,
    all_models: dict[str, ModelConfig],
    healthy_models: Optional[set[str]] = None,
    already_tried: Optional[set[str]] = None,
) -> List[str]:
    """
    Select fallback models. Uses explicit fallback_models list if configured,
    otherwise falls back to ELO-similarity ordering.

    Prefers healthy models over unhealthy ones within each ordering strategy.
    """
    if already_tried is None:
        already_tried = set()

    failed_config = all_models.get(failed_model)

    # Use explicit fallback list if configured on the failed model
    if failed_config and failed_config.fallback_models:
        result = []
        for name in failed_config.fallback_models:
            if name == failed_model or name in already_tried:
                continue
            if name not in all_models:
                continue
            # Skip unhealthy models to avoid wasting time on known-failed providers
            if healthy_models is not None and name not in healthy_models:
                continue
            result.append(name)
            if len(result) >= MAX_FALLBACK_MODELS - 1:
                break
        return result

    failed_elo = get_model_elo(failed_config) if failed_config else 0

    # Get candidate models (healthy first, then unhealthy as last resort)
    candidates = []
    for name, cfg in all_models.items():
        if name == failed_model or name in already_tried:
            continue
        is_healthy = healthy_models is None or name in healthy_models
        elo = get_model_elo(cfg)
        elo_diff = abs(elo - failed_elo) if failed_elo else elo
        # Sort key: (not healthy, elo_diff) - healthy models first, then by ELO similarity
        candidates.append((not is_healthy, elo_diff, name))

    candidates.sort()
    return [name for _, _, name in candidates[: MAX_FALLBACK_MODELS - 1]]
