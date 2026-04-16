"""Foundation Model / AI Gateway integration for running prompts.

Uses the OpenAI SDK to call Databricks serving endpoints (which expose an
OpenAI-compatible API).  Combined with ``mlflow.openai.autolog()`` (enabled
at app startup), every call automatically produces MLflow traces with token
usage, latencies, and structured spans — no manual span plumbing required.
"""

import re
import logging
from openai import AsyncOpenAI, APIStatusError, APITimeoutError
from server.config import get_workspace_host, get_oauth_token, get_workspace_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Eval-abort error hierarchy — raised by call_model when the error means
# we should stop the eval run immediately (not retry per-row).
# ---------------------------------------------------------------------------

class EvalAbortError(Exception):
    """Base for errors that should abort an eval run immediately."""


class TokenLimitError(EvalAbortError):
    """Prompt exceeds the model's context window."""


class RateLimitError(EvalAbortError):
    """Workspace or endpoint rate limit hit."""


# Keywords that indicate a context-length / token-limit error
_TOKEN_LIMIT_KEYWORDS = [
    "context_length_exceeded",
    "context window",
    "max tokens",
    "maximum context length",
    "token limit",
]


CHAT_TASKS = {
    "llm/v1/chat",
    "llm/v1/completions",
}

# Tasks to exclude (not useful for prompt testing)
EXCLUDE_TASKS = {
    "llm/v1/embeddings",
}

# Name patterns to exclude (internal/eval endpoints that clutter the list)
EXCLUDE_NAME_PATTERNS = [
    "internal-optimized-model-",
    "optimized-model-",
    "-v1-eval-",
    "-v3-eval-",
    "-mtpt-",
    "kie-",
]

# Foundation Model API endpoints always start with "databricks-"
FOUNDATION_PREFIX = "databricks-"


def _get_openai_client() -> AsyncOpenAI:
    """Build an AsyncOpenAI client pointed at the Databricks serving endpoint."""
    host = get_workspace_host()
    token = get_oauth_token()
    return AsyncOpenAI(
        api_key=token,
        base_url=f"{host}/serving-endpoints",
        timeout=120.0,
    )


def _clean_state(state_str: str) -> str:
    """Normalize state strings like 'EndpointStateReady.READY' to 'READY'."""
    if "." in state_str:
        return state_str.split(".")[-1]
    return state_str


def list_serving_endpoints(filter_chat_only: bool = True) -> list[dict]:
    """List available serving endpoints, filtering for chat-compatible ones.

    Returns only endpoints whose task is llm/v1/chat or llm/v1/completions,
    or Foundation Model API endpoints (databricks-*), excluding embeddings.
    """
    w = get_workspace_client()
    endpoints = []
    try:
        for ep in w.serving_endpoints.list():
            task = str(ep.task) if hasattr(ep, "task") and ep.task else "unknown"
            state = "UNKNOWN"
            if ep.state:
                state = _clean_state(
                    str(ep.state.ready) if hasattr(ep.state, "ready") else str(ep.state)
                )

            # Skip excluded tasks
            if task.lower() in EXCLUDE_TASKS:
                continue

            # Skip noisy internal/eval endpoints
            if any(pat in ep.name for pat in EXCLUDE_NAME_PATTERNS):
                continue

            if filter_chat_only:
                is_foundation = ep.name.startswith(FOUNDATION_PREFIX)
                is_chat = task.lower() in CHAT_TASKS
                if not (is_foundation or is_chat):
                    continue

            endpoints.append({
                "name": ep.name,
                "state": state,
                "task": task,
            })
    except Exception as e:
        logger.error("Error listing serving endpoints: %s", e)

    # Sort: foundation models first, then alphabetical
    endpoints.sort(key=lambda x: (0 if x["name"].startswith(FOUNDATION_PREFIX) else 1, x["name"]))
    return endpoints


async def call_model(
    endpoint_name: str,
    prompt: str,
    max_tokens: int = 4096,
    temperature: float = 1.0,
    system_prompt: str | None = None,
) -> dict:
    """Call a Databricks serving endpoint via the OpenAI SDK.

    mlflow.openai.autolog() (enabled at app startup) automatically traces
    every call, capturing token usage, latencies, and model parameters.
    """
    client = _get_openai_client()

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = await client.chat.completions.create(
            model=endpoint_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except APITimeoutError:
        raise TokenLimitError(
            "Request timed out — the prompt may be too long for this model. "
            "Try reducing the prompt length or the number of variables."
        )
    except APIStatusError as e:
        error_text = e.message or str(e)
        error_lower = error_text.lower()

        # Rate limit: HTTP 429 or REQUEST_LIMIT_EXCEEDED in body
        if e.status_code == 429 or "request_limit_exceeded" in error_lower:
            raise RateLimitError(
                f"Rate limit exceeded for endpoint '{endpoint_name}'. "
                "Try again in a few minutes or reduce concurrency."
            )

        # Token / context-length limit
        if any(kw in error_lower for kw in _TOKEN_LIMIT_KEYWORDS):
            raise TokenLimitError(
                "Prompt exceeds the model's context window. "
                "Try reducing the prompt length or the number of variables."
            )

        # Temperature-unsupported errors from external model proxies
        if e.status_code == 400 and "temperature" in error_text:
            search_text = error_text
            if "unsupported" in search_text.lower() or "not support" in search_text.lower():
                match = re.search(r"Only the default \(([^)]+)\) value is supported", search_text, re.IGNORECASE)
                hint = f" Try setting it to {match.group(1)}." if match else " Try adjusting temperature in settings."
                raise Exception(f"This model doesn't support the current temperature value.{hint}")

        raise Exception(f"Model API error ({e.status_code}): {error_text}")

    content = response.choices[0].message.content or "" if response.choices else ""
    usage = {}
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return {
        "content": content,
        "model": response.model or endpoint_name,
        "usage": usage,
    }
