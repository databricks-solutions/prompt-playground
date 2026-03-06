"""Tests for error detection in llm.call_model.

Covers:
- RateLimitError raised on HTTP 429 status
- RateLimitError raised when error body contains REQUEST_LIMIT_EXCEEDED
- TokenLimitError raised when error body contains context-length keywords
- EvalAbortError is the common base for both RateLimitError and TokenLimitError
- Generic model errors raise plain Exception (not EvalAbortError)
- Successful 200 response returns content, model, and usage
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from server.llm import call_model, EvalAbortError, TokenLimitError, RateLimitError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_session(status: int, body: str):
    """Build a mock aiohttp session that returns a response with given status and body."""
    mock_response = AsyncMock()
    mock_response.status = status
    mock_response.text = AsyncMock(return_value=body)
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post.return_value = mock_response
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    return mock_session


def _make_success_session(content: str = "hello", model: str = "test-model"):
    """Build a mock aiohttp session that returns a successful 200 response."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        "choices": [{"message": {"content": content}}],
        "model": model,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post.return_value = mock_response
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    return mock_session


@pytest.fixture(autouse=True)
def _patch_config(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "https://test.databricks.com")
    with patch("server.llm.get_workspace_host", return_value="https://test.databricks.com"), \
         patch("server.llm.get_oauth_token", return_value="test-token"):
        yield


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class TestExceptionHierarchy:

    def test_token_limit_error_is_eval_abort_error(self):
        assert issubclass(TokenLimitError, EvalAbortError)

    def test_rate_limit_error_is_eval_abort_error(self):
        assert issubclass(RateLimitError, EvalAbortError)

    def test_eval_abort_error_is_exception(self):
        assert issubclass(EvalAbortError, Exception)

    def test_token_limit_error_can_be_caught_as_eval_abort_error(self):
        with pytest.raises(EvalAbortError):
            raise TokenLimitError("context exceeded")

    def test_rate_limit_error_can_be_caught_as_eval_abort_error(self):
        with pytest.raises(EvalAbortError):
            raise RateLimitError("rate exceeded")


# ---------------------------------------------------------------------------
# Rate limit detection
# ---------------------------------------------------------------------------

class TestRateLimitDetection:

    @pytest.mark.asyncio
    async def test_raises_rate_limit_error_on_429(self):
        session = _make_mock_session(429, '{"error_code":"RATE_LIMIT","message":"too many requests"}')
        with patch("aiohttp.ClientSession", return_value=session):
            with pytest.raises(RateLimitError):
                await call_model("test-endpoint", "hello")

    @pytest.mark.asyncio
    async def test_raises_rate_limit_error_on_request_limit_exceeded_body(self):
        body = '{"error_code":"REQUEST_LIMIT_EXCEEDED","message":"Exceeded workspace input tokens per minute rate limit"}'
        session = _make_mock_session(400, body)
        with patch("aiohttp.ClientSession", return_value=session):
            with pytest.raises(RateLimitError):
                await call_model("test-endpoint", "hello")

    @pytest.mark.asyncio
    async def test_rate_limit_error_message_is_user_friendly(self):
        session = _make_mock_session(429, '{"message":"rate exceeded"}')
        with patch("aiohttp.ClientSession", return_value=session):
            with pytest.raises(RateLimitError) as exc_info:
                await call_model("test-endpoint", "hello")
        assert "rate limit" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_rate_limit_is_eval_abort_error(self):
        """RateLimitError must be catchable as EvalAbortError for fast-fail to work."""
        session = _make_mock_session(429, '{}')
        with patch("aiohttp.ClientSession", return_value=session):
            with pytest.raises(EvalAbortError):
                await call_model("test-endpoint", "hello")


# ---------------------------------------------------------------------------
# Token limit detection
# ---------------------------------------------------------------------------

class TestTokenLimitDetection:

    @pytest.mark.asyncio
    async def test_raises_token_limit_error_on_context_length_exceeded(self):
        body = '{"error_code":"BAD_REQUEST","message":"context_length_exceeded: prompt too long"}'
        session = _make_mock_session(400, body)
        with patch("aiohttp.ClientSession", return_value=session):
            with pytest.raises(TokenLimitError):
                await call_model("test-endpoint", "hello")

    @pytest.mark.asyncio
    async def test_raises_token_limit_error_on_context_window_phrase(self):
        body = '{"message":"This model has a context window of 8192 tokens"}'
        session = _make_mock_session(400, body)
        with patch("aiohttp.ClientSession", return_value=session):
            with pytest.raises(TokenLimitError):
                await call_model("test-endpoint", "hello")

    @pytest.mark.asyncio
    async def test_raises_token_limit_error_on_max_tokens_phrase(self):
        body = '{"message":"Input exceeds max tokens allowed"}'
        session = _make_mock_session(400, body)
        with patch("aiohttp.ClientSession", return_value=session):
            with pytest.raises(TokenLimitError):
                await call_model("test-endpoint", "hello")

    @pytest.mark.asyncio
    async def test_token_limit_error_message_is_user_friendly(self):
        body = '{"message":"context_length_exceeded"}'
        session = _make_mock_session(400, body)
        with patch("aiohttp.ClientSession", return_value=session):
            with pytest.raises(TokenLimitError) as exc_info:
                await call_model("test-endpoint", "hello")
        assert "context window" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_token_limit_is_eval_abort_error(self):
        """TokenLimitError must be catchable as EvalAbortError for fast-fail to work."""
        body = '{"message":"context_length_exceeded"}'
        session = _make_mock_session(400, body)
        with patch("aiohttp.ClientSession", return_value=session):
            with pytest.raises(EvalAbortError):
                await call_model("test-endpoint", "hello")


# ---------------------------------------------------------------------------
# Generic errors (should NOT be EvalAbortError)
# ---------------------------------------------------------------------------

class TestGenericErrors:

    @pytest.mark.asyncio
    async def test_generic_500_raises_plain_exception(self):
        body = '{"message":"Internal server error"}'
        session = _make_mock_session(500, body)
        with patch("aiohttp.ClientSession", return_value=session):
            with pytest.raises(Exception) as exc_info:
                await call_model("test-endpoint", "hello")
        assert not isinstance(exc_info.value, EvalAbortError)

    @pytest.mark.asyncio
    async def test_generic_400_raises_plain_exception(self):
        body = '{"message":"Invalid request format"}'
        session = _make_mock_session(400, body)
        with patch("aiohttp.ClientSession", return_value=session):
            with pytest.raises(Exception) as exc_info:
                await call_model("test-endpoint", "hello")
        assert not isinstance(exc_info.value, EvalAbortError)


# ---------------------------------------------------------------------------
# Successful response
# ---------------------------------------------------------------------------

class TestSuccessfulResponse:

    @pytest.mark.asyncio
    async def test_returns_content_model_usage_on_200(self):
        session = _make_success_session(content="test response", model="databricks-claude")
        with patch("aiohttp.ClientSession", return_value=session):
            result = await call_model("test-endpoint", "prompt")
        assert result["content"] == "test response"
        assert result["model"] == "databricks-claude"
        assert "total_tokens" in result["usage"]
