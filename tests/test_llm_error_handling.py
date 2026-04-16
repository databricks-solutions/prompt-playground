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
from openai import APIStatusError, APITimeoutError
from server.llm import call_model, EvalAbortError, TokenLimitError, RateLimitError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_api_status_error(status_code: int, message: str) -> APIStatusError:
    """Build an APIStatusError that mimics what the OpenAI SDK raises."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.headers = {}
    mock_response.json.return_value = {"error": {"message": message}}
    return APIStatusError(message=message, response=mock_response, body={"error": {"message": message}})


def _make_success_response(content: str = "hello", model: str = "test-model"):
    """Build a mock ChatCompletion response object."""
    mock_message = MagicMock()
    mock_message.content = content

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5
    mock_usage.total_tokens = 15

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.model = model
    mock_response.usage = mock_usage
    return mock_response


@pytest.fixture(autouse=True)
def _patch_config(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "https://test.databricks.com")
    # Reset cached client so each test starts fresh
    import server.llm
    server.llm._openai_client = None
    server.llm._openai_client_token = None
    with patch("server.llm.get_workspace_host", return_value="https://test.databricks.com"), \
         patch("server.llm.get_oauth_token", return_value="test-token"):
        yield


def _patch_openai_create(side_effect=None, return_value=None):
    """Patch the AsyncOpenAI client's chat.completions.create method."""
    mock_client = MagicMock()
    mock_create = AsyncMock(side_effect=side_effect, return_value=return_value)
    mock_client.chat.completions.create = mock_create
    return patch("server.llm._get_openai_client", return_value=mock_client)


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
        error = _make_api_status_error(429, "too many requests")
        with _patch_openai_create(side_effect=error):
            with pytest.raises(RateLimitError):
                await call_model("test-endpoint", "hello")

    @pytest.mark.asyncio
    async def test_raises_rate_limit_error_on_request_limit_exceeded_body(self):
        error = _make_api_status_error(400, "REQUEST_LIMIT_EXCEEDED: Exceeded workspace input tokens per minute rate limit")
        with _patch_openai_create(side_effect=error):
            with pytest.raises(RateLimitError):
                await call_model("test-endpoint", "hello")

    @pytest.mark.asyncio
    async def test_rate_limit_error_message_is_user_friendly(self):
        error = _make_api_status_error(429, "rate exceeded")
        with _patch_openai_create(side_effect=error):
            with pytest.raises(RateLimitError) as exc_info:
                await call_model("test-endpoint", "hello")
        assert "rate limit" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_rate_limit_is_eval_abort_error(self):
        """RateLimitError must be catchable as EvalAbortError for fast-fail to work."""
        error = _make_api_status_error(429, "rate exceeded")
        with _patch_openai_create(side_effect=error):
            with pytest.raises(EvalAbortError):
                await call_model("test-endpoint", "hello")


# ---------------------------------------------------------------------------
# Token limit detection
# ---------------------------------------------------------------------------

class TestTokenLimitDetection:

    @pytest.mark.asyncio
    async def test_raises_token_limit_error_on_context_length_exceeded(self):
        error = _make_api_status_error(400, "context_length_exceeded: prompt too long")
        with _patch_openai_create(side_effect=error):
            with pytest.raises(TokenLimitError):
                await call_model("test-endpoint", "hello")

    @pytest.mark.asyncio
    async def test_raises_token_limit_error_on_context_window_phrase(self):
        error = _make_api_status_error(400, "This model has a context window of 8192 tokens")
        with _patch_openai_create(side_effect=error):
            with pytest.raises(TokenLimitError):
                await call_model("test-endpoint", "hello")

    @pytest.mark.asyncio
    async def test_raises_token_limit_error_on_max_tokens_phrase(self):
        error = _make_api_status_error(400, "Input exceeds max tokens allowed")
        with _patch_openai_create(side_effect=error):
            with pytest.raises(TokenLimitError):
                await call_model("test-endpoint", "hello")

    @pytest.mark.asyncio
    async def test_token_limit_error_message_is_user_friendly(self):
        error = _make_api_status_error(400, "context_length_exceeded")
        with _patch_openai_create(side_effect=error):
            with pytest.raises(TokenLimitError) as exc_info:
                await call_model("test-endpoint", "hello")
        assert "context window" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_token_limit_is_eval_abort_error(self):
        """TokenLimitError must be catchable as EvalAbortError for fast-fail to work."""
        error = _make_api_status_error(400, "context_length_exceeded")
        with _patch_openai_create(side_effect=error):
            with pytest.raises(EvalAbortError):
                await call_model("test-endpoint", "hello")

    @pytest.mark.asyncio
    async def test_raises_token_limit_error_on_timeout(self):
        """APITimeoutError should map to TokenLimitError."""
        mock_request = MagicMock()
        mock_request.url = "https://test.databricks.com/serving-endpoints/test-endpoint/invocations"
        error = APITimeoutError(request=mock_request)
        with _patch_openai_create(side_effect=error):
            with pytest.raises(TokenLimitError):
                await call_model("test-endpoint", "hello")


# ---------------------------------------------------------------------------
# Generic errors (should NOT be EvalAbortError)
# ---------------------------------------------------------------------------

class TestGenericErrors:

    @pytest.mark.asyncio
    async def test_generic_500_raises_plain_exception(self):
        error = _make_api_status_error(500, "Internal server error")
        with _patch_openai_create(side_effect=error):
            with pytest.raises(Exception) as exc_info:
                await call_model("test-endpoint", "hello")
        assert not isinstance(exc_info.value, EvalAbortError)

    @pytest.mark.asyncio
    async def test_generic_400_raises_plain_exception(self):
        error = _make_api_status_error(400, "Invalid request format")
        with _patch_openai_create(side_effect=error):
            with pytest.raises(Exception) as exc_info:
                await call_model("test-endpoint", "hello")
        assert not isinstance(exc_info.value, EvalAbortError)


# ---------------------------------------------------------------------------
# Successful response
# ---------------------------------------------------------------------------

class TestSuccessfulResponse:

    @pytest.mark.asyncio
    async def test_returns_content_model_usage_on_200(self):
        response = _make_success_response(content="test response", model="databricks-claude")
        with _patch_openai_create(return_value=response):
            result = await call_model("test-endpoint", "prompt")
        assert result["content"] == "test response"
        assert result["model"] == "databricks-claude"
        assert "total_tokens" in result["usage"]
