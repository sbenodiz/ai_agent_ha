"""Tests for AskSageClient retry logic on overload / rate-limit responses."""

import asyncio
import json
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------- Mock homeassistant before importing agent ----------
_HA_MODULES = [
    "homeassistant",
    "homeassistant.core",
    "homeassistant.components",
    "homeassistant.components.frontend",
    "homeassistant.components.http",
    "homeassistant.config_entries",
    "homeassistant.exceptions",
    "homeassistant.helpers",
    "homeassistant.helpers.area_registry",
    "homeassistant.helpers.device_registry",
    "homeassistant.helpers.entity_registry",
    "homeassistant.helpers.aiohttp_client",
    "homeassistant.helpers.storage",
    "homeassistant.helpers.config_validation",
    "homeassistant.helpers.typing",
    "homeassistant.util",
    "homeassistant.util.dt",
]
for _mod in _HA_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Also mock voluptuous if not installed
try:
    import voluptuous  # noqa: F401
except ImportError:
    sys.modules["voluptuous"] = MagicMock()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from custom_components.ai_agent_ha.agent import (  # noqa: E402
    AskSageClient,
    _is_overload_response,
)


# ---------- Helpers ----------

def _make_mock_response(*, status=200, json_data=None, text=""):
    """Create a mock aiohttp response."""
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data or {})
    resp.text = AsyncMock(return_value=text)
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


def _make_mock_session(responses):
    """Create a mock session whose .post() yields *responses* in order.

    Each call to ``_session()`` returns the same context manager that
    round-robins through *responses* on successive ``.post()`` calls.
    """
    call_count = 0

    class _PostCtx:
        def __init__(self, resp):
            self._resp = resp

        async def __aenter__(self):
            return self._resp

        async def __aexit__(self, *_):
            pass

    class _SessionCtx:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        def post(self, *args, **kwargs):
            nonlocal call_count
            resp = responses[call_count]
            call_count += 1
            return _PostCtx(resp)

    return _SessionCtx()


# ---------- Unit tests for _is_overload_response ----------

class TestIsOverloadResponse:
    def test_detects_overloaded(self):
        assert _is_overload_response("Sorry, the model is overloaded, please try again in a few seconds.")

    def test_detects_try_again(self):
        assert _is_overload_response("Please try again later.")

    def test_detects_rate_limit(self):
        assert _is_overload_response("Rate limit exceeded.")

    def test_detects_too_many_requests(self):
        assert _is_overload_response("Too many requests, slow down.")

    def test_normal_response(self):
        assert not _is_overload_response("The temperature in the living room is 72°F.")


# ---------- Retry integration tests ----------

class TestAskSageRetry:
    @pytest.mark.asyncio
    async def test_retry_on_overload_then_success(self):
        """First call returns overload message, second call returns real answer."""
        overload_resp = _make_mock_response(
            status=200,
            json_data={"message": "Sorry, the model is overloaded, please try again in a few seconds."},
        )
        good_resp = _make_mock_response(
            status=200,
            json_data={"message": "The temperature is 72°F."},
        )

        mock_session = _make_mock_session([overload_resp, good_resp])
        client = AskSageClient(token="test-token", model="gpt-4o-mini")

        with patch.object(client, "_session", return_value=mock_session), \
             patch("custom_components.ai_agent_ha.agent.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await client.get_response([{"role": "user", "content": "What is the temperature?"}])

        assert "72°F" in result
        assert "overloaded" not in result.lower()
        mock_sleep.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_retry_exhausted_returns_friendly_message(self):
        """All 3 calls return overload; result contains 'temporarily overloaded'."""
        overload_msg = "Sorry, the model is overloaded, please try again in a few seconds."
        responses = [
            _make_mock_response(status=200, json_data={"message": overload_msg})
            for _ in range(3)
        ]

        mock_session = _make_mock_session(responses)
        client = AskSageClient(token="test-token", model="gpt-4o-mini")

        with patch.object(client, "_session", return_value=mock_session), \
             patch("custom_components.ai_agent_ha.agent.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await client.get_response([{"role": "user", "content": "test"}])

        parsed = json.loads(result)
        assert "temporarily overloaded" in parsed["response"]
        # Sleeps between attempts 1→2 and 2→3 (not after the last attempt)
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_http_429_retries(self):
        """First call returns HTTP 429, second returns 200 with valid response."""
        resp_429 = _make_mock_response(status=429, text="rate limited")
        resp_ok = _make_mock_response(
            status=200,
            json_data={"message": "All good now."},
        )

        mock_session = _make_mock_session([resp_429, resp_ok])
        client = AskSageClient(token="test-token", model="gpt-4o-mini")

        with patch.object(client, "_session", return_value=mock_session), \
             patch("custom_components.ai_agent_ha.agent.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await client.get_response([{"role": "user", "content": "test"}])

        assert "All good now" in result
        mock_sleep.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_http_503_retries(self):
        """First call returns HTTP 503, second returns 200."""
        resp_503 = _make_mock_response(status=503, text="service unavailable")
        resp_ok = _make_mock_response(
            status=200,
            json_data={"message": "Recovered."},
        )

        mock_session = _make_mock_session([resp_503, resp_ok])
        client = AskSageClient(token="test-token", model="gpt-4o-mini")

        with patch.object(client, "_session", return_value=mock_session), \
             patch("custom_components.ai_agent_ha.agent.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await client.get_response([{"role": "user", "content": "test"}])

        assert "Recovered" in result
        mock_sleep.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_http_429_exhausted_raises(self):
        """All 3 calls return HTTP 429 — raises after retries exhausted."""
        responses = [
            _make_mock_response(status=429, text="rate limited")
            for _ in range(3)
        ]

        mock_session = _make_mock_session(responses)
        client = AskSageClient(token="test-token", model="gpt-4o-mini")

        with patch.object(client, "_session", return_value=mock_session), \
             patch("custom_components.ai_agent_ha.agent.asyncio.sleep", new_callable=AsyncMock), \
             pytest.raises(Exception, match="service unavailable after 3 attempts"):
            await client.get_response([{"role": "user", "content": "test"}])
