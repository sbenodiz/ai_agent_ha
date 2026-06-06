"""Regression tests for issue #80: Anthropic API error 400 on all queries.

Covers the four fixes:
  1. AnthropicClient surfaces the API error body and raises NonRetryableAIError
     for deterministic 4xx errors (429/408 stay retryable).
  2. _get_ai_response does not burn retries on non-retryable errors.
  3. Oversized data responses are truncated before entering the conversation
     so they cannot blow the model context window.
  4. A failed query is rolled back from conversation history so it cannot
     poison subsequent queries.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add the parent directory to the path for direct imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _mock_aiohttp_session(response_text, status=200, headers=None):
    """Patch aiohttp.ClientSession to return a fixed HTTP body/status."""

    class _FakeResp:
        def __init__(self):
            self.status = status
            self.headers = headers or {}

        async def text(self):
            return response_text

        async def json(self):
            return json.loads(response_text)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        def post(self, *args, **kwargs):
            return _FakeResp()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    return patch(
        "custom_components.ai_agent_ha.agent.aiohttp.ClientSession", _FakeSession
    )


ERROR_400_BODY = json.dumps(
    {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "prompt is too long: 205547 tokens > 200000 maximum",
        },
    }
)


def _make_agent():
    from custom_components.ai_agent_ha.agent import AiAgentHaAgent

    hass = MagicMock()
    hass.data = {"ai_agent_ha": {"configs": {}}}
    config = {
        "ai_provider": "anthropic",
        "anthropic_token": "sk-ant-" + "x" * 40,
        "models": {"anthropic": "claude-sonnet-4-5-20250929"},
    }
    agent = AiAgentHaAgent(hass, config)
    agent._retry_delay = 0  # keep tests fast
    return agent


class TestAnthropicErrorSurfacing:
    """Fix 1: error body surfaced, 4xx non-retryable."""

    @pytest.mark.asyncio
    async def test_400_raises_non_retryable_with_body(self):
        try:
            from custom_components.ai_agent_ha.agent import (
                AnthropicClient,
                NonRetryableAIError,
            )
        except ImportError:
            pytest.skip("AnthropicClient not available")

        client = AnthropicClient("sk-ant-" + "x" * 40)
        with _mock_aiohttp_session(ERROR_400_BODY, status=400):
            with pytest.raises(NonRetryableAIError) as exc_info:
                await client.get_response([{"role": "user", "content": "hi"}])
        assert "prompt is too long" in str(exc_info.value)
        assert "400" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_429_stays_retryable_and_carries_retry_after(self):
        try:
            from custom_components.ai_agent_ha.agent import (
                AnthropicClient,
                NonRetryableAIError,
                RateLimitedAIError,
            )
        except ImportError:
            pytest.skip("AnthropicClient not available")

        client = AnthropicClient("sk-ant-" + "x" * 40)
        body = json.dumps(
            {"type": "error", "error": {"type": "rate_limit_error", "message": "rl"}}
        )
        with _mock_aiohttp_session(body, status=429, headers={"retry-after": "23"}):
            with pytest.raises(RateLimitedAIError) as exc_info:
                await client.get_response([{"role": "user", "content": "hi"}])
        assert not isinstance(exc_info.value, NonRetryableAIError)
        assert exc_info.value.retry_after == 23.0

    @pytest.mark.asyncio
    async def test_retry_loop_waits_out_rate_limit_window(self):
        try:
            from custom_components.ai_agent_ha.agent import RateLimitedAIError
        except ImportError:
            pytest.skip("agent module not available")

        agent = _make_agent()
        sleeps = []
        calls = {"n": 0}

        class FakeClient:
            async def get_response(self, messages, **kwargs):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RateLimitedAIError("429: rate limited", retry_after=17)
                return json.dumps({"request_type": "final_response", "response": "ok"})

        agent.ai_client = FakeClient()
        agent.conversation_history = [
            agent.system_prompt,
            {"role": "user", "content": "hi"},
        ]

        import asyncio as _asyncio

        real_sleep = _asyncio.sleep

        async def fake_sleep(t):
            sleeps.append(t)
            await real_sleep(0)

        with patch("custom_components.ai_agent_ha.agent.asyncio.sleep", fake_sleep):
            response = await agent._get_ai_response()
        assert json.loads(response)["response"] == "ok"
        assert sleeps and sleeps[0] == 17, f"retry-after not honored: {sleeps}"

    @pytest.mark.asyncio
    async def test_500_stays_retryable(self):
        try:
            from custom_components.ai_agent_ha.agent import (
                AnthropicClient,
                NonRetryableAIError,
            )
        except ImportError:
            pytest.skip("AnthropicClient not available")

        client = AnthropicClient("sk-ant-" + "x" * 40)
        with _mock_aiohttp_session("upstream blew up", status=500):
            with pytest.raises(Exception) as exc_info:
                await client.get_response([{"role": "user", "content": "hi"}])
        assert not isinstance(exc_info.value, NonRetryableAIError)
        assert "upstream blew up" in str(exc_info.value)


class TestNoRetryOnNonRetryable:
    """Fix 2: _get_ai_response fails fast instead of 10 retries."""

    @pytest.mark.asyncio
    async def test_get_ai_response_does_not_retry(self):
        try:
            from custom_components.ai_agent_ha.agent import NonRetryableAIError
        except ImportError:
            pytest.skip("agent module not available")

        agent = _make_agent()
        calls = {"n": 0}

        class FakeClient:
            async def get_response(self, messages, **kwargs):
                calls["n"] += 1
                raise NonRetryableAIError("Anthropic API error 400: prompt is too long")

        agent.ai_client = FakeClient()
        agent.conversation_history = [
            agent.system_prompt,
            {"role": "user", "content": "hi"},
        ]
        with pytest.raises(NonRetryableAIError):
            await agent._get_ai_response()
        assert calls["n"] == 1, "non-retryable error must not be retried"

    @pytest.mark.asyncio
    async def test_retryable_errors_still_retried(self):
        try:
            from custom_components.ai_agent_ha.agent import AiAgentHaAgent  # noqa: F401
        except ImportError:
            pytest.skip("agent module not available")

        agent = _make_agent()
        calls = {"n": 0}

        class FakeClient:
            async def get_response(self, messages, **kwargs):
                calls["n"] += 1
                raise Exception("Anthropic API error 500: transient")

        agent.ai_client = FakeClient()
        agent.conversation_history = [
            agent.system_prompt,
            {"role": "user", "content": "hi"},
        ]
        with pytest.raises(Exception, match="Failed after"):
            await agent._get_ai_response()
        assert calls["n"] == agent._max_retries


class TestDataMessageTruncation:
    """Fix 3: oversized data responses are capped."""

    def test_small_data_passthrough(self):
        try:
            from custom_components.ai_agent_ha.agent import AiAgentHaAgent  # noqa: F401
        except ImportError:
            pytest.skip("agent module not available")

        agent = _make_agent()
        data = [{"entity_id": "sensor.a", "state": "1"}]
        msg = agent._format_data_message(data)
        assert json.loads(msg) == {"data": data}

    def test_large_list_truncated_under_cap(self):
        try:
            from custom_components.ai_agent_ha.agent import AiAgentHaAgent  # noqa: F401
        except ImportError:
            pytest.skip("agent module not available")

        agent = _make_agent()
        # ~2600 entities x ~400 chars each, like the live repro of issue #80
        data = [
            {
                "entity_id": f"sensor.device_{i}_power",
                "state": "42.0",
                "attributes": {"friendly_name": f"Device {i} Power", "x": "y" * 300},
            }
            for i in range(2600)
        ]
        raw_len = len(json.dumps({"data": data}))
        assert raw_len > agent.MAX_DATA_MESSAGE_CHARS  # sanity: input is oversized

        msg = agent._format_data_message(data)
        assert len(msg) <= agent.MAX_DATA_MESSAGE_CHARS
        parsed = json.loads(msg)
        assert parsed["truncated"] is True
        assert parsed["total_items"] == 2600
        assert parsed["items_shown"] == len(parsed["data"])
        assert 0 < parsed["items_shown"] < 2600
        # order preserved: first items survive
        assert parsed["data"][0]["entity_id"] == "sensor.device_0_power"
        assert "note" in parsed

    def test_large_dict_truncated_under_cap(self):
        try:
            from custom_components.ai_agent_ha.agent import AiAgentHaAgent  # noqa: F401
        except ImportError:
            pytest.skip("agent module not available")

        agent = _make_agent()
        data = {f"area_{i}": {"name": "x" * 500} for i in range(1000)}
        msg = agent._format_data_message(data)
        assert len(msg) <= agent.MAX_DATA_MESSAGE_CHARS
        parsed = json.loads(msg)
        assert parsed["truncated"] is True


class TestHistoryRollbackOnFailure:
    """Fix 4: failed queries don't poison the conversation history."""

    @pytest.mark.asyncio
    async def test_failed_query_rolled_back_and_next_query_succeeds(self):
        try:
            from custom_components.ai_agent_ha.agent import (
                AnthropicClient,
                NonRetryableAIError,
            )
        except ImportError:
            pytest.skip("agent module not available")

        agent = _make_agent()
        behavior = {"fail": True}

        async def fake_get_response(self, messages, **kwargs):
            if behavior["fail"]:
                raise NonRetryableAIError("Anthropic API error 400: prompt is too long")
            return json.dumps(
                {"request_type": "final_response", "response": "All good"}
            )

        with patch.object(AnthropicClient, "get_response", fake_get_response):
            result1 = await agent.process_query("query that fails")
            assert result1["success"] is False
            assert "prompt is too long" in result1["error"]
            # rollback: only the system prompt remains
            roles = [m["role"] for m in agent.conversation_history]
            assert roles == ["system"], f"history not rolled back: {roles}"

            behavior["fail"] = False
            result2 = await agent.process_query("query that succeeds")
            assert result2["success"] is True
            assert result2["answer"] == "All good"


class TestWindowAlignment:
    """History window never starts with an assistant message after slicing."""

    @pytest.mark.asyncio
    async def test_window_starts_with_user_turn(self):
        try:
            from custom_components.ai_agent_ha.agent import AiAgentHaAgent  # noqa: F401
        except ImportError:
            pytest.skip("agent module not available")

        agent = _make_agent()
        captured = {}

        class FakeClient:
            async def get_response(self, messages, **kwargs):
                captured["messages"] = messages
                return json.dumps({"request_type": "final_response", "response": "ok"})

        agent.ai_client = FakeClient()
        # 12 messages so [-10:] slices mid-turn and would start with assistant
        history = [agent.system_prompt]
        for i in range(5):
            history.append({"role": "user", "content": f"u{i}"})
            history.append({"role": "assistant", "content": f"a{i}"})
        history.append({"role": "user", "content": "final"})
        agent.conversation_history = history
        assert history[-10:][0]["role"] == "assistant"  # sanity: slice is misaligned

        await agent._get_ai_response()
        sent = captured["messages"]
        assert sent[0]["role"] == "system"
        assert (
            sent[1]["role"] == "user"
        ), f"window starts mid-turn: {[m['role'] for m in sent]}"

    @pytest.mark.asyncio
    async def test_window_total_size_capped(self):
        try:
            from custom_components.ai_agent_ha.agent import AiAgentHaAgent  # noqa: F401
        except ImportError:
            pytest.skip("agent module not available")

        agent = _make_agent()
        captured = {}

        class FakeClient:
            async def get_response(self, messages, **kwargs):
                captured["messages"] = messages
                return json.dumps({"request_type": "final_response", "response": "ok"})

        agent.ai_client = FakeClient()
        # several at-cap data messages stack past the window budget
        history = [agent.system_prompt]
        for i in range(4):
            history.append({"role": "assistant", "content": f"req{i}"})
            history.append(
                {"role": "user", "content": json.dumps({"data": "x" * 49_000, "i": i})}
            )
        agent.conversation_history = history

        await agent._get_ai_response()
        sent = captured["messages"]
        non_system_chars = sum(len(m["content"]) for m in sent if m["role"] != "system")
        assert non_system_chars <= agent.MAX_WINDOW_CHARS
        # the most recent data message must survive
        assert any('"i": 3' in m["content"] for m in sent)


class TestMaxIterationsRollback:
    """Max-iterations failure also rolls history back."""

    @pytest.mark.asyncio
    async def test_max_iterations_rolls_back_history(self):
        try:
            from custom_components.ai_agent_ha.agent import AnthropicClient
        except ImportError:
            pytest.skip("agent module not available")

        agent = _make_agent()

        async def always_data_request(self, messages, **kwargs):
            # model never produces a final answer
            return json.dumps(
                {
                    "request_type": "data_request",
                    "request": "get_automations",
                    "parameters": {},
                }
            )

        async def fake_get_automations():
            return [{"id": "a1", "alias": "Test"}]

        agent.get_automations = fake_get_automations
        with patch.object(AnthropicClient, "get_response", always_data_request):
            result = await agent.process_query("never finishes")
        assert result["success"] is False
        assert "Maximum iterations" in result["error"]
        roles = [m["role"] for m in agent.conversation_history]
        assert roles == ["system"], f"history not rolled back: {roles}"
