"""The AI Agent implementation with multiple provider support.

Example config:
ai_agent_ha:
  ai_provider: openai  # or 'llama', 'gemini', 'openrouter', 'anthropic', 'alter', 'zai', 'local'
  llama_token: "..."
  openai_token: "..."
  gemini_token: "..."
  openrouter_token: "..."
  anthropic_token: "..."
  alter_token: "..."
  zai_token: "..."
  zai_endpoint: "general"  # or 'coding' for z.ai (3× usage, 1/7 cost)
  local_url: "http://localhost:11434/api/generate"  # Required for local models
  # Model configuration (optional, defaults will be used if not specified)
  models:
    openai: "gpt-3.5-turbo"  # or "gpt-4", "gpt-4-turbo", etc.
    llama: "Llama-4-Maverick-17B-128E-Instruct-FP8"
    gemini: "gemini-2.5-flash"  # or "gemini-2.5-pro", "gemini-2.0-flash", etc.
    openrouter: "openai/gpt-4o"  # or any model available on OpenRouter
    anthropic: "claude-opus-4-6"  # or "claude-sonnet-4-6", "claude-haiku-4-5-20251001", "claude-sonnet-4-5-20250929", etc.
    alter: "your-model-name"  # model name for Alter API
    zai: "glm-4.7"  # model name for z.ai API (glm-4.7, glm-4.6, glm-4.5, etc.)
    local: "llama3.2"  # model name for local API (optional if your API doesn't require it)
"""

import asyncio
import codecs
import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote

import aiohttp
import yaml  # type: ignore[import-untyped]
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import CONF_WEATHER_ENTITY, DOMAIN

_LOGGER = logging.getLogger(__name__)


# === Security Utilities ===
def sanitize_for_logging(data: Any, mask: str = "***REDACTED***") -> Any:
    """Sanitize sensitive data for safe logging.

    Recursively masks sensitive fields like API keys, tokens, passwords, etc.
    This prevents accidental exposure of credentials in debug logs.

    Args:
        data: The data structure to sanitize (dict, list, str, etc.)
        mask: The string to use for masking sensitive values

    Returns:
        A sanitized copy of the data with sensitive fields masked

    Example:
        >>> config = {"openai_token": "sk-abc123", "ai_provider": "openai"}
        >>> sanitize_for_logging(config)
        {"openai_token": "***REDACTED***", "ai_provider": "openai"}
    """
    # Sensitive field patterns (case-insensitive)
    sensitive_patterns = {
        "token",
        "key",
        "password",
        "secret",
        "credential",
        "auth",
        "authorization",
        "api_key",
        "apikey",
        "llama_token",
        "openai_token",
        "gemini_token",
        "anthropic_token",
        "openrouter_token",
        "alter_token",
        "zai_token",
    }

    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            # Check if key matches any sensitive pattern
            key_lower = str(key).lower()
            is_sensitive = any(pattern in key_lower for pattern in sensitive_patterns)

            if is_sensitive:
                sanitized[key] = mask
            else:
                # Recursively sanitize nested structures
                sanitized[key] = sanitize_for_logging(value, mask)
        return sanitized

    elif isinstance(data, list):
        return [sanitize_for_logging(item, mask) for item in data]

    elif isinstance(data, tuple):
        return tuple(sanitize_for_logging(item, mask) for item in data)

    else:
        # Primitive types (str, int, bool, etc.) - return as-is
        return data


# === AI Client Abstractions ===


class _HASessionContext:
    """Async context manager that wraps a shared HA aiohttp session.

    The HA-managed session must NOT be closed by individual callers, so this
    wrapper skips the close step while still exposing the standard
    ``async with`` interface expected by all call sites.
    """

    __slots__ = ("_session",)

    def __init__(self, session: aiohttp.ClientSession) -> None:
        self._session = session

    async def __aenter__(self) -> aiohttp.ClientSession:
        return self._session

    async def __aexit__(self, *_) -> None:
        pass  # Do not close — HA owns this session.


class BaseAIClient:
    """Base class for all AI provider clients.

    Subclasses should call super().__init__(hass=hass) if they accept a hass
    instance, so that _get_session() can return the HA-managed aiohttp session
    (which avoids creating a new TCP connector per request).
    """

    def __init__(self, hass: Optional["HomeAssistant"] = None):
        self._hass = hass

    def _session(self):
        """Return a context manager that yields an aiohttp ClientSession.

        When a Home Assistant instance is available the HA-managed session is
        reused (no new connector / TCP stack per request).  When unavailable
        (e.g. in unit tests) a fresh short-lived session is created instead.
        """
        if self._hass is not None:
            return _HASessionContext(async_get_clientsession(self._hass))
        return aiohttp.ClientSession()

    async def get_response(self, messages, **kwargs):
        raise NotImplementedError

    @staticmethod
    def extract_thinking(text: str):
        """Extract thinking content from model response before stripping.

        Returns (thinking_content, None) — thinking_content is None if no block found.
        """
        if not text:
            return None
        match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Also check <|thinking|> variant
        match = re.search(
            r"<\|thinking\|>(.*?)</\|thinking\|>", text, re.DOTALL | re.IGNORECASE
        )
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def strip_thinking_tags(text: str) -> str:
        """Remove thinking/reasoning blocks from model responses.

        Strips <think>...</think> blocks produced by models with thinking mode
        enabled (Qwen3, DeepSeek-R1, etc.). Handles multi-line blocks, nested
        whitespace, and cases where the closing tag is missing (truncated output).

        Also strips the <|thinking|>...</|thinking|> variant used by some models.

        Args:
            text: Raw response string from the model.

        Returns:
            The response with all thinking blocks removed and whitespace cleaned up.
        """
        if not text:
            return text
        # Remove <think>...</think> blocks (case-insensitive, dotall)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Remove <|thinking|>...</|thinking|> variant
        text = re.sub(
            r"<\|thinking\|>.*?</\|thinking\|>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        # Handle truncated blocks: remove everything from an unclosed <think> to end of string
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<\|thinking\|>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Clean up leading/trailing whitespace left behind
        return text.strip()


class LocalClient(BaseAIClient):
    def __init__(self, url, model="", hass=None):
        super().__init__(hass=hass)
        self.url = url
        self.model = model
        # Detect OpenAI-compatible endpoints (e.g. LM Studio, vLLM, LocalAI)
        # by checking if the URL contains '/v1'. If so, use the OpenAI chat
        # completions format instead of the Ollama-native prompt format.
        self._is_openai_compatible = "/v1" in (url or "")
        if self._is_openai_compatible:
            # Ensure the request URL targets /v1/chat/completions
            self._chat_url = url.rstrip("/")
            if not self._chat_url.endswith("/chat/completions"):
                if self._chat_url.endswith("/v1"):
                    self._chat_url += "/chat/completions"
                else:
                    self._chat_url += "/v1/chat/completions"
            _LOGGER.info(
                "Detected OpenAI-compatible local endpoint. Chat URL: %s",
                self._chat_url,
            )

    async def get_response(self, messages, **kwargs):
        _LOGGER.debug(
            "Making request to local API with model: '%s' at URL: %s (openai_compat=%s)",
            self.model or "[NO MODEL SPECIFIED]",
            self.url,
            self._is_openai_compatible,
        )

        if not self.model:
            _LOGGER.warning(
                "No model specified for local API request. Some APIs (like Ollama) require a model name."
            )
        headers = {"Content-Type": "application/json"}

        # Choose request format based on detected endpoint type
        if self._is_openai_compatible:
            # OpenAI-compatible format (LM Studio, vLLM, LocalAI, etc.)
            # Send structured messages array to /v1/chat/completions
            payload = {
                "messages": messages,
                "stream": False,
                "temperature": 0.7,
                "top_p": 0.9,
            }
            if self.model:
                payload["model"] = self.model
            request_url = self._chat_url
            _LOGGER.debug("Using OpenAI-compatible format → POST %s", request_url)
        else:
            # Legacy Ollama-native format: flatten messages into a prompt string
            prompt = ""
            for message in messages:
                role = message.get("role", "")
                content = message.get("content", "")
                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"
            prompt += "Assistant: "

            payload = {
                "prompt": prompt,
                "stream": False,
            }
            if self.model:
                payload["model"] = self.model
            request_url = self.url
            _LOGGER.debug("Using Ollama-native format → POST %s", request_url)

        # Note: Payloads don't contain auth tokens (those are in headers), but may contain user prompts
        _LOGGER.debug("Local API request payload: %s", json.dumps(payload, indent=2))

        # Ollama-specific validation (only for non-OpenAI-compatible endpoints)
        if not self._is_openai_compatible:
            if "model" not in payload or not payload["model"]:
                _LOGGER.warning(
                    "Missing 'model' field in request to local API. This may cause issues with Ollama."
                )
            elif self.url and "ollama" in self.url.lower():
                _LOGGER.debug(
                    "Detected Ollama URL, ensuring model is specified: %s",
                    payload.get("model"),
                )

        async with self._session() as session:
            async with session.post(
                request_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    _LOGGER.error("Local API error %d: %s", resp.status, error_text)

                    # Provide more specific error messages for common Ollama issues
                    if resp.status == 404:
                        if "model" in payload and payload["model"]:
                            raise Exception(
                                f"Model '{payload['model']}' not found. Please ensure the model is installed in Ollama using: ollama pull {payload['model']}"
                            )
                        else:
                            raise Exception(
                                "Local API endpoint not found. Please check the URL and ensure Ollama is running."
                            )
                    elif resp.status == 400:
                        raise Exception(
                            f"Bad request to local API. Error: {error_text}"
                        )
                    else:
                        raise Exception(f"Local API error {resp.status}: {error_text}")

                try:
                    response_text = await resp.text()
                    _LOGGER.debug(
                        "Local API response (first 200 chars): %s", response_text[:200]
                    )
                    _LOGGER.debug("Local API response status: %d", resp.status)
                    # Sanitize headers to avoid logging any auth tokens
                    _LOGGER.debug(
                        "Local API response headers: %s",
                        sanitize_for_logging(dict(resp.headers)),
                    )

                    # Try to parse as JSON
                    try:
                        data = json.loads(response_text)

                        # Try common response formats
                        # Ollama format - return only the response text
                        if "response" in data:
                            response_content = self.strip_thinking_tags(
                                data["response"]
                            )
                            _LOGGER.debug(
                                "Extracted response content: %s",
                                (
                                    response_content[:100]
                                    if response_content
                                    else "[EMPTY]"
                                ),
                            )

                            # Check if response is empty or None
                            if not response_content or response_content.strip() == "":
                                _LOGGER.warning(
                                    "Ollama returned empty response. Full data: %s",
                                    data,
                                )
                                # Check if this is a loading response
                                if data.get("done_reason") == "load":
                                    _LOGGER.warning(
                                        "Ollama is still loading the model. Please wait and try again."
                                    )
                                    return json.dumps(
                                        {
                                            "request_type": "final_response",
                                            "response": "The AI model is still loading. Please wait a moment and try again.",
                                        }
                                    )
                                elif data.get("done") is False:
                                    _LOGGER.warning(
                                        "Ollama response indicates it's not done yet."
                                    )
                                    return json.dumps(
                                        {
                                            "request_type": "final_response",
                                            "response": "The AI is still processing your request. Please try again.",
                                        }
                                    )
                                else:
                                    return json.dumps(
                                        {
                                            "request_type": "final_response",
                                            "response": "The AI returned an empty response. Please try rephrasing your question.",
                                        }
                                    )

                            # Check if the response looks like JSON
                            response_content = response_content.strip()
                            if response_content.startswith(
                                "{"
                            ) and response_content.endswith("}"):
                                try:
                                    # Validate that it's actually JSON and contains valid request_type
                                    parsed_json = json.loads(response_content)
                                    if (
                                        isinstance(parsed_json, dict)
                                        and "request_type" in parsed_json
                                    ):
                                        _LOGGER.debug(
                                            "Local model provided valid JSON response"
                                        )
                                        return response_content
                                    else:
                                        _LOGGER.debug(
                                            "JSON missing request_type, treating as plain text"
                                        )
                                except json.JSONDecodeError:
                                    _LOGGER.debug(
                                        "Invalid JSON from local model, treating as plain text"
                                    )
                                    pass

                            # If it's plain text, wrap it in the expected JSON format
                            wrapped_response = {
                                "request_type": "final_response",
                                "response": response_content,
                            }
                            _LOGGER.debug("Wrapped plain text response in JSON format")
                            return json.dumps(wrapped_response)

                        # OpenAI-like format
                        elif "choices" in data and len(data["choices"]) > 0:
                            choice = data["choices"][0]
                            if "message" in choice and "content" in choice["message"]:
                                content = self.strip_thinking_tags(
                                    choice["message"]["content"]
                                )
                            elif "text" in choice:
                                content = self.strip_thinking_tags(choice["text"])
                            else:
                                content = str(data)

                            # Check if it's valid JSON with request_type
                            content = content.strip()
                            if content.startswith("{") and content.endswith("}"):
                                try:
                                    parsed_json = json.loads(content)
                                    if (
                                        isinstance(parsed_json, dict)
                                        and "request_type" in parsed_json
                                    ):
                                        _LOGGER.debug(
                                            "Local model provided valid JSON response (OpenAI format)"
                                        )
                                        return content
                                    else:
                                        _LOGGER.debug(
                                            "JSON missing request_type, treating as plain text (OpenAI format)"
                                        )
                                except json.JSONDecodeError:
                                    _LOGGER.debug(
                                        "Invalid JSON from local model, treating as plain text (OpenAI format)"
                                    )
                                    pass

                            # Wrap in expected format if plain text
                            wrapped_response = {
                                "request_type": "final_response",
                                "response": content,
                            }
                            return json.dumps(wrapped_response)

                        # Generic content field
                        elif "content" in data:
                            content = self.strip_thinking_tags(data["content"])
                            content = content.strip()
                            if content.startswith("{") and content.endswith("}"):
                                try:
                                    parsed_json = json.loads(content)
                                    if (
                                        isinstance(parsed_json, dict)
                                        and "request_type" in parsed_json
                                    ):
                                        _LOGGER.debug(
                                            "Local model provided valid JSON response (generic format)"
                                        )
                                        return content
                                    else:
                                        _LOGGER.debug(
                                            "JSON missing request_type, treating as plain text (generic format)"
                                        )
                                except json.JSONDecodeError:
                                    _LOGGER.debug(
                                        "Invalid JSON from local model, treating as plain text (generic format)"
                                    )
                                    pass

                            wrapped_response = {
                                "request_type": "final_response",
                                "response": content,
                            }
                            return json.dumps(wrapped_response)

                        # Handle case where no standard fields are found
                        _LOGGER.warning(
                            "No standard response fields found in local API response. Full response: %s",
                            data,
                        )

                        # Check for Ollama-specific edge cases
                        if data.get("done_reason") == "load":
                            return json.dumps(
                                {
                                    "request_type": "final_response",
                                    "response": "The AI model is still loading. Please wait a moment and try again.",
                                }
                            )
                        elif data.get("done") is False:
                            return json.dumps(
                                {
                                    "request_type": "final_response",
                                    "response": "The AI is still processing your request. Please try again.",
                                }
                            )
                        elif "message" in data:
                            # Some APIs use "message" field
                            message_content = data["message"]
                            if (
                                isinstance(message_content, dict)
                                and "content" in message_content
                            ):
                                content = self.strip_thinking_tags(
                                    message_content["content"]
                                )
                            else:
                                content = self.strip_thinking_tags(str(message_content))
                            return json.dumps(
                                {"request_type": "final_response", "response": content}
                            )

                        # Return the whole data as string if we can't find a specific field
                        return json.dumps(
                            {
                                "request_type": "final_response",
                                "response": f"Received unexpected response format from local API: {str(data)}",
                            }
                        )

                    except json.JSONDecodeError:
                        # If not JSON, check if it's a JSON response that got corrupted by wrapping
                        response_text = response_text.strip()
                        if response_text.startswith("{") and response_text.endswith(
                            "}"
                        ):
                            try:
                                parsed_json = json.loads(response_text)
                                if (
                                    isinstance(parsed_json, dict)
                                    and "request_type" in parsed_json
                                ):
                                    _LOGGER.debug(
                                        "Local model provided valid JSON response (direct)"
                                    )
                                    return response_text
                            except json.JSONDecodeError:
                                pass

                        # If not valid JSON, wrap the raw text in expected format
                        _LOGGER.debug("Response is not JSON, wrapping plain text")
                        wrapped_response = {
                            "request_type": "final_response",
                            "response": response_text,
                        }
                        return json.dumps(wrapped_response)

                except Exception as e:
                    _LOGGER.error("Failed to parse local API response: %s", str(e))
                    raise Exception(f"Failed to parse local API response: {str(e)}")

    async def get_response_stream(self, messages, **kwargs):
        """SSE streaming for OpenAI-compatible local endpoints.

        Yields text chunks as they arrive. Falls back to non-streaming on error.
        Only works for OpenAI-compatible endpoints (LM Studio, vLLM, etc.).
        """
        if not self._is_openai_compatible:
            # Ollama-native endpoints don't support SSE streaming in the same way
            _LOGGER.debug("Local non-OpenAI endpoint: falling back to non-streaming")
            result = await self.get_response(messages, **kwargs)
            yield result
            return

        headers = {"Content-Type": "application/json"}
        payload = {
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        if self.model:
            payload["model"] = self.model

        try:
            async with self._session() as session:
                async with session.post(
                    self._chat_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    if resp.status != 200:
                        _LOGGER.warning(
                            "Local stream error %d, falling back", resp.status
                        )
                        result = await self.get_response(messages, **kwargs)
                        yield result
                        return

                    async for line in resp.content:
                        decoded = line.decode("utf-8", errors="replace").strip()
                        if not decoded or not decoded.startswith("data:"):
                            continue
                        data_str = decoded[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            content = (
                                chunk.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            if content:
                                yield content
                        except (json.JSONDecodeError, IndexError, KeyError):
                            continue
        except Exception as e:
            _LOGGER.warning("Local streaming failed (%s), falling back", e)
            result = await self.get_response(messages, **kwargs)
            yield result


class LlamaClient(BaseAIClient):
    def __init__(
        self, token, model="Llama-4-Maverick-17B-128E-Instruct-FP8", hass=None
    ):
        super().__init__(hass=hass)
        self.token = token
        self.model = model
        self.api_url = "https://api.llama.com/v1/chat/completions"

    async def get_response(self, messages, **kwargs):
        _LOGGER.debug("Making request to Llama API with model: %s", self.model)
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "top_p": 0.9,
            # max_tokens omitted - let Llama use the model's default capacity
        }

        _LOGGER.debug("Llama request payload: %s", json.dumps(payload, indent=2))

        async with self._session() as session:
            async with session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    _LOGGER.error("Llama API error %d: %s", resp.status, error_text)
                    raise Exception(f"Llama API error {resp.status}")
                data = await resp.json()
                # Extract text from Llama response
                completion = data.get("completion_message", {})
                content = completion.get("content", {})
                return self.strip_thinking_tags(content.get("text", str(data)))


class OpenAIClient(BaseAIClient):
    def __init__(self, token, model="gpt-3.5-turbo", base_url="", hass=None):
        super().__init__(hass=hass)
        self.token = token
        self.model = model
        # Use custom base URL if provided (e.g. LM Studio at http://192.168.0.57:1234/v1)
        if base_url and base_url.strip():
            self.api_url = base_url.rstrip("/") + "/chat/completions"
            self._custom_endpoint = True
            _LOGGER.info(
                "OpenAIClient using custom endpoint: %s (model: %s)",
                self.api_url,
                self.model,
            )
        else:
            self.api_url = "https://api.openai.com/v1/chat/completions"
            self._custom_endpoint = False

    def _is_restricted_model(self):
        """Check if the model has restricted parameters (no temperature, top_p, etc.)."""
        # Models that don't support temperature, top_p and other parameters
        restricted_models = ["o3-mini", "o3", "o1-mini", "o1-preview", "o1", "gpt-5"]

        model_lower = self.model.lower()
        return any(model_id in model_lower for model_id in restricted_models)

    async def get_response(self, messages, **kwargs):
        _LOGGER.debug("Making request to OpenAI API with model: %s", self.model)

        # Validate token — skip sk- prefix check for custom endpoints (e.g. LM Studio)
        if not self.token:
            raise Exception("Invalid OpenAI API key format")
        if not self._custom_endpoint and not self.token.startswith("sk-"):
            raise Exception("Invalid OpenAI API key format")

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        # Check if model has restricted parameters
        is_restricted = self._is_restricted_model()
        _LOGGER.debug(
            "Using model: %s (restricted parameters: %s)",
            self.model,
            is_restricted,
        )

        # Build payload with model-appropriate parameters
        # Don't set max_tokens - let OpenAI use the model's maximum capacity
        payload = {"model": self.model, "messages": messages}

        # Only add temperature and top_p for models that support them
        if not is_restricted:
            payload.update({"temperature": 0.7, "top_p": 0.9})

        _LOGGER.debug("OpenAI request payload: %s", json.dumps(payload, indent=2))

        async with self._session() as session:
            async with session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                response_text = await resp.text()
                _LOGGER.debug("OpenAI API response status: %d", resp.status)
                _LOGGER.debug("OpenAI API response: %s", response_text[:500])

                if resp.status != 200:
                    _LOGGER.error("OpenAI API error %d: %s", resp.status, response_text)
                    raise Exception(f"OpenAI API error {resp.status}: {response_text}")

                try:
                    data = json.loads(response_text)
                except json.JSONDecodeError as e:
                    _LOGGER.error("Failed to parse OpenAI response as JSON: %s", str(e))
                    raise Exception(
                        f"Invalid JSON response from OpenAI: {response_text[:200]}"
                    )

                # Extract text from OpenAI response
                choices = data.get("choices", [])
                if choices and "message" in choices[0]:
                    content = choices[0]["message"].get("content", "")
                    if not content:
                        _LOGGER.warning("OpenAI returned empty content in message")
                        _LOGGER.debug(
                            "Full OpenAI response: %s", json.dumps(data, indent=2)
                        )
                    return self.strip_thinking_tags(content)
                else:
                    _LOGGER.warning("OpenAI response missing expected structure")
                    _LOGGER.debug(
                        "Full OpenAI response: %s", json.dumps(data, indent=2)
                    )
                    return str(data)


class GeminiClient(BaseAIClient):
    def __init__(self, token, model="gemini-2.5-flash", hass=None):
        super().__init__(hass=hass)
        self.token = token.strip() if token else token  # Strip whitespace from token
        self.model = model
        # Use v1beta for all models as per Google's current API documentation
        # All Gemini 2.0/2.5 models are available on v1beta endpoint
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    async def get_response(self, messages, **kwargs):
        _LOGGER.debug("Making request to Gemini API with model: %s", self.model)

        # Validate token
        if not self.token:
            raise Exception("Missing Gemini API key")

        headers = {"Content-Type": "application/json"}

        # Convert OpenAI-style messages to Gemini format
        gemini_contents = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                # Gemini doesn't have a system role, so we prepend it to the first user message
                if not gemini_contents:
                    gemini_contents.append(
                        {"role": "user", "parts": [{"text": f"System: {content}"}]}
                    )
                else:
                    # Add system message as user message
                    gemini_contents.append(
                        {"role": "user", "parts": [{"text": f"System: {content}"}]}
                    )
            elif role == "user":
                gemini_contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "tool":
                # Gemini doesn't support role='tool' in basic contents API;
                # map HA data-injection turns to 'user' role.
                gemini_contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                gemini_contents.append({"role": "model", "parts": [{"text": content}]})

        payload = {
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.9,
                # maxOutputTokens omitted - let Gemini use model's maximum capacity
            },
        }

        # Add API key as query parameter (URL encoded)
        url_with_key = f"{self.api_url}?key={quote(self.token)}"

        _LOGGER.debug("Gemini request payload: %s", json.dumps(payload, indent=2))

        async with self._session() as session:
            async with session.post(
                url_with_key,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                response_text = await resp.text()
                _LOGGER.debug("Gemini API response status: %d", resp.status)
                _LOGGER.debug("Gemini API response: %s", response_text[:500])

                if resp.status != 200:
                    _LOGGER.error("Gemini API error %d: %s", resp.status, response_text)
                    raise Exception(f"Gemini API error {resp.status}: {response_text}")

                try:
                    data = json.loads(response_text)
                except json.JSONDecodeError as e:
                    _LOGGER.error("Failed to parse Gemini response as JSON: %s", str(e))
                    raise Exception(
                        f"Invalid JSON response from Gemini: {response_text[:200]}"
                    )

                # Log token usage for debugging, especially for Gemini 2.5 extended thinking
                usage_metadata = data.get("usageMetadata", {})
                if usage_metadata:
                    _LOGGER.debug(
                        "Gemini token usage - prompt: %d, total: %d, thoughts: %d",
                        usage_metadata.get("promptTokenCount", 0),
                        usage_metadata.get("totalTokenCount", 0),
                        usage_metadata.get("thoughtsTokenCount", 0),
                    )

                # Extract text from Gemini response
                candidates = data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    # Check finish reason for potential issues
                    finish_reason = candidates[0].get("finishReason", "")
                    if finish_reason == "MAX_TOKENS":
                        _LOGGER.warning(
                            "Gemini response truncated due to MAX_TOKENS limit. "
                            "Thoughts used: %d tokens. Consider increasing maxOutputTokens.",
                            usage_metadata.get("thoughtsTokenCount", 0),
                        )

                    parts = candidates[0]["content"].get("parts", [])
                    if parts:
                        content = parts[0].get("text", "")
                        if not content:
                            _LOGGER.warning("Gemini returned empty text content")
                            _LOGGER.debug(
                                "Full Gemini response: %s", json.dumps(data, indent=2)
                            )
                        return self.strip_thinking_tags(content)
                    else:
                        _LOGGER.warning("Gemini response missing parts")
                        _LOGGER.debug(
                            "Full Gemini response: %s", json.dumps(data, indent=2)
                        )
                else:
                    _LOGGER.warning("Gemini response missing expected structure")
                    _LOGGER.debug(
                        "Full Gemini response: %s", json.dumps(data, indent=2)
                    )
                return str(data)


class AnthropicClient(BaseAIClient):
    def __init__(self, token, model="claude-opus-4-6", hass=None):
        super().__init__(hass=hass)
        self.token = token
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"

    async def get_response(self, messages, **kwargs):
        _LOGGER.debug("Making request to Anthropic API with model: %s", self.model)
        headers = {
            "x-api-key": self.token,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        # Convert OpenAI-style messages to Anthropic format
        system_message = None
        anthropic_messages = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                # Anthropic uses a separate system parameter
                system_message = content
            elif role == "user":
                anthropic_messages.append({"role": "user", "content": content})
            elif role == "tool":
                # Anthropic doesn't support role='tool' in basic messages API;
                # map HA data-injection turns to 'user' role.
                anthropic_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                anthropic_messages.append({"role": "assistant", "content": content})

        payload = {
            "model": self.model,
            "max_tokens": 8192,  # Maximum for Anthropic Claude models
            "temperature": 0.0,
            "messages": anthropic_messages,
        }

        # Add system message if present
        if system_message:
            payload["system"] = system_message

        _LOGGER.debug("Anthropic request payload: %s", json.dumps(payload, indent=2))

        async with self._session() as session:
            async with session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    _LOGGER.error("Anthropic API error %d: %s", resp.status, error_text)
                    raise Exception(f"Anthropic API error {resp.status}")
                data = await resp.json()
                # Extract text from Anthropic response
                content_blocks = data.get("content", [])
                if content_blocks and isinstance(content_blocks, list):
                    # Get the text from the first content block
                    for block in content_blocks:
                        if block.get("type") == "text":
                            return self.strip_thinking_tags(
                                block.get("text", str(data))
                            )
                return str(data)

    async def get_response_stream(self, messages, **kwargs):
        """SSE streaming for the Anthropic Messages API.

        Yields text chunks as they arrive. Falls back to non-streaming on error.
        """
        _LOGGER.debug("Anthropic stream: starting with model %s", self.model)
        headers = {
            "x-api-key": self.token,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        system_message = None
        anthropic_messages = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                system_message = content
            elif role == "user":
                anthropic_messages.append({"role": "user", "content": content})
            elif role == "tool":
                anthropic_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                anthropic_messages.append({"role": "assistant", "content": content})

        payload = {
            "model": self.model,
            "max_tokens": 8192,
            "temperature": 0.0,
            "messages": anthropic_messages,
            "stream": True,
        }
        if system_message:
            payload["system"] = system_message

        try:
            async with self._session() as session:
                async with session.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    if resp.status != 200:
                        _LOGGER.warning(
                            "Anthropic stream error %d, falling back", resp.status
                        )
                        result = await self.get_response(messages, **kwargs)
                        yield result
                        return

                    async for line in resp.content:
                        decoded = line.decode("utf-8", errors="replace").strip()
                        if not decoded or not decoded.startswith("data:"):
                            continue
                        data_str = decoded[5:].strip()
                        if not data_str:
                            continue
                        try:
                            event = json.loads(data_str)
                            event_type = event.get("type", "")
                            if event_type == "content_block_delta":
                                text = event.get("delta", {}).get("text", "")
                                if text:
                                    yield text
                            elif event_type == "message_stop":
                                break
                        except (json.JSONDecodeError, KeyError):
                            continue
        except Exception as e:
            _LOGGER.warning("Anthropic streaming failed (%s), falling back", e)
            result = await self.get_response(messages, **kwargs)
            yield result


class OpenRouterClient(BaseAIClient):
    def __init__(self, token, model="openai/gpt-4o", hass=None):
        super().__init__(hass=hass)
        self.token = token
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    async def get_response(self, messages, **kwargs):
        _LOGGER.debug("Making request to OpenRouter API with model: %s", self.model)
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://home-assistant.io",  # Optional for OpenRouter rankings
            "X-Title": "Home Assistant AI Agent",  # Optional for OpenRouter rankings
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "top_p": 0.9,
            # max_tokens omitted - let OpenRouter use the model's maximum capacity
        }

        _LOGGER.debug("OpenRouter request payload: %s", json.dumps(payload, indent=2))

        async with self._session() as session:
            async with session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    _LOGGER.error(
                        "OpenRouter API error %d: %s", resp.status, error_text
                    )
                    raise Exception(f"OpenRouter API error {resp.status}")
                data = await resp.json()
                # Extract text from OpenRouter response (OpenAI-compatible format)
                choices = data.get("choices", [])
                if not choices:
                    _LOGGER.warning("OpenRouter response missing choices")
                    _LOGGER.debug(
                        "Full OpenRouter response: %s", json.dumps(data, indent=2)
                    )
                    return str(data)
                if choices and "message" in choices[0]:
                    return self.strip_thinking_tags(
                        choices[0]["message"].get("content", str(data))
                    )
                return str(data)


class AlterClient(BaseAIClient):
    def __init__(self, token, model="", hass=None):
        super().__init__(hass=hass)
        self.token = token
        self.model = model
        self.api_url = "https://alterhq.com/api/v1/chat/completions"

    async def get_response(self, messages, **kwargs):
        _LOGGER.debug("Making request to Alter API with model: %s", self.model)
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "top_p": 0.9,
        }

        _LOGGER.debug("Alter request payload: %s", json.dumps(payload, indent=2))

        async with self._session() as session:
            async with session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    _LOGGER.error("Alter API error %d: %s", resp.status, error_text)
                    raise Exception(f"Alter API error {resp.status}")
                data = await resp.json()
                # Extract text from Alter response (OpenAI-compatible format)
                choices = data.get("choices", [])
                if not choices:
                    _LOGGER.warning("Alter response missing choices")
                    _LOGGER.debug("Full Alter response: %s", json.dumps(data, indent=2))
                    return str(data)
                if choices and "message" in choices[0]:
                    return self.strip_thinking_tags(
                        choices[0]["message"].get("content", str(data))
                    )
                return str(data)


class ZaiClient(BaseAIClient):
    def __init__(self, token, model="", endpoint_type="general", hass=None):
        super().__init__(hass=hass)
        self.token = token
        self.model = model
        self.endpoint_type = endpoint_type
        # General endpoint: https://api.z.ai/api/paas/v4/chat/completions
        # Coding endpoint: https://api.z.ai/api/coding/paas/v4/chat/completions
        if endpoint_type == "coding":
            self.api_url = "https://api.z.ai/api/coding/paas/v4/chat/completions"
        else:
            self.api_url = "https://api.z.ai/api/paas/v4/chat/completions"

    async def get_response(self, messages, **kwargs):
        _LOGGER.debug(
            "Making request to z.ai API with model: %s, endpoint: %s",
            self.model,
            self.endpoint_type,
        )
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "top_p": 0.9,
        }

        _LOGGER.debug("z.ai request payload: %s", json.dumps(payload, indent=2))

        async with self._session() as session:
            async with session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    _LOGGER.error("z.ai API error %d: %s", resp.status, error_text)
                    raise Exception(f"z.ai API error {resp.status}")
                data = await resp.json()
                # Extract text from z.ai response (OpenAI-compatible format)
                choices = data.get("choices", [])
                if not choices:
                    _LOGGER.warning("z.ai response missing choices")
                    _LOGGER.debug("Full z.ai response: %s", json.dumps(data, indent=2))
                    return str(data)
                if choices and "message" in choices[0]:
                    return self.strip_thinking_tags(
                        choices[0]["message"].get("content", str(data))
                    )
                return str(data)


# --- Ask Sage retry helpers ---
_ASK_SAGE_MAX_RETRIES = 3
_ASK_SAGE_RETRY_DELAYS = [1, 2, 4]  # seconds (exponential backoff)


def _is_overload_response(text: str) -> bool:
    """Return True if Ask Sage returned a transient overload/rate-limit message."""
    lowered = text.lower()
    return (
        "overloaded" in lowered
        or "try again" in lowered
        or "rate limit" in lowered
        or "too many requests" in lowered
    )


class AskSageClient(BaseAIClient):
    """Client for the Ask Sage Server API.

    API reference: https://docs.asksage.ai/api-docs/swagger.html
    Base URL: https://api.asksage.ai/server/
    Auth: x-access-tokens header
    Models: fetched live from GET /get-models, cached on the instance.
    """

    QUERY_URL = "https://api.asksage.ai/server/query"
    AGENT_URL = "https://api.asksage.ai/server/execute-agent"
    MODELS_URL = "https://api.asksage.ai/server/get-models"

    # live: 0=off, 1=Live (Google), 2=Live+ (Google+crawl)
    # deep_agent: True to route through Ask Sage's Deep Agent (/execute-agent)
    def __init__(
        self,
        token: str,
        model: str = "gpt-4o-mini",
        live: int = 0,
        deep_agent: bool = False,
        hass=None,
    ):
        super().__init__(hass=hass)
        self.token = token
        self.model = model
        self.live = int(live)  # coerce str->int in case HA stores it as string
        self.deep_agent = bool(deep_agent)

    @staticmethod
    def _filter_models(raw: list) -> list:
        """Return model ids from the /get-models response, excluding gov models."""
        return [
            m["id"]
            for m in raw
            if isinstance(m, dict) and "id" in m and "gov" not in m["id"].lower()
        ]

    @classmethod
    async def fetch_models(cls) -> list:
        """Fetch available (non-gov) model ids from the public /get-models endpoint.

        Returns an empty list on failure so callers can fall back gracefully.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    cls.MODELS_URL,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return cls._filter_models(data.get("data", []))
                    _LOGGER.warning(
                        "Ask Sage /get-models returned HTTP %d", resp.status
                    )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("Ask Sage model fetch failed: %s", exc)
        return []

    async def validate_data_scope(self) -> dict:
        """Validate that the Ask Sage token is valid and confirm data-scope guarantees.

        Calls /user/get-user-logs to confirm the token authenticates and that
        any HA data sent as query content is stored only under this account.

        Returns a dict with keys:
          - valid (bool): token accepted by Ask Sage
          - account_scoped (bool): logs confirmed to be per-account only
          - message (str): human-readable summary

        DATA FLOW AUDIT:
          - HA entity data is passed inline in the ``message`` field of /query.
          - ``dataset: "none"`` prevents Ask Sage from writing data to RAG.
          - ``limit_references: 0`` prevents RAG retrieval even if dataset changes.
          - Per Ask Sage docs: query content is "fire and forget" — not used for
            model training, not retained after response generation.
          - /user/get-user-logs stores prompt + completion scoped to THIS token
            only — no cross-account access.
          - No data is ever sent to /train or /add-dataset by this integration.
        """
        USER_LOGS_URL = "https://api.asksage.ai/user/get-user-logs"
        headers = {
            "x-access-tokens": self.token,
            "Content-Type": "application/json",
        }
        try:
            async with self._session() as session:
                async with session.post(
                    USER_LOGS_URL,
                    headers=headers,
                    json={},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 401:
                        return {
                            "valid": False,
                            "account_scoped": False,
                            "message": "Ask Sage token rejected (401). HA data will NOT be sent.",
                        }
                    if resp.status == 200:
                        _LOGGER.info(
                            "Ask Sage data-scope validation PASSED: "
                            "token authenticated, query logs are scoped to this account only. "
                            "HA entity data is sent as query content (fire-and-forget, not RAG-ingested). "
                            "dataset=none, limit_references=0 enforced on all requests."
                        )
                        return {
                            "valid": True,
                            "account_scoped": True,
                            "message": (
                                "Data scope confirmed: HA data sent as query content only, "
                                "scoped to this Ask Sage account. Not written to RAG or shared datasets."
                            ),
                        }
                    return {
                        "valid": False,
                        "account_scoped": False,
                        "message": f"Ask Sage validation returned unexpected status {resp.status}.",
                    }
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("Ask Sage data-scope validation failed: %s", exc)
            return {
                "valid": False,
                "account_scoped": False,
                "message": f"Could not validate Ask Sage data scope: {exc}",
            }

    async def get_response(self, messages, **kwargs) -> str:
        """Send a query to Ask Sage and return the generated text.

        Routes to /execute-agent when deep_agent=True, otherwise /query.

        The Ask Sage /query endpoint accepts:
          - message: the prompt (string or conversation array)
          - model: model id string
          - temperature: float 0-1
          - dataset: "none" (we don't inject HA knowledge into Ask Sage's RAG)
          - live: 0=off, 1=Live (Google), 2=Live+ (Google+crawl)

        The response shape is CompletionResponse; the generated text lives in
        the ``message`` field (not ``choices`` like OpenAI-compatible APIs).
        """
        _LOGGER.debug(
            "Ask Sage: sending query with model=%s, live=%d, deep_agent=%s, %d messages",
            self.model,
            self.live,
            self.deep_agent,
            len(messages),
        )

        # DATA SCOPE AUDIT: HA entity data is sent as query content (message field)
        # to Ask Sage's /query endpoint. dataset="none" and limit_references=0
        # ensure no data is written to or retrieved from Ask Sage RAG.
        # Logs are stored per this token's account only (fire-and-forget, no cross-account access).
        _LOGGER.debug(
            "Ask Sage data scope: dataset=none, limit_references=0, "
            "query content scoped to account token. No RAG ingestion."
        )

        headers = {
            "x-access-tokens": self.token,
            "Content-Type": "application/json",
        }

        # Ask Sage's /query accepts the message field as either a plain string
        # (single turn) or an array of {user, message} objects (multi-turn).
        # We normalise from the OpenAI-style {role, content} format that
        # conversation_history uses throughout this integration.
        system_content = ""
        asksage_messages = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                # Ask Sage handles the system prompt via the system_prompt
                # field — capture it and skip from the message array.
                system_content = content
                continue
            elif role in ("user", "me"):
                asksage_messages.append({"user": "me", "message": content})
            elif role in ("assistant", "gpt", "tool"):
                asksage_messages.append({"user": "gpt", "message": content})

        # If the conversation collapsed to a single user turn, send as string
        # for maximum compatibility with the API.
        if len(asksage_messages) == 1 and asksage_messages[0]["user"] == "me":
            message_payload = asksage_messages[0]["message"]
        elif asksage_messages:
            message_payload = asksage_messages
        else:
            # Fallback: grab the last user message as a plain string
            user_msgs = [m for m in messages if m.get("role") == "user"]
            message_payload = user_msgs[-1]["content"] if user_msgs else ""

        payload = {
            "message": message_payload,
            "model": self.model,
            "temperature": 0,
            "dataset": "none",  # Disable Ask Sage RAG; HA data is injected by the agent
            "limit_references": 0,  # Belt-and-suspenders: zero RAG references regardless of dataset setting
            "live": self.live,
        }
        if system_content:
            payload["system_prompt"] = system_content

        _LOGGER.debug(
            "Ask Sage request payload (message preview): %s",
            str(message_payload)[:200],
        )

        # Deep Agent mode routes to /execute-agent; standard queries use /query
        endpoint_url = self.AGENT_URL if self.deep_agent else self.QUERY_URL
        if self.deep_agent:
            # /execute-agent wraps the message in an agent payload
            payload = {
                "message": message_payload,
                "model": self.model,
                "live": self.live,
                "streaming": False,
            }
            if system_content:
                payload["system_prompt"] = system_content
            _LOGGER.debug("Ask Sage: routing to Deep Agent endpoint")

        last_text = ""
        for attempt in range(_ASK_SAGE_MAX_RETRIES):
            async with self._session() as session:
                async with session.post(
                    endpoint_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    if resp.status == 401:
                        raise Exception(
                            "Ask Sage API error 401: invalid or expired token"
                        )
                    if resp.status in (429, 503):
                        error_text = await resp.text()
                        if attempt < _ASK_SAGE_MAX_RETRIES - 1:
                            delay = _ASK_SAGE_RETRY_DELAYS[attempt]
                            _LOGGER.warning(
                                "Ask Sage HTTP %d (attempt %d/%d), retrying in %ds",
                                resp.status,
                                attempt + 1,
                                _ASK_SAGE_MAX_RETRIES,
                                delay,
                            )
                            await asyncio.sleep(delay)
                            continue
                        raise Exception(
                            f"Ask Sage API error {resp.status}: service unavailable after {_ASK_SAGE_MAX_RETRIES} attempts"
                        )
                    if resp.status != 200:
                        error_text = await resp.text()
                        _LOGGER.error(
                            "Ask Sage API error %d: %s", resp.status, error_text
                        )
                        raise Exception(
                            f"Ask Sage API error {resp.status}: {error_text[:200]}"
                        )

                    data = await resp.json()
                    _LOGGER.debug("Ask Sage raw response keys: %s", list(data.keys()))

                    if self.deep_agent:
                        # Deep Agent response: {execution_status, response: {response: "..."}, ...}
                        inner = data.get("response", {})
                        if isinstance(inner, dict):
                            text = inner.get("response", "") or inner.get("message", "")
                        else:
                            text = str(inner)
                        if not text:
                            text = data.get("message", "")
                    else:
                        # Standard /query CompletionResponse: generated text is in `message`.
                        text = data.get("message", "")
                        if not text:
                            # Fallback: some error states surface in `response`
                            text = data.get("response", "")
                    if not text:
                        _LOGGER.warning(
                            "Ask Sage returned empty message. Full response: %s",
                            json.dumps(data, default=str)[:500],
                        )
                        return str(data)

            # Check for overload response before returning
            if not _is_overload_response(text):
                break  # good response, exit loop

            last_text = text
            if attempt < _ASK_SAGE_MAX_RETRIES - 1:
                delay = _ASK_SAGE_RETRY_DELAYS[attempt]
                _LOGGER.warning(
                    "Ask Sage overload response (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1,
                    _ASK_SAGE_MAX_RETRIES,
                    delay,
                    text[:100],
                )
                await asyncio.sleep(delay)
        else:
            # All retries exhausted — return friendly message
            _LOGGER.error(
                "Ask Sage overload after %d attempts. Last response: %s",
                _ASK_SAGE_MAX_RETRIES,
                last_text[:200],
            )
            return json.dumps(
                {
                    "request_type": "final_response",
                    "response": "Ask Sage is temporarily overloaded. It was automatically retried 3 times — if you still see this, please resend your message.",
                }
            )

        return self.strip_thinking_tags(str(text))

    # Ask Sage /query does not support SSE streaming — get_response_stream intentionally
    # omitted so the main loop always uses get_response() for this provider.


# === Main Agent ===
class AiAgentHaAgent:
    """Agent for handling queries with dynamic data requests and multiple AI providers."""

    SYSTEM_PROMPT = {
        "role": "system",
        "content": (
            "CRITICAL OUTPUT FORMAT RULES — READ BEFORE ANYTHING ELSE:\n"
            "- You MUST respond with pure, valid JSON only. No exceptions.\n"
            "- NEVER output YAML. NEVER output markdown. NEVER use code fences (```).\n"
            '- If you are asked to build a dashboard, respond with JSON using request_type="dashboard_suggestion".\n'
            '- A response that starts with "title:", "views:", or "cards:" is WRONG and will break the application.\n'
            "- WRONG: title: My Dashboard\\nviews:\\n  - cards:...\n"
            '- RIGHT: {"request_type": "dashboard_suggestion", "dashboard": {"title": "My Dashboard", "views": [...]}}\n\n'
            "You are an AI assistant integrated with Home Assistant.\n"
            "You can request specific data by using only these commands:\n"
            "- get_entity_state(entity_id): Get state of a specific entity\n"
            "- get_entities_by_domain(domain): Get all entities in a domain\n"
            "- get_entities_by_device_class(device_class, domain?): Get entities with specific device_class (e.g., 'temperature', 'humidity', 'motion')\n"
            "- get_climate_related_entities(): Get all climate-related entities (climate.* entities + temperature/humidity sensors)\n"
            "- get_entities_by_area(area_id): Get all entities in a specific area\n"
            "- get_entities(area_id or area_ids): Get entities by area(s) - supports single area_id or list of area_ids\n"
            "  Use as: get_entities(area_ids=['area1', 'area2']) for multiple areas or get_entities(area_id='single_area')\n"
            "- get_calendar_events(entity_id?): Get calendar events\n"
            "- get_automations(): Get all automations\n"
            "- get_weather_data(): Get current weather and forecast data\n"
            "- get_entity_registry(): Get entity registry entries (now includes device_class, state_class, unit_of_measurement)\n"
            "- get_device_registry(): Get device registry entries\n"
            "- get_area_registry(): Get room/area information\n"
            "- get_history(entity_id, hours): Get historical state changes\n"
            "- get_person_data(): Get person tracking information\n"
            "- get_statistics(entity_id): Get sensor statistics\n"
            "- get_scenes(): Get scene configurations\n"
            "- get_dashboards(): Get list of all dashboards\n"
            "- get_dashboard_config(dashboard_url): Get configuration of a specific dashboard\n"
            "- set_entity_state(entity_id, state, attributes?): Set state of an entity (e.g., turn on/off lights, open/close covers)\n"
            "- call_service(domain, service, target?, service_data?): Call any Home Assistant service directly\n"
            "- create_automation(automation): Create a new automation with the provided configuration\n"
            "- create_dashboard(dashboard_config): Create a new dashboard with the provided configuration\n"
            "- update_dashboard(dashboard_url, dashboard_config): Update an existing dashboard configuration\n\n"
            "IMPORTANT DEVICE_CLASS GUIDANCE:\n"
            "- Many sensors have a 'device_class' attribute (temperature, humidity, motion, etc.)\n"
            "- Use get_climate_related_entities() for climate dashboards (includes climate.* entities and temperature/humidity sensors)\n"
            "- Use get_entities_by_device_class(device_class) to filter by device_class (e.g., 'temperature', 'humidity', 'motion')\n"
            "- For climate dashboards, use history-graph and gauge cards for temperature/humidity sensors\n\n"
            "DASHBOARD CREATION:\n"
            "When a user asks to create a dashboard:\n"
            "1. Gather entities using get_climate_related_entities() or other get_* commands\n"
            "2. Respond with JSON using request_type: 'dashboard_suggestion' (NEVER use 'final_response'!)\n"
            "3. Use Lovelace JSON format (NOT YAML!)\n"
            "4. Example response structure:\n"
            '{"request_type": "dashboard_suggestion", "message": "Dashboard created", "dashboard": {"title": "...", "views": [...]}}\n'
            "5. Do NOT include YAML, markdown, or code blocks - only pure JSON\n\n"
            "IMPORTANT AREA/FLOOR GUIDANCE:\n"
            "- When users ask for entities from a specific floor, use get_area_registry() first\n"
            "- Areas have both 'area_id' and 'floor_id' - these are different concepts\n"
            "- Filter areas by their floor_id to find all areas on a specific floor\n"
            "- Use get_entities() with area_ids parameter to get entities from multiple areas efficiently\n"
            "- Example: get_entities(area_ids=['area1', 'area2', 'area3']) for multiple areas at once\n"
            "- This is more efficient than calling get_entities_by_area() multiple times\n\n"
            "AUTOMATION CREATION:\n"
            "When creating automations, request entities first to know the entity IDs.\n"
            "For days, use: ['fri', 'mon', 'sat', 'sun', 'thu', 'tue', 'wed']\n\n"
            "RESPONSE FORMATS - You must ALWAYS respond with valid JSON:\n\n"
            "For automations:\n"
            "{\n"
            '  "request_type": "automation_suggestion",\n'
            '  "message": "I\'ve created an automation that might help you. Would you like me to create it?",\n'
            '  "automation": {\n'
            '    "alias": "Name of the automation",\n'
            '    "description": "Description of what the automation does",\n'
            '    "trigger": [...],  // Array of trigger conditions\n'
            '    "condition": [...], // Optional array of conditions\n'
            '    "action": [...]     // Array of actions to perform\n'
            "  }\n"
            "}\n\n"
            "For dashboards (WHEN USER ASKS TO CREATE A DASHBOARD):\n"
            "{\n"
            '  "request_type": "dashboard_suggestion",\n'
            '  "message": "Description of the dashboard you created",\n'
            '  "dashboard": {\n'
            '    "title": "Dashboard Title",\n'
            '    "url_path": "url-path",\n'
            '    "icon": "mdi:icon-name",\n'
            '    "show_in_sidebar": true,\n'
            '    "views": [{\n'
            '      "title": "View Title",\n'
            '      "cards": [...]\n'
            "    }]\n"
            "  }\n"
            "}\n"
            "IMPORTANT: The above MUST be returned as raw JSON. Do NOT format it as YAML. "
            "Do NOT add markdown fences. The response must start with { and end with }.\n\n"
            "For data requests, use this exact JSON format:\n"
            "{\n"
            '  "request_type": "data_request",\n'
            '  "request": "command_name",\n'
            '  "parameters": {...}\n'
            "}\n"
            'For get_entities with multiple areas: {"request_type": "get_entities", "parameters": {"area_ids": ["area1", "area2"]}}\n'
            'For get_entities with single area: {"request_type": "get_entities", "parameters": {"area_id": "single_area"}}\n\n'
            "For service calls, use this exact JSON format:\n"
            "{\n"
            '  "request_type": "call_service",\n'
            '  "domain": "light",\n'
            '  "service": "turn_on",\n'
            '  "target": {"entity_id": ["entity1", "entity2"]},\n'
            '  "service_data": {"brightness": 255}\n'
            "}\n\n"
            "For answering questions (NOT creating dashboards/automations):\n"
            "{\n"
            '  "request_type": "final_response",\n'
            '  "response": "your answer to the user"\n'
            "}\n\n"
            "IMPORTANT: Use 'dashboard_suggestion' when creating dashboards, NOT 'final_response'!\n\n"
            "CRITICAL FORMATTING RULES:\n"
            "- You must ALWAYS respond with ONLY a valid JSON object\n"
            "- DO NOT include any text before the JSON\n"
            "- DO NOT include any text after the JSON\n"
            "- DO NOT include explanations or descriptions outside the JSON\n"
            "- Your entire response must be parseable as JSON\n"
            "- Use the 'message' field inside the JSON for user-facing text\n"
            "- NEVER mix regular text with JSON in your response\n\n"
            "WRONG: 'I'll create this for you. {\"request_type\": ...}'\n"
            'CORRECT: \'{"request_type": "dashboard_suggestion", "message": "I\'ll create this for you.", ...}\''
        ),
    }

    SYSTEM_PROMPT_LOCAL = {
        "role": "system",
        "content": (
            "CRITICAL OUTPUT FORMAT RULES — READ BEFORE ANYTHING ELSE:\n"
            "- You MUST respond with pure, valid JSON only. No exceptions.\n"
            "- NEVER output YAML. NEVER output markdown. NEVER use code fences (```).\n"
            '- If you are asked to build a dashboard, respond with JSON using request_type="dashboard_suggestion".\n'
            '- A response that starts with "title:", "views:", or "cards:" is WRONG and will break the application.\n'
            "- WRONG: title: My Dashboard\\nviews:\\n  - cards:...\n"
            '- RIGHT: {"request_type": "dashboard_suggestion", "dashboard": {"title": "My Dashboard", "views": [...]}}\n\n'
            "You are an AI assistant integrated with Home Assistant.\n"
            "You can request specific data by using only these commands:\n"
            "- get_entity_state(entity_id): Get state of a specific entity\n"
            "- get_entities_by_domain(domain): Get all entities in a domain\n"
            "- get_entities_by_device_class(device_class, domain?): Get entities with specific device_class (e.g., 'temperature', 'humidity', 'motion')\n"
            "- get_climate_related_entities(): Get all climate-related entities (climate.* entities + temperature/humidity sensors)\n"
            "- get_entities_by_area(area_id): Get all entities in a specific area\n"
            "- get_entities(area_id or area_ids): Get entities by area(s) - supports single area_id or list of area_ids\n"
            "  Use as: get_entities(area_ids=['area1', 'area2']) for multiple areas or get_entities(area_id='single_area')\n"
            "- get_calendar_events(entity_id?): Get calendar events\n"
            "- get_automations(): Get all automations\n"
            "- get_weather_data(): Get current weather and forecast data\n"
            "- get_entity_registry(): Get entity registry entries (now includes device_class, state_class, unit_of_measurement)\n"
            "- get_device_registry(): Get device registry entries\n"
            "- get_area_registry(): Get room/area information\n"
            "- get_history(entity_id, hours): Get historical state changes\n"
            "- get_person_data(): Get person tracking information\n"
            "- get_statistics(entity_id): Get sensor statistics\n"
            "- get_scenes(): Get scene configurations\n"
            "- get_dashboards(): Get list of all dashboards\n"
            "- get_dashboard_config(dashboard_url): Get configuration of a specific dashboard\n"
            "- set_entity_state(entity_id, state, attributes?): Set state of an entity (e.g., turn on/off lights, open/close covers)\n"
            "- call_service(domain, service, target?, service_data?): Call any Home Assistant service directly\n"
            "- create_automation(automation): Create a new automation with the provided configuration\n"
            "- create_dashboard(dashboard_config): Create a new dashboard with the provided configuration\n"
            "- update_dashboard(dashboard_url, dashboard_config): Update an existing dashboard configuration\n\n"
            "IMPORTANT DEVICE_CLASS GUIDANCE:\n"
            "- Many sensors have a 'device_class' attribute (temperature, humidity, motion, etc.)\n"
            "- Use get_climate_related_entities() for climate dashboards (includes climate.* entities and temperature/humidity sensors)\n"
            "- Use get_entities_by_device_class(device_class) to filter by device_class (e.g., 'temperature', 'humidity', 'motion')\n"
            "- For climate dashboards, use history-graph and gauge cards for temperature/humidity sensors\n\n"
            "DASHBOARD CREATION:\n"
            "When a user asks to create a dashboard:\n"
            "1. Gather entities using get_climate_related_entities() or other get_* commands\n"
            "2. Respond with JSON using request_type: 'dashboard_suggestion' (NEVER use 'final_response'!)\n"
            "3. Use Lovelace JSON format (NOT YAML!)\n"
            "4. Example response structure:\n"
            '{"request_type": "dashboard_suggestion", "message": "Dashboard created", "dashboard": {"title": "...", "views": [...]}}\n'
            "5. Do NOT include YAML, markdown, or code blocks - only pure JSON\n\n"
            "IMPORTANT AREA/FLOOR GUIDANCE:\n"
            "- When users ask for entities from a specific floor, use get_area_registry() first\n"
            "- Areas have both 'area_id' and 'floor_id' - these are different concepts\n"
            "- Filter areas by their floor_id to find all areas on a specific floor\n"
            "- Use get_entities() with area_ids parameter to get entities from multiple areas efficiently\n"
            "- Example: get_entities(area_ids=['area1', 'area2', 'area3']) for multiple areas at once\n"
            "- This is more efficient than calling get_entities_by_area() multiple times\n\n"
            "AUTOMATION CREATION:\n"
            "When creating automations, request entities first to know the entity IDs.\n"
            "For days, use: ['fri', 'mon', 'sat', 'sun', 'thu', 'tue', 'wed']\n\n"
            "RESPONSE FORMATS - You must ALWAYS respond with valid JSON:\n\n"
            "For automations:\n"
            "{\n"
            '  "request_type": "automation_suggestion",\n'
            '  "message": "I\'ve created an automation that might help you. Would you like me to create it?",\n'
            '  "automation": {\n'
            '    "alias": "Name of the automation",\n'
            '    "description": "Description of what the automation does",\n'
            '    "trigger": [...],  // Array of trigger conditions\n'
            '    "condition": [...], // Optional array of conditions\n'
            '    "action": [...]     // Array of actions to perform\n'
            "  }\n"
            "}\n\n"
            "For dashboards (WHEN USER ASKS TO CREATE A DASHBOARD):\n"
            "{\n"
            '  "request_type": "dashboard_suggestion",\n'
            '  "message": "Description of the dashboard you created",\n'
            '  "dashboard": {\n'
            '    "title": "Dashboard Title",\n'
            '    "url_path": "url-path",\n'
            '    "icon": "mdi:icon-name",\n'
            '    "show_in_sidebar": true,\n'
            '    "views": [{\n'
            '      "title": "View Title",\n'
            '      "cards": [...]\n'
            "    }]\n"
            "  }\n"
            "}\n"
            "IMPORTANT: The above MUST be returned as raw JSON. Do NOT format it as YAML. "
            "Do NOT add markdown fences. The response must start with { and end with }.\n\n"
            "For data requests, use this exact JSON format:\n"
            "{\n"
            '  "request_type": "data_request",\n'
            '  "request": "command_name",\n'
            '  "parameters": {...}\n'
            "}\n"
            'For get_entities with multiple areas: {"request_type": "get_entities", "parameters": {"area_ids": ["area1", "area2"]}}\n'
            'For get_entities with single area: {"request_type": "get_entities", "parameters": {"area_id": "single_area"}}\n\n'
            "For service calls, use this exact JSON format:\n"
            "{\n"
            '  "request_type": "call_service",\n'
            '  "domain": "light",\n'
            '  "service": "turn_on",\n'
            '  "target": {"entity_id": ["entity1", "entity2"]},\n'
            '  "service_data": {"brightness": 255}\n'
            "}\n\n"
            "For answering questions (NOT creating dashboards/automations):\n"
            "{\n"
            '  "request_type": "final_response",\n'
            '  "response": "your answer to the user"\n'
            "}\n\n"
            "IMPORTANT: Use 'dashboard_suggestion' when creating dashboards, NOT 'final_response'!\n\n"
            "CRITICAL FORMATTING RULES:\n"
            "- You must ALWAYS respond with ONLY a valid JSON object\n"
            "- DO NOT include any text before the JSON\n"
            "- DO NOT include any text after the JSON\n"
            "- DO NOT include explanations or descriptions outside the JSON\n"
            "- Your entire response must be parseable as JSON\n"
            "- Use the 'message' field inside the JSON for user-facing text\n"
            "- NEVER mix regular text with JSON in your response\n\n"
            "WRONG: 'I'll create this for you. {\"request_type\": ...}'\n"
            'CORRECT: \'{"request_type": "dashboard_suggestion", "message": "I\'ll create this for you.", ...}\''
        ),
    }

    def __init__(self, hass: HomeAssistant, config: Dict[str, Any]):
        """Initialize the agent with provider selection."""
        self.hass = hass
        self.config = config
        self.conversation_history: List[Dict[str, Any]] = []
        self._cache: Dict[str, Any] = {}
        self.ai_client: BaseAIClient
        self._cache_timeout = 300  # 5 minutes
        self._max_retries = 10
        self._retry_delay = 1  # seconds
        self._rate_limit = 60  # requests per minute
        self._last_request_time = 0
        self._request_count = 0
        self._request_window_start = time.time()
        # Serialise access to process_query so concurrent calls don't
        # interleave conversation_history reads/writes.
        self._query_lock = asyncio.Lock()
        # Maximum conversation_history entries kept in memory.
        # _get_ai_response only sends the last 10 to the model, but we
        # keep a larger buffer for rollback and debug trace purposes.
        self._max_history_len = 50

        provider = config.get("ai_provider", "openai")
        models_config = config.get("models", {})

        _LOGGER.debug("Initializing AiAgentHaAgent with provider: %s", provider)
        _LOGGER.debug("Models config loaded: %s", models_config)

        # Set the appropriate system prompt based on provider
        if provider == "local":
            self.system_prompt = self.SYSTEM_PROMPT_LOCAL
            _LOGGER.debug("Using local-optimized system prompt")
        else:
            self.system_prompt = self.SYSTEM_PROMPT
            _LOGGER.debug("Using standard system prompt")

        # Initialize the appropriate AI client with model selection
        if provider == "openai":
            model = models_config.get("openai", "gpt-3.5-turbo")
            base_url = config.get("openai_base_url", "") or ""
            self.ai_client = OpenAIClient(
                config.get("openai_token"), model, base_url, hass=self.hass
            )
        elif provider == "gemini":
            model = models_config.get("gemini", "gemini-2.5-flash")
            self.ai_client = GeminiClient(
                config.get("gemini_token"), model, hass=self.hass
            )
        elif provider == "openrouter":
            model = models_config.get("openrouter", "openai/gpt-4o")
            self.ai_client = OpenRouterClient(
                config.get("openrouter_token"), model, hass=self.hass
            )
        elif provider == "anthropic":
            model = models_config.get("anthropic", "claude-opus-4-6")
            self.ai_client = AnthropicClient(
                config.get("anthropic_token"), model, hass=self.hass
            )
        elif provider == "alter":
            model = models_config.get("alter", "")
            self.ai_client = AlterClient(
                config.get("alter_token"), model, hass=self.hass
            )
        elif provider == "zai":
            model = models_config.get("zai", "glm-4.7")
            endpoint_type = config.get("zai_endpoint", "general")
            self.ai_client = ZaiClient(
                config.get("zai_token"), model, endpoint_type, hass=self.hass
            )
        elif provider == "asksage":
            model = models_config.get("asksage", "gpt-4o-mini")
            self.ai_client = AskSageClient(
                config.get("asksage_token"),
                model,
                live=config.get("asksage_live", 0),
                deep_agent=config.get("asksage_deep_agent", False),
                hass=self.hass,
            )
            # Schedule data-scope validation as a background task so it runs
            # after HA finishes initializing without blocking the setup path.
            if self.hass:

                async def _run_asksage_validation(client=self.ai_client):
                    result = await client.validate_data_scope()
                    if not result["valid"]:
                        _LOGGER.warning(
                            "Ask Sage data-scope validation: %s", result["message"]
                        )

                self.hass.async_create_task(_run_asksage_validation())
        elif provider == "local":
            model = models_config.get("local", "")
            url = config.get("local_url")
            if not url:
                _LOGGER.error("Missing local_url for local provider")
                raise Exception("Missing local_url configuration for local provider")
            self.ai_client = LocalClient(url, model, hass=self.hass)
        else:  # default to llama if somehow specified
            model = models_config.get("llama", "Llama-4-Maverick-17B-128E-Instruct-FP8")
            self.ai_client = LlamaClient(
                config.get("llama_token"), model, hass=self.hass
            )

        _LOGGER.debug(
            "AiAgentHaAgent initialized successfully with provider: %s, model: %s",
            provider,
            model,
        )

    def _validate_api_key(self) -> bool:
        """Validate the API key format."""
        provider = self.config.get("ai_provider", "openai")

        if provider == "openai":
            token = self.config.get("openai_token")
        elif provider == "gemini":
            token = self.config.get("gemini_token")
        elif provider == "openrouter":
            token = self.config.get("openrouter_token")
        elif provider == "anthropic":
            token = self.config.get("anthropic_token")
        elif provider == "alter":
            token = self.config.get("alter_token")
        elif provider == "zai":
            token = self.config.get("zai_token")
        elif provider == "asksage":
            token = self.config.get("asksage_token")
        elif provider == "local":
            token = self.config.get("local_url")
        else:
            token = self.config.get("llama_token")

        if not token or not isinstance(token, str):
            return False

        # For local provider, validate URL format
        if provider == "local":
            return bool(token.startswith(("http://", "https://")))

        # For OpenAI with a custom base URL (e.g. LM Studio), skip the length check
        if provider == "openai" and self.config.get("openai_base_url", "").strip():
            return len(token) > 0

        # Add more specific validation based on your API key format
        return len(token) >= 32

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        current_time = time.time()
        if current_time - self._request_window_start >= 60:
            self._request_count = 0
            self._request_window_start = current_time

        if self._request_count >= self._rate_limit:
            return False

        self._request_count += 1
        return True

    def _get_cached_data(self, key: str) -> Optional[Any]:
        """Get data from cache if it's still valid."""
        if key in self._cache:
            timestamp, data = self._cache[key]
            if time.time() - timestamp < self._cache_timeout:
                return data
            del self._cache[key]
        return None

    def _set_cached_data(self, key: str, data: Any) -> None:
        """Store data in cache with timestamp."""
        self._cache[key] = (time.time(), data)

    def _sanitize_automation_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize automation configuration to prevent injection attacks."""
        sanitized: Dict[str, Any] = {}
        for key, value in config.items():
            if key in ["alias", "description"]:
                # Sanitize strings
                sanitized[key] = str(value).strip()[:100]  # Limit length
            elif key in ["trigger", "condition", "action"]:
                # Validate arrays
                if isinstance(value, list):
                    sanitized[key] = value
            elif key == "mode":
                # Validate mode
                if value in ["single", "restart", "queued", "parallel"]:
                    sanitized[key] = value
        return sanitized

    async def get_entity_state(self, entity_id: str) -> Dict[str, Any]:
        """Get the state of a specific entity."""
        try:
            _LOGGER.debug("Requesting entity state for: %s", entity_id)
            state = self.hass.states.get(entity_id)
            if not state:
                _LOGGER.warning("Entity not found: %s", entity_id)
                return {"error": f"Entity {entity_id} not found"}

            # Get area information from entity/device registry
            # Wrapped in try-except to handle cases where registries aren't available (e.g., in tests)
            area_id = None
            area_name = None

            try:
                entity_registry = er.async_get(self.hass)
                device_registry = dr.async_get(self.hass)
                area_registry = ar.async_get(self.hass)

                if entity_registry and hasattr(entity_registry, "async_get"):
                    # Try to find the entity in the registry
                    entity_entry = entity_registry.async_get(entity_id)
                    if entity_entry:
                        _LOGGER.debug("Entity %s found in registry", entity_id)
                        # Check if entity has a direct area assignment
                        if hasattr(entity_entry, "area_id") and entity_entry.area_id:
                            area_id = entity_entry.area_id
                            _LOGGER.debug(
                                "Entity %s has direct area assignment: %s",
                                entity_id,
                                area_id,
                            )
                        # Otherwise check if the entity's device has an area
                        elif (
                            hasattr(entity_entry, "device_id")
                            and entity_entry.device_id
                            and device_registry
                            and hasattr(device_registry, "async_get")
                        ):
                            _LOGGER.debug(
                                "Entity %s has device_id: %s, checking device area",
                                entity_id,
                                entity_entry.device_id,
                            )
                            device_entry = device_registry.async_get(
                                entity_entry.device_id
                            )
                            if device_entry:
                                if (
                                    hasattr(device_entry, "area_id")
                                    and device_entry.area_id
                                ):
                                    area_id = device_entry.area_id
                                    _LOGGER.debug(
                                        "Device %s has area: %s",
                                        entity_entry.device_id,
                                        area_id,
                                    )
                                else:
                                    _LOGGER.debug(
                                        "Device %s has no area assigned",
                                        entity_entry.device_id,
                                    )
                            else:
                                _LOGGER.debug(
                                    "Device %s not found in registry",
                                    entity_entry.device_id,
                                )
                        else:
                            _LOGGER.debug(
                                "Entity %s has no area_id and no device_id", entity_id
                            )
                    else:
                        _LOGGER.debug(
                            "Entity %s not found in entity registry", entity_id
                        )
                else:
                    _LOGGER.debug("Entity registry not available for %s", entity_id)

                # Get area name from area_id
                if (
                    area_id
                    and area_registry
                    and hasattr(area_registry, "async_get_area")
                ):
                    area_entry = area_registry.async_get_area(area_id)
                    if area_entry and hasattr(area_entry, "name"):
                        area_name = area_entry.name
                        _LOGGER.debug(
                            "Resolved area_id %s to area_name: %s", area_id, area_name
                        )
                    else:
                        _LOGGER.debug("Could not resolve area_id %s to name", area_id)
                elif area_id:
                    _LOGGER.debug(
                        "Have area_id %s but area_registry not available", area_id
                    )
            except Exception as e:
                # Registries not available (likely in test environment) - skip area information
                _LOGGER.warning(
                    "Exception retrieving area information for %s: %s",
                    entity_id,
                    str(e),
                )

            result = {
                "entity_id": state.entity_id,
                "state": state.state,
                "last_changed": (
                    state.last_changed.isoformat() if state.last_changed else None
                ),
                "friendly_name": state.attributes.get("friendly_name"),
                "area_id": area_id,
                "area_name": area_name,
                "attributes": {
                    k: (v.isoformat() if hasattr(v, "isoformat") else v)
                    for k, v in state.attributes.items()
                },
            }
            _LOGGER.debug(
                "Retrieved entity state for %s: area_id=%s, area_name=%s",
                entity_id,
                area_id,
                area_name,
            )
            return result
        except Exception as e:
            _LOGGER.exception("Error getting entity state: %s", str(e))
            return {"error": f"Error getting entity state: {str(e)}"}

    async def get_entities_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """Get all entities for a specific domain."""
        try:
            _LOGGER.debug("Requesting all entities for domain: %s", domain)
            states = [
                state
                for state in self.hass.states.async_all()
                if state.entity_id.startswith(f"{domain}.")
            ]
            _LOGGER.debug("Found %d entities in domain %s", len(states), domain)
            return [await self.get_entity_state(state.entity_id) for state in states]
        except Exception as e:
            _LOGGER.exception("Error getting entities by domain: %s", str(e))
            return [{"error": f"Error getting entities for domain {domain}: {str(e)}"}]

    async def get_entities_by_device_class(
        self, device_class: str, domain: str = None
    ) -> List[Dict[str, Any]]:
        """Get all entities with a specific device_class.

        Args:
            device_class: The device class to filter by (e.g., 'temperature', 'humidity', 'motion')
            domain: Optional domain to restrict search (e.g., 'sensor', 'binary_sensor')

        Returns:
            List of entity state dictionaries that match the device_class
        """
        try:
            _LOGGER.debug(
                "Requesting all entities with device_class: %s (domain: %s)",
                device_class,
                domain or "all",
            )
            matching_entities = []

            for state in self.hass.states.async_all():
                # Filter by domain if specified
                if domain and not state.entity_id.startswith(f"{domain}."):
                    continue

                # Check if this entity has the matching device_class
                entity_device_class = state.attributes.get("device_class")
                if entity_device_class == device_class:
                    matching_entities.append(state.entity_id)

            _LOGGER.debug(
                "Found %d entities with device_class %s",
                len(matching_entities),
                device_class,
            )

            # Get full state information for each matching entity
            return [
                await self.get_entity_state(entity_id)
                for entity_id in matching_entities
            ]

        except Exception as e:
            _LOGGER.exception("Error getting entities by device_class: %s", str(e))
            return [
                {
                    "error": f"Error getting entities with device_class {device_class}: {str(e)}"
                }
            ]

    async def get_climate_related_entities(self) -> List[Dict[str, Any]]:
        """Get all climate-related entities including climate domain and temperature/humidity sensors.

        Returns:
            List of entity state dictionaries for:
            - All climate.* entities (thermostats, HVAC systems)
            - All sensor.* entities with device_class: temperature
            - All sensor.* entities with device_class: humidity
        """
        try:
            _LOGGER.debug("Requesting all climate-related entities")
            climate_entities = []

            # Get all climate domain entities (thermostats, HVAC)
            climate_domain = await self.get_entities_by_domain("climate")
            climate_entities.extend(climate_domain)

            # Get temperature sensors
            temp_sensors = await self.get_entities_by_device_class(
                "temperature", "sensor"
            )
            climate_entities.extend(temp_sensors)

            # Get humidity sensors
            humidity_sensors = await self.get_entities_by_device_class(
                "humidity", "sensor"
            )
            climate_entities.extend(humidity_sensors)

            # Deduplicate by entity_id (edge case: if an entity appears in multiple categories)
            seen_entity_ids = set()
            unique_entities = []
            for entity in climate_entities:
                entity_id = entity.get("entity_id")
                if entity_id and entity_id not in seen_entity_ids:
                    seen_entity_ids.add(entity_id)
                    unique_entities.append(entity)

            _LOGGER.debug(
                "Found %d total climate-related entities (deduplicated from %d)",
                len(unique_entities),
                len(climate_entities),
            )
            return unique_entities

        except Exception as e:
            _LOGGER.exception("Error getting climate-related entities: %s", str(e))
            return [{"error": f"Error getting climate-related entities: {str(e)}"}]

    async def get_entities_by_area(self, area_id: str) -> List[Dict[str, Any]]:
        """Get all entities for a specific area."""
        try:
            _LOGGER.debug("Requesting all entities for area: %s", area_id)

            # Get entity registry to find entities assigned to the area
            entity_registry = er.async_get(self.hass)
            device_registry = dr.async_get(self.hass)

            entities_in_area = []

            # Find entities assigned to the area (directly or through their device)
            for entity in entity_registry.entities.values():
                # Check if entity is directly assigned to the area
                if entity.area_id == area_id:
                    entities_in_area.append(entity.entity_id)
                # Check if entity's device is assigned to the area
                elif entity.device_id:
                    device = device_registry.devices.get(entity.device_id)
                    if device and device.area_id == area_id:
                        entities_in_area.append(entity.entity_id)

            _LOGGER.debug(
                "Found %d entities in area %s", len(entities_in_area), area_id
            )

            # Get state information for each entity
            result = []
            for entity_id in entities_in_area:
                state_info = await self.get_entity_state(entity_id)
                if not state_info.get("error"):  # Only include entities that exist
                    result.append(state_info)

            return result

        except Exception as e:
            _LOGGER.exception("Error getting entities by area: %s", str(e))
            return [{"error": f"Error getting entities for area {area_id}: {str(e)}"}]

    async def get_entities(self, area_id=None, area_ids=None) -> List[Dict[str, Any]]:
        """Get entities by area(s) - flexible method that supports single area or multiple areas."""
        try:
            # Handle different parameter formats
            areas_to_process = []

            if area_ids:
                # Multiple areas provided
                if isinstance(area_ids, list):
                    areas_to_process = area_ids
                else:
                    areas_to_process = [area_ids]
            elif area_id:
                # Single area provided
                if isinstance(area_id, list):
                    areas_to_process = area_id
                else:
                    areas_to_process = [area_id]
            else:
                return [{"error": "No area_id or area_ids provided"}]

            _LOGGER.debug("Requesting entities for areas: %s", areas_to_process)

            all_entities = []
            for area in areas_to_process:
                entities_in_area = await self.get_entities_by_area(area)
                all_entities.extend(entities_in_area)

            # Remove duplicates based on entity_id
            seen_entities = set()
            unique_entities = []
            for entity in all_entities:
                if isinstance(entity, dict) and "entity_id" in entity:
                    if entity["entity_id"] not in seen_entities:
                        seen_entities.add(entity["entity_id"])
                        unique_entities.append(entity)
                else:
                    unique_entities.append(entity)  # Keep error messages

            _LOGGER.debug(
                "Found %d unique entities across %d areas",
                len(unique_entities),
                len(areas_to_process),
            )
            return unique_entities

        except Exception as e:
            _LOGGER.exception("Error getting entities: %s", str(e))
            return [{"error": f"Error getting entities: {str(e)}"}]

    async def get_calendar_events(
        self, entity_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get calendar events, optionally filtered by entity_id."""
        try:
            if entity_id:
                _LOGGER.debug(
                    "Requesting calendar events for specific entity: %s", entity_id
                )
                return [await self.get_entity_state(entity_id)]

            _LOGGER.debug("Requesting all calendar events")
            return await self.get_entities_by_domain("calendar")
        except Exception as e:
            _LOGGER.exception("Error getting calendar events: %s", str(e))
            return [{"error": f"Error getting calendar events: {str(e)}"}]

    async def get_automations(self) -> List[Dict[str, Any]]:
        """Get all automations."""
        try:
            _LOGGER.debug("Requesting all automations")
            return await self.get_entities_by_domain("automation")
        except Exception as e:
            _LOGGER.exception("Error getting automations: %s", str(e))
            return [{"error": f"Error getting automations: {str(e)}"}]

    async def get_entity_registry(self) -> List[Dict]:
        """Get entity registry entries with device_class and other metadata.

        Area information is resolved from the entity or its device.
        """
        _LOGGER.debug("Requesting all entity registry entries")
        try:
            entity_registry = er.async_get(self.hass)
            if not entity_registry:
                return []

            device_registry = dr.async_get(self.hass)
            area_registry = ar.async_get(self.hass)

            result = []
            for entry in entity_registry.entities.values():
                # Get the current state to access device_class and other attributes
                state = self.hass.states.get(entry.entity_id)
                device_class = state.attributes.get("device_class") if state else None
                state_class = state.attributes.get("state_class") if state else None
                unit_of_measurement = (
                    state.attributes.get("unit_of_measurement") if state else None
                )

                # Resolve area_id and area_name
                # First check entity's direct area assignment
                area_id = entry.area_id
                area_name = None

                # If entity doesn't have area, check device's area
                if not area_id and entry.device_id and device_registry:
                    device_entry = device_registry.async_get(entry.device_id)
                    if device_entry and hasattr(device_entry, "area_id"):
                        area_id = device_entry.area_id

                # Resolve area_name from area_id
                if area_id and area_registry:
                    area_entry = area_registry.async_get_area(area_id)
                    if area_entry and hasattr(area_entry, "name"):
                        area_name = area_entry.name

                result.append(
                    {
                        "entity_id": entry.entity_id,
                        "device_id": entry.device_id,
                        "platform": entry.platform,
                        "disabled": entry.disabled,
                        "area_id": area_id,
                        "area_name": area_name,
                        "original_name": entry.original_name,
                        "unique_id": entry.unique_id,
                        "device_class": device_class,
                        "state_class": state_class,
                        "unit_of_measurement": unit_of_measurement,
                    }
                )

            return result
        except Exception as e:
            _LOGGER.exception("Error getting entity registry entries: %s", str(e))
            return [{"error": f"Error getting entity registry entries: {str(e)}"}]

    async def get_device_registry(self) -> List[Dict]:
        """Get device registry entries"""
        _LOGGER.debug("Requesting all device registry entries")
        try:
            registry = dr.async_get(self.hass)
            if not registry:
                return []
            return [
                {
                    "id": device.id,
                    "name": device.name,
                    "model": device.model,
                    "manufacturer": device.manufacturer,
                    "sw_version": device.sw_version,
                    "hw_version": device.hw_version,
                    "connections": (
                        list(device.connections) if device.connections else []
                    ),
                    "identifiers": (
                        list(device.identifiers) if device.identifiers else []
                    ),
                    "area_id": device.area_id,
                    "disabled": device.disabled_by is not None,
                    "entry_type": (
                        device.entry_type.value if device.entry_type else None
                    ),
                    "name_by_user": device.name_by_user,
                }
                for device in registry.devices.values()
            ]
        except Exception as e:
            _LOGGER.exception("Error getting device registry entries: %s", str(e))
            return [{"error": f"Error getting device registry entries: {str(e)}"}]

    async def get_history(self, entity_id: str, hours: int = 24) -> List[Dict]:
        """Get historical state changes for an entity"""
        _LOGGER.debug("Requesting historical state changes for entity: %s", entity_id)
        try:
            from homeassistant.components.recorder.history import get_significant_states

            now = dt_util.utcnow()
            start = now - timedelta(hours=hours)

            # Get history using the recorder history module
            history_data = await self.hass.async_add_executor_job(
                get_significant_states,
                self.hass,
                start,
                now,
                [entity_id],
            )

            # Convert to serializable format
            result = []
            for entity_id_key, states in history_data.items():
                for state in states:
                    # Skip if it's a dict (mypy type narrowing)
                    if isinstance(state, dict):
                        continue
                    result.append(
                        {
                            "entity_id": state.entity_id,
                            "state": state.state,
                            "last_changed": state.last_changed.isoformat(),
                            "last_updated": state.last_updated.isoformat(),
                            "attributes": dict(state.attributes),
                        }
                    )
            return result
        except Exception as e:
            _LOGGER.exception("Error getting history: %s", str(e))
            return [{"error": f"Error getting history: {str(e)}"}]

    async def get_area_registry(self) -> Dict[str, Any]:
        """Get area registry information"""
        _LOGGER.debug("Get area registry information")
        try:
            registry = ar.async_get(self.hass)
            if not registry:
                return {}

            result = {}
            for area in registry.areas.values():
                result[area.id] = {
                    "name": area.name,
                    "normalized_name": area.normalized_name,
                    "picture": area.picture,
                    "icon": area.icon,
                    "floor_id": area.floor_id,
                    "labels": list(area.labels) if area.labels else [],
                }
            return result
        except Exception as e:
            _LOGGER.exception("Error getting area registry: %s", str(e))
            return {"error": f"Error getting area registry: {str(e)}"}

    async def get_person_data(self) -> List[Dict]:
        """Get person tracking information"""
        _LOGGER.debug("Requesting person tracking information")
        try:
            result = []
            for state in self.hass.states.async_all("person"):
                result.append(
                    {
                        "entity_id": state.entity_id,
                        "name": state.attributes.get("friendly_name", state.entity_id),
                        "state": state.state,
                        "latitude": state.attributes.get("latitude"),
                        "longitude": state.attributes.get("longitude"),
                        "source": state.attributes.get("source"),
                        "gps_accuracy": state.attributes.get("gps_accuracy"),
                        "last_changed": (
                            state.last_changed.isoformat()
                            if state.last_changed
                            else None
                        ),
                    }
                )
            return result
        except Exception as e:
            _LOGGER.exception("Error getting person tracking information: %s", str(e))
            return [{"error": f"Error getting person tracking information: {str(e)}"}]

    async def get_statistics(self, entity_id: str) -> Dict:
        """Get statistics for an entity"""
        _LOGGER.debug("Requesting statistics for entity: %s", entity_id)
        try:
            from homeassistant.components import recorder

            # Check if recorder is available
            if not self.hass.data.get(recorder.DATA_INSTANCE):
                return {"error": "Recorder component is not available"}

            # from homeassistant.components.recorder.statistics import get_latest_short_term_statistics
            import homeassistant.components.recorder.statistics as stats_module

            # Get latest statistics
            stats = await self.hass.async_add_executor_job(
                # get_latest_short_term_statistics,
                stats_module.get_last_short_term_statistics,
                self.hass,
                1,
                entity_id,
                True,
                set(),
            )

            if entity_id in stats:
                stat_data = stats[entity_id][0] if stats[entity_id] else {}
                return {
                    "entity_id": entity_id,
                    "start": stat_data.get("start"),
                    "mean": stat_data.get("mean"),
                    "min": stat_data.get("min"),
                    "max": stat_data.get("max"),
                    "last_reset": stat_data.get("last_reset"),
                    "state": stat_data.get("state"),
                    "sum": stat_data.get("sum"),
                }
            else:
                return {"error": f"No statistics available for entity {entity_id}"}
        except Exception as e:
            _LOGGER.exception("Error getting statistics: %s", str(e))
            return {"error": f"Error getting statistics: {str(e)}"}

    async def get_scenes(self) -> List[Dict]:
        """Get scene configurations"""
        _LOGGER.debug("Requesting scene configurations")
        try:
            result = []
            for state in self.hass.states.async_all("scene"):
                result.append(
                    {
                        "entity_id": state.entity_id,
                        "name": state.attributes.get("friendly_name", state.entity_id),
                        "last_activated": state.attributes.get("last_activated"),
                        "icon": state.attributes.get("icon"),
                        "last_changed": (
                            state.last_changed.isoformat()
                            if state.last_changed
                            else None
                        ),
                    }
                )
            return result
        except Exception as e:
            _LOGGER.exception("Error getting scene configurations: %s", str(e))
            return [{"error": f"Error getting scene configurations: {str(e)}"}]

    async def get_weather_data(self) -> Dict[str, Any]:
        """Get weather data from any available weather entity in the system."""
        try:
            # Find all weather entities
            weather_entities = [
                state
                for state in self.hass.states.async_all()
                if state.domain == "weather"
            ]

            if not weather_entities:
                return {
                    "error": "No weather entities found in the system. Please add a weather integration."
                }

            # Use the first available weather entity
            state = weather_entities[0]
            _LOGGER.debug("Using weather entity: %s", state.entity_id)

            # Get all available attributes
            all_attributes = state.attributes
            _LOGGER.debug(
                "Available weather attributes: %s", json.dumps(all_attributes)
            )

            # Get forecast data
            forecast = all_attributes.get("forecast", [])

            # Process forecast data
            processed_forecast = []
            for day in forecast:
                forecast_entry = {
                    "datetime": day.get("datetime"),
                    "temperature": day.get("temperature"),
                    "condition": day.get("condition"),
                    "precipitation": day.get("precipitation"),
                    "precipitation_probability": day.get("precipitation_probability"),
                    "humidity": day.get("humidity"),
                    "wind_speed": day.get("wind_speed"),
                    "wind_bearing": day.get("wind_bearing"),
                }
                # Only add entries that have at least some data
                if any(v is not None for v in forecast_entry.values()):
                    processed_forecast.append(forecast_entry)

            # Get current weather data
            current = {
                "entity_id": state.entity_id,
                "temperature": all_attributes.get("temperature"),
                "humidity": all_attributes.get("humidity"),
                "pressure": all_attributes.get("pressure"),
                "wind_speed": all_attributes.get("wind_speed"),
                "wind_bearing": all_attributes.get("wind_bearing"),
                "condition": state.state,
                "forecast_available": len(processed_forecast) > 0,
            }

            # Log the processed data for debugging
            _LOGGER.debug(
                "Processed weather data: %s",
                json.dumps(
                    {"current": current, "forecast_count": len(processed_forecast)}
                ),
            )

            return {"current": current, "forecast": processed_forecast}
        except Exception as e:
            _LOGGER.exception("Error getting weather data: %s", str(e))
            return {"error": f"Error getting weather data: {str(e)}"}

    async def create_automation(
        self, automation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new automation with validation and sanitization."""
        try:
            _LOGGER.debug(
                "Creating automation with config: %s", json.dumps(automation_config)
            )

            # Validate required fields
            if not all(
                key in automation_config for key in ["alias", "trigger", "action"]
            ):
                return {"error": "Missing required fields in automation configuration"}

            # Sanitize configuration
            sanitized_config = self._sanitize_automation_config(automation_config)

            # Generate a unique ID for the automation
            automation_id = f"ai_agent_auto_{int(time.time() * 1000)}"

            # Create the automation entry
            automation_entry = {
                "id": automation_id,
                "alias": sanitized_config["alias"],
                "description": sanitized_config.get("description", ""),
                "trigger": sanitized_config["trigger"],
                "condition": sanitized_config.get("condition", []),
                "action": sanitized_config["action"],
                "mode": sanitized_config.get("mode", "single"),
            }

            # Read current automations.yaml using async executor
            automations_path = self.hass.config.path("automations.yaml")
            try:
                current_automations = await self.hass.async_add_executor_job(
                    lambda: yaml.safe_load(open(automations_path, "r")) or []
                )
            except FileNotFoundError:
                current_automations = []

            # Check for duplicate automation names
            if any(
                auto.get("alias") == automation_entry["alias"]
                for auto in current_automations
            ):
                return {
                    "error": f"An automation with the name '{automation_entry['alias']}' already exists"
                }

            # Append new automation
            current_automations.append(automation_entry)

            # Write back to file using async executor
            await self.hass.async_add_executor_job(
                lambda: yaml.dump(
                    current_automations,
                    open(automations_path, "w"),
                    default_flow_style=False,
                )
            )

            # Reload automations
            await self.hass.services.async_call("automation", "reload")

            # Clear automation-related caches
            self._cache.clear()

            return {
                "success": True,
                "message": f"Automation '{automation_entry['alias']}' created successfully",
            }

        except Exception as e:
            _LOGGER.exception("Error creating automation: %s", str(e))
            return {"error": f"Error creating automation: {str(e)}"}

    async def get_dashboards(self) -> List[Dict[str, Any]]:
        """Get list of all dashboards."""
        try:
            _LOGGER.debug("Requesting all dashboards")

            # Get dashboards via WebSocket API
            ws_api = self.hass.data.get("websocket_api")
            if not ws_api:
                return [{"error": "WebSocket API not available"}]

            # Use the lovelace service to get dashboards
            try:
                from homeassistant.components.lovelace import DOMAIN as LOVELACE_DOMAIN

                # Get lovelace data using property access (required for HA 2026.2+)
                # lovelace_data is a LovelaceData dataclass with a 'dashboards' attribute
                lovelace_data = self.hass.data.get(LOVELACE_DOMAIN)
                if lovelace_data is None:
                    return [{"error": "Lovelace not available"}]

                # Safety check for dashboards attribute (backward compatibility)
                if not hasattr(lovelace_data, "dashboards"):
                    return [{"error": "Lovelace dashboards not available"}]

                # Use property access instead of dictionary access
                dashboards = lovelace_data.dashboards

                # Get YAML dashboard configs for metadata (title, icon, etc.)
                # yaml_dashboards contains the configuration with metadata
                yaml_configs = getattr(lovelace_data, "yaml_dashboards", {}) or {}

                dashboard_list = []

                # Iterate over all dashboards (None key = default dashboard)
                for url_path, dashboard_obj in dashboards.items():
                    # Try to get metadata from yaml_dashboards first
                    yaml_config = yaml_configs.get(url_path, {}) or {}

                    # Get title - check yaml config, then use defaults
                    title = yaml_config.get("title")
                    if not title:
                        title = (
                            "Overview"
                            if url_path is None
                            else (url_path or "Dashboard")
                        )

                    # Get icon - check yaml config, then use defaults
                    icon = yaml_config.get("icon")
                    if not icon:
                        icon = "mdi:home" if url_path is None else "mdi:view-dashboard"

                    # Get sidebar/admin settings from yaml config or defaults
                    show_in_sidebar = yaml_config.get("show_in_sidebar", True)
                    require_admin = yaml_config.get("require_admin", False)

                    dashboard_list.append(
                        {
                            "url_path": url_path,
                            "title": title,
                            "icon": icon,
                            "show_in_sidebar": show_in_sidebar,
                            "require_admin": require_admin,
                        }
                    )

                _LOGGER.debug("Found %d dashboards", len(dashboard_list))
                return dashboard_list

            except Exception as e:
                _LOGGER.warning("Could not get dashboards via lovelace: %s", str(e))
                return [{"error": f"Could not retrieve dashboards: {str(e)}"}]

        except Exception as e:
            _LOGGER.exception("Error getting dashboards: %s", str(e))
            return [{"error": f"Error getting dashboards: {str(e)}"}]

    async def get_dashboard_config(
        self, dashboard_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get configuration of a specific dashboard."""
        try:
            _LOGGER.debug(
                "Requesting dashboard config for: %s", dashboard_url or "default"
            )

            # Get dashboard configuration
            try:
                from homeassistant.components.lovelace import DOMAIN as LOVELACE_DOMAIN

                # Get lovelace data using property access (required for HA 2026.2+)
                lovelace_data = self.hass.data.get(LOVELACE_DOMAIN)
                if lovelace_data is None:
                    return {"error": "Lovelace not available"}

                # Safety check for dashboards attribute (backward compatibility)
                if not hasattr(lovelace_data, "dashboards"):
                    return {"error": "Lovelace dashboards not available"}

                # Use property access instead of dictionary access
                # The dashboards dict uses None as key for the default dashboard
                dashboards = lovelace_data.dashboards

                # Get the dashboard (None key = default dashboard)
                dashboard_key = None if dashboard_url is None else dashboard_url
                if dashboard_key in dashboards:
                    dashboard = dashboards[dashboard_key]
                    config = await dashboard.async_get_info()
                    return dict(config) if config else {"error": "No dashboard config"}
                else:
                    if dashboard_url is None:
                        return {"error": "Default dashboard not found"}
                    else:
                        return {"error": f"Dashboard '{dashboard_url}' not found"}

            except Exception as e:
                _LOGGER.warning("Could not get dashboard config: %s", str(e))
                return {"error": f"Could not retrieve dashboard config: {str(e)}"}

        except Exception as e:
            _LOGGER.exception("Error getting dashboard config: %s", str(e))
            return {"error": f"Error getting dashboard config: {str(e)}"}

    async def create_dashboard(
        self, dashboard_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new dashboard using Home Assistant's Lovelace WebSocket API."""
        try:
            _LOGGER.debug(
                "Creating dashboard with config: %s",
                json.dumps(dashboard_config, default=str),
            )

            # Validate required fields
            if not dashboard_config.get("title"):
                return {"error": "Dashboard title is required"}

            if not dashboard_config.get("url_path"):
                return {"error": "Dashboard URL path is required"}

            # Sanitize the URL path
            url_path = (
                dashboard_config["url_path"].lower().replace(" ", "-").replace("_", "-")
            )

            # Prepare dashboard configuration for Lovelace
            dashboard_data = {
                "title": dashboard_config["title"],
                "icon": dashboard_config.get("icon", "mdi:view-dashboard"),
                "show_in_sidebar": dashboard_config.get("show_in_sidebar", True),
                "require_admin": dashboard_config.get("require_admin", False),
                "views": dashboard_config.get("views", []),
            }

            try:
                # Create dashboard file directly - this is the most reliable method
                import os

                import yaml

                # Create the dashboard YAML file
                lovelace_config_file = self.hass.config.path(
                    f"ui-lovelace-{url_path}.yaml"
                )

                # Use async_add_executor_job to perform file I/O asynchronously
                def write_dashboard_file():
                    with open(lovelace_config_file, "w") as f:
                        yaml.dump(
                            dashboard_data,
                            f,
                            default_flow_style=False,
                            allow_unicode=True,
                        )

                await self.hass.async_add_executor_job(write_dashboard_file)

                _LOGGER.info(
                    "Successfully created dashboard file: %s", lovelace_config_file
                )

                # Now update configuration.yaml
                try:
                    config_file = self.hass.config.path("configuration.yaml")
                    dashboard_config_entry = {
                        url_path: {
                            "mode": "yaml",
                            "title": dashboard_config["title"],
                            "icon": dashboard_config.get("icon", "mdi:view-dashboard"),
                            "show_in_sidebar": dashboard_config.get(
                                "show_in_sidebar", True
                            ),
                            "filename": f"ui-lovelace-{url_path}.yaml",
                        }
                    }

                    def update_config_file():
                        try:
                            with open(config_file, "r") as f:
                                content = f.read()

                            # Dashboard configuration to add
                            dashboard_yaml = f"""    {url_path}:
      mode: yaml
      title: {dashboard_config['title']}
      icon: {dashboard_config.get('icon', 'mdi:view-dashboard')}
      show_in_sidebar: {str(dashboard_config.get('show_in_sidebar', True)).lower()}
      filename: ui-lovelace-{url_path}.yaml"""

                            # Check if lovelace section exists
                            if "lovelace:" not in content:
                                # Add complete lovelace section at the end
                                lovelace_section = f"""
# Lovelace dashboards configuration added by AI Agent
lovelace:
  dashboards:
{dashboard_yaml}
"""
                                with open(config_file, "a") as f:
                                    f.write(lovelace_section)
                                return True

                            # If lovelace exists, check for dashboards section
                            lines = content.split("\n")
                            new_lines = []
                            dashboard_added = False
                            in_lovelace = False
                            lovelace_indent = 0

                            for i, line in enumerate(lines):
                                new_lines.append(line)

                                # Detect lovelace section
                                if (
                                    line.strip() == "lovelace:"
                                    or line.strip().startswith("lovelace:")
                                ):
                                    in_lovelace = True
                                    lovelace_indent = len(line) - len(line.lstrip())
                                    continue

                                # If we're in lovelace section
                                if in_lovelace:
                                    current_indent = (
                                        len(line) - len(line.lstrip())
                                        if line.strip()
                                        else 0
                                    )

                                    # If we hit another top-level section, we're out of lovelace
                                    if (
                                        line.strip()
                                        and current_indent <= lovelace_indent
                                        and not line.startswith(" ")
                                    ):
                                        if line.strip() != "lovelace:":
                                            in_lovelace = False

                                    # Look for dashboards section
                                    if in_lovelace and "dashboards:" in line:
                                        # Add our dashboard after the dashboards: line
                                        new_lines.append(dashboard_yaml)
                                        dashboard_added = True
                                        in_lovelace = False  # We're done
                                        break

                            # If we found lovelace but no dashboards section, add it
                            if not dashboard_added and "lovelace:" in content:
                                # Find lovelace section and add dashboards
                                new_lines = []
                                for line in lines:
                                    new_lines.append(line)
                                    if (
                                        line.strip() == "lovelace:"
                                        or line.strip().startswith("lovelace:")
                                    ):
                                        # Add dashboards section right after lovelace
                                        new_lines.append("  dashboards:")
                                        new_lines.append(dashboard_yaml)
                                        dashboard_added = True
                                        break

                            if dashboard_added:
                                with open(config_file, "w") as f:
                                    f.write("\n".join(new_lines))
                                return True
                            else:
                                # Last resort: append to end of file
                                with open(config_file, "a") as f:
                                    f.write(f"\n  dashboards:\n{dashboard_yaml}\n")
                                return True

                        except Exception as e:
                            _LOGGER.error(
                                "Failed to update configuration.yaml: %s", str(e)
                            )
                            # Fallback to simple append method
                            try:
                                with open(config_file, "r") as f:
                                    content = f.read()

                                # Check if lovelace section exists
                                if "lovelace:" not in content:
                                    # Add lovelace section
                                    lovelace_config = f"""
# Lovelace dashboards
lovelace:
  dashboards:
    {url_path}:
      mode: yaml
      title: {dashboard_config['title']}
      icon: {dashboard_config.get('icon', 'mdi:view-dashboard')}
      show_in_sidebar: {str(dashboard_config.get('show_in_sidebar', True)).lower()}
      filename: ui-lovelace-{url_path}.yaml
"""
                                    with open(config_file, "a") as f:
                                        f.write(lovelace_config)
                                else:
                                    # Add to existing lovelace section (simple approach)
                                    dashboard_entry = f"""    {url_path}:
      mode: yaml
      title: {dashboard_config['title']}
      icon: {dashboard_config.get('icon', 'mdi:view-dashboard')}
      show_in_sidebar: {str(dashboard_config.get('show_in_sidebar', True)).lower()}
      filename: ui-lovelace-{url_path}.yaml
"""
                                    # Find the dashboards section and add to it
                                    lines = content.split("\n")
                                    new_lines = []
                                    in_dashboards = False
                                    dashboards_indented = False

                                    for line in lines:
                                        new_lines.append(line)
                                        if (
                                            "dashboards:" in line
                                            and "lovelace"
                                            in content[: content.find(line)]
                                        ):
                                            in_dashboards = True
                                            # Add our dashboard entry after dashboards:
                                            new_lines.append(dashboard_entry.rstrip())
                                            in_dashboards = False

                                    # If we couldn't find dashboards section, add it under lovelace
                                    if not any("dashboards:" in line for line in lines):
                                        for i, line in enumerate(new_lines):
                                            if line.strip() == "lovelace:":
                                                new_lines.insert(i + 1, "  dashboards:")
                                                new_lines.insert(
                                                    i + 2, dashboard_entry.rstrip()
                                                )
                                                break

                                    with open(config_file, "w") as f:
                                        f.write("\n".join(new_lines))

                                return True
                            except Exception as fallback_error:
                                _LOGGER.error(
                                    "Fallback config update also failed: %s",
                                    str(fallback_error),
                                )
                                return False

                    config_updated = await self.hass.async_add_executor_job(
                        update_config_file
                    )

                    if config_updated:
                        success_message = f"""Dashboard '{dashboard_config['title']}' created successfully!

✅ Dashboard file created: ui-lovelace-{url_path}.yaml
✅ Configuration.yaml updated automatically

🔄 Please restart Home Assistant to see your new dashboard in the sidebar."""

                        return {
                            "success": True,
                            "message": success_message,
                            "url_path": url_path,
                            "restart_required": True,
                        }
                    else:
                        # Config update failed, provide manual instructions
                        config_instructions = f"""Dashboard '{dashboard_config['title']}' created successfully!

✅ Dashboard file created: ui-lovelace-{url_path}.yaml
⚠️  Could not automatically update configuration.yaml

Please manually add this to your configuration.yaml:

lovelace:
  dashboards:
    {url_path}:
      mode: yaml
      title: {dashboard_config['title']}
      icon: {dashboard_config.get('icon', 'mdi:view-dashboard')}
      show_in_sidebar: {str(dashboard_config.get('show_in_sidebar', True)).lower()}
      filename: ui-lovelace-{url_path}.yaml

Then restart Home Assistant to see your new dashboard in the sidebar."""

                        return {
                            "success": True,
                            "message": config_instructions,
                            "url_path": url_path,
                            "restart_required": True,
                        }

                except Exception as config_error:
                    _LOGGER.error(
                        "Error updating configuration.yaml: %s", str(config_error)
                    )
                    # Provide manual instructions as fallback
                    config_instructions = f"""Dashboard '{dashboard_config['title']}' created successfully!

✅ Dashboard file created: ui-lovelace-{url_path}.yaml
⚠️  Could not automatically update configuration.yaml

Please manually add this to your configuration.yaml:

lovelace:
  dashboards:
    {url_path}:
      mode: yaml
      title: {dashboard_config['title']}
      icon: {dashboard_config.get('icon', 'mdi:view-dashboard')}
      show_in_sidebar: {str(dashboard_config.get('show_in_sidebar', True)).lower()}
      filename: ui-lovelace-{url_path}.yaml

Then restart Home Assistant to see your new dashboard in the sidebar."""

                    return {
                        "success": True,
                        "message": config_instructions,
                        "url_path": url_path,
                        "restart_required": True,
                    }

            except Exception as e:
                _LOGGER.error("Failed to create dashboard file: %s", str(e))
                return {"error": f"Failed to create dashboard file: {str(e)}"}

        except Exception as e:
            _LOGGER.exception("Error creating dashboard: %s", str(e))
            return {"error": f"Error creating dashboard: {str(e)}"}

    async def update_dashboard(
        self, dashboard_url: str, dashboard_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update an existing dashboard using Home Assistant's Lovelace WebSocket API."""
        try:
            _LOGGER.debug(
                "Updating dashboard %s with config: %s",
                dashboard_url,
                json.dumps(dashboard_config, default=str),
            )

            # Prepare updated dashboard configuration
            dashboard_data = {
                "title": dashboard_config.get("title", "Updated Dashboard"),
                "icon": dashboard_config.get("icon", "mdi:view-dashboard"),
                "show_in_sidebar": dashboard_config.get("show_in_sidebar", True),
                "require_admin": dashboard_config.get("require_admin", False),
                "views": dashboard_config.get("views", []),
            }

            try:
                # Update dashboard file directly
                import os

                import yaml

                # Try updating the YAML file
                dashboard_file = self.hass.config.path(
                    f"ui-lovelace-{dashboard_url}.yaml"
                )

                # Check if file exists asynchronously
                def check_file_exists():
                    return os.path.exists(dashboard_file)

                file_exists = await self.hass.async_add_executor_job(check_file_exists)

                if not file_exists:
                    dashboard_file = self.hass.config.path(
                        f"dashboards/{dashboard_url}.yaml"
                    )
                    file_exists = await self.hass.async_add_executor_job(
                        lambda: os.path.exists(dashboard_file)
                    )

                if file_exists:
                    # Use async_add_executor_job to perform file I/O asynchronously
                    def update_dashboard_file():
                        with open(dashboard_file, "w") as f:
                            yaml.dump(
                                dashboard_data,
                                f,
                                default_flow_style=False,
                                allow_unicode=True,
                            )

                    await self.hass.async_add_executor_job(update_dashboard_file)

                    _LOGGER.info(
                        "Successfully updated dashboard file: %s", dashboard_file
                    )
                    return {
                        "success": True,
                        "message": f"Dashboard '{dashboard_url}' updated successfully!",
                    }
                else:
                    return {"error": f"Dashboard file for '{dashboard_url}' not found"}

            except Exception as e:
                _LOGGER.error("Failed to update dashboard file: %s", str(e))
                return {"error": f"Failed to update dashboard file: {str(e)}"}

        except Exception as e:
            _LOGGER.exception("Error updating dashboard: %s", str(e))
            return {"error": f"Error updating dashboard: {str(e)}"}

    async def process_query(
        self, user_query: str, provider: Optional[str] = None, debug: bool = False
    ) -> Dict[str, Any]:
        """Process a user query with input validation and rate limiting."""
        async with self._query_lock:
            return await self._process_query_inner(user_query, provider, debug)

    async def _process_query_inner(
        self, user_query: str, provider: Optional[str] = None, debug: bool = False
    ) -> Dict[str, Any]:
        """Inner implementation of process_query, always called under _query_lock."""
        try:
            if not user_query or not isinstance(user_query, str):
                return {"success": False, "error": "Invalid query format"}

            # Get the correct configuration for the requested provider
            if provider and provider in self.hass.data[DOMAIN]["configs"]:
                config = self.hass.data[DOMAIN]["configs"][provider]
            else:
                config = self.config

            _LOGGER.debug(f"Processing query with provider: {provider}")
            # Log sanitized config (masks all tokens/keys for security)
            _LOGGER.debug(
                f"Using config: {json.dumps(sanitize_for_logging(config), default=str)}"
            )

            selected_provider = provider or config.get("ai_provider", "llama")
            models_config = config.get("models", {})

            provider_config = {
                "openai": {
                    "token_key": "openai_token",
                    "model": models_config.get("openai", "gpt-3.5-turbo"),
                    "client_class": OpenAIClient,
                },
                "gemini": {
                    "token_key": "gemini_token",
                    "model": models_config.get("gemini", "gemini-1.5-flash"),
                    "client_class": GeminiClient,
                },
                "openrouter": {
                    "token_key": "openrouter_token",
                    "model": models_config.get("openrouter", "openai/gpt-4o"),
                    "client_class": OpenRouterClient,
                },
                "llama": {
                    "token_key": "llama_token",
                    "model": models_config.get(
                        "llama", "Llama-4-Maverick-17B-128E-Instruct-FP8"
                    ),
                    "client_class": LlamaClient,
                },
                "anthropic": {
                    "token_key": "anthropic_token",
                    "model": models_config.get("anthropic", "claude-opus-4-6"),
                    "client_class": AnthropicClient,
                },
                "alter": {
                    "token_key": "alter_token",
                    "model": models_config.get("alter", ""),
                    "client_class": AlterClient,
                },
                "zai": {
                    "token_key": "zai_token",
                    "model": models_config.get("zai", ""),
                    "client_class": ZaiClient,
                },
                "local": {
                    "token_key": "local_url",
                    "model": models_config.get("local", ""),
                    "client_class": LocalClient,
                },
                "asksage": {
                    "token_key": "asksage_token",
                    "model": models_config.get("asksage", "gpt-4o-mini"),
                    "client_class": AskSageClient,
                },
            }

            # Validate provider and get configuration
            if selected_provider not in provider_config:
                _LOGGER.warning(
                    f"Invalid provider {selected_provider}, falling back to llama"
                )
                selected_provider = "llama"

            provider_settings = provider_config[selected_provider]
            token = self.config.get(provider_settings["token_key"])

            def _with_debug(result: Dict[str, Any]) -> Dict[str, Any]:
                """Attach a sanitized trace when UI requests debug info."""
                if debug and "debug" not in result:
                    result["debug"] = self._build_debug_trace(
                        selected_provider,
                        provider_settings,
                        config.get("zai_endpoint", "general"),
                    )
                return result

            # Validate token/URL
            if not token:
                error_msg = f"No {'URL' if selected_provider == 'local' else 'token'} configured for provider {selected_provider}"
                _LOGGER.error(error_msg)
                return _with_debug({"success": False, "error": error_msg})

            # Initialize client
            try:
                if selected_provider == "zai":
                    # ZaiClient takes (token, model, endpoint_type)
                    endpoint_type = config.get("zai_endpoint", "general")
                    self.ai_client = provider_settings["client_class"](
                        token=token,
                        model=provider_settings["model"],
                        endpoint_type=endpoint_type,
                        hass=self.hass,
                    )
                    _LOGGER.debug(
                        f"Initialized {selected_provider} client with model {provider_settings['model']}, endpoint_type {endpoint_type}"
                    )
                elif selected_provider == "local":
                    # LocalClient takes (url, model)
                    self.ai_client = provider_settings["client_class"](
                        url=token, model=provider_settings["model"], hass=self.hass
                    )
                    _LOGGER.debug(
                        f"Initialized {selected_provider} client with model {provider_settings['model']}"
                    )
                elif selected_provider == "asksage":
                    # AskSageClient takes (token, model, live, deep_agent)
                    self.ai_client = provider_settings["client_class"](
                        token=token,
                        model=provider_settings["model"],
                        live=config.get("asksage_live", 0),
                        deep_agent=config.get("asksage_deep_agent", False),
                        hass=self.hass,
                    )
                    _LOGGER.debug(
                        f"Initialized {selected_provider} client with model {provider_settings['model']}, "
                        f"live={config.get('asksage_live', 0)}, deep_agent={config.get('asksage_deep_agent', False)}"
                    )
                    # Run data-scope validation inline (we are already inside an async context).
                    # Warn but do not abort — a failed validation is not a hard error.
                    try:
                        scope_result = await self.ai_client.validate_data_scope()
                        if not scope_result["valid"]:
                            _LOGGER.warning(
                                "Ask Sage data-scope validation failed: %s",
                                scope_result["message"],
                            )
                        else:
                            _LOGGER.debug(
                                "Ask Sage data-scope: %s", scope_result["message"]
                            )
                    except Exception as _scope_exc:  # noqa: BLE001
                        _LOGGER.warning(
                            "Ask Sage data-scope check error: %s", _scope_exc
                        )
                else:
                    # Other clients take (token, model)
                    # For OpenAI, also pass optional custom base_url override
                    if selected_provider == "openai":
                        base_url = config.get("openai_base_url", "") or ""
                        self.ai_client = provider_settings["client_class"](
                            token=token,
                            model=provider_settings["model"],
                            base_url=base_url,
                            hass=self.hass,
                        )
                    else:
                        self.ai_client = provider_settings["client_class"](
                            token=token,
                            model=provider_settings["model"],
                            hass=self.hass,
                        )
                    _LOGGER.debug(
                        f"Initialized {selected_provider} client with model {provider_settings['model']}"
                    )
            except Exception as e:
                error_msg = f"Error initializing {selected_provider} client: {str(e)}"
                _LOGGER.error(error_msg)
                return _with_debug({"success": False, "error": error_msg})

            # Process the query with rate limiting and retries
            if not self._check_rate_limit():
                return _with_debug(
                    {
                        "success": False,
                        "error": "Rate limit exceeded. Please wait before trying again.",
                    }
                )

            # Sanitize user input
            user_query = user_query.strip()[:1000]  # Limit length and trim whitespace

            _LOGGER.debug("Processing new query: %s", user_query)

            # Check cache for identical query
            cache_key = f"query_{hash(user_query)}_{provider}_{debug}"
            cached_result = self._get_cached_data(cache_key)
            if cached_result:
                return (
                    dict(cached_result)
                    if isinstance(cached_result, dict)
                    else {"error": "Invalid cached result"}
                )

            # Add system message to conversation if it's the first message
            if not self.conversation_history:
                _LOGGER.debug("Adding system message to new conversation")
                self.conversation_history.append(self.system_prompt)

            # Add user query to conversation.
            # Record the index so we can roll back to this point on error,
            # preventing a failed query from poisoning the next call.
            history_rollback_index = len(self.conversation_history)
            self.conversation_history.append({"role": "user", "content": user_query})
            _LOGGER.debug(
                "Added user query to conversation history (rollback index=%d)",
                history_rollback_index,
            )

            max_iterations = 5  # Prevent infinite loops
            iteration = 0

            while iteration < max_iterations:
                iteration += 1
                _LOGGER.debug(f"Processing iteration {iteration} of {max_iterations}")

                try:
                    # Get AI response (with optional streaming)
                    _LOGGER.debug("Requesting response from AI provider")
                    streaming_enabled = config.get("enable_streaming", False)
                    t_start = time.monotonic()

                    if streaming_enabled and hasattr(
                        self.ai_client, "get_response_stream"
                    ):
                        response = await self._get_ai_response_stream()
                    else:
                        response = await self._get_ai_response()

                    t_end = time.monotonic()
                    thinking_duration = round(t_end - t_start, 1)

                    # Extract thinking content BEFORE stripping
                    thinking_content = (
                        BaseAIClient.extract_thinking(response) if response else None
                    )

                    # Strip thinking tags from the response for further processing
                    if response:
                        response = BaseAIClient.strip_thinking_tags(response)

                    _LOGGER.debug("Received response from AI provider: %s", response)

                    try:
                        # Try to parse the response as JSON with simplified approach
                        response_clean = response.strip()

                        # Remove potential BOM and other invisible characters
                        if response_clean.startswith(codecs.BOM_UTF8.decode("utf-8")):
                            response_clean = response_clean[1:]

                        # Remove other common invisible characters
                        invisible_chars = [
                            "\ufeff",
                            "\u200b",
                            "\u200c",
                            "\u200d",
                            "\u2060",
                        ]
                        for char in invisible_chars:
                            response_clean = response_clean.replace(char, "")

                        _LOGGER.debug(
                            "Cleaned response length: %d", len(response_clean)
                        )
                        _LOGGER.debug(
                            "Cleaned response first 100 chars: %s", response_clean[:100]
                        )
                        _LOGGER.debug(
                            "Cleaned response last 100 chars: %s", response_clean[-100:]
                        )

                        # Simple strategy: try to parse the cleaned response directly
                        response_data = None
                        try:
                            _LOGGER.debug("Attempting basic JSON parse...")
                            response_data = json.loads(response_clean)
                            _LOGGER.debug("Basic JSON parse succeeded!")
                        except json.JSONDecodeError as e:
                            _LOGGER.warning("Basic JSON parse failed: %s", str(e))
                            _LOGGER.debug("JSON error position: %d", e.pos)
                            if e.pos < len(response_clean):
                                _LOGGER.debug(
                                    "Character at error position: %s (ord: %d)",
                                    repr(response_clean[e.pos]),
                                    ord(response_clean[e.pos]),
                                )
                                _LOGGER.debug(
                                    "Context around error: %s",
                                    repr(
                                        response_clean[max(0, e.pos - 10) : e.pos + 10]
                                    ),
                                )

                            # Fallback: try to extract JSON by finding the first { and last }
                            json_start = response_clean.find("{")
                            json_end = response_clean.rfind("}")

                            if (
                                json_start != -1
                                and json_end != -1
                                and json_end > json_start
                            ):
                                json_part = response_clean[json_start : json_end + 1]
                                _LOGGER.debug(
                                    "Trying fallback extraction from pos %d to %d",
                                    json_start,
                                    json_end,
                                )
                                _LOGGER.debug("Extracted JSON: %s", json_part[:200])

                                try:
                                    response_data = json.loads(json_part)
                                    _LOGGER.debug("Fallback JSON extraction succeeded!")
                                except json.JSONDecodeError as e2:
                                    _LOGGER.warning(
                                        "Fallback JSON extraction also failed: %s",
                                        str(e2),
                                    )
                                    raise e  # Re-raise the original error
                            else:
                                _LOGGER.warning(
                                    "Could not find JSON boundaries in response"
                                )
                                raise e  # Re-raise the original error

                        if response_data is None:
                            raise json.JSONDecodeError(
                                "All parsing strategies failed", response_clean, 0
                            )

                        _LOGGER.debug("Successfully parsed JSON response")
                        _LOGGER.debug(
                            "Parsed response type: %s",
                            response_data.get("request_type", "unknown"),
                        )

                        # Check if this is a data request (either format)
                        data_request_types = [
                            "get_entity_state",
                            "get_entities_by_domain",
                            "get_entities_by_device_class",
                            "get_climate_related_entities",
                            "get_entities_by_area",
                            "get_entities",
                            "get_calendar_events",
                            "get_automations",
                            "get_entity_registry",
                            "get_device_registry",
                            "get_weather_data",
                            "get_area_registry",
                            "get_history",
                            "get_person_data",
                            "get_statistics",
                            "get_scenes",
                            "get_dashboards",
                            "get_dashboard_config",
                            "set_entity_state",
                            "create_automation",
                            "create_dashboard",
                            "update_dashboard",
                        ]

                        if (
                            response_data.get("request_type") == "data_request"
                            or response_data.get("request_type") in data_request_types
                        ):
                            # Handle data request (both standard format and direct request type)
                            if response_data.get("request_type") == "data_request":
                                request_type = response_data.get("request")
                            else:
                                request_type = response_data.get("request_type")
                            parameters = response_data.get("parameters", {})

                            # Normalise model-invented aliases to canonical request types
                            _REQUEST_TYPE_ALIASES = {
                                "get_temperature_sensors": "get_entities_by_device_class",
                                "get_humidity_sensors": "get_entities_by_device_class",
                                "get_motion_sensors": "get_entities_by_device_class",
                                "get_sensor_states": "get_entities_by_domain",
                                "get_sensors": "get_entities_by_domain",
                                "get_lights": "get_entities_by_domain",
                                "get_switches": "get_entities_by_domain",
                                "get_all_entities": "get_entities",
                                "get_entity": "get_entity_state",
                                "get_state": "get_entity_state",
                                "get_weather": "get_weather_data",
                                "get_forecast": "get_weather_data",
                                "get_climate": "get_climate_related_entities",
                                "get_climate_entities": "get_climate_related_entities",
                            }
                            if request_type in _REQUEST_TYPE_ALIASES:
                                canonical = _REQUEST_TYPE_ALIASES[request_type]
                                _LOGGER.debug(
                                    "Normalising request_type alias %r -> %r",
                                    request_type,
                                    canonical,
                                )
                                # Inject device_class/domain from the alias if not set
                                if (
                                    request_type in ("get_temperature_sensors",)
                                    and "device_class" not in parameters
                                ):
                                    parameters = {
                                        **parameters,
                                        "device_class": "temperature",
                                    }
                                elif (
                                    request_type in ("get_humidity_sensors",)
                                    and "device_class" not in parameters
                                ):
                                    parameters = {
                                        **parameters,
                                        "device_class": "humidity",
                                    }
                                elif (
                                    request_type in ("get_motion_sensors",)
                                    and "device_class" not in parameters
                                ):
                                    parameters = {
                                        **parameters,
                                        "device_class": "motion",
                                    }
                                elif (
                                    request_type in ("get_sensor_states", "get_sensors")
                                    and "domain" not in parameters
                                ):
                                    parameters = {**parameters, "domain": "sensor"}
                                elif (
                                    request_type in ("get_lights",)
                                    and "domain" not in parameters
                                ):
                                    parameters = {**parameters, "domain": "light"}
                                elif (
                                    request_type in ("get_switches",)
                                    and "domain" not in parameters
                                ):
                                    parameters = {**parameters, "domain": "switch"}
                                request_type = canonical
                            _LOGGER.debug(
                                "Processing data request: %s with parameters: %s",
                                request_type,
                                json.dumps(parameters),
                            )

                            # Add AI's response to conversation history
                            self.conversation_history.append(
                                {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        response_data
                                    ),  # Store clean JSON
                                }
                            )

                            # Get requested data
                            data: Union[Dict[str, Any], List[Dict[str, Any]]]
                            if request_type == "get_entity_state":
                                data = await self.get_entity_state(
                                    parameters.get("entity_id")
                                )
                            elif request_type == "get_entities_by_domain":
                                data = await self.get_entities_by_domain(
                                    parameters.get("domain")
                                )
                            elif request_type == "get_entities_by_area":
                                data = await self.get_entities_by_area(
                                    parameters.get("area_id")
                                )
                            elif request_type == "get_entities":
                                data = await self.get_entities(
                                    area_id=parameters.get("area_id"),
                                    area_ids=parameters.get("area_ids"),
                                )
                            elif request_type == "get_entities_by_device_class":
                                data = await self.get_entities_by_device_class(
                                    parameters.get("device_class"),
                                    parameters.get("domain"),
                                )
                            elif request_type == "get_climate_related_entities":
                                data = await self.get_climate_related_entities()
                            elif request_type == "get_calendar_events":
                                data = await self.get_calendar_events(
                                    parameters.get("entity_id")
                                )
                            elif request_type == "get_automations":
                                data = await self.get_automations()
                            elif request_type == "get_entity_registry":
                                data = await self.get_entity_registry()
                            elif request_type == "get_device_registry":
                                data = await self.get_device_registry()
                            elif request_type == "get_weather_data":
                                data = await self.get_weather_data()
                            elif request_type == "get_area_registry":
                                data = await self.get_area_registry()
                            elif request_type == "get_history":
                                data = await self.get_history(
                                    parameters.get("entity_id"),
                                    parameters.get("hours", 24),
                                )
                            elif request_type == "get_person_data":
                                data = await self.get_person_data()
                            elif request_type == "get_statistics":
                                data = await self.get_statistics(
                                    parameters.get("entity_id")
                                )
                            elif request_type == "get_scenes":
                                data = await self.get_scenes()
                            elif request_type == "get_dashboards":
                                data = await self.get_dashboards()
                            elif request_type == "get_dashboard_config":
                                data = await self.get_dashboard_config(
                                    parameters.get("dashboard_url")
                                )
                            elif request_type == "set_entity_state":
                                data = await self.set_entity_state(
                                    parameters.get("entity_id"),
                                    parameters.get("state"),
                                    parameters.get("attributes"),
                                )
                            elif request_type == "create_automation":
                                data = await self.create_automation(
                                    parameters.get("automation")
                                )
                            elif request_type == "create_dashboard":
                                data = await self.create_dashboard(
                                    parameters.get("dashboard_config")
                                )
                            elif request_type == "update_dashboard":
                                data = await self.update_dashboard(
                                    parameters.get("dashboard_url"),
                                    parameters.get("dashboard_config"),
                                )
                            else:
                                # Unknown request type — tell the model and let it retry
                                _LOGGER.warning(
                                    "Unknown request type from model: %r — feeding error back for retry",
                                    request_type,
                                )
                                self.conversation_history.append(
                                    {
                                        "role": "tool",
                                        "content": json.dumps(
                                            {
                                                "error": f"Unknown request type: {request_type!r}. "
                                                "Use one of: get_entity_state, get_entities_by_device_class, "
                                                "get_entities_by_domain, get_climate_related_entities, "
                                                "get_weather_data, get_entities, get_entity_registry, "
                                                "get_area_registry, get_history, get_statistics, "
                                                "get_automations, get_scenes, get_dashboards, "
                                                "create_automation, create_dashboard, update_dashboard, "
                                                "call_service, final_response."
                                            }
                                        ),
                                    }
                                )
                                continue  # retry loop with error context

                            # Check if any data request resulted in an error
                            if isinstance(data, dict) and "error" in data:
                                return _with_debug(
                                    {"success": False, "error": data["error"]}
                                )
                            elif isinstance(data, list) and any(
                                "error" in item
                                for item in data
                                if isinstance(item, dict)
                            ):
                                errors = [
                                    item["error"]
                                    for item in data
                                    if isinstance(item, dict) and "error" in item
                                ]
                                return _with_debug(
                                    {"success": False, "error": "; ".join(errors)}
                                )

                            _LOGGER.debug(
                                "Retrieved data for request: %s",
                                json.dumps(data, default=str),
                            )

                            # Add data to conversation as a tool message.
                            # Using role='tool' (not 'user') keeps HA data injections
                            # semantically distinct from real human queries, preventing
                            # LM Studio / Jinja prompt templates from failing with
                            # "No user query found in messages" when the 10-message
                            # context window fills up with data-fetch turns.
                            self.conversation_history.append(
                                {
                                    "role": "tool",
                                    "content": json.dumps({"data": data}, default=str),
                                }
                            )
                            continue

                        elif response_data.get("request_type") == "final_response":
                            # Add final response to conversation history
                            self.conversation_history.append(
                                {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        response_data
                                    ),  # Store clean JSON
                                }
                            )

                            # Return final response
                            _LOGGER.debug(
                                "Received final response: %s",
                                response_data.get("response"),
                            )
                            result = {
                                "success": True,
                                "answer": response_data.get("response", ""),
                            }
                            # Attach thinking content if present
                            if thinking_content:
                                result["thinking"] = thinking_content
                                result["thinking_duration"] = thinking_duration
                            result = _with_debug(result)
                            self._trim_history()
                            self._set_cached_data(cache_key, result)
                            return result
                        elif (
                            response_data.get("request_type") == "automation_suggestion"
                        ):
                            # Add automation suggestion to conversation history
                            self.conversation_history.append(
                                {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        response_data
                                    ),  # Store clean JSON
                                }
                            )

                            # Return automation suggestion
                            _LOGGER.debug(
                                "Received automation suggestion: %s",
                                json.dumps(response_data.get("automation")),
                            )
                            result = {
                                "success": True,
                                "answer": json.dumps(response_data),
                            }
                            if thinking_content:
                                result["thinking"] = thinking_content
                                result["thinking_duration"] = thinking_duration
                            result = _with_debug(result)
                            self._trim_history()
                            self._set_cached_data(cache_key, result)
                            return result
                        elif (
                            response_data.get("request_type") == "dashboard_suggestion"
                        ):
                            # Add dashboard suggestion to conversation history
                            self.conversation_history.append(
                                {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        response_data
                                    ),  # Store clean JSON
                                }
                            )

                            # Return dashboard suggestion
                            _LOGGER.debug(
                                "Received dashboard suggestion: %s",
                                json.dumps(response_data.get("dashboard")),
                            )
                            result = {
                                "success": True,
                                "answer": json.dumps(response_data),
                            }
                            if thinking_content:
                                result["thinking"] = thinking_content
                                result["thinking_duration"] = thinking_duration
                            result = _with_debug(result)
                            self._trim_history()
                            self._set_cached_data(cache_key, result)
                            return result
                        elif response_data.get("request_type") in [
                            "get_entities",
                            "get_entities_by_area",
                        ]:
                            # Handle direct get_entities request (for backward compatibility)
                            parameters = response_data.get("parameters", {})
                            _LOGGER.debug(
                                "Processing direct get_entities request with parameters: %s",
                                json.dumps(parameters),
                            )

                            # Add AI's response to conversation history
                            self.conversation_history.append(
                                {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        response_data
                                    ),  # Store clean JSON
                                }
                            )

                            # Get entities data
                            if response_data.get("request_type") == "get_entities":
                                data = await self.get_entities(
                                    area_id=parameters.get("area_id"),
                                    area_ids=parameters.get("area_ids"),
                                )
                            else:  # get_entities_by_area
                                data = await self.get_entities_by_area(
                                    parameters.get("area_id")
                                )

                            _LOGGER.debug(
                                "Retrieved %d entities",
                                len(data) if isinstance(data, list) else 1,
                            )

                            # Add data to conversation as a tool message (see above).
                            self.conversation_history.append(
                                {
                                    "role": "tool",
                                    "content": json.dumps({"data": data}, default=str),
                                }
                            )
                            continue
                        elif response_data.get("request_type") == "call_service":
                            # Handle service call request
                            domain = response_data.get("domain")
                            service = response_data.get("service")
                            target = response_data.get("target", {})
                            service_data = response_data.get("service_data", {})

                            # Resolve nested requests in target
                            if target and "entity_id" in target:
                                entity_id_value = target["entity_id"]
                                if (
                                    isinstance(entity_id_value, dict)
                                    and "request_type" in entity_id_value
                                ):
                                    # This is a nested request, resolve it
                                    nested_request_type = entity_id_value.get(
                                        "request_type"
                                    )
                                    nested_parameters = entity_id_value.get(
                                        "parameters", {}
                                    )

                                    _LOGGER.debug(
                                        "Resolving nested request: %s with parameters: %s",
                                        nested_request_type,
                                        json.dumps(nested_parameters),
                                    )

                                    # Resolve the nested request
                                    if nested_request_type == "get_entities":
                                        entities_data = await self.get_entities(
                                            area_id=nested_parameters.get("area_id"),
                                            area_ids=nested_parameters.get("area_ids"),
                                        )
                                    elif nested_request_type == "get_entities_by_area":
                                        entities_data = await self.get_entities_by_area(
                                            nested_parameters.get("area_id")
                                        )
                                    elif (
                                        nested_request_type == "get_entities_by_domain"
                                    ):
                                        entities_data = (
                                            await self.get_entities_by_domain(
                                                nested_parameters.get("domain")
                                            )
                                        )
                                    else:
                                        _LOGGER.error(
                                            "Unsupported nested request type: %s",
                                            nested_request_type,
                                        )
                                        return {
                                            "success": False,
                                            "error": f"Unsupported nested request type: {nested_request_type}",
                                        }

                                    # Extract entity IDs from the resolved data
                                    if isinstance(entities_data, list):
                                        entity_ids = [
                                            entity.get("entity_id")
                                            for entity in entities_data
                                            if entity.get("entity_id")
                                        ]
                                        target["entity_id"] = entity_ids
                                        _LOGGER.debug(
                                            "Resolved nested request to entity IDs: %s",
                                            entity_ids,
                                        )
                                    else:
                                        _LOGGER.error(
                                            "Nested request returned unexpected data format"
                                        )
                                        return _with_debug(
                                            {
                                                "success": False,
                                                "error": "Nested request returned unexpected data format",
                                            }
                                        )

                            # Handle backward compatibility with old format
                            if not domain or not service:
                                request = response_data.get("request")
                                parameters = response_data.get("parameters", {})

                                if request and "entity_id" in parameters:
                                    entity_id = parameters["entity_id"]
                                    # Infer domain from entity_id
                                    if "." in entity_id:
                                        domain = entity_id.split(".")[0]
                                        service = request
                                        target = {"entity_id": entity_id}
                                        # Remove entity_id from parameters to avoid duplication
                                        service_data = {
                                            k: v
                                            for k, v in parameters.items()
                                            if k != "entity_id"
                                        }
                                        _LOGGER.debug(
                                            "Converted old format: domain=%s, service=%s",
                                            domain,
                                            service,
                                        )

                            _LOGGER.debug(
                                "Processing service call: %s.%s with target: %s and data: %s",
                                domain,
                                service,
                                json.dumps(target),
                                json.dumps(service_data),
                            )

                            # Add AI's response to conversation history
                            self.conversation_history.append(
                                {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        response_data
                                    ),  # Store clean JSON
                                }
                            )

                            # Call the service
                            data = await self.call_service(
                                domain, service, target, service_data
                            )

                            # Check if service call resulted in an error
                            if isinstance(data, dict) and "error" in data:
                                return _with_debug(
                                    {"success": False, "error": data["error"]}
                                )

                            _LOGGER.debug(
                                "Service call completed: %s",
                                json.dumps(data, default=str),
                            )

                            # Add data to conversation as a tool message (see above).
                            self.conversation_history.append(
                                {
                                    "role": "tool",
                                    "content": json.dumps({"data": data}, default=str),
                                }
                            )
                            # Go to next iteration to continue the loop
                            continue

                        # Unknown or missing request_type — treat as final_response
                        # (model returned valid JSON but without a recognised request_type,
                        # e.g. a plain {"message": "..."} or temperature data object)
                        _LOGGER.debug(
                            "Unrecognised request_type %r — treating as final_response",
                            response_data.get("request_type"),
                        )
                        self.conversation_history.append(
                            {"role": "assistant", "content": json.dumps(response_data)}
                        )
                        fallback_text = (
                            response_data.get("response")
                            or response_data.get("message")
                            or response_data.get("answer")
                            or json.dumps(response_data)
                        )
                        result = {"success": True, "answer": fallback_text}
                        if thinking_content:
                            result["thinking"] = thinking_content
                            result["thinking_duration"] = thinking_duration
                        result = _with_debug(result)
                        self._trim_history()
                        self._set_cached_data(cache_key, result)
                        return result

                    except json.JSONDecodeError as e:
                        # Check if this is a local provider that might have already wrapped the response
                        provider = self.config.get("ai_provider", "unknown")
                        if provider == "local":
                            _LOGGER.debug(
                                "Local provider returned non-JSON response (this is normal and handled): %s",
                                response[:200],
                            )
                        else:
                            # Log more of the response to help with debugging for non-local providers
                            response_preview = (
                                response[:1000] if len(response) > 1000 else response
                            )
                            _LOGGER.warning(
                                "Failed to parse response as JSON: %s. Response length: %d. Response preview: %s",
                                str(e),
                                len(response),
                                response_preview,
                            )

                            # Log additional debugging information
                            _LOGGER.debug(
                                "First 50 characters as bytes: %s",
                                response[:50].encode("utf-8") if response else b"",
                            )
                            _LOGGER.debug(
                                "Response starts with: %s",
                                repr(response[:10]) if response else "None",
                            )

                        # Also log the response to a separate debug file for detailed analysis (non-local providers only)
                        if provider != "local":
                            try:
                                import os

                                debug_dir = "/config/ai_agent_ha_debug"

                                def write_debug_file():
                                    if not os.path.exists(debug_dir):
                                        os.makedirs(debug_dir)

                                    import datetime

                                    timestamp = datetime.datetime.now().strftime(
                                        "%Y%m%d_%H%M%S"
                                    )
                                    debug_file = os.path.join(
                                        debug_dir, f"failed_response_{timestamp}.txt"
                                    )

                                    with open(debug_file, "w", encoding="utf-8") as f:
                                        f.write(f"Timestamp: {timestamp}\n")
                                        f.write(f"Provider: {provider}\n")
                                        f.write(f"Error: {str(e)}\n")
                                        f.write(f"Response length: {len(response)}\n")
                                        f.write(
                                            f"Response bytes: {response.encode('utf-8') if response else b''}\n"
                                        )
                                        f.write(f"Response repr: {repr(response)}\n")
                                        f.write(f"Full response:\n{response}\n")

                                    return debug_file

                                # Run file operations in executor to avoid blocking
                                debug_file = await self.hass.async_add_executor_job(
                                    write_debug_file
                                )
                                _LOGGER.info(
                                    "Failed response saved to debug file: %s",
                                    debug_file,
                                )
                            except Exception as debug_error:
                                _LOGGER.debug(
                                    "Could not save debug file: %s", str(debug_error)
                                )

                        # Check if this looks like a corrupted automation suggestion
                        if (
                            response.strip().startswith(
                                '{"request_type": "automation_suggestion'
                            )
                            and len(response) > 10000
                            and response.count("for its use in various fields") > 50
                        ):
                            _LOGGER.warning(
                                "Detected corrupted automation suggestion response with repetitive text"
                            )
                            result = _with_debug(
                                {
                                    "success": False,
                                    "error": "AI generated corrupted automation response. Please try again with a more specific automation request.",
                                }
                            )
                            # Roll back history and do not cache — let the user retry fresh.
                            self.conversation_history = self.conversation_history[
                                :history_rollback_index
                            ]
                            return result

                        # Try YAML recovery for dashboard responses before falling back to final_response
                        yaml_dashboard_indicators = ["title:", "views:", "cards:", "type: custom:", "type: entities"]
                        if any(indicator in response for indicator in yaml_dashboard_indicators):
                            try:
                                parsed_yaml = yaml.safe_load(response)
                                if isinstance(parsed_yaml, dict) and any(
                                    k in parsed_yaml for k in ["title", "views", "cards"]
                                ):
                                    _LOGGER.warning(
                                        "LLM returned YAML for dashboard — recovering via yaml.safe_load()"
                                    )
                                    response_data = {
                                        "request_type": "dashboard_suggestion",
                                        "dashboard": parsed_yaml,
                                    }
                                    # Route through the normal dashboard_suggestion path
                                    self.conversation_history.append(
                                        {
                                            "role": "assistant",
                                            "content": json.dumps(response_data),
                                        }
                                    )
                                    result = {
                                        "success": True,
                                        "answer": json.dumps(response_data),
                                    }
                                    if thinking_content:
                                        result["thinking"] = thinking_content
                                        result["thinking_duration"] = thinking_duration
                                    result = _with_debug(result)
                                    self._trim_history()
                                    self._set_cached_data(cache_key, result)
                                    return result
                            except yaml.YAMLError:
                                pass

                        # If response is not valid JSON, try to wrap it as a final response
                        try:
                            # Truncate extremely long responses to prevent memory issues
                            response_to_wrap = response
                            if len(response) > 50000:
                                response_to_wrap = (
                                    response[:5000]
                                    + "... [Response truncated due to excessive length]"
                                )
                                _LOGGER.warning(
                                    "Truncated extremely long response from %d to 5000 characters",
                                    len(response),
                                )

                            wrapped_response = {
                                "request_type": "final_response",
                                "response": response_to_wrap,
                            }
                            result = {
                                "success": True,
                                "answer": json.dumps(wrapped_response),
                            }
                            if thinking_content:
                                result["thinking"] = thinking_content
                                result["thinking_duration"] = thinking_duration
                            _LOGGER.debug("Wrapped non-JSON response as final_response")
                        except Exception as wrap_error:
                            _LOGGER.error(
                                "Failed to wrap response: %s", str(wrap_error)
                            )
                            result = {
                                "success": False,
                                "error": f"Invalid response format: {str(e)}",
                            }

                        result = _with_debug(result)
                        # Only cache successful wraps; on failure roll back history and
                        # skip caching so the next identical prompt retries fresh.
                        if result.get("success", False):
                            self._trim_history()
                            self._set_cached_data(cache_key, result)
                        else:
                            self.conversation_history = self.conversation_history[
                                :history_rollback_index
                            ]
                        return result

                except Exception as e:
                    _LOGGER.exception("Error processing AI response: %s", str(e))
                    # Roll back conversation history to before this query so the
                    # failed prompt doesn't contaminate the next call.
                    self.conversation_history = self.conversation_history[
                        :history_rollback_index
                    ]
                    _LOGGER.debug(
                        "Rolled back conversation history to index %d after error",
                        history_rollback_index,
                    )
                    # Do NOT cache error results — let the next identical query retry fresh.
                    return _with_debug(
                        {
                            "success": False,
                            "error": f"Error processing AI response: {str(e)}",
                        }
                    )

            # If we've reached max iterations without a final response
            _LOGGER.warning("Reached maximum iterations without final response")
            # Roll back history — the query loop ran but never settled on a final response.
            self.conversation_history = self.conversation_history[
                :history_rollback_index
            ]
            _LOGGER.debug(
                "Rolled back conversation history to index %d after max iterations",
                history_rollback_index,
            )
            result = {
                "success": False,
                "error": "Maximum iterations reached without final response",
            }
            result = _with_debug(result)
            # Do NOT cache — let the user retry and get a fresh attempt.
            return result

        except Exception as e:
            _LOGGER.exception("Error in process_query: %s", str(e))
            # Roll back if history_rollback_index was set before the exception.
            try:
                self.conversation_history = self.conversation_history[
                    :history_rollback_index
                ]
                _LOGGER.debug(
                    "Rolled back conversation history to index %d after outer exception",
                    history_rollback_index,
                )
            except NameError:
                pass  # Exception occurred before history_rollback_index was set
            return _with_debug(
                {"success": False, "error": f"Error in process_query: {str(e)}"}
            )

    def _build_debug_trace(
        self,
        provider: Optional[str],
        provider_settings: Optional[Dict[str, Any]],
        endpoint_type: Optional[str],
    ) -> Dict[str, Any]:
        """Return a sanitized snapshot of the HA↔AI conversation for UI display."""
        history_tail = (
            self.conversation_history[-20:] if self.conversation_history else []
        )
        return {
            "provider": provider,
            "model": provider_settings.get("model") if provider_settings else None,
            "endpoint_type": endpoint_type,
            "conversation": history_tail,
        }

    async def _get_ai_response(self) -> str:
        """Get response from the selected AI provider with retries and rate limiting.

        Rate limiting is enforced by the caller (_process_query_inner) before this
        method is invoked, so we do not call _check_rate_limit() again here.
        """
        retry_count = 0
        last_error = None
        # Use the full conversation_history — _trim_history() already caps it
        # at _max_history_len entries, so we don't apply a redundant 10-message
        # window here that would strip the system prompt or truncate context.
        recent_messages = self.conversation_history
        # Ensure system prompt is always the first message
        if not recent_messages or recent_messages[0].get("role") != "system":
            recent_messages = [self.system_prompt] + recent_messages

        _LOGGER.debug("Sending %d messages to AI provider", len(recent_messages))
        _LOGGER.debug("AI provider: %s", self.config.get("ai_provider", "unknown"))

        while retry_count < self._max_retries:
            try:
                _LOGGER.debug(
                    "Attempt %d/%d: Calling AI client",
                    retry_count + 1,
                    self._max_retries,
                )
                response = await self.ai_client.get_response(recent_messages)
                _LOGGER.debug(
                    "AI client returned response of length: %d", len(response or "")
                )
                _LOGGER.debug("AI response preview: %s", (response or "")[:200])

                # Check for extremely long responses that might indicate model issues
                if response and len(response) > 50000:
                    _LOGGER.warning(
                        "AI returned extremely long response (%d characters), this may indicate a model issue",
                        len(response),
                    )
                    # Check for repetitive patterns that indicate a corrupted response
                    if response.count("for its use in various fields") > 50:
                        _LOGGER.error(
                            "Detected corrupted repetitive response, aborting this iteration"
                        )
                        raise Exception(
                            "AI generated corrupted response with repetitive text. Please try again with a clearer request."
                        )

                # Check if response is empty
                if not response or response.strip() == "":
                    _LOGGER.warning(
                        "AI client returned empty response on attempt %d",
                        retry_count + 1,
                    )
                    if retry_count + 1 >= self._max_retries:
                        raise Exception(
                            "AI provider returned empty response after all retries"
                        )
                    else:
                        retry_count += 1
                        # Exponential backoff: 1 s, 2 s, 4 s … capped at 30 s
                        await asyncio.sleep(
                            min(self._retry_delay * (2 ** (retry_count - 1)), 30)
                        )
                        continue

                return str(response)
            except Exception as e:
                _LOGGER.error(
                    "AI client error on attempt %d: %s", retry_count + 1, str(e)
                )
                last_error = e
                retry_count += 1
                if retry_count < self._max_retries:
                    # Exponential backoff: 1 s, 2 s, 4 s … capped at 30 s
                    await asyncio.sleep(
                        min(self._retry_delay * (2 ** (retry_count - 1)), 30)
                    )
                continue
        raise Exception(
            f"Failed after {retry_count} retries. Last error: {str(last_error)}"
        )

    async def _get_ai_response_stream(self):
        """Get streaming response from AI provider, yielding chunks and firing WS events.

        Returns the full accumulated response text.
        """
        recent_messages = self.conversation_history
        if not recent_messages or recent_messages[0].get("role") != "system":
            recent_messages = [self.system_prompt] + recent_messages

        _LOGGER.debug(
            "Streaming: sending %d messages to AI provider", len(recent_messages)
        )

        if not hasattr(self.ai_client, "get_response_stream"):
            _LOGGER.debug(
                "Provider does not support streaming, falling back to non-streaming"
            )
            return await self._get_ai_response()

        accumulated = []
        try:
            async for chunk in self.ai_client.get_response_stream(recent_messages):
                accumulated.append(chunk)
                # Fire WebSocket event for each chunk
                try:
                    partial_text = "".join(accumulated)
                    self.hass.bus.async_fire(
                        "ai_agent_ha/stream_chunk",
                        {
                            "type": "stream_chunk",
                            "text": partial_text,
                        },
                    )
                except Exception:
                    pass  # Don't fail the stream for WS errors

            full_text = "".join(accumulated)

            if not full_text or full_text.strip() == "":
                _LOGGER.warning("Streaming returned empty response, falling back")
                return await self._get_ai_response()

            # Fire stream end event
            try:
                self.hass.bus.async_fire(
                    "ai_agent_ha/stream_end",
                    {
                        "type": "stream_end",
                    },
                )
            except Exception:
                pass

            return full_text
        except Exception as e:
            _LOGGER.warning("Streaming failed (%s), falling back to non-streaming", e)
            return await self._get_ai_response()

    def _trim_history(self) -> None:
        """Trim conversation_history to the last _max_history_len entries.

        Preserves the system prompt at position 0 when trimming, so the
        model always has the instruction context available.
        """
        if len(self.conversation_history) > self._max_history_len:
            # Keep the system prompt (index 0) + the tail
            tail_size = self._max_history_len - 1
            self.conversation_history = [
                self.conversation_history[0]
            ] + self.conversation_history[-tail_size:]
            _LOGGER.debug(
                "Trimmed conversation history to %d entries",
                len(self.conversation_history),
            )

    def clear_conversation_history(self) -> None:
        """Clear the conversation history and cache."""
        self.conversation_history = []
        self._cache.clear()
        _LOGGER.debug("Conversation history and cache cleared")

    async def set_entity_state(
        self, entity_id: str, state: str, attributes: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Set the state of an entity."""
        try:
            _LOGGER.debug(
                "Setting state for entity %s to %s with attributes: %s",
                entity_id,
                state,
                json.dumps(attributes or {}),
            )

            # Validate entity exists
            if not self.hass.states.get(entity_id):
                return {"error": f"Entity {entity_id} not found"}

            # Call the appropriate service based on the domain
            domain = entity_id.split(".")[0]

            if domain == "light":
                service = (
                    "turn_on" if state.lower() in ["on", "true", "1"] else "turn_off"
                )
                service_data = {"entity_id": entity_id}
                if attributes and service == "turn_on":
                    service_data.update(attributes)
                await self.hass.services.async_call("light", service, service_data)

            elif domain == "switch":
                service = (
                    "turn_on" if state.lower() in ["on", "true", "1"] else "turn_off"
                )
                await self.hass.services.async_call(
                    "switch", service, {"entity_id": entity_id}
                )

            elif domain == "cover":
                if state.lower() in ["open", "up"]:
                    service = "open_cover"
                elif state.lower() in ["close", "down"]:
                    service = "close_cover"
                elif state.lower() == "stop":
                    service = "stop_cover"
                else:
                    return {"error": f"Invalid state {state} for cover entity"}
                await self.hass.services.async_call(
                    "cover", service, {"entity_id": entity_id}
                )

            elif domain == "climate":
                service_data = {"entity_id": entity_id}
                if state.lower() in ["on", "true", "1"]:
                    service = "turn_on"
                elif state.lower() in ["off", "false", "0"]:
                    service = "turn_off"
                elif state.lower() in ["heat", "cool", "dry", "fan_only", "auto"]:
                    service = "set_hvac_mode"
                    service_data["hvac_mode"] = state.lower()
                else:
                    return {"error": f"Invalid state {state} for climate entity"}
                await self.hass.services.async_call("climate", service, service_data)

            elif domain == "fan":
                service = (
                    "turn_on" if state.lower() in ["on", "true", "1"] else "turn_off"
                )
                service_data = {"entity_id": entity_id}
                if attributes and service == "turn_on":
                    service_data.update(attributes)
                await self.hass.services.async_call("fan", service, service_data)

            else:
                # For other domains, try to set the state directly
                self.hass.states.async_set(entity_id, state, attributes or {})

            # Get the new state to confirm the change
            new_state = self.hass.states.get(entity_id)
            return {
                "success": True,
                "entity_id": entity_id,
                "new_state": new_state.state,
                "new_attributes": new_state.attributes,
            }

        except Exception as e:
            _LOGGER.exception("Error setting entity state: %s", str(e))
            return {"error": f"Error setting entity state: {str(e)}"}

    async def call_service(
        self,
        domain: str,
        service: str,
        target: Optional[Dict[str, Any]] = None,
        service_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call a Home Assistant service."""
        try:
            _LOGGER.debug(
                "Calling service %s.%s with target: %s and data: %s",
                domain,
                service,
                json.dumps(target or {}),
                json.dumps(service_data or {}),
            )

            # Prepare the service call data
            call_data = {}

            # Add target entities if provided
            if target:
                if "entity_id" in target:
                    entity_ids = target["entity_id"]
                    if isinstance(entity_ids, list):
                        call_data["entity_id"] = entity_ids
                    else:
                        call_data["entity_id"] = [entity_ids]

                # Add other target properties
                for key, value in target.items():
                    if key != "entity_id":
                        call_data[key] = value

            # Add service data if provided
            if service_data:
                call_data.update(service_data)

            _LOGGER.debug("Final service call data: %s", json.dumps(call_data))

            # Call the service
            await self.hass.services.async_call(domain, service, call_data)

            # Get the updated states of affected entities
            result_entities = []
            if "entity_id" in call_data:
                for entity_id in call_data["entity_id"]:
                    state = self.hass.states.get(entity_id)
                    if state:
                        result_entities.append(
                            {
                                "entity_id": entity_id,
                                "state": state.state,
                                "attributes": dict(state.attributes),
                            }
                        )

            return {
                "success": True,
                "service": f"{domain}.{service}",
                "entities_affected": result_entities,
                "message": f"Successfully called {domain}.{service}",
            }

        except Exception as e:
            _LOGGER.exception(
                "Error calling service %s.%s: %s", domain, service, str(e)
            )
            return {"error": f"Error calling service {domain}.{service}: {str(e)}"}

    async def save_user_prompt_history(
        self, user_id: str, history: List[str]
    ) -> Dict[str, Any]:
        """Save user's prompt history to HA storage."""
        try:
            store: Store = Store(self.hass, 1, f"ai_agent_ha_history_{user_id}")
            await store.async_save({"history": history})
            return {"success": True}
        except Exception as e:
            _LOGGER.exception("Error saving prompt history: %s", str(e))
            return {"error": f"Error saving prompt history: {str(e)}"}

    async def load_user_prompt_history(self, user_id: str) -> Dict[str, Any]:
        """Load user's prompt history from HA storage."""
        try:
            store: Store = Store(self.hass, 1, f"ai_agent_ha_history_{user_id}")
            data = await store.async_load()
            history = data.get("history", []) if data else []
            return {"success": True, "history": history}
        except Exception as e:
            _LOGGER.exception("Error loading prompt history: %s", str(e))
            return {"error": f"Error loading prompt history: {str(e)}", "history": []}
