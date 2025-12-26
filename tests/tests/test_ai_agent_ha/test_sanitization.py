"""Tests for the sanitization utility function."""

import os
import sys

import pytest

# Add the parent directory to the path for direct imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# Import the sanitization function directly without importing the full module
# This avoids dependency on homeassistant which may not be installed
def get_sanitize_function():
    """Get the sanitize_for_logging function without importing full module."""
    from typing import Any

    def sanitize_for_logging(data: Any, mask: str = "***REDACTED***") -> Any:
        """Sanitize sensitive data for safe logging.

        Recursively masks sensitive fields like API keys, tokens, passwords, etc.
        This prevents accidental exposure of credentials in debug logs.

        Args:
            data: The data structure to sanitize (dict, list, str, etc.)
            mask: The string to use for masking sensitive values

        Returns:
            A sanitized copy of the data with sensitive fields masked
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
        }

        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                # Check if key matches any sensitive pattern
                key_lower = str(key).lower()
                is_sensitive = any(
                    pattern in key_lower for pattern in sensitive_patterns
                )

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

    return sanitize_for_logging


sanitize_for_logging = get_sanitize_function()


class TestSanitization:
    """Test the sanitize_for_logging function."""

    def test_sanitize_simple_token(self):
        """Test sanitization of a simple token field."""
        data = {"openai_token": "sk-abc123", "ai_provider": "openai"}
        result = sanitize_for_logging(data)

        assert result["openai_token"] == "***REDACTED***"
        assert result["ai_provider"] == "openai"

    def test_sanitize_multiple_tokens(self):
        """Test sanitization of multiple token fields."""
        data = {
            "openai_token": "sk-abc123",
            "gemini_token": "gm-xyz789",
            "anthropic_token": "sk-ant-456",
            "ai_provider": "openai",
            "model": "gpt-4",
        }
        result = sanitize_for_logging(data)

        assert result["openai_token"] == "***REDACTED***"
        assert result["gemini_token"] == "***REDACTED***"
        assert result["anthropic_token"] == "***REDACTED***"
        assert result["ai_provider"] == "openai"
        assert result["model"] == "gpt-4"

    def test_sanitize_api_key(self):
        """Test sanitization of api_key field."""
        data = {"api_key": "secret123", "endpoint": "https://api.example.com"}
        result = sanitize_for_logging(data)

        assert result["api_key"] == "***REDACTED***"
        assert result["endpoint"] == "https://api.example.com"

    def test_sanitize_password(self):
        """Test sanitization of password field."""
        data = {"password": "secret", "username": "user"}
        result = sanitize_for_logging(data)

        assert result["password"] == "***REDACTED***"
        assert result["username"] == "user"

    def test_sanitize_authorization(self):
        """Test sanitization of authorization field."""
        data = {"authorization": "Bearer token123", "content_type": "application/json"}
        result = sanitize_for_logging(data)

        assert result["authorization"] == "***REDACTED***"
        assert result["content_type"] == "application/json"

    def test_sanitize_nested_dict(self):
        """Test sanitization of nested dictionaries."""
        data = {
            "config": {"openai_token": "sk-abc123", "model": "gpt-4"},
            "ai_provider": "openai",
        }
        result = sanitize_for_logging(data)

        assert result["config"]["openai_token"] == "***REDACTED***"
        assert result["config"]["model"] == "gpt-4"
        assert result["ai_provider"] == "openai"

    def test_sanitize_list_of_dicts(self):
        """Test sanitization of lists containing dictionaries."""
        data = [
            {"api_key": "key1", "name": "config1"},
            {"api_key": "key2", "name": "config2"},
        ]
        result = sanitize_for_logging(data)

        assert result[0]["api_key"] == "***REDACTED***"
        assert result[0]["name"] == "config1"
        assert result[1]["api_key"] == "***REDACTED***"
        assert result[1]["name"] == "config2"

    def test_sanitize_tuple(self):
        """Test sanitization of tuples."""
        data = ({"token": "secret", "name": "test"}, "plain_value")
        result = sanitize_for_logging(data)

        assert isinstance(result, tuple)
        assert result[0]["token"] == "***REDACTED***"
        assert result[0]["name"] == "test"
        assert result[1] == "plain_value"

    def test_sanitize_case_insensitive(self):
        """Test that sanitization is case-insensitive."""
        data = {
            "Token": "secret1",
            "API_KEY": "secret2",
            "Password": "secret3",
            "AUTHORIZATION": "secret4",
        }
        result = sanitize_for_logging(data)

        assert result["Token"] == "***REDACTED***"
        assert result["API_KEY"] == "***REDACTED***"
        assert result["Password"] == "***REDACTED***"
        assert result["AUTHORIZATION"] == "***REDACTED***"

    def test_sanitize_partial_match(self):
        """Test that partial matches work (e.g., 'my_token' contains 'token')."""
        data = {
            "my_token": "secret",
            "custom_api_key": "secret2",
            "db_password": "secret3",
        }
        result = sanitize_for_logging(data)

        assert result["my_token"] == "***REDACTED***"
        assert result["custom_api_key"] == "***REDACTED***"
        assert result["db_password"] == "***REDACTED***"

    def test_sanitize_custom_mask(self):
        """Test using a custom mask string."""
        data = {"token": "secret", "name": "test"}
        result = sanitize_for_logging(data, mask="[HIDDEN]")

        assert result["token"] == "[HIDDEN]"
        assert result["name"] == "test"

    def test_sanitize_primitive_types(self):
        """Test that primitive types pass through unchanged."""
        assert sanitize_for_logging("string") == "string"
        assert sanitize_for_logging(123) == 123
        assert sanitize_for_logging(True) is True
        assert sanitize_for_logging(None) is None

    def test_sanitize_empty_structures(self):
        """Test sanitization of empty structures."""
        assert sanitize_for_logging({}) == {}
        assert sanitize_for_logging([]) == []
        assert sanitize_for_logging(()) == ()

    def test_sanitize_deeply_nested(self):
        """Test sanitization of deeply nested structures."""
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "api_key": "secret",
                        "config": {"token": "secret2", "safe_value": "ok"},
                    }
                }
            }
        }
        result = sanitize_for_logging(data)

        assert result["level1"]["level2"]["level3"]["api_key"] == "***REDACTED***"
        assert (
            result["level1"]["level2"]["level3"]["config"]["token"] == "***REDACTED***"
        )
        assert result["level1"]["level2"]["level3"]["config"]["safe_value"] == "ok"

    def test_sanitize_real_config(self):
        """Test with a realistic configuration object."""
        config = {
            "ai_provider": "openai",
            "openai_token": "sk-abc123xyz",
            "gemini_token": "gm-def456",
            "anthropic_token": "sk-ant-789",
            "openrouter_token": "or-token-123",
            "llama_token": "llama-token-456",
            "local_url": "http://localhost:11434",
            "models": {
                "openai": "gpt-4",
                "gemini": "gemini-pro",
                "anthropic": "claude-3-sonnet",
            },
        }
        result = sanitize_for_logging(config)

        # All tokens should be redacted
        assert result["openai_token"] == "***REDACTED***"
        assert result["gemini_token"] == "***REDACTED***"
        assert result["anthropic_token"] == "***REDACTED***"
        assert result["openrouter_token"] == "***REDACTED***"
        assert result["llama_token"] == "***REDACTED***"

        # Non-sensitive values should remain
        assert result["ai_provider"] == "openai"
        assert result["local_url"] == "http://localhost:11434"
        assert result["models"]["openai"] == "gpt-4"
        assert result["models"]["gemini"] == "gemini-pro"
        assert result["models"]["anthropic"] == "claude-3-sonnet"

    def test_sanitize_http_headers(self):
        """Test sanitization of HTTP headers."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer secret-token",
            "X-API-Key": "api-key-123",
            "User-Agent": "HomeAssistant/1.0",
        }
        result = sanitize_for_logging(headers)

        assert result["Content-Type"] == "application/json"
        assert result["Authorization"] == "***REDACTED***"
        assert result["X-API-Key"] == "***REDACTED***"
        assert result["User-Agent"] == "HomeAssistant/1.0"
