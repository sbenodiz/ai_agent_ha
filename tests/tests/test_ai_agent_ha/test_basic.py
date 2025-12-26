"""Basic tests that don't require Home Assistant."""

import pytest
import sys
import os
import importlib.util
from unittest.mock import MagicMock, patch


def _import_const_directly():
    """Import const.py directly without going through __init__.py."""
    const_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "custom_components",
        "ai_agent_ha",
        "const.py",
    )
    const_path = os.path.abspath(const_path)

    spec = importlib.util.spec_from_file_location("const", const_path)
    const_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(const_module)
    return const_module


def test_domain_constant():
    """Test that the domain constant is defined correctly."""
    const = _import_const_directly()
    assert const.DOMAIN == "ai_agent_ha"


def test_ai_providers_constant():
    """Test that AI providers are defined correctly."""
    const = _import_const_directly()
    assert isinstance(const.AI_PROVIDERS, list)
    assert len(const.AI_PROVIDERS) > 0
    assert "openai" in const.AI_PROVIDERS
    assert "anthropic" in const.AI_PROVIDERS


def test_version_constant():
    """Test version handling."""
    # This test will work in CI where homeassistant is available
    if "homeassistant" in sys.modules or _homeassistant_available():
        try:
            # Only test config flow if homeassistant is available
            from custom_components.ai_agent_ha.config_flow import AiAgentHaConfigFlow

            assert hasattr(AiAgentHaConfigFlow, "VERSION")
            assert AiAgentHaConfigFlow.VERSION == 1
        except ImportError:
            pytest.skip("Home Assistant not available")
    else:
        pytest.skip("Home Assistant not available for local testing")


def _homeassistant_available():
    """Check if homeassistant is available."""
    try:
        import homeassistant

        return True
    except ImportError:
        return False


def test_basic_functionality():
    """Test basic functionality that doesn't require Home Assistant."""
    const = _import_const_directly()

    # Basic validation
    assert isinstance(const.DOMAIN, str)
    assert len(const.DOMAIN) > 0
    assert isinstance(const.AI_PROVIDERS, list)
    assert len(const.AI_PROVIDERS) > 0
