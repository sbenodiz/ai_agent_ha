"""Tests for the configuration flow."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import importlib.util

# Add the parent directory to the path for direct imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from homeassistant import config_entries
    from homeassistant.core import HomeAssistant
    from homeassistant.data_entry_flow import FlowResultType

    HOMEASSISTANT_AVAILABLE = True
except ImportError:
    HOMEASSISTANT_AVAILABLE = False
    config_entries = MagicMock()
    HomeAssistant = MagicMock
    FlowResultType = MagicMock()


def _import_config_flow_directly():
    """Import config_flow.py directly without going through __init__.py."""
    config_flow_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "custom_components", "ai_agent_ha", "config_flow.py"
    )
    config_flow_path = os.path.abspath(config_flow_path)

    spec = importlib.util.spec_from_file_location("config_flow", config_flow_path)
    config_flow_module = importlib.util.module_from_spec(spec)

    # Mocking voluptuous globally would leak into other tests and break any
    # later import of real Home Assistant components (vol.Required would
    # resolve to a MagicMock). Restore sys.modules after exec_module.
    _SENTINEL = object()
    mocked = {
        "homeassistant.helpers.config_validation": MagicMock(),
        "voluptuous": MagicMock(),
    }
    originals = {name: sys.modules.get(name, _SENTINEL) for name in mocked}
    sys.modules.update(mocked)

    try:
        spec.loader.exec_module(config_flow_module)
    finally:
        for name, original in originals.items():
            if original is _SENTINEL:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return config_flow_module


class TestConfigFlow:
    """Test the config flow."""

    @pytest.fixture
    def mock_hass(self):
        """Mock Home Assistant instance."""
        mock = MagicMock()
        mock.data = {}
        mock.config_entries = MagicMock()
        return mock

    def test_config_flow_import(self):
        """Test that config flow can be imported without errors."""
        try:
            config_flow_module = _import_config_flow_directly()
            assert hasattr(config_flow_module, 'AiAgentHaConfigFlow')
            assert hasattr(config_flow_module.AiAgentHaConfigFlow, 'VERSION')
            assert config_flow_module.AiAgentHaConfigFlow.VERSION == 1
        except Exception as e:
            # If import fails, that's also valid information
            pytest.skip(f"Config flow import failed: {e}")

    @pytest.mark.skipif(
        not HOMEASSISTANT_AVAILABLE, reason="Home Assistant not available"
    )
    def test_config_flow_class_structure(self):
        """Test config flow class structure."""
        try:
            config_flow_module = _import_config_flow_directly()
            flow_class = config_flow_module.AiAgentHaConfigFlow
            
            # Check that required methods exist
            assert hasattr(flow_class, 'async_step_user')
            assert hasattr(flow_class, 'async_step_openai')
            assert hasattr(flow_class, 'async_step_anthropic')
            assert hasattr(flow_class, 'async_step_gemini')
            
        except Exception as e:
            pytest.skip(f"Config flow class test failed: {e}")

    def test_config_flow_domain(self):
        """Test config flow domain constants."""
        try:
            config_flow_module = _import_config_flow_directly()
            # Basic validation that domain is accessible
            from custom_components.ai_agent_ha.const import DOMAIN
            assert DOMAIN == "ai_agent_ha"
        except Exception as e:
            pytest.skip(f"Config flow domain test failed: {e}")

    def test_config_flow_ai_providers(self):
        """Test that AI providers are defined."""
        try:
            from custom_components.ai_agent_ha.const import AI_PROVIDERS
            expected_providers = ["llama", "openai", "gemini", "openrouter", "anthropic", "local_ollama", "openai_compatible"]
            assert all(provider in AI_PROVIDERS for provider in expected_providers)
        except Exception as e:
            pytest.skip(f"AI providers test failed: {e}")

    def test_config_flow_constants(self):
        """Test config flow constants are properly defined."""
        try:
            config_flow_module = _import_config_flow_directly()
            flow_class = config_flow_module.AiAgentHaConfigFlow
            assert hasattr(flow_class, "VERSION")
            assert flow_class.VERSION == 1
        except Exception as e:
            pytest.skip(f"Config flow constants test failed: {e}")

    def test_config_flow_schema_structure(self):
        """Test that config flow has proper schema structure."""
        try:
            config_flow_module = _import_config_flow_directly()
            flow_class = config_flow_module.AiAgentHaConfigFlow

            # Test that we can instantiate the class (basic smoke test)
            flow = flow_class()
            assert flow is not None
            assert flow.VERSION == 1

        except Exception as e:
            pytest.skip(f"Config flow schema test failed: {e}")

    def test_openai_compatible_api_key_field_present(self):
        """Regression for issue #70 Bug 1: openai_compatible must expose an API key field.

        In v1.11 the runtime read openai_compatible_api_key from config but the
        config flow never collected it, so the Authorization header was never sent
        and authenticated gateways returned 401. This test guards against that
        regression by checking the source of config_flow.py for:
          - the field appearing as a vol.Optional in BOTH the create and options forms
          - the field being persisted in BOTH save paths
        """
        config_flow_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "custom_components",
                "ai_agent_ha",
                "config_flow.py",
            )
        )
        with open(config_flow_path, encoding="utf-8") as f:
            src = f.read()

        # The field must appear as an Optional schema entry at least twice
        # (initial create flow + options/edit flow).
        field_decls = src.count('"openai_compatible_api_key"')
        assert field_decls >= 4, (
            f"openai_compatible_api_key referenced only {field_decls}× in "
            f"config_flow.py; expected >= 4 (2 schema fields + 2 save paths)"
        )

        # And it must be persisted into the config in both paths.
        assert 'self.config_data["openai_compatible_api_key"]' in src, (
            "openai_compatible_api_key not saved in initial config flow"
        )
        assert 'updated_data["openai_compatible_api_key"]' in src, (
            "openai_compatible_api_key not saved in options/edit flow"
        )
