"""Integration tests for AI Agent HA."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import os

# Add the parent directory to the path for direct imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import homeassistant
    from homeassistant.core import HomeAssistant, ServiceCall
    from homeassistant.config_entries import ConfigEntry

    HOMEASSISTANT_AVAILABLE = True
except ImportError:
    HOMEASSISTANT_AVAILABLE = False
    HomeAssistant = MagicMock
    ServiceCall = MagicMock
    ConfigEntry = MagicMock


class TestIntegration:
    """Test the full integration functionality."""

    @pytest.fixture
    def mock_hass(self):
        """Mock Home Assistant instance with full services."""
        mock = MagicMock()
        mock.data = {}
        mock.services = MagicMock()
        mock.services.async_register = Mock()
        mock.config = MagicMock()
        mock.config.path = MagicMock(return_value="/mock/path")
        mock.bus = MagicMock()
        mock.states = MagicMock()
        mock.http = MagicMock()
        mock.http.async_register_static_paths = AsyncMock()
        return mock

    @pytest.fixture
    def mock_config_entry(self):
        """Mock config entry."""
        mock = MagicMock()
        mock.version = 1
        mock.data = {
            "ai_provider": "openai",
            "openai_token": "test_token",
            "openai_model": "gpt-3.5-turbo",
        }
        return mock

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not HOMEASSISTANT_AVAILABLE, reason="Home Assistant not available"
    )
    async def test_full_integration_setup(self, mock_hass, mock_config_entry):
        """Test the full integration setup process."""
        with patch.dict(
            "sys.modules",
            {
                "homeassistant.components.frontend": MagicMock(),
                "homeassistant.components.http": MagicMock(),
                "homeassistant.helpers.config_validation": MagicMock(),
                "homeassistant.exceptions": MagicMock(),
                "homeassistant.helpers.storage": MagicMock(),
                "voluptuous": MagicMock(),
                "openai": MagicMock(),
            },
        ), patch(
            "custom_components.ai_agent_ha.agent.AiAgentHaAgent"
        ) as mock_agent_class:

            # Mock agent instance
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            from custom_components.ai_agent_ha import async_setup_entry

            result = await async_setup_entry(mock_hass, mock_config_entry)
            assert result is True

            # Verify services were registered
            assert mock_hass.services.async_register.called

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not HOMEASSISTANT_AVAILABLE, reason="Home Assistant not available"
    )
    async def test_service_calls(self, mock_hass, mock_config_entry):
        """Test service call functionality."""
        with patch.dict(
            "sys.modules",
            {
                "homeassistant.components.frontend": MagicMock(),
                "homeassistant.components.http": MagicMock(),
                "homeassistant.helpers.config_validation": MagicMock(),
                "homeassistant.exceptions": MagicMock(),
                "homeassistant.helpers.storage": MagicMock(),
                "voluptuous": MagicMock(),
                "openai": MagicMock(),
            },
        ), patch(
            "custom_components.ai_agent_ha.agent.AiAgentHaAgent"
        ) as mock_agent_class:

            # Mock agent instance with query method
            mock_agent = MagicMock()
            mock_agent.send_query = AsyncMock(return_value="Test response")
            mock_agent_class.return_value = mock_agent

            from custom_components.ai_agent_ha import async_setup_entry

            # Setup the integration
            await async_setup_entry(mock_hass, mock_config_entry)

            # Get the service handler that was registered
            service_calls = [
                call
                for call in mock_hass.services.async_register.call_args_list
                if call[0][1] == "query"
            ]

            assert len(service_calls) > 0, "Query service should be registered"

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not HOMEASSISTANT_AVAILABLE, reason="Home Assistant not available"
    )
    async def test_frontend_panel_registration(self, mock_hass, mock_config_entry):
        """Test frontend panel registration."""
        with patch.dict(
            "sys.modules",
            {
                "homeassistant.components.frontend": MagicMock(),
                "homeassistant.components.http": MagicMock(),
                "homeassistant.helpers.config_validation": MagicMock(),
                "homeassistant.exceptions": MagicMock(),
                "homeassistant.helpers.storage": MagicMock(),
                "voluptuous": MagicMock(),
                "openai": MagicMock(),
            },
        ), patch(
            "custom_components.ai_agent_ha.agent.AiAgentHaAgent"
        ) as mock_agent_class:

            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            from custom_components.ai_agent_ha import async_setup_entry

            result = await async_setup_entry(mock_hass, mock_config_entry)
            assert result is True

            # Verify static paths were registered for frontend
            mock_hass.http.async_register_static_paths.assert_called()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not HOMEASSISTANT_AVAILABLE, reason="Home Assistant not available"
    )
    async def test_unload_entry(self, mock_hass, mock_config_entry):
        """Test unloading the integration."""
        with patch.dict(
            "sys.modules",
            {
                "homeassistant.components.frontend": MagicMock(),
                "homeassistant.components.http": MagicMock(),
                "homeassistant.helpers.config_validation": MagicMock(),
                "homeassistant.exceptions": MagicMock(),
                "homeassistant.helpers.storage": MagicMock(),
                "voluptuous": MagicMock(),
                "openai": MagicMock(),
            },
        ), patch(
            "custom_components.ai_agent_ha.agent.AiAgentHaAgent"
        ) as mock_agent_class:

            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            from custom_components.ai_agent_ha import (
                async_setup_entry,
                async_unload_entry,
            )

            # Setup first
            await async_setup_entry(mock_hass, mock_config_entry)

            # Then unload
            result = await async_unload_entry(mock_hass, mock_config_entry)
            assert result is True

    def test_manifest_validation(self):
        """Test that manifest.json is valid."""
        import json

        manifest_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "custom_components",
            "ai_agent_ha",
            "manifest.json",
        )

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Check required fields
        assert "domain" in manifest
        assert "name" in manifest
        assert "version" in manifest
        assert "requirements" in manifest
        assert "dependencies" in manifest
        assert "config_flow" in manifest
        assert manifest["config_flow"] is True

    def test_services_yaml_validation(self):
        """Test that services.yaml is valid."""
        import yaml

        services_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "custom_components",
            "ai_agent_ha",
            "services.yaml",
        )

        with open(services_path, "r") as f:
            services = yaml.safe_load(f)

        # Check that query service is defined
        assert "query" in services
        assert "description" in services["query"]
        assert "fields" in services["query"]
