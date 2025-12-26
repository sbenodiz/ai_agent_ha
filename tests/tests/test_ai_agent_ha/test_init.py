"""Test for AI Agent HA setup."""

from unittest.mock import patch, MagicMock, AsyncMock
import pytest
import sys
import os

# Add the parent directory to the path for direct imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from homeassistant.core import HomeAssistant
    from homeassistant.setup import async_setup_component
    from homeassistant.config_entries import ConfigEntry

    HOMEASSISTANT_AVAILABLE = True
except ImportError:
    HOMEASSISTANT_AVAILABLE = False
    HomeAssistant = MagicMock
    async_setup_component = MagicMock
    ConfigEntry = MagicMock

# Import const directly to avoid __init__.py issues
try:
    from custom_components.ai_agent_ha.const import DOMAIN
except ImportError:
    # Fallback for local testing
    DOMAIN = "ai_agent_ha"


@pytest.mark.asyncio
@pytest.mark.skipif(not HOMEASSISTANT_AVAILABLE, reason="Home Assistant not available")
async def test_async_setup():
    """Test the basic async_setup function."""
    # Mock all imports to avoid any import issues
    with patch.dict(
        "sys.modules",
        {
            "homeassistant.components.frontend": MagicMock(),
            "homeassistant.components.http": MagicMock(),
            "homeassistant.helpers.config_validation": MagicMock(),
            "voluptuous": MagicMock(),
            "homeassistant.exceptions": MagicMock(),
            "homeassistant.helpers.storage": MagicMock(),
        },
    ):
        from custom_components.ai_agent_ha import async_setup

        mock_hass = MagicMock()
        mock_config = MagicMock()

        result = await async_setup(mock_hass, mock_config)
        assert result is True


@pytest.mark.asyncio
@pytest.mark.skipif(not HOMEASSISTANT_AVAILABLE, reason="Home Assistant not available")
async def test_setup_entry():
    """Test setting up an entry."""
    # Mock all imports comprehensively
    with patch.dict(
        "sys.modules",
        {
            "homeassistant.components.frontend": MagicMock(),
            "homeassistant.components.http": MagicMock(),
            "homeassistant.helpers.config_validation": MagicMock(),
            "homeassistant.exceptions": MagicMock(),
            "homeassistant.helpers.storage": MagicMock(),
            "voluptuous": MagicMock(),
        },
    ), patch("custom_components.ai_agent_ha.agent.AiAgentHaAgent") as mock_agent:

        from custom_components.ai_agent_ha import async_setup_entry

        # Create mock hass and entry
        mock_hass = MagicMock()
        mock_hass.data = {}
        mock_hass.services = MagicMock()
        mock_hass.http = MagicMock()
        mock_hass.http.async_register_static_paths = AsyncMock()
        mock_hass.config = MagicMock()
        mock_hass.config.path = MagicMock(return_value="/mock/path")
        mock_hass.bus = MagicMock()

        mock_entry = MagicMock()
        mock_entry.version = 1
        mock_entry.data = {"ai_provider": "openai", "openai_token": "fake_token"}

        # The function should return True
        result = await async_setup_entry(mock_hass, mock_entry)
        assert result is True

        # Just verify the function completes successfully
        # The agent creation depends on complex Home Assistant internals
        # that are difficult to mock completely in unit tests
