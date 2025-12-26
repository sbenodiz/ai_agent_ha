"""Fixtures for AI Agent HA tests."""

import asyncio
from unittest.mock import patch, MagicMock

import pytest

try:
    from homeassistant.core import HomeAssistant
    from homeassistant.setup import async_setup_component

    HOMEASSISTANT_AVAILABLE = True
except ImportError:
    HOMEASSISTANT_AVAILABLE = False
    # Mock Home Assistant classes for local testing
    HomeAssistant = MagicMock
    async_setup_component = MagicMock


@pytest.fixture
async def hass():
    """Return a Home Assistant instance for testing."""
    if not HOMEASSISTANT_AVAILABLE:
        # Return a mock for local testing
        mock_hass = MagicMock()
        mock_hass.data = {}
        mock_hass.services = MagicMock()
        mock_hass.config = MagicMock()
        mock_hass.config.components = set()
        yield mock_hass
        return

    hass = HomeAssistant()
    hass.config.components.add("persistent_notification")

    # Start Home Assistant
    await hass.async_start()

    yield hass

    # Stop Home Assistant
    await hass.async_stop()


@pytest.fixture
def mock_agent():
    """Mock the AI Agent."""
    with patch("custom_components.ai_agent_ha.agent.AiAgentHaAgent") as mock:
        yield mock
