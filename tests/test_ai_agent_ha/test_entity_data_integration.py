"""Integration tests for entity/data retrieval methods on AiAgentHaAgent.

These tests instantiate the real ``AiAgentHaAgent`` and exercise its
data-retrieval methods against a mocked Home Assistant interface
(``hass.states``, ``hass.data``, registries, recorder helpers).
"""

import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Make the repo root importable so ``custom_components.ai_agent_ha`` resolves
# regardless of where pytest is invoked from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import homeassistant  # noqa: F401

    from custom_components.ai_agent_ha.agent import AiAgentHaAgent

    AGENT_AVAILABLE = True
except Exception:  # pragma: no cover - environment guard
    AGENT_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not AGENT_AVAILABLE,
    reason="Home Assistant or agent module not importable",
)


def _make_state(entity_id, state_value="on", attributes=None, last_changed=None):
    """Build a fake hass State object with the attributes the agent reads."""
    s = MagicMock()
    s.entity_id = entity_id
    s.state = state_value
    s.attributes = attributes or {}
    s.last_changed = last_changed or datetime(
        2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc
    )
    s.last_updated = s.last_changed
    s.domain = entity_id.split(".", 1)[0]
    return s


@pytest.fixture
def mock_hass():
    h = MagicMock()
    h.data = {}
    h.config = MagicMock()
    h.config.path = MagicMock(side_effect=lambda *parts: "/mock/" + "/".join(parts))
    h.services = MagicMock()
    h.services.async_call = AsyncMock(return_value=None)
    h.bus = MagicMock()
    h.states = MagicMock()
    h.states.async_all = MagicMock(return_value=[])
    h.states.get = MagicMock(return_value=None)

    async def _exec(func, *args, **kwargs):
        return func(*args, **kwargs)

    h.async_add_executor_job = AsyncMock(side_effect=_exec)
    return h


@pytest.fixture
def agent_config():
    return {"ai_provider": "openai", "openai_token": "sk-test-not-real"}


@pytest.fixture
def agent(mock_hass, agent_config):
    return AiAgentHaAgent(mock_hass, agent_config)


class TestEntityDataRetrieval:
    """Integration-style tests for entity/data retrieval methods."""

    @pytest.mark.asyncio
    async def test_get_entities_by_domain(self, agent, mock_hass):
        states = [
            _make_state("light.kitchen", "on", {"friendly_name": "Kitchen Light"}),
            _make_state("light.living", "off", {"friendly_name": "Living Light"}),
            _make_state("switch.garage", "on"),
            _make_state("sensor.temperature", "21.5"),
        ]
        mock_hass.states.async_all = MagicMock(return_value=states)
        mock_hass.states.get = MagicMock(
            side_effect=lambda eid: next(
                (s for s in states if s.entity_id == eid), None
            )
        )

        lights = await agent.get_entities_by_domain("light")
        assert len(lights) == 2
        assert {e["entity_id"] for e in lights} == {
            "light.kitchen",
            "light.living",
        }
        assert all(e["entity_id"].startswith("light.") for e in lights)

        switches = await agent.get_entities_by_domain("switch")
        assert len(switches) == 1
        assert switches[0]["entity_id"] == "switch.garage"

        nothing = await agent.get_entities_by_domain("doesnotexist")
        assert nothing == []

    @pytest.mark.asyncio
    async def test_get_entity_state(self, agent, mock_hass):
        s = _make_state(
            "light.kitchen",
            "on",
            {"friendly_name": "Kitchen", "brightness": 200},
        )
        mock_hass.states.get = MagicMock(
            side_effect=lambda eid: s if eid == "light.kitchen" else None
        )

        result = await agent.get_entity_state("light.kitchen")
        assert result["entity_id"] == "light.kitchen"
        assert result["state"] == "on"
        assert result["friendly_name"] == "Kitchen"
        assert result["attributes"]["brightness"] == 200
        assert "last_changed" in result

        missing = await agent.get_entity_state("light.unknown")
        assert "error" in missing
        assert "light.unknown" in missing["error"]

    @pytest.mark.asyncio
    async def test_get_entities_by_area(self, agent, mock_hass):
        kitchen_light = MagicMock(
            entity_id="light.kitchen", area_id="kitchen", device_id=None
        )
        living_light = MagicMock(
            entity_id="light.living", area_id="living", device_id=None
        )
        sensor_no_area = MagicMock(
            entity_id="sensor.x", area_id=None, device_id=None
        )

        entity_reg = MagicMock()
        entity_reg.entities = {
            "light.kitchen": kitchen_light,
            "light.living": living_light,
            "sensor.x": sensor_no_area,
        }
        device_reg = MagicMock()
        device_reg.devices = {}

        states_by_id = {
            "light.kitchen": _make_state(
                "light.kitchen", "on", {"friendly_name": "Kitchen"}
            ),
            "light.living": _make_state("light.living", "off"),
        }
        mock_hass.states.get = MagicMock(
            side_effect=lambda eid: states_by_id.get(eid)
        )

        with patch(
            "homeassistant.helpers.entity_registry.async_get",
            return_value=entity_reg,
        ), patch(
            "homeassistant.helpers.device_registry.async_get",
            return_value=device_reg,
        ), patch(
            "homeassistant.helpers.area_registry.async_get",
            return_value=MagicMock(),
        ):
            result = await agent.get_entities_by_area("kitchen")

        assert len(result) == 1
        assert result[0]["entity_id"] == "light.kitchen"

    @pytest.mark.asyncio
    async def test_get_entities_with_area_filter(self, agent, mock_hass):
        kitchen_light = MagicMock(
            entity_id="light.kitchen", area_id="kitchen", device_id=None
        )
        living_light = MagicMock(
            entity_id="light.living", area_id="living", device_id=None
        )
        garage_switch = MagicMock(
            entity_id="switch.garage", area_id="garage", device_id=None
        )

        entity_reg = MagicMock()
        entity_reg.entities = {
            e.entity_id: e
            for e in (kitchen_light, living_light, garage_switch)
        }
        device_reg = MagicMock()
        device_reg.devices = {}

        all_states = {
            "light.kitchen": _make_state("light.kitchen", "on"),
            "light.living": _make_state("light.living", "off"),
            "switch.garage": _make_state("switch.garage", "on"),
        }
        mock_hass.states.get = MagicMock(
            side_effect=lambda eid: all_states.get(eid)
        )

        with patch(
            "homeassistant.helpers.entity_registry.async_get",
            return_value=entity_reg,
        ), patch(
            "homeassistant.helpers.device_registry.async_get",
            return_value=device_reg,
        ), patch(
            "homeassistant.helpers.area_registry.async_get",
            return_value=MagicMock(),
        ):
            single = await agent.get_entities(area_id="kitchen")
            assert len(single) == 1
            assert single[0]["entity_id"] == "light.kitchen"

            multi = await agent.get_entities(area_ids=["kitchen", "garage"])
            ids = {e["entity_id"] for e in multi if "entity_id" in e}
            assert ids == {"light.kitchen", "switch.garage"}

        nothing = await agent.get_entities()
        assert nothing == [{"error": "No area_id or area_ids provided"}]

    @pytest.mark.asyncio
    async def test_get_calendar_events(self, agent, mock_hass):
        states = [
            _make_state("calendar.work", "on", {"friendly_name": "Work"}),
            _make_state("calendar.personal", "off"),
            _make_state("light.kitchen", "on"),
        ]
        mock_hass.states.async_all = MagicMock(return_value=states)
        mock_hass.states.get = MagicMock(
            side_effect=lambda eid: next(
                (s for s in states if s.entity_id == eid), None
            )
        )

        result = await agent.get_calendar_events()
        assert len(result) == 2
        assert all(e["entity_id"].startswith("calendar.") for e in result)

        single = await agent.get_calendar_events(entity_id="calendar.work")
        assert len(single) == 1
        assert single[0]["entity_id"] == "calendar.work"

    @pytest.mark.asyncio
    async def test_get_automations(self, agent, mock_hass):
        states = [
            _make_state("automation.morning", "on"),
            _make_state("automation.evening", "off"),
            _make_state("light.kitchen", "on"),
            _make_state("switch.garage", "on"),
        ]
        mock_hass.states.async_all = MagicMock(return_value=states)
        mock_hass.states.get = MagicMock(
            side_effect=lambda eid: next(
                (s for s in states if s.entity_id == eid), None
            )
        )

        result = await agent.get_automations()
        assert len(result) == 2
        assert all(e["entity_id"].startswith("automation.") for e in result)

    @pytest.mark.asyncio
    async def test_get_history(self, agent, mock_hass):
        s1 = _make_state(
            "sensor.temperature",
            "21.5",
            last_changed=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
        s2 = _make_state(
            "sensor.temperature",
            "22.0",
            last_changed=datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc),
        )
        history_data = {"sensor.temperature": [s1, s2]}

        with patch(
            "homeassistant.components.recorder.history.get_significant_states",
            return_value=history_data,
        ):
            result = await agent.get_history("sensor.temperature", hours=24)

        assert len(result) == 2
        assert all(item["entity_id"] == "sensor.temperature" for item in result)
        assert "last_changed" in result[0]
        assert "last_updated" in result[0]

    @pytest.mark.asyncio
    async def test_get_statistics(self, agent, mock_hass):
        from homeassistant.components import recorder

        # Mark recorder as available so the early-return branch is skipped.
        mock_hass.data[recorder.DATA_INSTANCE] = MagicMock()

        stats_data = {
            "sensor.energy": [
                {
                    "start": "2024-01-01T00:00:00Z",
                    "mean": 100.0,
                    "min": 90.0,
                    "max": 110.0,
                    "last_reset": None,
                    "state": 105.0,
                    "sum": 1000.0,
                }
            ]
        }
        with patch(
            "homeassistant.components.recorder.statistics.get_last_short_term_statistics",
            return_value=stats_data,
        ):
            result = await agent.get_statistics("sensor.energy")
        assert result["entity_id"] == "sensor.energy"
        assert result["mean"] == 100.0
        assert result["max"] == 110.0

        with patch(
            "homeassistant.components.recorder.statistics.get_last_short_term_statistics",
            return_value={},
        ):
            empty = await agent.get_statistics("sensor.unknown")
        assert "error" in empty

    @pytest.mark.asyncio
    async def test_get_weather_data(self, agent, mock_hass):
        weather_state = _make_state(
            "weather.home",
            "sunny",
            {
                "friendly_name": "Home",
                "temperature": 22.5,
                "humidity": 60,
                "pressure": 1013,
                "wind_speed": 5.0,
                "wind_bearing": 180,
                "forecast": [
                    {
                        "datetime": "2024-01-02T12:00:00Z",
                        "temperature": 20.0,
                        "condition": "rainy",
                    },
                    {
                        "datetime": "2024-01-03T12:00:00Z",
                        "temperature": 18.0,
                        "condition": "cloudy",
                    },
                ],
            },
        )
        light_state = _make_state("light.kitchen", "on")
        mock_hass.states.async_all = MagicMock(
            return_value=[weather_state, light_state]
        )

        result = await agent.get_weather_data()
        assert "current" in result
        assert result["current"]["entity_id"] == "weather.home"
        assert result["current"]["temperature"] == 22.5
        assert result["current"]["forecast_available"] is True
        assert len(result["forecast"]) == 2

        mock_hass.states.async_all = MagicMock(return_value=[light_state])
        empty = await agent.get_weather_data()
        assert "error" in empty
