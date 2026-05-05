"""Integration tests for registry and dashboard methods on AiAgentHaAgent."""

import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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


def _make_state(entity_id, state_value="on", attributes=None):
    s = MagicMock()
    s.entity_id = entity_id
    s.state = state_value
    s.attributes = attributes or {}
    s.last_changed = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    s.last_updated = s.last_changed
    s.domain = entity_id.split(".", 1)[0]
    return s


@pytest.fixture
def mock_hass(tmp_path):
    h = MagicMock()
    h.data = {}
    h.config = MagicMock()
    # Use a real tmp dir so dashboard create/update writes to inspectable files.
    h.config.path = MagicMock(
        side_effect=lambda *parts: str(tmp_path.joinpath(*parts))
    )
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


class TestRegistryMethods:
    """Integration tests for the registry-retrieval helpers."""

    @pytest.mark.asyncio
    async def test_get_entity_registry(self, agent, mock_hass):
        e1 = MagicMock(
            entity_id="light.kitchen",
            device_id="dev1",
            platform="hue",
            disabled=False,
            area_id="kitchen",
            original_name="Hue Bulb",
            unique_id="abc123",
        )
        e2 = MagicMock(
            entity_id="sensor.temp",
            device_id=None,
            platform="mqtt",
            disabled=False,
            area_id=None,
            original_name="Temp",
            unique_id="def456",
        )

        ent_reg = MagicMock()
        ent_reg.entities = {e1.entity_id: e1, e2.entity_id: e2}

        kitchen_area = MagicMock()
        kitchen_area.name = "Kitchen"
        area_reg = MagicMock()
        area_reg.async_get_area = MagicMock(
            side_effect=lambda aid: kitchen_area if aid == "kitchen" else None
        )

        dev_reg = MagicMock()
        dev_reg.async_get = MagicMock(return_value=None)

        light_state = _make_state(
            "light.kitchen", "on", {"device_class": "light"}
        )
        sensor_state = _make_state(
            "sensor.temp",
            "21.5",
            {
                "device_class": "temperature",
                "state_class": "measurement",
                "unit_of_measurement": "C",
            },
        )
        mock_hass.states.get = MagicMock(
            side_effect=lambda eid: {
                "light.kitchen": light_state,
                "sensor.temp": sensor_state,
            }.get(eid)
        )

        with patch(
            "homeassistant.helpers.entity_registry.async_get",
            return_value=ent_reg,
        ), patch(
            "homeassistant.helpers.device_registry.async_get",
            return_value=dev_reg,
        ), patch(
            "homeassistant.helpers.area_registry.async_get",
            return_value=area_reg,
        ):
            result = await agent.get_entity_registry()

        assert len(result) == 2
        by_id = {r["entity_id"]: r for r in result}
        assert by_id["light.kitchen"]["area_id"] == "kitchen"
        assert by_id["light.kitchen"]["area_name"] == "Kitchen"
        assert by_id["sensor.temp"]["device_class"] == "temperature"
        assert by_id["sensor.temp"]["unit_of_measurement"] == "C"

        with patch(
            "homeassistant.helpers.entity_registry.async_get",
            return_value=None,
        ), patch(
            "homeassistant.helpers.device_registry.async_get",
            return_value=None,
        ), patch(
            "homeassistant.helpers.area_registry.async_get", return_value=None
        ):
            assert await agent.get_entity_registry() == []

    @pytest.mark.asyncio
    async def test_get_device_registry(self, agent):
        d1 = MagicMock(
            id="dev1",
            name="Hue Bridge",
            model="Bridge v2",
            manufacturer="Philips",
            sw_version="1.0",
            hw_version=None,
            connections=set(),
            identifiers={("hue", "abc")},
            area_id="living",
            disabled_by=None,
            entry_type=None,
            name_by_user=None,
        )
        d2 = MagicMock(
            id="dev2",
            name="Sensor",
            model="X",
            manufacturer="MQTT",
            sw_version="2.0",
            hw_version="rev1",
            connections={("mac", "00:11")},
            identifiers={("mqtt", "x")},
            area_id=None,
            disabled_by="user",
            entry_type=None,
            name_by_user="My Sensor",
        )
        dev_reg = MagicMock()
        dev_reg.devices = {d1.id: d1, d2.id: d2}

        with patch(
            "homeassistant.helpers.device_registry.async_get",
            return_value=dev_reg,
        ):
            result = await agent.get_device_registry()
        assert len(result) == 2
        by_id = {r["id"]: r for r in result}
        assert by_id["dev1"]["manufacturer"] == "Philips"
        assert by_id["dev1"]["disabled"] is False
        assert by_id["dev2"]["disabled"] is True
        assert by_id["dev2"]["name_by_user"] == "My Sensor"

        with patch(
            "homeassistant.helpers.device_registry.async_get", return_value=None
        ):
            assert await agent.get_device_registry() == []

    @pytest.mark.asyncio
    async def test_get_area_registry(self, agent):
        kitchen = MagicMock(
            id="kitchen",
            normalized_name="kitchen",
            picture=None,
            icon="mdi:silverware",
            floor_id="ground",
            labels={"main"},
        )
        kitchen.name = "Kitchen"
        living = MagicMock(
            id="living",
            normalized_name="living_room",
            picture=None,
            icon=None,
            floor_id="ground",
            labels=set(),
        )
        living.name = "Living Room"
        area_reg = MagicMock()
        area_reg.areas = {kitchen.id: kitchen, living.id: living}

        with patch(
            "homeassistant.helpers.area_registry.async_get", return_value=area_reg
        ):
            result = await agent.get_area_registry()
        assert set(result.keys()) == {"kitchen", "living"}
        assert result["kitchen"]["name"] == "Kitchen"
        assert result["kitchen"]["floor_id"] == "ground"
        assert isinstance(result["living"]["labels"], list)

        with patch(
            "homeassistant.helpers.area_registry.async_get", return_value=None
        ):
            assert await agent.get_area_registry() == {}

    @pytest.mark.asyncio
    async def test_get_person_data(self, agent, mock_hass):
        person_state = _make_state(
            "person.alice",
            "home",
            {
                "friendly_name": "Alice",
                "latitude": 40.0,
                "longitude": -74.0,
                "source": "device_tracker.alice_phone",
                "gps_accuracy": 10,
            },
        )
        light_state = _make_state("light.kitchen", "on")

        def async_all(domain=None):
            states = [person_state, light_state]
            if domain == "person":
                return [s for s in states if s.entity_id.startswith("person.")]
            return states

        mock_hass.states.async_all = MagicMock(side_effect=async_all)

        result = await agent.get_person_data()
        assert len(result) == 1
        assert result[0]["entity_id"] == "person.alice"
        assert result[0]["state"] == "home"
        assert result[0]["latitude"] == 40.0

        mock_hass.states.async_all = MagicMock(side_effect=lambda domain=None: [])
        assert await agent.get_person_data() == []

    @pytest.mark.asyncio
    async def test_get_scenes(self, agent, mock_hass):
        scene_state = _make_state(
            "scene.movie_night",
            "scening",
            {
                "friendly_name": "Movie Night",
                "icon": "mdi:movie",
                "last_activated": "2024-01-01T18:00:00Z",
            },
        )
        light_state = _make_state("light.kitchen", "on")

        def async_all(domain=None):
            states = [scene_state, light_state]
            if domain == "scene":
                return [s for s in states if s.entity_id.startswith("scene.")]
            return states

        mock_hass.states.async_all = MagicMock(side_effect=async_all)

        result = await agent.get_scenes()
        assert len(result) == 1
        assert result[0]["entity_id"] == "scene.movie_night"
        assert result[0]["icon"] == "mdi:movie"


class TestDashboardManagement:
    """Integration tests for dashboard CRUD methods."""

    @pytest.mark.asyncio
    async def test_get_dashboards(self, agent, mock_hass):
        from homeassistant.components.lovelace import DOMAIN as LOVELACE_DOMAIN

        mock_hass.data["websocket_api"] = MagicMock()

        lovelace_data = MagicMock()
        lovelace_data.dashboards = {None: MagicMock(), "kitchen": MagicMock()}
        lovelace_data.yaml_dashboards = {
            None: {"title": "Home"},
            "kitchen": {
                "title": "Kitchen",
                "icon": "mdi:silverware",
                "show_in_sidebar": False,
                "require_admin": True,
            },
        }
        mock_hass.data[LOVELACE_DOMAIN] = lovelace_data

        result = await agent.get_dashboards()
        assert len(result) == 2
        by_url = {d["url_path"]: d for d in result}
        assert by_url[None]["title"] == "Home"
        assert by_url["kitchen"]["icon"] == "mdi:silverware"
        assert by_url["kitchen"]["show_in_sidebar"] is False
        assert by_url["kitchen"]["require_admin"] is True

        mock_hass.data.pop("websocket_api")
        no_ws = await agent.get_dashboards()
        assert no_ws == [{"error": "WebSocket API not available"}]

    @pytest.mark.asyncio
    async def test_get_dashboard_config(self, agent, mock_hass):
        from homeassistant.components.lovelace import DOMAIN as LOVELACE_DOMAIN

        default_dash = MagicMock()
        default_dash.async_get_info = AsyncMock(
            return_value={"title": "Home", "views": []}
        )
        kitchen_dash = MagicMock()
        kitchen_dash.async_get_info = AsyncMock(
            return_value={"title": "Kitchen", "views": [{"path": "k"}]}
        )

        lovelace_data = MagicMock()
        lovelace_data.dashboards = {None: default_dash, "kitchen": kitchen_dash}
        mock_hass.data[LOVELACE_DOMAIN] = lovelace_data

        default = await agent.get_dashboard_config()
        assert default["title"] == "Home"

        kitchen = await agent.get_dashboard_config("kitchen")
        assert kitchen["title"] == "Kitchen"

        missing = await agent.get_dashboard_config("nonexistent")
        assert "error" in missing
        assert "nonexistent" in missing["error"]

    @pytest.mark.asyncio
    async def test_create_dashboard(self, agent, mock_hass, tmp_path):
        config_yaml = tmp_path / "configuration.yaml"
        config_yaml.write_text("# default config\n")

        result = await agent.create_dashboard(
            {
                "title": "My Dashboard",
                "url_path": "my-dashboard",
                "icon": "mdi:view-dashboard",
                "views": [{"title": "Home"}],
            }
        )
        assert result.get("success") is True
        assert result["url_path"] == "my-dashboard"

        produced = tmp_path / "ui-lovelace-my-dashboard.yaml"
        assert produced.exists()
        contents = produced.read_text()
        assert "My Dashboard" in contents

        assert "my-dashboard" in config_yaml.read_text()

        bad = await agent.create_dashboard({"url_path": "x"})
        assert "error" in bad

        bad2 = await agent.create_dashboard({"title": "x"})
        assert "error" in bad2

    @pytest.mark.asyncio
    async def test_update_dashboard(self, agent, mock_hass, tmp_path):
        existing = tmp_path / "ui-lovelace-existing.yaml"
        existing.write_text("title: Old Title\nviews: []\n")

        result = await agent.update_dashboard(
            "existing",
            {
                "title": "New Title",
                "icon": "mdi:home",
                "views": [{"title": "Updated"}],
            },
        )
        assert result.get("success") is True

        updated = existing.read_text()
        assert "New Title" in updated
        assert "Old Title" not in updated

        missing = await agent.update_dashboard(
            "does-not-exist", {"title": "X"}
        )
        assert "error" in missing
