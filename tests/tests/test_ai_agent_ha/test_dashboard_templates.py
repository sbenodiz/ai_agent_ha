"""Tests for dashboard templates functionality."""

import os
import sys
from unittest.mock import Mock, patch

import pytest

# Add the parent directory to the path for direct imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from custom_components.ai_agent_ha.dashboard_templates import (
        CARD_EXAMPLES,
        COMMON_ICONS,
        DASHBOARD_TEMPLATES,
        get_template_for_entities,
    )
except ImportError:
    # Fallback for local testing
    DASHBOARD_TEMPLATES = {}
    CARD_EXAMPLES = {}
    COMMON_ICONS = {}

    def get_template_for_entities(entities, dashboard_type="general"):
        return {}


class TestDashboardTemplates:
    """Test dashboard templates and related functionality."""

    def test_dashboard_templates_structure(self):
        """Test that dashboard templates have correct structure."""
        assert isinstance(DASHBOARD_TEMPLATES, dict)

        # Check required keys in each template
        for template_name, template in DASHBOARD_TEMPLATES.items():
            assert isinstance(template, dict)
            assert "title" in template
            assert "url_path" in template
            assert "icon" in template
            assert "show_in_sidebar" in template
            assert "views" in template
            assert isinstance(template["views"], list)

            # Check view structure
            for view in template["views"]:
                assert isinstance(view, dict)
                assert "title" in view
                assert "cards" in view
                assert isinstance(view["cards"], list)

    def test_card_examples_structure(self):
        """Test that card examples have correct structure."""
        assert isinstance(CARD_EXAMPLES, dict)

        for card_type, card_info in CARD_EXAMPLES.items():
            assert isinstance(card_info, dict)
            assert "description" in card_info
            assert "example" in card_info
            assert isinstance(card_info["example"], dict)

    def test_common_icons_structure(self):
        """Test that common icons are properly defined."""
        assert isinstance(COMMON_ICONS, dict)

        # Check that all values are valid icon strings
        for key, icon in COMMON_ICONS.items():
            assert isinstance(icon, str)
            assert icon.startswith("mdi:")

    def test_get_template_for_entities_empty(self):
        """Test get_template_for_entities with empty entities list."""
        result = get_template_for_entities([], "general")

        assert isinstance(result, dict)
        assert "title" in result
        assert "url_path" in result
        assert "icon" in result
        assert "show_in_sidebar" in result
        assert "views" in result
        assert len(result["views"]) == 1
        assert result["views"][0]["title"] == "Overview"
        assert isinstance(result["views"][0]["cards"], list)

    def test_get_template_for_entities_with_lights(self):
        """Test get_template_for_entities with light entities."""
        entities = [
            {"entity_id": "light.living_room", "state": "on"},
            {"entity_id": "light.bedroom", "state": "off"},
            {"entity_id": "light.kitchen", "state": "on"},
        ]

        result = get_template_for_entities(entities, "lights")

        assert result["title"] == "Lights Dashboard"
        assert result["url_path"] == "lights"
        assert result["icon"] == "mdi:lightbulb"

        # Check that lights are grouped correctly
        cards = result["views"][0]["cards"]
        light_card = next(
            (
                card
                for card in cards
                if card.get("type") == "entities" and "Lights" in card.get("title", "")
            ),
            None,
        )
        assert light_card is not None
        assert "light.living_room" in light_card["entities"]
        assert "light.bedroom" in light_card["entities"]
        assert "light.kitchen" in light_card["entities"]

    def test_get_template_for_entities_with_climate(self):
        """Test get_template_for_entities with climate entities."""
        entities = [
            {"entity_id": "climate.main_thermostat", "state": "heat"},
            {"entity_id": "sensor.temperature", "state": "22.5"},
        ]

        result = get_template_for_entities(entities, "climate")

        assert result["title"] == "Climate Dashboard"
        assert result["url_path"] == "climate"
        assert result["icon"] == "mdi:thermometer"

        # Check that climate entities are handled correctly
        cards = result["views"][0]["cards"]
        thermostat_card = next(
            (card for card in cards if card.get("type") == "thermostat"), None
        )
        assert thermostat_card is not None
        assert thermostat_card["entity"] == "climate.main_thermostat"

    def test_get_template_for_entities_with_media(self):
        """Test get_template_for_entities with media player entities."""
        entities = [
            {"entity_id": "media_player.living_room_tv", "state": "playing"},
            {"entity_id": "media_player.bedroom_speaker", "state": "idle"},
        ]

        result = get_template_for_entities(entities, "media")

        assert result["title"] == "Media Dashboard"
        assert result["url_path"] == "media"
        assert result["icon"] == "mdi:play"

        # Check that media entities are handled correctly
        cards = result["views"][0]["cards"]
        media_cards = [card for card in cards if card.get("type") == "media-control"]
        assert len(media_cards) == 2
        entity_ids = [card["entity"] for card in media_cards]
        assert "media_player.living_room_tv" in entity_ids
        assert "media_player.bedroom_speaker" in entity_ids

    def test_get_template_for_entities_with_security(self):
        """Test get_template_for_entities with security entities."""
        entities = [
            {"entity_id": "binary_sensor.front_door", "state": "off"},
            {"entity_id": "binary_sensor.back_door", "state": "on"},
            {"entity_id": "alarm_control_panel.home", "state": "armed_home"},
        ]

        result = get_template_for_entities(entities, "security")

        assert result["title"] == "Security Dashboard"
        assert result["url_path"] == "security"
        assert result["icon"] == "mdi:security"

        # Check that security entities are grouped correctly
        cards = result["views"][0]["cards"]

        # Check for binary_sensor entities (grouped under "Sensors")
        sensor_card = next(
            (
                card
                for card in cards
                if card.get("type") == "entities" and card.get("entities", [])
            ),
            None,
        )

        # Check for alarm_control_panel entities (grouped under "Switches" since they're not explicitly handled)
        alarm_entities = []
        for card in cards:
            if card.get("type") == "entities" and card.get("entities", []):
                alarm_entities.extend(card.get("entities", []))

        # Verify entities are included
        all_entities = []
        for card in cards:
            if "entities" in card:
                all_entities.extend(card.get("entities", []))
            elif "entity" in card:
                all_entities.append(card.get("entity"))

        assert "binary_sensor.front_door" in all_entities
        assert "binary_sensor.back_door" in all_entities
        assert "alarm_control_panel.home" in all_entities
        assert sensor_card is not None

    def test_get_template_for_entities_mixed_domains(self):
        """Test get_template_for_entities with mixed domain entities."""
        entities = [
            {"entity_id": "light.living_room", "state": "on"},
            {"entity_id": "sensor.temperature", "state": "22.5"},
            {"entity_id": "switch.fan", "state": "on"},
            {"entity_id": "weather.home", "state": "sunny"},
        ]

        result = get_template_for_entities(entities, "general")

        # Check that entities are grouped by domain
        cards = result["views"][0]["cards"]

        # Should have cards for each domain
        card_types = [card.get("type") for card in cards]
        assert "entities" in card_types  # For lights and switches
        assert "weather-forecast" in card_types  # For weather

        # Check that all entities are included
        all_entities = []
        for card in cards:
            if "entities" in card:
                all_entities.extend(card.get("entities", []))

        assert "light.living_room" in all_entities
        assert "sensor.temperature" in all_entities
        assert "switch.fan" in all_entities

    def test_get_template_for_entities_string_entities(self):
        """Test get_template_for_entities with string entity IDs."""
        entities = ["light.living_room", "light.bedroom", "climate.main"]

        result = get_template_for_entities(entities, "mixed")

        assert result["title"] == "Mixed Dashboard"
        assert result["url_path"] == "mixed"

        # Check that string entities are handled correctly
        cards = result["views"][0]["cards"]
        light_entities = []
        for card in cards:
            if card.get("type") == "entities" and "Lights" in str(
                card.get("title", "")
            ):
                light_entities.extend(card.get("entities", []))

        assert "light.living_room" in light_entities
        assert "light.bedroom" in light_entities

    def test_template_customization(self):
        """Test that templates can be customized."""
        entities = [
            {"entity_id": "light.living_room", "state": "on"},
            {"entity_id": "light.kitchen", "state": "on"},
        ]

        result = get_template_for_entities(entities, "custom")

        # Check that custom dashboard type is handled
        assert result["title"] == "Custom Dashboard"
        assert result["url_path"] == "custom"
        assert result["icon"] == "mdi:view-dashboard"  # Default icon

    def test_empty_entity_list_limits(self):
        """Test behavior with empty or very large entity lists."""
        # Empty list
        result = get_template_for_entities([], "empty")
        assert len(result["views"][0]["cards"]) == 0

        # Large list (should handle gracefully)
        large_entities = [
            {"entity_id": f"light.light_{i}", "state": "on"} for i in range(100)
        ]
        result = get_template_for_entities(large_entities, "large")

        # Should still create a valid template
        assert isinstance(result, dict)
        assert "views" in result
        assert len(result["views"]) == 1

    def test_climate_dashboard_with_temperature_sensors(self):
        """Test climate dashboard creation with temperature sensors (no climate.* entities)."""
        entities = [
            {
                "entity_id": "sensor.bedroom_temperature",
                "state": "22.5",
                "attributes": {
                    "device_class": "temperature",
                    "unit_of_measurement": "°C",
                },
            },
            {
                "entity_id": "sensor.kitchen_temperature",
                "state": "23.1",
                "attributes": {
                    "device_class": "temperature",
                    "unit_of_measurement": "°C",
                },
            },
        ]

        result = get_template_for_entities(entities, "climate")

        # Should create a climate dashboard with temperature sensors
        assert result["title"] == "Climate Dashboard"
        assert result["url_path"] == "climate"

        # Check that temperature sensors are included in cards
        cards = result["views"][0]["cards"]

        # Should have history graph for temperature
        history_cards = [
            c
            for c in cards
            if c.get("type") == "history-graph"
            and c.get("title") == "Temperature History"
        ]
        assert len(history_cards) == 1
        assert "sensor.bedroom_temperature" in history_cards[0]["entities"]
        assert "sensor.kitchen_temperature" in history_cards[0]["entities"]

        # Should have entities card for temperature sensors
        entity_cards = [
            c
            for c in cards
            if c.get("type") == "entities" and c.get("title") == "Temperature Sensors"
        ]
        assert len(entity_cards) == 1
        assert "sensor.bedroom_temperature" in entity_cards[0]["entities"]
        assert "sensor.kitchen_temperature" in entity_cards[0]["entities"]

    def test_climate_dashboard_with_humidity_sensors(self):
        """Test climate dashboard creation with humidity sensors."""
        entities = [
            {
                "entity_id": "sensor.living_room_humidity",
                "state": "55",
                "attributes": {"device_class": "humidity", "unit_of_measurement": "%"},
            },
            {
                "entity_id": "sensor.bedroom_humidity",
                "state": "52",
                "attributes": {"device_class": "humidity", "unit_of_measurement": "%"},
            },
        ]

        result = get_template_for_entities(entities, "climate")

        cards = result["views"][0]["cards"]

        # Should have history graph for humidity
        history_cards = [
            c
            for c in cards
            if c.get("type") == "history-graph" and c.get("title") == "Humidity History"
        ]
        assert len(history_cards) == 1
        assert "sensor.living_room_humidity" in history_cards[0]["entities"]
        assert "sensor.bedroom_humidity" in history_cards[0]["entities"]

        # Should have entities card for humidity sensors
        entity_cards = [
            c
            for c in cards
            if c.get("type") == "entities" and c.get("title") == "Humidity Sensors"
        ]
        assert len(entity_cards) == 1

    def test_climate_dashboard_with_mixed_sensors(self):
        """Test climate dashboard with climate entities, temperature, and humidity sensors."""
        entities = [
            {"entity_id": "climate.thermostat", "state": "heat", "attributes": {}},
            {
                "entity_id": "sensor.bedroom_temperature",
                "state": "22.5",
                "attributes": {"device_class": "temperature"},
            },
            {
                "entity_id": "sensor.living_room_humidity",
                "state": "55",
                "attributes": {"device_class": "humidity"},
            },
        ]

        result = get_template_for_entities(entities, "climate")

        cards = result["views"][0]["cards"]

        # Should have thermostat card for climate entity
        thermostat_cards = [c for c in cards if c.get("type") == "thermostat"]
        assert len(thermostat_cards) == 1
        assert thermostat_cards[0]["entity"] == "climate.thermostat"

        # Should have temperature history
        temp_history = [
            c
            for c in cards
            if c.get("type") == "history-graph"
            and c.get("title") == "Temperature History"
        ]
        assert len(temp_history) == 1

        # Should have humidity history
        humidity_history = [
            c
            for c in cards
            if c.get("type") == "history-graph" and c.get("title") == "Humidity History"
        ]
        assert len(humidity_history) == 1

    def test_sensors_without_device_class_not_in_climate(self):
        """Test that sensors without device_class are not categorized as climate sensors."""
        entities = [
            {
                "entity_id": "sensor.power_usage",
                "state": "150",
                "attributes": {"device_class": "power", "unit_of_measurement": "W"},
            },
            {
                "entity_id": "sensor.generic_sensor",
                "state": "42",
                "attributes": {},  # No device_class
            },
        ]

        result = get_template_for_entities(entities, "climate")

        cards = result["views"][0]["cards"]

        # Should NOT have temperature or humidity cards
        temp_cards = [c for c in cards if "Temperature" in c.get("title", "")]
        humidity_cards = [c for c in cards if "Humidity" in c.get("title", "")]

        assert len(temp_cards) == 0
        assert len(humidity_cards) == 0

        # Should have "Other Sensors" card
        other_sensor_cards = [c for c in cards if c.get("title") == "Other Sensors"]
        assert len(other_sensor_cards) == 1
        assert "sensor.power_usage" in other_sensor_cards[0]["entities"]
        assert "sensor.generic_sensor" in other_sensor_cards[0]["entities"]
