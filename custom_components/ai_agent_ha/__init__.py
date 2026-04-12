"""The AI Agent HA integration."""

from __future__ import annotations

import json
import logging
import time

import voluptuous as vol
from homeassistant.components import websocket_api
from homeassistant.components.frontend import async_register_built_in_panel
from homeassistant.components.http import StaticPathConfig
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, SupportsResponse
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType

from .agent import AiAgentHaAgent
from .const import AI_PROVIDERS, DOMAIN

_LOGGER = logging.getLogger(__name__)

# Config schema - this integration only supports config entries
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

# Define service schema to accept a custom prompt
SERVICE_SCHEMA = vol.Schema(
    {
        vol.Optional("prompt"): cv.string,
    }
)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the AI Agent HA component."""
    return True


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate old config entries to new version."""
    _LOGGER.debug("Migrating config entry from version %s", entry.version)

    if entry.version == 1:
        # No migration needed for version 1
        return True

    # Future migrations would go here
    # if entry.version < 2:
    #     # Migrate from version 1 to 2
    #     new_data = dict(entry.data)
    #     # Add migration logic here
    #     hass.config_entries.async_update_entry(entry, data=new_data, version=2)

    _LOGGER.info("Migration to version %s successful", entry.version)
    return True


async def _cleanup_legacy_dashboard_config(hass: HomeAssistant) -> None:
    """Remove stale lovelace dashboard entries written by pre-v1.2.4 create_dashboard.

    The old file-based approach wrote entries to configuration.yaml like:
        lovelace:
          dashboards:
            lighting:
              mode: yaml
              filename: ui-lovelace-lighting.yaml

    These entries fail HA config validation if the url_path has no hyphen,
    and are no longer needed since v1.2.4 uses the Lovelace Storage API.
    """
    import os

    config_file = hass.config.path("configuration.yaml")

    def _do_cleanup():
        try:
            with open(config_file, "r") as f:
                content = f.read()

            # Only proceed if there's a lovelace section with our marker
            if "ui-lovelace-" not in content:
                return []  # Nothing to clean

            import yaml as _yaml

            try:
                config = _yaml.safe_load(content)
            except Exception:
                return []  # Don't touch if we can't parse

            if not isinstance(config, dict):
                return []

            lovelace = config.get("lovelace")
            if not isinstance(lovelace, dict):
                return []

            dashboards = lovelace.get("dashboards")
            if not isinstance(dashboards, dict):
                return []

            # Find entries with ui-lovelace- filenames (AI-generated)
            stale_keys = [
                k
                for k, v in dashboards.items()
                if isinstance(v, dict)
                and str(v.get("filename", "")).startswith("ui-lovelace-")
            ]

            if not stale_keys:
                return []

            # Remove stale entries
            for key in stale_keys:
                del dashboards[key]
                _LOGGER.warning(
                    "Removed stale AI-generated lovelace dashboard entry '%s' from "
                    "configuration.yaml. This entry was created by a pre-v1.2.4 version "
                    "of AI Agent HA and is no longer needed.",
                    key,
                )

            # If dashboards section is now empty, remove it
            if not dashboards:
                del lovelace["dashboards"]

            # If lovelace section is now empty, remove the whole key
            if not lovelace:
                del config["lovelace"]

            new_content = _yaml.dump(
                config, default_flow_style=False, allow_unicode=True
            )

            with open(config_file, "w") as f:
                f.write(new_content)

            return stale_keys

        except Exception as e:
            _LOGGER.error("Error during legacy lovelace config cleanup: %s", str(e))
            return []

    def _cleanup_orphan_files(stale_keys):
        """Delete orphaned ui-lovelace-*.yaml files."""
        for key in stale_keys:
            for filename in [f"ui-lovelace-{key}.yaml", f"dashboards/{key}.yaml"]:
                filepath = hass.config.path(filename)
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        _LOGGER.warning(
                            "Removed orphaned dashboard file: %s", filepath
                        )
                except Exception as e:
                    _LOGGER.error(
                        "Could not remove orphaned file %s: %s", filepath, str(e)
                    )

    stale_keys = await hass.async_add_executor_job(_do_cleanup)
    if stale_keys:
        await hass.async_add_executor_job(_cleanup_orphan_files, stale_keys)
        _LOGGER.warning(
            "AI Agent HA: Cleaned up %d stale lovelace dashboard configuration "
            "entries from configuration.yaml. These were created by a pre-v1.2.4 "
            "version. Dashboards now use HA Storage API and no configuration.yaml "
            "changes are needed.",
            len(stale_keys),
        )
        hass.components.persistent_notification.async_create(
            f"AI Agent HA cleaned up {len(stale_keys)} stale dashboard configuration "
            "entries from configuration.yaml that were created by a previous version. "
            "No action needed — dashboards now use HA Storage API.",
            title="AI Agent HA: Legacy Config Cleaned",
            notification_id="ai_agent_ha_legacy_cleanup",
        )


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up AI Agent HA from a config entry."""
    # Clean up any stale lovelace entries from pre-v1.2.4 file-based dashboard creation
    await _cleanup_legacy_dashboard_config(hass)

    try:
        # Handle version compatibility
        if not hasattr(entry, "version") or entry.version != 1:
            _LOGGER.warning(
                "Config entry has version %s, expected 1. Attempting compatibility mode.",
                getattr(entry, "version", "unknown"),
            )

        # Convert ConfigEntry to dict and ensure all required keys exist
        config_data = dict(entry.data)

        # Ensure backward compatibility - check for required keys
        if "ai_provider" not in config_data:
            _LOGGER.error(
                "Config entry missing required 'ai_provider' key. Entry data: %s",
                config_data,
            )
            raise ConfigEntryNotReady("Config entry missing required 'ai_provider' key")

        if DOMAIN not in hass.data:
            hass.data[DOMAIN] = {"agents": {}, "configs": {}}

        provider = config_data["ai_provider"]

        # Validate provider — derived from AI_PROVIDERS in const.py so this
        # check stays in sync automatically whenever a new provider is added.
        if provider not in AI_PROVIDERS:
            _LOGGER.error("Unknown AI provider: %s", provider)
            raise ConfigEntryNotReady(f"Unknown AI provider: {provider}")

        # Store config for this provider
        hass.data[DOMAIN]["configs"][provider] = config_data

        # Create agent for this provider
        _LOGGER.debug(
            "Creating AI agent for provider %s with config: %s",
            provider,
            {
                k: v
                for k, v in config_data.items()
                if k
                not in [
                    "llama_token",
                    "openai_token",
                    "gemini_token",
                    "openrouter_token",
                    "anthropic_token",
                    "alter_token",
                    "zai_token",
                    "asksage_token",
                ]
            },
        )
        hass.data[DOMAIN]["agents"][provider] = AiAgentHaAgent(hass, config_data)

        _LOGGER.info("Successfully set up AI Agent HA for provider: %s", provider)

    except KeyError as err:
        _LOGGER.error("Missing required configuration key: %s", err)
        raise ConfigEntryNotReady(f"Missing required configuration key: {err}")
    except Exception as err:
        _LOGGER.exception("Unexpected error setting up AI Agent HA")
        raise ConfigEntryNotReady(f"Error setting up AI Agent HA: {err}")

    # Modify the query service handler to use the correct provider
    # ── State-entity response channel ──────────────────────────────
    # The event bus is fire-and-forget; if the frontend panel isn't
    # listening when the event fires (slow LLM round-trips, browser
    # refresh) the response is lost.  We now *also* persist every
    # response into a state entity so the panel can poll it as a
    # fallback.
    RESPONSE_ENTITY = f"sensor.{DOMAIN}_last_response"

    def _write_response_state(result: dict) -> None:
        """Persist *result* to a HA state entity for frontend fallback."""
        try:
            ts = time.time()
            # Truncate large payloads for state — HA limits attribute size.
            payload = json.dumps(result, default=str)
            if len(payload) > 16000:
                # Keep answer + success/error but drop debug blob
                trimmed = {k: v for k, v in result.items() if k != "debug"}
                payload = json.dumps(trimmed, default=str)
            hass.states.async_set(
                RESPONSE_ENTITY,
                str(ts),  # state = timestamp (changes each response)
                {"response_json": payload, "ts": ts},
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.debug("Could not write response state: %s", exc)

    async def async_handle_query(call):
        """Handle the query service call."""
        try:
            # Check if agents are available
            if DOMAIN not in hass.data or not hass.data[DOMAIN].get("agents"):
                _LOGGER.error(
                    "No AI agents available. Please configure the integration first."
                )
                result = {"error": "No AI agents configured"}
                _write_response_state(result)
                hass.bus.async_fire("ai_agent_ha_response", result)
                return

            provider = call.data.get("provider")
            if provider not in hass.data[DOMAIN]["agents"]:
                # Get the first available provider
                available_providers = list(hass.data[DOMAIN]["agents"].keys())
                if not available_providers:
                    _LOGGER.error("No AI agents available")
                    result = {"error": "No AI agents configured"}
                    _write_response_state(result)
                    hass.bus.async_fire("ai_agent_ha_response", result)
                    return
                provider = available_providers[0]
                _LOGGER.debug(f"Using fallback provider: {provider}")

            agent = hass.data[DOMAIN]["agents"][provider]
            result = await agent.process_query(
                call.data.get("prompt", ""),
                provider=provider,
                debug=call.data.get("debug", False),
            )
            _write_response_state(result)
            hass.bus.async_fire("ai_agent_ha_response", result)
        except Exception as e:
            _LOGGER.error("Query service error: %s", str(e), exc_info=True)
            result = {"error": "Unable to process request. Check Home Assistant logs for details."}
            _write_response_state(result)
            hass.bus.async_fire("ai_agent_ha_response", result)

    async def async_handle_create_automation(call):
        """Handle the create_automation service call."""
        try:
            # Check if agents are available
            if DOMAIN not in hass.data or not hass.data[DOMAIN].get("agents"):
                _LOGGER.error(
                    "No AI agents available. Please configure the integration first."
                )
                return {"error": "No AI agents configured"}

            provider = call.data.get("provider")
            if provider not in hass.data[DOMAIN]["agents"]:
                # Get the first available provider
                available_providers = list(hass.data[DOMAIN]["agents"].keys())
                if not available_providers:
                    _LOGGER.error("No AI agents available")
                    return {"error": "No AI agents configured"}
                provider = available_providers[0]
                _LOGGER.debug(f"Using fallback provider: {provider}")

            agent = hass.data[DOMAIN]["agents"][provider]
            result = await agent.create_automation(call.data.get("automation", {}))
            return result
        except Exception as e:
            _LOGGER.error("Create automation service error: %s", str(e), exc_info=True)
            return {"error": "Unable to process request. Check Home Assistant logs for details."}

    async def async_handle_save_prompt_history(call):
        """Handle the save_prompt_history service call."""
        try:
            # Check if agents are available
            if DOMAIN not in hass.data or not hass.data[DOMAIN].get("agents"):
                _LOGGER.error(
                    "No AI agents available. Please configure the integration first."
                )
                return {"error": "No AI agents configured"}

            provider = call.data.get("provider")
            if provider not in hass.data[DOMAIN]["agents"]:
                # Get the first available provider
                available_providers = list(hass.data[DOMAIN]["agents"].keys())
                if not available_providers:
                    _LOGGER.error("No AI agents available")
                    return {"error": "No AI agents configured"}
                provider = available_providers[0]
                _LOGGER.debug(f"Using fallback provider: {provider}")

            agent = hass.data[DOMAIN]["agents"][provider]
            user_id = call.context.user_id if call.context.user_id else "default"
            result = await agent.save_user_prompt_history(
                user_id, call.data.get("history", [])
            )
            return result
        except Exception as e:
            _LOGGER.error("Save prompt history service error: %s", str(e), exc_info=True)
            return {"error": "Unable to process request. Check Home Assistant logs for details."}

    async def async_handle_load_prompt_history(call):
        """Handle the load_prompt_history service call."""
        try:
            # Check if agents are available
            if DOMAIN not in hass.data or not hass.data[DOMAIN].get("agents"):
                _LOGGER.error(
                    "No AI agents available. Please configure the integration first."
                )
                return {"error": "No AI agents configured"}

            provider = call.data.get("provider")
            if provider not in hass.data[DOMAIN]["agents"]:
                # Get the first available provider
                available_providers = list(hass.data[DOMAIN]["agents"].keys())
                if not available_providers:
                    _LOGGER.error("No AI agents available")
                    return {"error": "No AI agents configured"}
                provider = available_providers[0]
                _LOGGER.debug(f"Using fallback provider: {provider}")

            agent = hass.data[DOMAIN]["agents"][provider]
            user_id = call.context.user_id if call.context.user_id else "default"
            result = await agent.load_user_prompt_history(user_id)
            _LOGGER.debug("Load prompt history result: %s", result)
            return result
        except Exception as e:
            _LOGGER.error("Load prompt history service error: %s", str(e), exc_info=True)
            return {"error": "Unable to process request. Check Home Assistant logs for details."}

    async def async_handle_create_dashboard(call):
        """Handle the create_dashboard service call."""
        try:
            # Check if agents are available
            if DOMAIN not in hass.data or not hass.data[DOMAIN].get("agents"):
                _LOGGER.error(
                    "No AI agents available. Please configure the integration first."
                )
                return {"error": "No AI agents configured"}

            provider = call.data.get("provider")
            if provider not in hass.data[DOMAIN]["agents"]:
                # Get the first available provider
                available_providers = list(hass.data[DOMAIN]["agents"].keys())
                if not available_providers:
                    _LOGGER.error("No AI agents available")
                    return {"error": "No AI agents configured"}
                provider = available_providers[0]
                _LOGGER.debug(f"Using fallback provider: {provider}")

            agent = hass.data[DOMAIN]["agents"][provider]

            # Parse dashboard config if it's a string
            dashboard_config = call.data.get("dashboard_config", {})
            if isinstance(dashboard_config, str):
                try:
                    import json

                    dashboard_config = json.loads(dashboard_config)
                except json.JSONDecodeError as e:
                    _LOGGER.error("Invalid JSON in dashboard_config: %s", str(e))
                    return {"error": "Invalid JSON in dashboard_config. Check Home Assistant logs for details."}

            result = await agent.create_dashboard(dashboard_config)
            return result
        except Exception as e:
            _LOGGER.error("Create dashboard service error: %s", str(e), exc_info=True)
            return {"error": "Unable to process request. Check Home Assistant logs for details."}

    async def async_handle_update_dashboard(call):
        """Handle the update_dashboard service call."""
        try:
            # Check if agents are available
            if DOMAIN not in hass.data or not hass.data[DOMAIN].get("agents"):
                _LOGGER.error(
                    "No AI agents available. Please configure the integration first."
                )
                return {"error": "No AI agents configured"}

            provider = call.data.get("provider")
            if provider not in hass.data[DOMAIN]["agents"]:
                # Get the first available provider
                available_providers = list(hass.data[DOMAIN]["agents"].keys())
                if not available_providers:
                    _LOGGER.error("No AI agents available")
                    return {"error": "No AI agents configured"}
                provider = available_providers[0]
                _LOGGER.debug(f"Using fallback provider: {provider}")

            agent = hass.data[DOMAIN]["agents"][provider]

            # Parse dashboard config if it's a string
            dashboard_config = call.data.get("dashboard_config", {})
            if isinstance(dashboard_config, str):
                try:
                    import json

                    dashboard_config = json.loads(dashboard_config)
                except json.JSONDecodeError as e:
                    _LOGGER.error("Invalid JSON in dashboard_config: %s", str(e))
                    return {"error": "Invalid JSON in dashboard_config. Check Home Assistant logs for details."}

            dashboard_url = call.data.get("dashboard_url", "")
            if not dashboard_url:
                return {"error": "Dashboard URL is required"}

            result = await agent.update_dashboard(dashboard_url, dashboard_config)
            return result
        except Exception as e:
            _LOGGER.error("Update dashboard service error: %s", str(e), exc_info=True)
            return {"error": "Unable to process request. Check Home Assistant logs for details."}

    @websocket_api.websocket_command(
        {vol.Required("type"): "ai_agent_ha/get_providers"}
    )
    @websocket_api.async_response
    async def ws_get_providers(hass, connection, msg):
        """Return safe provider info from hass.data — no credentials exposed."""
        from .const import AI_PROVIDERS as _AI_PROVIDERS

        domain_data = hass.data.get(DOMAIN, {})
        agents = domain_data.get("agents", {})
        configs = domain_data.get("configs", {})

        provider_display = {
            "openai": "OpenAI",
            "llama": "Llama",
            "gemini": "Google Gemini",
            "openrouter": "OpenRouter",
            "anthropic": "Anthropic",
            "alter": "Alter",
            "zai": "z.ai",
            "local": "Local Model",
            "asksage": "Ask Sage",
        }

        providers = []
        for provider_key, agent in agents.items():
            cfg = configs.get(provider_key, {})
            models = cfg.get("models", {})
            model_name = models.get(provider_key, "")
            providers.append(
                {
                    "value": provider_key,
                    "label": provider_display.get(provider_key, provider_key),
                    "model": model_name,
                    "persist_chat_history": cfg.get("persist_chat_history", False),
                    "enable_streaming": cfg.get("enable_streaming", False),
                }
            )

        connection.send_message(websocket_api.result_message(msg["id"], providers))

    websocket_api.async_register_command(hass, ws_get_providers)

    @websocket_api.websocket_command(
        {vol.Required("type"): "ai_agent_ha/get_last_response"}
    )
    @websocket_api.async_response
    async def ws_get_last_response(hass, connection, msg):
        """Return the last query response from the state entity.

        The frontend uses this as a fallback when the event-bus delivery
        is missed (slow LLM round-trips, browser refresh, etc.).
        """
        state_obj = hass.states.get(RESPONSE_ENTITY)
        if state_obj and state_obj.attributes.get("response_json"):
            try:
                data = json.loads(state_obj.attributes["response_json"])
                data["_ts"] = state_obj.attributes.get("ts", 0)
                connection.send_message(
                    websocket_api.result_message(msg["id"], data)
                )
            except (json.JSONDecodeError, TypeError):
                connection.send_message(
                    websocket_api.result_message(msg["id"], None)
                )
        else:
            connection.send_message(
                websocket_api.result_message(msg["id"], None)
            )

    websocket_api.async_register_command(hass, ws_get_last_response)

    # Register services
    hass.services.async_register(DOMAIN, "query", async_handle_query)
    hass.services.async_register(
        DOMAIN, "create_automation", async_handle_create_automation
    )
    hass.services.async_register(
        DOMAIN, "save_prompt_history", async_handle_save_prompt_history
    )
    hass.services.async_register(
        DOMAIN, "load_prompt_history", async_handle_load_prompt_history
    )
    hass.services.async_register(
        DOMAIN,
        "create_dashboard",
        async_handle_create_dashboard,
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN,
        "update_dashboard",
        async_handle_update_dashboard,
        supports_response=SupportsResponse.OPTIONAL,
    )

    # Register static path for frontend
    await hass.http.async_register_static_paths(
        [
            StaticPathConfig(
                "/frontend/ai_agent_ha",
                hass.config.path("custom_components/ai_agent_ha/frontend"),
                False,
            )
        ]
    )

    # Panel registration with proper error handling
    panel_name = "ai_agent_ha"
    try:
        if await _panel_exists(hass, panel_name):
            _LOGGER.debug("AI Agent HA panel already exists, skipping registration")
            return True

        _LOGGER.debug("Registering AI Agent HA panel")
        async_register_built_in_panel(
            hass,
            component_name="custom",
            sidebar_title="AI Agent HA",
            sidebar_icon="mdi:robot",
            frontend_url_path=panel_name,
            require_admin=False,
            config={
                "_panel_custom": {
                    "name": "ai_agent_ha-panel",
                    "module_url": "/frontend/ai_agent_ha/ai_agent_ha-panel.js",
                    "embed_iframe": False,
                }
            },
        )
        _LOGGER.debug("AI Agent HA panel registered successfully")
    except Exception as e:
        _LOGGER.warning("Panel registration error: %s", str(e))

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    if await _panel_exists(hass, "ai_agent_ha"):
        try:
            from homeassistant.components.frontend import async_remove_panel

            async_remove_panel(hass, "ai_agent_ha")
            _LOGGER.debug("AI Agent HA panel removed successfully")
        except Exception as e:
            _LOGGER.debug("Error removing panel: %s", str(e))

    # Remove services
    hass.services.async_remove(DOMAIN, "query")
    hass.services.async_remove(DOMAIN, "create_automation")
    hass.services.async_remove(DOMAIN, "save_prompt_history")
    hass.services.async_remove(DOMAIN, "load_prompt_history")
    hass.services.async_remove(DOMAIN, "create_dashboard")
    hass.services.async_remove(DOMAIN, "update_dashboard")
    # Remove data
    if DOMAIN in hass.data:
        hass.data.pop(DOMAIN)

    return True


async def _panel_exists(hass: HomeAssistant, panel_name: str) -> bool:
    """Check if a panel already exists."""
    try:
        return hasattr(hass.data, "frontend_panels") and panel_name in hass.data.get(
            "frontend_panels", {}
        )
    except Exception as e:
        _LOGGER.debug("Error checking panel existence: %s", str(e))
        return False
