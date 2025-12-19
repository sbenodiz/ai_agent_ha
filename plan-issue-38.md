# Plan: Fix Deprecated Lovelace API Access (Issue #38)

## Problem Summary

The code in `custom_components/ai_agent_ha/agent.py` uses dictionary-style access to retrieve Lovelace dashboard data, which is deprecated and will stop working in Home Assistant 2026.2.

**Current problematic code patterns:**
```python
# Line 2002-2003 in get_dashboards():
lovelace_config = self.hass.data.get(LOVELACE_DOMAIN, {})
dashboards = lovelace_config.get(CONF_DASHBOARDS, {})

# Line 2071-2085 in get_dashboard_config():
lovelace_config = self.hass.data.get(LOVELACE_DOMAIN, {})
dashboard = lovelace_config.get("default_dashboard")
dashboards = lovelace_config.get("dashboards", {})
```

## Root Cause

Home Assistant is transitioning from dictionary-style data access to property-based access for the Lovelace component data. The deprecation warning states:

> "Detected that custom integration accessed lovelace_data['dashboards'] instead of lovelace_data.dashboards. This will stop working in Home Assistant 2026.2"

## Verified API from Home Assistant Source Code

The Lovelace component uses a **dataclass structure** (from `homeassistant/components/lovelace/__init__.py`):

```python
@dataclass
class LovelaceData:
    """Dataclass to store information in hass.data."""
    mode: str
    dashboards: dict[str | None, dashboard.LovelaceConfig]
    resources: resources.ResourceYAMLCollection | resources.ResourceStorageCollection
    yaml_dashboards: dict[str | None, ConfigType]
```

**Key details:**
- `dashboards` is a property (attribute) on the `LovelaceData` dataclass
- Keys are URL path strings, with `None` representing the default dashboard
- Values are `LovelaceStorage` or `LovelaceYAML` objects (both inherit from `LovelaceConfig`)
- The old `__getitem__` and `get` methods still work but log deprecation warnings

## Solution

Replace dictionary-style access (`lovelace_data['dashboards']`, `lovelace_data.get(...)`) with property access (`lovelace_data.dashboards`).

## Implementation Steps

### Step 1: Fix `get_dashboards()` method (lines 1986-2039)

**Before:**
```python
from homeassistant.components.lovelace import CONF_DASHBOARDS
from homeassistant.components.lovelace import DOMAIN as LOVELACE_DOMAIN

lovelace_config = self.hass.data.get(LOVELACE_DOMAIN, {})
dashboards = lovelace_config.get(CONF_DASHBOARDS, {})
```

**After:**
```python
from homeassistant.components.lovelace import DOMAIN as LOVELACE_DOMAIN

lovelace_data = self.hass.data.get(LOVELACE_DOMAIN)
if lovelace_data is None:
    return [{"error": "Lovelace not available"}]

# Use property access instead of dictionary access (required for HA 2026.2+)
# lovelace_data is a LovelaceData dataclass with a 'dashboards' attribute
dashboards = lovelace_data.dashboards
```

### Step 2: Fix `get_dashboard_config()` method (lines 2041-2101)

**Before:**
```python
lovelace_config = self.hass.data.get(LOVELACE_DOMAIN, {})
dashboard = lovelace_config.get("default_dashboard")
dashboards = lovelace_config.get("dashboards", {})
```

**After:**
```python
lovelace_data = self.hass.data.get(LOVELACE_DOMAIN)
if lovelace_data is None:
    return {"error": "Lovelace not available"}

# Use property access instead of dictionary access (required for HA 2026.2+)
# The dashboards dict uses None as key for the default dashboard
dashboards = lovelace_data.dashboards
if dashboard_url is None:
    dashboard = dashboards.get(None)  # None key = default dashboard
else:
    dashboard = dashboards.get(dashboard_url)
```

### Step 3: Backward Compatibility

The `LovelaceData` dataclass has been in Home Assistant for some time. For safety, we can use `hasattr()` checks to verify the expected structure exists before accessing properties.

## Files to Modify

- `custom_components/ai_agent_ha/agent.py` (lines 1986-2101)

## Testing

After implementation:
1. Verify dashboards are retrieved correctly
2. Verify dashboard config retrieval works
3. Ensure no deprecation warnings are logged
4. Test with both default and custom dashboards

## References

- [Home Assistant Core - Lovelace __init__.py](https://github.com/home-assistant/core/blob/dev/homeassistant/components/lovelace/__init__.py) - Source code showing LovelaceData dataclass
- [GitHub Issue #861 - DW deprecated warning](https://github.com/dwainscheeren/dwains-lovelace-dashboard/issues/861)
- [Home Assistant Multiple Dashboards Documentation](https://www.home-assistant.io/dashboards/dashboards/)
