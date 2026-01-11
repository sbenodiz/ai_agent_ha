# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with AI Agent HA.

## Table of Contents

- [Getting Help](#getting-help)
- [Enabling Debug Logs](#enabling-debug-logs)
- [Collecting and Sharing Logs](#collecting-and-sharing-logs)
- [Common Issues](#common-issues)
  - [Climate vs Temperature/Humidity Sensors](#climate-vs-temperaturehumidity-sensors)
  - [API Key Issues](#api-key-issues)
  - [Model Not Found](#model-not-found)
  - [Timeout Errors](#timeout-errors)
  - [Entity Not Found](#entity-not-found)
  - [Dashboard Not Appearing](#dashboard-not-appearing)
  - [Configuration Not Showing Up](#configuration-not-showing-up)
  - [Frontend Not Updating](#frontend-not-updating)
- [Reporting Issues](#reporting-issues)

## Getting Help

If you encounter issues:

1. Check this troubleshooting guide for common solutions
2. Enable debug logs (see below) to get detailed information
3. Search [existing issues](https://github.com/sbenodiz/ai_agent_ha/issues) to see if your problem has been reported
4. If needed, create a new issue with detailed information and logs

## Enabling Debug Logs

Debug logs provide detailed information about what AI Agent HA is doing and can help identify issues.

### Step 1: Enable Debug Logging

Add the following to your Home Assistant `configuration.yaml` file:

```yaml
logger:
  default: info
  logs:
    custom_components.ai_agent_ha: debug
```

### Step 2: Restart Home Assistant

After adding the logging configuration, restart Home Assistant for the changes to take effect.

### Step 3: Reproduce the Issue

Once debug logging is enabled, reproduce the issue you're experiencing. This ensures the relevant information is captured in the logs.

## Collecting and Sharing Logs

### Method 1: Using Home Assistant UI (Recommended)

1. Go to **Settings** → **System** → **Logs**
2. Click the **Load Full Logs** button
3. Use the search box to filter for `ai_agent_ha`
4. Copy the relevant log entries

### Method 2: Using Log File

1. Navigate to your Home Assistant configuration directory
2. Open the `home-assistant.log` file
3. Search for entries containing `ai_agent_ha`
4. Copy the relevant log entries

### What to Include

When sharing logs, include:
- Timestamps of when the issue occurred
- Any error messages or warnings
- The full log entries for `ai_agent_ha` around the time of the issue
- The request you made to the AI (if applicable)

**Note**: AI Agent HA automatically sanitizes API keys and tokens in debug logs, so they won't be exposed. However, you may still want to remove personal information like addresses or specific device names if you prefer.

## Common Issues

### Climate vs Temperature/Humidity Sensors

**Issue**: You have temperature and humidity sensors, but the AI says "no climate entities found" when you ask to create a climate dashboard.

**Explanation**: This is a common confusion. In Home Assistant terminology:
- **Climate entities** (`climate.*`) are devices that can control temperature, such as thermostats, HVAC systems, and air conditioners
- **Sensor entities** (`sensor.*`) only measure and report data (like temperature and humidity) but cannot control anything

**Solution**:

If you want to create a dashboard with your temperature and humidity sensors:

```
"Create a dashboard showing all my temperature and humidity sensors"
```

Or be more specific:

```
"Create a sensor dashboard with all temperature and humidity readings"
```

If you actually have thermostats or HVAC controls and they're not being found:

1. Verify your climate devices are properly configured in Home Assistant:
   - Go to **Settings** → **Devices & Services**
   - Check that your thermostats/HVAC systems are showing up
   - Verify they have entity IDs starting with `climate.`

2. Try being more specific in your request:
   ```
   "Create a dashboard with my Nest thermostat and bedroom AC"
   ```

3. Check the entity states:
   - Go to **Developer Tools** → **States**
   - Search for `climate.`
   - Verify your climate entities are available and have valid states

### API Key Issues

**Symptoms**:
- "Invalid API key" errors
- "Unauthorized" responses
- Authentication failures

**Solutions**:

1. **Verify your API key is correct**:
   - Copy the key directly from your provider's dashboard
   - Ensure there are no extra spaces or characters
   - Check that the key hasn't expired

 2. **Check provider-specific requirements**:
    - **OpenAI**: Keys start with `sk-`
    - **Anthropic (Claude)**: Keys start with `sk-ant-`
    - **Google Gemini**: Get key from [Google AI Studio](https://aistudio.google.com/app/apikey)
    - **OpenRouter**: Get key from [OpenRouter](https://openrouter.ai/keys)
    - **Alter**: Get key from [Alter HQ](https://alterhq.com)
    - **z.ai**: Get key from [Z.ai Platform](https://z.ai/manage-apikey/apikey-list)
    - **Llama**: Get key from your Llama provider

3. **Reconfigure the integration**:
   - Go to **Settings** → **Devices & Services**
   - Find **AI Agent HA**
   - Click **Configure** and re-enter your API key

4. **Check API quota and billing**:
   - Verify your provider account has available credits
   - Check if you've exceeded rate limits
   - Ensure billing is set up correctly

### Model Not Found

**Symptoms**:
- "Model not found" errors
- "Invalid model" messages
- Model validation failures

**Solutions**:

1. **Verify the model name**:
   - Check your provider's documentation for exact model names
   - Model names are case-sensitive
   - Use the exact format from the provider

 2. **Check model availability**:
    - **OpenAI**: See [OpenAI Models](https://platform.openai.com/docs/models)
    - **Anthropic**: See [Claude Models](https://docs.anthropic.com/en/docs/models-overview)
    - **Google Gemini**: See [Gemini Models](https://ai.google.dev/models)
    - **OpenRouter**: See [OpenRouter Models](https://openrouter.ai/models)
    - **Alter**: Check your Alter provider's documentation
    - **z.ai**: See [Z.ai Documentation](https://docs.z.ai/api-reference/introduction)
    - **Llama**: Check your Llama provider's documentation

3. **Try a predefined model**:
   - In AI Agent HA settings, use the dropdown to select a predefined model
   - This ensures you're using a valid model name

4. **Check your API plan**:
   - Some models require specific API tiers or plans
   - Verify your account has access to the model you're trying to use

### Timeout Errors

**Symptoms**:
- Requests taking too long
- Timeout errors
- No response from AI

**Solutions**:

1. **Check network connectivity**:
   - Verify your Home Assistant instance can reach the internet
   - Test connectivity: `ping api.openai.com` (or your provider's API endpoint)
   - Check firewall rules

 2. **Verify provider status**:
    - Check if your AI provider is experiencing outages
    - OpenAI: https://status.openai.com/
    - Anthropic: https://status.anthropic.com/
    - Google: https://status.cloud.google.com/
    - z.ai: Check [Z.ai status page](https://docs.z.ai)

3. **Simplify your request**:
   - Complex requests may take longer to process
   - Try breaking down large requests into smaller ones
   - Reduce the amount of context in your question

4. **Check system resources**:
   - Verify Home Assistant has sufficient CPU and memory
   - Look for high system load in **Settings** → **System**

### Entity Not Found

**Symptoms**:
- AI says entities don't exist
- Cannot find specific devices
- "No entities available" messages

**Solutions**:

1. **Verify the entity exists**:
   - Go to **Developer Tools** → **States**
   - Search for the entity name
   - Check that it appears in the list

2. **Check entity naming**:
   - Use the exact entity ID (e.g., `light.living_room`)
   - Or use the friendly name shown in Home Assistant
   - Entity names are case-sensitive

3. **Verify entity is available**:
   - Check that the device is online and responding
   - Look for "unavailable" or "unknown" states
   - Restart the device integration if needed

4. **Use broader terms**:
   - Instead of "my Philips Hue bulb", try "living room light"
   - Instead of specific names, use room or device type
   - Let the AI discover entities based on type/location

5. **Check entity permissions**:
   - Ensure AI Agent HA has permission to access all entities
   - No specific entity restrictions are configured

### Dashboard Not Appearing

**Symptoms**:
- Dashboard created but not visible
- Missing from sidebar
- Cannot find dashboard

**Solutions**:

1. **Restart Home Assistant** (Most Common Fix):
   - Dashboards require a restart to appear in the sidebar
   - Go to **Settings** → **System** → **Restart**
   - Wait for Home Assistant to fully restart

2. **Check dashboard was created**:
   - Go to **Settings** → **Dashboards**
   - Look for your dashboard in the list
   - Verify it's enabled and not hidden

3. **Verify sidebar settings**:
   - The dashboard should be set to "Show in sidebar"
   - Check **Settings** → **Dashboards** → [Your Dashboard] → **Settings**

4. **Clear browser cache**:
   - Hard refresh: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac)
   - Clear browser cache for Home Assistant
   - Try a different browser or incognito mode

5. **Check for errors**:
   - Enable debug logs (see above)
   - Look for dashboard creation errors
   - Verify YAML syntax if you modified the dashboard

### Configuration Not Showing Up

**Symptoms**:
- Integration not appearing in Settings
- Cannot find AI Agent HA configuration
- Missing from Devices & Services

**Solutions**:

1. **Verify installation**:
   - Check that files are in `custom_components/ai_agent_ha/`
   - Verify all required files are present
   - Look for `manifest.json` in the directory

2. **Check for errors in logs**:
   - Go to **Settings** → **System** → **Logs**
   - Look for errors mentioning `ai_agent_ha`
   - Common issues: missing dependencies, syntax errors

3. **Restart Home Assistant**:
   - After installation, always restart
   - Go to **Settings** → **System** → **Restart**

4. **Clear Home Assistant cache**:
   ```bash
   rm -rf /config/.storage/core.restore_state
   ```
   Then restart Home Assistant

5. **Reinstall the integration**:
   - Remove the integration completely
   - Delete the `custom_components/ai_agent_ha` directory
   - Reinstall via HACS or manually
   - Restart Home Assistant

### Frontend Not Updating

**Symptoms**:
- UI changes not appearing
- Old interface still showing
- Style changes not applied

**Solutions**:

1. **Clear browser cache**:
   - Hard refresh: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac)
   - Clear browser cache completely
   - Close and reopen browser

2. **Disable browser cache** (for testing):
   - Open browser DevTools (F12)
   - Go to Network tab
   - Check "Disable cache"
   - Reload the page

3. **Clear Home Assistant frontend cache**:
   - Add `?refresh` to the URL
   - Example: `http://your-ha:8123/ai_agent_ha?refresh`

4. **Check file paths**:
   - Verify panel files are in correct location
   - Check file permissions
   - Look for JavaScript console errors (F12 → Console)

## Reporting Issues

When reporting issues, please include:

### 1. System Information
- Home Assistant version
- AI Agent HA version
- Python version
- Installation method (HACS or manual)

### 2. Configuration
- AI provider being used
- Model name
- Any custom configuration (remove sensitive data)

### 3. Steps to Reproduce
1. What you asked the AI
2. What you expected to happen
3. What actually happened

### 4. Logs
- Enable debug logging (see above)
- Include relevant log entries
- Remove sensitive information (API keys, tokens, addresses)

### 5. Screenshots (if applicable)
- Error messages
- UI issues
- Configuration screens

### Where to Report

- **Bugs**: [GitHub Issues](https://github.com/sbenodiz/ai_agent_ha/issues)
- **Questions**: [GitHub Discussions](https://github.com/sbenodiz/ai_agent_ha/discussions)
- **Security Issues**: See [Security Policy](../SECURITY.md)

### Issue Template

Use this template when creating an issue:

```markdown
## System Information
- Home Assistant version:
- AI Agent HA version:
- AI Provider:
- Model:

## Description
[Brief description of the issue]

## Steps to Reproduce
1.
2.
3.

## Expected Behavior
[What you expected to happen]

## Actual Behavior
[What actually happened]

## Logs
```
[Paste relevant logs here - remember to remove sensitive information]
```

## Screenshots
[If applicable]

## Additional Context
[Any other relevant information]
```

---

**Need more help?** Check out the other documentation:
- [Dashboard Creation Guide](DASHBOARD_CREATION.md)
- [Development Guide](DEVELOPMENT.md)
- [Contributing Guidelines](../CONTRIBUTING.md)
- [Main README](../README.md)
