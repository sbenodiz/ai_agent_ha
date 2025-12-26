# Development Guide for AI Agent HA

This guide provides information about the development workflow and technical aspects of the AI Agent HA project.

## Development Environment Setup

### Prerequisites

- Python 3.12+
- Git
- Home Assistant development environment
- API keys for one or more supported AI providers (OpenAI, Google Gemini, Anthropic, OpenRouter, Alter, z.ai, Llama)

### Setting Up a Development Environment

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/ai_agent_ha.git
   cd ai_agent_ha
   ```

2. **Set up a Home Assistant development environment**:
   - Option 1: Use a dedicated Home Assistant development instance
   - Option 2: Use a test Home Assistant container
   - Option 3: Use Home Assistant Core in development mode

   For details on setting up a Home Assistant development environment, see the [Home Assistant Developer Documentation](https://developers.home-assistant.io/docs/development_environment).

3. **Install the integration in development mode**:
   - Symlink or copy the `custom_components/ai_agent_ha` folder to your Home Assistant `custom_components` directory
   - Restart Home Assistant
   - Add the integration through the Home Assistant UI

## Project Architecture

### Core Components

- **__init__.py**: Integration initialization and setup
- **agent.py**: Core AI agent logic and AI provider integration
- **config_flow.py**: Configuration flow UI and logic
- **const.py**: Constants and configuration options
- **dashboard_templates.py**: Templates for dashboard creation
- **frontend/**: Frontend UI components
- **services.yaml**: Service definitions
- **translations/**: Localization files

### Data Flow

1. **User Input**: User provides a natural language request
2. **Agent Processing**:
   - Context collection (entities, states, etc.)
   - AI provider query
   - Response parsing and validation
3. **Action Execution**:
   - Service calls
   - Entity control
   - Automation/dashboard creation
4. **Response Delivery**: Results returned to user

## Development Workflow

### Adding a New Feature

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement your changes**:
   - Follow the coding standards in CONTRIBUTING.md
   - Add appropriate docstrings and comments
   - Update relevant documentation

3. **Test your changes**:
   - Test with multiple AI providers if applicable
   - Test with various user inputs
   - Check for edge cases and error handling

4. **Submit a pull request**:
   - Push your branch to your fork
   - Create a PR to the main repository
   - Fill out the PR template completely

### Adding Support for a New AI Provider

1. **Create a new client class** in `agent.py`:
   ```python
   class NewProviderClient:
       """Client for New AI Provider."""
       
       def __init__(self, api_key: str, model: str):
           """Initialize the client."""
           self.api_key = api_key
           self.model = model
           # Additional setup
           
       async def query(self, prompt: str, context: dict) -> str:
           """Send a query to the AI provider.
           
           Args:
               prompt: The user's prompt
               context: The context dictionary with Home Assistant data
               
           Returns:
               The AI provider's response
           """
           # Implementation here
   ```

2. **Add provider configuration** to `config_flow.py`:
   - Add provider option in the config flow
   - Create appropriate config fields
   - Add validation for the new provider's API key and models

3. **Update constants** in `const.py`:
   - Add provider name constant
   - Add default models list
   - Add any provider-specific configuration options

4. **Update documentation**:
   - Add provider setup instructions to README.md
   - Update configuration examples

### Frontend Development

1. **Understand the existing frontend**:
   - Examine `frontend/ai_agent_ha-panel.js`
   - Note Home Assistant frontend patterns

2. **Make frontend changes**:
   - Follow Home Assistant frontend guidelines
   - Test on multiple browsers and screen sizes
   - Ensure accessibility compliance

3. **Test frontend changes**:
   - Verify real-time updates
   - Check responsive design
   - Test with various AI responses

## Testing

### Manual Testing

Test the following scenarios:
- Light control: "Turn on the living room lights"
- Climate control: "Set the thermostat to 72 degrees"
- Entity state queries: "What's the temperature in the bedroom?"
- Automation creation: "Create an automation to turn off lights at 11 PM"
- Dashboard creation: "Create a security dashboard"
- Error handling: Test with invalid entities or requests
- Configuration flow: Test setup and options flow

### Debug Logs

Enable debug logging in Home Assistant:
```yaml
logger:
  default: info
  logs:
    custom_components.ai_agent_ha: debug
```

Check the Home Assistant logs at `<config_dir>/home-assistant.log` or in the "Logs" section of the Home Assistant UI.

## Common Development Tasks

### Adding a New Command Pattern

To add support for a new command pattern (e.g., a new type of automation):

1. **Identify the command pattern** in user prompts
2. **Update the AI prompt templates** with examples
3. **Add handling code** in the agent logic
4. **Test thoroughly** with various phrasings

### Adding a New Dashboard Template

To add a new dashboard template:

1. **Create the template** in `dashboard_templates.py`
2. **Design for flexibility** (avoid hardcoded entity IDs)
3. **Add documentation** for the template
4. **Test with different entity configurations**

### Troubleshooting Development Issues

- **Configuration not showing up**: Restart Home Assistant and clear browser cache
- **API key validation errors**: Verify API key format and permissions
- **Frontend not updating**: Check for JavaScript errors in browser console
- **Agent not responding**: Check debug logs for connection issues

## Release Process

1. **Update version number** in `manifest.json`
2. **Update CHANGELOG.md** with notable changes
3. **Create a release** on GitHub
4. **Tag the release** with the version number

## Additional Resources

- [Home Assistant Developer Documentation](https://developers.home-assistant.io/)
- [Home Assistant Custom Component Development](https://developers.home-assistant.io/docs/creating_component_index)
- [Home Assistant Frontend Development](https://developers.home-assistant.io/docs/frontend/custom-ui/)
- [AI Provider Documentation]:
  - [OpenAI API](https://platform.openai.com/docs/api-reference)
  - [Google Gemini API](https://ai.google.dev/tutorials/rest_quickstart)
  - [Anthropic Claude API](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
  - [OpenRouter API](https://openrouter.ai/docs)

## Getting Help

If you encounter issues during development:
- Check the [GitHub Discussions](https://github.com/sbenodiz/ai_agent_ha/discussions)
- Review existing issues and pull requests
- Ask questions in the discussions area

---

Happy coding! 🚀 