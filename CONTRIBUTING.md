# Contributing to AI Agent HA

Thank you for your interest in contributing to AI Agent HA! This document provides guidelines and information for contributors to help make the process smooth and collaborative.

## 🚀 Getting Started

### Prerequisites

- Home Assistant 2023.3+ (for testing)
- Python 3.12+
- Git
- Basic understanding of Home Assistant custom components
- API keys for at least one supported AI provider (OpenAI, Google Gemini, Anthropic, OpenRouter, Alter, z.ai, or Llama)

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/ai_agent_ha.git
   cd ai_agent_ha
   ```

3. **Set up Home Assistant development environment**:
   - Install Home Assistant in development mode
   - Copy the `custom_components/ai_agent_ha` folder to your Home Assistant `custom_components` directory
   - Restart Home Assistant

4. **Configure the integration**:
   - Go to Settings → Devices & Services → Add Integration
   - Search for "AI Agent HA"
   - Configure with your AI provider credentials

## 🏗️ Project Structure

```
ai_agent_ha/
├── custom_components/ai_agent_ha/
│   ├── __init__.py              # Integration initialization
│   ├── agent.py                 # Core AI agent logic
│   ├── config_flow.py           # Configuration flow
│   ├── const.py                 # Constants and configuration
│   ├── dashboard_templates.py   # Dashboard creation templates
│   ├── frontend/                # Frontend chat interface
│   ├── services.yaml            # Service definitions
│   └── translations/            # Localization files
├── docs/                        # Documentation
├── README.md                    # Project documentation
└── hacs.json                    # HACS configuration
```

## 📋 How to Contribute

### Reporting Issues

Before creating an issue, please:
1. **Check existing issues** to avoid duplicates
2. **Use the latest version** of the integration
3. **Test with multiple AI providers** if applicable

When reporting issues, include:
- Home Assistant version
- AI Agent HA version
- AI provider and model used
- Detailed steps to reproduce
- Error logs (if any)
- Expected vs actual behavior

### Suggesting Features

Feature requests are welcome! Please:
1. **Check existing feature requests** first
2. **Describe the use case** clearly
3. **Explain the expected behavior**
4. **Consider multiple AI providers** if applicable

### Code Contributions

#### Types of Contributions

- **Bug fixes**: Fix reported issues
- **New features**: Add new functionality
- **AI provider support**: Add support for new AI providers
- **Documentation**: Improve docs and examples
- **Translations**: Add or update language translations
- **Frontend improvements**: Enhance the chat interface
- **Dashboard templates**: Add new dashboard creation templates

#### Development Guidelines

##### General Code Style

- Follow **PEP 8** Python style guide
- Use **type hints** for all functions and methods
- Write **descriptive variable and function names**
- Add **docstrings** for all public functions and classes
- Keep functions **focused and single-purpose**

##### Home Assistant Specific

- Follow [Home Assistant development guidelines](https://developers.home-assistant.io/docs/development_guidelines)
- Use Home Assistant's **logging framework**
- Handle **exceptions gracefully**
- Use **async/await** for I/O operations
- Follow Home Assistant's **configuration flow patterns**

##### AI Provider Integration

When adding support for new AI providers:

1. **Create a new client class** in `agent.py`
2. **Follow the existing pattern** of other providers
3. **Add provider configuration** to `config_flow.py`
4. **Update constants** in `const.py`
5. **Add models list** to the configuration
6. **Test with multiple models** from the provider
7. **Update documentation** with setup instructions

Example provider implementation:
```python
class NewProviderClient:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
    
    async def query(self, prompt: str, context: dict) -> str:
        # Implementation here
        pass
```

##### Frontend Development

- Use **vanilla JavaScript** for compatibility
- Follow **accessibility guidelines**
- Test on **multiple browsers**
- Maintain **responsive design**
- Keep **consistent styling** with Home Assistant

##### Dashboard Templates

When adding new dashboard templates:

1. **Add template to** `dashboard_templates.py`
2. **Use generic entity patterns** (avoid hardcoded entity IDs)
3. **Include appropriate card types** for each entity
4. **Test with different entity configurations**
5. **Add documentation** for the template

#### Testing

Before submitting a pull request:

1. **Test with multiple AI providers** and models
2. **Test common use cases**:
   - Light control
   - Climate control
   - Automation creation
   - Dashboard creation
   - Entity state queries
3. **Test error handling** with invalid requests
4. **Check Home Assistant logs** for errors
5. **Test configuration flow** setup and options

#### Pull Request Process

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the guidelines above

3. **Test thoroughly** with different scenarios

4. **Update documentation** if needed:
   - Update README.md for new features
   - Add or update docstrings
   - Update configuration examples

5. **Commit with clear messages**:
   ```bash
   git commit -m "feat: add support for new AI provider"
   git commit -m "fix: handle timeout errors gracefully"
   git commit -m "docs: update configuration examples"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a pull request** with:
   - Clear title and description
   - List of changes made
   - Testing performed
   - Screenshots (if UI changes)
   - Breaking changes (if any)

## 🌐 Internationalization

To add or update translations:

1. **Check existing translations** in `translations/`
2. **Use the English file** (`en.json`) as a template
3. **Follow the JSON structure** exactly
4. **Test with Home Assistant** in the target language
5. **Update the language list** in documentation

## 📚 Documentation

### Code Documentation

- **Document all public functions** with docstrings
- **Include parameter types** and return types
- **Provide usage examples** for complex functions
- **Document configuration options** clearly

### User Documentation

- **Update README.md** for new features
- **Add configuration examples** for new providers
- **Include troubleshooting tips** for common issues
- **Add screenshots** for UI changes

## 🐛 Debugging

### Common Issues

1. **API key issues**: Check provider documentation
2. **Model not found**: Verify model name with provider
3. **Timeout errors**: Check network connectivity
4. **Entity not found**: Verify entity exists in Home Assistant

### Debug Logging

Enable debug logging in Home Assistant:
```yaml
logger:
  default: info
  logs:
    custom_components.ai_agent_ha: debug
```

## 📞 Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Home Assistant Community**: For Home Assistant specific questions

## 🎯 Code of Conduct

### Our Pledge

We are committed to creating a welcoming and inclusive environment for all contributors, regardless of experience level, background, or identity.

### Expected Behavior

- Be respectful and constructive in all interactions
- Welcome newcomers and help them get started
- Focus on the technical merits of contributions
- Provide helpful feedback and suggestions
- Acknowledge the work of others

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Personal attacks or inflammatory language
- Spam or off-topic discussions
- Sharing private information without consent

## 🏆 Recognition

Contributors are recognized in:
- README.md acknowledgments
- Git commit history
- Release notes for significant contributions

## 📄 License

By contributing to AI Agent HA, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to AI Agent HA! Your efforts help make Home Assistant more intelligent and user-friendly. 🤖✨