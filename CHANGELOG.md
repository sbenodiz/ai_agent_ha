# Changelog

All notable changes to the AI Agent HA project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-04-11

### Fixed
- **Dashboard YAML Recovery**: LLM responses returning YAML instead of JSON for dashboard requests are now automatically recovered via `yaml.safe_load()` and routed to the visual dashboard preview flow
- Strengthened system prompt with a CRITICAL JSON-only output block at the very top to prevent LLMs (especially Claude Opus via Ask Sage) from returning YAML

### Changed
- Added YAML→JSON recovery path in `JSONDecodeError` handler — detects dashboard-like YAML indicators (`title:`, `views:`, `cards:`) and converts to `dashboard_suggestion` JSON before falling back to `final_response`
- Added reinforcement text after dashboard JSON structure in both `SYSTEM_PROMPT` and `SYSTEM_PROMPT_LOCAL`

## [Unreleased]

### Fixed
- **CRITICAL**: Fixed Anthropic API system prompt being overwritten by data payloads
  - Data responses now correctly use "user" role instead of "system" role
  - Ensures Claude receives all formatting instructions for dashboard/automation creation
  - Resolves issue where Claude would return YAML in `final_response` instead of JSON with `dashboard_suggestion`
- Fixed climate dashboard creation for users with only temperature/humidity sensors (no climate.* entities)
  - Added `get_entities_by_device_class()` helper function
  - Added `get_climate_related_entities()` to combine climate.* entities with temperature/humidity sensors

### Added
- `get_entities_by_device_class(device_class, domain)` function to filter entities by device_class attribute
- `get_climate_related_entities()` function for comprehensive climate dashboard support
  - Includes climate.* entities (thermostats, HVAC)
  - Includes sensor.* entities with device_class: temperature
  - Includes sensor.* entities with device_class: humidity
  - Automatic deduplication to prevent duplicate entities
- Enhanced dashboard templates with temperature/humidity sensor support
  - History graphs for temperature and humidity visualization
  - Entity cards showing current sensor values
  - Properly categorized sensor groups by device_class
- Updated Anthropic provider to use Claude Sonnet 4.5 as default model
- Added `claude-sonnet-4-5-20250929` to available Anthropic models
- Enhanced `get_entity_registry()` to include device_class, state_class, and unit_of_measurement attributes
- Device class guidance in system prompts for improved AI understanding
- Unit tests for new climate-related functions and critical system prompt fix

## [0.99.6] - 2025-11-05
### Fixed
- Fixed UI issue with Clear Chat button overlap
- Improved UI layout and responsiveness

### Added
- Added local frontend testing capability for development
- Enhanced test infrastructure for frontend development

## [0.99.5] - 2025-11-04
### Added
- Support for GPT-5 model from OpenAI
- Added GPT-5 to the list of available OpenAI models

### Fixed
- Fixed linting issues throughout codebase
- Improved code quality and consistency

## [0.99.4] - 2025-11-03
### Fixed
- Fixed test suite issues
- Improved test coverage and reliability
- Resolved issue #16 related to test failures

## [0.99.3] - 2025-07-04
### Changed
- **Breaking**: Now requires Python 3.12+ for Home Assistant compatibility
- Updated all GitHub Actions workflows to use Python 3.12
- Updated mypy configuration for Python 3.12 compatibility
- Improved type annotations throughout codebase

### Fixed
- Fixed mypy type checking errors with Home Assistant 2025.1.x
- Fixed code formatting issues with black formatter
- Fixed test compatibility with Python 3.12
- Resolved CI/CD pipeline failures

### Added
- Comprehensive documentation updates for Python 3.12 requirement
- Enhanced development environment setup instructions
- Better error handling for AI provider imports

## [0.99.2] - Previous Release
### Added
- Contribution guidelines for the project
- Issue and pull request templates
- Code of Conduct
- Security policy
- Development guide
- Changelog

## [1.0.0] - YYYY-MM-DD (Replace with actual release date)
### Added
- Initial release of AI Agent HA
- Support for multiple AI providers (OpenAI, Google Gemini, Anthropic Claude, OpenRouter, Llama)
- Entity control through natural language
- Automation creation
- Dashboard creation
- Entity state queries
- Home Assistant panel integration
- Configuration flow setup
- Documentation

## How to Update This Changelog

For each new release, create a new section with:
- `[version number] - YYYY-MM-DD` as the heading
- Group changes under the following subheadings as needed:
  - **Added** - for new features
  - **Changed** - for changes in existing functionality
  - **Deprecated** - for soon-to-be removed features
  - **Removed** - for now removed features
  - **Fixed** - for bug fixes
  - **Security** - for security improvements and fixes
  
Example:
```
## [1.1.0] - 2023-12-15
### Added
- New feature X
- New provider Y

### Changed
- Improved handling of Z

### Fixed
- Bug in feature A
```

When adding items to the Unreleased section, follow the same format. When creating a release, rename "Unreleased" to the new version number and release date, then create a new "Unreleased" section. 