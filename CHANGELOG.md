# Changelog

All notable changes to the AI Agent HA project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.14] - 2026-06-06

### Fixed
- **Anthropic provider failed every query with `Anthropic API error 400` on larger installs** ([#80](https://github.com/sbenodiz/ai_agent_ha/issues/80))
  - Root cause (confirmed live against the real Anthropic API): data requests such as
    `get_entity_registry` dumped the *entire* install (800KB+ on ~2000+ entity setups)
    into a single conversation message, exceeding Claude's 200k-token context window.
    The API rejected it with `400 prompt is too long`, the integration discarded the
    error body, retried the identical payload 10 times, and the oversized message
    persisted in conversation history — making **every subsequent query fail too**.
  - Data responses are now capped (~50k chars) with a truncation notice instructing
    the model to request narrower data; the per-request message window is also capped
    so multiple data messages can't stack past context or rate-limit budgets.
  - Provider error bodies are now surfaced to the UI (e.g.
    `Anthropic API error 400: prompt is too long: 205547 tokens > 200000 maximum`)
    instead of a bare status code.
  - Deterministic 4xx errors are no longer retried (previously 10 futile retries and
    ~45s of backoff); HTTP 429 now honors the server's `retry-after` header so
    per-minute token rate limit windows (e.g. Anthropic tier 1: 30k input tokens/min)
    can actually reset between attempts.
  - Failed queries are rolled back from conversation history so they can't poison
    subsequent queries; the history window can no longer start mid-turn after slicing.
  - Anthropic request timeout raised from 30s to 300s, matching every other provider.
  - Agent loop allows up to 8 iterations (was 5) since capped data responses can
    require additional narrower data requests.

## [1.13] - 2026-06-01

### Fixed
- **OpenAI provider crashed with `'list' object has no attribute 'strip'`** ([#75](https://github.com/sbenodiz/ai_agent_ha/issues/75))
  - The OpenAI Responses API (`/v1/responses`, the default for the OpenAI provider)
    returns the assistant text nested in `output[].content[].text`, and the raw HTTP
    body has no top-level `output_text` (that field is an SDK-only convenience
    property). The response parser fell through to a fallback that returned the
    `content` **list** instead of a string, so the agent then called `.strip()` on a
    list and failed on every retry — the sidebar showed "AI client error on attempt
    1" repeated 10 times for any prompt.
  - The parser now descends into `output[].content[]` and concatenates every
    `output_text` block into a string, and correctly skips non-text items such as the
    leading `reasoning` item emitted by o-series models.
  - Added a defensive guard so any client that returns a non-string degrades
    gracefully instead of exhausting all retries with an opaque `AttributeError`.

## [1.12] - 2026-05-27

### Fixed
- **OpenAI-Compatible provider was missing the API key field** ([#70](https://github.com/sbenodiz/ai_agent_ha/issues/70))
  - The runtime read `openai_compatible_api_key` from config, but the config flow never
    collected it, so the `Authorization: Bearer ...` header was never sent. Authenticated
    gateways (Open WebUI, LiteLLM, hosted vLLM) returned 401 even when the same key
    worked in the gateway's own UI.
  - Added an optional API key field to both the initial config and the options/edit flow.
- **OpenAI provider with a custom Base URL hit `/responses`** ([#70](https://github.com/sbenodiz/ai_agent_ha/issues/70))
  - Third-party "OpenAI-compatible" servers implement `/chat/completions`, not OpenAI's
    newer Responses API. The OpenAI client now routes to `/chat/completions` when the
    Base URL is not `api.openai.com`, and keeps the Responses API for real OpenAI.

## [1.11] (and earlier accumulated changes)

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