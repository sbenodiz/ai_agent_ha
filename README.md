# AI Agent HA Extended

**Actively maintained fork of AI Agent HA — with expanded provider support, bug fixes, and enhanced UI**

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/JLay2026/ai_agent_ha)
![Version](https://img.shields.io/badge/version-v1.08.15-blue.svg)

---

## What's new in this fork vs the original

| Feature | Upstream | This Fork |
|---|---|---|
| Ask Sage AI provider | ❌ | ✅ |
| Local model (LM Studio/Ollama) auto-detect | Basic | ✅ Full OpenAI-compat |
| HA context sent to cloud models | ❌ Broken | ✅ Fixed |
| Dashboard suggestion rendering | ❌ Broken | ✅ Fixed |
| Conversation history cap (prevents memory bloat) | ❌ | ✅ 50-entry cap |
| History rollback on error | ❌ | ✅ |
| Concurrent query protection | ❌ | ✅ asyncio.Lock |
| Ask Sage overload retry | ❌ | ✅ 3× backoff |
| Thinking/reasoning strip | ❌ | ✅ |
| Anthropic Claude updated models | ❌ claude-sonnet-4-5 | ✅ claude-opus-4-6 |
| Test coverage | ❌ | ✅ 114 tests |
| Active releases | Irregular | ✅ v1.08.x series |

---

## Quick Install (HACS)

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/JLay2026/ai_agent_ha)

1. Open **HACS** in your Home Assistant instance.
2. Click the **three-dot menu** (⋮) in the top-right corner and select **Custom repositories**.
3. Enter the repository URL:
   ```
   https://github.com/JLay2026/ai_agent_ha
   ```
4. Set **Category** to `Integration` and click **Add**.
5. Search for **AI Agent HA Extended** in HACS and click **Download**.
6. Restart Home Assistant.
7. Go to **Settings → Devices & Services → Add Integration** and search for **AI Agent HA**.

---

## Supported AI Providers

This fork supports **9 AI providers**:

| Provider | Notes |
|---|---|
| **OpenAI** | GPT-4o, GPT-4, GPT-3.5, and all current models via API key |
| **Google Gemini** | Gemini Pro and Flash series |
| **Anthropic (Claude)** | Updated to latest models including `claude-opus-4-6` |
| **OpenRouter** | Access to many models via a single API key |
| **Alter** | Alter AI provider integration |
| **z.ai** | z.ai cloud provider |
| **Llama** | Meta Llama model support |
| **Local Model (LM Studio / Ollama)** | Auto-detects any OpenAI-compatible local server. Works with LM Studio, Ollama with `/v1` endpoint, and any server implementing the OpenAI API spec |
| **Ask Sage** *(new in this fork)* | Enterprise-grade secure cloud AI. Authenticates via `x-access-tokens` header. Supports live web search toggle (off / news / all) and deep agent mode. Model list is live-fetched at config time (government-restricted models are automatically excluded) |

---

## Features

### Home Assistant Context (Fixed in this fork)
- Session-persistent context injection: entities, weather, and history are sent to the AI on every query
- HA context delivery to cloud models was broken upstream — fully fixed in v1.08.10

### Dashboard Creation
- Clean dashboard suggestion cards rendered with `ha-card`, including icon, title, summary, and action buttons
- Multi-JSON response parsing fixed in v1.08.13; UI overhauled in v1.08.14

### Automation Creation
- Suggest and create Home Assistant automations directly from the chat interface

### Conversation Management
- 50-entry conversation history cap prevents memory bloat over long sessions
- History rollback on error keeps conversation state clean after failed requests
- Concurrent query protection via `asyncio.Lock` prevents race conditions

### Reliability & Quality
- Ask Sage overload retry with exponential backoff (3× with 1s / 2s / 4s delays)
- Thinking/reasoning token stripping for models that emit reasoning traces
- 114 automated tests covering core functionality

### UI Improvements
- Static provider · model label in the chat footer (replaced dropdown) for cleaner UI
- Dashboard suggestion cards with ha-card styling, icons, and action buttons

---

## Installation

### Via HACS (Recommended)

See [Quick Install](#quick-install-hacs) above.

### Manual Installation

1. Download or clone this repository:
   ```bash
   git clone https://github.com/JLay2026/ai_agent_ha.git
   ```
2. Copy the `custom_components/ai_agent_ha` directory into your Home Assistant `config/custom_components/` folder:
   ```
   config/
   └── custom_components/
       └── ai_agent_ha/
   ```
3. Restart Home Assistant.
4. Go to **Settings → Devices & Services → Add Integration** and search for **AI Agent HA**.

---

## Configuration

Navigate to **Settings → Devices & Services**, find the **AI Agent HA** integration, and click **Configure**.

### OpenAI
- **API Key**: Your OpenAI API key from [platform.openai.com](https://platform.openai.com)
- **Model**: Select from available GPT models

### Google Gemini
- **API Key**: Your Google AI API key from [aistudio.google.com](https://aistudio.google.com)
- **Model**: Select Gemini Pro or Flash variant

### Anthropic (Claude)
- **API Key**: Your Anthropic API key from [console.anthropic.com](https://console.anthropic.com)
- **Model**: Includes latest models up to `claude-opus-4-6`

### OpenRouter
- **API Key**: Your OpenRouter API key from [openrouter.ai](https://openrouter.ai)
- **Model**: Any model available on OpenRouter

### Alter / z.ai / Llama
- **API Key**: Provider-specific API key
- **Model**: Provider-available models

### Local Model (LM Studio / Ollama)
- **Base URL**: URL of your local OpenAI-compatible server (e.g., `http://localhost:1234/v1` for LM Studio, `http://localhost:11434/v1` for Ollama)
- **Model**: Model name as reported by your local server
- Auto-detects any OpenAI-compatible endpoint — no special configuration needed beyond the base URL

### Ask Sage *(new)*
- **Token**: Your Ask Sage `x-access-tokens` authentication token
- **Model**: Selected from a live-fetched list of available models (government-restricted models are automatically excluded)
- **Live Search**: Toggle web search context — `off`, `news`, or `all`
- **Deep Agent Mode**: Enable for more thorough, multi-step reasoning responses

---

## Upstream Attribution

This is a fork of [sbenodiz/ai_agent_ha](https://github.com/sbenodiz/ai_agent_ha). All original work is credited to the upstream author. This fork adds bug fixes and new features while remaining open to upstream merges.

---

## Changelog

### v1.08.15
Dashboard suggestion card upgraded to native HA components — `ha-card` with elevation shadow, `ha-chip` badges for each view, and `ha-icon` with primary color accent. Matches the look and feel of native Home Assistant UI.

### v1.08.14
Clean dashboard suggestion card UI — `ha-card` components with icon, title, summary, and action buttons for a polished, native-looking suggestion experience.

### v1.08.13
Fixed dashboard and automation suggestion rendering — resolves multi-JSON response parsing that caused suggestions to fail to display.

### v1.08.12
Replaced model dropdown with a static provider · model label in the chat footer for a cleaner, less cluttered UI.

### v1.08.11
Ask Sage retry on overload with exponential backoff — 3 retries with 1s, 2s, and 4s delays before surfacing an error to the user.

### v1.08.10
Fixed Ask Sage `system_prompt` — HA context (entities, weather, history) is now correctly injected into all Ask Sage queries.

### v1.08.9
Anthropic model updates (`claude-opus-4-6`), temperature and timeout fixes; Ask Sage live search and `deep_agent` toggles added; data scope validation improvements.

---

## Contributing

Pull requests are welcome. For significant changes, please open an issue first to discuss what you'd like to change.

## License

See [LICENSE](LICENSE) in this repository.
