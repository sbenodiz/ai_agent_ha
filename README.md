# AI Agent HA Extended

**Local-first AI for Home Assistant — with secure cloud options when you want them.**

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/custom-components/hacs)
[![Version](https://img.shields.io/badge/version-v1.2.19.1-blue.svg)](https://github.com/JLay2026/ai_agent_ha/releases/tag/v1.2.19.1)

AI Agent HA Extended brings conversational AI to Home Assistant with a focus on privacy, reliability, and practical home automation. Run a local model on LM Studio and your data never leaves your network — or connect to cloud providers through a single integration with consistent security guarantees across all paths.

---

## Why This Fork

The upstream project ([sbenodiz/ai_agent_ha](https://github.com/sbenodiz/ai_agent_ha)) established the core architecture. This fork exists because local model support was left incomplete, dashboard and automation rendering were broken, and the HA community needed a maintained path forward.

- **30+ releases** since forking — actively maintained with subincrement hotfix versioning
- **8 open PRs** to upstream remain unmerged
- **114+ tests** covering conversation loop, provider integrations, and JSON parsing
- Not a competition with upstream — if improvements get merged back, great

---

## What This Fork Does That Upstream Doesn't

### AI-Generated Dashboards
Ask for a dashboard in natural language and get a visual preview in chat. The agent queries your actual HA entities, builds a complete Lovelace YAML configuration, and presents it for review before deployment. Supports new dashboards and adding views to existing ones.

### AI-Generated Automations
Describe an automation in plain English. The agent generates valid HA automation YAML, renders it as a reviewable card in chat, and deploys it on approval.

### Typed Response Envelope (v1.2.19+)
Every response from the AI is classified into a structured envelope — `text`, `dashboard`, or `automation` — before reaching the frontend. The panel switches on type rather than parsing raw JSON from model output. This eliminates prompt bleed (where multi-step model conversations leak into chat) and makes rendering deterministic across models and providers.

### Dashboard Visual/YAML Toggle
View generated dashboards as visual card previews or raw YAML. Toggle per-message or set a persistent default in settings. Client-side JSON-to-YAML conversion.

### Reliable Local Model Support
Full OpenAI-compatible auto-detection for LM Studio and Ollama. Robust fallback parsing for structured JSON output from smaller models. Thinking/reasoning tag stripping for models with chain-of-thought (Qwen 3, DeepSeek-R1).

### Multi-Provider Architecture
Single integration, consistent interface across 9+ providers. Switch models without reconfiguring. Ask Sage provides community-credited access to GPT-4o, Claude, and others with zero RAG ingestion enforced in code.

### Security by Design
- Local models: zero external network calls
- Ask Sage: `dataset: "none"` and `limit_references: 0` enforced on every query
- 50-entry conversation history cap
- Chat persistence opt-in, user-scoped, browser-local only
- No credential logging

---

## Current Capabilities (v1.2.19)

| Capability | Status |
|---|---|
| Natural language dashboard generation | ✅ Visual preview + one-click deploy |
| Natural language automation generation | ✅ Review card + approve/reject |
| Typed response envelope | ✅ Deterministic rendering across models |
| Dashboard visual/YAML toggle | ✅ Per-message + persistent default |
| Entity-aware queries | ✅ Agent fetches real HA states for context |
| Multi-step reasoning | ✅ Entity lookup → data injection → generation |
| Local models (LM Studio / Ollama) | ✅ Full OpenAI-compat support |
| Ask Sage (cloud broker) | ✅ Community credits, multi-model, zero RAG |
| Direct cloud providers | ✅ OpenAI, Anthropic, Gemini, OpenRouter, Alter, z.ai, Llama |
| Inline temperature charts | ✅ SVG bar chart in chat |
| Chat history persistence | ✅ Opt-in, user-scoped localStorage |
| Thinking/reasoning display | ✅ Show thinking toggle + debug trace panel |
| Concurrent query protection | ✅ asyncio.Lock |
| Query correlation | ✅ Prevents stale cross-query response contamination |

---

## Choose Your Path

### Path A — Fully Local (Privacy First)

| | |
|---|---|
| **Backend** | LM Studio or Ollama on your local network |
| **Minimum model** | 9B parameters recommended; Qwen 2.5 9B is the tested baseline |
| **Data sovereignty** | Zero external network calls — all HA entity data stays on-premise |
| **Honest UX** | Responses take 5–30 seconds depending on hardware. Structured JSON output is less reliable than cloud models, but the integration includes robust fallback parsing. |
| **Best for** | Privacy-conscious households, sensitive entity data, air-gapped setups |

### Path B — Secure Cloud (Ask Sage)

| | |
|---|---|
| **Auth** | Single token — no per-provider API keys |
| **Models** | GPT-4o, Claude Opus 4.6, and others through one broker |
| **Access** | Community credits available — lower barrier than direct provider subscriptions |
| **RAG enforcement** | `dataset: none` and `limit_references: 0` on every query — enforced in code |
| **Extra features** | Live web search toggle (off / news / all), deep agent mode |
| **Best for** | Users who want managed cloud access without juggling per-model subscriptions |

### Path C — Direct Cloud Providers

| | |
|---|---|
| **Providers** | OpenAI, Anthropic (claude-opus-4-6), Google Gemini, OpenRouter, Alter, z.ai, Llama |
| **Data handling** | Standard API terms per provider |
| **Best for** | Users with existing provider subscriptions |

---

## Roadmap

Active development priorities, roughly in order:

### Near-Term (v1.2.x)
| Feature | Description |
|---|---|
| **Response delivery hardening** | Poll-primary response channel with debug/thinking data preservation. Fixes edge cases where dashboards require a page refresh to appear. |
| **Post-refresh recovery** | Automatically recover the last response after a page refresh so in-progress dashboards aren't lost. |
| **UI unblock during loading** | Allow Clear Chat and new prompts while the AI is thinking. New prompts cancel the in-flight query. |
| **Header settings menu** | Consolidated gear menu for Show thinking, Dashboard display mode, and future toggles. |

### Mid-Term (v1.3.x)
| Feature | Description |
|---|---|
| **WebSocket push channel** | Dedicated WS command replacing the event bus for response delivery. Foundation for streaming and real-time notifications. |
| **Streaming support** | Token-by-token response streaming for supported providers (local models, OpenAI-compatible APIs). Blocked by Ask Sage API for that provider. |
| **Prompt response speed** | Reduce latency through WS-based dispatch, conversation history trimming, entity pre-fetching, and prompt caching. |
| **Dashboard editing mode** | Natural language interface for modifying existing dashboards and views — describe changes, preview a diff, deploy. |
| **Automation push to HA** | Approve button writes directly to HA's automation registry and triggers a reload. |
| **In-chat model switcher** | Clickable model pill in the UI for switching providers/models without leaving the chat. |

### Longer-Term
| Feature | Description |
|---|---|
| **Mushroom card styling** | Detect Mushroom Cards installation and use mushroom-* card types in generated dashboards. |
| **Temperature visual response card** | Inline temperature widget rendered directly in chat for weather/climate queries. |
| **Quick actions bar toggle** | Show/hide the predefined prompts bar. |
| **localStorage chat persistence improvements** | Enhanced chat history with provider-scoped storage. |

This roadmap reflects current development priorities and may shift based on community feedback and upstream changes.

---

## Quick Install (HACS)

1. Open **HACS** in Home Assistant.
2. Go to **Integrations** → click the three-dot menu → **Custom repositories**.
3. Enter `https://github.com/JLay2026/ai_agent_ha` and select **Integration**.
4. Search for **AI Agent HA Extended** and click **Download**.
5. Restart Home Assistant.
6. Go to **Settings → Devices & Services → Add Integration** and search for **AI Agent**.

---

## Local Model Setup

### LM Studio (Recommended)

1. **Install LM Studio** from [lmstudio.ai](https://lmstudio.ai) on a machine on your local network.
2. **Download a model** — Qwen 2.5 9B is the tested baseline. Use the search bar inside LM Studio: search `qwen2.5-9b`.
3. **Start the local server**: In LM Studio, go to the **Developer** tab → **Start Server** → default port is `1234`.
4. **Configure the integration in HA**:
   - Provider: **Local Model**
   - Base URL: `http://YOUR_LAN_IP:1234/v1`
   - API Key: any non-empty string (e.g. `lmstudio`)
   - Model name: exactly as shown in LM Studio's loaded model list
5. **Test** by sending a simple message: "What lights are on?"

### Ollama

Follow the same steps but set the base URL to `http://YOUR_LAN_IP:11434/v1`. Ollama's OpenAI-compatible endpoint is used automatically.

### Model Recommendations

| Model | Size | Quality | Notes |
|---|---|---|---|
| Qwen 2.5 9B | 9B | ✅ Good | Tested baseline, recommended starting point |
| Qwen 3 8B | 8B | ✅ Good | Strong instruction following |
| Llama 3.1 8B | 8B | ✅ Good | Reliable JSON output |
| Mistral 7B | 7B | ⚠️ Marginal | May struggle with structured output for automations |
| Models < 7B | <7B | ❌ Not recommended | JSON output too unreliable for automations/dashboards |

**On response times**: Expect 5–30 seconds for typical HA queries on consumer hardware (e.g. an RTX 3080 or Apple M2). Automation and dashboard suggestions require structured JSON output; the integration includes fallback parsing, but larger models produce cleaner results.

---

## Ask Sage Setup

1. Get a token at [asksage.ai](https://asksage.ai). Community credits are available — check the site for the current program.
2. In the HA integration config, select **Ask Sage** as the provider.
3. Enter your **token**.
4. **Model**: The integration fetches available models live from Ask Sage. Government-specific models are excluded from the list automatically.
5. **Live search**: `off` / `news` / `all` — controls whether Ask Sage includes web search results in its context. `off` recommended for pure home automation queries.
6. **Deep agent**: Enables Ask Sage's multi-step reasoning mode. Slower but better for complex queries.

> The integration always sends `dataset: "none"` and `limit_references: 0` to Ask Sage. You cannot accidentally enable RAG ingestion through this integration.

---

## All Supported Providers

| Provider | Auth Method | Model Selection |
|---|---|---|
| **Local Model** (LM Studio / Ollama) | API key (any string) | Manual entry — match your loaded model name |
| **Ask Sage** | Token | Live-fetched list (gov models excluded) |
| **OpenAI** | API key | Manual entry (e.g. `gpt-4o`, `gpt-4-turbo`) |
| **Anthropic** | API key | Manual entry (e.g. `claude-opus-4-6`) |
| **Google Gemini** | API key | Manual entry (e.g. `gemini-1.5-pro`) |
| **OpenRouter** | API key | Manual entry (any OpenRouter model slug) |
| **Alter** | API key | Manual entry |
| **z.ai** | API key | Manual entry |
| **Llama** | API key | Manual entry |

---

## Security

### Data Sovereignty

| Tier | Provider | Data Leaves Network | Notes |
|---|---|---|---|
| Local | LM Studio / Ollama | **Never** | All processing on-premise |
| Secure Cloud | Ask Sage | Query text only | Zero RAG ingestion enforced, community credits |
| Standard Cloud | OpenAI / Anthropic / Gemini / etc. | Query text only | Standard provider terms apply |

### What the Integration Enforces

- **Ask Sage**: `dataset: "none"` and `limit_references: 0` sent on every query — enforced in code, not just recommended.
- **Conversation history cap**: Hard limit of 50 entries.
- **Chat history storage**: Browser-local, scoped to authenticated HA user (`hass.user.id`). Never transmitted.
- **Chat history is opt-in**: Disabled by default.
- **No credential logging**: API keys and tokens are never written to logs or stored in chat history.

---

## Upstream Attribution

This integration is forked from [sbenodiz/ai_agent_ha](https://github.com/sbenodiz/ai_agent_ha), which remains the original author and foundation. The upstream project established the core architecture, HA integration pattern, and initial provider support. This fork adds features, fixes regressions, and maintains active releases — it does not replace or supersede upstream, and remains open to merging changes in either direction.

---

## Contributing

Pull requests are welcome. For significant changes — new providers, architectural changes, new UI components — please open an issue first to discuss the approach. Bug fixes and test additions can go straight to a PR.

The test suite has 114+ tests covering the core conversation loop, provider integrations, and JSON parsing. New features should include tests.

---

## License

See [LICENSE](https://github.com/JLay2026/ai_agent_ha/blob/main/LICENSE) in the repository.
