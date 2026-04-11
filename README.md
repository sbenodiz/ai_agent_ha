# AI Agent HA Extended

**Local-first AI for Home Assistant — with secure cloud options when you want them.**

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/custom-components/hacs)
[![Version](https://img.shields.io/badge/version-v1.1.0-blue.svg)](https://github.com/JLay2026/ai_agent_ha/releases/tag/v1.1.0)

AI Agent HA Extended brings fully local AI to Home Assistant — run Qwen 2.5 9B on LM Studio and your home data never leaves your network. For users who prefer cloud AI, Ask Sage provides community-credited access to GPT-4o, Claude, and others through a single subscription with zero RAG ingestion enforced. Both paths are first-class, tested, and secure by design.

---

## Why This Fork

The upstream project ([sbenodiz/ai_agent_ha](https://github.com/sbenodiz/ai_agent_ha)) laid solid groundwork, but local model adoption was left as an open question — and the HA community kept asking. This fork exists to answer that question directly.

Key context:

- **Upstream left local model support incomplete.** Configuration errors, missing OpenAI-compat auto-detection, and no model guidance made local setup unreliable in practice.
- **8 open PRs** from this fork to upstream remain unmerged.
- **19 releases** since forking — actively maintained, not a one-off patch.

This fork is not a competition with upstream. If upstream merges the improvements, great. Until then, this fork is where the fixes live.

---

## Choose Your Path

### Path A — Fully Local (Privacy First)

| | |
|---|---|
| **Backend** | LM Studio or Ollama on your local network |
| **Minimum model** | 9B parameters recommended; Qwen 2.5 9B is the tested baseline |
| **Data sovereignty** | Zero external network calls — all HA entity data stays on-premise |
| **Honest UX** | Responses take 5–30 seconds depending on hardware. Structured JSON output is less reliable than cloud models, but the integration includes robust fallback parsing for automations and dashboard suggestions. |
| **Best for** | Privacy-conscious households, sensitive entity data, air-gapped setups |

### Path B — Secure Cloud (Ask Sage)

| | |
|---|---|
| **Auth** | Single token — no per-provider API keys |
| **Models** | GPT-4o, Claude Opus 4.6, and others through one broker |
| **Access** | Community credits available — lower barrier than direct provider subscriptions |
| **RAG enforcement** | `dataset: none` and `limit_references: 0` on every query — your data is never written to or retrieved from Ask Sage's RAG system |
| **Extra features** | Live web search toggle (off / news / all), deep agent mode |
| **Best for** | Users who want managed cloud access without juggling per-model subscriptions |

### Path C — Direct Cloud Providers

| | |
|---|---|
| **Providers** | OpenAI, Anthropic (claude-opus-4-6), Google Gemini, OpenRouter, Alter, z.ai, Llama |
| **Data handling** | Standard API terms per provider |
| **Best for** | Users with existing provider subscriptions |

---

## Security

Security is a design constraint here, not an afterthought.

### Data Sovereignty

| Tier | Provider | Data Leaves Network | Notes |
|---|---|---|---|
| Local | LM Studio / Ollama | **Never** | All processing on-premise |
| Secure Cloud | Ask Sage | Query text only | Zero RAG ingestion enforced, community credits |
| Standard Cloud | OpenAI / Anthropic / Gemini / etc. | Query text only | Standard provider terms apply |

### What the Integration Enforces (Regardless of Provider)

- **Ask Sage**: `dataset: "none"` and `limit_references: 0` are sent on every query. Your data is never written to or retrieved from Ask Sage's RAG system — this is enforced in code, not just recommended.
- **Conversation history cap**: Hard limit of 50 entries. No unbounded accumulation of your home's entity history.
- **Chat history storage**: Stored locally in the browser, scoped to the authenticated HA user (`hass.user.id`). Never transmitted.
- **Chat history is opt-in**: Disabled by default — important for shared or family HA instances where multiple users share a browser.
- **No credential logging**: API keys and tokens are never written to logs or stored in chat history.

### Local Model Security Note

When using LM Studio or Ollama on your local network, HA entity data, states, and automation history never leave your premises. The model processes everything locally. This is the recommended configuration for households with sensitive automation data.

---

## Quick Install (HACS)

1. Open **HACS** in Home Assistant.
2. Go to **Integrations** → click the three-dot menu → **Custom repositories**.
3. Enter `https://github.com/JLay2026/ai_agent_ha` and select **Integration**.
4. Search for **AI Agent HA Extended** and click **Download**.
5. Restart Home Assistant.
6. Go to **Settings → Devices & Services → Add Integration** and search for **AI Agent**.

---

## What's New in v1.1.0 vs Upstream

| Feature | Upstream | This Fork |
|---|---|---|
| Local model support (LM Studio / Ollama) | Partial, broken | ✅ Full OpenAI-compat auto-detect |
| Minimum model guidance | None | ✅ 9B+ recommended, Qwen 2.5 9B tested |
| Ask Sage provider | ❌ | ✅ Community credits, multi-model broker |
| HA context sent to models | ❌ Broken | ✅ Fixed (system_prompt injection) |
| Dashboard suggestion rendering | ❌ Broken | ✅ Fixed + ha-card UI |
| Dashboard placement choice | ❌ | ✅ New dashboard or add to existing |
| Automation suggestion rendering | ❌ | ✅ Fixed + hardened |
| Inline temperature chart | ❌ | ✅ SVG bar chart in chat bubble |
| Conversation history cap | ❌ | ✅ 50-entry cap |
| History rollback on error | ❌ | ✅ |
| Concurrent query protection | ❌ | ✅ `asyncio.Lock` |
| Ask Sage overload retry | ❌ | ✅ 3× exponential backoff |
| Thinking / reasoning strip | ❌ | ✅ |
| Anthropic Claude models | Outdated | ✅ claude-opus-4-6 |
| Chat history persistence | ❌ | ✅ Opt-in, user-scoped, local only |
| Test coverage | None | ✅ 114+ tests |
| Active releases | Irregular | ✅ 19 releases, actively maintained |

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

## Configuration Notes

- **Local Model**: The base URL must end in `/v1`. The integration sends standard OpenAI-format requests. If LM Studio reports a model name with spaces or slashes, use it exactly.
- **Ask Sage**: Do not set a custom base URL — the integration uses Ask Sage's production endpoint. Token is the only required credential.
- **OpenAI**: Standard API key from platform.openai.com. Organization ID is optional.
- **Anthropic**: Use `claude-opus-4-6` for the most capable model available. Older model names in upstream are outdated and will error.
- **OpenRouter**: Your API key from openrouter.ai. Model slugs follow the `provider/model-name` format (e.g. `anthropic/claude-3-opus`).
- **Chat history persistence**: Disabled by default. Enable in the integration's UI settings. History is stored in `localStorage` under a key scoped to `hass.user.id` — different HA users on the same browser get separate histories.

---

## Changelog

### v1.1.0
First independently versioned release. Introduces opt-in `localStorage` chat history persistence, scoped to the authenticated HA user. Security-first design documented explicitly. Full local model setup guide with model recommendations. All prior v1.08.x changes included.

### v1.08.19
Automation suggestion card hardened against malformed JSON from smaller models.

### v1.08.18
Dashboard placement UX: choose whether to create a new dashboard or add the suggested view to an existing one.

### v1.08.17
Inline SVG temperature chart rendered directly in the chat bubble for temperature history queries.

### v1.08.16
Ask Sage retry messages clarified — user-visible status during 3× backoff cycle.

### v1.08.15
Dashboard suggestion card rebuilt with `ha-card` and `ha-chip` badges for consistent HA UI integration.

### v1.08.14
Clean dashboard suggestion card — removed debug output and layout artifacts.

### v1.08.13
Fixed multi-JSON parsing for dashboard and automation suggestions — handles models that emit multiple JSON blocks in one response.

### v1.08.12
Static provider·model label replaces the dynamic dropdown that was causing config state confusion.

### v1.08.11
Ask Sage retry on overload: 3× exponential backoff with user-visible status.

### v1.08.10
Ask Sage `system_prompt` fix — HA context (entity states, history) now correctly injected into every query.

### v1.08.9
Anthropic model list updated to current names. Ask Sage live search and `deep_agent` toggles added to config flow.

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
