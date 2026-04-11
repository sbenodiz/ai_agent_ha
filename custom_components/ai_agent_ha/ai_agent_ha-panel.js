import {
  LitElement,
  html,
  svg,
  css,
} from "https://unpkg.com/lit-element@2.4.0/lit-element.js?module";

console.log("AI Agent HA Panel loading..."); // Debug log

// Try to find the actionable JSON object in a potentially multi-JSON response.
// The model often emits several data objects followed by the final response object.
// We scan all top-level JSON objects and return both the actionable one and the full list.
function extractAllJson(text) {
  const objects = [];
  let depth = 0;
  let start = -1;
  for (let i = 0; i < text.length; i++) {
    if (text[i] === '{') {
      if (depth === 0) start = i;
      depth++;
    } else if (text[i] === '}') {
      depth--;
      if (depth === 0 && start !== -1) {
        try {
          const obj = JSON.parse(text.slice(start, i + 1));
          objects.push(obj);
        } catch (e) { /* skip malformed */ }
        start = -1;
      }
    }
  }
  const actionable = [...objects].reverse().find(o => o.request_type);
  return { actionable: actionable || objects[objects.length - 1] || null, all: objects };
}

function extractTemperatureChart(allObjects) {
  for (const obj of allObjects) {
    if (!obj.data || !Array.isArray(obj.data)) continue;
    const tempEntities = obj.data.filter(e =>
      e.attributes?.device_class === 'temperature' ||
      e.entity_id?.includes('temperature') ||
      e.friendly_name?.toLowerCase().includes('temperature')
    );
    if (tempEntities.length >= 2) {
      return tempEntities
        .filter(e => e.state && !isNaN(parseFloat(e.state)))
        .map(e => ({
          label: (e.friendly_name || e.entity_id)
            .replace(/ temperature$/i, '')
            .replace(/sensor\./i, ''),
          value: parseFloat(e.state),
          unit: e.attributes?.unit_of_measurement || '°F',
          isOutdoor: !e.area_id || e.friendly_name?.toLowerCase().includes('outdoor') || e.entity_id?.includes('outdoor')
        }))
        .sort((a, b) => b.isOutdoor - a.isOutdoor);
    }
  }
  return null;
}

// Extract a JSON object from text that may contain markdown fences, prose, or YAML.
// Tries: strip code fences → direct parse → find first { → nested { scan.
function extractJSONFromText(text) {
  if (!text || typeof text !== 'string') return null;
  // Strip markdown code fences
  const fenceMatch = text.match(/```(?:json|yaml)?\s*([\s\S]*?)```/);
  if (fenceMatch) text = fenceMatch[1].trim();
  // Try direct parse
  try { return JSON.parse(text); } catch(e) { /* continue */ }
  // Find first { and try parsing from each one
  for (let i = text.indexOf('{'); i !== -1 && i < text.length; i = text.indexOf('{', i + 1)) {
    try { return JSON.parse(text.slice(i)); } catch(e) { /* continue */ }
  }
  return null;
}

// Legacy key (pre-v1.1.0) — cleared on first opt-in load
const CHAT_STORAGE_KEY = 'ai_agent_ha_chat_history';

// v1.1.0 secure persistence — opt-in, per-user, per-provider
const CHAT_STORAGE_VERSION = 1;
const CHAT_STORAGE_MAX_MESSAGES = 50;

function chatStorageKey(userId, provider) {
  return `ai_agent_ha_v1_${userId || 'anon'}_${provider || 'default'}`;
}

function scrubForStorage(text) {
  if (!text || typeof text !== 'string') return text;
  return text.replace(/\b(sk-[A-Za-z0-9]{20,}|[A-Za-z0-9_\-]{32,})\b/g, '[REDACTED]');
}

const PROVIDERS = {
  openai: "OpenAI",
  llama: "Llama",
  gemini: "Google Gemini",
  openrouter: "OpenRouter",
  anthropic: "Anthropic",
  alter: "Alter",
  zai: "z.ai",
  asksage: "Ask Sage",
  local: "Local Model",
};

const CARD_TYPE_META = {
  'weather-forecast':  { icon: 'mdi:weather-partly-cloudy', color: '#4fc3f7', label: 'Weather' },
  'thermostat':        { icon: 'mdi:thermostat',            color: '#ef5350', label: 'Thermostat' },
  'gauge':             { icon: 'mdi:gauge',                 color: '#66bb6a', label: 'Gauge' },
  'sensor':            { icon: 'mdi:eye',                   color: '#ab47bc', label: 'Sensor' },
  'entity':            { icon: 'mdi:power-plug',            color: '#7e57c2', label: 'Entity' },
  'button':            { icon: 'mdi:gesture-tap-button',    color: '#26a69a', label: 'Button' },
  'light':             { icon: 'mdi:lightbulb',             color: '#ffca28', label: 'Light' },
  'switch':            { icon: 'mdi:toggle-switch',         color: '#42a5f5', label: 'Switch' },
  'history-graph':     { icon: 'mdi:chart-line',            color: '#26c6da', label: 'History' },
  'statistics-graph':  { icon: 'mdi:chart-bar',             color: '#ff7043', label: 'Stats' },
  'media-control':     { icon: 'mdi:remote',                color: '#ec407a', label: 'Media' },
  'map':               { icon: 'mdi:map',                   color: '#9ccc65', label: 'Map' },
  'picture':           { icon: 'mdi:image',                 color: '#8d6e63', label: 'Picture' },
  'markdown':          { icon: 'mdi:text',                  color: '#78909c', label: 'Text' },
  'glance':            { icon: 'mdi:view-grid',             color: '#5c6bc0', label: 'Glance' },
  'entities':          { icon: 'mdi:format-list-bulleted',  color: '#5c6bc0', label: 'Entities' },
  'alarm-panel':       { icon: 'mdi:shield-home',           color: '#ef5350', label: 'Alarm' },
  'logbook':           { icon: 'mdi:clipboard-list',        color: '#78909c', label: 'Log' },
  'calendar':          { icon: 'mdi:calendar',              color: '#42a5f5', label: 'Calendar' },
  'energy-distribution': { icon: 'mdi:lightning-bolt',      color: '#ffca28', label: 'Energy' },
  'plant-status':      { icon: 'mdi:flower',                color: '#66bb6a', label: 'Plant' },
  'humidifier':        { icon: 'mdi:air-humidifier',        color: '#4fc3f7', label: 'Humidity' },
  'irrigation':        { icon: 'mdi:sprinkler',             color: '#66bb6a', label: 'Irrigation'},
  'timer':             { icon: 'mdi:timer',                 color: '#ffca28', label: 'Timer' },
  'input-boolean':     { icon: 'mdi:toggle-switch',         color: '#42a5f5', label: 'Toggle' },
};
const DEFAULT_CARD_META = { icon: 'mdi:card', color: '#90a4ae', label: 'Card' };

class AiAgentHaPanel extends LitElement {
  static get properties() {
    return {
      hass: { type: Object, reflect: false, attribute: false },
      narrow: { type: Boolean, reflect: false, attribute: false },
      panel: { type: Object, reflect: false, attribute: false },
      _messages: { type: Array, reflect: false, attribute: false },
      _isLoading: { type: Boolean, reflect: false, attribute: false },
      _error: { type: String, reflect: false, attribute: false },
      _pendingAutomation: { type: Object, reflect: false, attribute: false },
      _promptHistory: { type: Array, reflect: false, attribute: false },
      _showPredefinedPrompts: { type: Boolean, reflect: false, attribute: false },
      _showPromptHistory: { type: Boolean, reflect: false, attribute: false },
      _selectedPrompts: { type: Array, reflect: false, attribute: false },
      _selectedProvider: { type: String, reflect: false, attribute: false },
      _availableProviders: { type: Array, reflect: false, attribute: false },
      _showProviderDropdown: { type: Boolean, reflect: false, attribute: false },
      _showThinking: { type: Boolean, reflect: false, attribute: false },
      _thinkingExpanded: { type: Boolean, reflect: false, attribute: false },
      _debugInfo: { type: Object, reflect: false, attribute: false },
      _dashboardPickerActive: { type: Boolean, reflect: false, attribute: false },
      _existingDashboards: { type: Array, reflect: false, attribute: false },
      _dashboardPickerLoading: { type: Boolean, reflect: false, attribute: false },
      _activeSuggestionDashboard: { type: Object, reflect: false, attribute: false },
      _persistenceEnabled: { type: Boolean, reflect: false, attribute: false },
      _isStreaming: { type: Boolean, reflect: false, attribute: false },
      _streamingText: { type: String, reflect: false, attribute: false },
      _dashboardChangeActive: { type: Object, reflect: false, attribute: false },
      _dashboardChangeText: { type: String, reflect: false, attribute: false }
    };
  }

  static get styles() {
    return css`
      :host {
        background: var(--primary-background-color);
        -webkit-font-smoothing: antialiased;
        display: flex;
        flex-direction: column;
        height: 100vh;
      }
      .header {
        background: var(--app-header-background-color);
        color: var(--app-header-text-color);
        padding: 16px 24px;
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 20px;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        position: relative;
        z-index: 100;
      }
      .clear-button {
        margin-left: auto;
        border: none;
        border-radius: 16px;
        background: var(--error-color);
        color: #fff;
        cursor: pointer;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 8px 16px;
        font-weight: 500;
        font-size: 13px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.08);
        min-width: unset;
        width: auto;
        height: 36px;
        flex-shrink: 0;
        position: relative;
        z-index: 101;
        font-family: inherit;
      }
      .clear-button:hover {
        background: var(--error-color);
        opacity: 0.92;
        transform: translateY(-1px);
        box-shadow: 0 2px 6px rgba(0,0,0,0.13);
      }
      .clear-button:active {
        transform: translateY(0);
        box-shadow: 0 1px 2px rgba(0,0,0,0.08);
      }
      .clear-button ha-icon {
        --mdc-icon-size: 16px;
        margin-right: 2px;
        color: #fff;
      }
      .clear-button span {
        color: #fff;
        font-weight: 500;
      }
      .content {
        flex-grow: 1;
        padding: 24px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
      }
      .chat-container {
        width: 100%;
        padding: 0;
        display: flex;
        flex-direction: column;
        flex-grow: 1;
        height: 100%;
      }
      .messages {
        overflow-y: auto;
        border: 1px solid var(--divider-color);
        border-radius: 12px;
        margin-bottom: 24px;
        padding: 0;
        background: var(--primary-background-color);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        flex-grow: 1;
        width: 100%;
      }
      .prompts-section {
        margin-bottom: 12px;
        padding: 12px 16px;
        background: var(--secondary-background-color);
        border-radius: 16px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid var(--divider-color);
      }
      .prompts-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
        font-size: 14px;
        font-weight: 500;
        color: var(--secondary-text-color);
      }
      .prompts-toggle {
        display: flex;
        align-items: center;
        gap: 4px;
        cursor: pointer;
        color: var(--primary-color);
        font-size: 12px;
        font-weight: 500;
        padding: 2px 6px;
        border-radius: 4px;
        transition: background-color 0.2s ease;
      }
      .prompts-toggle:hover {
        background: var(--primary-color);
        color: var(--text-primary-color);
      }
      .prompts-toggle ha-icon {
        --mdc-icon-size: 14px;
      }
      .prompt-bubbles {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-bottom: 8px;
      }
      .prompt-bubble {
        background: var(--primary-background-color);
        border: 1px solid var(--divider-color);
        border-radius: 20px;
        padding: 6px 12px;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 12px;
        line-height: 1.3;
        color: var(--primary-text-color);
        white-space: nowrap;
        max-width: 200px;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      .prompt-bubble:hover {
        border-color: var(--primary-color);
        background: var(--primary-color);
        color: var(--text-primary-color);
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      }
      .prompt-bubble:active {
        transform: translateY(0);
      }
      .history-bubble {
        background: var(--primary-background-color);
        border: 1px solid var(--accent-color);
        border-radius: 20px;
        padding: 6px 12px;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 12px;
        line-height: 1.3;
        color: var(--accent-color);
        white-space: nowrap;
        max-width: 180px;
        overflow: hidden;
        text-overflow: ellipsis;
        display: flex;
        align-items: center;
        gap: 6px;
      }
      .history-bubble:hover {
        background: var(--accent-color);
        color: var(--text-primary-color);
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      }
      .history-delete {
        opacity: 0;
        transition: opacity 0.2s ease;
        color: var(--error-color);
        cursor: pointer;
        --mdc-icon-size: 14px;
      }
      .history-bubble:hover .history-delete {
        opacity: 1;
        color: var(--text-primary-color);
      }
      .message {
        margin-bottom: 16px;
        padding: 12px 16px;
        border-radius: 12px;
        max-width: 80%;
        line-height: 1.5;
        animation: fadeIn 0.3s ease-out;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        word-wrap: break-word;
      }
      .user-message {
        background: var(--primary-color);
        color: var(--text-primary-color);
        margin-left: auto;
        border-bottom-right-radius: 4px;
      }
      .assistant-message {
        background: var(--secondary-background-color);
        margin-right: auto;
        border-bottom-left-radius: 4px;
      }
      .chart-container {
        margin-top: 8px;
        overflow-x: auto;
      }
      .input-container {
        position: relative;
        width: 100%;
        background: var(--card-background-color);
        border: 1px solid var(--divider-color);
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        margin-bottom: 24px;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
      }
      .input-container:focus-within {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 2px rgba(var(--primary-color-rgb), 0.1);
      }
      .input-main {
        display: flex;
        align-items: flex-end;
        padding: 12px;
        gap: 12px;
      }
      .input-wrapper {
        flex-grow: 1;
        position: relative;
        border: 1px solid var(--divider-color);
      }
      textarea {
        width: 100%;
        min-height: 24px;
        max-height: 200px;
        padding: 12px 16px 12px 16px;
        border: none;
        outline: none;
        resize: none;
        font-size: 16px;
        line-height: 1.5;
        background: transparent;
        color: var(--primary-text-color);
        font-family: inherit;
      }
      textarea::placeholder {
        color: var(--secondary-text-color);
      }
      .input-footer {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 16px 12px 16px;
        border-top: 1px solid var(--divider-color);
        background: var(--card-background-color);
        border-radius: 0 0 12px 12px;
      }
      .provider-selector {
        position: relative;
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .provider-label {
        font-size: 0.75rem;
        color: var(--secondary-text-color, #888);
        white-space: nowrap;
        padding: 4px 8px;
        border-radius: 12px;
        background: var(--secondary-background-color, rgba(0,0,0,0.05));
      }
      .persistence-indicator {
        font-size: 0.75rem;
        opacity: 0.6;
        cursor: default;
      }
      .thinking-toggle {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        color: var(--secondary-text-color);
        cursor: pointer;
        user-select: none;
      }
      .thinking-toggle input {
        margin: 0;
      }
      .thinking-panel {
        border: 1px dashed var(--divider-color);
        border-radius: 10px;
        padding: 10px 12px;
        margin: 12px 0;
        background: var(--secondary-background-color);
      }
      .thinking-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        cursor: pointer;
        gap: 10px;
      }
      .thinking-title {
        font-weight: 600;
        color: var(--primary-text-color);
        font-size: 14px;
      }
      .thinking-subtitle {
        display: block;
        font-size: 12px;
        color: var(--secondary-text-color);
        margin-top: 2px;
      }
      .thinking-body {
        margin-top: 10px;
        display: flex;
        flex-direction: column;
        gap: 10px;
        max-height: 240px;
        overflow-y: auto;
      }
      .thinking-entry {
        border: 1px solid var(--divider-color);
        border-radius: 8px;
        padding: 8px;
        background: var(--primary-background-color);
      }
      .thinking-entry .badge {
        display: inline-block;
        background: var(--secondary-background-color);
        color: var(--secondary-text-color);
        font-size: 11px;
        padding: 2px 6px;
        border-radius: 6px;
        margin-bottom: 6px;
      }
      .thinking-entry pre {
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        font-size: 12px;
      }
      .thinking-empty {
        color: var(--secondary-text-color);
        font-size: 12px;
      }
      .thinking-block {
        margin-bottom: 8px;
        border: 1px solid var(--divider-color, #e0e0e0);
        border-radius: 8px;
        overflow: hidden;
      }
      .thinking-summary {
        padding: 8px 12px;
        cursor: pointer;
        font-size: 13px;
        color: var(--secondary-text-color, #666);
        background: var(--card-background-color, #fafafa);
        list-style: none;
        display: flex;
        align-items: center;
        gap: 6px;
      }
      .thinking-summary::-webkit-details-marker {
        display: none;
      }
      .thinking-summary:hover {
        background: var(--primary-background-color, #f0f0f0);
      }
      .thinking-content-inner {
        padding: 8px 12px;
        font-size: 13px;
        color: var(--secondary-text-color, #666);
        white-space: pre-wrap;
        max-height: 300px;
        overflow-y: auto;
      }
      .thinking-pulse-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--primary-color, #03a9f4);
        animation: thinkingPulse 1.5s ease-in-out infinite;
        margin-right: 6px;
      }
      @keyframes thinkingPulse {
        0%, 100% { opacity: 0.3; transform: scale(0.8); }
        50% { opacity: 1; transform: scale(1.2); }
      }
      .thinking-active {
        padding: 8px 12px;
        font-size: 13px;
        color: var(--secondary-text-color, #666);
        display: flex;
        align-items: center;
      }
      .streaming-text {
        white-space: pre-wrap;
        word-wrap: break-word;
      }
      .send-button {
        --mdc-theme-primary: var(--primary-color);
        --mdc-theme-on-primary: var(--text-primary-color);
        --mdc-typography-button-font-size: 14px;
        --mdc-typography-button-text-transform: none;
        --mdc-typography-button-letter-spacing: 0;
        --mdc-typography-button-font-weight: 500;
        --mdc-button-height: 36px;
        --mdc-button-padding: 0 16px;
        border-radius: 8px;
        transition: all 0.2s ease;
        min-width: 80px;
      }
      .send-button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
      }
      .send-button:active {
        transform: translateY(0);
      }
      .send-button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
      }
      .loading {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 16px;
        padding: 12px 16px;
        border-radius: 12px;
        background: var(--secondary-background-color);
        margin-right: auto;
        max-width: 80%;
        animation: fadeIn 0.3s ease-out;
      }
      .loading-dots {
        display: flex;
        gap: 4px;
      }
      .dot {
        width: 8px;
        height: 8px;
        background: var(--primary-color);
        border-radius: 50%;
        animation: bounce 1.4s infinite ease-in-out;
      }
      .dot:nth-child(1) { animation-delay: -0.32s; }
      .dot:nth-child(2) { animation-delay: -0.16s; }
      @keyframes bounce {
        0%, 80%, 100% {
          transform: scale(0);
        }
        40% {
          transform: scale(1.0);
        }
      }
      @keyframes fadeIn {
        from {
          opacity: 0;
          transform: translateY(10px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }
      .error {
        color: var(--error-color);
        padding: 16px;
        margin: 8px 0;
        border-radius: 12px;
        background: var(--error-background-color);
        border: 1px solid var(--error-color);
        animation: fadeIn 0.3s ease-out;
      }
      .automation-suggestion {
        background: var(--secondary-background-color);
        border: 1px solid var(--primary-color);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        position: relative;
        z-index: 10;
      }
      .automation-title {
        font-weight: 500;
        margin-bottom: 8px;
        color: var(--primary-color);
        font-size: 16px;
      }
      .automation-description {
        margin-bottom: 16px;
        color: var(--secondary-text-color);
        line-height: 1.4;
      }
      .automation-actions {
        display: flex;
        gap: 8px;
        margin-top: 16px;
        justify-content: flex-end;
      }
      .automation-actions ha-button {
        --mdc-button-height: 40px;
        --mdc-button-padding: 0 20px;
        --mdc-typography-button-font-size: 14px;
        --mdc-typography-button-font-weight: 600;
        border-radius: 20px;
      }
      .automation-actions ha-button:first-child {
        --mdc-theme-primary: var(--success-color, #4caf50);
        --mdc-theme-on-primary: #fff;
      }
      .automation-actions ha-button:last-child {
        --mdc-theme-primary: var(--error-color);
        --mdc-theme-on-primary: #fff;
      }
      .automation-details {
        margin-top: 8px;
        padding: 8px;
        background: var(--primary-background-color);
        border-radius: 8px;
        font-family: monospace;
        font-size: 12px;
        white-space: pre-wrap;
        overflow-x: auto;
        max-height: 200px;
        overflow-y: auto;
        border: 1px solid var(--divider-color);
      }
      .dashboard-suggestion-card {
        margin-top: 12px;
        width: 100%;
        box-sizing: border-box;
      }
      .dashboard-suggestion-card .card-header {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 16px 16px 0;
        font-size: 1rem;
        font-weight: 600;
        color: var(--primary-text-color);
      }
      .dashboard-suggestion-card .card-header-icon {
        color: var(--primary-color);
        --mdc-icon-size: 22px;
      }
      .dashboard-suggestion-card .card-content {
        padding: 12px 16px;
        display: flex;
        flex-direction: column;
        gap: 8px;
      }
      .dashboard-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }
      .dashboard-meta {
        font-size: 0.78rem;
        color: var(--secondary-text-color);
      }
      .dashboard-suggestion-card .card-actions {
        display: flex;
        gap: 8px;
        padding: 8px 16px 16px;
      }
      .no-providers {
        color: var(--error-color);
        font-size: 14px;
        padding: 8px;
      }
      .dashboard-picker {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 4px;
      }
      .dashboard-picker-select {
        flex: 1;
        min-width: 140px;
        padding: 6px 10px;
        border-radius: 8px;
        border: 1px solid var(--divider-color, rgba(255,255,255,0.15));
        background: var(--card-background-color, #1e1e2e);
        color: var(--primary-text-color);
        font-size: 0.85rem;
        cursor: pointer;
      }
      .dashboard-preview {
        max-height: 320px;
        overflow-y: auto;
        border: 1px solid var(--divider-color, rgba(255,255,255,0.12));
        border-radius: 10px;
        padding: 10px;
        background: var(--secondary-background-color, rgba(0,0,0,0.15));
        margin-bottom: 10px;
      }
      .preview-view-label {
        font-size: 11px;
        font-weight: 600;
        color: var(--secondary-text-color);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 6px 0 4px;
      }
      .preview-group-label {
        font-size: 10px;
        font-weight: 600;
        color: var(--primary-color);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 4px 0 2px;
        padding-left: 2px;
      }
      .preview-card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
        gap: 6px;
        margin-bottom: 8px;
      }
      .preview-card-tile {
        height: 70px;
        border-radius: 8px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 4px;
        cursor: default;
        transition: transform 0.15s;
      }
      .preview-card-tile:hover {
        transform: scale(1.04);
      }
      .preview-card-title {
        font-size: 10px;
        color: var(--secondary-text-color);
        text-align: center;
        max-width: 76px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        padding: 0 2px;
      }
      .preview-more-views {
        font-size: 11px;
        color: var(--secondary-text-color);
        text-align: center;
        padding: 4px 0;
      }
      .dashboard-change-row {
        display: flex;
        gap: 6px;
        align-items: center;
        margin-top: 6px;
      }
      .dashboard-change-input {
        flex: 1;
        padding: 7px 10px;
        border-radius: 8px;
        border: 1px solid var(--divider-color, rgba(255,255,255,0.15));
        background: var(--card-background-color, #1e1e2e);
        color: var(--primary-text-color);
        font-size: 13px;
      }
      .dashboard-change-input:focus {
        outline: none;
        border-color: var(--primary-color);
      }
    `;
  }

  constructor() {
    super();
    this._messages = this._loadMessages();
    this._isLoading = false;
    this._error = null;
    this._pendingAutomation = null;
    this._promptHistory = [];
    this._promptHistoryLoaded = false;
    this._showPredefinedPrompts = true;
    this._showPromptHistory = true;
    this._predefinedPrompts = [
      "Build a new automation to turn off all lights at 10:00 PM every day",
      "What's the current temperature inside and outside?",
      "Turn on all the lights in the living room",
      "Show me today's weather forecast",
      "What devices are currently on?",
      "Show me the energy usage for today",
      "Are all the doors and windows locked?",
      "Turn on movie mode in the living room",
      "What's the status of my security system?",
      "Show me who's currently home",
      "Turn off all devices when I leave home"
    ];
    this._selectedPrompts = this._getRandomPrompts();
    this._selectedProvider = null;
    this._availableProviders = [];
    this._showProviderDropdown = false;
    this.providersLoaded = false;
    this._eventSubscriptionSetup = false;
    this._serviceCallTimeout = null;
    this._showThinking = false;
    this._thinkingExpanded = false;
    this._debugInfo = null;
    this._dashboardPickerActive = false;
    this._existingDashboards = [];
    this._dashboardPickerLoading = false;
    this._activeSuggestionDashboard = null;
    this._persistenceEnabled = false;
    this._isStreaming = false;
    this._streamingText = '';
    this._streamChunkUnsub = null;
    this._streamEndUnsub = null;
    this._dashboardChangeActive = null;
    this._dashboardChangeText = '';
    console.debug("AI Agent HA Panel constructor called");
  }

  _loadMessages() {
    // Legacy loader for backward compat (pre-opt-in). Returns any old messages.
    try {
      const raw = localStorage.getItem(CHAT_STORAGE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      console.debug('AI Agent HA: restored', parsed.length, 'messages from legacy localStorage');
      return parsed;
    } catch (e) {
      console.warn('AI Agent HA: failed to load chat history from localStorage:', e);
      return [];
    }
  }

  _saveMessages() {
    // Legacy save (pre-opt-in). Only saves if persistence is NOT enabled (fallback).
    if (this._persistenceEnabled) return;
    try {
      const toSave = this._messages.length > CHAT_STORAGE_MAX_MESSAGES
        ? this._messages.slice(-CHAT_STORAGE_MAX_MESSAGES)
        : this._messages;
      localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(toSave));
    } catch (e) {
      console.warn('AI Agent HA: failed to save chat history to localStorage:', e);
    }
  }

  _saveHistoryToStorage() {
    if (!this._persistenceEnabled) return;
    if (!this.hass?.user?.id) return;
    try {
      const key = chatStorageKey(this.hass.user.id, this._selectedProvider);
      const toStore = this._messages
        .filter(m => m.type && m.text)
        .slice(-CHAT_STORAGE_MAX_MESSAGES)
        .map(m => ({ type: m.type, text: scrubForStorage(m.text) }));
      localStorage.setItem(key, JSON.stringify({
        version: CHAT_STORAGE_VERSION,
        provider: this._selectedProvider,
        timestamp: Date.now(),
        messages: toStore
      }));
    } catch (e) {
      console.warn('AI Agent HA: failed to save chat history:', e);
    }
  }

  _loadHistoryFromStorage() {
    if (!this._persistenceEnabled) return;
    if (!this.hass?.user?.id) return;
    try {
      const key = chatStorageKey(this.hass.user.id, this._selectedProvider);
      const raw = localStorage.getItem(key);
      if (!raw) return;
      const stored = JSON.parse(raw);
      if (stored.version !== CHAT_STORAGE_VERSION) {
        localStorage.removeItem(key);
        return;
      }
      const sevenDays = 7 * 24 * 60 * 60 * 1000;
      if (stored.provider !== this._selectedProvider) return;
      if (Date.now() - stored.timestamp > sevenDays) {
        localStorage.removeItem(key);
        return;
      }
      if (stored.messages && stored.messages.length > 0) {
        this._messages = stored.messages;
        // Clear legacy storage now that we have opt-in persistence
        try { localStorage.removeItem(CHAT_STORAGE_KEY); } catch (_) {}
        this.requestUpdate();
      }
    } catch (e) {
      console.warn('AI Agent HA: failed to load chat history:', e);
      try {
        const key = chatStorageKey(this.hass.user.id, this._selectedProvider);
        localStorage.removeItem(key);
      } catch (_) {}
    }
  }

  _getRandomPrompts() {
    // Shuffle array and take first 3 items
    const shuffled = [...this._predefinedPrompts].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, 3);
  }

  async connectedCallback() {
    super.connectedCallback();
    console.debug("AI Agent HA Panel connected");
    if (this.hass && !this._eventSubscriptionSetup) {
      this._eventSubscriptionSetup = true;
      this.hass.connection.subscribeEvents(
        (event) => this._handleLlamaResponse(event),
        'ai_agent_ha_response'
      );
      console.debug("Event subscription set up in connectedCallback()");

      // Subscribe to streaming events
      this._streamChunkUnsub = this.hass.connection.subscribeEvents(
        (event) => this._handleStreamChunk(event),
        'ai_agent_ha/stream_chunk'
      );
      this._streamEndUnsub = this.hass.connection.subscribeEvents(
        (event) => this._handleStreamEnd(event),
        'ai_agent_ha/stream_end'
      );
      console.debug("Streaming event subscriptions set up");

      // Load prompt history from Home Assistant storage
      await this._loadPromptHistory();
    }

    // Clear all ai_agent_ha storage on HA logout
    this._logoutHandler = () => {
      try {
        Object.keys(localStorage)
          .filter(k => k.startsWith('ai_agent_ha_v1_'))
          .forEach(k => localStorage.removeItem(k));
      } catch (e) {}
    };
    window.addEventListener('hass-logout', this._logoutHandler);

    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
      if (!this.shadowRoot.querySelector('.provider-selector')?.contains(e.target)) {
        this._showProviderDropdown = false;
      }
    });
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    if (this._logoutHandler) {
      window.removeEventListener('hass-logout', this._logoutHandler);
    }
    // Clean up streaming subscriptions
    if (this._streamChunkUnsub) {
      try { this._streamChunkUnsub.then(unsub => unsub()); } catch (_) {}
    }
    if (this._streamEndUnsub) {
      try { this._streamEndUnsub.then(unsub => unsub()); } catch (_) {}
    }
  }

  async updated(changedProps) {
    console.debug("Updated called with:", changedProps);

    // Persist chat history whenever messages change
    if (changedProps.has('_messages')) {
      this._saveMessages();
    }

    // Set up event subscription when hass becomes available
    if (changedProps.has('hass') && this.hass && !this._eventSubscriptionSetup) {
      this._eventSubscriptionSetup = true;
      this.hass.connection.subscribeEvents(
        (event) => this._handleLlamaResponse(event),
        'ai_agent_ha_response'
      );
      console.debug("Event subscription set up in updated()");
    }

    // Load providers when hass becomes available
    if (changedProps.has('hass') && this.hass && !this.providersLoaded) {
      this.providersLoaded = true;

      try {
        // Primary: use ai_agent_ha/get_providers — authoritative, no credential exposure
        let providers = [];
        let persistenceEnabled = false;
        let streamingEnabled = false;

        try {
          const providerData = await this.hass.callWS({ type: 'ai_agent_ha/get_providers' });
          if (providerData && providerData.length > 0) {
            providers = providerData.map(p => ({
              value: p.value,
              label: p.label,
              model: p.model || ''
            }));
            persistenceEnabled = providerData.some(p => p.persist_chat_history);
            streamingEnabled = providerData.some(p => p.enable_streaming);
            console.debug("AI Agent HA: providers loaded via get_providers WS:", providers);
          }
        } catch (wsErr) {
          console.warn("AI Agent HA: get_providers WS failed, falling back to config_entries/get:", wsErr);
          // Fallback: parse config_entries/get (title + unique_id only — entry.data not exposed by HA)
          const allEntries = await this.hass.callWS({ type: 'config_entries/get' });
          const aiAgentEntries = allEntries.filter(entry => entry.domain === 'ai_agent_ha');
          providers = aiAgentEntries
            .map(entry => {
              const provider = this._resolveProviderFromEntry(entry);
              if (!provider) return null;
              return { value: provider, label: PROVIDERS[provider] || provider, model: '' };
            })
            .filter(Boolean);
          console.debug("AI Agent HA: providers loaded via config_entries fallback:", providers);
        }

        if (providers.length > 0) {
          this._availableProviders = providers;
          this._persistenceEnabled = persistenceEnabled;

          console.debug("Available AI providers:", this._availableProviders);
          console.debug("Chat persistence enabled:", this._persistenceEnabled);

          if (
            (!this._selectedProvider || !providers.find(p => p.value === this._selectedProvider)) &&
            providers.length > 0
          ) {
            this._selectedProvider = providers[0].value;
          }

          // Load persisted history if enabled
          if (this._persistenceEnabled) {
            this._loadHistoryFromStorage();
          }
        } else {
          console.debug("No 'ai_agent_ha' providers found.");
          this._availableProviders = [];
        }
      } catch (error) {
        console.error("Error fetching AI providers:", error);
        this._error = error.message || 'Failed to load AI provider configurations.';
        this._availableProviders = [];
      }
      this.requestUpdate();
    }

    // Load prompt history when hass becomes available and we haven't loaded it yet
    if (changedProps.has('hass') && this.hass && !this._promptHistoryLoaded) {
      this._promptHistoryLoaded = true;
      await this._loadPromptHistory();
    }

    // Load prompt history when provider changes
    if (changedProps.has('_selectedProvider') && this._selectedProvider && this.hass) {
      await this._loadPromptHistory();
    }

    if (changedProps.has('_messages') || changedProps.has('_isLoading')) {
      this._scrollToBottom();
    }
  }

  _renderPromptsSection() {
    return html`
      <div class="prompts-section">
        <div class="prompts-header">
          <span>Quick Actions</span>
          <div style="display: flex; gap: 12px;">
            <div class="prompts-toggle" @click=${() => this._togglePredefinedPrompts()}>
              <ha-icon icon="${this._showPredefinedPrompts ? 'mdi:chevron-up' : 'mdi:chevron-down'}"></ha-icon>
              <span>Suggestions</span>
            </div>
            ${this._promptHistory.length > 0 ? html`
              <div class="prompts-toggle" @click=${() => this._togglePromptHistory()}>
                <ha-icon icon="${this._showPromptHistory ? 'mdi:chevron-up' : 'mdi:chevron-down'}"></ha-icon>
                <span>Recent</span>
              </div>
            ` : ''}
          </div>
        </div>

        ${this._showPredefinedPrompts ? html`
          <div class="prompt-bubbles">
            ${this._selectedPrompts.map(prompt => html`
              <div class="prompt-bubble" @click=${() => this._usePrompt(prompt)}>
                ${prompt}
              </div>
            `)}
          </div>
        ` : ''}

        ${this._showPromptHistory && this._promptHistory.length > 0 ? html`
          <div class="prompt-bubbles">
            ${this._promptHistory.slice(-3).reverse().map((prompt, index) => html`
              <div class="history-bubble" @click=${(e) => this._useHistoryPrompt(e, prompt)}>
                <span style="flex-grow: 1; overflow: hidden; text-overflow: ellipsis;">${prompt}</span>
                <ha-icon
                  class="history-delete"
                  icon="mdi:close"
                  @click=${(e) => this._deleteHistoryItem(e, prompt)}
                ></ha-icon>
              </div>
            `)}
          </div>
        ` : ''}
      </div>
    `;
  }

  _togglePredefinedPrompts() {
    this._showPredefinedPrompts = !this._showPredefinedPrompts;
    // Refresh random selection when toggling on
    if (this._showPredefinedPrompts) {
      this._selectedPrompts = this._getRandomPrompts();
    }
  }

  _togglePromptHistory() {
    this._showPromptHistory = !this._showPromptHistory;
  }

  _usePrompt(prompt) {
    if (this._isLoading) return;
    const promptEl = this.shadowRoot.querySelector('#prompt');
    if (promptEl) {
      promptEl.value = prompt;
      promptEl.focus();
    }
  }

  _useHistoryPrompt(event, prompt) {
    event.stopPropagation();
    if (this._isLoading) return;
    const promptEl = this.shadowRoot.querySelector('#prompt');
    if (promptEl) {
      promptEl.value = prompt;
      promptEl.focus();
    }
  }

  async _deleteHistoryItem(event, prompt) {
    event.stopPropagation();
    this._promptHistory = this._promptHistory.filter(p => p !== prompt);
    await this._savePromptHistory();
    this.requestUpdate();
  }

  async _addToHistory(prompt) {
    if (!prompt || prompt.trim().length === 0) return;

    // Remove duplicates and add to front
    this._promptHistory = this._promptHistory.filter(p => p !== prompt);
    this._promptHistory.push(prompt);

    // Keep only last 20 prompts
    if (this._promptHistory.length > 20) {
      this._promptHistory = this._promptHistory.slice(-20);
    }

    await this._savePromptHistory();
    this.requestUpdate();
  }

  async _loadPromptHistory() {
    if (!this.hass) {
      console.debug('Hass not available, skipping prompt history load');
      return;
    }

    console.debug('Loading prompt history...');
    try {
      const result = await this.hass.callService('ai_agent_ha', 'load_prompt_history', {
        provider: this._selectedProvider
      });
      console.debug('Prompt history service result:', result);

      if (result && result.response && result.response.history) {
        this._promptHistory = result.response.history;
        console.debug('Loaded prompt history from service:', this._promptHistory);
        this.requestUpdate();
      } else if (result && result.history) {
        this._promptHistory = result.history;
        console.debug('Loaded prompt history from service (direct):', this._promptHistory);
        this.requestUpdate();
      } else {
        console.debug('No prompt history returned from service, checking localStorage');
        // Fallback to localStorage if service returns no data
        this._loadFromLocalStorage();
      }
    } catch (error) {
      console.error('Error loading prompt history from service:', error);
      // Fallback to localStorage if service fails
      this._loadFromLocalStorage();
    }
  }

  _loadFromLocalStorage() {
    try {
      const savedList = localStorage.getItem('ai_agent_ha_prompt_history');
      if (savedList) {
        const parsedList = JSON.parse(savedList);
        const saved = parsedList.history && parsedList.provider === this._selectedProvider ? parsedList.history : null;
        if (saved) {
          this._promptHistory = JSON.parse(saved);
          console.debug('Loaded prompt history from localStorage:', this._promptHistory);
          this.requestUpdate();
        } else {
          console.debug('No prompt history in localStorage');
          this._promptHistory = [];
        }
      }
    } catch (e) {
      console.error('Error loading from localStorage:', e);
      this._promptHistory = [];
    }
  }

  async _savePromptHistory() {
    if (!this.hass) {
      console.debug('Hass not available, saving to localStorage only');
      this._saveToLocalStorage();
      return;
    }

    console.debug('Saving prompt history:', this._promptHistory);
    try {
      const result = await this.hass.callService('ai_agent_ha', 'save_prompt_history', {
        history: this._promptHistory,
        provider: this._selectedProvider
      });
      console.debug('Save prompt history result:', result);

      // Also save to localStorage as backup
      this._saveToLocalStorage();
    } catch (error) {
      console.error('Error saving prompt history to service:', error);
      // Fallback to localStorage if service fails
      this._saveToLocalStorage();
    }
  }

  _saveToLocalStorage() {
    try {
      const data = {
        provider: this._selectedProvider,
        history: JSON.stringify(this._promptHistory)
      }
      localStorage.setItem('ai_agent_ha_prompt_history', JSON.stringify(data));
      console.debug('Saved prompt history to localStorage');
    } catch (e) {
      console.error('Error saving to localStorage:', e);
    }
  }

  render() {
    console.debug("Rendering with state:", {
      messages: this._messages,
      isLoading: this._isLoading,
      error: this._error
    });
    console.debug("Messages array:", this._messages);

    return html`
      <div class="header">
        <ha-icon icon="mdi:robot"></ha-icon>
        AI Agent HA
        <button
          class="clear-button"
          @click=${this._clearChat}
          ?disabled=${this._isLoading}
        >
          <ha-icon icon="mdi:delete-sweep"></ha-icon>
          <span>Clear Chat</span>
        </button>
      </div>
      <div class="content">
        <div class="chat-container">
          <div class="messages" id="messages">
            ${this._messages.map(msg => html`
              <div class="message ${msg.type}-message">
                ${msg.thinking ? html`
                  <details class="thinking-block">
                    <summary class="thinking-summary">
                      <span>💭</span>
                      Thought for ${msg.thinking_duration || '?'}s
                    </summary>
                    <div class="thinking-content-inner">${msg.thinking}</div>
                  </details>
                ` : ''}
                ${msg.text && !(msg.dashboard && msg.text.trim().startsWith('{')) ? msg.text : ''}
                ${this._renderTempChart(msg.chartData)}
                ${msg.automation ? html`
                  <div class="automation-suggestion">
                    <div class="automation-title">${msg.automation.alias || msg.automation.id || 'New Automation'}</div>
                    <div class="automation-description">${msg.automation.description || (msg.automation.trigger?.[0] ? `Trigger: ${JSON.stringify(msg.automation.trigger[0])}` : 'Review the automation below')}</div>
                    <div class="automation-details">
                      ${JSON.stringify(msg.automation, null, 2)}
                    </div>
                    <div class="automation-actions">
                      <ha-button
                        @click=${() => this._approveAutomation(msg.automation)}
                        .disabled=${this._isLoading}
                      >Approve</ha-button>
                      <ha-button
                        @click=${() => this._rejectAutomation()}
                        .disabled=${this._isLoading}
                      >Reject</ha-button>
                    </div>
                  </div>
                ` : ''}
                ${msg.dashboard ? html`
                  <ha-card class="dashboard-suggestion-card">
                    <div class="card-header">
                      <ha-icon icon="${msg.dashboard.icon || 'mdi:view-dashboard'}" class="card-header-icon"></ha-icon>
                      <span>${msg.dashboard.title || 'New Dashboard'}</span>
                    </div>
                    <div class="card-content">

                      ${this._renderDashboardPreview(msg.dashboard)}

                      <div class="dashboard-meta">
                        ${(() => {
                          const views = msg.dashboard.views || [];
                          const totalCards = views.reduce((sum, v) => sum + (v.cards ? v.cards.length : 0), 0);
                          return `${views.length} view${views.length !== 1 ? 's' : ''} \u00b7 ${totalCards} card${totalCards !== 1 ? 's' : ''}`;
                        })()}
                      </div>

                      ${this._dashboardChangeActive === msg.dashboard ? html`
                        <div class="dashboard-change-row">
                          <input
                            class="dashboard-change-input"
                            placeholder="Describe changes (e.g. add a weather card, remove the timer)..."
                            .value=${this._dashboardChangeText || ''}
                            @input=${e => { this._dashboardChangeText = e.target.value; }}
                            @keydown=${e => { if (e.key === 'Enter' && this._dashboardChangeText?.trim()) this._requestDashboardChange(msg.dashboard); }}
                          />
                          <ha-button
                            @click=${() => this._requestDashboardChange(msg.dashboard)}
                            .disabled=${this._isLoading || !this._dashboardChangeText?.trim()}
                          >Apply</ha-button>
                          <ha-button
                            @click=${() => { this._dashboardChangeActive = null; this._dashboardChangeText = ''; this.requestUpdate(); }}
                          >Cancel</ha-button>
                        </div>
                      ` : ''}

                    </div>
                    <div class="card-actions">
                      ${!this._dashboardPickerActive || this._activeSuggestionDashboard !== msg.dashboard ? html`
                        <ha-button
                          @click=${() => this._approveDashboard(msg.dashboard)}
                          .disabled=${this._isLoading}
                        >New Dashboard</ha-button>
                        <ha-button
                          @click=${() => this._showDashboardPicker(msg.dashboard)}
                          .disabled=${this._isLoading || this._dashboardPickerLoading}
                        >${this._dashboardPickerLoading && this._activeSuggestionDashboard === msg.dashboard ? 'Loading...' : 'Add to Existing'}</ha-button>
                        <ha-button
                          @click=${() => { this._dashboardChangeActive = msg.dashboard; this._dashboardChangeText = ''; this.requestUpdate(); }}
                          .disabled=${this._isLoading}
                        >Request Changes</ha-button>
                      ` : html`
                        <div class="dashboard-picker">
                          <select class="dashboard-picker-select" id="dashboard-picker-select-${msg.dashboard.title?.replace(/\s/g,'_')}">
                            ${this._existingDashboards.map(d => html`
                              <option value="${d.url_path}">${d.title || d.url_path}</option>
                            `)}
                          </select>
                          <ha-button
                            @click=${() => {
                              const sel = this.shadowRoot.querySelector(`#dashboard-picker-select-${msg.dashboard.title?.replace(/\s/g,'_')}`);
                              if (sel) this._addViewToExisting(msg.dashboard, sel.value);
                            }}
                            .disabled=${this._isLoading}
                          >Add View</ha-button>
                          <ha-button
                            @click=${() => { this._dashboardPickerActive = false; this._activeSuggestionDashboard = null; this.requestUpdate(); }}
                          >Cancel</ha-button>
                        </div>
                      `}
                    </div>
                  </ha-card>
                ` : ''}
              </div>
            `)}
            ${this._isStreaming && this._streamingText ? html`
              <div class="message assistant-message">
                <div class="streaming-text">${this._streamingText}</div>
              </div>
            ` : ''}
            ${this._isLoading ? html`
              <div class="loading">
                ${this._isStreaming ? html`
                  <div class="thinking-active">
                    <span class="thinking-pulse-dot"></span>
                    Receiving...
                  </div>
                ` : html`
                  <div class="thinking-active">
                    <span class="thinking-pulse-dot"></span>
                    Thinking...
                  </div>
                  <div class="loading-dots">
                    <div class="dot"></div>
                    <div class="dot"></div>
                    <div class="dot"></div>
                  </div>
                `}
              </div>
            ` : ''}
            ${this._error ? html`
              <div class="error">${this._error}</div>
            ` : ''}
            ${this._showThinking ? this._renderThinkingPanel() : ''}
          </div>
          ${this._renderPromptsSection()}
          <div class="input-container">
            <div class="input-main">
              <div class="input-wrapper">
                <textarea
                  id="prompt"
                  placeholder="Ask me anything about your Home Assistant..."
                  ?disabled=${this._isLoading}
                  @keydown=${this._handleKeyDown}
                  @input=${this._autoResize}
                ></textarea>
              </div>
            </div>

            <div class="input-footer">
              <div class="provider-selector">
                ${(() => {
                  const p = this._availableProviders.find(p => p.value === this._selectedProvider);
                  const providerLabel = p ? p.label : (this._selectedProvider || 'No provider configured');
                  const modelLabel = p?.model ? ` \u00b7 ${p.model}` : '';
                  return html`<span class="provider-label">${providerLabel}${modelLabel}</span>
                    ${this._persistenceEnabled ? html`<span class="persistence-indicator" title="Chat history is saved locally">\u{1F4BE}</span>` : ''}`;
                })()}
              </div>
              <label class="thinking-toggle">
                <input
                  type="checkbox"
                  .checked=${this._showThinking}
                  @change=${(e) => this._toggleShowThinking(e)}
                />
                Show thinking
              </label>

              <ha-button
                class="send-button"
                @click=${this._sendMessage}
                .disabled=${this._isLoading || !this._hasProviders()}
              >
                <ha-icon icon="mdi:send"></ha-icon>
              </ha-button>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  _scrollToBottom() {
    const messages = this.shadowRoot.querySelector('#messages');
    if (messages) {
      messages.scrollTop = messages.scrollHeight;
    }
  }

  _autoResize(e) {
    const textarea = e.target;
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
  }

  _handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey && !this._isLoading) {
      e.preventDefault();
      this._sendMessage();
    }
  }

  _toggleProviderDropdown() {
    this._showProviderDropdown = !this._showProviderDropdown;
    console.log("Toggling provider dropdown:", this._showProviderDropdown);
    this.requestUpdate(); // Añade esta línea para forzar la actualización
  }

  async _selectProvider(provider) {
    this._selectedProvider = provider;
    console.debug("Provider changed to:", provider);
    await this._loadPromptHistory();
    this.requestUpdate();
  }

  _getSelectedProviderLabel() {
    const provider = this._availableProviders.find(p => p.value === this._selectedProvider);
    return provider ? provider.label : 'Select Model';
  }

  async _sendMessage() {
    const promptEl = this.shadowRoot.querySelector('#prompt');
    const prompt = promptEl.value.trim();
    if (!prompt || this._isLoading) return;

    console.debug("Sending message:", prompt);
    console.debug("Sending message with provider:", this._selectedProvider);

    // Add to history
    await this._addToHistory(prompt);

    // Add user message
    this._messages = [...this._messages, { type: 'user', text: prompt }];
    promptEl.value = '';
    promptEl.style.height = 'auto';
    this._isLoading = true;
    this._error = null;
    this._debugInfo = null;
    this._thinkingExpanded = false; // keep collapsed until a trace arrives

    // Clear any existing timeout
    if (this._serviceCallTimeout) {
      clearTimeout(this._serviceCallTimeout);
    }

    // Set timeout to clear loading state after 300 seconds
    // Increased from 60s to support local models (LM Studio, Ollama) which
    // may take longer to generate responses on local hardware.
    this._serviceCallTimeout = setTimeout(() => {
      if (this._isLoading) {
        console.warn("Service call timeout - clearing loading state");
        this._isLoading = false;
        this._error = 'Request timed out. Please try again.';
        this._messages = [...this._messages, {
          type: 'assistant',
          text: 'Sorry, the request timed out. Please try again.'
        }];
        this.requestUpdate();
      }
    }, 300000); // 300 second timeout (matches backend aiohttp timeout)

    try {
      console.debug("Calling ai_agent_ha service");
      await this.hass.callService('ai_agent_ha', 'query', {
        prompt: prompt,
        provider: this._selectedProvider,
        debug: this._showThinking
      });
    } catch (error) {
      console.error("Error calling service:", error);
      this._clearLoadingState();
      this._error = error.message || 'An error occurred while processing your request';
      this._messages = [...this._messages, {
        type: 'assistant',
        text: `Error: ${this._error}`
      }];
    }
  }

  _clearLoadingState() {
    this._isLoading = false;
    if (this._serviceCallTimeout) {
      clearTimeout(this._serviceCallTimeout);
      this._serviceCallTimeout = null;
    }
  }

  _handleStreamChunk(event) {
    const text = event.data?.text || '';
    if (!text) return;
    this._isStreaming = true;
    this._streamingText = text;
    this.requestUpdate();
    this._scrollToBottom();
  }

  _handleStreamEnd(event) {
    // Stream ended — the final response will arrive via ai_agent_ha_response
    this._isStreaming = false;
    console.debug("Stream ended");
    this.requestUpdate();
  }

  _handleLlamaResponse(event) {
    console.debug("Received llama response:", event);
    
    try {
      this._clearLoadingState();
      this._isStreaming = false;
      this._streamingText = '';
      this._debugInfo = this._showThinking ? (event.data.debug || null) : null;
      if (this._showThinking && this._debugInfo) {
        this._thinkingExpanded = true;
      }
    if (event.data.success) {
      // Check if the answer is empty
      if (!event.data.answer || event.data.answer.trim() === '') {
        console.warn("AI agent returned empty response");
        this._messages = [
          ...this._messages,
          { type: 'assistant', text: 'I received your message but I\'m not sure how to respond. Could you please try rephrasing your question?' }
        ];
        return;
      }

      let message = { type: 'assistant', text: event.data.answer, _rawAnswer: event.data.answer };

      // Capture thinking content from response
      if (event.data.thinking) {
        message.thinking = event.data.thinking;
        message.thinking_duration = event.data.thinking_duration || null;
      }

      // Check if the response contains an automation or dashboard suggestion
      try {
        console.debug("Attempting to parse response:", event.data.answer?.substring(0, 200));
        const { actionable: response, all: allObjects } = extractAllJson(event.data.answer || '');
        console.debug("Extracted actionable JSON:", response);

        // Detect temperature chart data from intermediate data objects
        const tempReadings = extractTemperatureChart(allObjects);
        if (tempReadings) {
          message.chartData = { type: 'temperature', readings: tempReadings };
        }

        if (response) {
          if (response.request_type === 'automation_suggestion') {
            console.debug("Found automation suggestion");
            message.automation = response.automation;
            message.text = response.message || 'I found an automation that might help you. Would you like me to create it?';
          } else if (response.request_type === 'dashboard_suggestion') {
            console.debug("Found dashboard suggestion:", response.dashboard);
            message.dashboard = response.dashboard;
            message.text = response.message || 'I created a dashboard configuration for you. Would you like me to create it?';
          } else if (response.request_type === 'final_response') {
            message.text = response.response || response.message || event.data.answer;
          } else if (response.message) {
            message.text = response.message;
          } else if (response.response) {
            message.text = response.response;
          }
        }
      } catch (e) {
        console.debug("Response parsing error:", e);
      }

      // Safety: if dashboard was found but text still looks like raw JSON, replace it
      if (message.dashboard && message.text && message.text.trim().startsWith('{')) {
        message.text = 'I created a dashboard for you. Review it below.';
      }

      // YAML / markdown bleed guard: if text looks like it contains a dashboard
      // response wrapped in prose, code fences, or YAML, try to extract it.
      if (!message.dashboard && message.text) {
        const trimmed = message.text.trim();
        const looksLikeDashboard = /^(dashboard:|title:|views:)/m.test(trimmed)
          || /```(?:json|yaml)?\s*[\s\S]*?(dashboard|views|cards)/i.test(trimmed);
        if (looksLikeDashboard) {
          const recovered = extractJSONFromText(trimmed);
          if (recovered && typeof recovered === 'object') {
            // Unwrap top-level "dashboard" key if present
            const dash = recovered.dashboard || recovered;
            if (dash.views || dash.cards || dash.title) {
              console.warn('Recovered dashboard from YAML/markdown bleed in message text');
              message.dashboard = dash;
              message.text = recovered.message || 'I created a dashboard for you. Review it below.';
            }
          }
          // If extraction failed, still suppress the raw YAML/markdown
          if (!message.dashboard) {
            console.warn('Detected YAML dashboard bleed in message text — suppressing raw display');
            message.text = 'The AI returned a dashboard in an unexpected format. Please try your request again.';
          }
        }
      }

      console.debug("Adding message to UI:", message);
      this._messages = [...this._messages, message];
      this._saveHistoryToStorage();
    } else {
      this._error = event.data.error || 'An error occurred';
      this._messages = [
        ...this._messages,
        { type: 'assistant', text: `Error: ${this._error}` }
      ];
    }
    } catch (error) {
      console.error("Error in _handleLlamaResponse:", error);
      this._clearLoadingState();
      this._error = 'An error occurred while processing the response';
      this._messages = [...this._messages, {
        type: 'assistant',
        text: 'Sorry, an error occurred while processing the response. Please try again.'
      }];
      this.requestUpdate();
    }
  }

  async _approveAutomation(automation) {
    if (this._isLoading) return;
    this._isLoading = true;
    try {
      const result = await this.hass.callService('ai_agent_ha', 'create_automation', {
        automation: automation
      });

      console.debug("Automation creation result:", result);

      // The result should be an object with a message property
      if (result && result.message) {
        this._messages = [...this._messages, {
          type: 'assistant',
          text: result.message
        }];
      } else {
        // Fallback success message if no message is provided
        this._messages = [...this._messages, {
          type: 'assistant',
          text: `Automation "${automation.alias}" has been created successfully!`
        }];
      }
    } catch (error) {
      console.error("Error creating automation:", error);
      this._error = error.message || 'An error occurred while creating the automation';
      this._messages = [...this._messages, {
        type: 'assistant',
        text: `Error: ${this._error}`
      }];
    } finally {
      this._clearLoadingState();
    }
  }

  _rejectAutomation() {
    this._messages = [...this._messages, {
      type: 'assistant',
      text: 'Automation creation cancelled. Would you like to try something else?'
    }];
  }

  _getCardGroup(card) {
    const cardType = (card?.type || '').toLowerCase();
    const entity = (card?.entity || card?.entities?.[0] || '').toLowerCase();
    if (cardType.startsWith('light') || entity.startsWith('light.'))
      return 'Lights';
    if (cardType === 'media-control' || entity.startsWith('media_player.'))
      return 'Media';
    if (cardType === 'irrigation' || cardType.includes('valve') ||
        entity.startsWith('valve.') ||
        (entity.startsWith('input_boolean.') && (entity.includes('irrigat') || entity.includes('zone') || entity.includes('sprinkler'))))
      return 'Irrigation';
    if (entity.startsWith('switch.'))
      return 'Switches';
    if (entity.startsWith('sensor.') || entity.startsWith('binary_sensor.'))
      return 'Sensors';
    if (cardType === 'weather-forecast' || entity.startsWith('weather.'))
      return 'Weather';
    if (cardType === 'thermostat' || entity.startsWith('climate.'))
      return 'Climate';
    return 'Other';
  }

  _renderDashboardPreview(dashboard) {
    const views = dashboard?.views;
    if (!views || !Array.isArray(views) || views.length === 0) return html``;
    const displayViews = views.slice(0, 3);
    const remaining = views.length - 3;
    return html`
      <div class="dashboard-preview">
        ${displayViews.map(view => html`
          <div class="preview-view-label">${view.title || 'View'}</div>
          ${(view.cards && view.cards.length > 0) ? (() => {
            // Group cards by category
            const groups = {};
            view.cards.forEach(card => {
              const group = this._getCardGroup(card);
              if (!groups[group]) groups[group] = [];
              groups[group].push(card);
            });
            const groupOrder = ['Lights', 'Climate', 'Weather', 'Switches', 'Sensors', 'Irrigation', 'Media', 'Other'];
            const sortedGroups = Object.keys(groups).sort((a, b) => {
              const ai = groupOrder.indexOf(a);
              const bi = groupOrder.indexOf(b);
              return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
            });
            return html`${sortedGroups.map(groupName => html`
              ${sortedGroups.length > 1 ? html`<div class="preview-group-label">${groupName}</div>` : ''}
              <div class="preview-card-grid">
                ${groups[groupName].map(card => {
                  const cardType = card?.type || '';
                  const meta = CARD_TYPE_META[cardType] || DEFAULT_CARD_META;
                  const title = card?.title || meta.label;
                  const truncTitle = title.length > 12 ? title.slice(0, 12) + '\u2026' : title;
                  return html`
                    <div class="preview-card-tile"
                         style="background:${meta.color}1a;border:1px solid ${meta.color}66">
                      <ha-icon icon="${meta.icon}" style="color:${meta.color};--mdc-icon-size:22px"></ha-icon>
                      <span class="preview-card-title">${truncTitle}</span>
                    </div>
                  `;
                })}
              </div>
            `)}`;
          })() : ''}
        `)}
        ${remaining > 0 ? html`<div class="preview-more-views">+${remaining} more view${remaining !== 1 ? 's' : ''}</div>` : ''}
      </div>
    `;
  }

  async _requestDashboardChange(dashboard) {
    if (this._isLoading) return;
    const changeText = this._dashboardChangeText?.trim();
    if (!changeText) return;

    // Sanitise title to prevent prompt injection
    const safeTitle = (dashboard.title || 'Untitled').slice(0, 100).replace(/"/g, '\\"');

    this._dashboardChangeActive = null;
    this._dashboardChangeText = '';

    this._messages = [...this._messages, { type: 'user', text: `Please update the dashboard: ${changeText}` }];
    this._isLoading = true;
    this._error = null;
    this._debugInfo = null;
    this._thinkingExpanded = false;
    this.requestUpdate();

    // Timeout fallback — consistent with _approveDashboard pattern
    if (this._serviceCallTimeout) clearTimeout(this._serviceCallTimeout);
    this._serviceCallTimeout = setTimeout(() => {
      if (this._isLoading) {
        this._clearLoadingState();
        this._error = 'Request timed out. Please try again.';
        this.requestUpdate();
      }
    }, 300000);

    try {
      await this.hass.callService('ai_agent_ha', 'query', {
        prompt: `Please update the "${safeTitle}" dashboard with the following changes: ${changeText}. Return a complete updated dashboard_suggestion JSON.`,
        provider: this._selectedProvider,
        debug: this._showThinking
      });
    } catch (e) {
      console.error('Error requesting dashboard change:', e);
      this._clearLoadingState();
      this._error = e.message || 'Failed to request dashboard changes';
      this.requestUpdate();
    }
  }

  async _approveDashboard(dashboard) {
    if (this._isLoading) return;
    this._isLoading = true;
    try {
      const title = dashboard.title || 'My Dashboard';
      // Generate url_path from title — must contain a hyphen for HA validation
      let urlPath = title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
      if (!urlPath.includes('-')) {
        urlPath = urlPath + '-dashboard';
      }

      // Step 1: Create the dashboard entry via WebSocket
      console.debug("Creating dashboard entry via WS:", urlPath);
      await this.hass.callWS({
        type: 'lovelace/dashboards/create',
        url_path: urlPath,
        title: title,
        icon: dashboard.icon || 'mdi:view-dashboard',
        show_in_sidebar: true,
        require_admin: false,
        mode: 'storage',
      });

      // Step 2: Save the views/cards config to the new dashboard
      console.debug("Saving dashboard config via WS:", urlPath);
      await this.hass.callWS({
        type: 'lovelace/config/save',
        url_path: urlPath,
        config: { views: dashboard.views || [] },
      });

      this._messages = [...this._messages, {
        type: 'assistant',
        text: `Dashboard "${title}" created successfully!\n\nThe dashboard is now available in your sidebar — no restart required.\nURL: /lovelace-${urlPath}`
      }];
    } catch (error) {
      console.error("Error creating dashboard:", error);
      this._error = error.message || 'An error occurred while creating the dashboard';
      this._messages = [...this._messages, {
        type: 'assistant',
        text: `Error: ${this._error}`
      }];
    } finally {
      this._clearLoadingState();
    }
  }

  _rejectDashboard() {
    this._messages = [...this._messages, {
      type: 'assistant',
      text: 'Dashboard creation cancelled. Would you like me to create a different dashboard?'
    }];
  }

  async _showDashboardPicker(dashboard) {
    this._dashboardPickerLoading = true;
    this._activeSuggestionDashboard = dashboard;
    this.requestUpdate();
    try {
      // Fetch existing dashboards via Lovelace WebSocket API
      const dashboards = await this.hass.callWS({ type: 'lovelace/dashboards/list' });
      // Filter to user-created dashboards (exclude system ones without url_path)
      this._existingDashboards = (dashboards || []).filter(d => d.url_path);
      if (this._existingDashboards.length === 0) {
        // No existing dashboards — fall back to creating new
        this._messages = [...this._messages, {
          type: 'assistant',
          text: 'No existing dashboards found. Creating a new dashboard instead.'
        }];
        await this._approveDashboard(dashboard);
        return;
      }
      this._dashboardPickerActive = true;
    } catch (error) {
      console.error('Error fetching dashboards:', error);
      const errMsg = (error.message || String(error)).toLowerCase();
      if (errMsg.includes('unknown command') || errMsg.includes('unknown_command')) {
        // HA is likely in YAML mode — fall back to creating a new dashboard
        this._messages = [...this._messages, {
          type: 'assistant',
          text: 'Dashboard listing not available (YAML mode). Creating a new dashboard instead.'
        }];
        await this._approveDashboard(dashboard);
      } else {
        this._messages = [...this._messages, {
          type: 'assistant',
          text: `Could not load existing dashboards: ${error.message || error}. Please try creating a new dashboard instead.`
        }];
      }
    } finally {
      this._dashboardPickerLoading = false;
      this.requestUpdate();
    }
  }

  async _addViewToExisting(dashboard, targetDashboardUrl) {
    if (this._isLoading) return;
    this._isLoading = true;
    this._dashboardPickerActive = false;
    this._activeSuggestionDashboard = null;
    try {
      // Pass just the views array as the config to update_dashboard
      const viewsConfig = { views: dashboard.views || [] };
      const result = await this.hass.callService('ai_agent_ha', 'update_dashboard', {
        dashboard_url: targetDashboardUrl,
        dashboard_config: viewsConfig
      }, {}, true);
      const targetName = this._existingDashboards.find(d => d.url_path === targetDashboardUrl)?.title || targetDashboardUrl;
      this._messages = [...this._messages, {
        type: 'assistant',
        text: result?.message || `"${dashboard.title}" view${dashboard.views?.length !== 1 ? 's' : ''} added to "${targetName}" successfully.`
      }];
    } catch (error) {
      console.error('Error adding view to dashboard:', error);
      this._messages = [...this._messages, {
        type: 'assistant',
        text: `Error adding view: ${error.message || error}`
      }];
    } finally {
      this._clearLoadingState();
      this._existingDashboards = [];
      this.requestUpdate();
    }
  }

  shouldUpdate(changedProps) {
    // Only update if internal state changes, not on every hass update
    return changedProps.has('_messages') ||
           changedProps.has('_isLoading') ||
           changedProps.has('_error') ||
           changedProps.has('_promptHistory') ||
           changedProps.has('_showPredefinedPrompts') ||
           changedProps.has('_showPromptHistory') ||
           changedProps.has('_availableProviders') ||
           changedProps.has('_selectedProvider') ||
           changedProps.has('_showProviderDropdown') ||
           changedProps.has('_dashboardPickerActive') ||
           changedProps.has('_existingDashboards') ||
           changedProps.has('_dashboardPickerLoading') ||
           changedProps.has('_persistenceEnabled') ||
           changedProps.has('_isStreaming') ||
           changedProps.has('_streamingText') ||
           changedProps.has('_dashboardChangeActive') ||
           changedProps.has('_dashboardChangeText');
  }

  _clearChat() {
    this._messages = [];
    this._clearLoadingState();
    this._error = null;
    this._pendingAutomation = null;
    this._debugInfo = null;
    this._isStreaming = false;
    this._streamingText = '';
    // Clear persisted chat history (both legacy and new)
    try {
      localStorage.removeItem(CHAT_STORAGE_KEY);
      if (this._persistenceEnabled) {
        localStorage.removeItem(chatStorageKey(this.hass?.user?.id, this._selectedProvider));
      }
      console.debug('AI Agent HA: cleared chat history from localStorage');
    } catch (e) {
      console.warn('AI Agent HA: failed to clear chat history from localStorage:', e);
    }
    // Clear dashboard suggestion UI state
    this._dashboardChangeActive = null;
    this._dashboardChangeText = '';
    this._dashboardPickerActive = false;
    this._activeSuggestionDashboard = null;
    // Don't clear prompt history - users might want to keep it
  }

  _resolveProviderFromEntry(entry) {
    if (!entry) return null;

    const providerFromData = entry.data?.ai_provider || entry.options?.ai_provider;
    if (providerFromData && PROVIDERS[providerFromData]) {
      return providerFromData;
    }

    const uniqueId = entry.unique_id || entry.uniqueId;
    if (uniqueId && uniqueId.startsWith("ai_agent_ha_")) {
      const fromUniqueId = uniqueId.replace("ai_agent_ha_", "");
      if (PROVIDERS[fromUniqueId]) {
        return fromUniqueId;
      }
    }

    const titleMap = {
      "ai agent ha (openrouter)": "openrouter",
      "ai agent ha (google gemini)": "gemini",
      "ai agent ha (openai)": "openai",
      "ai agent ha (llama)": "llama",
      "ai agent ha (anthropic (claude))": "anthropic",
      "ai agent ha (alter)": "alter",
      "ai agent ha (z.ai)": "zai",
      "ai agent ha (local model)": "local",
      "ai agent ha (ask sage)": "asksage",
    };

    if (entry.title) {
      const lowerTitle = entry.title.toLowerCase();
      if (titleMap[lowerTitle]) {
        return titleMap[lowerTitle];
      }

      const match = entry.title.match(/\(([^)]+)\)/);
      if (match && match[1]) {
        const normalized = match[1].toLowerCase().replace(/[^a-z0-9]/g, "");
        const providerKey = Object.keys(PROVIDERS).find(
          key => key.replace(/[^a-z0-9]/g, "") === normalized
        );
        if (providerKey) {
          return providerKey;
        }
      }
    }

    return null;
  }

  _getProviderInfo(providerId) {
    return this._availableProviders.find(p => p.value === providerId);
  }

  _hasProviders() {
    return this._availableProviders && this._availableProviders.length > 0;
  }

  _toggleThinkingPanel() {
    this._thinkingExpanded = !this._thinkingExpanded;
  }

  _toggleShowThinking(e) {
    this._showThinking = e.target.checked;
    if (!this._showThinking) {
      this._thinkingExpanded = false;
    }
  }

  _renderTempChart(chartData) {
    if (!chartData || chartData.type !== 'temperature') return html``;
    const readings = chartData.readings;
    if (!readings || readings.length === 0) return html``;

    const values = readings.map(r => r.value);
    const min = Math.min(...values) - 3;
    const max = Math.max(...values) + 3;
    const range = max - min || 1;
    const barH = 22;
    const gap = 10;
    const labelW = 140;
    const chartW = 200;
    const svgW = labelW + chartW + 60;
    const svgH = readings.length * (barH + gap) + 20;

    return html`
      <div class="chart-container">
        <svg width="${svgW}" height="${svgH}" style="display:block;margin-top:10px;max-width:100%">
          ${readings.map((r, i) => {
            const barWidth = Math.max(4, Math.round(((r.value - min) / range) * chartW));
            const y = i * (barH + gap) + 10;
            const color = r.isOutdoor ? '#5B9BD5' : '#4CAF82';
            return svg`
              <text x="${labelW - 8}" y="${y + barH / 2 + 5}"
                    text-anchor="end" font-size="12" fill="var(--secondary-text-color, #aaa)"
                    font-family="sans-serif">${r.label}</text>
              <rect x="${labelW}" y="${y}" width="${barWidth}" height="${barH}"
                    rx="4" fill="${color}" opacity="0.85"></rect>
              <text x="${labelW + barWidth + 6}" y="${y + barH / 2 + 5}"
                    font-size="12" fill="var(--primary-text-color, #fff)"
                    font-family="sans-serif" font-weight="600">${r.value}${r.unit}</text>
            `;
          })}
        </svg>
      </div>
    `;
  }

  _renderThinkingPanel() {
    if (!this._debugInfo) {
      return '';
    }

    const subtitleParts = [];
    if (this._debugInfo.provider) subtitleParts.push(this._debugInfo.provider);
    if (this._debugInfo.model) subtitleParts.push(this._debugInfo.model);
    if (this._debugInfo.endpoint_type) subtitleParts.push(this._debugInfo.endpoint_type);
    const subtitle = subtitleParts.join(" · ");
    const conversation = this._debugInfo.conversation || [];

    return html`
      <div class="thinking-panel">
        <div class="thinking-header" @click=${() => this._toggleThinkingPanel()}>
          <div>
            <span class="thinking-title">Thinking trace</span>
            ${subtitle ? html`<span class="thinking-subtitle">${subtitle}</span>` : ''}
          </div>
          <ha-icon icon=${this._thinkingExpanded ? 'mdi:chevron-up' : 'mdi:chevron-down'}></ha-icon>
        </div>
        ${this._thinkingExpanded ? html`
          <div class="thinking-body">
            ${conversation.length === 0 ? html`
              <div class="thinking-empty">No trace captured.</div>
            ` : conversation.map((entry, index) => html`
              <div class="thinking-entry">
                <div class="badge">${entry.role || 'unknown'}</div>
                <pre>${entry.content || ''}</pre>
              </div>
            `)}
          </div>
        ` : ''}
      </div>
    `;
  }
}

customElements.define("ai_agent_ha-panel", AiAgentHaPanel);

console.log("AI Agent HA Panel registered");

