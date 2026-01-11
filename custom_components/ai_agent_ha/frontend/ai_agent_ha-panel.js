import {
  LitElement,
  html,
  css,
} from "https://unpkg.com/lit-element@2.4.0/lit-element.js?module";

console.log("AI Agent HA Panel loading..."); // Debug log

const PROVIDERS = {
  openai: "OpenAI",
  llama: "Llama",
  gemini: "Google Gemini",
  openrouter: "OpenRouter",
  anthropic: "Anthropic",
  alter: "Alter",
  zai: "z.ai",
  local: "Local Model",
};

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
      _debugInfo: { type: Object, reflect: false, attribute: false }
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
      .provider-button {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        background: var(--secondary-background-color);
        border: 1px solid var(--divider-color);
        border-radius: 8px;
        cursor: pointer;
        font-size: 14px;
        font-weight: 500;
        color: var(--primary-text-color);
        transition: all 0.2s ease;
        min-width: 150px;
        -webkit-appearance: none;
        -moz-appearance: none;
        appearance: none;
        background-image: url('data:image/svg+xml;charset=US-ASCII,<svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M7 10l5 5 5-5H7z" fill="currentColor"/></svg>');
        background-repeat: no-repeat;
        background-position: right 8px center;
        padding-right: 30px;
      }
      .provider-button:hover {
        background-color: var(--primary-background-color);
        border-color: var(--primary-color);
      }
      .provider-button:focus {
        outline: none;
        border-color: var(--primary-color);
        box-shadow: 0 0 0 2px rgba(var(--primary-color-rgb), 0.2);
      }
      .provider-label {
        font-size: 12px;
        color: var(--secondary-text-color);
        margin-right: 8px;
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
      .dashboard-suggestion {
        background: var(--secondary-background-color);
        border: 1px solid var(--info-color, #2196f3);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        position: relative;
        z-index: 10;
      }
      .dashboard-title {
        font-weight: 500;
        margin-bottom: 8px;
        color: var(--info-color, #2196f3);
        font-size: 16px;
      }
      .dashboard-description {
        margin-bottom: 16px;
        color: var(--secondary-text-color);
        line-height: 1.4;
      }
      .dashboard-actions {
        display: flex;
        gap: 8px;
        margin-top: 16px;
        justify-content: flex-end;
      }
      .dashboard-actions ha-button {
        --mdc-button-height: 40px;
        --mdc-button-padding: 0 20px;
        --mdc-typography-button-font-size: 14px;
        --mdc-typography-button-font-weight: 600;
        border-radius: 20px;
      }
      .dashboard-actions ha-button:first-child {
        --mdc-theme-primary: var(--info-color, #2196f3);
        --mdc-theme-on-primary: #fff;
      }
      .dashboard-actions ha-button:last-child {
        --mdc-theme-primary: var(--error-color);
        --mdc-theme-on-primary: #fff;
      }
      .dashboard-details {
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
      .no-providers {
        color: var(--error-color);
        font-size: 14px;
        padding: 8px;
      }
    `;
  }

  constructor() {
    super();
    this._messages = [];
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
    console.debug("AI Agent HA Panel constructor called");
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
      // Load prompt history from Home Assistant storage
      await this._loadPromptHistory();
    }

    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
      if (!this.shadowRoot.querySelector('.provider-selector')?.contains(e.target)) {
        this._showProviderDropdown = false;
      }
    });
  }

  async updated(changedProps) {
    console.debug("Updated called with:", changedProps);

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
        // Uses the WebSocket API to get all entries with their complete data
        const allEntries = await this.hass.callWS({ type: 'config_entries/get' });

        const aiAgentEntries = allEntries.filter(
          entry => entry.domain === 'ai_agent_ha'
        );

        if (aiAgentEntries.length > 0) {
          const providers = aiAgentEntries
            .map(entry => {
              const provider = this._resolveProviderFromEntry(entry);
              if (!provider) return null;

              return {
                value: provider,
                label: PROVIDERS[provider] || provider
              };
            })
            .filter(Boolean);

          this._availableProviders = providers;

          console.debug("Available AI providers (mapped from data/title):", this._availableProviders);

          if (
            (!this._selectedProvider || !providers.find(p => p.value === this._selectedProvider)) &&
            providers.length > 0
          ) {
            this._selectedProvider = providers[0].value;
          }
        } else {
          console.debug("No 'ai_agent_ha' config entries found via WebSocket.");
          this._availableProviders = [];
        }
      } catch (error) {
        console.error("Error fetching config entries via WebSocket:", error);
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
                ${msg.text}
                ${msg.automation ? html`
                  <div class="automation-suggestion">
                    <div class="automation-title">${msg.automation.alias}</div>
                    <div class="automation-description">${msg.automation.description}</div>
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
                  <div class="dashboard-suggestion">
                    <div class="dashboard-title">${msg.dashboard.title}</div>
                    <div class="dashboard-description">Dashboard with ${msg.dashboard.views ? msg.dashboard.views.length : 0} view(s)</div>
                    <div class="dashboard-details">
                      ${JSON.stringify(msg.dashboard, null, 2)}
                    </div>
                    <div class="dashboard-actions">
                      <ha-button
                        @click=${() => this._approveDashboard(msg.dashboard)}
                        .disabled=${this._isLoading}
                      >Create Dashboard</ha-button>
                      <ha-button
                        @click=${() => this._rejectDashboard()}
                        .disabled=${this._isLoading}
                      >Cancel</ha-button>
                    </div>
                  </div>
                ` : ''}
              </div>
            `)}
            ${this._isLoading ? html`
              <div class="loading">
                <span>AI Agent is thinking</span>
                <div class="loading-dots">
                  <div class="dot"></div>
                  <div class="dot"></div>
                  <div class="dot"></div>
                </div>
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
                <span class="provider-label">Model:</span>
                <select
                  class="provider-button"
                  @change=${(e) => this._selectProvider(e.target.value)}
                  .value=${this._selectedProvider || ''}
                >
                  ${this._availableProviders.map(provider => html`
                    <option
                      value=${provider.value}
                      ?selected=${provider.value === this._selectedProvider}
                    >
                      ${provider.label}
                    </option>
                  `)}
                </select>
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

    // Set timeout to clear loading state after 60 seconds
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
    }, 60000); // 60 second timeout

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

  _handleLlamaResponse(event) {
    console.debug("Received llama response:", event);
    
    try {
      this._clearLoadingState();
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

      let message = { type: 'assistant', text: event.data.answer };

      // Check if the response contains an automation or dashboard suggestion
      try {
        console.debug("Attempting to parse response as JSON:", event.data.answer);
        let jsonText = event.data.answer;
        
        // Try to extract JSON from mixed text+JSON responses
        const jsonMatch = jsonText.match(/\{[\s\S]*\}/);
        if (jsonMatch && jsonMatch[0] !== jsonText.trim()) {
          console.debug("Found JSON within mixed response, extracting:", jsonMatch[0]);
          jsonText = jsonMatch[0];
        }
        
        const response = JSON.parse(jsonText);
        console.debug("Parsed JSON response:", response);
        
        if (response.request_type === 'automation_suggestion') {
          console.debug("Found automation suggestion");
          message.automation = response.automation;
          message.text = response.message || 'I found an automation that might help you. Would you like me to create it?';
        } else if (response.request_type === 'dashboard_suggestion') {
          console.debug("Found dashboard suggestion:", response.dashboard);
          message.dashboard = response.dashboard;
          message.text = response.message || 'I created a dashboard configuration for you. Would you like me to create it?';
        } else if (response.request_type === 'final_response') {
          // If it's a final response, use the response field
          message.text = response.response || response.message || event.data.answer;
        } else if (response.message) {
          // If there's a message field, use it
          message.text = response.message;
        } else if (response.response) {
          // If there's a response field, use it
          message.text = response.response;
        }
        // If none of the above, keep the original event.data.answer as message.text
      } catch (e) {
        // Not a JSON response, treat as normal message
        console.debug("Response is not JSON, using as-is:", event.data.answer);
        console.debug("JSON parse error:", e);
        // message.text is already set to event.data.answer
      }

      console.debug("Adding message to UI:", message);
      this._messages = [...this._messages, message];
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

  async _approveDashboard(dashboard) {
    if (this._isLoading) return;
    this._isLoading = true;
    try {
      const result = await this.hass.callService('ai_agent_ha', 'create_dashboard', {
        dashboard_config: dashboard
      });

      console.debug("Dashboard creation result:", result);

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
          text: `Dashboard "${dashboard.title}" has been created successfully!`
        }];
      }
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
           changedProps.has('_showProviderDropdown');
  }

  _clearChat() {
    this._messages = [];
    this._clearLoadingState();
    this._error = null;
    this._pendingAutomation = null;
    this._debugInfo = null;
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
