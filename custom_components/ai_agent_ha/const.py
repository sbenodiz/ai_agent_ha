"""Constants for the AI Agent HA integration."""

DOMAIN = "ai_agent_ha"
VERSION = "1.2.15"
CONF_API_KEY = "api_key"
CONF_WEATHER_ENTITY = "weather_entity"

# AI Provider configuration keys
CONF_LLAMA_TOKEN = "llama_token"  # nosec B105
CONF_OPENAI_TOKEN = "openai_token"  # nosec B105
CONF_GEMINI_TOKEN = "gemini_token"  # nosec B105
CONF_OPENROUTER_TOKEN = "openrouter_token"  # nosec B105
CONF_ANTHROPIC_TOKEN = "anthropic_token"  # nosec B105
CONF_ALTER_TOKEN = "alter_token"  # nosec B105
CONF_ZAI_TOKEN = "zai_token"  # nosec B105
CONF_LOCAL_URL = "local_url"
CONF_LOCAL_MODEL = "local_model"
CONF_OPENAI_BASE_URL = (
    "openai_base_url"  # Optional custom endpoint for OpenAI-compatible servers
)
CONF_ASKSAGE_TOKEN = "asksage_token"  # nosec B105
CONF_ASKSAGE_LIVE = "asksage_live"  # 0=off, 1=Live (Google), 2=Live+ (Google+crawl)
CONF_ASKSAGE_DEEP_AGENT = "asksage_deep_agent"  # bool — enable Deep Agent mode

# Available AI providers
AI_PROVIDERS = [
    "llama",
    "openai",
    "gemini",
    "openrouter",
    "anthropic",
    "alter",
    "zai",
    "local",
    "asksage",
]

# Chat persistence
CONF_PERSIST_CHAT_HISTORY = "persist_chat_history"

# SSE Streaming
CONF_ENABLE_STREAMING = "enable_streaming"

# AI Provider constants
CONF_MODELS = "models"

# Supported AI providers
DEFAULT_AI_PROVIDER = "openai"
