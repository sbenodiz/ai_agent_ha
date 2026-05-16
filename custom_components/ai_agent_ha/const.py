"""Constants for the AI Agent HA integration."""

DOMAIN = "ai_agent_ha"
CONF_API_KEY = "api_key"
CONF_WEATHER_ENTITY = "weather_entity"

# AI Provider configuration keys
CONF_LLAMA_TOKEN = "llama_token"  # nosec B105
CONF_OPENAI_TOKEN = "openai_token"  # nosec B105
CONF_OPENAI_BASE_URL = (
    "openai_base_url"  # nosec B105 - configuration key, not a credential
)
CONF_GEMINI_TOKEN = "gemini_token"  # nosec B105
CONF_OPENROUTER_TOKEN = "openrouter_token"  # nosec B105
CONF_ANTHROPIC_TOKEN = "anthropic_token"  # nosec B105
CONF_ALTER_TOKEN = "alter_token"  # nosec B105
CONF_ZAI_TOKEN = "zai_token"  # nosec B105
CONF_LOCAL_OLLAMA_URL = "local_ollama_url"
CONF_LOCAL_OLLAMA_MODEL = "local_ollama_model"
CONF_OPENAI_COMPATIBLE_URL = "openai_compatible_url"
CONF_LOCAL_URL = "local_url"  # legacy alias for local_ollama_url

# Available AI providers
AI_PROVIDERS = [
    "llama",
    "openai",
    "gemini",
    "openrouter",
    "anthropic",
    "alter",
    "zai",
    "local_ollama",
    "openai_compatible",
]

# AI Provider constants
CONF_MODELS = "models"

# Supported AI providers
DEFAULT_AI_PROVIDER = "openai"
