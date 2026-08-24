"""Constants used throughout the LLM Proxy."""

# Default retry and fallback settings
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_FALLBACK_ATTEMPTS = 10
DEFAULT_DISCONNECT_CHECK_INTERVAL = 50

# =============================================================================
# PERFORMANCE OPTIMIZATION CONFIGURATION
# =============================================================================

# Token encoding cache size (LRU cache max entries)
TOKEN_ENCODING_CACHE_SIZE = 10000

# API key cache TTL in seconds
API_KEY_CACHE_TTL_SECONDS = 60

# Lockout cleanup interval in seconds
LOCKOUT_CLEANUP_INTERVAL_SECONDS = 60
