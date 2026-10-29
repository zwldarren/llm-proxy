/**
 * Centralized localStorage key definitions.
 * Every storage key used in the application should be defined here
 * to avoid duplication, typos, and scattered string literals.
 */
export const STORAGE_KEYS = {
  // Legacy auth keys. Credentials now live only in HttpOnly cookies; these
  // names remain so an upgrade can delete values left by older versions.
  AUTH_TOKEN: "auth_token",
  SESSION_API_KEY: "session_api_key",
  MUST_CHANGE_PASSWORD: "must_change_password",
  // Device-local UX hints: usernames that have registered or used a passkey on
  // this browser. Used only to choose the primary login action; never exposed.
  PASSKEY_ACCOUNTS: "passkey_accounts",

  // Chat
  CHAT_MESSAGES: "llm-proxy-chat-messages",
  CHAT_RUNS: "llm-proxy-chat-runs",
  CHAT_MODEL: "llm-proxy-chat-model",
  CHAT_ENDPOINT: "llm-proxy-chat-endpoint",
  CHAT_TOOLS: "llm-proxy-chat-tools",
  CHAT_WEB_SEARCH: "llm-proxy-chat-web-search",
  SPEECH_VOICE: "speech_voice",
  SPEECH_SPEED: "speech_speed",
  SPEECH_MODEL: "speech_model",

  // Images
  IMAGES_MODEL: "llm-proxy-images-model",

  // UI state
  MODELS_VIEW_MODE: "llm-proxy:models-view-mode",
  PLAZA_VIEW_MODE: "llm-proxy:plaza-view-mode",
  PROVIDERS_VIEW_MODE: "llm-proxy:providers-view-mode",
  API_KEYS_VIEW_MODE: "llm-proxy:api-keys-view-mode",
  MCP_SERVERS_VIEW_MODE: "llm-proxy:mcp-servers-view-mode",
  SIDEBAR_EXPANDED_GROUPS: "llm-proxy:sidebar-expanded-groups",

  // Locale
  LOCALE: "locale",
} as const;
