/**
 * Provider constants shared by the config UI and the OAuth sign-in flow.
 *
 * Each of these is a contract with the backend: the provider type is what the
 * adapter registry dispatches on, the callback path is where the OAuth redirect
 * lands, and the message type is what the callback page uses to reach the
 * dashboard. They live here so a rename is one edit instead of a hunt across
 * components and translations.
 */

/** Provider type whose upstream credentials come from an OAuth sign-in. */
export const OPENAI_CHATGPT_PROVIDER_TYPE = "openai-chatgpt";

/**
 * postMessage contract between the OAuth callback page
 * (`_render_callback_html`) and the dashboard listening for it.
 */
export const OAUTH_CALLBACK_MESSAGE_TYPE = "llm-proxy-provider-oauth";

/** Route the backend serves for the OAuth redirect (see `provider_oauth.py`). */
export const OAUTH_CALLBACK_PATH = "/api/config/providers/oauth/callback";

/**
 * Loopback callback OpenAI accepts for deployments that cannot expose a public
 * callback route: the operator pastes the redirected URL back into the console.
 */
export const OAUTH_LOOPBACK_REDIRECT_URI = "http://127.0.0.1:1455/auth/callback";

/** Plan-usage scope the granted token must carry (`PLAN_USAGE_SCOPE` upstream). */
export const PLAN_USAGE_SCOPE = "chatgpt.tokens.use.direct";

/** Client id sent to OpenAI when dynamic client registration is in use. */
export const DYNAMIC_AGENT_CLIENT_ID = "dynamic_agent_client";

const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "::1", "[::1]"]);

/** True when the console is served from a host OpenAI treats as local. */
export function isLoopbackHost(hostname: string): boolean {
  return LOOPBACK_HOSTS.has(hostname.trim().toLowerCase());
}

/**
 * Redirect URI to offer in the sign-in dialog.
 *
 * A loopback console can complete the flow on its own callback route; any other
 * deployment has to use the loopback URI and paste the result back, because
 * OpenAI only accepts a pre-registered callback.
 */
export function defaultOAuthRedirectUri(
  location: Pick<Location, "origin" | "hostname"> | undefined = typeof window === "undefined"
    ? undefined
    : window.location
): string {
  if (!location || !isLoopbackHost(location.hostname)) {
    return OAUTH_LOOPBACK_REDIRECT_URI;
  }
  return `${location.origin}${OAUTH_CALLBACK_PATH}`;
}
