/** Cookie/header pair used by the backend's double-submit CSRF check. */
const CSRF_COOKIE = "llm_proxy_csrf";
const CSRF_HEADER = "X-CSRF-Token";
const SAFE_METHODS: Record<string, true> = {
  GET: true,
  HEAD: true,
  OPTIONS: true,
  TRACE: true,
};

function readCookie(name: string): string | null {
  const prefix = `${name}=`;
  for (const part of document.cookie.split(";")) {
    const entry = part.trim();
    if (entry.startsWith(prefix)) return decodeURIComponent(entry.slice(prefix.length));
  }
  return null;
}

/**
 * Headers required when an unsafe request may authenticate with the session
 * cookie. The token is sent unconditionally on unsafe methods so the console's
 * own `/api` and `/v1` calls share one transport path.
 */
export function csrfHeaders(method: string | undefined): Record<string, string> {
  if (SAFE_METHODS[(method ?? "GET").toUpperCase()]) return {};
  const token = readCookie(CSRF_COOKIE);
  return token ? { [CSRF_HEADER]: token } : {};
}
