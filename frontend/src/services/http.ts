import { tokenStorage } from "./storage";

/** Default request timeout in milliseconds (30 seconds). */
const DEFAULT_TIMEOUT_MS = 30_000;

/**
 * Request config: RequestInit plus an optional per-request timeout override.
 * Long-running endpoints (e.g. image generation) pass a larger timeoutMs so
 * the default 30s abort does not fire while the backend is still working.
 */
export interface RequestConfig extends RequestInit {
  /** Override the default request timeout in milliseconds. */
  timeoutMs?: number;
}

/** Auth endpoints accessible without a token (login + first-run setup + logout).
 * Logout is included so a 401 from the logout call itself (when the JWT was
 * already cleared) does not re-trigger handleUnauthorized() and loop.
 */
const PUBLIC_AUTH_PATHS = new Set([
  "/api/auth/login",
  "/api/auth/setup",
  "/api/auth/setup-status",
  "/api/auth/logout",
]);

function isPublicAuthPath(url: string): boolean {
  return PUBLIC_AUTH_PATHS.has(url.split("?")[0]);
}

export class HttpError extends Error {
  status: number;
  statusText: string;
  data: unknown;

  constructor(status: number, statusText: string, data: unknown) {
    super(`HTTP Error ${status}: ${statusText}`);
    this.status = status;
    this.statusText = statusText;
    this.data = data;
  }
}

export class NetworkError extends Error {
  constructor(message: string = "Network error - please check your connection") {
    super(message);
    this.name = "NetworkError";
  }
}

export class TimeoutError extends Error {
  constructor(message: string = "Request timed out - the server took too long to respond") {
    super(message);
    this.name = "TimeoutError";
  }
}

// Re-entrancy guard: prevents recursive handleUnauthorized() calls (e.g. when
// the logout API call itself returns 401 after the JWT was already cleared).
let isHandlingUnauthorized = false;

/** Detect the forced-password-change signal on a 403 body. The JWT
 * middleware returns the code at the top level of the error body. */
function hasPasswordChangeRequiredCode(data: unknown): boolean {
  if (!data || typeof data !== "object") return false;
  return (data as { code?: unknown }).code === "password_change_required";
}

/** Funnel the user into the forced password change flow. Triggered when any
 * request is rejected with 403 `password_change_required` — e.g. an admin
 * reset the password while this session was still live. The store flag is
 * set before the router push so the navigation guard sees it (the guard
 * redirects the forced route away when the flag is not set). */
function handlePasswordChangeRequired(): void {
  // Dynamic imports to avoid circular dependencies
  Promise.all([import("@/stores/auth"), import("@/router")])
    .then(([{ useAuthStore }, { default: router }]) => {
      try {
        useAuthStore().setMustChangePassword(true);
      } catch (e) {
        console.error("Failed to set mustChangePassword on auth store:", e);
      }
      try {
        const currentRoute = router.currentRoute.value;
        if (currentRoute.name !== "forceChangePassword" && currentRoute.name !== "login") {
          router.push("/force-change-password").catch(() => {});
        }
      } catch (e) {
        console.error("Failed to redirect to forced password change:", e);
      }
    })
    .catch((e) => console.error("Failed to handle forced password change:", e));
}

export function handleUnauthorized(): void {
  if (isHandlingUnauthorized) return;
  isHandlingUnauthorized = true;

  // Remove token from localStorage immediately so queued HTTP requests
  // won't include the expired JWT before the reactive store is updated.
  tokenStorage.remove();

  // Clear reactive store state synchronously (no API call needed).
  // We intentionally do NOT call the backend /logout endpoint here:
  // - The token is already invalid/expired; the backend can't attribute it.
  // - Calling logout creates noise in audit logs (duplicate logout entries
  //   with no user identity).
  // - Session API key cleanup happens on the server when the JWT expires.
  import("@/stores/auth").then(({ useAuthStore }) => {
    try {
      const store = useAuthStore();
      store.clearCredentials();
    } catch (e) {
      console.error("Failed to clear authStore on unauthorized error:", e);
    }
  });

  // Dynamic router import to avoid circular dependency
  import("@/router").then(({ default: router }) => {
    try {
      const currentRoute = router.currentRoute.value;
      // Use route name (not fullPath) to detect login page — fullPath fails
      // when the URL includes a query string like "/login?redirect=...".
      if (currentRoute.name !== "login") {
        const redirect = currentRoute.fullPath;
        router.push(`/login?redirect=${encodeURIComponent(redirect)}`).finally(() => {
          isHandlingUnauthorized = false;
        });
      } else {
        isHandlingUnauthorized = false;
      }
    } catch (e) {
      isHandlingUnauthorized = false;
      console.error("Failed to redirect using router:", e);
    }
  });
}

async function request<T>(url: string, config: RequestConfig = {}): Promise<T> {
  const token = tokenStorage.get();
  const headers = new Headers(config.headers);
  headers.set("Content-Type", "application/json");
  if (token && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  // Build an AbortSignal that combines the caller's signal (if any) with a
  // timeout so that no request can hang indefinitely. Endpoints that need
  // longer than the default (e.g. image generation) pass timeoutMs.
  const timeoutSignal = AbortSignal.timeout(config.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  const signal = config.signal ? AbortSignal.any([config.signal, timeoutSignal]) : timeoutSignal;

  let response: Response;
  try {
    response = await fetch(url, {
      ...config,
      headers,
      signal,
    });
  } catch (e) {
    if (e instanceof DOMException && e.name === "TimeoutError") {
      throw new TimeoutError(
        "Request timed out. The server took too long to respond. Please try again."
      );
    }
    if (e instanceof TypeError) {
      throw new NetworkError("Unable to reach the server. Please try again later.");
    }
    throw e;
  }

  if (!response.ok) {
    if (response.status === 401 && !isPublicAuthPath(url)) {
      handleUnauthorized();
    }

    let data: unknown;
    const text = await response.text();
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }

    if (response.status === 403 && hasPasswordChangeRequiredCode(data)) {
      handlePasswordChangeRequired();
    }

    throw new HttpError(response.status, response.statusText, data);
  }

  if (response.status === 204) {
    return {} as T;
  }

  try {
    return await response.json();
  } catch {
    return (await response.text()) as unknown as T;
  }
}

export const http = {
  get: <T>(url: string, config?: RequestConfig) => request<T>(url, { ...config, method: "GET" }),
  post: <T>(url: string, body: unknown, config?: RequestConfig) =>
    request<T>(url, { ...config, method: "POST", body: JSON.stringify(body) }),
  put: <T>(url: string, body: unknown, config?: RequestConfig) =>
    request<T>(url, { ...config, method: "PUT", body: JSON.stringify(body) }),
  delete: <T>(url: string, config?: RequestConfig) =>
    request<T>(url, { ...config, method: "DELETE" }),
};
