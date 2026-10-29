import { csrfHeaders } from "./csrf";
import { errorCodeFromPayload } from "@/utils/apiError";

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
  /** Internal: this request already retried after a step-up prompt. */
  stepUpAttempted?: boolean;
}

/** Auth endpoints that are genuinely anonymous.
 *
 * Only endpoints reachable without a console session belong here: the list
 * exists to stop a 401 from triggering handleUnauthorized() when the failure
 * is not "your session is gone" (a wrong password at step-up, an expired
 * invite/2FA/challenge token, an invalid email-verification link).
 *
 * Authenticated endpoints must NOT be added, even when their 401 means
 * something else: an expired or revoked session hitting a passkey
 * registration or password change has to clear the stale token and send the
 * user to /login instead of leaving them retrying against a dead session.
 */
const PUBLIC_AUTH_PATHS = new Set([
  "/api/auth/login",
  "/api/auth/setup",
  "/api/auth/setup-status",
  "/api/auth/capabilities",
  "/api/auth/session",
  "/api/auth/logout",
  "/api/auth/2fa/verify",
  "/api/auth/2fa/recover",
  "/api/auth/passkeys/login/begin",
  "/api/auth/passkeys/login/finish",
  "/api/auth/step-up",
  "/api/auth/email/verify",
  "/api/auth/password/forgot",
  "/api/auth/password/reset",
]);

function isPublicAuthPath(url: string): boolean {
  const path = url.split("?")[0];
  return (
    PUBLIC_AUTH_PATHS.has(path) ||
    path.startsWith("/api/auth/invitations/") ||
    path.startsWith("/api/auth/oauth/")
  );
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

/**
 * The user dismissed the step-up dialog.
 *
 * Distinct from the original ``step_up_required`` rejection: re-throwing that
 * payload made every caller report "re-authentication required" for what was
 * a deliberate cancel.
 */
export class StepUpCancelledError extends Error {
  constructor(message: string = "Re-authentication cancelled") {
    super(message);
    this.name = "StepUpCancelledError";
  }
}

// Re-entrancy guard: prevents recursive handleUnauthorized() calls (e.g. when
// the logout API call itself returns 401 after the JWT was already cleared).
let isHandlingUnauthorized = false;

function hasErrorCode(data: unknown, expected: string): boolean {
  return errorCodeFromPayload(data) === expected;
}

function handleMfaEnrollmentRequired(): void {
  Promise.all([import("@/router")])
    .then(([{ default: router }]) => {
      if (router.currentRoute.value.name !== "security") {
        router.push("/account/security").catch(() => {});
      }
    })
    .catch(() => {});
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

export function handleUnauthorized(reason: string | null = null): void {
  if (isHandlingUnauthorized) return;
  isHandlingUnauthorized = true;

  // Clear the full local session (identity + all cached stores). The JWT and
  // /v1 key live in HttpOnly cookies, which the browser will drop when they
  // expire; the store must forget the identity immediately so no queued render
  // treats this tab as signed in.
  // Credentials alone are not enough: the per-user list caches (catalog is
  // allowlist-filtered server-side) would otherwise survive and be served to
  // the next login on this browser.
  // We intentionally do NOT call the backend /logout endpoint here:
  // - The token is already invalid/expired; the backend can't attribute it.
  // - Calling logout creates noise in audit logs (duplicate logout entries
  //   with no user identity).
  // - Session API key cleanup happens on the server when the JWT expires.
  import("@/stores/auth").then(({ useAuthStore }) => {
    try {
      const store = useAuthStore();
      store.clearLocalSession();
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
        const params = new URLSearchParams({ redirect: currentRoute.fullPath });
        if (reason === "session_revoked") {
          params.set("reason", reason);
        }
        router.push(`/login?${params.toString()}`).finally(() => {
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
  const headers = new Headers(config.headers);
  headers.set("Content-Type", "application/json");
  for (const [name, value] of Object.entries(csrfHeaders(config.method))) {
    if (!headers.has(name)) headers.set(name, value);
  }

  // Build an AbortSignal that combines the caller's signal (if any) with a
  // timeout so that no request can hang indefinitely. Endpoints that need
  // longer than the default (e.g. image generation) pass timeoutMs.
  const timeoutSignal = AbortSignal.timeout(config.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  const signal = config.signal ? AbortSignal.any([config.signal, timeoutSignal]) : timeoutSignal;

  let response: Response;
  try {
    response = await fetch(url, {
      credentials: "same-origin",
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
    let data: unknown;
    const text = await response.text();
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }

    if (response.status === 401 && !isPublicAuthPath(url)) {
      handleUnauthorized(errorCodeFromPayload(data));
    }

    if (response.status === 403 && hasErrorCode(data, "password_change_required")) {
      handlePasswordChangeRequired();
    }
    if (response.status === 403 && hasErrorCode(data, "mfa_enrollment_required")) {
      handleMfaEnrollmentRequired();
    }
    if (
      response.status === 403 &&
      hasErrorCode(data, "step_up_required") &&
      !config.stepUpAttempted
    ) {
      // One dialog serves every queued sensitive request. Retry the original
      // call once the user proves a fresh factor; the retry gets its own
      // timeout signal, so time spent in the dialog does not consume it.
      const { useStepUpStore } = await import("@/stores/stepUp");
      try {
        await useStepUpStore().requireStepUp();
      } catch {
        throw new StepUpCancelledError();
      }
      return request<T>(url, { ...config, stepUpAttempted: true });
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
  patch: <T>(url: string, body: unknown, config?: RequestConfig) =>
    request<T>(url, { ...config, method: "PATCH", body: JSON.stringify(body) }),
  delete: <T>(url: string, config?: RequestConfig) =>
    request<T>(url, { ...config, method: "DELETE" }),
};
