// frontend/src/services/http.test.ts
import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it, vi } from "vitest";

const authStoreMock = vi.hoisted(() => ({
  clearCredentials: vi.fn(),
  clearLocalSession: vi.fn(),
  setMustChangePassword: vi.fn(),
}));
const routerMock = vi.hoisted(() => ({
  currentRoute: { value: { name: "models", fullPath: "/models" } },
  push: vi.fn(() => Promise.resolve()),
}));

vi.mock("@/stores/auth", () => ({
  useAuthStore: () => authStoreMock,
}));

vi.mock("@/router", () => ({
  default: routerMock,
}));

beforeEach(() => {
  vi.resetModules();
  setActivePinia(createPinia());
  authStoreMock.clearCredentials.mockClear();
  authStoreMock.clearLocalSession.mockClear();
  authStoreMock.setMustChangePassword.mockClear();
  routerMock.push.mockClear();
});

describe("step-up retry", () => {
  it("retries the original request once after re-authentication", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            error: {
              code: "step_up_required",
              message: "Please re-authenticate before performing this action",
            },
          }),
          { status: 403, headers: { "Content-Type": "application/json" } }
        )
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ ok: true }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      );
    vi.stubGlobal("fetch", fetchMock);

    const { http } = await import("./http");
    const { useStepUpStore } = await import("@/stores/stepUp");
    const store = useStepUpStore();

    const result = http.post<{ ok: boolean }>("/api/team/invitations", { email: "a@example.com" });
    await vi.waitFor(() => expect(store.required).toBe(true));
    store.complete();

    await expect(result).resolves.toEqual({ ok: true });
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[1][1].body).toBe(JSON.stringify({ email: "a@example.com" }));
  });

  it.each([
    ["password_change_required", "/force-change-password"],
    ["mfa_enrollment_required", "/account/security"],
  ])("routes a nested %s signal without treating it as a generic 403", async (code, route) => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValueOnce(
        new Response(JSON.stringify({ error: { code, message: "Action required" } }), {
          status: 403,
          headers: { "Content-Type": "application/json" },
        })
      )
    );

    const { http } = await import("./http");
    await expect(http.get("/api/me/profile")).rejects.toThrow("HTTP Error 403");

    await vi.waitFor(() => expect(routerMock.push).toHaveBeenCalledWith(route));
    if (code === "password_change_required") {
      expect(authStoreMock.setMustChangePassword).toHaveBeenCalledWith(true);
    }
  });
});

describe("cookie transport", () => {
  it("sends the double-submit CSRF header on unsafe requests", async () => {
    document.cookie = "llm_proxy_csrf=csrf-value; path=/";
    const fetchMock = vi.fn().mockResolvedValueOnce(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    );
    vi.stubGlobal("fetch", fetchMock);

    const { http } = await import("./http");
    await http.post("/api/me/username", { new_username: "admin" });

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(init.credentials).toBe("same-origin");
    expect(new Headers(init.headers).get("X-CSRF-Token")).toBe("csrf-value");
    document.cookie = "llm_proxy_csrf=; Max-Age=0; path=/";
  });

  it("does not add a CSRF header to safe reads", async () => {
    document.cookie = "llm_proxy_csrf=csrf-value; path=/";
    const fetchMock = vi.fn().mockResolvedValueOnce(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    );
    vi.stubGlobal("fetch", fetchMock);

    const { http } = await import("./http");
    await http.get("/api/me/profile");

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(new Headers(init.headers).has("X-CSRF-Token")).toBe(false);
    document.cookie = "llm_proxy_csrf=; Max-Age=0; path=/";
  });
});

describe("handleUnauthorized", () => {
  it("tears down the full local session, not just credentials", async () => {
    // A 401 (expired/revoked token) must not leave per-user store caches
    // behind for the next login on this browser.
    // Dynamic import: handleUnauthorized is guarded by the module-level
    // isHandlingUnauthorized flag, so each test needs a fresh module
    // (paired with vi.resetModules above).
    const { handleUnauthorized } = await import("./http");
    handleUnauthorized();

    await vi.waitFor(() => expect(authStoreMock.clearLocalSession).toHaveBeenCalled());
    expect(authStoreMock.clearCredentials).not.toHaveBeenCalled();
  });

  it("tells the login screen when a newer sign-in revoked the session", async () => {
    const { handleUnauthorized } = await import("./http");
    handleUnauthorized("session_revoked");

    await vi.waitFor(() =>
      expect(routerMock.push).toHaveBeenCalledWith(
        "/login?redirect=%2Fmodels&reason=session_revoked"
      )
    );
  });
});
