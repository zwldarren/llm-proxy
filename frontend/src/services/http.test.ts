// frontend/src/services/http.test.ts
import { beforeEach, describe, expect, it, vi } from "vitest";

const authStoreMock = vi.hoisted(() => ({
  clearCredentials: vi.fn(),
  clearLocalSession: vi.fn(),
}));

vi.mock("@/stores/auth", () => ({
  useAuthStore: () => authStoreMock,
}));

vi.mock("@/router", () => ({
  default: {
    currentRoute: { value: { name: "models" } },
    push: vi.fn(() => Promise.resolve()),
  },
}));

beforeEach(() => {
  vi.resetModules();
  authStoreMock.clearCredentials.mockClear();
  authStoreMock.clearLocalSession.mockClear();
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
});
