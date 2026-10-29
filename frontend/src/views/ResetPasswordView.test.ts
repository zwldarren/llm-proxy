import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";

const api = vi.hoisted(() => ({ resetPassword: vi.fn() }));
const toast = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn() }));
const router = vi.hoisted(() => ({ push: vi.fn(), replace: vi.fn() }));
const route = vi.hoisted(() => ({ query: {} as Record<string, string> }));

// Partial mock: utils/password.ts and utils/error.ts build the app i18n
// instance at import time.
vi.mock("vue-i18n", async (importOriginal) => {
  const actual = await importOriginal<typeof import("vue-i18n")>();
  return { ...actual, useI18n: () => ({ t: (key: string) => key }) };
});
vi.mock("vue-router", () => ({
  useRoute: () => route,
  useRouter: () => router,
}));
vi.mock("vue-sonner", () => ({ toast }));
vi.mock("@/services/api/auth", () => ({ authApi: api }));
vi.mock("@lucide/vue", () => {
  const Stub = () => null;
  return { ArrowLeft: Stub, Loader2: Stub };
});
vi.mock("@/components/common/AuthShell.vue", () => ({
  __esModule: true,
  default: { template: "<div><slot /></div>" },
}));

import ResetPasswordView from "./ResetPasswordView.vue";

const NEW_PASSWORD = "Brand-New-Password-456!";

describe("ResetPasswordView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    route.query = { token: "reset-token" };
  });

  it("sets the password once both fields agree", async () => {
    api.resetPassword.mockResolvedValue({ success: true });
    const wrapper = mount(ResetPasswordView);
    await flushPromises();

    await wrapper.find("#reset-new-password").setValue(NEW_PASSWORD);
    await wrapper.find("#reset-confirm-password").setValue(NEW_PASSWORD);
    await wrapper.find("form").trigger("submit.prevent");
    await flushPromises();

    expect(api.resetPassword).toHaveBeenCalledWith("reset-token", NEW_PASSWORD);
    // Nobody is signed in by a reset: every session was revoked server-side.
    expect(router.replace).toHaveBeenCalledWith("/login");
  });

  it("reports a link without a token instead of showing a dead form", async () => {
    route.query = {};
    const wrapper = mount(ResetPasswordView);
    await flushPromises();

    expect(wrapper.find("#reset-missing-token").text()).toBe("auth.resetMissingToken");
    expect(wrapper.find("form").exists()).toBe(false);
    expect(api.resetPassword).not.toHaveBeenCalled();
  });

  it("does not submit mismatched passwords", async () => {
    const wrapper = mount(ResetPasswordView);
    await flushPromises();

    await wrapper.find("#reset-new-password").setValue(NEW_PASSWORD);
    await wrapper.find("#reset-confirm-password").setValue("Brand-New-Password-456?");
    await wrapper.find("form").trigger("submit.prevent");
    await flushPromises();

    expect(api.resetPassword).not.toHaveBeenCalled();
    expect(wrapper.find("#reset-error").text()).toBe("auth.passwordMismatch");
  });

  it("does not submit a password below the strength policy", async () => {
    const wrapper = mount(ResetPasswordView);
    await flushPromises();

    await wrapper.find("#reset-new-password").setValue("short");
    await wrapper.find("#reset-confirm-password").setValue("short");
    await wrapper.find("form").trigger("submit.prevent");
    await flushPromises();

    expect(api.resetPassword).not.toHaveBeenCalled();
    expect(wrapper.find("#reset-error").text()).toBe("auth.passwordTooShort");
  });

  it("keeps the form and shows the reason when the link is spent", async () => {
    api.resetPassword.mockRejectedValue(new Error("boom"));
    const wrapper = mount(ResetPasswordView);
    await flushPromises();

    await wrapper.find("#reset-new-password").setValue(NEW_PASSWORD);
    await wrapper.find("#reset-confirm-password").setValue(NEW_PASSWORD);
    await wrapper.find("form").trigger("submit.prevent");
    await flushPromises();

    expect(router.replace).not.toHaveBeenCalled();
    expect(wrapper.find("#reset-error").exists()).toBe(true);
    expect(wrapper.find("form").exists()).toBe(true);
  });
});
