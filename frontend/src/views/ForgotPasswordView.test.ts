import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";

const api = vi.hoisted(() => ({
  getCapabilities: vi.fn(),
  requestPasswordReset: vi.fn(),
}));
const router = vi.hoisted(() => ({ push: vi.fn() }));
const route = vi.hoisted(() => ({ query: {} as Record<string, string> }));

// Partial mock: utils/error.ts builds the app i18n instance at import time.
vi.mock("vue-i18n", async (importOriginal) => {
  const actual = await importOriginal<typeof import("vue-i18n")>();
  return { ...actual, useI18n: () => ({ t: (key: string) => key }) };
});
vi.mock("vue-router", () => ({
  useRoute: () => route,
  useRouter: () => router,
}));
vi.mock("@/services/api/auth", () => ({ authApi: api }));
vi.mock("@lucide/vue", () => {
  const Stub = () => null;
  return { ArrowLeft: Stub, Loader2: Stub };
});
vi.mock("@/components/common/AuthShell.vue", () => ({
  __esModule: true,
  default: { template: "<div><slot /></div>" },
}));

import ForgotPasswordView from "./ForgotPasswordView.vue";

describe("ForgotPasswordView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    route.query = {};
    api.getCapabilities.mockResolvedValue({ password_reset: true });
  });

  it("asks for an identifier and reports the request generically", async () => {
    api.requestPasswordReset.mockResolvedValue({ requested: true });
    const wrapper = mount(ForgotPasswordView);
    await flushPromises();

    await wrapper.find("#forgot-identifier").setValue("  alice@example.com  ");
    await wrapper.find("form").trigger("submit.prevent");
    await flushPromises();

    expect(api.requestPasswordReset).toHaveBeenCalledWith("alice@example.com");
    // The message must not depend on what the backend found: no account was
    // named, and the confirmation is the same sentence either way.
    expect(wrapper.find("#forgot-sent").text()).toBe("auth.forgotSent");
    expect(wrapper.find("form").exists()).toBe(false);
  });

  it("starts from the identifier the sign-in page already knows", async () => {
    route.query = { identifier: "operator" };
    const wrapper = mount(ForgotPasswordView);
    await flushPromises();

    expect((wrapper.find("#forgot-identifier").element as HTMLInputElement).value).toBe("operator");
  });

  it("refuses to submit an empty identifier", async () => {
    const wrapper = mount(ForgotPasswordView);
    await flushPromises();

    await wrapper.find("form").trigger("submit.prevent");
    await flushPromises();

    expect(api.requestPasswordReset).not.toHaveBeenCalled();
    expect(wrapper.find("#forgot-error").text()).toBe("auth.identifierRequired");
  });

  it("sends the user to an administrator when the instance cannot email links", async () => {
    api.getCapabilities.mockResolvedValue({ password_reset: false });
    const wrapper = mount(ForgotPasswordView);
    await flushPromises();

    expect(wrapper.find("#forgot-unavailable").text()).toBe("auth.forgotUnavailable");
    expect(wrapper.find("form").exists()).toBe(false);
  });

  it("treats an unreadable capability as unavailable", async () => {
    api.getCapabilities.mockRejectedValue(new Error("offline"));
    const wrapper = mount(ForgotPasswordView);
    await flushPromises();

    expect(wrapper.find("#forgot-unavailable").exists()).toBe(true);
    expect(wrapper.find("form").exists()).toBe(false);
  });

  it("keeps the form when the request itself fails", async () => {
    api.requestPasswordReset.mockRejectedValue(new Error("boom"));
    const wrapper = mount(ForgotPasswordView);
    await flushPromises();

    await wrapper.find("#forgot-identifier").setValue("alice");
    await wrapper.find("form").trigger("submit.prevent");
    await flushPromises();

    expect(wrapper.find("#forgot-sent").exists()).toBe(false);
    expect(wrapper.find("form").exists()).toBe(true);
  });
});
