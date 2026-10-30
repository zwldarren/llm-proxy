import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { STORAGE_KEYS } from "@/constants/storageKeys";

const auth = vi.hoisted(() => ({
  login: vi.fn(),
  loginWithPasskey: vi.fn(),
  listOAuthProviders: vi.fn(),
  getCapabilities: vi.fn(),
}));
const router = vi.hoisted(() => ({ push: vi.fn() }));
const route = vi.hoisted(() => ({ query: {} as Record<string, string> }));

// Partially mock so modules that build the app i18n instance (utils/error.ts)
// still resolve ``createI18n``; only the composable is stubbed.
vi.mock("vue-i18n", async (importOriginal) => {
  const actual = await importOriginal<typeof import("vue-i18n")>();
  return { ...actual, useI18n: () => ({ t: (key: string) => key }) };
});
vi.mock("vue-router", () => ({
  useRoute: () => route,
  useRouter: () => router,
  RouterLink: { props: ["to", "id"], template: "<a><slot /></a>" },
}));
vi.mock("@/stores/auth", () => ({
  useAuthStore: () => ({
    isAuthenticated: true,
    login: auth.login,
    loginWithPasskey: auth.loginWithPasskey,
  }),
}));
vi.mock("@/services/api/auth", () => ({
  authApi: {
    listOAuthProviders: auth.listOAuthProviders,
    getCapabilities: auth.getCapabilities,
  },
}));
vi.mock("@lucide/vue", () => {
  const Stub = () => null;
  return {
    ArrowLeft: Stub,
    Check: Stub,
    ChevronDown: Stub,
    ChevronUp: Stub,
    CircleAlert: Stub,
    Eye: Stub,
    EyeOff: Stub,
    Fingerprint: Stub,
    Info: Stub,
    KeyRound: Stub,
    Languages: Stub,
    Loader2: Stub,
    Monitor: Stub,
    Moon: Stub,
    Sun: Stub,
  };
});

import LoginView from "./LoginView.vue";

describe("LoginView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    route.query = {};
    auth.listOAuthProviders.mockResolvedValue({ providers: [] });
    auth.getCapabilities.mockResolvedValue({ password_reset: true });
    // happy-dom has no WebAuthn; the login page only offers passkeys when the
    // browser exposes it in a secure context.
    vi.stubGlobal("PublicKeyCredential", class {});
    Object.defineProperty(window, "isSecureContext", { configurable: true, value: true });
  });

  afterEach(() => {
    localStorage.clear();
    vi.unstubAllGlobals();
  });

  it("keeps the password primary on a browser with no proven passkey", async () => {
    const wrapper = mount(LoginView);
    await flushPromises();

    const buttons = wrapper.findAll("button");
    const passkey = buttons.find((button) => button.text().includes("auth.passkeyLogin"));
    const password = buttons.find((button) => button.text().includes("auth.loginButton"));

    expect(passkey).toBeDefined();
    expect(password).toBeDefined();
    expect(passkey!.classes()).toContain("border");
    expect(password!.classes()).toContain("bg-primary");
  });

  it("promotes passkey sign-in after this browser has proven one", async () => {
    localStorage.setItem(STORAGE_KEYS.PASSKEY_ACCOUNTS, JSON.stringify(["operator"]));
    const wrapper = mount(LoginView);
    await flushPromises();

    const buttons = wrapper.findAll("button");
    const passkey = buttons.find((button) => button.text().includes("auth.passkeyLogin"));
    const password = buttons.find((button) => button.text().includes("auth.loginButton"));

    expect(passkey).toBeDefined();
    expect(password).toBeDefined();
    expect(passkey!.classes()).toContain("bg-primary");
    expect(password!.classes()).toContain("border");
  });

  it("does not promote passkey for a different account on the same browser", async () => {
    localStorage.setItem(STORAGE_KEYS.PASSKEY_ACCOUNTS, JSON.stringify(["operator"]));
    const wrapper = mount(LoginView);
    await flushPromises();
    await wrapper.find("#username").setValue("another-user");
    await flushPromises();

    const buttons = wrapper.findAll("button");
    const passkey = buttons.find((button) => button.text().includes("auth.passkeyLogin"));
    const password = buttons.find((button) => button.text().includes("auth.loginButton"));

    expect(passkey!.classes()).toContain("border");
    expect(password!.classes()).toContain("bg-primary");
  });

  it("points at the identifier instead of sending an empty form", async () => {
    const wrapper = mount(LoginView);
    await flushPromises();

    await wrapper.find("form").trigger("submit.prevent");
    await flushPromises();

    expect(auth.login).not.toHaveBeenCalled();
    expect(wrapper.find("#identifier-error").text()).toBe("auth.identifierRequired");
    expect(wrapper.find("#password-error").exists()).toBe(false);
  });

  it("asks for the password when only the identifier was filled in", async () => {
    const wrapper = mount(LoginView);
    await flushPromises();

    await wrapper.find("#username").setValue("operator");
    await wrapper.find("form").trigger("submit.prevent");
    await flushPromises();

    expect(auth.login).not.toHaveBeenCalled();
    expect(wrapper.find("#password-error").text()).toBe("auth.enterPassword");
    expect(wrapper.find("#identifier-error").exists()).toBe(false);
  });

  it("lets the operator reveal the typed password", async () => {
    const wrapper = mount(LoginView);
    await flushPromises();

    expect(wrapper.find("#password").attributes("type")).toBe("password");

    const reveal = wrapper
      .findAll("button")
      .find((button) => button.attributes("aria-label") === "auth.showPassword");
    expect(reveal).toBeDefined();
    await reveal!.trigger("click");

    expect(wrapper.find("#password").attributes("type")).toBe("text");
    expect(
      wrapper
        .findAll("button")
        .some((button) => button.attributes("aria-label") === "auth.hidePassword")
    ).toBe(true);
  });

  it("explains that a newer sign-in ended the current session", async () => {
    route.query = { reason: "session_revoked" };
    const wrapper = mount(LoginView);
    await flushPromises();

    expect(wrapper.find("#session-notice").text()).toContain("auth.sessionRevoked");
  });

  it("hides the passkey button when the browser cannot use WebAuthn", async () => {
    vi.stubGlobal("PublicKeyCredential", undefined);
    const wrapper = mount(LoginView);
    await flushPromises();

    const buttons = wrapper.findAll("button");
    expect(buttons.some((button) => button.text().includes("auth.passkeyLogin"))).toBe(false);
    expect(buttons.some((button) => button.text().includes("auth.loginButton"))).toBe(true);
  });

  it("does not follow an external post-login redirect", async () => {
    route.query = { redirect: "https://example.com/steal" };
    auth.login.mockResolvedValue({});
    const wrapper = mount(LoginView);
    await flushPromises();

    await wrapper.find("#username").setValue("operator");
    await wrapper.find("#password").setValue("password");
    await wrapper.find("form").trigger("submit.prevent");
    await flushPromises();

    expect(router.push).toHaveBeenCalledWith("/");
  });

  it("preserves a same-origin post-login redirect", async () => {
    route.query = { redirect: "/logs?q=error" };
    auth.login.mockResolvedValue({});
    const wrapper = mount(LoginView);
    await flushPromises();

    await wrapper.find("#username").setValue("operator");
    await wrapper.find("#password").setValue("password");
    await wrapper.find("form").trigger("submit.prevent");
    await flushPromises();

    expect(router.push).toHaveBeenCalledWith("/logs?q=error");
  });

  it("lets a user leave an MFA challenge and return to sign-in", async () => {
    auth.login.mockResolvedValue({ mfa_required: true, challenge_token: "challenge" });
    const wrapper = mount(LoginView);
    await flushPromises();

    await wrapper.find("#username").setValue("operator");
    await wrapper.find("#password").setValue("password");
    await wrapper.find("form").trigger("submit.prevent");
    await flushPromises();

    expect(wrapper.find("h1").text()).toBe("auth.mfaTitle");
    expect(wrapper.find("#mfa-code").exists()).toBe(true);

    const back = wrapper
      .findAll("button")
      .find((button) => button.text().includes("auth.backToLogin"));
    expect(back).toBeDefined();
    await back!.trigger("click");

    expect(wrapper.find("h1").text()).toBe("auth.loginTitle");
    expect(wrapper.find("#mfa-code").exists()).toBe(false);
    expect(wrapper.find("#username").exists()).toBe(true);
  });

  it("offers a reset link when the instance can send email", async () => {
    const wrapper = mount(LoginView);
    await flushPromises();

    expect(wrapper.find("#forgot-password-link").text()).toBe("auth.forgotPassword");
    expect(wrapper.find("#forgot-password-contact").exists()).toBe(false);
  });

  it("points an admin-less reset to an administrator when SMTP is off", async () => {
    // A reset that cannot be emailed would strand the user waiting for mail, so
    // the same entry becomes an instruction instead of a link.
    auth.getCapabilities.mockResolvedValue({ password_reset: false });
    const wrapper = mount(LoginView);
    await flushPromises();

    expect(wrapper.find("#forgot-password-link").exists()).toBe(false);
    expect(wrapper.find("#forgot-password-contact").text()).toBe("auth.forgotContactAdmin");
  });

  it("keeps the recovery entry off the MFA screen", async () => {
    auth.login.mockResolvedValue({ mfa_required: true, challenge_token: "challenge" });
    const wrapper = mount(LoginView);
    await flushPromises();

    await wrapper.find("#username").setValue("operator");
    await wrapper.find("#password").setValue("password");
    await wrapper.find("form").trigger("submit.prevent");
    await flushPromises();

    expect(wrapper.find("#forgot-password-link").exists()).toBe(false);
    expect(wrapper.find("#forgot-password-contact").exists()).toBe(false);
  });
});
