import { enableAutoUnmount, flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ConsoleAuthSettings } from "@/services/api/consoleAuth";

const api = vi.hoisted(() => ({ get: vi.fn(), update: vi.fn(), testEmail: vi.fn() }));
const toast = vi.hoisted(() => ({ success: vi.fn() }));
const router = vi.hoisted(() => ({ beforeRouteLeave: vi.fn(), push: vi.fn() }));

vi.mock("vue-i18n", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}));
vi.mock("vue-sonner", () => ({ toast }));
vi.mock("vue-router", () => ({
  onBeforeRouteLeave: router.beforeRouteLeave,
  useRouter: () => ({ push: router.push }),
}));
vi.mock("@/services/api/consoleAuth", () => ({ consoleAuthApi: api }));
vi.mock("@/utils/error", () => ({
  getErrorMessage: (value: unknown) => (value instanceof Error ? value.message : String(value)),
}));
vi.mock("@/components/layout/AppLayout.vue", () => ({
  default: { template: '<div><slot name="header" /><slot /></div>' },
}));
vi.mock("@/components/common/PageHeader.vue", () => ({
  default: { template: "<div />" },
}));
vi.mock("@/components/common/LoadingState.vue", () => ({
  default: { template: "<div />" },
}));
vi.mock("@lucide/vue", () => {
  const Stub = () => null;
  return { ChevronDown: Stub, Loader2: Stub, Plus: Stub, ShieldCheck: Stub, Trash2: Stub };
});

import ConsoleAuthView from "./ConsoleAuthView.vue";
import ConfirmDialog from "@/components/common/ConfirmDialog.vue";

enableAutoUnmount(afterEach);

function makeSettings(): ConsoleAuthSettings {
  return {
    public_base_url: "https://proxy.example.com",
    require_2fa_for_admins: false,
    require_2fa_after_passkey: false,
    smtp: {
      enabled: false,
      host: "",
      port: 587,
      username: "",
      password: "",
      from_address: "",
      starttls: true,
      use_ssl: false,
    },
    resend: {
      enabled: false,
      api_key: "",
      from_address: "",
    },
    oidc_providers: [],
  };
}

describe("ConsoleAuthView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.get.mockImplementation(async () => makeSettings());
    api.update.mockImplementation(async (value: ConsoleAuthSettings) =>
      JSON.parse(JSON.stringify(value))
    );
  });

  it("tracks clean, dirty, and saved states around manual save", async () => {
    const wrapper = mount(ConsoleAuthView);
    await flushPromises();

    const save = wrapper.find('button[type="submit"]');
    expect(save.attributes("disabled")).toBeDefined();
    expect(wrapper.text()).toContain("consoleAuth.noPendingChanges");

    await wrapper.find("#public-url").setValue("https://proxy-next.example.com");
    expect(save.attributes("disabled")).toBeUndefined();
    expect(wrapper.text()).toContain("consoleAuth.unsavedChanges");

    await wrapper.find("form").trigger("submit.prevent");
    await flushPromises();

    expect(api.update).toHaveBeenCalledOnce();
    expect(api.update.mock.calls[0][0].public_base_url).toBe("https://proxy-next.example.com");
    expect(toast.success).toHaveBeenCalledWith("common.saved");
    expect(save.attributes("disabled")).toBeDefined();
    expect(wrapper.text()).toContain("consoleAuth.noPendingChanges");
  });

  it("keeps saved provider IDs immutable while allowing new IDs", async () => {
    const settings = makeSettings();
    settings.oidc_providers.push({
      id: "corp",
      name: "Corporate SSO",
      issuer: "https://id.example.com",
      client_id: "llm-proxy",
      client_secret: "",
      scopes: "openid email profile",
      enabled: true,
    });
    api.get.mockResolvedValue(settings);

    const wrapper = mount(ConsoleAuthView);
    await flushPromises();

    expect(wrapper.find("#provider-0-id").attributes("disabled")).toBeDefined();
    expect(wrapper.text()).toContain("consoleAuth.providerIdLockedHint");

    await wrapper
      .findAll("button")
      .find((button) => button.text().includes("consoleAuth.addProvider"))!
      .trigger("click");

    expect(wrapper.find("#provider-1-id").attributes("disabled")).toBeUndefined();
  });

  it("collapses saved provider details and expands new providers", async () => {
    const settings = makeSettings();
    settings.oidc_providers.push({
      id: "corp",
      name: "Corporate SSO",
      issuer: "https://id.example.com",
      client_id: "llm-proxy",
      client_secret: "",
      scopes: "openid email profile",
      enabled: true,
    });
    api.get.mockResolvedValue(settings);

    const wrapper = mount(ConsoleAuthView);
    await flushPromises();

    expect(wrapper.find("#provider-0-details").attributes("style")).toContain("display: none");
    await wrapper
      .findAll("button")
      .find((button) => button.text().includes("consoleAuth.showProviderDetails"))!
      .trigger("click");
    expect(wrapper.find("#provider-0-details").attributes("style") || "").not.toContain(
      "display: none"
    );

    await wrapper
      .findAll("button")
      .find((button) => button.text().includes("consoleAuth.addProvider"))!
      .trigger("click");
    expect(wrapper.find("#provider-1-details").attributes("style") || "").not.toContain(
      "display: none"
    );
  });

  it("derives each provider's callback URL from the public base URL", async () => {
    const settings = makeSettings();
    settings.oidc_providers.push({
      id: "corp",
      name: "Corporate SSO",
      issuer: "https://id.example.com",
      client_id: "llm-proxy",
      client_secret: "",
      scopes: "openid email profile",
      enabled: true,
    });
    api.get.mockResolvedValue(settings);

    const wrapper = mount(ConsoleAuthView);
    await flushPromises();

    expect(wrapper.text()).toContain("https://proxy.example.com/api/auth/oauth/corp/callback");

    // The URL follows the field, so the admin sees what they will register
    // before saving a base URL they cannot yet see the effect of.
    await wrapper.find("#public-url").setValue("https://proxy-next.example.com/");
    expect(wrapper.text()).toContain("https://proxy-next.example.com/api/auth/oauth/corp/callback");
  });

  it("discards edits back to the last saved snapshot", async () => {
    const wrapper = mount(ConsoleAuthView);
    await flushPromises();

    await wrapper.find("#public-url").setValue("https://proxy-next.example.com");
    const discard = wrapper
      .findAll("button")
      .find((button) => button.text().includes("consoleAuth.discardChanges"))!;
    await discard.trigger("click");

    expect(wrapper.find<HTMLInputElement>("#public-url").element.value).toBe(
      "https://proxy.example.com"
    );
    expect(wrapper.text()).toContain("consoleAuth.noPendingChanges");
    expect(wrapper.find('button[type="submit"]').attributes("disabled")).toBeDefined();
  });

  it("blocks route changes, asks before leaving, and blocks page unload only while dirty", async () => {
    const wrapper = mount(ConsoleAuthView);
    await flushPromises();
    const guard = router.beforeRouteLeave.mock.calls[0][0] as (to: { fullPath: string }) => boolean;
    expect(guard({ fullPath: "/config/settings" })).toBe(true);

    await wrapper.find("#public-url").setValue("https://proxy-next.example.com");

    // A dirty form cancels the navigation outright and raises the console's own
    // dialog — the native confirm() ignored the app's language and styling.
    expect(guard({ fullPath: "/config/settings" })).toBe(false);
    expect(router.push).not.toHaveBeenCalled();

    wrapper.findComponent(ConfirmDialog).vm.$emit("confirm");
    await flushPromises();
    expect(router.push).toHaveBeenCalledWith("/config/settings");

    const unload = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(unload);
    expect(unload.defaultPrevented).toBe(false);
  });

  it("asks before dropping an OIDC provider, secret included", async () => {
    const settings = makeSettings();
    settings.oidc_providers.push({
      id: "corp",
      name: "Corporate SSO",
      issuer: "https://id.example.com",
      client_id: "llm-proxy",
      client_secret: "typed-but-not-saved",
      scopes: "openid email profile",
      enabled: true,
    });
    api.get.mockResolvedValue(settings);

    const wrapper = mount(ConsoleAuthView);
    await flushPromises();
    expect(wrapper.find("#provider-0-id").exists()).toBe(true);

    await wrapper.find('[aria-label="consoleAuth.removeProvider"]').trigger("click");

    // Nothing is spliced out on the first click: the removal carries an
    // unsaved client secret with it.
    expect(wrapper.find("#provider-0-id").exists()).toBe(true);

    const removalDialog = wrapper.findAllComponents(ConfirmDialog).at(1)!;
    removalDialog.vm.$emit("confirm");
    await flushPromises();

    expect(wrapper.find("#provider-0-id").exists()).toBe(false);
  });

  it("shows inline SMTP errors and focuses the first invalid field", async () => {
    const wrapper = mount(ConsoleAuthView, { attachTo: document.body });
    await flushPromises();

    expect(wrapper.find("#smtp-host").exists()).toBe(false);
    await wrapper.find('[aria-label="consoleAuth.smtpEnabled"]').trigger("click");
    expect(wrapper.find("#smtp-host").exists()).toBe(true);
    await wrapper.find("form").trigger("submit.prevent");
    await flushPromises();

    expect(api.update).not.toHaveBeenCalled();
    expect(wrapper.text()).toContain("consoleAuth.validationSummary");
    expect(wrapper.find("#smtp-host").attributes("aria-invalid")).toBe("true");
    expect(wrapper.find("#smtp-from").attributes("aria-invalid")).toBe("true");
    expect(document.activeElement?.id).toBe("smtp-host");

    await wrapper.find("#smtp-host").setValue("smtp.example.com");
    await wrapper.find("#smtp-from").setValue("proxy@example.com");
    await wrapper.find("#smtp-port").setValue("2525");
    await wrapper.find("form").trigger("submit.prevent");
    await flushPromises();
    expect(api.update).toHaveBeenCalledOnce();
    // The port is a number in the wire contract, whatever the input's text is.
    expect(api.update.mock.calls[0][0].smtp.port).toBe(2525);
  });

  it("validates the Resend block before saving", async () => {
    const wrapper = mount(ConsoleAuthView, { attachTo: document.body });
    await flushPromises();

    expect(wrapper.find("#resend-api-key").exists()).toBe(false);
    await wrapper.find('[aria-label="consoleAuth.resendEnabled"]').trigger("click");
    expect(wrapper.find("#resend-api-key").exists()).toBe(true);
    await wrapper.find("form").trigger("submit.prevent");
    await flushPromises();

    expect(api.update).not.toHaveBeenCalled();
    expect(wrapper.find("#resend-api-key").attributes("aria-invalid")).toBe("true");
    expect(wrapper.find("#resend-from").attributes("aria-invalid")).toBe("true");

    await wrapper.find("#resend-api-key").setValue("re_test_key");
    await wrapper.find("#resend-from").setValue("noreply@example.com");
    await wrapper.find("form").trigger("submit.prevent");
    await flushPromises();

    expect(api.update).toHaveBeenCalledOnce();
    expect(api.update.mock.calls[0][0].resend).toEqual({
      enabled: true,
      api_key: "re_test_key",
      from_address: "noreply@example.com",
    });
  });

  it("sends a test email through the saved transport", async () => {
    const settings = makeSettings();
    settings.resend = {
      enabled: true,
      api_key: "****",
      from_address: "noreply@example.com",
    };
    api.get.mockResolvedValue(settings);
    api.testEmail.mockResolvedValue({ sent: true, transport: "resend" });

    const wrapper = mount(ConsoleAuthView);
    await flushPromises();

    const sendButton = () =>
      wrapper
        .findAll("button")
        .find((button) => button.text().includes("consoleAuth.testEmailSend"))!;
    expect(sendButton().attributes("disabled")).toBeUndefined();

    // The endpoint reads the saved config, so unsaved edits must not look tested.
    await wrapper.find("#public-url").setValue("https://proxy-next.example.com");
    expect(sendButton().attributes("disabled")).toBeDefined();
    await wrapper.find("#public-url").setValue("https://proxy.example.com");

    await wrapper.find("#test-email-to").setValue("ops@example.com");
    await sendButton().trigger("click");
    await flushPromises();

    expect(api.testEmail).toHaveBeenCalledWith("ops@example.com");
    expect(toast.success).toHaveBeenCalledWith("consoleAuth.testEmailSent");
  });

  it("rejects an insecure public URL before saving", async () => {
    const wrapper = mount(ConsoleAuthView);
    await flushPromises();

    await wrapper.find("#public-url").setValue("http://proxy.example.com");
    await wrapper.find("form").trigger("submit.prevent");
    await flushPromises();

    expect(api.update).not.toHaveBeenCalled();
    expect(wrapper.find("#public-url-error").text()).toBe("consoleAuth.publicUrlInvalid");
    expect(wrapper.find("#public-url").attributes("aria-describedby")).toBe("public-url-error");
  });

  it("preserves edits and focuses the error summary when saving fails", async () => {
    api.update.mockRejectedValue(new Error("Save failed"));
    const wrapper = mount(ConsoleAuthView, { attachTo: document.body });
    await flushPromises();

    await wrapper.find("#public-url").setValue("https://proxy-next.example.com");
    await wrapper.find("form").trigger("submit.prevent");
    await flushPromises();

    expect(wrapper.find<HTMLInputElement>("#public-url").element.value).toBe(
      "https://proxy-next.example.com"
    );
    expect(wrapper.text()).toContain("Save failed");
    expect(document.activeElement?.id).toBe("console-auth-error");
  });
});
