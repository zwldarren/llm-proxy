import { enableAutoUnmount, flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ConsoleAuthSettings } from "@/services/api/consoleAuth";

const api = vi.hoisted(() => ({ get: vi.fn(), update: vi.fn() }));
const toast = vi.hoisted(() => ({ success: vi.fn() }));
const router = vi.hoisted(() => ({ beforeRouteLeave: vi.fn() }));

vi.mock("vue-i18n", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}));
vi.mock("vue-sonner", () => ({ toast }));
vi.mock("vue-router", () => ({ onBeforeRouteLeave: router.beforeRouteLeave }));
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

enableAutoUnmount(afterEach);

function makeSettings(): ConsoleAuthSettings {
  return {
    public_base_url: "https://proxy.example.com",
    require_2fa_for_admins: false,
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

  it("blocks route changes and page unload only while dirty", async () => {
    const wrapper = mount(ConsoleAuthView);
    await flushPromises();
    const guard = router.beforeRouteLeave.mock.calls[0][0] as () => boolean;
    expect(guard()).toBe(true);

    await wrapper.find("#public-url").setValue("https://proxy-next.example.com");
    const originalConfirm = window.confirm;
    window.confirm = vi.fn().mockReturnValue(false);
    expect(guard()).toBe(false);
    const unload = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(unload);
    expect(unload.defaultPrevented).toBe(true);
    window.confirm = originalConfirm;
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
    await wrapper.find("form").trigger("submit.prevent");
    await flushPromises();
    expect(api.update).toHaveBeenCalledOnce();
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
