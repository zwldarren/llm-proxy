import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ConsoleAuthSettings } from "@/services/api/consoleAuth";

const api = vi.hoisted(() => ({ get: vi.fn(), update: vi.fn() }));
const toast = vi.hoisted(() => ({ success: vi.fn() }));

vi.mock("vue-i18n", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}));
vi.mock("vue-sonner", () => ({ toast }));
vi.mock("@/services/api/consoleAuth", () => ({ consoleAuthApi: api }));
vi.mock("@/components/layout/AppLayout.vue", () => ({
  default: { template: "<div><slot /></div>" },
}));
vi.mock("@/components/common/PageHeader.vue", () => ({
  default: { template: "<div />" },
}));
vi.mock("@/components/common/LoadingState.vue", () => ({
  default: { template: "<div />" },
}));
vi.mock("@lucide/vue", () => {
  const Stub = () => null;
  return { Loader2: Stub, Plus: Stub, ShieldCheck: Stub, Trash2: Stub };
});

import ConsoleAuthView from "./ConsoleAuthView.vue";

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
});
