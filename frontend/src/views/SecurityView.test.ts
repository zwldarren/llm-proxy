import { flushPromises, mount } from "@vue/test-utils";
import { enableAutoUnmount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { OAuthIdentity, OAuthProvider } from "@/types/schemas";

const api = vi.hoisted(() => ({
  listPasskeys: vi.fn(),
  listOAuthProviders: vi.fn(),
  getTwoFactor: vi.fn(),
  listIdentities: vi.fn(),
  getProfile: vi.fn(),
}));
const router = vi.hoisted(() => ({ push: vi.fn(), replace: vi.fn() }));
const route = vi.hoisted(() => ({ query: {} as Record<string, string> }));

/**
 * Keys and their interpolation values, so a test can prove the dialog names a
 * provider the operator recognises rather than its stored id.
 */
vi.mock("vue-i18n", async (importOriginal) => {
  const actual = await importOriginal<typeof import("vue-i18n")>();
  return {
    ...actual,
    useI18n: () => ({
      t: (key: string, params?: Record<string, unknown>) =>
        params ? `${key}|${Object.values(params).join("|")}` : key,
      locale: { value: "en" },
    }),
  };
});
vi.mock("vue-router", () => ({
  useRoute: () => route,
  useRouter: () => router,
  RouterLink: { props: ["to"], template: "<a><slot /></a>" },
}));
vi.mock("@/stores/auth", () => ({
  useAuthStore: () => ({
    username: "ops",
    isAdmin: false,
    clearLocalSession: vi.fn(),
    logout: vi.fn(),
  }),
}));
vi.mock("vue-sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn() },
}));
vi.mock("@/services/api/auth", () => ({
  authApi: {
    listPasskeys: api.listPasskeys,
    listOAuthProviders: api.listOAuthProviders,
  },
}));
vi.mock("@/services/api/me", () => ({ meApi: { getProfile: api.getProfile } }));
vi.mock("@/services/api/security", () => ({
  securityApi: {
    getTwoFactor: api.getTwoFactor,
    listIdentities: api.listIdentities,
    unlinkIdentity: vi.fn(),
  },
}));
vi.mock("@lucide/vue", () => {
  const Stub = () => null;
  return {
    Check: Stub,
    Copy: Stub,
    Download: Stub,
    Eye: Stub,
    EyeOff: Stub,
    Fingerprint: Stub,
    KeyRound: Stub,
    Link2: Stub,
    Loader2: Stub,
    LockKeyhole: Stub,
    Pencil: Stub,
    ShieldCheck: Stub,
    Trash2: Stub,
    Unlink: Stub,
    X: Stub,
  };
});

import ConfirmDialog from "@/components/common/ConfirmDialog.vue";
import SecurityView from "./SecurityView.vue";

enableAutoUnmount(afterEach);

const CORP: OAuthProvider = { id: "corp", name: "Corporate SSO" };
const GITHUB: OAuthProvider = { id: "github", name: "GitHub" };

function identity(overrides: Partial<OAuthIdentity> = {}): OAuthIdentity {
  return {
    id: 1,
    provider: "corp",
    email: "ops@example.com",
    created_at: "2026-01-01T00:00:00Z",
    last_login_at: null,
    ...overrides,
  };
}

function mountView(providers: OAuthProvider[], identities: OAuthIdentity[]) {
  api.listOAuthProviders.mockResolvedValue({ providers });
  api.listIdentities.mockResolvedValue({ identities });
  return mount(SecurityView, {
    attachTo: document.body,
    global: {
      stubs: {
        AccountShell: { template: "<div><slot /></div>" },
        QrCode: { template: "<div />" },
      },
    },
  });
}

/** Link buttons only: they are the ones labelled with a provider name. */
function linkButtons(wrapper: ReturnType<typeof mount>) {
  return wrapper
    .findAll("button")
    .filter((button) => button.text().startsWith("accountSecurity.link"));
}

describe("SecurityView linked providers", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.listPasskeys.mockResolvedValue([]);
    api.getTwoFactor.mockResolvedValue({
      enabled: false,
      recovery_codes_remaining: 0,
      required_for_admins: false,
    });
    api.getProfile.mockResolvedValue({
      id: 1,
      username: "ops",
      email: null,
      email_verified: false,
      has_password: true,
      role: "member",
      must_change_password: false,
      created_at: "2026-01-01T00:00:00Z",
      allowed_models: null,
      last_login_at: null,
      session: null,
    });
  });

  it("names a linked provider by its display name and does not offer to link it again", async () => {
    const wrapper = mountView([CORP, GITHUB], [identity()]);
    await flushPromises();

    expect(wrapper.text()).toContain("Corporate SSO · ops@example.com");

    // "Link GitHub" is offered; "Link Corporate SSO" would only round-trip
    // through the identity provider to report the link it already has.
    const offered = linkButtons(wrapper).map((button) => button.text());
    expect(offered).toEqual(["accountSecurity.link GitHub"]);
  });

  it("offers no link button, and no empty-state copy, once every provider is linked", async () => {
    const wrapper = mountView([CORP], [identity()]);
    await flushPromises();

    expect(linkButtons(wrapper)).toHaveLength(0);
    expect(wrapper.text()).not.toContain("accountSecurity.noProvidersAvailable");
  });

  it("marks an identity whose provider was removed instead of leaving a bare id", async () => {
    const wrapper = mountView([], [identity({ provider: "legacy-sso" })]);
    await flushPromises();

    // The stored id is all that is left to name it by, so say why.
    expect(wrapper.text()).toContain("legacy-sso · ops@example.com");
    expect(wrapper.text()).toContain("accountSecurity.identityProviderUnavailable");
    expect(wrapper.text()).toContain("accountSecurity.noProvidersAvailable");
  });

  it("names the provider in the unlink confirmation too", async () => {
    const wrapper = mountView([CORP], [identity()]);
    await flushPromises();

    const unlink = wrapper
      .findAll("button")
      .find((button) => button.text().includes("accountSecurity.unlink"))!;
    await unlink.trigger("click");

    const dialog = wrapper
      .findAllComponents(ConfirmDialog)
      .find((candidate) => candidate.props("title") === "accountSecurity.unlinkTitle")!;
    expect(dialog.props("description")).toBe("accountSecurity.unlinkDescription|Corporate SSO");
  });
});
