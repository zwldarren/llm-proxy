import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";

const api = vi.hoisted(() => ({
  getPublic: vi.fn(),
  listProviders: vi.fn(),
  acceptInvitation: vi.fn(),
}));
const router = vi.hoisted(() => ({ push: vi.fn() }));

// Partial mock: utils/password.ts builds the app i18n instance at import time.
vi.mock("vue-i18n", async (importOriginal) => {
  const actual = await importOriginal<typeof import("vue-i18n")>();
  return { ...actual, useI18n: () => ({ t: (key: string) => key }) };
});
vi.mock("vue-router", () => ({
  useRoute: () => ({ params: { token: "invite-token" } }),
  useRouter: () => router,
}));
vi.mock("vue-sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));
vi.mock("@/services/api/invitations", () => ({
  invitationsApi: { getPublic: api.getPublic, accept: api.acceptInvitation },
}));
vi.mock("@/services/api/auth", () => ({
  authApi: { listOAuthProviders: api.listProviders },
}));
vi.mock("@/stores/auth", () => ({
  useAuthStore: () => ({ acceptInvitation: api.acceptInvitation }),
}));
vi.mock("@lucide/vue", () => {
  const Stub = () => null;
  return { Loader2: Stub };
});
vi.mock("@/components/common/AuthShell.vue", () => ({
  __esModule: true,
  default: { template: "<div><slot /></div>" },
}));

import { HttpError } from "@/services/http";
import InviteAcceptView from "./InviteAcceptView.vue";

describe("InviteAcceptView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.listProviders.mockResolvedValue({ providers: [] });
  });

  it("sends a freshly accepted invitee to the passkey prompt", async () => {
    api.getPublic.mockResolvedValue({
      email: "new@example.com",
      role: "viewer",
      expires_at: "2026-01-01T00:00:00Z",
    });
    api.acceptInvitation.mockResolvedValue({ access_token: "token" });

    const wrapper = mount(InviteAcceptView);
    await flushPromises();

    await wrapper.find("#invite-username").setValue("newuser");
    await wrapper.find("#invite-password").setValue("Passw0rd!x");
    await wrapper.find("#invite-confirm").setValue("Passw0rd!x");
    await wrapper.find("form").trigger("submit.prevent");
    await flushPromises();

    expect(api.acceptInvitation).toHaveBeenCalledWith("invite-token", "newuser", "Passw0rd!x");
    expect(router.push).toHaveBeenCalledWith("/account/security?from=invite");
  });

  it.each([400, 404, 410])(
    "explains a dead invitation (%s) instead of showing an empty form",
    async (status) => {
      api.getPublic.mockRejectedValue(new HttpError(status, "Gone", {}));

      const wrapper = mount(InviteAcceptView);
      await flushPromises();

      expect(wrapper.text()).toContain("auth.inviteExpired");
      expect(wrapper.text()).not.toContain("HTTP Error");
      expect(wrapper.find("form").exists()).toBe(false);
    }
  );
});
