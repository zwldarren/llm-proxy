// frontend/src/views/ChatView.test.ts
import { mount } from "@vue/test-utils";
import { createPinia } from "pinia";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { defineComponent, h, nextTick, ref } from "vue";

vi.mock("@/utils/icons", () => ({
  getModelIconUrl: vi.fn(() => null),
  isMonoIcon: vi.fn(() => false),
}));

vi.mock("vue-i18n", () => ({
  useI18n: () => ({ t: (key: string) => key }),
  createI18n: vi.fn(() => ({})),
}));

vi.mock("vue-router", () => ({
  useRouter: () => ({ push: vi.fn() }),
  useRoute: () => ({ query: {} }),
}));

vi.mock("@/router", () => ({
  default: { beforeEach: vi.fn(), afterEach: vi.fn(), currentRoute: { value: {} } },
}));

// The Chat page reads the session API key off the auth store; the tests drive
// its presence to check the "no key" path without a real login.
const auth = vi.hoisted(() => ({
  sessionApiKey: "session-key" as string | null,
  isAdmin: false,
}));
vi.mock("@/stores/auth", () => ({
  useAuthStore: () => auth,
}));

vi.mock("@/services/api/chat", () => ({
  chatApi: {
    getModels: vi.fn(),
    streamChatCompletion: vi.fn(),
  },
}));

vi.mock("@/services/api/config", () => ({
  configApi: {
    getModels: vi.fn(() => Promise.resolve([])),
    getProviders: vi.fn(() => Promise.resolve([])),
  },
  webSearchApi: { getConfig: vi.fn(() => Promise.resolve({ enabled: true })) },
}));

import { chatApi } from "@/services/api/chat";
import { TooltipProvider } from "@/components/ui/tooltip";
import ChatView from "./ChatView.vue";

const AppLayoutStub = defineComponent({
  name: "AppLayout",
  setup(_, { slots }) {
    return () => h("div", [slots.header?.(), slots.default?.()]);
  },
});

/** Leaf chat surfaces are irrelevant here; stub them so only the header runs. */
const global = {
  plugins: [createPinia()],
  stubs: {
    AppLayout: AppLayoutStub,
    ChatSettings: true,
    ChatMessage: true,
    ChatEmptyState: true,
    ConfirmDialog: true,
  },
};

async function flushPromises() {
  await new Promise((r) => setTimeout(r, 0));
}

/**
 * ChatView's header uses tooltips, which need the provider App.vue supplies.
 * `inner` lets a test swap the ChatView for a KeepAlive-wrapped copy.
 */
function render(inner = "<ChatView />") {
  const Host = defineComponent({
    components: { ChatView, TooltipProvider },
    setup: () => ({ show: ref(true) }),
    template: `<TooltipProvider>${inner}</TooltipProvider>`,
  });
  return mount(Host, { global });
}

/** The header placeholder doubles as the retry control (aria-label tells them apart). */
const retryButton = (wrapper: ReturnType<typeof render>) =>
  wrapper.find<HTMLButtonElement>("button[aria-label='chat.retryModels']");

const models = (...ids: string[]) => ({
  data: ids.map((id) => ({ id, provider: "openai" })),
});

beforeEach(() => {
  auth.sessionApiKey = "session-key";
  localStorage.clear();
  vi.mocked(chatApi.getModels).mockReset();
});

describe("ChatView model list", () => {
  it("loads models on mount and selects the first one", async () => {
    vi.mocked(chatApi.getModels).mockResolvedValue(models("gpt-4o", "claude-sonnet-4"));

    const wrapper = render();
    await flushPromises();

    expect(chatApi.getModels).toHaveBeenCalledTimes(1);
    // The selector (not the retry placeholder) is what renders once loaded.
    expect(wrapper.find("[data-slot='select-trigger']").exists()).toBe(true);
    expect(retryButton(wrapper).exists()).toBe(false);
  });

  it("keeps the model selector retryable after a failed load", async () => {
    vi.mocked(chatApi.getModels).mockRejectedValueOnce(new Error("boom"));

    const wrapper = render();
    await flushPromises();

    const button = retryButton(wrapper);
    expect(button.exists()).toBe(true);
    expect(button.element.disabled).toBe(false);

    // A click retries in place — the page must recover without a full reload.
    vi.mocked(chatApi.getModels).mockResolvedValueOnce(models("gpt-4o"));
    await button.trigger("click");
    await flushPromises();

    expect(chatApi.getModels).toHaveBeenCalledTimes(2);
    expect(wrapper.find("[data-slot='select-trigger']").exists()).toBe(true);
  });

  it("reloads models when the kept-alive page is entered again", async () => {
    vi.mocked(chatApi.getModels).mockRejectedValueOnce(new Error("backend restarting"));

    const wrapper = render('<KeepAlive><ChatView v-if="show" /></KeepAlive>');
    await flushPromises();
    expect(chatApi.getModels).toHaveBeenCalledTimes(1);
    expect(retryButton(wrapper).exists()).toBe(true);

    vi.mocked(chatApi.getModels).mockResolvedValueOnce(models("gpt-4o"));

    wrapper.vm.show = false;
    await nextTick();
    wrapper.vm.show = true;
    await nextTick();
    await flushPromises();

    // KeepAlive never re-runs onMounted: activation is the only retry point.
    expect(chatApi.getModels).toHaveBeenCalledTimes(2);
    expect(wrapper.find("[data-slot='select-trigger']").exists()).toBe(true);
  });

  it("never calls /v1/models without a session API key", async () => {
    auth.sessionApiKey = null;

    const wrapper = render();
    await flushPromises();

    // An empty Bearer is rejected as an invalid key and locks the IP out, so
    // the request must not be sent at all.
    expect(chatApi.getModels).not.toHaveBeenCalled();
    expect(retryButton(wrapper).exists()).toBe(true);
  });
});
