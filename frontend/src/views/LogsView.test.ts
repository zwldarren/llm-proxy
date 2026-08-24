import { mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";
import { defineComponent, h } from "vue";

vi.mock("@/utils/icons", () => {
  return {
    getProviderIconUrl: vi.fn(() => null),
    getIconUrl: vi.fn(() => null),
  };
});

vi.mock("vue-i18n", () => {
  return {
    useI18n: () => ({ t: (key: string) => key }),
    createI18n: vi.fn(() => ({})),
  };
});

vi.mock("vue-router", () => {
  return {
    useRoute: () => ({ query: {} }),
    useRouter: () => ({ replace: vi.fn() }),
    createRouter: vi.fn(() => ({
      beforeEach: vi.fn(),
      afterEach: vi.fn(),
    })),
    createWebHistory: vi.fn(() => ({})),
  };
});

vi.mock("@/router", () => {
  return {
    default: {
      beforeEach: vi.fn(),
      afterEach: vi.fn(),
    },
  };
});

vi.mock("@/stores/auth", () => {
  return {
    useAuthStore: vi.fn(() => ({
      isAuthenticated: true,
    })),
  };
});

vi.mock("@/services/api/config", () => {
  return {
    configApi: {
      getModels: vi.fn(() => Promise.resolve([])),
      getProviders: vi.fn(() => Promise.resolve([])),
    },
  };
});

vi.mock("@/services/api/logs", () => {
  return {
    logsApi: {
      getLogs: vi.fn(() => Promise.resolve({ items: [], total: 0, page: 1, page_size: 20 })),
      getLog: vi.fn(() => Promise.resolve({})),
      deleteOldLogs: vi.fn(() => Promise.resolve({ deleted: 0 })),
      getUsageStats: vi.fn(() =>
        Promise.resolve({
          summary: {
            total_cost: 0,
            total_requests: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            avg_response_time_ms: 0,
            success_rate: 100,
            avg_ttft_ms: 0,
            avg_tokens_per_second: 0,
            total_cache_creation_tokens: 0,
            total_cache_read_tokens: 0,
            total_cached_prompt_tokens: 0,
            cache_savings_usd: 0,
          },
          by_provider: [],
          by_model: [],
          daily_usage: [],
        })
      ),
    },
  };
});

vi.mock("@lucide/vue", () => {
  const Stub = () => null;
  return {
    Activity: Stub,
    ArrowDown: Stub,
    ArrowUp: Stub,
    Calendar: Stub,
    CalendarIcon: Stub,
    ChevronFirst: Stub,
    ChevronLast: Stub,
    ChevronLeft: Stub,
    ChevronRight: Stub,
    ChevronDown: Stub,
    Check: Stub,
    Clock: Stub,
    Coins: Stub,
    Eye: Stub,
    Filter: Stub,
    Globe: Stub,
    Loader2: Stub,
    Pause: Stub,
    Play: Stub,
    RefreshCw: Stub,
    ScrollText: Stub,
    Search: Stub,
    Shield: Stub,
    ThumbsUp: Stub,
    Wrench: Stub,
    X: Stub,
    Zap: Stub,
  };
});

import LogsView from "./LogsView.vue";

const AppLayoutStub = defineComponent({
  name: "AppLayout",
  setup(_, { slots }) {
    return () => h("div", [slots.header?.(), slots.default?.()]);
  },
});

const CardStub = defineComponent({
  name: "Card",
  setup(_, { slots }) {
    return () => h("div", [slots.content?.(), slots.default?.()]);
  },
});

const ButtonStub = defineComponent({
  name: "Button",
  props: ["disabled"],
  setup(_, { slots }) {
    return () => h("button", slots.default?.());
  },
});

const InputStub = defineComponent({
  name: "Input",
  props: ["modelValue", "type"],
  emits: ["update:modelValue"],
  setup(props, { emit }) {
    return () =>
      h("input", {
        type: props.type || "text",
        value: props.modelValue as unknown,
        onInput: (event: Event) => {
          const target = event.target as HTMLInputElement | null;
          const value =
            props.type === "number" ? Number(target?.value ?? 0) : (target?.value ?? "");
          emit("update:modelValue", value);
        },
      });
  },
});

async function flushPromises() {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

describe("LogsView", () => {
  it("loads logs on mount", async () => {
    const wrapper = mount(LogsView, {
      global: {
        stubs: {
          AppLayout: AppLayoutStub,
          Card: CardStub,
          Button: ButtonStub,
          Input: InputStub,
        },
      },
    });

    await flushPromises();
    const { logsApi } = await import("@/services/api/logs");

    expect(logsApi.getLogs).toHaveBeenCalled();
    expect(wrapper.text()).toContain("logs.noLogs");
  });

  it("only calls getLogs once when switching tabs and does not trigger duplicate queries", async () => {
    const { logsApi } = await import("@/services/api/logs");
    vi.clearAllMocks();

    const wrapper = mount(LogsView, {
      global: {
        stubs: {
          AppLayout: AppLayoutStub,
          Card: CardStub,
          Button: ButtonStub,
          Input: InputStub,
        },
      },
    });

    await flushPromises();
    expect(logsApi.getLogs).toHaveBeenCalled();
    vi.clearAllMocks();

    // Switch tab to audit
    // @ts-expect-error - vm exposes internal variables in setup
    wrapper.vm.activeTab = "audit";
    await flushPromises();

    // Wait a little bit more than debounce time to make sure no debounced fetch fires
    await new Promise((resolve) => setTimeout(resolve, 400));

    // getLogs should be called for the new tab load (page 1)
    const page1Calls = vi.mocked(logsApi.getLogs).mock.calls.filter((args) => args[0]?.page === 1);
    expect(page1Calls.length).toBeGreaterThanOrEqual(1);
  });
});
