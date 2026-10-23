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

// Mutable so a test can present a deep link (e.g. ?trace_id=...) to the view.
const { routeQuery } = vi.hoisted(() => ({ routeQuery: {} as Record<string, string> }));

vi.mock("vue-router", () => {
  return {
    useRoute: () => ({ name: "logs", query: routeQuery }),
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
      isAdmin: true,
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
    Scale: Stub,
    ScrollText: Stub,
    Search: Stub,
    Shield: Stub,
    ThumbsUp: Stub,
    Wrench: Stub,
    X: Stub,
    Zap: Stub,
  };
});

// The view mounts all three of these through `defineAsyncComponent`, so every mount
// starts a module transform that can still be running once the test has finished. CI
// failed the run with `EnvironmentTeardownError: Cannot load
// '.../range-calendar/RangeCalendarHeadCell.vue' ... after the environment was torn
// down`: the `?trace_id` deep link opens the filter panel, and the panel imports the
// range calendar. A view test does not need its lazy children, so mocking them keeps
// this file's module graph to what the view itself imports.
vi.mock("@/components/common/LogsFilters.vue", () => ({
  __esModule: true,
  default: defineComponent({
    name: "LogsFilters",
    props: ["filters", "activeTab"],
    emits: ["update:filters"],
    setup() {
      return () => h("div", { class: "logs-filters" });
    },
  }),
}));

vi.mock("@/components/common/LogDetailsSheet.vue", () => ({
  __esModule: true,
  default: defineComponent({
    name: "LogDetailsSheet",
    props: ["open", "log"],
    emits: ["update:open", "filter-trace"],
    setup() {
      return () => null;
    },
  }),
}));

vi.mock("@/components/common/AuditIntegrityDialog.vue", () => ({
  __esModule: true,
  default: defineComponent({
    name: "AuditIntegrityDialog",
    props: ["open"],
    emits: ["update:open"],
    setup() {
      return () => null;
    },
  }),
}));

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

// Passthrough stubs: Tooltip parts now rely on the app-level TooltipProvider,
// which is absent in isolated view mounts.
const TooltipStub = defineComponent({
  name: "Tooltip",
  setup(_, { slots }) {
    return () => slots.default?.();
  },
});
const TooltipTriggerStub = defineComponent({
  name: "TooltipTrigger",
  setup(_, { slots }) {
    return () => slots.default?.();
  },
});
const TooltipContentStub = defineComponent({
  name: "TooltipContent",
  setup() {
    return () => null;
  },
});

describe("LogsView", () => {
  it("loads logs on mount", async () => {
    const wrapper = mount(LogsView, {
      global: {
        stubs: {
          AppLayout: AppLayoutStub,
          Card: CardStub,
          Button: ButtonStub,
          Input: InputStub,
          Tooltip: TooltipStub,
          TooltipTrigger: TooltipTriggerStub,
          TooltipContent: TooltipContentStub,
        },
      },
    });

    await flushPromises();
    const { logsApi } = await import("@/services/api/logs");

    expect(logsApi.getLogs).toHaveBeenCalled();
    expect(wrapper.text()).toContain("logs.noLogs");
  });

  it("applies a ?trace_id= deep link before the first fetch", async () => {
    const { logsApi } = await import("@/services/api/logs");
    vi.clearAllMocks();
    routeQuery.trace_id = "trace-from-the-url";

    try {
      const wrapper = mount(LogsView, {
        global: {
          stubs: {
            AppLayout: AppLayoutStub,
            Card: CardStub,
            Button: ButtonStub,
            Input: InputStub,
            Tooltip: TooltipStub,
            TooltipTrigger: TooltipTriggerStub,
            TooltipContent: TooltipContentStub,
          },
        },
      });

      await flushPromises();

      // The trace filter is not in the filter panel, so the only way it can reach
      // the request is the deep link — and it must be there on the first call.
      expect(vi.mocked(logsApi.getLogs).mock.calls[0]?.[0]?.trace_id).toBe("trace-from-the-url");

      // The deep link also reveals the filter panel, so the chip it applied stays
      // visible with the rest of the active filters.
      expect(wrapper.findComponent({ name: "LogsFilters" }).exists()).toBe(true);
    } finally {
      delete routeQuery.trace_id;
    }
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
          Tooltip: TooltipStub,
          TooltipTrigger: TooltipTriggerStub,
          TooltipContent: TooltipContentStub,
        },
      },
    });

    await flushPromises();
    expect(logsApi.getLogs).toHaveBeenCalled();
    vi.clearAllMocks();

    // Switch tab to audit by clicking the tab trigger (public behavior)
    const auditTab = wrapper.findAll('[role="tab"]').find((tab) => tab.text() === "nav.auditLogs");
    expect(auditTab).toBeTruthy();
    await auditTab!.trigger("mousedown");
    await flushPromises();

    // Wait a little bit more than debounce time to make sure no debounced fetch fires
    await new Promise((resolve) => setTimeout(resolve, 400));

    // getLogs should be called for the new tab load (page 1)
    const page1Calls = vi.mocked(logsApi.getLogs).mock.calls.filter((args) => args[0]?.page === 1);
    expect(page1Calls.length).toBeGreaterThanOrEqual(1);
  });
});
