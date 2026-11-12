// frontend/src/views/HomeView.test.ts
import { mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";
import { defineComponent, h } from "vue";

vi.mock("@/utils/icons", () => ({
  getModelIconUrl: vi.fn(() => null),
  isMonoIcon: vi.fn(() => false),
  getProviderIconUrl: vi.fn(() => null),
  getIconUrl: vi.fn(() => null),
}));

vi.mock("vue-i18n", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}));

vi.mock("vue-router", () => ({
  useRouter: () => ({ push: vi.fn() }),
  useRoute: () => ({ query: {} }),
}));

vi.mock("@/services/api/logs", () => ({
  logsApi: {
    getHourlyUsage: vi.fn(() => Promise.resolve({ buckets: [] })),
    getUsageStats: vi.fn(() =>
      Promise.resolve({
        summary: {
          total_cost: 3.81,
          total_requests: 1234,
          total_input_tokens: 100000,
          total_output_tokens: 20000,
          avg_response_time_ms: 1500,
          success_rate: 100,
          avg_ttft_ms: 480,
          avg_tokens_per_second: 41,
          total_cache_creation_tokens: 0,
          total_cache_read_tokens: 5000,
          total_cached_prompt_tokens: 0,
          cache_savings_usd: 0.42,
        },
        by_provider: [],
        by_model: [],
        daily_usage: [],
      })
    ),
  },
}));

// Stub the heavy chart + async-loaded usage children so we test HomeView's
// data flow, not chart.js rendering. Async components need global.stubs by name.
vi.mock("@/components/usage", () => ({
  UsageByModel: defineComponent({
    name: "UsageByModel",
    props: ["byModel"],
    template: "<div data-test='by-model'>home.byModel</div>",
  }),
  UsageByProvider: defineComponent({
    name: "UsageByProvider",
    props: ["byProvider"],
    template: "<div data-test='by-provider'>home.byProvider</div>",
  }),
  UsageTrendsChart: defineComponent({
    name: "UsageTrendsChart",
    props: ["dailyUsage"],
    template: "<div data-test='trends'>home.usageTrends</div>",
  }),
  UsageMetricStrip: defineComponent({
    name: "UsageMetricStrip",
    props: ["summary", "byProvider"],
    template: "<div data-test='strip'/>",
  }),
  TokenBreakdownChart: defineComponent({
    name: "TokenBreakdownChart",
    props: ["buckets", "granularity"],
    template: "<div data-test='token-breakdown'/>",
  }),
}));

vi.mock("@lucide/vue", () => {
  const Stub = () => null;
  return {
    BarChart3: Stub,
    CalendarIcon: Stub,
    Check: Stub,
    ChevronDown: Stub,
    ChevronUp: Stub,
    Cpu: Stub,
    Layers: Stub,
    Loader2: Stub,
    Settings: Stub,
  };
});

import { logsApi } from "@/services/api/logs";
import type { UsageStatsResponse } from "@/types/schemas";
import HomeView from "./HomeView.vue";

const AppLayoutStub = defineComponent({
  name: "AppLayout",
  setup(_, { slots }) {
    return () => h("div", [slots.header?.(), slots.default?.()]);
  },
});

async function flushPromises() {
  await new Promise((r) => setTimeout(r, 0));
}

describe("HomeView", () => {
  it("fetches usage stats on mount with a default date range", async () => {
    mount(HomeView, { global: { stubs: { AppLayout: AppLayoutStub } } });
    await flushPromises();
    const { logsApi } = await import("@/services/api/logs");
    expect(logsApi.getUsageStats).toHaveBeenCalledTimes(1);
    const args = vi.mocked(logsApi.getUsageStats).mock.calls[0]![0] as {
      start_date: string;
      end_date: string;
    };
    expect(args.start_date).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(args.end_date).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  });

  it("renders the metric strip + trends + breakdowns after data loads", async () => {
    const wrapper = mount(HomeView, { global: { stubs: { AppLayout: AppLayoutStub } } });
    await flushPromises();
    expect(wrapper.find("[data-test='strip']").exists()).toBe(true);
    expect(wrapper.find("[data-test='trends']").exists()).toBe(true);
    expect(wrapper.find("[data-test='by-provider']").exists()).toBe(true);
    expect(wrapper.find("[data-test='by-model']").exists()).toBe(true);
  });

  it("does not let an older range response repaint the newer range", async () => {
    vi.clearAllMocks();

    const older = deferred<UsageStatsResponse>();
    const newer = deferred<UsageStatsResponse>();
    vi.mocked(logsApi.getUsageStats)
      .mockResolvedValueOnce(makeStats("initial-model"))
      .mockImplementationOnce(() => older.promise)
      .mockImplementationOnce(() => newer.promise);

    const wrapper = mount(HomeView, { global: { stubs: { AppLayout: AppLayoutStub } } });
    await flushPromises();

    type ViewMethods = {
      selectPreset: (preset: "7d" | "30d") => void;
    };
    const view = wrapper.vm as unknown as ViewMethods;
    view.selectPreset("7d");
    view.selectPreset("30d");

    newer.resolve(makeStats("newer-range"));
    await flushPromises();
    older.resolve(makeStats("older-range"));
    await flushPromises();

    const usage = wrapper.findComponent({ name: "UsageByModel" });
    expect(usage.props("byModel")).toEqual([expect.objectContaining({ model: "newer-range" })]);
  });
});

/** The project's TypeScript lib predates Promise.withResolvers. */
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function makeStats(model: string): UsageStatsResponse {
  return {
    summary: {
      total_cost: 0,
      total_requests: 1,
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
    by_model: [
      {
        model,
        provider: "openai",
        requests: 1,
        cost: 0,
        input_tokens: 0,
        output_tokens: 0,
        cache_creation_tokens: 0,
        cache_read_tokens: 0,
        cached_prompt_tokens: 0,
      },
    ],
    daily_usage: [],
  };
}
