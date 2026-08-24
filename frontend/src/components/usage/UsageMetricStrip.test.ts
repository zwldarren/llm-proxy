import { mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";

vi.mock("vue-i18n", () => ({ useI18n: () => ({ t: (k: string) => k }) }));
vi.mock("@lucide/vue", () => {
  const Stub = () => null;
  return {
    Activity: Stub,
    CheckCircle2: Stub,
    Clock: Stub,
    Cpu: Stub,
    DollarSign: Stub,
    Package: Stub,
    Zap: Stub,
  };
});

import UsageMetricStrip from "./UsageMetricStrip.vue";
import type { UsageByProvider, UsageSummary } from "@/types/schemas";

const summary: UsageSummary = {
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
};

const byProvider: UsageByProvider[] = [
  {
    provider: "anthropic",
    requests: 800,
    cost: 3.0,
    input_tokens: 50000,
    output_tokens: 15000,
    cache_creation_tokens: 0,
    cache_read_tokens: 5000,
    cached_prompt_tokens: 0,
  },
  {
    provider: "ollama",
    requests: 434,
    cost: 0.81,
    input_tokens: 30000,
    output_tokens: 5000,
    cache_creation_tokens: 0,
    cache_read_tokens: 0,
    cached_prompt_tokens: 0,
  },
];

function mountStrip() {
  return mount(UsageMetricStrip, { props: { summary, byProvider } });
}

describe("UsageMetricStrip", () => {
  it("renders the six primary cells with their values and labels", () => {
    const text = mountStrip().text();
    expect(text).toContain("home.totalRequests");
    expect(text).toContain("1.2K"); // formatNumberWithSuffix(1234)
    expect(text).toContain("home.totalCost");
    expect(text).toContain("$3.81"); // formatCostWithPrecision(3.81, 2)
    expect(text).toContain("home.totalTokens");
    expect(text).toContain("home.successRate");
    expect(text).toContain("100.0%"); // formatPercentage(100)
    expect(text).toContain("home.avgThroughput");
    expect(text).toContain("41.0"); // avg_tokens_per_second.toFixed(1)
  });

  it("shows the optimal success tier at 100%", () => {
    expect(mountStrip().text()).toContain("home.statusOptimal");
  });

  it("shows the critical success tier below 95%", () => {
    const wrapper = mount(UsageMetricStrip, {
      props: { summary: { ...summary, success_rate: 90 }, byProvider },
    });
    expect(wrapper.text()).toContain("home.statusCritical");
  });

  it("renders the secondary strip (token distribution + cache, with savings)", () => {
    const text = mountStrip().text();
    expect(text).toContain("home.tokenDistribution");
    expect(text).toContain("home.cacheEfficiency");
    expect(text).toContain("$0.420"); // formatCostWithPrecision(0.42) default precision 3
  });

  it("renders no Card / no rounded-xl nested boxes", () => {
    const html = mountStrip().html();
    expect(html).not.toContain('class="card-container');
    // primary strip is a grid of cells separated by divide-x, not bordered boxes
    expect(html).not.toContain("rounded-xl border");
  });

  it("computes cache hit rate excluding non-cache-reporting providers", () => {
    // With the test data: 5000 cache_read_tokens from anthropic (50K input_tokens),
    // ollama has 50K input_tokens but 0 cache tokens.
    // Old behavior: 5000 / 100000 = 5.0%
    // New behavior: 5000 / 50000 = 10.0% (only anthropic's input tokens)
    const text = mountStrip().text();
    expect(text).toContain("10.0%");
  });

  it("shows 0% cache hit when no providers report cache data", () => {
    const noCacheProviders: UsageByProvider[] = [
      {
        provider: "ollama",
        requests: 100,
        cost: 0,
        input_tokens: 10000,
        output_tokens: 5000,
        cache_creation_tokens: 0,
        cache_read_tokens: 0,
        cached_prompt_tokens: 0,
      },
    ];
    const wrapper = mount(UsageMetricStrip, {
      props: {
        summary: { ...summary, total_cache_read_tokens: 0, total_cached_prompt_tokens: 0 },
        byProvider: noCacheProviders,
      },
    });
    expect(wrapper.text()).toContain("0.0%");
  });
});
