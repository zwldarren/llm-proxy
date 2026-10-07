import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { DEFAULT_SMART_ROUTING } from "@/constants/defaults";

vi.mock("@/services/api/config", () => {
  const api = () => ({ getConfig: vi.fn(), updateConfig: vi.fn() });
  return {
    circuitBreakerApi: { ...api(), getStates: vi.fn(), reset: vi.fn(), resetAll: vi.fn() },
    configApi: api(),
    corsApi: api(),
    keepaliveApi: api(),
    mcpSecurityApi: api(),
    providerSelectionApi: api(),
    rateLimitsApi: api(),
    requestPolicyApi: api(),
    resilienceApi: api(),
    securityApi: api(),
    smartRoutingApi: api(),
    webSearchApi: api(),
  };
});

import { smartRoutingApi } from "@/services/api/config";
import { useSettingsStore } from "./settings";

beforeEach(() => {
  setActivePinia(createPinia());
  vi.mocked(smartRoutingApi.getConfig).mockReset();
});

describe("settings store: smart routing judge", () => {
  it("fills the judge block from defaults for a row stored before it existed", async () => {
    vi.mocked(smartRoutingApi.getConfig).mockResolvedValue({
      enabled: true,
      mode_weights: { fast: 0.35, auto: 0.65, best: 1.0 },
    } as never);

    const config = await useSettingsStore().fetchSmartRouting();

    expect(config.judge.enabled).toBe(false);
    expect(config.judge.model).toBe("");
    expect(config.judge.modes).toEqual(DEFAULT_SMART_ROUTING.judge.modes);
    expect(config.judge.shadow).toBe(true);
    expect(config.judge.confidence_below).toBeNull();
  });

  it("preserves a stored judge block, gate and all", async () => {
    vi.mocked(smartRoutingApi.getConfig).mockResolvedValue({
      enabled: true,
      mode_weights: { fast: 0.35, auto: 0.65, best: 1.0 },
      judge: {
        enabled: true,
        model: "tev1:0.8b",
        modes: ["auto"],
        deadline_s: 0.4,
        confidence_below: 0.6,
        complexity_between: null,
        shadow: false,
        shadow_sample_rate: 0,
        context_turns: 1,
        context_chars: 1000,
      },
    } as never);

    const config = await useSettingsStore().fetchSmartRouting();

    expect(config.judge).toEqual({
      enabled: true,
      model: "tev1:0.8b",
      modes: ["auto"],
      deadline_s: 0.4,
      confidence_below: 0.6,
      complexity_between: null,
      shadow: false,
      shadow_sample_rate: 0,
      context_turns: 1,
      context_chars: 1000,
    });
  });

  it("copies the band instead of aliasing the response", async () => {
    const stored: [number, number] = [0.33, 0.67];
    vi.mocked(smartRoutingApi.getConfig).mockResolvedValue({
      enabled: true,
      mode_weights: { fast: 0.35, auto: 0.65, best: 1.0 },
      judge: { complexity_between: stored },
    } as never);

    const config = await useSettingsStore().fetchSmartRouting();

    expect(config.judge.complexity_between).toEqual(stored);
    expect(config.judge.complexity_between).not.toBe(stored);
  });

  it("preserves an explicitly empty judged-mode list", async () => {
    vi.mocked(smartRoutingApi.getConfig).mockResolvedValue({
      enabled: true,
      mode_weights: { fast: 0.35, auto: 0.65, best: 1.0 },
      judge: { enabled: true, model: "tev1:0.8b", modes: [] },
    } as never);

    const config = await useSettingsStore().fetchSmartRouting();

    // Empty is a meaningful choice (no virtual model is ever judged) and must
    // not be replaced by the defaults on the next load.
    expect(config.judge.modes).toEqual([]);
  });
});
