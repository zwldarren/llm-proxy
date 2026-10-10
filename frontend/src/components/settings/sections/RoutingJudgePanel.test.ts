import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ref } from "vue";

vi.mock("vue-i18n", () => ({ useI18n: () => ({ t: (k: string) => k }) }));
vi.mock("@lucide/vue", () => {
  const Stub = () => null;
  return { Check: Stub, ChevronDown: Stub, ChevronUp: Stub, Loader2: Stub };
});

vi.mock("@/services/api/config", () => ({
  configApi: { getModels: vi.fn() },
}));

import { configApi } from "@/services/api/config";
import { DEFAULT_ROUTING_JUDGE, DEFAULT_SMART_ROUTING } from "@/constants/defaults";
import type { AutoSaveState } from "@/composables/useSettingAutoSave";
import type { RoutingJudgeConfig, SmartRoutingConfig } from "@/types/schemas";
import RoutingJudgePanel from "./RoutingJudgePanel.vue";

function makeAutoSave(judge: Partial<RoutingJudgeConfig> = {}) {
  const state = ref<SmartRoutingConfig>({
    ...DEFAULT_SMART_ROUTING,
    enabled: true,
    judge: { ...DEFAULT_ROUTING_JUDGE, ...judge },
  });
  const autoSave = {
    state,
    pending: ref(false),
    error: ref<string | null>(null),
    initialize: vi.fn(),
    save: vi.fn(async () => {}),
  } as unknown as AutoSaveState<SmartRoutingConfig>;
  return { autoSave, state };
}

const MODELS = [
  { name: "gpt-4o", supports_systemone: false },
  { name: "tev1:0.8b", supports_systemone: true },
  { name: "typesafe/jev", supports_systemone: true },
];

beforeEach(() => {
  vi.mocked(configApi.getModels).mockReset();
  vi.mocked(configApi.getModels).mockResolvedValue(MODELS as never);
});

async function mountPanel(judge: Partial<RoutingJudgeConfig> = {}) {
  const { autoSave, state } = makeAutoSave(judge);
  const wrapper = mount(RoutingJudgePanel, { props: { autoSave } });
  await flushPromises();
  return { wrapper, state };
}

describe("RoutingJudgePanel", () => {
  it("switches the judge on", async () => {
    const { wrapper, state } = await mountPanel();

    expect(wrapper.find('[data-testid="judge-enabled"]').exists()).toBe(true);
    await wrapper.find('[data-testid="judge-enabled"]').trigger("click");

    expect(state.value.judge.enabled).toBe(true);
  });

  it("keeps only System One models in the picker, plus whatever is configured", async () => {
    const { wrapper } = await mountPanel({ enabled: true, model: "legacy/judge" });

    // The picker keeps the configured name visible even when it is no longer
    // marked as a System One model, so the operator can see what is set.
    const source = (wrapper.vm as unknown as { modelOptions: string[] }).modelOptions;
    expect(source).toEqual(["legacy/judge", "tev1:0.8b", "typesafe/jev"]);
  });

  it("explains itself when no System One model exists yet", async () => {
    vi.mocked(configApi.getModels).mockResolvedValue([
      { name: "gpt-4o", supports_systemone: false },
    ] as never);

    const { wrapper } = await mountPanel({ enabled: true });

    expect(wrapper.text()).toContain("smartRouting.judgeModelNone");
  });

  it("derives the gate from the two stored predicates", async () => {
    const confidence = await mountPanel({ enabled: true, confidence_below: 0.6 });
    expect(confidence.wrapper.text()).toContain("smartRouting.judgeConfidenceBelow");
    expect(confidence.wrapper.text()).not.toContain("smartRouting.judgeBandDescription");

    const band = await mountPanel({ enabled: true, complexity_between: [0.33, 0.67] });
    expect(band.wrapper.text()).toContain("smartRouting.judgeBandDescription");
    expect(band.wrapper.text()).not.toContain("smartRouting.judgeConfidenceBelow");

    const open = await mountPanel({ enabled: true });
    expect(open.wrapper.text()).not.toContain("smartRouting.judgeBandDescription");
    expect(open.wrapper.text()).not.toContain("smartRouting.judgeConfidenceBelow");
  });

  it("shows both gate predicates when a config sets both", async () => {
    const both = await mountPanel({
      enabled: true,
      confidence_below: 0.6,
      complexity_between: [0.33, 0.67],
    });

    // The backend treats the predicates as an OR and does not reject both being
    // set, so neither may be hidden from the operator.
    expect(both.wrapper.text()).toContain("smartRouting.judgeConfidenceBelow");
    expect(both.wrapper.text()).toContain("smartRouting.judgeBandDescription");
  });

  it("clamps the band so start and end stay ordered", async () => {
    const { wrapper, state } = await mountPanel({
      enabled: true,
      complexity_between: [0.33, 0.67],
    });
    const vm = wrapper.vm as unknown as {
      setBandLow: (value: number) => void;
      setBandHigh: (value: number) => void;
    };

    vm.setBandLow(0.9);
    expect(state.value.judge.complexity_between![0]).toBeLessThan(
      state.value.judge.complexity_between![1]
    );

    vm.setBandHigh(0.1);
    expect(state.value.judge.complexity_between![0]).toBeLessThan(
      state.value.judge.complexity_between![1]
    );
  });

  it("shows the shadow sample rate only while shadow mode is on", async () => {
    const shadowed = await mountPanel({ enabled: true, shadow: true });
    expect(shadowed.wrapper.text()).toContain("smartRouting.judgeShadowSampleRate");

    const live = await mountPanel({ enabled: true, shadow: false });
    expect(live.wrapper.text()).not.toContain("smartRouting.judgeShadowSampleRate");
  });

  it("maps the ADR-0029 postures: all-off is shadow-only, anchor and always named", async () => {
    const shadowOnly = await mountPanel({ enabled: true });
    expect((shadowOnly.wrapper.vm as unknown as { trigger: string }).trigger).toBe("shadow");

    const anchor = await mountPanel({ enabled: true, not_low_anchor: true });
    expect((anchor.wrapper.vm as unknown as { trigger: string }).trigger).toBe("anchor");

    const open = await mountPanel({ enabled: true, always: true });
    expect((open.wrapper.vm as unknown as { trigger: string }).trigger).toBe("always");
  });

  it("carries the shadow switch with the shadow-only posture", async () => {
    // Shadow-only is meaningless with the shadow window off: the gate never
    // fires and nothing is consulted, so the posture must carry the switch.
    const { wrapper, state } = await mountPanel({ enabled: true, shadow: false });
    const vm = wrapper.vm as unknown as { setTrigger: (value: unknown) => void };

    vm.setTrigger("shadow");
    expect(state.value.judge.shadow).toBe(true);
    expect(state.value.judge.always).toBe(false);
    expect(state.value.judge.confidence_below).toBeNull();
  });

  it("toggles both consumption bars together, preserving a pre-set bar", async () => {
    const { wrapper, state } = await mountPanel({ enabled: true });
    const vm = wrapper.vm as unknown as { setConsumption: (on: boolean) => void };

    vm.setConsumption(true);
    expect(state.value.judge.apply_min_confidence).toBe(0.9);
    expect(state.value.judge.apply_min_escalation).toBe(0.5);

    vm.setConsumption(false);
    expect(state.value.judge.apply_min_confidence).toBeNull();
    expect(state.value.judge.apply_min_escalation).toBeNull();

    // A pre-set bar survives the switch going on: only nulls are defaulted.
    state.value.judge.apply_min_escalation = 0.7;
    vm.setConsumption(true);
    expect(state.value.judge.apply_min_confidence).toBe(0.9);
    expect(state.value.judge.apply_min_escalation).toBe(0.7);
  });

  it("adds and removes judged modes, keeping the configured order", async () => {
    const { wrapper, state } = await mountPanel({ enabled: true, modes: ["auto", "best"] });
    const boxes = wrapper.findAll('[role="checkbox"]');
    expect(boxes).toHaveLength(3);

    await boxes[2]!.trigger("click");
    expect(state.value.judge.modes).toEqual(["auto", "best", "fast"]);

    await boxes[2]!.trigger("click");
    expect(state.value.judge.modes).toEqual(["auto", "best"]);
  });
});
