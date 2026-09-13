import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import i18n from "@/i18n";
import type { PricingTier } from "@/types/schemas";
import PricingTierEditor from "./PricingTierEditor.vue";

function mountEditor(modelValue: PricingTier[] | null = null) {
  return mount(PricingTierEditor, {
    props: { modelValue },
    global: { plugins: [i18n] },
  });
}

function findButton(wrapper: ReturnType<typeof mountEditor>, text: string) {
  const button = wrapper.findAll("button").find((b) => b.text().includes(text));
  if (!button) throw new Error(`button not found: ${text}`);
  return button;
}

describe("PricingTierEditor", () => {
  it("renders the empty hint when no tiers are configured", () => {
    const wrapper = mountEditor(null);
    expect(wrapper.text()).toContain("No tiers");
    expect(wrapper.findAll('[aria-label="Remove tier"]')).toHaveLength(0);
  });

  it("sorts the emitted list by threshold when adding a tier", async () => {
    const wrapper = mountEditor([{ threshold: 300000, input_cost_per_1m: 5 }]);

    await findButton(wrapper, "Add tier").trigger("click");

    const emitted = wrapper.emitted("update:modelValue");
    expect(emitted).toBeTruthy();
    expect(emitted!.at(-1)?.[0]).toEqual([
      { threshold: 300000, input_cost_per_1m: 5 },
      {
        threshold: 600000,
        input_cost_per_1m: null,
        output_cost_per_1m: null,
      },
    ]);
  });

  it("emits null after the last tier is removed", async () => {
    const wrapper = mountEditor([{ threshold: 200000, input_cost_per_1m: 6 }]);

    await wrapper.find('[aria-label="Remove tier"]').trigger("click");

    const emitted = wrapper.emitted("update:modelValue");
    expect(emitted!.at(-1)?.[0]).toBeNull();
  });

  it("warns about duplicate thresholds", () => {
    const wrapper = mountEditor([
      { threshold: 200000, input_cost_per_1m: 6 },
      { threshold: 200000, input_cost_per_1m: 9 },
    ]);

    expect(wrapper.text()).toContain("Duplicate thresholds");
  });

  it("shows a delete control per tier", () => {
    const wrapper = mountEditor([
      { threshold: 128000, input_cost_per_1m: 2 },
      { threshold: 272000, input_cost_per_1m: 5 },
    ]);

    expect(wrapper.findAll('[aria-label="Remove tier"]')).toHaveLength(2);
  });
});
