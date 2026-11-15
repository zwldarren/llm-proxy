import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import i18n from "@/i18n";
import type { PricingTimeWindow } from "@/types/schemas";
import { NumberInput } from "@/components/ui/number-input";
import PricingTimeWindowEditor from "./PricingTimeWindowEditor.vue";

function mountEditor(modelValue: PricingTimeWindow[] | null = null) {
  return mount(PricingTimeWindowEditor, {
    props: { modelValue },
    global: { plugins: [i18n] },
  });
}

function findButton(wrapper: ReturnType<typeof mountEditor>, text: string) {
  const button = wrapper.findAll("button").find((b) => b.text().includes(text));
  if (!button) throw new Error(`button not found: ${text}`);
  return button;
}

/** The clock rows of one band, in render order. */
function rangeInputs(wrapper: ReturnType<typeof mountEditor>) {
  return wrapper.findAll('input[aria-label^="Range start"]');
}

const peakRanges: PricingTimeWindow[] = [
  { start: "01:00", end: "04:00", days: [], timezone: "UTC", input_cost_per_1m: 2 },
  { start: "06:00", end: "10:00", days: [], timezone: "UTC", input_cost_per_1m: 2 },
];

describe("PricingTimeWindowEditor", () => {
  it("renders the empty hint when no windows are configured", () => {
    const wrapper = mountEditor(null);
    expect(wrapper.text()).toContain("No bands");
    expect(rangeInputs(wrapper)).toHaveLength(0);
  });

  it("shows one band with a row per range when the windows share timezone, days and rates", () => {
    // "Peak hours are 01:00-04:00 and 06:00-10:00 UTC" must read as one band.
    const wrapper = mountEditor(peakRanges);

    expect(rangeInputs(wrapper)).toHaveLength(2);
    expect(wrapper.findAll('[aria-label="Remove band"]')).toHaveLength(1);
    expect(wrapper.findAll('[aria-label="Remove range"]')).toHaveLength(2);
  });

  it("keeps windows that price differently in separate bands", () => {
    const wrapper = mountEditor([
      { start: "01:00", end: "04:00", timezone: "UTC", input_cost_per_1m: 2 },
      { start: "06:00", end: "10:00", timezone: "UTC", input_cost_per_1m: 5 },
    ]);

    expect(wrapper.findAll('[aria-label="Remove band"]')).toHaveLength(2);
  });

  it("adds a range to the band rather than a second band", async () => {
    const wrapper = mountEditor(peakRanges);

    await findButton(wrapper, "Add range").trigger("click");

    const emitted = wrapper.emitted("update:modelValue")!;
    expect(emitted.at(-1)?.[0]).toEqual([
      ...peakRanges,
      { start: "10:00", end: "11:00", days: [], timezone: "UTC", input_cost_per_1m: 2 },
    ]);
  });

  it("applies a rate to every range in the band", async () => {
    const wrapper = mountEditor(peakRanges);

    await wrapper.findAllComponents(NumberInput)[0]!.vm.$emit("update:modelValue", 3);

    const emitted = wrapper.emitted("update:modelValue")!;
    expect(emitted.at(-1)?.[0]).toEqual([
      { ...peakRanges[0], input_cost_per_1m: 3 },
      { ...peakRanges[1], input_cost_per_1m: 3 },
    ]);
  });

  it("applies a timezone to every range in the band", async () => {
    const wrapper = mountEditor(peakRanges);

    await wrapper.find('input[aria-label="Timezone (IANA)"]').setValue("Asia/Shanghai");

    const emitted = wrapper.emitted("update:modelValue")!;
    expect((emitted.at(-1)?.[0] as PricingTimeWindow[]).map((window) => window.timezone)).toEqual([
      "Asia/Shanghai",
      "Asia/Shanghai",
    ]);
  });

  it("removes only the range whose row was cleared", async () => {
    const wrapper = mountEditor(peakRanges);

    await wrapper.findAll('[aria-label="Remove range"]')[1]!.trigger("click");

    const emitted = wrapper.emitted("update:modelValue")!;
    expect(emitted.at(-1)?.[0]).toEqual([peakRanges[0]]);
  });

  it("drops the band when its last range is removed", async () => {
    const wrapper = mountEditor([peakRanges[0]!]);

    await wrapper.find('[aria-label="Remove range"]').trigger("click");

    const emitted = wrapper.emitted("update:modelValue")!;
    expect(emitted.at(-1)?.[0]).toBeNull();
  });

  it("stores the empty list instead of null for a provider mapping", async () => {
    const wrapper = mount(PricingTimeWindowEditor, {
      props: { modelValue: [peakRanges[0]!], inheritWhenEmpty: false },
      global: { plugins: [i18n] },
    });

    await wrapper.find('[aria-label="Remove range"]').trigger("click");

    const emitted = wrapper.emitted("update:modelValue")!;
    expect(emitted.at(-1)?.[0]).toEqual([]);
  });

  it("normalizes a typed 24-hour clock instead of asking for AM or PM", async () => {
    const wrapper = mountEditor([peakRanges[0]!]);

    const start = rangeInputs(wrapper)[0]!;
    await start.setValue("930");
    await start.trigger("blur");

    const emitted = wrapper.emitted("update:modelValue")!;
    expect((emitted.at(-1)?.[0] as PricingTimeWindow[])[0]?.start).toBe("09:30");
  });

  it("reverts an unreadable clock rather than storing it", async () => {
    const wrapper = mountEditor([peakRanges[0]!]);

    const start = rangeInputs(wrapper)[0]!;
    await start.setValue("9:99");
    await start.trigger("blur");

    expect(wrapper.emitted("update:modelValue")).toBeUndefined();
    expect((start.element as HTMLInputElement).value).toBe("01:00");
  });

  it("warns when two ranges of one band overlap", () => {
    const wrapper = mountEditor([
      { start: "01:00", end: "04:00", timezone: "UTC", input_cost_per_1m: 2 },
      { start: "03:00", end: "05:00", timezone: "UTC", input_cost_per_1m: 2 },
    ]);

    expect(wrapper.text()).toContain("overlap");
  });

  it("warns when the timezone cannot be resolved", () => {
    const wrapper = mountEditor([
      { start: "01:00", end: "04:00", timezone: "Asia/Atlantis", input_cost_per_1m: 2 },
    ]);

    expect(wrapper.text()).toContain("Unknown timezone");
  });
});

describe("PricingTimeWindowEditor accessibility", () => {
  it("names the band group so repeated per-band inputs stay distinguishable", () => {
    const wrapper = mountEditor([
      ...peakRanges,
      { start: "20:00", end: "22:00", timezone: "Asia/Shanghai" },
    ]);

    expect(wrapper.findAll("legend").map((legend) => legend.text())).toEqual([
      "Peak band 1",
      "Peak band 2",
    ]);
  });
});
