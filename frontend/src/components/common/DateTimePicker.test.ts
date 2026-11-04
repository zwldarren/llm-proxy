import { mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";
import { nextTick } from "vue";

vi.mock("vue-i18n", () => {
  return {
    useI18n: () => ({ t: (key: string) => key }),
  };
});

import DateTimePicker from "./DateTimePicker.vue";

const VALUE = "2031-04-05T09:30:00.000Z";

async function mountPicker(modelValue: string | null) {
  const wrapper = mount(DateTimePicker, {
    props: { modelValue, clearLabel: "Clear", placeholder: "Pick a date" },
  });
  await nextTick();
  return wrapper;
}

/**
 * Clear used to be an `<svg>` nested inside the popover trigger button: a bare
 * SVG has no role, so its aria-label was dropped, and a click handler on it is
 * pointer-only — no tab stop, no Enter, no Space. It is now a real sibling
 * button, which is what these assertions protect.
 */
describe("DateTimePicker clear control", () => {
  it("exposes Clear as its own labelled button", async () => {
    const wrapper = await mountPicker(VALUE);

    const clear = wrapper.get('button[aria-label="Clear"]');
    expect(clear.attributes("type")).toBe("button");
    // 24px minimum target, not the 14px icon it replaced.
    expect(clear.classes()).toContain("size-6");
  });

  it("clears the value without opening the calendar", async () => {
    const wrapper = await mountPicker(VALUE);

    await wrapper.get('button[aria-label="Clear"]').trigger("click");

    expect(wrapper.emitted("update:modelValue")).toEqual([[null]]);
    // The trigger button is a separate element, so no popover toggle follows.
    expect(wrapper.emitted("update:open")).toBeUndefined();
  });

  it("renders no clear control when the field is empty", async () => {
    const wrapper = await mountPicker(null);

    expect(wrapper.find('button[aria-label="Clear"]').exists()).toBe(false);
  });
});
