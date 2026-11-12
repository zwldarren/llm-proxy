import { mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";

vi.mock("vue-i18n", () => ({ useI18n: () => ({ t: (key: string) => key }) }));

import AccessScopePicker from "./AccessScopePicker.vue";

const props = {
  items: [
    { id: "alpha", name: "Alpha" },
    { id: "beta", name: "Beta" },
  ],
  allLabel: "All models",
  searchPlaceholder: "Search",
  emptyAvailableText: "None available",
  emptySearchText: "No matches",
  emptyText: "No models selected",
};

describe("AccessScopePicker", () => {
  it("does not hand a cleared deny-all selection back when modes toggle", async () => {
    const wrapper = mount(AccessScopePicker, { props: { ...props, modelValue: ["alpha"] } });

    await wrapper
      .findAll("button")
      .find((button) => button.text() === "accessPicker.clear")!
      .trigger("click");
    expect(wrapper.emitted("update:modelValue")?.at(-1)).toEqual([[]]);

    await wrapper.setProps({ modelValue: [] });
    await wrapper.find('[aria-label="All models"]').trigger("click");
    expect(wrapper.emitted("update:modelValue")?.at(-1)).toEqual([null]);

    await wrapper.setProps({ modelValue: null });
    await wrapper.find('[aria-label="accessPicker.custom"]').trigger("click");
    expect(wrapper.emitted("update:modelValue")?.at(-1)).toEqual([[]]);
  });
});
