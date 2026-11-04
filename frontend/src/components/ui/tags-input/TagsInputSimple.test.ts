import { mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";

vi.mock("vue-i18n", () => ({
  useI18n: () => ({ t: (key: string, params: { name: string }) => `${key}:${params.name}` }),
}));

import TagsInputSimple from "./TagsInputSimple.vue";

describe("TagsInputSimple", () => {
  it("drops Reka's empty label reference and names the remove action", () => {
    const wrapper = mount(TagsInputSimple, {
      props: { modelValue: ["alpha"], ariaLabel: "Allowed commands" },
    });

    expect(wrapper.html()).not.toContain('aria-labelledby=""');
    expect(wrapper.get("button").attributes("aria-label")).toBe("common.removeItem:alpha");
    expect(wrapper.get("input").attributes("aria-label")).toBe("Allowed commands");
  });
});
