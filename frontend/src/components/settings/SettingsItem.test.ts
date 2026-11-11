import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import SettingsItem from "./SettingsItem.vue";

describe("SettingsItem", () => {
  it("keeps the control on the title's line on a phone", () => {
    const wrapper = mount(SettingsItem, {
      props: { title: "Log input/output", description: "A long explanation of the setting." },
      slots: { action: '<button type="button">toggle</button>' },
    });

    // The title and the control share the first grid row; the description
    // takes the second one. A control that drifted below a long description
    // stopped reading as belonging to its setting.
    const title = wrapper.find(".col-start-2.row-start-1");
    const action = wrapper.find(".col-start-3.row-start-1.row-span-2");
    const description = wrapper.find(".col-start-2.row-start-2");

    expect(title.exists()).toBe(true);
    expect(action.exists()).toBe(true);
    expect(description.exists()).toBe(true);
    expect(title.text()).toContain("Log input/output");
    expect(action.text()).toContain("toggle");
  });

  it("puts the control in the second column of a nested row", () => {
    const wrapper = mount(SettingsItem, {
      props: { title: "Nested", description: "Details", variant: "nested" },
      slots: { action: '<button type="button">toggle</button>' },
    });

    expect(wrapper.find(".col-start-1.row-start-1").exists()).toBe(true);
    expect(wrapper.find(".col-start-2.row-start-1.row-span-2").exists()).toBe(true);
  });
});
