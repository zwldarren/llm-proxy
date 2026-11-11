import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import SettingsItem from "./SettingsItem.vue";

describe("SettingsItem", () => {
  it("gives the control its own line on a phone and the title column the whole row", () => {
    const wrapper = mount(SettingsItem, {
      props: { title: "Extra Sensitive Keys", description: "A long explanation of the setting." },
      slots: { action: '<button type="button">toggle</button>' },
    });

    // The title and the description own the full width until sm, and the
    // control moves below them. Sharing one row with a control that has an
    // intrinsic width (a text input, a pair of steppers) squeezed the title
    // column to 32px on a 390px screen, one character per line.
    const title = wrapper.find(".col-start-2.row-start-1");
    const description = wrapper.find(".col-start-2.row-start-2");
    const action = wrapper.find(".col-start-1.col-span-2.row-start-3");

    expect(title.exists()).toBe(true);
    expect(description.exists()).toBe(true);
    expect(action.exists()).toBe(true);
    expect(title.text()).toContain("Extra Sensitive Keys");
    expect(action.text()).toContain("toggle");

    // From sm up it is the desktop row again: icon, text, control on the right
    // edge of the first row, spanning both text rows.
    const classes = action.classes();
    for (const cls of ["sm:col-start-3", "sm:col-span-1", "sm:row-start-1", "sm:row-span-2"]) {
      expect(classes).toContain(cls);
    }
  });

  it("stacks a nested row the same way, and returns to two columns from sm", () => {
    const wrapper = mount(SettingsItem, {
      props: { title: "Nested", description: "Details", variant: "nested" },
      slots: { action: '<button type="button">toggle</button>' },
    });

    expect(wrapper.find(".col-start-1.row-start-1").exists()).toBe(true);
    expect(wrapper.find(".col-start-1.row-start-3").exists()).toBe(true);

    const classes = wrapper.find(".col-start-1.row-start-3").classes();
    for (const cls of ["sm:col-start-2", "sm:row-start-1", "sm:row-span-2"]) {
      expect(classes).toContain(cls);
    }
  });
});
