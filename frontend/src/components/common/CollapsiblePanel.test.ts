import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import CollapsiblePanel from "./CollapsiblePanel.vue";

describe("CollapsiblePanel", () => {
  it("removes closed content from focus and the accessibility tree", async () => {
    const wrapper = mount(CollapsiblePanel, {
      props: { open: false },
      slots: { default: "<button>Hidden action</button>" },
    });

    expect(wrapper.attributes("inert")).toBe("");

    await wrapper.setProps({ open: true });
    expect(wrapper.attributes("inert")).toBeUndefined();
  });
});
