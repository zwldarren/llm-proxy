import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import SegmentedControl from "./SegmentedControl.vue";
import SegmentedControlItem from "./SegmentedControlItem.vue";

const toggleHarness = {
  components: { SegmentedControl, SegmentedControlItem },
  data: () => ({ value: "list" as "list" | "table" }),
  template: `
    <SegmentedControl aria-label="View mode">
      <SegmentedControlItem :active="value === 'list'" @click="value = 'list'">
        List
      </SegmentedControlItem>
      <SegmentedControlItem :active="value === 'table'" @click="value = 'table'">
        Table
      </SegmentedControlItem>
    </SegmentedControl>
  `,
};

describe("SegmentedControl", () => {
  it("exposes a radiogroup with the active option checked", async () => {
    const wrapper = mount(toggleHarness, { attachTo: document.body });
    const group = wrapper.find('[data-slot="segmented-control"]');
    let items = wrapper.findAll('[data-slot="segmented-control-item"]');

    expect(group.attributes("role")).toBe("radiogroup");
    expect(items[0].attributes("role")).toBe("radio");
    expect(items[0].attributes("aria-checked")).toBe("true");
    expect(items[1].attributes("aria-checked")).toBe("false");

    await items[1].trigger("click");
    items = wrapper.findAll('[data-slot="segmented-control-item"]');

    expect(items[0].attributes("aria-checked")).toBe("false");
    expect(items[1].attributes("aria-checked")).toBe("true");
    wrapper.unmount();
  });

  it("keeps one roving tab stop and moves it with the selection", async () => {
    const wrapper = mount(toggleHarness, { attachTo: document.body });
    let items = wrapper.findAll('[data-slot="segmented-control-item"]');

    expect(items[0].attributes("tabindex")).toBe("0");
    expect(items[1].attributes("tabindex")).toBe("-1");

    await items[1].trigger("click");
    items = wrapper.findAll('[data-slot="segmented-control-item"]');

    expect(items[0].attributes("tabindex")).toBe("-1");
    expect(items[1].attributes("tabindex")).toBe("0");
    wrapper.unmount();
  });

  it("moves focus and selection with the arrow keys", async () => {
    const wrapper = mount(toggleHarness, { attachTo: document.body });
    const items = wrapper.findAll('[data-slot="segmented-control-item"]');
    const [first, second] = items.map((item) => item.element as HTMLElement);

    first.focus();
    await wrapper.find('[data-slot="segmented-control"]').trigger("keydown", { key: "ArrowRight" });

    expect(document.activeElement).toBe(second);
    expect(wrapper.findComponent({ name: "SegmentedControl" }).exists()).toBe(true);
    expect(items[1].attributes("aria-checked")).toBe("true");
    wrapper.unmount();
  });

  it("uses a nav landmark instead of a radiogroup for navigational tabs", () => {
    const wrapper = mount({
      components: { SegmentedControl, SegmentedControlItem },
      template: `
        <SegmentedControl kind="nav" aria-label="Account">
          <SegmentedControlItem kind="nav" as-child :active="true">
            <a href="#profile">Profile</a>
          </SegmentedControlItem>
        </SegmentedControl>
      `,
    });
    const group = wrapper.find('[data-slot="segmented-control"]');
    const item = wrapper.find('[data-slot="segmented-control-item"]');

    expect(group.element.tagName).toBe("NAV");
    expect(group.attributes("role")).toBeUndefined();
    expect(group.attributes("aria-label")).toBe("Account");
    expect(item.attributes("aria-current")).toBe("page");
    expect(item.attributes("aria-checked")).toBeUndefined();
  });

  it("disables an inactive option without removing it from the group", () => {
    const wrapper = mount({
      components: { SegmentedControl, SegmentedControlItem },
      template: `
        <SegmentedControl aria-label="Scope">
          <SegmentedControlItem :active="true">All</SegmentedControlItem>
          <SegmentedControlItem :disabled="true">Custom</SegmentedControlItem>
        </SegmentedControl>
      `,
    });
    const item = wrapper.findAll('[data-slot="segmented-control-item"]')[1];

    expect(item.attributes("disabled")).toBeDefined();
    expect(item.attributes("aria-disabled")).toBe("true");
    expect(item.attributes("tabindex")).toBe("-1");
  });

  it("cancels the default action of a disabled item rendered as an anchor", async () => {
    const wrapper = mount(
      {
        components: { SegmentedControl, SegmentedControlItem },
        template: `
          <SegmentedControl kind="nav" aria-label="Account">
            <SegmentedControlItem kind="nav" as-child disabled>
              <a href="#security">Security</a>
            </SegmentedControlItem>
          </SegmentedControl>
        `,
      },
      { attachTo: document.body }
    );
    const link = wrapper.find("a");
    let prevented: boolean | undefined;
    link.element.addEventListener("click", (event) => {
      prevented = event.defaultPrevented;
    });

    await link.trigger("click");

    expect(prevented).toBe(true);
    expect(link.attributes("tabindex")).toBe("-1");
    wrapper.unmount();
  });
});
