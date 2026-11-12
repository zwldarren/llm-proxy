import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import BudgetMeter from "./BudgetMeter.vue";

describe("BudgetMeter", () => {
  it("does not announce unknown spend as zero", () => {
    const wrapper = mount(BudgetMeter, { props: { ratio: null, label: "Budget spend" } });

    expect(wrapper.find('[role="progressbar"]').exists()).toBe(false);
    expect(wrapper.find('[aria-hidden="true"]').exists()).toBe(true);
    expect(wrapper.text()).toBe("");
  });

  it("clamps an overspent ratio and exposes a bounded progress value", () => {
    const wrapper = mount(BudgetMeter, { props: { ratio: 1.8, label: "Budget spend" } });

    expect(wrapper.find('[role="progressbar"]').attributes("aria-valuenow")).toBe("100");
  });
});
