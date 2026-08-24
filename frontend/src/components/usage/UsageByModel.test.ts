// frontend/src/components/usage/UsageByModel.test.ts
import { mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";
import { defineComponent, h } from "vue";

vi.mock("vue-i18n", () => ({ useI18n: () => ({ t: (k: string) => k }) }));
vi.mock("@/composables/useTheme", () => ({ useTheme: () => ({ isDark: { value: false } }) }));
vi.mock("@lucide/vue", () => {
  const Stub = () => null;
  return { ArrowDown: Stub, ChevronLeft: Stub, ChevronRight: Stub, Cpu: Stub };
});

const ScrollAreaStub = defineComponent({
  name: "ScrollArea",
  setup(_, { slots }) {
    return () => h("div", slots.default?.());
  },
});
const SeparatorStub = defineComponent({
  name: "Separator",
  setup() {
    return () => h("hr");
  },
});
const TooltipProviderStub = defineComponent({
  name: "TooltipProvider",
  setup(_, { slots }) {
    return () => h("div", slots.default?.());
  },
});
const TooltipStub = defineComponent({
  name: "Tooltip",
  setup(_, { slots }) {
    return () => h("span", slots.default?.());
  },
});
const TooltipTriggerStub = defineComponent({
  name: "TooltipTrigger",
  setup(_, { slots }) {
    return () => h("span", slots.default?.());
  },
});
const TooltipContentStub = defineComponent({
  name: "TooltipContent",
  setup(_, { slots }) {
    return () => h("span", slots.default?.());
  },
});
const ButtonStub = defineComponent({
  name: "Button",
  props: ["disabled"],
  setup(_, { slots }) {
    return () => h("button", { disabled: _.disabled }, slots.default?.());
  },
});

import UsageByModel from "./UsageByModel.vue";

const fixture = Array.from({ length: 15 }, (_, i) => ({
  model: `model-${i}`,
  provider: "openai",
  requests: 100 - i,
  cost: 1,
}));

function mountModel() {
  return mount(UsageByModel, {
    props: { byModel: fixture },
    global: {
      stubs: {
        ScrollArea: ScrollAreaStub,
        Separator: SeparatorStub,
        TooltipProvider: TooltipProviderStub,
        Tooltip: TooltipStub,
        TooltipTrigger: TooltipTriggerStub,
        TooltipContent: TooltipContentStub,
        Button: ButtonStub,
      },
    },
  });
}

describe("UsageByModel", () => {
  it("renders the first page of 10 and paginates to the next", async () => {
    const wrapper = mountModel();
    // model-0..model-9 are on page 1 (sorted by requests desc)
    expect(wrapper.text()).toContain("model-0");
    expect(wrapper.text()).not.toContain("model-10");
    // click next (the second ghost Button with a ChevronRight stub)
    const next = wrapper.findAll("button").at(-1)!;
    await next.trigger("click");
    expect(wrapper.text()).toContain("model-10");
    expect(wrapper.text()).not.toContain("model-0");
  });
});
