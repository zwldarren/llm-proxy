// frontend/src/components/usage/UsageByProvider.test.ts
import { mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";
import { defineComponent, h } from "vue";

vi.mock("vue-i18n", () => ({ useI18n: () => ({ t: (k: string) => k }) }));
vi.mock("@/composables/useTheme", () => ({ useTheme: () => ({ isDark: { value: false } }) }));
vi.mock("@lucide/vue", () => {
  const Stub = () => null;
  return { ArrowDown: Stub, Database: Stub };
});

// Stub ScrollArea + Tooltip* so list rows render directly.
const ScrollAreaStub = defineComponent({
  name: "ScrollArea",
  setup(_, { slots }) {
    return () => h("div", slots.default?.());
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

import UsageByProvider from "./UsageByProvider.vue";

const fixture = [
  {
    provider: "anthropic",
    requests: 100,
    cost: 1,
    input_tokens: 10,
    output_tokens: 10,
    cache_creation_tokens: 0,
    cache_read_tokens: 0,
    cached_prompt_tokens: 0,
  },
  {
    provider: "openai",
    requests: 500,
    cost: 5,
    input_tokens: 10,
    output_tokens: 10,
    cache_creation_tokens: 0,
    cache_read_tokens: 0,
    cached_prompt_tokens: 0,
  },
  {
    provider: "gemini",
    requests: 50,
    cost: 9,
    input_tokens: 10,
    output_tokens: 10,
    cache_creation_tokens: 0,
    cache_read_tokens: 0,
    cached_prompt_tokens: 0,
  },
];

function mountProvider() {
  return mount(UsageByProvider, {
    props: { byProvider: fixture },
    global: {
      stubs: {
        ScrollArea: ScrollAreaStub,
        TooltipProvider: TooltipProviderStub,
        Tooltip: TooltipStub,
        TooltipTrigger: TooltipTriggerStub,
        TooltipContent: TooltipContentStub,
      },
    },
  });
}

describe("UsageByProvider", () => {
  it("sorts by requests desc by default", () => {
    const text = mountProvider().text();
    const a = text.indexOf("openai");
    const b = text.indexOf("anthropic");
    expect(a).toBeLessThan(b); // 500 before 100
  });

  it("re-sorts by cost when the cost header is clicked", async () => {
    const wrapper = mountProvider();
    const buttons = wrapper.findAll("button");
    const costBtn = buttons.find((b) => b.text().includes("home.costUsd"));
    await costBtn!.trigger("click");
    const text = wrapper.text();
    expect(text.indexOf("gemini")).toBeLessThan(text.indexOf("openai")); // 9 before 5
  });
});
