// frontend/src/components/config/ParameterOverridesBuilder.test.ts
import { mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";
import { defineComponent, h } from "vue";

vi.mock("vue-i18n", () => ({ useI18n: () => ({ t: (k: string) => k }) }));
vi.mock("vue-sonner", () => ({ toast: { error: vi.fn() } }));
vi.mock("@lucide/vue", () => {
  const Stub = () => null;
  return { Plus: Stub, Trash2: Stub, Code2: Stub, ListTree: Stub };
});

const ButtonStub = defineComponent({
  name: "Button",
  emits: ["click"],
  setup(_, { slots, emit }) {
    return () => h("button", { onClick: () => emit("click") }, slots.default?.());
  },
});
const InputStub = defineComponent({
  name: "Input",
  props: ["modelValue"],
  emits: ["update:modelValue"],
  setup(props, { emit }) {
    return () =>
      h("input", {
        value: props.modelValue,
        onInput: (e: Event) => emit("update:modelValue", (e.target as HTMLInputElement).value),
      });
  },
});
const LabelStub = defineComponent({
  name: "Label",
  setup(_, { slots }) {
    return () => h("label", slots.default?.());
  },
});
const TextareaStub = defineComponent({
  name: "Textarea",
  props: ["modelValue"],
  emits: ["update:modelValue"],
  setup(props, { emit }) {
    return () =>
      h("textarea", {
        value: props.modelValue,
        onInput: (e: Event) => emit("update:modelValue", (e.target as HTMLInputElement).value),
      });
  },
});

import ParameterOverridesBuilder from "./ParameterOverridesBuilder.vue";
import type { ParameterOverridesConfig } from "@/types/parameterOverrides";

const globalStubs = {
  Button: ButtonStub,
  Input: InputStub,
  Label: LabelStub,
  Textarea: TextareaStub,
};

describe("ParameterOverridesBuilder", () => {
  it("keeps empty-key rows when the parent applies a local edit (no mid-edit rebuild)", async () => {
    const wrapper = mount(ParameterOverridesBuilder, {
      props: { modelValue: {} },
      global: { stubs: globalStubs },
    });

    // Add two entries, then fill in the first row's key.
    const addButton = wrapper.findAll("button").at(-1)!;
    await addButton.trigger("click");
    await addButton.trigger("click");
    expect(wrapper.findAll("input").length).toBe(4);

    await wrapper.findAll("input").at(0)!.setValue("temperature");
    const emitted = wrapper.emitted("update:modelValue");
    expect(emitted).toBeTruthy();

    // Parent applies the emitted value back (v-model round trip).
    await wrapper.setProps({
      modelValue: emitted!.at(-1)![0] as ParameterOverridesConfig,
    });

    // Regression: the second, still-empty row must not be dropped.
    expect(wrapper.findAll("input").length).toBe(4);
  });

  it("rebuilds entries from external modelValue changes", async () => {
    const wrapper = mount(ParameterOverridesBuilder, {
      props: { modelValue: {} },
      global: { stubs: globalStubs },
    });

    await wrapper.setProps({ modelValue: { temperature: 0.7, max_tokens: 4096 } });
    expect(wrapper.findAll("input").length).toBe(4);
  });
});
