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
  emits: ["update:modelValue", "input", "blur"],
  setup(props, { emit }) {
    return () =>
      h("textarea", {
        value: props.modelValue,
        onInput: (e: Event) => {
          emit("update:modelValue", (e.target as HTMLTextAreaElement).value);
          emit("input");
        },
        onBlur: () => emit("blur"),
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

  it("does not offer the visual editor for an operations value", async () => {
    const operations = [{ mode: "set", path: "temperature", value: 0.2 }];
    const wrapper = mount(ParameterOverridesBuilder, {
      props: { modelValue: { operations } as ParameterOverridesConfig },
      global: { stubs: globalStubs },
    });

    const segments = wrapper.findAll('[data-slot="segmented-control-item"]');

    // The key/value list cannot express an operation, so offering it would have
    // shown "no parameters configured" and then dropped the list on the next
    // keystroke.
    expect(segments[0]!.attributes("aria-disabled")).toBe("true");
    expect(wrapper.text()).toContain("parameterOverrides.operationsJsonOnly");
    expect(wrapper.find("textarea").exists()).toBe(true);
  });

  it("flags invalid JSON in place instead of keeping the old value silently", async () => {
    const wrapper = mount(ParameterOverridesBuilder, {
      props: { modelValue: {} },
      global: { stubs: globalStubs },
    });

    await wrapper.findAll('[data-slot="segmented-control-item"]')[1]!.trigger("click");
    const textarea = wrapper.find("textarea");
    await textarea.setValue("{ not json");
    await textarea.trigger("blur");

    // The error belongs next to the field: the parent still holds the previous
    // object, and a toast that fades left a payload missing what was typed.
    expect(wrapper.text()).toContain("common.invalidJson");
    expect(textarea.attributes("aria-invalid")).toBe("true");
    expect(wrapper.emitted("update:modelValue")).toBeFalsy();
  });
});
