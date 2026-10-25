import { mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";

vi.mock("vue-i18n", () => ({ useI18n: () => ({ t: (k: string) => k }) }));

// The real sheet portals its content and the real select opens a popup; both
// need a browser surface this test does not have. The stubs keep the template
// wiring (model value and update handler) observable.
vi.mock("@/components/ui/sheet", () => ({
  Sheet: { template: "<div><slot /></div>" },
  SheetContent: { template: "<div><slot /></div>" },
}));
vi.mock("@/components/ui/sheet/SheetTitle.vue", () => ({
  default: { template: "<div><slot /></div>" },
}));
vi.mock("@/components/ui/sheet/SheetDescription.vue", () => ({
  default: { template: "<div><slot /></div>" },
}));
vi.mock("@/components/ui/select", () => ({
  Select: {
    props: { modelValue: { type: String, default: "" } },
    emits: ["update:modelValue"],
    template:
      '<select :value="modelValue" @change="$emit(\'update:modelValue\', $event.target.value)"><slot /></select>',
  },
  SelectContent: { template: "<slot />" },
  SelectItem: { props: ["value"], template: '<option :value="value"><slot /></option>' },
  SelectTrigger: { template: "<slot />" },
  SelectValue: { template: "<span />" },
}));

import type { TracingProviderEditor } from "@/composables/useTracingProviders";
import type { TracingProvider } from "@/types/schemas";
import TracingProviderSheet from "./TracingProviderSheet.vue";

function makeProvider(settings: Record<string, unknown>): TracingProvider {
  return { id: "provider-1", provider: "langfuse", name: "langfuse 1", enabled: true, settings };
}

function makeEditor(provider: TracingProvider) {
  const updateProviderField = vi.fn();
  const editor = {
    getProviderFieldValue: (p: TracingProvider, field: string) => p.settings[field],
    updateProviderField,
  } as unknown as TracingProviderEditor;
  return { editor, updateProviderField, provider };
}

describe("TracingProviderSheet", () => {
  it("shows the backend default, full, when no content mode was saved", () => {
    const { editor, provider } = makeEditor(makeProvider({ public_key: "pk", secret_key: "sk" }));

    const wrapper = mount(TracingProviderSheet, { props: { provider, editor } });

    expect((wrapper.find("select").element as HTMLSelectElement).value).toBe("full");
  });

  it("saves the selected content mode", async () => {
    const { editor, provider, updateProviderField } = makeEditor(
      makeProvider({ public_key: "pk", secret_key: "sk", send_content: "full" })
    );

    const wrapper = mount(TracingProviderSheet, { props: { provider, editor } });
    await wrapper.find("select").setValue("inherit");

    expect(updateProviderField).toHaveBeenCalledWith("provider-1", "send_content", "inherit");
  });

  it("keeps an explicitly saved mode visible", () => {
    const { editor, provider } = makeEditor(
      makeProvider({ public_key: "pk", secret_key: "sk", send_content: "none" })
    );

    const wrapper = mount(TracingProviderSheet, { props: { provider, editor } });

    expect((wrapper.find("select").element as HTMLSelectElement).value).toBe("none");
  });
});
