import { mount } from "@vue/test-utils";
import { ref } from "vue";
import { describe, expect, it, vi } from "vitest";

vi.mock("vue-i18n", () => ({ useI18n: () => ({ t: (k: string) => k, te: () => true }) }));

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
import type { TracingProvider, TracingProviderDetails } from "@/types/schemas";
import TracingProviderSheet from "./TracingProviderSheet.vue";

const LANGFUSE: TracingProviderDetails = {
  name: "langfuse",
  required_fields: ["public_key", "secret_key"],
  optional_fields: ["base_url", "send_content"],
  fields: [
    { name: "base_url", type: "text", required: false, description: "Langfuse base URL." },
    { name: "public_key", type: "text", required: true, description: "Langfuse public key." },
    { name: "secret_key", type: "password", required: true, description: "Langfuse secret key." },
    {
      name: "send_content",
      type: "select",
      required: false,
      choices: ["inherit", "none", "full"],
      default: "full",
      description: "Which content to export.",
    },
  ],
};

const OTEL: TracingProviderDetails = {
  name: "otel",
  required_fields: ["endpoint"],
  optional_fields: ["protocol", "headers"],
  fields: [
    { name: "endpoint", type: "text", required: true, description: "Collector endpoint." },
    {
      name: "protocol",
      type: "select",
      required: false,
      choices: ["http", "grpc"],
      default: "http",
      description: "OTLP transport.",
    },
    { name: "headers", type: "headers", required: false, description: "Extra headers." },
    {
      name: "insecure",
      type: "boolean",
      required: false,
      default: false,
      description: "Send gRPC without TLS.",
    },
    {
      name: "sample_rate",
      type: "number",
      required: false,
      min: 0,
      max: 1,
      description: "Sampling rate.",
    },
  ],
};

const LANGSMITH: TracingProviderDetails = {
  name: "langsmith",
  required_fields: ["api_key"],
  optional_fields: ["project", "endpoint"],
  fields: [
    { name: "api_key", type: "password", required: true, description: "LangSmith API key." },
    { name: "project", type: "text", required: false, default: "default", description: "Project." },
    {
      name: "endpoint",
      type: "text",
      required: false,
      default: "https://api.smith.langchain.com/otel/v1/traces",
      description: "OTLP gateway.",
    },
  ],
};

function makeProvider(settings: Record<string, unknown>, provider = "langfuse"): TracingProvider {
  return { id: "provider-1", provider, name: `${provider} 1`, enabled: true, settings };
}

function makeEditor(
  provider: TracingProvider,
  types: TracingProviderDetails[]
): { editor: TracingProviderEditor; updateProviderField: ReturnType<typeof vi.fn> } {
  const updateProviderField = vi.fn();
  const editor = {
    providerTypes: ref(types),
    getProviderFieldValue: (p: TracingProvider, field: string) => p.settings[field],
    updateProviderField,
  } as unknown as TracingProviderEditor;
  return { editor, updateProviderField };
}

function fieldInput(wrapper: ReturnType<typeof mount>, provider: string, name: string) {
  return wrapper.find(`input[aria-label="tracing.fields.${provider}.${name}.label"]`);
}

describe("TracingProviderSheet", () => {
  it("shows the backend default, full, when no content mode was saved", () => {
    const provider = makeProvider({ public_key: "pk", secret_key: "sk" });
    const { editor } = makeEditor(provider, [LANGFUSE]);

    const wrapper = mount(TracingProviderSheet, { props: { provider, editor } });

    expect((wrapper.find("select").element as HTMLSelectElement).value).toBe("full");
  });

  it("saves the selected content mode", async () => {
    const provider = makeProvider({
      public_key: "pk",
      secret_key: "sk",
      send_content: "full",
    });
    const { editor, updateProviderField } = makeEditor(provider, [LANGFUSE]);

    const wrapper = mount(TracingProviderSheet, { props: { provider, editor } });
    await wrapper.find("select").setValue("inherit");

    expect(updateProviderField).toHaveBeenCalledWith("provider-1", "send_content", "inherit");
  });

  it("keeps an explicitly saved mode visible", () => {
    const provider = makeProvider({
      public_key: "pk",
      secret_key: "sk",
      send_content: "none",
    });
    const { editor } = makeEditor(provider, [LANGFUSE]);

    const wrapper = mount(TracingProviderSheet, { props: { provider, editor } });

    expect((wrapper.find("select").element as HTMLSelectElement).value).toBe("none");
  });

  it("renders the fields the selected provider declares", () => {
    const provider = makeProvider({ api_key: "lsv2-key" }, "langsmith");
    const { editor } = makeEditor(provider, [LANGFUSE, OTEL, LANGSMITH]);

    const wrapper = mount(TracingProviderSheet, { props: { provider, editor } });

    expect(fieldInput(wrapper, "langsmith", "api_key").attributes("type")).toBe("password");
    expect((fieldInput(wrapper, "langsmith", "project").element as HTMLInputElement).value).toBe(
      "default"
    );
    // Fields of other providers are not rendered.
    expect(fieldInput(wrapper, "langfuse", "public_key").exists()).toBe(false);
  });

  it("renders a declared boolean as a switch", async () => {
    const provider = makeProvider({ endpoint: "https://collector:4318/v1/traces" }, "otel");
    const { editor, updateProviderField } = makeEditor(provider, [OTEL]);

    const wrapper = mount(TracingProviderSheet, { props: { provider, editor } });
    const insecure = wrapper.find('[aria-label="tracing.fields.otel.insecure.label"]');
    await insecure.trigger("click");

    expect(updateProviderField).toHaveBeenCalledWith("provider-1", "insecure", true);
  });

  it("bounds a declared number field to its metadata range", () => {
    const provider = makeProvider({ endpoint: "https://collector:4318/v1/traces" }, "otel");
    const { editor } = makeEditor(provider, [OTEL]);

    const wrapper = mount(TracingProviderSheet, { props: { provider, editor } });
    const input = fieldInput(wrapper, "otel", "sample_rate");

    // A sampling rate outside 0-1 must not be enterable; the backend rejects it.
    expect(input.attributes("aria-valuemin")).toBe("0");
    expect(input.attributes("aria-valuemax")).toBe("1");
  });

  it("edits a headers map as key/value rows", async () => {
    const provider = makeProvider({ endpoint: "https://collector:4318/v1/traces" }, "otel");
    const { editor, updateProviderField } = makeEditor(provider, [OTEL]);

    const wrapper = mount(TracingProviderSheet, { props: { provider, editor } });
    const addButton = wrapper
      .findAll("button")
      .find((button) => button.text() === "tracing.addHeader");
    await addButton!.trigger("click");

    expect(updateProviderField).toHaveBeenCalledWith("provider-1", "headers", {
      "header-1": "",
    });
  });

  it("writes header values back into the settings object", async () => {
    const provider = makeProvider(
      { endpoint: "https://collector:4318/v1/traces", headers: { Authorization: "Bearer x" } },
      "otel"
    );
    const { editor, updateProviderField } = makeEditor(provider, [OTEL]);

    const wrapper = mount(TracingProviderSheet, { props: { provider, editor } });
    const valueInput = wrapper.find(
      'input[aria-label="tracing.fields.otel.headers.label tracing.headerValue"]'
    );
    await valueInput.setValue("Bearer y");

    expect(updateProviderField).toHaveBeenCalledWith("provider-1", "headers", {
      Authorization: "Bearer y",
    });
  });
});
