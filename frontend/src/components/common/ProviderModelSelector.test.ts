import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("vue-i18n", () => ({ useI18n: () => ({ t: (k: string) => k }) }));
vi.mock("@lucide/vue", () => {
  const Stub = () => null;
  return {
    AlertCircle: Stub,
    ChevronDown: Stub,
    Loader2: Stub,
    RefreshCw: Stub,
    Search: Stub,
  };
});
// The popover is a portal that only renders while open; the stub keeps the
// list in the tree so a test can assert on the rows.
vi.mock("@/components/ui/popover", () => {
  const slot = "<div><slot /></div>";
  return {
    Popover: {
      name: "Popover",
      props: { open: Boolean },
      emits: ["update:open"],
      template: slot,
    },
    PopoverContent: { name: "PopoverContent", template: slot },
    PopoverTrigger: { name: "PopoverTrigger", template: slot },
  };
});
vi.mock("@/components/ui/tooltip", () => {
  const Stub = { template: "<div><slot /></div>" };
  return { Tooltip: Stub, TooltipContent: Stub, TooltipTrigger: Stub };
});

vi.mock("@/services/api/config", () => ({
  configApi: { getProviderModels: vi.fn() },
}));

import { configApi } from "@/services/api/config";
import ProviderModelSelector from "./ProviderModelSelector.vue";

const models = [
  {
    id: "anthropic/claude-sonnet-4.5",
    name: "Claude Sonnet 4.5",
    description: "Frontier model",
    owned_by: null,
    context_length: 200000,
    architecture: { input_modalities: ["text", "image"], output_modalities: ["text"] },
    supported_parameters: ["tools", "reasoning"],
    pricing: { prompt: 0.000003, completion: 0.000015, request: null },
  },
  {
    id: "google/veo-3.1",
    name: "Veo 3.1",
    description: null,
    owned_by: null,
    context_length: null,
    architecture: { input_modalities: ["text"], output_modalities: ["video"] },
    supported_parameters: [],
    pricing: null,
  },
];

beforeEach(() => {
  vi.mocked(configApi.getProviderModels).mockReset();
});

async function mountOpen() {
  vi.mocked(configApi.getProviderModels).mockResolvedValue({
    provider_name: "openrouter",
    provider_type: "openrouter",
    models,
  });
  const wrapper = mount(ProviderModelSelector, {
    props: { providerName: "openrouter", modelValue: "" },
  });
  wrapper.findComponent({ name: "Popover" }).vm.$emit("update:open", true);
  await flushPromises();
  return wrapper;
}

describe("ProviderModelSelector", () => {
  it("renders the metadata the provider listing now carries", async () => {
    const wrapper = await mountOpen();

    expect(configApi.getProviderModels).toHaveBeenCalledWith("openrouter");

    const rows = wrapper.findAll('[role="option"]');
    expect(rows).toHaveLength(2);

    const text = rows[0].text();
    expect(text).toContain("200K"); // context_length, formatted
    expect(text).toContain("$3.00"); // 0.000003/token -> per 1M
    expect(text).toContain("$15.00");

    // A non-text model shows what it produces instead of a price per token.
    const videoRow = rows[1].text();
    expect(videoRow).toContain("text→video");
    expect(videoRow).not.toContain("$");
  });

  it("finds models by modality, not just by name", async () => {
    const wrapper = await mountOpen();

    await wrapper.find("input").setValue("video");
    await flushPromises();

    const rows = wrapper.findAll('[role="option"]');
    expect(rows).toHaveLength(1);
    expect(rows[0].text()).toContain("google/veo-3.1");
  });

  it("emits the model id when a row is chosen", async () => {
    const wrapper = await mountOpen();

    await wrapper.findAll('[role="option"]')[0].trigger("click");

    expect(wrapper.emitted("update:modelValue")).toEqual([["anthropic/claude-sonnet-4.5"]]);
  });
});
