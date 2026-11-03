import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import i18n from "@/i18n";
import ModelsSyncDialog from "./ModelsSyncDialog.vue";

const pricingPreview = vi.fn();
const metadataPreview = vi.fn();
const applyPricing = vi.fn();
const applyMetadata = vi.fn();

vi.mock("@/services/api/config", () => ({
  pricingApi: {
    fetchPreview: () => pricingPreview(),
    applyPricing: (payload: unknown) => applyPricing(payload),
  },
  metadataApi: {
    fetchPreview: () => metadataPreview(),
    applyMetadata: (payload: unknown) => applyMetadata(payload),
  },
}));

vi.mock("vue-sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

/** One new mapping and one whose stored rate moves (the "Overwrite" row). */
const PRICING_PREVIEW = {
  success: true,
  dry_run: true,
  total_models: 2,
  total_provider_mappings: 2,
  updated_count: 2,
  skipped_count: 0,
  unchanged_count: 0,
  results: [
    {
      mapping_id: 1,
      model_name: "gpt-4o-mini",
      provider_model_name: "gpt-4o-mini",
      provider: "openai",
      updated: false,
      message: "",
      selected_source: "openai",
      new_input_cost: 0.15,
      new_output_cost: 0.6,
      available_sources: [
        {
          source: "openai",
          input_cost_per_1m: 0.15,
          output_cost_per_1m: 0.6,
          tiers: [],
          service_tier_pricing: {},
        },
      ],
    },
    {
      mapping_id: 2,
      model_name: "deepseek-chat",
      provider_model_name: "deepseek-chat",
      provider: "deepseek",
      old_input_cost: 0.27,
      old_output_cost: 1.1,
      new_input_cost: 0.28,
      new_output_cost: 1.12,
      updated: false,
      message: "",
      selected_source: "deepseek",
      available_sources: [
        {
          source: "deepseek",
          input_cost_per_1m: 0.28,
          output_cost_per_1m: 1.12,
          tiers: [],
          service_tier_pricing: {},
        },
      ],
    },
  ],
};

const emptyMeta = (over: Record<string, unknown>) => ({
  source: "",
  supports_images: null,
  attachment: null,
  reasoning: null,
  tool_call: null,
  structured_output: null,
  temperature: null,
  open_weights: null,
  status: null,
  family: null,
  knowledge: null,
  release_date: null,
  context_length: null,
  max_output_tokens: null,
  ...over,
});

/** One gap-fill (auto-selected) and one model models.dev does not know. */
const METADATA_PREVIEW = {
  success: true,
  total_models: 2,
  changed_count: 1,
  unchanged_count: 0,
  nodata_count: 1,
  results: [
    {
      model_name: "gpt-4o",
      old: emptyMeta({ source: "", family: "gpt" }),
      available_sources: [
        emptyMeta({ source: "openai", family: "gpt", max_output_tokens: 16384, tool_call: true }),
      ],
      selected_source: "openai",
      message: "",
    },
    {
      model_name: "qwen3",
      old: emptyMeta({ source: "" }),
      available_sources: [],
      selected_source: null,
      message: "",
    },
  ],
};

const stubs = {
  // Keep the dialog body in the render tree instead of a body teleport.
  Dialog: { template: "<div><slot /></div>" },
  DialogContent: { template: "<div><slot /></div>" },
  DialogHeader: { template: "<div><slot /></div>" },
  DialogTitle: { template: "<h2><slot /></h2>" },
  DialogDescription: { template: "<p><slot /></p>" },
};

async function mountDialog() {
  const wrapper = mount(ModelsSyncDialog, {
    props: { open: true },
    global: { plugins: [i18n], stubs },
  });
  await flushPromises();
  return wrapper;
}

type Wrapper = Awaited<ReturnType<typeof mountDialog>>;

function buttonByText(wrapper: Wrapper, text: string) {
  const button = wrapper.findAll("button").find((b) => b.text().includes(text));
  if (!button) throw new Error(`button not found: ${text}`);
  return button;
}

beforeEach(() => {
  vi.clearAllMocks();
  pricingPreview.mockResolvedValue(PRICING_PREVIEW);
  metadataPreview.mockResolvedValue(METADATA_PREVIEW);
  applyPricing.mockResolvedValue({ success: true, applied_count: 1, failed_count: 0, results: [] });
  applyMetadata.mockResolvedValue({
    success: true,
    applied_count: 1,
    failed_count: 0,
    results: [],
  });
});

describe("ModelsSyncDialog", () => {
  it("lands on the review with a count per section and one write summary", async () => {
    const wrapper = await mountDialog();

    expect(pricingPreview).toHaveBeenCalledOnce();
    expect(metadataPreview).toHaveBeenCalledOnce();
    // Both sections are loaded up front, so neither tab is a blind click.
    const tabs = wrapper.findAll('[role="tab"]');
    expect(tabs.map((tab) => tab.text())).toEqual(["Pricing 2", "Capabilities 1"]);
    // Selected by default: the new mapping and the pure gap-fill.
    expect(wrapper.text()).toContain("2 rows will be written");
    expect(buttonByText(wrapper, "Apply 2 updates").exists()).toBe(true);
  });

  it("says why a row that overwrites is left unchecked", async () => {
    const wrapper = await mountDialog();

    expect(wrapper.text()).toContain("1 row left unchecked (marked Overwrite)");
    // The row itself is badged with the effect of writing it.
    expect(wrapper.text()).toContain("Overwrite");
  });

  it("reports a failed section without blocking the other one", async () => {
    metadataPreview.mockRejectedValue(new Error("upstream 503"));
    const wrapper = await mountDialog();

    expect(wrapper.text()).toContain("Capabilities failed to load and is not included");
    expect(wrapper.text()).toContain("1 row will be written");
  });

  it("writes both sections in one confirmation", async () => {
    const wrapper = await mountDialog();

    await buttonByText(wrapper, "Apply 2 updates").trigger("click");
    // The bottom bar expands in place: summary, the rows by name, then confirm.
    expect(wrapper.text()).toContain("1 mapping repriced · 1 model updated");
    expect(wrapper.text()).toContain("gpt-4o-mini · openai");

    await buttonByText(wrapper, "Confirm and apply").trigger("click");
    await flushPromises();

    expect(applyPricing).toHaveBeenCalledOnce();
    expect(applyPricing.mock.calls[0]![0]).toEqual({
      updates: [{ mapping_id: 1, input_cost_per_1m: 0.15, output_cost_per_1m: 0.6 }],
    });
    expect(applyMetadata).toHaveBeenCalledOnce();
    expect(applyMetadata.mock.calls[0]![0]).toEqual({
      updates: [{ model_name: "gpt-4o", max_output_tokens: 16384, tool_call: true }],
    });
    expect(wrapper.emitted("applied")).toBeTruthy();
    expect(wrapper.emitted("update:open")?.at(-1)?.[0]).toBe(false);
  });

  it("keeps the confirmation open and reports rows when a write fails", async () => {
    applyMetadata.mockRejectedValue(new Error("boom"));
    const wrapper = await mountDialog();

    await buttonByText(wrapper, "Apply 2 updates").trigger("click");
    await buttonByText(wrapper, "Confirm and apply").trigger("click");
    await flushPromises();

    expect(wrapper.emitted("update:open")).toBeFalsy();
    expect(wrapper.text()).toContain("1 metadata updates failed");
  });

  it("skips a section with nothing selected", async () => {
    metadataPreview.mockResolvedValue({
      ...METADATA_PREVIEW,
      results: [METADATA_PREVIEW.results[1]],
    });
    const wrapper = await mountDialog();

    await buttonByText(wrapper, "Apply 1 update").trigger("click");
    await buttonByText(wrapper, "Confirm and apply").trigger("click");
    await flushPromises();

    expect(applyPricing).toHaveBeenCalledOnce();
    expect(applyMetadata).not.toHaveBeenCalled();
  });
});
