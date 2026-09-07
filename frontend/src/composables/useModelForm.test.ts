// frontend/src/composables/useModelForm.test.ts
import { describe, expect, it } from "vitest";
import { nextTick } from "vue";
import { emptyModelForm, useModelForm } from "./useModelForm";
import type { ModelProviderMapping, ModelRead } from "@/types/schemas";

function savedProvider(overrides: Partial<ModelProviderMapping> = {}): ModelProviderMapping {
  return {
    provider_name: "openai",
    provider_model_name: "gpt-4o",
    ...overrides,
  };
}

function savedModel(overrides: Partial<ModelRead> = {}): ModelRead {
  return {
    id: 1,
    name: "gpt-4o",
    providers: [savedProvider()],
    ...overrides,
  };
}

function validDraft() {
  const form = useModelForm();
  form.openForCreate();
  form.form.value.name = "gpt-4o";
  form.form.value.providers[0]!.provider_name = "openai";
  form.form.value.providers[0]!.provider_model_name = "gpt-4o";
  return form;
}

describe("useModelForm capabilities", () => {
  it("maps proxy-bound capabilities onto their supports_* fields", () => {
    const form = useModelForm();
    expect(form.getCapability("vision")).toBe(false);

    form.setCapability("vision", true);
    expect(form.form.value.supports_images).toBe(true);
    expect(form.getCapability("vision")).toBe(true);
  });

  it("maps informational capabilities onto their same-named fields", () => {
    const form = useModelForm();
    form.setCapability("reasoning", true);
    expect(form.form.value.reasoning).toBe(true);
    expect(form.form.value.supports_images).toBe(false);
  });
});

describe("useModelForm null proxies", () => {
  it("proxies empty string to null and back for the description", () => {
    const form = useModelForm();
    expect(form.descriptionModel.value).toBe("");

    form.descriptionModel.value = "Hello";
    expect(form.form.value.description).toBe("Hello");

    form.descriptionModel.value = "";
    expect(form.form.value.description).toBeNull();
  });

  it("uses the none sentinel for an unset status", () => {
    const form = useModelForm();
    expect(form.statusModel.value).toBe("none");

    form.statusModel.value = "beta";
    expect(form.form.value.status).toBe("beta");

    form.statusModel.value = "none";
    expect(form.form.value.status).toBeNull();
  });

  it("trims family and stores null for blank input", () => {
    const form = useModelForm();
    form.familyModel.value = "  claude-sonnet  ";
    expect(form.form.value.family).toBe("claude-sonnet");

    form.familyModel.value = "   ";
    expect(form.form.value.family).toBeNull();
  });
});

describe("useModelForm provider rows", () => {
  it("adds and removes provider rows", () => {
    const form = useModelForm();
    form.openForCreate();
    expect(form.form.value.providers).toHaveLength(1);

    form.addProvider();
    expect(form.form.value.providers).toHaveLength(2);

    form.removeProvider(0);
    expect(form.form.value.providers).toHaveLength(1);
  });

  it("edits a row only after saveProviderEdit commits it", () => {
    const form = useModelForm();
    form.openForCreate();

    form.openProviderEditDialog(0);
    expect(form.showProviderEditDialog.value).toBe(true);

    form.editingProviderData.value.provider_name = "azure";
    expect(form.form.value.providers[0]!.provider_name).toBe("");

    form.saveProviderEdit();
    expect(form.form.value.providers[0]!.provider_name).toBe("azure");
    expect(form.showProviderEditDialog.value).toBe(false);
    expect(form.editingProviderIndex.value).toBeNull();
  });
});

describe("useModelForm seeding", () => {
  it("openForCreate seeds one blank provider row", () => {
    const form = useModelForm();
    form.openForCreate();
    expect(form.form.value).toEqual({
      ...emptyModelForm(),
      providers: [
        expect.objectContaining({ provider_name: "", provider_model_name: "", priority: 0 }),
      ],
    });
  });

  it("openForEdit sorts providers by priority descending and null-fills costs", () => {
    const form = useModelForm();
    form.openForEdit(
      savedModel({
        providers: [
          savedProvider({ provider_name: "low", priority: 1 }),
          savedProvider({ provider_name: "high", priority: 10, input_cost_per_1m: 0.5 }),
        ],
      })
    );

    expect(form.form.value.providers.map((p) => p.provider_name)).toEqual(["high", "low"]);
    expect(form.form.value.providers[0]!.input_cost_per_1m).toBe(0.5);
    expect(form.form.value.providers[1]!.input_cost_per_1m).toBeNull();
    expect(form.form.value.providers[1]!.parameter_overrides).toEqual({});
  });

  it("openForEdit keeps the model parameter overrides for the builder", () => {
    const form = useModelForm();
    form.openForEdit(savedModel({ parameter_overrides: { temperature: 0.7 } }));
    expect(form.parameterOverrides.value).toEqual({ temperature: 0.7 });
  });
});

describe("useModelForm cross-field rules", () => {
  it("clears the routing profile when auto-eligibility is switched off", async () => {
    const form = useModelForm();
    form.openForCreate();
    form.form.value.auto_eligible = true;
    await nextTick();
    form.form.value.quality_tier = "PREMIUM";
    form.form.value.routing_assignments = ["fast"];

    form.form.value.auto_eligible = false;
    await nextTick();

    expect(form.form.value.quality_tier).toBeNull();
    expect(form.form.value.routing_assignments).toBeNull();
  });

  it("resets the icon preview failure when the name or icon changes", async () => {
    const form = useModelForm();
    form.openForCreate();
    form.iconPreviewFailed.value = true;

    form.form.value.icon_url = "https://example.com/icon.svg";
    await nextTick();
    expect(form.iconPreviewFailed.value).toBe(false);
  });
});

describe("useModelForm validate", () => {
  it("rejects a blank name first", () => {
    const form = useModelForm();
    form.openForCreate();
    expect(form.validate()).toBe("models.nameRequired");
  });

  it("rejects a draft without providers", () => {
    const form = useModelForm();
    form.openForCreate();
    form.form.value.name = "gpt-4o";
    form.removeProvider(0);
    expect(form.validate()).toBe("models.atLeastOneProvider");
  });

  it("rejects provider rows missing a provider or model name", () => {
    const form = useModelForm();
    form.openForCreate();
    form.form.value.name = "gpt-4o";
    expect(form.validate()).toBe("models.providerNameRequired");

    form.form.value.providers[0]!.provider_name = "openai";
    form.form.value.providers[0]!.provider_model_name = "   ";
    expect(form.validate()).toBe("models.providerModelNameRequired");
  });

  it("requires a quality tier for auto-eligible models", () => {
    const form = validDraft();
    form.form.value.auto_eligible = true;
    expect(form.validate()).toBe("models.qualityTierRequired");

    form.form.value.quality_tier = "BALANCED";
    expect(form.validate()).toBeNull();
  });

  it("accepts a complete draft", () => {
    expect(validDraft().validate()).toBeNull();
  });
});

describe("useModelForm buildPayload", () => {
  it("trims the model name", () => {
    const form = validDraft();
    form.form.value.name = "  gpt-4o  ";
    expect(form.buildPayload().name).toBe("gpt-4o");
  });

  it("clears the routing profile when the draft is not auto-eligible", () => {
    // A stored model can carry a tier while auto_eligible is false; the
    // watcher never fires in that case, so the payload cleanup must.
    const form = useModelForm();
    form.openForEdit(savedModel({ auto_eligible: false, quality_tier: "PREMIUM" }));

    const payload = form.buildPayload();
    expect(payload.quality_tier).toBeNull();
    expect(payload.routing_assignments).toBeNull();
  });

  it("sends null parameter overrides when none are configured", () => {
    const form = validDraft();
    expect(form.buildPayload().parameter_overrides).toBeNull();

    form.parameterOverrides.value = { temperature: 0.7 };
    expect(form.buildPayload().parameter_overrides).toEqual({ temperature: 0.7 });
  });

  it("null-fills unset provider cost fields", () => {
    const form = validDraft();
    const payload = form.buildPayload();
    expect(payload.providers[0]).toEqual(
      expect.objectContaining({
        provider_name: "openai",
        provider_model_name: "gpt-4o",
        input_cost_per_1m: null,
        web_search_cost_per_1k: null,
        parameter_overrides: {},
      })
    );
  });
});
