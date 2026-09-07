// frontend/src/composables/useModelTable.test.ts
import { describe, expect, it } from "vitest";
import { ref } from "vue";
import { useModelTable } from "./useModelTable";
import type { ModelProviderMapping, ModelRead } from "@/types/schemas";

function provider(name: string, cost: number | null = null): ModelProviderMapping {
  return {
    provider_name: name,
    provider_model_name: "m",
    input_cost_per_1m: cost,
  };
}

function model(name: string, overrides: Partial<ModelRead> = {}): ModelRead {
  return { id: 1, name, providers: [], ...overrides };
}

describe("useModelTable", () => {
  it("filters by search query across name and description", () => {
    const models = ref([
      model("gpt-4o", { description: "Flagship" }),
      model("claude-opus", { description: null }),
    ]);
    const table = useModelTable(models);

    table.searchQuery.value = "flagship";
    expect(table.filteredAndSortedModels.value.map((m) => m.name)).toEqual(["gpt-4o"]);
  });

  it("lists unique sorted provider names and filters by provider", () => {
    const models = ref([
      model("a", { providers: [provider("openai"), provider("azure")] }),
      model("b", { providers: [provider("openai")] }),
      model("c", { providers: [provider("anthropic")] }),
    ]);
    const table = useModelTable(models);

    expect(table.availableProviders.value).toEqual(["anthropic", "azure", "openai"]);

    table.handleProviderFilter("openai");
    expect(table.filteredAndSortedModels.value.map((m) => m.name)).toEqual(["a", "b"]);

    // Clicking the active chip again clears the filter.
    table.handleProviderFilter("openai");
    expect(table.filteredAndSortedModels.value).toHaveLength(3);
  });

  it("sorts a cost column by the shared pricing rule, unpriced models last", () => {
    const models = ref([
      model("unpriced"),
      model("wide", {
        input_cost_per_1m: 1,
        providers: [provider("openai", 100)],
      }),
      model("tight", { input_cost_per_1m: 1 }),
      model("pricey", { input_cost_per_1m: 2 }),
    ]);
    const table = useModelTable(models);

    table.onSort("input_cost");
    expect(table.filteredAndSortedModels.value.map((m) => m.name)).toEqual([
      "tight",
      "wide",
      "pricey",
      "unpriced",
    ]);

    // Toggling the same column flips direction but keeps unpriced last.
    table.onSort("input_cost");
    expect(table.filteredAndSortedModels.value.map((m) => m.name)).toEqual([
      "pricey",
      "wide",
      "tight",
      "unpriced",
    ]);
  });

  it("resets direction to asc when switching columns", () => {
    const models = ref([model("b"), model("a")]);
    const table = useModelTable(models);

    table.onSort("name");
    expect(table.sortDir.value).toBe("desc");
    table.onSort("context_length");
    expect(table.sortField.value).toBe("context_length");
    expect(table.sortDir.value).toBe("asc");
  });

  it("clearFilters restores search, provider filter, and default sort", () => {
    const models = ref([model("b"), model("a", { providers: [provider("openai")] })]);
    const table = useModelTable(models);

    table.searchQuery.value = "b";
    table.handleProviderFilter("openai");
    table.onSort("input_cost");
    table.clearFilters();

    expect(table.searchQuery.value).toBe("");
    expect(table.selectedProviderFilter.value).toBe("");
    expect(table.sortField.value).toBe("name");
    expect(table.sortDir.value).toBe("asc");
    expect(table.filteredAndSortedModels.value.map((m) => m.name)).toEqual(["a", "b"]);
  });
});
