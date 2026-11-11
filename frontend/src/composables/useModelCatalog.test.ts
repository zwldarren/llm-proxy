import { effectScope, nextTick } from "vue";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("vue-i18n", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}));

vi.mock("@/stores/models", () => ({
  useModelStore: () => ({ models: [] }),
}));

vi.mock("@/services/api/chat", () => ({
  chatApi: { getModels: vi.fn() },
}));

vi.mock("@/utils/error", () => ({
  getErrorMessage: (error: unknown) => (error instanceof Error ? error.message : String(error)),
}));

import { useModelCatalog } from "@/composables/useModelCatalog";

const STORAGE_KEY = "review-test-model";

describe("useModelCatalog", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("keeps the selected model synchronized across storage changes", async () => {
    const firstScope = effectScope();
    const first = firstScope.run(() =>
      useModelCatalog({ storageKey: STORAGE_KEY, namespace: "chat" })
    )!;
    const secondScope = effectScope();
    const second = secondScope.run(() =>
      useModelCatalog({ storageKey: STORAGE_KEY, namespace: "chat" })
    )!;

    first.selectedModel.value = "alpha";
    await nextTick();
    expect(localStorage.getItem(STORAGE_KEY)).toBe("alpha");

    localStorage.setItem(STORAGE_KEY, "beta");
    window.dispatchEvent(
      new StorageEvent("storage", {
        key: STORAGE_KEY,
        newValue: "beta",
        storageArea: localStorage,
      })
    );
    await nextTick();

    expect(second.selectedModel.value).toBe("beta");
    firstScope.stop();
    secondScope.stop();
  });
});
