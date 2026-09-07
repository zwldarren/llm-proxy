// frontend/src/stores/apiKeys.test.ts
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import type { ApiKeySpendSummary } from "@/services/api/apiKeys";

vi.mock("@/services/api/apiKeys", () => ({
  apiKeysApi: {
    getApiKeys: vi.fn(),
    createApiKey: vi.fn(),
    updateApiKey: vi.fn(),
    deleteApiKey: vi.fn(),
    getSpendSummary: vi.fn(),
    resetBudget: vi.fn(),
  },
}));

import { apiKeysApi } from "@/services/api/apiKeys";
import { useApiKeyStore } from "./apiKeys";

const spend = (name: string): ApiKeySpendSummary => ({
  name,
  total_spend_usd: 1,
  total_requests: 1,
  period_spend_usd: null,
  period_start: null,
  budget_usd: null,
  budget_period: null,
  budget_reset_day: null,
});

beforeEach(() => {
  setActivePinia(createPinia());
  vi.mocked(apiKeysApi.getSpendSummary).mockReset();
});

describe("useApiKeyStore reset", () => {
  it("clears the per-key spend cache so the next session refetches", async () => {
    vi.mocked(apiKeysApi.getSpendSummary).mockResolvedValue([spend("key-a")]);
    const store = useApiKeyStore();

    await store.fetchSpendSummary();
    expect(store.spendByKey["key-a"]).toBeDefined();

    store.reset();
    expect(store.spendByKey).toEqual({});

    // A non-forced fetch after reset must hit the network again — otherwise
    // the next user on this browser sees the previous user's spend figures.
    await store.fetchSpendSummary();
    expect(apiKeysApi.getSpendSummary).toHaveBeenCalledTimes(2);
  });
});
