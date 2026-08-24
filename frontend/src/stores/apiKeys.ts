import { defineStore } from "pinia";
import { ref } from "vue";
import {
  apiKeysApi,
  type ApiKeyCreate,
  type ApiKeyRead,
  type ApiKeyResponse,
  type ApiKeySpendSummary,
  type ApiKeyUpdate,
} from "@/services/api/apiKeys";
import { createResourceStore } from "@/composables/useResourceStore";

export const useApiKeyStore = defineStore("apiKeys", () => {
  const store = createResourceStore<ApiKeyRead, ApiKeyCreate, ApiKeyUpdate>({
    name: "API key",
    fetchFn: () => apiKeysApi.getApiKeys(),
    createFn: (data) => apiKeysApi.createApiKey(data) as unknown as Promise<ApiKeyRead>,
    updateFn: (name, data) => apiKeysApi.updateApiKey(name, data),
    deleteFn: (name) => apiKeysApi.deleteApiKey(name),
  });

  // Per-key spend summary, keyed by key name for O(1) list-cell lookups.
  const spendByKey = ref<Record<string, ApiKeySpendSummary>>({});
  const spendLoading = ref(false);
  // Tracks whether the summary was fetched at least once: an empty result is
  // a valid cache state (no keys / no spend yet), not a reason to refetch.
  const spendLoaded = ref(false);
  // Sequence guard: a forced refresh may overlap an in-flight fetch, and only
  // the latest request is allowed to commit its result.
  let spendFetchSeq = 0;

  // createApiKey returns ApiKeyResponse (includes the generated key), not ApiKeyRead.
  // Omitted models/mcpServers mean "no restriction" (allow all); an explicit
  // empty array would mean deny-all, so undefined is the permissive default.
  async function createApiKey(data: ApiKeyCreate): Promise<ApiKeyResponse> {
    try {
      const res = await apiKeysApi.createApiKey(data);
      // Refresh the spend summary too: a key created with a budget should
      // show its (empty) budget window immediately, not after a reload.
      await Promise.all([store.fetchItems(true), fetchSpendSummary(true)]);
      return res;
    } catch (err) {
      console.error("Failed to create API key:", err);
      throw err;
    }
  }

  async function fetchSpendSummary(force = false): Promise<void> {
    // A forced refresh is always honored, even while another fetch is in
    // flight; only redundant non-forced calls are skipped.
    if (spendLoading.value && !force) return;
    if (!force && spendLoaded.value) return;
    const seq = ++spendFetchSeq;
    spendLoading.value = true;
    try {
      const rows = await apiKeysApi.getSpendSummary();
      // Discard a stale response when a newer fetch has already been issued.
      if (seq !== spendFetchSeq) return;
      spendByKey.value = Object.fromEntries(rows.map((row) => [row.name, row]));
      spendLoaded.value = true;
    } catch (err) {
      // Spend is auxiliary display data: a failure must not break key listing.
      console.error("Failed to fetch API key spend summary:", err);
    } finally {
      if (seq === spendFetchSeq) spendLoading.value = false;
    }
  }

  async function resetBudget(name: string): Promise<void> {
    await apiKeysApi.resetBudget(name);
    await Promise.all([store.fetchItems(true), fetchSpendSummary(true)]);
  }

  return {
    apiKeys: store.items,
    loading: store.loading,
    loaded: store.loaded,
    ready: store.ready,
    spendByKey,
    spendLoading,
    fetchApiKeys: store.fetchItems,
    prefetch: store.prefetch,
    createApiKey,
    updateApiKey: store.updateItem,
    deleteApiKey: store.deleteItem,
    fetchSpendSummary,
    resetBudget,
    reset: store.reset,
  };
});
