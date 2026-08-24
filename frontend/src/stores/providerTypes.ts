import { computed, ref } from "vue";
import { defineStore } from "pinia";
import { useI18n } from "vue-i18n";
import { toast } from "vue-sonner";
import { configApi } from "@/services/api/config";
import type { ProviderTypeInfo } from "@/types/schemas";
import { getErrorMessage } from "@/utils/error";

/**
 * Available provider types fetched from the backend catalog.
 *
 * The catalog is derived from the adapter registry server-side, so adding a
 * provider adapter is a backend-only change — no per-provider frontend list.
 * The fetch is single-flight (concurrent `ensureLoaded()` callers share one
 * request) and is retried on failure: `ready` stays false, so the next call
 * tries again. Failures surface a toast (the list stays empty so the UI
 * never shows stale data).
 */
export const useProviderTypesStore = defineStore("providerTypes", () => {
  const { t } = useI18n();
  const types = ref<ProviderTypeInfo[]>([]);
  const loading = ref(false);
  const ready = ref(false);

  let fetchPromise: Promise<boolean> | null = null;

  const byType = computed(() => new Map(types.value.map((t) => [t.type, t])));

  function getType(type: string): ProviderTypeInfo | undefined {
    return byType.value.get(type);
  }

  /**
   * Load the provider-type catalog once. Resolves true when the catalog is
   * available (from cache or a fresh fetch), false on fetch failure — callers
   * may surface the failure (toast) and show an empty list.
   */
  async function ensureLoaded(): Promise<boolean> {
    if (ready.value) return true;
    if (fetchPromise) return fetchPromise;

    loading.value = true;
    fetchPromise = (async () => {
      try {
        types.value = await configApi.getProviderTypes();
        ready.value = true;
        return true;
      } catch (e) {
        // Keep the list empty so the UI never shows stale data; a later
        // ensureLoaded() call retries. Surface the failure to the user.
        console.error("Failed to load provider types", e);
        toast.error(t("errors.providerTypesLoadFailed"), {
          description: getErrorMessage(e),
        });
        return false;
      } finally {
        loading.value = false;
        fetchPromise = null;
      }
    })();
    return fetchPromise;
  }

  function reset() {
    types.value = [];
    loading.value = false;
    ready.value = false;
    fetchPromise = null;
  }

  return { types, loading, ready, byType, getType, ensureLoaded, reset };
});
