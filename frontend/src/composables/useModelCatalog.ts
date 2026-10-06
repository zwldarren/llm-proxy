import { computed, ref, watch, type ComputedRef, type Ref } from "vue";
import { useI18n } from "vue-i18n";
import { toast } from "vue-sonner";
import { chatApi } from "@/services/api/chat";
import { useModelStore } from "@/stores/models";
import { getErrorMessage } from "@/utils/error";

/** A model as the playground pickers show it: id plus provider for its icon. */
export interface CatalogModel {
  id: string;
  provider: string;
}

interface ModelCatalogOptions {
  /** This picker's session API key; an empty key must never reach `/v1/models`. */
  apiKey: Ref<string>;
  /** localStorage key holding this picker's last selection. */
  storageKey: string;
  /** i18n namespace holding the apiKeyRequired/retryModels/noModelsAvailable copy. */
  namespace: "chat" | "images";
}

export interface ModelCatalog {
  models: Ref<CatalogModel[]>;
  selectedModel: Ref<string | null>;
  isLoadingModels: Ref<boolean>;
  /** Why the list is missing, if it is. An empty catalog is not a failure. */
  modelsError: Ref<string | null>;
  /** Placeholder copy: failed load vs. empty catalog. Both states are retryable. */
  modelsPlaceholderLabel: ComputedRef<string>;
  /** Loads `/v1/models`, reporting success so callers can stay silent or not. */
  loadModels: () => Promise<boolean>;
  /** User-initiated retry — same load, but failure is reported out loud. */
  refreshModels: () => Promise<void>;
}

/**
 * The `/v1/models` catalog shared by the Chat and Images model pickers.
 *
 * A failed load used to blank the list and leave an inert placeholder, so the
 * page stayed stuck until a full reload. Here a failure keeps whatever is
 * already on screen, records why, and leaves the picker able to retry in place.
 * The catalog also reloads itself when the session key changes (a different
 * session must not be served from this cache) or when the model config store
 * updates (it supplies provider names for models that omit them).
 */
export function useModelCatalog(options: ModelCatalogOptions): ModelCatalog {
  const { t } = useI18n();
  const modelStore = useModelStore();

  const models = ref<CatalogModel[]>([]);
  const selectedModel = ref<string | null>(localStorage.getItem(options.storageKey));
  const isLoadingModels = ref(false);
  const modelsError = ref<string | null>(null);
  /** Ticket of the newest in-flight call; an older caller must not write. */
  let requestTicket = 0;

  watch(selectedModel, (value) => {
    if (value) localStorage.setItem(options.storageKey, value);
    else localStorage.removeItem(options.storageKey);
  });

  /** Provider name from the model config, for models that omit it. */
  const providerFromConfig = (id: string): string =>
    modelStore.models.find((cm) => cm.name === id)?.providers?.[0]?.provider_name || "";

  const loadModels = async (): Promise<boolean> => {
    const key = options.apiKey.value.trim();
    if (!key) {
      // The proxy authenticates /v1/models with the session API key and rejects
      // an empty Bearer as an *invalid key*: it counts toward the per-IP lockout
      // and its 401 tears the session down. Surface the state instead of firing.
      modelsError.value = t(`${options.namespace}.apiKeyRequired`);
      models.value = [];
      selectedModel.value = null;
      return false;
    }

    const ticket = ++requestTicket;
    isLoadingModels.value = true;
    modelsError.value = null;
    try {
      const res = await chatApi.getModels(key);
      if (ticket !== requestTicket) return false;

      models.value = (res.data ?? []).map((m) => ({
        id: m.id,
        provider: m.provider || providerFromConfig(m.id),
      }));

      // Restore the saved model when it still exists, otherwise take the first.
      const saved = localStorage.getItem(options.storageKey);
      selectedModel.value =
        saved && models.value.some((m) => m.id === saved) ? saved : (models.value[0]?.id ?? null);
      return true;
    } catch (error) {
      if (ticket !== requestTicket) return false;
      console.error("Failed to load models:", error);
      modelsError.value = getErrorMessage(error);
      // A failed refresh must not take a working selector away from a user who
      // is mid-conversation: keep the list already on screen and only clear the
      // selection when there is nothing left to select.
      if (models.value.length === 0) selectedModel.value = null;
      return false;
    } finally {
      if (ticket === requestTicket) isLoadingModels.value = false;
    }
  };

  const refreshModels = async () => {
    if (isLoadingModels.value) return;
    if (!(await loadModels())) {
      toast.error(t(`${options.namespace}.modelsLoadFailed`), {
        description: modelsError.value ?? undefined,
      });
    }
  };

  // The key is restored from storage and can legitimately be absent on first
  // render: wait for it, and reload when it changes (a new session).
  watch(options.apiKey, (key, previous) => {
    if (!key.trim()) return;
    if (previous && previous !== key) void loadModels();
    else if (models.value.length === 0 && !isLoadingModels.value) void loadModels();
  });

  watch(
    () => modelStore.models,
    () => void loadModels()
  );

  /** Header placeholder copy: why the picker is not there. */
  const modelsPlaceholderLabel = computed(() =>
    modelsError.value
      ? t(`${options.namespace}.retryModels`)
      : t(`${options.namespace}.noModelsAvailable`)
  );

  return {
    models,
    selectedModel,
    isLoadingModels,
    modelsError,
    modelsPlaceholderLabel,
    loadModels,
    refreshModels,
  };
}
