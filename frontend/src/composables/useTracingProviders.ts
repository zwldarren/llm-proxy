import { ref } from "vue";
import type { Ref } from "vue";
import { useI18n } from "vue-i18n";
import { meTracingApi } from "@/services/api/config";
import type { TracingConfig, TracingProvider, TracingProviderDetails } from "@/types/schemas";

export interface TracingProviderEditor {
  /** Supported provider types fetched from the API. */
  providerTypes: Ref<TracingProviderDetails[]>;
  /** Whether provider types are being loaded. */
  loadingProviderTypes: Ref<boolean>;
  /** Load supported provider types from the backend. */
  loadProviderTypes: (api?: typeof meTracingApi) => Promise<void>;
  /** Normalize providers and advance local ID counters from an API response. */
  prepareInitialProviders: (providers: TracingProvider[]) => TracingProvider[];
  /** Add a new provider card. */
  addProvider: (providerType?: string) => void;
  /** Remove a provider by local ID. */
  removeProvider: (id: string) => void;
  updateProviderName: (id: string, name: string) => void;
  updateProviderEnabled: (id: string, enabled: boolean) => void;
  getProviderFieldValue: (provider: TracingProvider, fieldName: string) => unknown;
  updateProviderField: (providerId: string, fieldName: string, value: unknown) => void;
  /** Find a provider by local ID. */
  getProviderById: (id: string) => TracingProvider | undefined;
  /** Human-readable label for a provider type. */
  providerTypeLabel: (providerType: string) => string;
  /** Build a payload-ready TracingConfig without frontend-only fields. */
  buildSavePayload: (config: TracingConfig) => TracingConfig;
}

/**
 * Composable for editing a list of tracing providers.
 *
 * Operates on the reactive `TracingConfig` state (usually owned by
 * `useSettingAutoSave`). It handles local stable IDs and discovery of
 * supported provider types from the backend.
 */
export function useTracingProviderEditor(state: Ref<TracingConfig>): TracingProviderEditor {
  const { t } = useI18n();

  let providerIdCounter = 1;

  const providerTypes = ref<TracingProviderDetails[]>([]);
  const loadingProviderTypes = ref(false);

  // ── ID helpers ──────────────────────────────────────────────────────

  function generateProviderId(): string {
    return `provider-${providerIdCounter++}`;
  }

  function advanceCounters(providers: TracingProvider[]): void {
    for (const p of providers) {
      const match = p.id?.match(/^provider-(\d+)$/);
      if (match && match[1] !== undefined) {
        const num = parseInt(match[1], 10);
        if (num >= providerIdCounter) providerIdCounter = num + 1;
      }
    }
  }

  // ── Normalization ─────────────────────────────────────────────────

  function normalizeProvider(provider: TracingProvider): TracingProvider {
    return {
      ...provider,
      id: provider.id || generateProviderId(),
    };
  }

  function prepareInitialProviders(providers: TracingProvider[]): TracingProvider[] {
    const normalized = providers.map(normalizeProvider);
    advanceCounters(normalized);
    return normalized;
  }

  // ── Provider CRUD ───────────────────────────────────────────────────

  function updateProviderById(
    id: string,
    updater: (provider: TracingProvider) => TracingProvider
  ): void {
    state.value.providers = state.value.providers.map((p) => (p.id === id ? updater(p) : p));
  }

  function addProvider(providerType = "langfuse"): void {
    const typeInfo = providerTypes.value.find((p) => p.name === providerType);
    let baseName: string;
    if (typeInfo?.name === "langfuse") {
      baseName = t("tracing.providerType.langfuse");
    } else {
      baseName = typeInfo?.name || providerType;
    }

    const settings: Record<string, unknown> = {};
    if (typeInfo?.fields) {
      for (const field of typeInfo.fields) {
        if (field.default !== undefined && field.default !== null) {
          settings[field.name] = field.default;
        }
      }
    }

    state.value.providers = [
      ...state.value.providers,
      {
        id: generateProviderId(),
        provider: providerType,
        name: `${baseName} ${providerIdCounter - 1}`,
        enabled: true,
        settings,
      },
    ];
  }

  function removeProvider(id: string): void {
    state.value.providers = state.value.providers.filter((p) => p.id !== id);
  }

  function updateProviderName(id: string, name: string): void {
    updateProviderById(id, (p) => ({ ...p, name }));
  }

  function updateProviderEnabled(id: string, enabled: boolean): void {
    updateProviderById(id, (p) => ({ ...p, enabled }));
  }

  // ── Generic field accessors ─────────────────────────────────────────

  function getProviderFieldValue(provider: TracingProvider, fieldName: string): unknown {
    return provider.settings[fieldName];
  }

  function updateProviderField(providerId: string, fieldName: string, value: unknown): void {
    updateProviderById(providerId, (p) => {
      const settings = { ...p.settings };
      if (value === "" || value === null || value === undefined) {
        delete settings[fieldName];
      } else {
        settings[fieldName] = value;
      }
      return { ...p, settings };
    });
  }

  function getProviderById(id: string): TracingProvider | undefined {
    return state.value.providers.find((p) => p.id === id);
  }

  function providerTypeLabel(providerType: string): string {
    if (providerType === "langfuse") return t("tracing.providerType.langfuse");
    const found = providerTypes.value.find((p) => p.name === providerType);
    return found?.name || providerType;
  }

  // ── API helpers ───────────────────────────────────────────────────

  async function loadProviderTypes(api: typeof meTracingApi = meTracingApi): Promise<void> {
    try {
      loadingProviderTypes.value = true;
      const res = await api.getProviders();
      // Only show langfuse provider (audit_log is internal)
      providerTypes.value = res.providers.filter(
        (p) => p.name && p.name !== "audit_log" && p.name !== "otlp"
      );
    } catch (e) {
      console.error("Failed to load tracing provider types", e);
    } finally {
      loadingProviderTypes.value = false;
    }
  }

  /**
   * Build the payload for saving, stripping frontend-only fields.
   */
  function buildSavePayload(config: TracingConfig): TracingConfig {
    return {
      enabled: config.enabled,
      providers: config.providers.map(
        ({ id: _id, masked_settings: _masked_settings, ...provider }) => ({
          ...provider,
          settings: { ...provider.settings },
        })
      ),
    };
  }

  return {
    providerTypes,
    loadingProviderTypes,
    loadProviderTypes,
    prepareInitialProviders,
    addProvider,
    removeProvider,
    updateProviderName,
    updateProviderEnabled,
    getProviderFieldValue,
    updateProviderField,
    getProviderById,
    providerTypeLabel,
    buildSavePayload,
  };
}

/**
 * Check whether a tracing provider is considered configured enough to send traces.
 * Mirrors the backend validation: enabled and all required fields are filled.
 */
function isProviderConfigured(
  provider: TracingProvider,
  providerTypes: TracingProviderDetails[]
): boolean {
  if (!provider.enabled) return false;
  const typeInfo = providerTypes.find((p) => p.name === provider.provider);
  if (!typeInfo) return true;
  for (const fieldName of typeInfo.required_fields) {
    const value = provider.settings[fieldName];
    if (value === undefined || value === null || value === "") return false;
  }
  return true;
}

/**
 * Count how many providers are enabled and have the minimum required fields.
 */
export function configuredProviderCount(
  providers: TracingProvider[],
  providerTypes: TracingProviderDetails[]
): number {
  return providers.filter((p) => isProviderConfigured(p, providerTypes)).length;
}
