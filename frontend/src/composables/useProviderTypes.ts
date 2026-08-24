import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { useProviderTypesStore } from "@/stores/providerTypes";
import type { ProviderTypeInfo } from "@/types/schemas";
import { getProviderIconUrl, iconUrlFromMetadata, isMonoProvider } from "@/utils/icons";

interface ProviderTypeOption {
  label: string;
  value: string;
}

/** Structural subset of a provider record that the icon helpers read. */
interface ProviderLike {
  type: string;
  icon_url?: string | null;
}

/**
 * Reactive access to the backend provider-type catalog.
 *
 * Labels are picked per the active locale (name_zh for zh, name_en otherwise)
 * and fall back to the raw type name, so a provider type missing metadata
 * still renders sensibly. Call `ensureLoaded()` from the consuming view
 * (single-flight, cached in the store).
 */
export function useProviderTypes() {
  const store = useProviderTypesStore();
  const { locale } = useI18n();

  const isZh = computed(() => locale.value.toLowerCase().startsWith("zh"));

  function labelOf(info: ProviderTypeInfo): string {
    const name = isZh.value ? info.name_zh || info.name_en : info.name_en || info.name_zh;
    return name || info.type;
  }

  /** Select/filter options in backend display-name order. */
  const typeOptions = computed<ProviderTypeOption[]>(() =>
    store.types.map((t) => ({ label: labelOf(t), value: t.type }))
  );

  /**
   * Icon URL for a provider row, in resolution order:
   * custom icon_url -> backend type metadata -> static map.
   */
  function providerIconUrl(provider: ProviderLike): string | null {
    return (
      provider.icon_url ||
      iconUrlFromMetadata(store.getType(provider.type)) ||
      getProviderIconUrl(provider.type)
    );
  }

  /** Mono styling: the backend-declared variant wins; fall back to the static map. */
  function providerIsMono(provider: ProviderLike): boolean {
    const info = store.getType(provider.type);
    return info?.icon_id ? info.icon_variant === "mono" : isMonoProvider(provider.type);
  }

  return {
    typeOptions,
    providerIconUrl,
    providerIsMono,
    ensureLoaded: store.ensureLoaded,
  };
}
