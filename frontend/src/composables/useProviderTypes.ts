import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { useProviderTypesStore } from "@/stores/providerTypes";
import type { ProviderTypeInfo } from "@/types/schemas";
import { providerIcon, type BrandIcon } from "@/utils/icons";

interface ProviderTypeOption {
  label: string;
  value: string;
}

/** Structural subset of a provider record that the icon helpers read. */
interface ProviderLike {
  type: string;
  name: string;
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

  /**
   * Select/filter options in backend display-name order.
   *
   * `types` is read defensively: this runs inside a render computed, and a
   * single missing field must degrade to an empty option list for one frame
   * (the next `ensureLoaded()` repopulates it) instead of taking the whole
   * console down through the global error boundary.
   */
  const typeOptions = computed<ProviderTypeOption[]>(() =>
    (store.types ?? []).map((t) => ({ label: labelOf(t), value: t.type }))
  );

  /**
   * Mark for a provider row, in resolution order: custom icon_url -> the
   * branding the provider type's adapter declares -> a catalog match on the
   * type/name. Null means "no logo": the caller's `BrandMark` then renders the
   * provider name's initial.
   */
  function providerMark(provider: ProviderLike): BrandIcon | null {
    return providerIcon({
      type: provider.type,
      name: provider.name,
      iconUrl: provider.icon_url,
      brand: store.getType(provider.type),
    });
  }

  return {
    typeOptions,
    providerMark,
    ensureLoaded: store.ensureLoaded,
  };
}
