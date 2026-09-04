import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { toast } from "vue-sonner";
import { useClipboard } from "@vueuse/core";
import { CAPABILITY_ORDER } from "@/components/plaza/capabilities";
import type { ModelCatalogEntry } from "@/types/schemas";

/**
 * Shared row logic for the plaza model components (list item + table row).
 *
 * Both row shapes render the same data — capability icons in fixed order,
 * the sanitized homepage link, the copy-name affordance, and the
 * quality-tier badge variant — so the logic lives here once.
 */
export function usePlazaModelRow(model: () => ModelCatalogEntry) {
  const { t } = useI18n();

  const capabilities = computed(() =>
    CAPABILITY_ORDER.filter((cap) => model().capabilities?.includes(cap))
  );

  /** Homepage URL only when it is an absolute http(s) link. */
  const safeHomepageUrl = computed(() => {
    const url = model().homepage_url;
    if (!url) return null;
    return /^(https?):\/\//i.test(url) ? url : null;
  });

  const { copy, copied } = useClipboard({ legacy: true, copiedDuring: 1500 });

  async function copyName() {
    try {
      await copy(model().name);
    } catch {
      toast.error(t("plaza.copyFailed"));
    }
  }

  function tierBadgeVariant(tier: string | null | undefined): "default" | "secondary" | "outline" {
    if (tier === "PREMIUM") return "default";
    if (tier === "BALANCED") return "secondary";
    return "outline";
  }

  return { capabilities, safeHomepageUrl, copied, copyName, tierBadgeVariant };
}
