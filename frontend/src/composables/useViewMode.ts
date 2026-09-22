import { useMediaQuery, useStorage } from "@vueuse/core";
import { computed, type WritableComputedRef } from "vue";

/**
 * Shared "table" | "list" view-mode preference for the config pages
 * (API Keys, Providers, Models, MCP Servers).
 *
 * The preference is persisted per page in localStorage. Values written by
 * the retired card view ("cards") are migrated to the list view on read.
 *
 * Below the console breakpoint (1024px — the same gate LogsView uses) the
 * table view is not usable: its fixed-width columns overflow 375–1023px and
 * there is no scroll affordance. The list view is forced there while the
 * stored preference is left untouched, so it comes back on a wider viewport.
 */
export type ViewMode = "table" | "list";

/**
 * Viewport gate below which console tables are unusable: their fixed-width
 * columns overflow 375–1023px with no scroll affordance. Shared by every
 * table/list view-mode switch (this composable) and by LogsView's own
 * table/list fallback so all console surfaces flip at the same width.
 */
export const CONSOLE_DESKTOP_QUERY = "(min-width: 1024px)";

export function useViewMode(storageKey: string): WritableComputedRef<ViewMode> {
  const mode = useStorage<ViewMode | "cards">(storageKey, "table");
  if ((mode.value as string) === "cards") {
    mode.value = "list";
  }

  const isDesktop = useMediaQuery(CONSOLE_DESKTOP_QUERY);

  return computed({
    get: () => (isDesktop.value ? (mode.value as ViewMode) : "list"),
    set: (value: ViewMode) => {
      mode.value = value;
    },
  });
}
