import { useStorage, type RemovableRef } from "@vueuse/core";

/**
 * Shared "table" | "list" view-mode preference for the config pages
 * (API Keys, Providers, Models, MCP Servers).
 *
 * The preference is persisted per page in localStorage. Values written by
 * the retired card view ("cards") are migrated to the list view on read.
 */
export type ViewMode = "table" | "list";

export function useViewMode(storageKey: string): RemovableRef<ViewMode> {
  const mode = useStorage<ViewMode | "cards">(storageKey, "table");
  if ((mode.value as string) === "cards") {
    mode.value = "list";
  }
  return mode as RemovableRef<ViewMode>;
}
