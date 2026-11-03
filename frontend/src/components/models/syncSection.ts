/**
 * Contract between the models.dev sync dialog and its review sections.
 *
 * The dialog owns the flow — fetch, one bottom bar, one write — while each
 * section (pricing, capabilities) owns its rows, filters and selection. A
 * section exposes a `summary` in this shape plus `fetchPreview()` and
 * `buildUpdates()`, so the dialog can show what apply would write and send
 * both sections in a single request without knowing their row shapes.
 */

/** The two independent field sets models.dev can supply. */
export type SyncSectionKey = "pricing" | "capabilities";

/** Where a section's preview fetch stands. */
export type SyncSectionStatus = "idle" | "loading" | "ready" | "error";

/** What the dialog reads off a section. */
export interface SyncSectionState {
  status: SyncSectionStatus;
  error: string | null;
  /** Rows carrying any diff against the stored configuration. */
  changedCount: number;
  /** Selected rows that actually differ, i.e. what apply would write. */
  applicableCount: number;
  /**
   * Rows that would overwrite a stored value and are therefore *not*
   * preselected. The dialog names this count, so the preselect rule is visible
   * instead of being a surprise.
   */
  blockedCount: number;
  /**
   * The rows behind `applicableCount`, named as "model · provider" (pricing) or
   * the model name (capabilities), in the order the confirmation lists them.
   */
  pending: string[];
}

/** Placeholder state for a section whose component is not mounted yet. */
export const EMPTY_SECTION_STATE: SyncSectionState = {
  status: "idle",
  error: null,
  changedCount: 0,
  applicableCount: 0,
  blockedCount: 0,
  pending: [],
};
