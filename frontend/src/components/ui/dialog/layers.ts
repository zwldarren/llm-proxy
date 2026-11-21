/**
 * Stacking layers shared by every overlay surface.
 *
 * `base` is what a surface opened by a page control uses. `top` is for a
 * surface that *interrupts* another one: a confirm over a sheet, a step-up over
 * whatever asked for it. Inside one layer the paint order is DOM order, which
 * is declaration order in the view — so a confirm that shares `base` with the
 * sheet it is guarding lands underneath that sheet whenever the view declares
 * the confirm first. The card is then visible but dead: the sheet's scrim eats
 * the click, the sheet itself has `pointer-events: none` while the confirm is
 * up, and the operator reads it as a frozen page.
 */
export const SURFACE_LAYERS = {
  base: "z-50",
  top: "z-[100]",
} as const;

export type SurfaceLayer = keyof typeof SURFACE_LAYERS;
