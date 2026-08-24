/**
 * Categorical color scale for data-viz surfaces (Usage trends chart series
 * and the hue dots in its model filter).
 *
 * The chart never displays more than 10 series, so the scale is built around
 * exactly 10 hue slots evenly spaced across the allowed hue arc — the widest
 * possible separation (29.5° between adjacent slots). Slots are assigned in a
 * golden-style permuted order (stride 7, coprime with 10) so any prefix —
 * Top 5, Top 10 — is spread across the whole arc instead of clustering in
 * one hue neighborhood. Saturation and lightness alternate by slot parity,
 * so the closest hue pairs always also differ in tone.
 *
 * Names are ranked by the caller (usage order); ranks beyond the first 10
 * (only reachable via manual selection in the filter popover) reuse the
 * slots with a half-step hue shift and swapped tone, keeping them distinct
 * from their prime-slot neighbors.
 *
 * The hue arc excludes reds/oranges (0–55°) to avoid confusion with the
 * semantic status colors (error, warning). Chroma stays in the muted
 * Monochrome Editorial range — colors disambiguate, never decorate.
 */

/** Number of prime hue slots — matches the chart's hard 10-series cap. */
const SLOT_COUNT = 10;

/**
 * Slot assignment stride, coprime with SLOT_COUNT. Consecutive ranks land 7
 * slots (≈206°) apart, so any prefix of the ranking spans the full arc.
 */
const SLOT_STRIDE = 7;

/** Allowed categorical hue arc: [55°, 350°), skipping status red/orange. */
const HUE_MIN = 55;
const HUE_ARC = 295;
const HUE_STEP = HUE_ARC / SLOT_COUNT;
/** Prime slots sit half a step inside the arc so the shifted cycle stays in range. */
const HUE_BASE = HUE_MIN + HUE_STEP / 2;

/** Muted two-tone ramps, alternated by slot parity so hue neighbors differ in tone. */
const SATURATION_RAMP = [28, 38] as const;
const LIGHTNESS_RAMP_DARK = [60, 70] as const;
const LIGHTNESS_RAMP_LIGHT = [48, 38] as const;

/** Fallback for names outside the scale's set — Neutral Slate token. */
const FALLBACK = "hsl(220 5% 50%)";

type CategoricalColorScale = (name: string, isDark: boolean) => string;

/**
 * Build a color scale over a ranked set of category names.
 *
 * @param names Category names in priority order (e.g. ranked by usage) — the
 * first 10 receive the prime, maximally separated slot colors.
 *
 * Stability contract: the same ranked list always yields the same name→color
 * mapping, so a category keeps its color across Top-5/Top-10/filter toggles
 * and page reloads, and chart series match the hue dots in the model filter.
 */
export function createCategoricalColorScale(names: readonly string[]): CategoricalColorScale {
  const indexByName = new Map(names.map((name, i) => [name, i]));

  return (name, isDark) => {
    const rank = indexByName.get(name);
    if (rank === undefined) return FALLBACK;

    const cycle = Math.floor(rank / SLOT_COUNT);
    const slot = (((rank * SLOT_STRIDE) % SLOT_COUNT) + SLOT_COUNT) % SLOT_COUNT;
    const tone = (slot + cycle) % 2;

    // Later cycles shift half a slot step and wrap inside the allowed arc.
    const rawHue = HUE_BASE + slot * HUE_STEP + cycle * (HUE_STEP / 2);
    const hue = HUE_MIN + ((rawHue - HUE_MIN) % HUE_ARC);

    const saturation = SATURATION_RAMP[tone];
    const lightness = (isDark ? LIGHTNESS_RAMP_DARK : LIGHTNESS_RAMP_LIGHT)[tone];

    return `hsl(${Math.round(hue)} ${saturation}% ${lightness}%)`;
  };
}
