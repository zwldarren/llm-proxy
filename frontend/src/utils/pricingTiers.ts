/**
 * Shared vocabulary and presentation rules for context-pricing tiers.
 *
 * Three surfaces deal in tiers — the model editor, the pricing popover's
 * read-only breakdown, and the models.dev sync panel's diff chips — so the
 * rate-dimension list and the threshold wording live here once instead of
 * being restated per component.
 */

import type { ComposerTranslation } from "vue-i18n";
import type { PricingTier } from "@/types/schemas";

/** Every rate dimension a tier can override (``threshold`` excluded). */
export type TierRateKey = Exclude<keyof PricingTier, "threshold">;

/** The tier rate dimensions, in display order. */
export const TIER_RATE_KEYS: readonly TierRateKey[] = [
  "input_cost_per_1m",
  "output_cost_per_1m",
  "cached_read_cost_per_1m",
  "cached_write_cost_per_1m",
  "audio_input_cost_per_1m",
  "audio_output_cost_per_1m",
  "image_input_cost_per_1m",
];

/** "≥ 272,000 tokens" label for a tier threshold. */
export function tierThresholdLabel(t: ComposerTranslation, threshold: number): string {
  return t("models.pricingTierFrom", { tokens: threshold.toLocaleString("en-US") });
}
