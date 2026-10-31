/**
 * Shared vocabulary and presentation rules for price overlays.
 *
 * Four surfaces deal in overlays — the model editor's tier, window and
 * service-tier editors, and the pricing popover's read-only breakdown — so the
 * rate-dimension list, the money formatting and the section rows live here once
 * instead of being restated per component. A new rate dimension is added here
 * and reaches every surface.
 */

import type { ComposerTranslation } from "vue-i18n";
import type {
  PricingTier,
  PricingTimeWindow,
  ServiceTierPricing,
  TokenRateOverrides,
} from "@/types/schemas";

/** Every rate dimension an overlay can override (``threshold`` excluded). */
export type TierRateKey = Exclude<keyof PricingTier, "threshold">;

/** The overlay rate dimensions, in display order. */
export const TIER_RATE_KEYS: readonly TierRateKey[] = [
  "input_cost_per_1m",
  "output_cost_per_1m",
  "cached_read_cost_per_1m",
  "cached_write_cost_per_1m",
  "reasoning_cost_per_1m",
  "audio_input_cost_per_1m",
  "audio_output_cost_per_1m",
  "image_input_cost_per_1m",
];

/** A rate dimension paired with its label key, for the overlay editors. */
export interface RateField {
  key: TierRateKey;
  labelKey: string;
}

/** The dimensions the editors show first: the rates most models actually set. */
export const PRIMARY_RATE_FIELDS: readonly RateField[] = [
  { key: "input_cost_per_1m", labelKey: "models.costDimInput" },
  { key: "output_cost_per_1m", labelKey: "models.costDimOutput" },
  { key: "cached_read_cost_per_1m", labelKey: "models.costDimCachedRead" },
  { key: "cached_write_cost_per_1m", labelKey: "models.costDimCachedWrite" },
];

/** The remaining dimensions, behind the editors' "more dimensions" disclosure. */
export const EXTRA_RATE_FIELDS: readonly RateField[] = [
  { key: "reasoning_cost_per_1m", labelKey: "models.costDimReasoning" },
  { key: "audio_input_cost_per_1m", labelKey: "models.costDimAudioInput" },
  { key: "audio_output_cost_per_1m", labelKey: "models.costDimAudioOutput" },
  { key: "image_input_cost_per_1m", labelKey: "models.costDimImageInput" },
];

/** "$0.5" for a rate, trimming to at most 6 decimals without trailing zeros. */
export function formatCost(v: number): string {
  const formatted = Number.parseFloat(v.toFixed(6));
  // Very small non-zero rates fall back to exponent form so they never read "$0".
  if (formatted === 0 && v > 0) return `$${v.toExponential(2)}`;
  return `$${formatted}`;
}

/** A rate, or an em dash when the overlay leaves that dimension unset. */
export function formatRate(v: number | null | undefined): string {
  return v == null ? "—" : formatCost(v);
}

/** "≥ 272,000 tokens" label for a tier threshold. */
export function tierThresholdLabel(t: ComposerTranslation, threshold: number): string {
  return t("models.pricingTierFrom", { tokens: threshold.toLocaleString("en-US") });
}

/** One read-only row of an overlay section in the pricing popover. */
export interface RateRow {
  key: string;
  label: string;
  text: string;
}

/** IN/OUT plus the cached-read and reasoning rates when the overlay sets them. */
function overlayRateText(t: ComposerTranslation, rates: TokenRateOverrides): string {
  const parts = [
    `${t("models.inputShort")} ${formatRate(rates.input_cost_per_1m)}`,
    `${t("models.outputShort")} ${formatRate(rates.output_cost_per_1m)}`,
  ];
  if (rates.cached_read_cost_per_1m != null) {
    parts.push(`${t("models.cachedShort")} ${formatCost(rates.cached_read_cost_per_1m)}`);
  }
  if (rates.reasoning_cost_per_1m != null) {
    parts.push(`${t("models.reasoningShort")} ${formatCost(rates.reasoning_cost_per_1m)}`);
  }
  return parts.join(" · ");
}

/** Context tiers as popover rows, keyed by threshold. */
export function tierRateRows(t: ComposerTranslation, tiers: PricingTier[]): RateRow[] {
  return tiers.map((tier) => ({
    key: `tier-${tier.threshold}`,
    label: tierThresholdLabel(t, tier.threshold),
    text: overlayRateText(t, tier),
  }));
}

/** Service tiers as popover rows, keyed by mode name. */
export function serviceTierRateRows(
  t: ComposerTranslation,
  pricing: ServiceTierPricing
): RateRow[] {
  return Object.entries(pricing).map(([name, rates]) => ({
    key: `mode-${name}`,
    label: name,
    text: overlayRateText(t, rates ?? {}),
  }));
}

/** Short weekday names indexed 0=Monday, localized — no i18n keys needed. */
export function weekdayLabels(locale: string): string[] {
  return Array.from({ length: 7 }, (_, index) => {
    // 2024-01-01 is a Monday, so the index maps straight onto ISO weekdays.
    const date = new Date(2024, 0, 1 + index);
    return new Intl.DateTimeFormat(locale, { weekday: "short" }).format(date);
  });
}

/** "Mon, Fri" or "every day" for a window's start days. */
export function windowDaysLabel(t: ComposerTranslation, window: PricingTimeWindow, locale: string) {
  if (!window.days?.length) return t("models.pricingWindowEveryDay");
  const labels = weekdayLabels(locale);
  return window.days.map((day) => labels[day] ?? String(day)).join(", ");
}

/** Time windows as popover rows, keyed by their clock and start days. */
export function windowRateRows(
  t: ComposerTranslation,
  windows: PricingTimeWindow[],
  locale = "en-US"
): RateRow[] {
  return windows.map((window, index) => ({
    key: `window-${index}`,
    label: `${window.start}–${window.end} · ${windowDaysLabel(t, window, locale)}`,
    text: overlayRateText(t, window),
  }));
}

/** "HH:MM" as minutes since midnight, or null when the value is not a clock. */
export function minutesOfDay(value: string | undefined): number | null {
  const match = /^([01]\d|2[0-3]):([0-5]\d)$/.exec(value ?? "");
  return match ? Number(match[1]) * 60 + Number(match[2]) : null;
}

/** Absolute-minute span a window covers when it starts on `day`. */
function spanOf(window: PricingTimeWindow, day: number): [number, number] {
  const start = (minutesOfDay(window.start) ?? 0) + day * 1440;
  let end = (minutesOfDay(window.end) ?? 0) + day * 1440;
  if (end <= start) end += 1440;
  return [start, end];
}

/**
 * Indexes of windows that overlap a peer, compared in real (UTC) time.
 *
 * The backend rejects these (first-match-wins would make the bill depend on
 * list order), so flagging them here saves a failed save round-trip. A window
 * that crosses midnight also covers the next weekday's small hours, so the
 * comparison is over the spans each window actually covers — a Sunday
 * 22:00–08:00 band collides with a Monday 00:00–02:00 band even though they
 * share no start day. Windows read their own timezone clock, so two bands in
 * different zones colliding on the same real instant must be flagged too; the
 * backend compares them in UTC, and this mirrors it with the same rule: each
 * band is tested under the offset pairs the two zones actually shared (never a
 * DST pairing no real date produces).
 */
export function overlappingWindowIndexes(windows: PricingTimeWindow[]): Set<number> {
  const flagged = new Set<number>();
  for (let i = 0; i < windows.length; i += 1) {
    for (let j = i + 1; j < windows.length; j += 1) {
      const left = windows[i]!;
      const right = windows[j]!;
      const leftZone = left.timezone ?? "UTC";
      const rightZone = right.timezone ?? "UTC";
      const offsetPairs =
        leftZone === rightZone
          ? ([[0, 0]] as Array<[number, number]>)
          : coexistingOffsets(leftZone, rightZone);
      const leftSpans = spansOf(left);
      const rightSpans = spansOf(right);
      const collides = offsetPairs.some(([leftOffset, rightOffset]) =>
        leftSpans.some(([ls, le]) =>
          rightSpans.some(
            ([rs, re]) => ls - leftOffset < re - rightOffset && rs - rightOffset < le - leftOffset
          )
        )
      );
      if (collides) {
        flagged.add(i);
        flagged.add(j);
      }
    }
  }
  return flagged;
}

/**
 * Instants sampled to learn each zone's offsets: one per quarter of the
 * reference year catches both DST regimes, and sampling the *same* instant for
 * both zones keeps only offset pairs that really coexisted.
 */
const OFFSET_SAMPLE_INSTANTS = ["01-15", "04-15", "07-15", "10-15"].map(
  (monthDay) => new Date(`2025-${monthDay}T00:00:00Z`)
);

/** A zone's UTC offset in minutes east of UTC, read through the platform tz db. */
function offsetMinutes(timeZone: string, at: Date): number {
  const name = new Intl.DateTimeFormat("en-US", {
    timeZone,
    timeZoneName: "longOffset",
  })
    .formatToParts(at)
    .find((part) => part.type === "timeZoneName")?.value;
  // "GMT+08:00" / "GMT-03:30"; plain "GMT" is UTC.
  const match = /GMT([+-])(\d{2}):(\d{2})/.exec(name ?? "");
  if (!match) return 0;
  const sign = match[1] === "-" ? -1 : 1;
  return sign * (Number(match[2]) * 60 + Number(match[3]));
}

/** The UTC-offset pairs two zones shared on the sampled instants. */
function coexistingOffsets(leftZone: string, rightZone: string): Array<[number, number]> {
  const pairs = new Map<string, [number, number]>();
  for (const at of OFFSET_SAMPLE_INSTANTS) {
    const pair: [number, number] = [offsetMinutes(leftZone, at), offsetMinutes(rightZone, at)];
    pairs.set(pair.join(","), pair);
  }
  return [...pairs.values()];
}

/** Every span a window covers over two consecutive weeks, Monday of week 0 = 0. */
function spansOf(window: PricingTimeWindow): Array<[number, number]> {
  const days = window.days?.length ? window.days : [0, 1, 2, 3, 4, 5, 6];
  return days.flatMap((day) => [spanOf(window, day), spanOf(window, day + 7)]);
}
