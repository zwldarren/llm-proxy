<script setup lang="ts">
import { ChevronDown, ChevronUp, Plus, Trash2, X } from "@lucide/vue";
import { computed, ref } from "vue";
import { useI18n } from "vue-i18n";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { NumberInput } from "@/components/ui/number-input";
import { TimeInput } from "@/components/ui/time-input";
import type { PricingTimeWindow, TokenRateOverrides } from "@/types/schemas";
import { clockFromMinutes, isKnownTimezone } from "@/utils/clock";
import {
  EXTRA_RATE_FIELDS,
  PRIMARY_RATE_FIELDS,
  flattenWindowBands,
  groupWindowBands,
  minutesOfDay,
  overlappingWindowIndexes,
  weekdayLabels,
  windowDaysLabel as formatWindowDays,
  type PricingWindowBand,
  type RateField,
  type TierRateKey,
  type WindowRange,
} from "@/utils/pricingTiers";

/**
 * Editor for time-of-day pricing windows (peak / off-peak rates).
 *
 * One card is a *band*: a timezone, the weekdays its ranges start on, and the
 * rates they all share. A band holds any number of clock ranges, which is how a
 * provider states "peak hours are 01:00–04:00 and 06:00–10:00 UTC" without the
 * timezone, the days and the rates being re-entered per range.
 *
 * Storage stays the flat `PricingTimeWindow[]` the API takes — one window per
 * range — so the band exists only here; `groupWindowBands` recovers it from a
 * saved list. A request that STARTED inside a window bills at its rates for the
 * whole request and `end` at or before `start` wraps past midnight; requests
 * outside every window bill at the base rates. The parent owns persistence.
 */
defineOptions({ name: "PricingTimeWindowEditor" });

const props = withDefaults(
  defineProps<{
    modelValue?: PricingTimeWindow[] | null;
    disabled?: boolean;
    /** An emptied editor means "inherit the model-level list" unless the parent
     * says otherwise: a provider mapping may instead store the empty list, which
     * is the explicit statement that this upstream has no windows. */
    inheritWhenEmpty?: boolean;
  }>(),
  { inheritWhenEmpty: true }
);

const emit = defineEmits<{
  (e: "update:modelValue", value: PricingTimeWindow[] | null): void;
}>();

const { t, locale } = useI18n();

const windows = computed(() => props.modelValue ?? []);
const bands = computed<PricingWindowBand[]>(() => groupWindowBands(windows.value));
const expanded = ref<Set<number>>(new Set());

/** Localized short weekday names, indexed 0=Monday. */
const dayLabels = computed(() => weekdayLabels(locale.value));

/**
 * Two ranges may not cover the same real instant — the backend rejects that,
 * because first-match-wins would make the bill depend on list order — so flag
 * the offending ranges before a failed save round-trip. Indexes are checked on
 * the stored shape and mapped back onto the band each range belongs to.
 */
const overlap = computed(() => {
  const windowsByRange: PricingTimeWindow[] = [];
  const positions: Array<{ band: number; range: number }> = [];
  bands.value.forEach((band, bandIndex) => {
    band.ranges.forEach((range, rangeIndex) => {
      windowsByRange.push({
        ...band.rates,
        start: range.start,
        end: range.end,
        days: band.days,
        timezone: band.timezone,
      });
      positions.push({ band: bandIndex, range: rangeIndex });
    });
  });

  const flaggedRanges = new Set<string>();
  const flaggedBands = new Set<number>();
  for (const index of overlappingWindowIndexes(windowsByRange)) {
    const position = positions[index];
    if (!position) continue;
    flaggedRanges.add(`${position.band}:${position.range}`);
    flaggedBands.add(position.band);
  }
  return { flaggedRanges, flaggedBands };
});

/** Bands whose timezone the platform cannot resolve (the backend would drop them). */
const unknownTimezones = computed(
  () =>
    new Set(bands.value.flatMap((band, index) => (isKnownTimezone(band.timezone) ? [] : [index])))
);

function emitBands(next: PricingWindowBand[]) {
  const list = flattenWindowBands(next);
  const cleared = props.inheritWhenEmpty === false ? [] : null;
  emit("update:modelValue", list.length ? list : cleared);
}

function patchBand(index: number, patch: Partial<PricingWindowBand>) {
  emitBands(bands.value.map((band, i) => (i === index ? { ...band, ...patch } : band)));
}

function patchRange(bandIndex: number, rangeIndex: number, patch: Partial<WindowRange>) {
  const band = bands.value[bandIndex];
  if (!band) return;
  const ranges = band.ranges.map((range, i) => (i === rangeIndex ? { ...range, ...patch } : range));
  patchBand(bandIndex, { ranges });
}

function setRate(bandIndex: number, key: TierRateKey, value: number | null) {
  const band = bands.value[bandIndex];
  if (!band) return;
  const rates: TokenRateOverrides = { ...band.rates };
  if (value == null) delete rates[key];
  else rates[key] = value;
  patchBand(bandIndex, { rates });
}

function toggleDay(bandIndex: number, day: number) {
  const band = bands.value[bandIndex];
  if (!band) return;
  const days = new Set(band.days);
  if (days.has(day)) days.delete(day);
  else days.add(day);
  patchBand(bandIndex, { days: [...days].sort((a, b) => a - b) });
}

/**
 * A range that cannot collide with the ones already there: the band's next
 * hour, or an overnight band when the card is new.
 */
function suggestRange(band: PricingWindowBand): WindowRange {
  const last = band.ranges.at(-1);
  if (!last) return { start: "22:00", end: "08:00" };
  const start = minutesOfDay(last.end) ?? 0;
  return { start: clockFromMinutes(start), end: clockFromMinutes(start + 60) };
}

function addRange(bandIndex: number) {
  const band = bands.value[bandIndex];
  if (!band) return;
  patchBand(bandIndex, { ranges: [...band.ranges, suggestRange(band)] });
}

/** The last range of a band takes the band with it: a band is its ranges. */
function removeRange(bandIndex: number, rangeIndex: number) {
  const band = bands.value[bandIndex];
  if (!band) return;
  const ranges = band.ranges.filter((_, i) => i !== rangeIndex);
  if (ranges.length) patchBand(bandIndex, { ranges });
  else removeBand(bandIndex);
}

/** A new band starts from the timezone in use, off-peak and with no rates yet. */
function addBand() {
  emitBands([
    ...bands.value,
    {
      timezone: bands.value[0]?.timezone ?? "UTC",
      days: [],
      ranges: [{ start: "22:00", end: "08:00" }],
      rates: {},
    },
  ]);
}

function removeBand(index: number) {
  expanded.value.delete(index);
  emitBands(bands.value.filter((_, i) => i !== index));
}

function toggleExpanded(index: number) {
  const next = new Set(expanded.value);
  if (next.has(index)) next.delete(index);
  else next.add(index);
  expanded.value = next;
}

function isExpanded(index: number): boolean {
  return (
    expanded.value.has(index) ||
    EXTRA_RATE_FIELDS.some((field) => bands.value[index]?.rates[field.key] != null)
  );
}

/** What a band covers, for the accessible names of its repeated rate inputs. */
function bandLabel(index: number): string {
  const band = bands.value[index];
  if (!band) return "";
  const clocks = band.ranges.map((range) => `${range.start}–${range.end}`).join(", ");
  return [clocks, band.timezone].filter(Boolean).join(" · ");
}

/** What one range covers — the clock rows repeat, so each needs its own name. */
function rangeLabel(bandIndex: number, rangeIndex: number): string {
  const band = bands.value[bandIndex];
  const range = band?.ranges[rangeIndex];
  if (!band || !range) return "";
  return `${range.start}–${range.end} · ${band.timezone}`;
}

function rateAriaLabel(index: number, field: RateField): string {
  return `${t(field.labelKey)} · ${bandLabel(index)}`;
}
</script>

<template>
  <div class="border-t border-border pt-4 mt-2">
    <div class="flex items-start justify-between gap-3">
      <div class="min-w-0">
        <p class="text-sm font-medium">{{ t("models.pricingWindows") }}</p>
        <p class="text-[11px] text-muted-foreground leading-relaxed max-w-lg">
          {{ t("models.pricingWindowsHelp") }}
        </p>
      </div>
      <Button
        type="button"
        variant="outline"
        size="sm"
        class="shrink-0 h-8"
        :disabled="disabled"
        @click="addBand"
      >
        <Plus class="w-3.5 h-3.5 mr-1.5" />
        {{ t("models.addPricingWindow") }}
      </Button>
    </div>

    <p
      v-if="bands.length === 0"
      class="mt-3 rounded-md border border-dashed border-border px-3 py-2.5 text-[11px] text-muted-foreground"
    >
      {{ t("models.pricingWindowsEmpty") }}
    </p>

    <div v-else class="mt-3 space-y-2.5">
      <fieldset
        v-for="(band, bandIndex) in bands"
        :key="bandIndex"
        class="min-w-0 rounded-lg border bg-muted/10"
        :class="
          overlap.flaggedBands.has(bandIndex) ? 'border-status-warning/60' : 'border-border/70'
        "
      >
        <!-- Names the group for assistive tech: a band's inputs repeat per band. -->
        <legend class="sr-only">
          {{ t("models.pricingWindowBand", { index: bandIndex + 1 }) }}
        </legend>
        <!-- The band's identity: timezone and start days, chosen once per band. -->
        <div class="flex flex-wrap items-center gap-2 px-3 py-2">
          <Input
            :model-value="band.timezone"
            :disabled="disabled"
            class="h-8 w-40 text-data text-xs"
            placeholder="UTC"
            spellcheck="false"
            autocomplete="off"
            :aria-label="t('models.pricingWindowTimezone')"
            :aria-invalid="unknownTimezones.has(bandIndex) || undefined"
            @update:model-value="(value) => patchBand(bandIndex, { timezone: String(value) })"
          />
          <span class="text-[11px] text-muted-foreground">{{ t("models.pricingWindowDays") }}</span>
          <div class="flex flex-wrap items-center gap-1">
            <button
              v-for="(label, day) in dayLabels"
              :key="day"
              type="button"
              class="rounded px-1.5 py-0.5 text-[11px] transition-colors disabled:pointer-events-none disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60"
              :class="
                band.days.includes(day)
                  ? 'bg-foreground text-background'
                  : 'text-muted-foreground hover:bg-muted'
              "
              :disabled="disabled"
              :aria-pressed="band.days.includes(day)"
              :aria-label="dayLabels[day]"
              @click="toggleDay(bandIndex, day)"
            >
              {{ label }}
            </button>
          </div>
          <span class="text-[11px] text-muted-foreground">{{
            formatWindowDays(t, band, locale)
          }}</span>
          <button
            type="button"
            class="ml-auto rounded p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-destructive disabled:pointer-events-none disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60"
            :disabled="disabled"
            :aria-label="t('models.removePricingWindow')"
            @click="removeBand(bandIndex)"
          >
            <Trash2 class="w-3.5 h-3.5" />
          </button>
        </div>

        <!-- The clock ranges this band prices; all of them share the rates below. -->
        <div class="space-y-1.5 px-3 pb-2">
          <p class="text-[11px] text-muted-foreground">{{ t("models.pricingWindowRanges") }}</p>
          <div
            v-for="(range, rangeIndex) in band.ranges"
            :key="rangeIndex"
            class="flex flex-wrap items-center gap-2"
          >
            <TimeInput
              :model-value="range.start"
              :disabled="disabled"
              class="h-8 w-20 text-xs"
              :aria-label="`${t('models.pricingWindowStart')} · ${rangeLabel(bandIndex, rangeIndex)}`"
              :aria-invalid="overlap.flaggedRanges.has(`${bandIndex}:${rangeIndex}`) || undefined"
              @update:model-value="
                (value) => value != null && patchRange(bandIndex, rangeIndex, { start: value })
              "
            />
            <span class="text-[11px] text-muted-foreground">→</span>
            <TimeInput
              :model-value="range.end"
              :disabled="disabled"
              class="h-8 w-20 text-xs"
              :aria-label="`${t('models.pricingWindowEnd')} · ${rangeLabel(bandIndex, rangeIndex)}`"
              :aria-invalid="overlap.flaggedRanges.has(`${bandIndex}:${rangeIndex}`) || undefined"
              @update:model-value="
                (value) => value != null && patchRange(bandIndex, rangeIndex, { end: value })
              "
            />
            <button
              type="button"
              class="rounded p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-destructive disabled:pointer-events-none disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60"
              :disabled="disabled"
              :aria-label="t('models.removePricingWindowRange')"
              @click="removeRange(bandIndex, rangeIndex)"
            >
              <X class="w-3.5 h-3.5" />
            </button>
          </div>
          <button
            type="button"
            class="inline-flex items-center gap-1 rounded text-[11px] text-muted-foreground transition-colors hover:text-foreground disabled:pointer-events-none disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60"
            :disabled="disabled"
            @click="addRange(bandIndex)"
          >
            <Plus class="w-3 h-3" />
            {{ t("models.addPricingWindowRange") }}
          </button>
        </div>

        <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 px-3 pb-3">
          <div v-for="field in PRIMARY_RATE_FIELDS" :key="field.key" class="space-y-1">
            <Label class="text-[11px] text-muted-foreground">{{ t(field.labelKey) }}</Label>
            <NumberInput
              :model-value="band.rates[field.key] ?? null"
              :min="0"
              step="0.01"
              :disabled="disabled"
              :placeholder="t('models.pricingTierInherit')"
              class="h-8 text-data text-xs"
              :aria-label="rateAriaLabel(bandIndex, field)"
              @update:model-value="(value) => setRate(bandIndex, field.key, value)"
            />
          </div>
        </div>

        <div class="px-3 pb-2">
          <button
            type="button"
            class="inline-flex items-center gap-1 rounded text-[11px] text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60"
            @click="toggleExpanded(bandIndex)"
          >
            <component :is="isExpanded(bandIndex) ? ChevronUp : ChevronDown" class="w-3 h-3" />
            {{ t("models.pricingTierMoreDims") }}
          </button>
        </div>

        <div v-if="isExpanded(bandIndex)" class="grid grid-cols-2 sm:grid-cols-4 gap-2 px-3 pb-3">
          <div v-for="field in EXTRA_RATE_FIELDS" :key="field.key" class="space-y-1">
            <Label class="text-[11px] text-muted-foreground">{{ t(field.labelKey) }}</Label>
            <NumberInput
              :model-value="band.rates[field.key] ?? null"
              :min="0"
              step="0.01"
              :disabled="disabled"
              :placeholder="t('models.pricingTierInherit')"
              class="h-8 text-data text-xs"
              :aria-label="rateAriaLabel(bandIndex, field)"
              @update:model-value="(value) => setRate(bandIndex, field.key, value)"
            />
          </div>
        </div>
      </fieldset>

      <p v-if="unknownTimezones.size" class="text-[11px] text-status-warning">
        {{ t("models.pricingWindowUnknownTimezone") }}
      </p>
      <p v-if="overlap.flaggedRanges.size" class="text-[11px] text-status-warning">
        {{ t("models.pricingWindowsOverlap") }}
      </p>
    </div>
  </div>
</template>
