<script setup lang="ts">
import { ChevronDown, ChevronUp, Plus, Trash2 } from "@lucide/vue";
import { computed, ref } from "vue";
import { useI18n } from "vue-i18n";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { NumberInput } from "@/components/ui/number-input";
import type { PricingTimeWindow, TokenRateOverrides } from "@/types/schemas";
import {
  EXTRA_RATE_FIELDS,
  PRIMARY_RATE_FIELDS,
  minutesOfDay,
  overlappingWindowIndexes,
  weekdayLabels,
  windowDaysLabel as formatWindowDays,
  type RateField,
  type TierRateKey,
} from "@/utils/pricingTiers";

/**
 * Editor for time-of-day pricing windows (peak / off-peak rates).
 *
 * A window reprices a request that STARTED inside it, read on the window's own
 * timezone clock; a request outside every window bills at the base rates. `end`
 * at or before `start` wraps past midnight, and `days` names the weekday the
 * window starts on (empty = every day). The parent owns persistence — this
 * component only emits windows sorted the way the backend evaluates them.
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
const expanded = ref<Set<number>>(new Set());

/** Localized short weekday names, indexed 0=Monday. */
const dayLabels = computed(() => weekdayLabels(locale.value));

/**
 * Overlapping windows are rejected by the backend (first-match-wins would make
 * the bill depend on list order), so flag them here before a save round-trip.
 * The rule itself lives in the shared overlay vocabulary, which the backend's
 * validator mirrors.
 */
const overlapping = computed(() => overlappingWindowIndexes(windows.value));

function emitUpdate(list: PricingTimeWindow[]) {
  list.sort((a, b) => {
    const zone = (a.timezone ?? "UTC").localeCompare(b.timezone ?? "UTC");
    if (zone !== 0) return zone;
    return (minutesOfDay(a.start) ?? 0) - (minutesOfDay(b.start) ?? 0);
  });
  const cleared = props.inheritWhenEmpty === false ? [] : null;
  emit("update:modelValue", list.length ? list : cleared);
}

function patchWindow(index: number, patch: Partial<PricingTimeWindow>) {
  const list = windows.value.map((window) => ({ ...window, days: [...(window.days ?? [])] }));
  list[index] = { ...list[index]!, ...patch };
  emitUpdate(list);
}

function setRate(index: number, key: TierRateKey, value: number | null) {
  patchWindow(index, { [key]: value } as Partial<TokenRateOverrides>);
}

/** Add a window that does not overlap the existing ones: overnight, off-peak. */
function addWindow() {
  const list = windows.value.map((window) => ({ ...window }));
  list.push({
    start: "22:00",
    end: "08:00",
    days: [],
    timezone: list[0]?.timezone ?? "UTC",
    input_cost_per_1m: null,
    output_cost_per_1m: null,
  });
  emitUpdate(list);
}

function removeWindow(index: number) {
  const list = windows.value.map((window) => ({ ...window }));
  list.splice(index, 1);
  expanded.value.delete(index);
  emitUpdate(list);
}

function toggleDay(index: number, day: number) {
  const window = windows.value[index];
  if (!window) return;
  const days = new Set(window.days ?? []);
  if (days.has(day)) days.delete(day);
  else days.add(day);
  patchWindow(index, { days: [...days].sort((a, b) => a - b) });
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
    EXTRA_RATE_FIELDS.some((field) => windows.value[index]?.[field.key] != null)
  );
}

/** Rate inputs repeat per window, so the accessible name carries its clock. */
function rateAriaLabel(index: number, field: RateField): string {
  const window = windows.value[index];
  const clock = window ? `${window.start}–${window.end}` : "";
  return `${t(field.labelKey)} · ${clock}`;
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
        @click="addWindow"
      >
        <Plus class="w-3.5 h-3.5 mr-1.5" />
        {{ t("models.addPricingWindow") }}
      </Button>
    </div>

    <p
      v-if="windows.length === 0"
      class="mt-3 rounded-md border border-dashed border-border px-3 py-2.5 text-[11px] text-muted-foreground"
    >
      {{ t("models.pricingWindowsEmpty") }}
    </p>

    <div v-else class="mt-3 space-y-2">
      <div
        v-for="(window, index) in windows"
        :key="index"
        class="rounded-lg border bg-muted/10"
        :class="overlapping.has(index) ? 'border-status-warning/60' : 'border-border/70'"
      >
        <div class="flex flex-wrap items-center gap-2 px-3 py-2">
          <Input
            :model-value="window.start"
            type="time"
            :disabled="disabled"
            class="h-8 w-28 text-data text-xs"
            :aria-label="t('models.pricingWindowStart')"
            @update:model-value="(value) => patchWindow(index, { start: String(value) })"
          />
          <span class="text-[11px] text-muted-foreground">→</span>
          <Input
            :model-value="window.end"
            type="time"
            :disabled="disabled"
            class="h-8 w-28 text-data text-xs"
            :aria-label="t('models.pricingWindowEnd')"
            @update:model-value="(value) => patchWindow(index, { end: String(value) })"
          />
          <Input
            :model-value="window.timezone ?? 'UTC'"
            :disabled="disabled"
            class="h-8 w-44 text-data text-xs"
            placeholder="Asia/Shanghai"
            :aria-label="t('models.pricingWindowTimezone')"
            @update:model-value="(value) => patchWindow(index, { timezone: String(value) })"
          />
          <button
            type="button"
            class="ml-auto rounded p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-destructive disabled:pointer-events-none disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60"
            :disabled="disabled"
            :aria-label="t('models.removePricingWindow')"
            @click="removeWindow(index)"
          >
            <Trash2 class="w-3.5 h-3.5" />
          </button>
        </div>

        <div class="flex flex-wrap items-center gap-1 px-3 pb-2">
          <span class="mr-1 text-[11px] text-muted-foreground">
            {{ t("models.pricingWindowDays") }}
          </span>
          <button
            v-for="(label, day) in dayLabels"
            :key="day"
            type="button"
            class="rounded px-1.5 py-0.5 text-[11px] transition-colors disabled:pointer-events-none disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60"
            :class="
              window.days?.includes(day)
                ? 'bg-foreground text-background'
                : 'text-muted-foreground hover:bg-muted'
            "
            :disabled="disabled"
            :aria-pressed="window.days?.includes(day) ?? false"
            :aria-label="dayLabels[day]"
            @click="toggleDay(index, day)"
          >
            {{ label }}
          </button>
          <span class="ml-1 text-[11px] text-muted-foreground">{{
            formatWindowDays(t, window, locale)
          }}</span>
        </div>

        <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 px-3 pb-3">
          <div v-for="field in PRIMARY_RATE_FIELDS" :key="field.key" class="space-y-1">
            <Label class="text-[11px] text-muted-foreground">{{ t(field.labelKey) }}</Label>
            <NumberInput
              :model-value="window[field.key] ?? null"
              :min="0"
              step="0.01"
              :disabled="disabled"
              :placeholder="t('models.pricingTierInherit')"
              class="h-8 text-data text-xs"
              :aria-label="rateAriaLabel(index, field)"
              @update:model-value="(value) => setRate(index, field.key, value)"
            />
          </div>
        </div>

        <div class="px-3 pb-2">
          <button
            type="button"
            class="inline-flex items-center gap-1 rounded text-[11px] text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60"
            @click="toggleExpanded(index)"
          >
            <component :is="isExpanded(index) ? ChevronUp : ChevronDown" class="w-3 h-3" />
            {{ t("models.pricingTierMoreDims") }}
          </button>
        </div>

        <div v-if="isExpanded(index)" class="grid grid-cols-2 sm:grid-cols-4 gap-2 px-3 pb-3">
          <div v-for="field in EXTRA_RATE_FIELDS" :key="field.key" class="space-y-1">
            <Label class="text-[11px] text-muted-foreground">{{ t(field.labelKey) }}</Label>
            <NumberInput
              :model-value="window[field.key] ?? null"
              :min="0"
              step="0.01"
              :disabled="disabled"
              :placeholder="t('models.pricingTierInherit')"
              class="h-8 text-data text-xs"
              :aria-label="rateAriaLabel(index, field)"
              @update:model-value="(value) => setRate(index, field.key, value)"
            />
          </div>
        </div>
      </div>

      <p v-if="overlapping.size" class="text-[11px] text-status-warning">
        {{ t("models.pricingWindowsOverlap") }}
      </p>
    </div>
  </div>
</template>
