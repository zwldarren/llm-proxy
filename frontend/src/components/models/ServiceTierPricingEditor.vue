<script setup lang="ts">
import { ChevronDown, ChevronUp, Plus, Trash2 } from "@lucide/vue";
import { computed, ref } from "vue";
import { useI18n } from "vue-i18n";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { NumberInput } from "@/components/ui/number-input";
import type { ServiceTierPricing, ServiceTierRates } from "@/types/schemas";
import { EXTRA_RATE_FIELDS, PRIMARY_RATE_FIELDS, type RateField } from "@/utils/pricingTiers";

/**
 * Editor for service-tier rate overrides (owned by models.dev `experimental.modes`).
 *
 * Each entry is one mode the provider serves the same model at — OpenAI's `flex`
 * (cheaper, slower) and `priority` (dearer, faster) — keyed by the name a client
 * sends as `service_tier`. A request served at a tier with no entry bills at the
 * base rates, and a blank dimension inherits the rate the turn would otherwise
 * have been billed at (context tier or time window included).
 */
defineOptions({ name: "ServiceTierPricingEditor" });

const props = withDefaults(
  defineProps<{
    modelValue?: ServiceTierPricing | null;
    disabled?: boolean;
    /** An emptied editor means "inherit the model-level map" unless the parent
     * says otherwise: a provider mapping may instead store the empty map, which
     * is the explicit statement that this upstream has no tier prices. */
    inheritWhenEmpty?: boolean;
  }>(),
  { inheritWhenEmpty: true }
);

const emit = defineEmits<{
  (e: "update:modelValue", value: ServiceTierPricing | null): void;
}>();

const { t } = useI18n();

type RateKey = keyof ServiceTierRates;

/** Suggested mode names, offered as the default for a new row. */
const SUGGESTED_MODES = ["flex", "priority", "fast"];

const rows = computed(() =>
  Object.entries(props.modelValue ?? {}).map(([name, rates]) => ({ name, rates: rates ?? {} }))
);
const expanded = ref<Set<number>>(new Set());

function emitRows(next: Array<{ name: string; rates: ServiceTierRates }>) {
  const record: ServiceTierPricing = {};
  for (const row of next) {
    const name = row.name.trim().toLowerCase();
    if (!name) continue;
    record[name] = row.rates;
  }
  const cleared = props.inheritWhenEmpty === false ? {} : null;
  emit("update:modelValue", Object.keys(record).length ? record : cleared);
}

function addRow() {
  const used = new Set(rows.value.map((row) => row.name));
  const name = SUGGESTED_MODES.find((mode) => !used.has(mode)) ?? "";
  emitRows([...rows.value, { name, rates: {} }]);
}

function removeRow(index: number) {
  const next = rows.value.map((row) => ({ ...row, rates: { ...row.rates } }));
  next.splice(index, 1);
  expanded.value.delete(index);
  emitRows(next);
}

function renameRow(index: number, name: string) {
  const next = rows.value.map((row) => ({ ...row, rates: { ...row.rates } }));
  next[index]!.name = name;
  emitRows(next);
}

function setRate(index: number, key: RateKey, value: number | null) {
  const next = rows.value.map((row) => ({ ...row, rates: { ...row.rates } }));
  next[index]!.rates[key] = value;
  emitRows(next);
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
    EXTRA_RATE_FIELDS.some((field) => rows.value[index]?.rates[field.key] != null)
  );
}

/** Rate inputs repeat per mode, so the accessible name carries the mode. */
function rateAriaLabel(index: number, field: RateField): string {
  const mode = rows.value[index]?.name || t("models.serviceTierName");
  return `${t(field.labelKey)} · ${mode}`;
}
</script>

<template>
  <div class="border-t border-border pt-4 mt-2">
    <div class="flex items-start justify-between gap-3">
      <div class="min-w-0">
        <p class="text-sm font-medium">{{ t("models.serviceTiers") }}</p>
        <p class="text-[11px] text-muted-foreground leading-relaxed max-w-lg">
          {{ t("models.serviceTiersHelp") }}
        </p>
      </div>
      <Button
        type="button"
        variant="outline"
        size="sm"
        class="shrink-0 h-8"
        :disabled="disabled"
        @click="addRow"
      >
        <Plus class="w-3.5 h-3.5 mr-1.5" />
        {{ t("models.addServiceTier") }}
      </Button>
    </div>

    <p
      v-if="rows.length === 0"
      class="mt-3 rounded-md border border-dashed border-border px-3 py-2.5 text-[11px] text-muted-foreground"
    >
      {{ t("models.serviceTiersEmpty") }}
    </p>

    <div v-else class="mt-3 space-y-2">
      <div
        v-for="(row, index) in rows"
        :key="index"
        class="rounded-lg border border-border/70 bg-muted/10"
      >
        <div class="flex items-center gap-2 px-3 py-2">
          <div class="w-40 shrink-0">
            <Input
              :model-value="row.name"
              :disabled="disabled"
              class="h-8 text-data text-xs"
              :placeholder="t('models.serviceTierName')"
              :aria-label="t('models.serviceTierName')"
              @update:model-value="(value) => renameRow(index, String(value))"
            />
          </div>
          <span class="min-w-0 truncate text-[11px] text-muted-foreground">
            {{ t("models.serviceTierNameHelp") }}
          </span>
          <button
            type="button"
            class="ml-auto rounded p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-destructive disabled:pointer-events-none disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60"
            :disabled="disabled"
            :aria-label="t('models.removeServiceTier')"
            @click="removeRow(index)"
          >
            <Trash2 class="w-3.5 h-3.5" />
          </button>
        </div>

        <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 px-3 pb-3">
          <div v-for="field in PRIMARY_RATE_FIELDS" :key="field.key" class="space-y-1">
            <Label class="text-[11px] text-muted-foreground">{{ t(field.labelKey) }}</Label>
            <NumberInput
              :model-value="row.rates[field.key] ?? null"
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
              :model-value="row.rates[field.key] ?? null"
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
    </div>
  </div>
</template>
