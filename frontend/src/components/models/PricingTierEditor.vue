<script setup lang="ts">
import { ChevronDown, ChevronUp, Plus, Trash2 } from "@lucide/vue";
import { computed, ref } from "vue";
import { useI18n } from "vue-i18n";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { NumberInput } from "@/components/ui/number-input";
import type { PricingTier } from "@/types/schemas";

/**
 * Editor for context-based pricing tiers.
 *
 * A tier applies from its input-token threshold onward; rates left blank
 * inherit the model/provider base price. The parent owns persistence — this
 * component only emits normalized (sorted, deduplicated-checked) lists.
 */
defineOptions({ name: "PricingTierEditor" });

const props = defineProps<{
  modelValue?: PricingTier[] | null;
  disabled?: boolean;
}>();

const emit = defineEmits<{
  (e: "update:modelValue", value: PricingTier[] | null): void;
}>();

const { t } = useI18n();

type RateKey = Exclude<keyof PricingTier, "threshold">;

const PRIMARY_FIELDS: Array<{ key: RateKey; labelKey: string }> = [
  { key: "input_cost_per_1m", labelKey: "models.costDimInput" },
  { key: "output_cost_per_1m", labelKey: "models.costDimOutput" },
  { key: "cached_read_cost_per_1m", labelKey: "models.costDimCachedRead" },
  { key: "cached_write_cost_per_1m", labelKey: "models.costDimCachedWrite" },
];

const EXTRA_FIELDS: Array<{ key: RateKey; labelKey: string }> = [
  { key: "audio_input_cost_per_1m", labelKey: "models.costDimAudioInput" },
  { key: "audio_output_cost_per_1m", labelKey: "models.costDimAudioOutput" },
  { key: "image_input_cost_per_1m", labelKey: "models.costDimImageInput" },
];

const tiers = computed(() => props.modelValue ?? []);
const expanded = ref<Set<number>>(new Set());

const duplicateThresholds = computed(() => {
  const seen = new Set<number>();
  const dupes = new Set<number>();
  for (const tier of tiers.value) {
    if (seen.has(tier.threshold)) dupes.add(tier.threshold);
    seen.add(tier.threshold);
  }
  return [...dupes];
});

function emitUpdate(list: PricingTier[]) {
  list.sort((a, b) => a.threshold - b.threshold);
  emit("update:modelValue", list.length ? list : null);
}

function setThreshold(index: number, value: number | null) {
  const list = tiers.value.map((tier) => ({ ...tier }));
  list[index]!.threshold = value === null ? 0 : Math.max(0, Math.round(value));
  emitUpdate(list);
}

function setRate(index: number, key: RateKey, value: number | null) {
  const list = tiers.value.map((tier) => ({ ...tier }));
  list[index]![key] = value;
  emitUpdate(list);
}

/** Default to one step past the highest tier (or the common 200k/272k band). */
function addTier() {
  const list = tiers.value.map((tier) => ({ ...tier }));
  const highest = list.reduce((max, tier) => Math.max(max, tier.threshold), 0);
  list.push({
    threshold: highest === 0 ? 200000 : highest * 2,
    input_cost_per_1m: null,
    output_cost_per_1m: null,
  });
  emitUpdate(list);
}

function removeTier(index: number) {
  const list = tiers.value.map((tier) => ({ ...tier }));
  list.splice(index, 1);
  expanded.value.delete(index);
  emitUpdate(list);
}

function toggleExpanded(index: number) {
  const next = new Set(expanded.value);
  if (next.has(index)) next.delete(index);
  else next.add(index);
  expanded.value = next;
}

function isExpanded(index: number): boolean {
  return expanded.value.has(index) || EXTRA_FIELDS.some((f) => tiers.value[index]?.[f.key] != null);
}
</script>

<template>
  <div class="border-t border-border pt-4 mt-2">
    <div class="flex items-start justify-between gap-3">
      <div class="min-w-0">
        <p class="text-sm font-medium">{{ t("models.pricingTiers") }}</p>
        <p class="text-[11px] text-muted-foreground leading-relaxed max-w-lg">
          {{ t("models.pricingTiersHelp") }}
        </p>
      </div>
      <Button
        type="button"
        variant="outline"
        size="sm"
        class="shrink-0 h-8"
        :disabled="disabled"
        @click="addTier"
      >
        <Plus class="w-3.5 h-3.5 mr-1.5" />
        {{ t("models.addPricingTier") }}
      </Button>
    </div>

    <p
      v-if="tiers.length === 0"
      class="mt-3 rounded-md border border-dashed border-border px-3 py-2.5 text-[11px] text-muted-foreground"
    >
      {{ t("models.pricingTiersEmpty") }}
    </p>

    <div v-else class="mt-3 space-y-2">
      <div
        v-for="(tier, index) in tiers"
        :key="index"
        class="rounded-lg border border-border/70 bg-muted/10"
      >
        <div class="flex items-center gap-2 px-3 py-2">
          <span class="text-data text-xs font-medium text-muted-foreground">≥</span>
          <!-- Width lives on the wrapper: NumberFieldRoot hosts the absolutely
               positioned stepper buttons, so a w-32 on NumberInput would shrink
               only the inner input and leave the chevrons detached. -->
          <div class="w-32 shrink-0">
            <NumberInput
              :model-value="tier.threshold"
              :min="0"
              :step="1000"
              :disabled="disabled"
              class="h-8 text-data text-xs"
              :aria-label="t('models.pricingTierThreshold')"
              @update:model-value="(value) => setThreshold(index, value)"
            />
          </div>
          <span class="text-[11px] text-muted-foreground">
            {{ t("models.pricingTierTokens") }}
          </span>
          <button
            type="button"
            class="ml-auto rounded p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-destructive disabled:pointer-events-none disabled:opacity-40"
            :disabled="disabled"
            :aria-label="t('models.removePricingTier')"
            @click="removeTier(index)"
          >
            <Trash2 class="w-3.5 h-3.5" />
          </button>
        </div>

        <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 px-3 pb-3">
          <div v-for="field in PRIMARY_FIELDS" :key="field.key" class="space-y-1">
            <Label class="text-[11px] text-muted-foreground">{{ t(field.labelKey) }}</Label>
            <NumberInput
              :model-value="tier[field.key] ?? null"
              :min="0"
              step="0.01"
              :disabled="disabled"
              :placeholder="t('models.pricingTierInherit')"
              class="h-8 text-data text-xs"
              @update:model-value="(value) => setRate(index, field.key, value)"
            />
          </div>
        </div>

        <div class="px-3 pb-2">
          <button
            type="button"
            class="inline-flex items-center gap-1 text-[11px] text-muted-foreground transition-colors hover:text-foreground"
            @click="toggleExpanded(index)"
          >
            <component :is="isExpanded(index) ? ChevronUp : ChevronDown" class="w-3 h-3" />
            {{ t("models.pricingTierMoreDims") }}
          </button>
        </div>

        <div v-if="isExpanded(index)" class="grid grid-cols-2 sm:grid-cols-3 gap-2 px-3 pb-3">
          <div v-for="field in EXTRA_FIELDS" :key="field.key" class="space-y-1">
            <Label class="text-[11px] text-muted-foreground">{{ t(field.labelKey) }}</Label>
            <NumberInput
              :model-value="tier[field.key] ?? null"
              :min="0"
              step="0.01"
              :disabled="disabled"
              :placeholder="t('models.pricingTierInherit')"
              class="h-8 text-data text-xs"
              @update:model-value="(value) => setRate(index, field.key, value)"
            />
          </div>
        </div>
      </div>

      <p v-if="duplicateThresholds.length" class="text-[11px] text-status-warning">
        {{ t("models.pricingTiersDuplicate", { thresholds: duplicateThresholds.join(", ") }) }}
      </p>
    </div>
  </div>
</template>
