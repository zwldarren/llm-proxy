<script setup lang="ts">
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import type { ModelRead } from "@/types/schemas";

/**
 * Rich pricing cell for the models table.
 * Shows the effective input/output price (range-aware across provider-level
 * overrides and the model default) and opens a popover with the full pricing
 * breakdown across all pricing dimensions.
 */

const props = defineProps<{ model: ModelRead }>();

const { t } = useI18n();

type CostKey =
  | "input_cost_per_1m"
  | "output_cost_per_1m"
  | "cached_read_cost_per_1m"
  | "cached_write_cost_per_1m"
  | "audio_input_cost_per_1m"
  | "audio_output_cost_per_1m"
  | "image_input_cost_per_1m"
  | "cost_per_image"
  | "audio_cost_per_minute"
  | "tts_cost_per_1m_chars"
  | "web_search_cost_per_1k";

interface CostField {
  key: CostKey;
  labelKey: string;
  /** Short badge shown in the compact cell when this dimension is priced. */
  dimKey?: string;
}

const COST_FIELDS: CostField[] = [
  { key: "input_cost_per_1m", labelKey: "models.inputCost" },
  { key: "output_cost_per_1m", labelKey: "models.outputCost" },
  {
    key: "cached_read_cost_per_1m",
    labelKey: "models.cachedReadCost",
    dimKey: "models.dimCached",
  },
  {
    key: "cached_write_cost_per_1m",
    labelKey: "models.cachedWriteCost",
    dimKey: "models.dimCached",
  },
  {
    key: "audio_input_cost_per_1m",
    labelKey: "models.audioInputCost",
    dimKey: "models.dimAudio",
  },
  {
    key: "audio_output_cost_per_1m",
    labelKey: "models.audioOutputCost",
    dimKey: "models.dimAudio",
  },
  {
    key: "image_input_cost_per_1m",
    labelKey: "models.imageInputCost",
    dimKey: "models.dimImage",
  },
  { key: "cost_per_image", labelKey: "models.costPerImage", dimKey: "models.dimImage" },
  {
    key: "audio_cost_per_minute",
    labelKey: "models.audioCostPerMinute",
    dimKey: "models.dimAudio",
  },
  { key: "tts_cost_per_1m_chars", labelKey: "models.ttsCostPer1mChars", dimKey: "models.dimTts" },
  {
    key: "web_search_cost_per_1k",
    labelKey: "models.webSearchCostPer1k",
    dimKey: "models.dimSearch",
  },
];

const providers = computed(() => props.model.providers ?? []);

/** All configured values for a cost key: provider overrides + model default. */
function valuesFor(key: CostKey): number[] {
  const vals: number[] = [];
  for (const p of providers.value) {
    const v = p[key];
    if (v != null) vals.push(v);
  }
  const d = props.model[key];
  if (d != null) vals.push(d);
  return vals;
}

function formatCost(v: number): string {
  // Trim to at most 6 decimal places without trailing zeros.
  const formatted = Number.parseFloat(v.toFixed(6));
  if (formatted === 0 && v > 0) return `$${v.toExponential(2)}`;
  return `$${formatted}`;
}

/** Effective price as a single value or "min–max" range. */
function rangeText(key: CostKey): string | null {
  const vals = valuesFor(key);
  if (vals.length === 0) return null;
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  return min === max ? formatCost(min) : `${formatCost(min)}–${formatCost(max)}`;
}

const inputText = computed(() => rangeText("input_cost_per_1m"));
const outputText = computed(() => rangeText("output_cost_per_1m"));

/** Short labels for extra priced dimensions (cached, audio, image, ...). */
const activeDims = computed(() => {
  const dims = new Set<string>();
  for (const f of COST_FIELDS) {
    if (f.dimKey && valuesFor(f.key).length > 0) dims.add(t(f.dimKey));
  }
  return [...dims];
});

interface CostRow extends CostField {
  value: number;
}

/** Model-level default pricing rows (only configured dimensions). */
const defaultRows = computed<CostRow[]>(() =>
  COST_FIELDS.flatMap((f) => {
    const v = props.model[f.key];
    return v != null ? [{ ...f, value: v }] : [];
  })
);

/** Per-provider override rows (only providers with at least one override). */
const providerGroups = computed(() =>
  providers.value
    .map((p) => ({
      name: p.provider_name || t("models.unknownProvider"),
      rows: COST_FIELDS.flatMap((f) => {
        const v = p[f.key];
        return v != null ? [{ ...f, value: v }] : [];
      }),
    }))
    .filter((g) => g.rows.length > 0)
);

const hasAnyPricing = computed(
  () => defaultRows.value.length > 0 || providerGroups.value.length > 0
);
</script>

<template>
  <Popover v-if="hasAnyPricing">
    <PopoverTrigger as-child>
      <button
        type="button"
        class="w-full rounded-md px-1.5 py-1 -my-1 text-right cursor-pointer transition-colors hover:bg-muted/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring/60"
        :aria-label="t('models.pricingDetails')"
        :title="t('models.pricingDetails')"
      >
        <div class="text-data text-xs leading-tight">
          <span class="font-medium text-action-blue">{{ inputText ?? "—" }}</span>
          <span class="mx-1 text-muted-foreground/50">/</span>
          <span class="font-medium text-status-success">{{ outputText ?? "—" }}</span>
        </div>
        <div
          v-if="activeDims.length"
          class="mt-0.5 text-[11px] font-medium uppercase tracking-wider text-muted-foreground/80"
        >
          {{ activeDims.join(" · ") }}
        </div>
      </button>
    </PopoverTrigger>
    <PopoverContent align="end" class="w-84 p-0">
      <!-- Header -->
      <div class="border-b border-border/60 bg-muted/10 px-3.5 py-2.5">
        <p class="text-xs font-semibold text-foreground">{{ t("models.pricingDetails") }}</p>
        <p class="mt-0.5 truncate font-mono text-[11px] text-muted-foreground">
          {{ model.name }}
        </p>
      </div>

      <div class="max-h-80 space-y-4 overflow-y-auto p-3.5">
        <!-- Model default pricing -->
        <section v-if="defaultRows.length">
          <p
            class="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground"
          >
            {{ t("models.modelDefault") }}
          </p>
          <div class="divide-y divide-border/40">
            <div
              v-for="row in defaultRows"
              :key="row.key"
              class="flex items-baseline justify-between gap-3 py-1"
            >
              <span class="text-xs text-muted-foreground">{{ t(row.labelKey) }}</span>
              <span class="text-data text-xs text-foreground">{{ formatCost(row.value) }}</span>
            </div>
          </div>
        </section>

        <!-- Per-provider overrides -->
        <section v-for="group in providerGroups" :key="group.name">
          <div class="mb-1.5 flex items-center gap-1.5">
            <span
              class="rounded border border-border/60 bg-background/55 px-1.5 py-0 font-mono text-[11px] text-muted-foreground"
            >
              {{ group.name }}
            </span>
            <span class="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
              {{ t("models.providerOverride") }}
            </span>
          </div>
          <div class="divide-y divide-border/40">
            <div
              v-for="row in group.rows"
              :key="row.key"
              class="flex items-baseline justify-between gap-3 py-1"
            >
              <span class="text-xs text-muted-foreground">{{ t(row.labelKey) }}</span>
              <span class="text-data text-xs text-foreground">{{ formatCost(row.value) }}</span>
            </div>
          </div>
        </section>
      </div>
    </PopoverContent>
  </Popover>

  <span v-else class="text-data text-xs text-muted-foreground">—</span>
</template>
