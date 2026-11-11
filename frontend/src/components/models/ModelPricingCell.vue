<script setup lang="ts">
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import PricingRateList from "@/components/models/PricingRateList.vue";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { costRange, type ModelCostKey } from "@/utils/modelPricing";
import {
  formatCost,
  serviceTierRateRows,
  tierRateRows,
  windowRateRows,
} from "@/utils/pricingTiers";
import type {
  ModelRead,
  PricingTier,
  PricingTimeWindow,
  ServiceTierPricing,
} from "@/types/schemas";

/**
 * Pricing cell for the models table/list.
 *
 * Two display modes:
 * - `field="input" | "output" | "cached"`: a single scannable mono value (or
 *   "min–max" range when provider overrides differ) — used by the per-column
 *   table layout.
 * - default: a stacked IN/OUT/CACHED block — used by the list layout. The
 *   CACHED row renders as "—" when no cached-read price is configured.
 *
 * Either way the cell stays a popover trigger that opens the full pricing
 * breakdown across all pricing dimensions (reasoning, cached write, audio,
 * image, ...) and every price overlay the model or its providers carry: context
 * tiers, time windows and service tiers.
 */

const props = defineProps<{
  model: ModelRead;
  field?: "input" | "output" | "cached";
}>();

const { t, locale } = useI18n();

interface CostField {
  key: ModelCostKey;
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
  { key: "reasoning_cost_per_1m", labelKey: "models.reasoningCost" },
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

/** Effective price as a single value or "min–max" range. */
function rangeText(key: ModelCostKey): string | null {
  const range = costRange(props.model, key);
  if (range == null) return null;
  return range.min === range.max
    ? formatCost(range.min)
    : `${formatCost(range.min)}–${formatCost(range.max)}`;
}

const inputText = computed(() => rangeText("input_cost_per_1m"));
const outputText = computed(() => rangeText("output_cost_per_1m"));
const cachedText = computed(() => rangeText("cached_read_cost_per_1m"));

const FIELD_TEXTS = { input: inputText, output: outputText, cached: cachedText } as const;

/** The single value shown when `field` is set. */
const fieldText = computed(() => (props.field ? FIELD_TEXTS[props.field].value : null));

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

/** Model-level context tiers (empty when none configured). */
const defaultTiers = computed<PricingTier[]>(() => props.model.pricing_tiers ?? []);

/** Model-level time windows (empty when none configured). */
const defaultWindows = computed<PricingTimeWindow[]>(() => props.model.pricing_time_windows ?? []);

/** Model-level service-tier rates (empty when none configured). */
const defaultServiceTiers = computed<ServiceTierPricing>(
  () => props.model.service_tier_pricing ?? {}
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
      tiers: p.pricing_tiers ?? [],
      windows: p.pricing_time_windows ?? [],
      serviceTiers: p.service_tier_pricing ?? {},
    }))
    .filter(
      (g) =>
        g.rows.length > 0 ||
        g.tiers.length > 0 ||
        g.windows.length > 0 ||
        Object.keys(g.serviceTiers).length > 0
    )
);

const hasTiers = computed(
  () => defaultTiers.value.length > 0 || providerGroups.value.some((g) => g.tiers.length > 0)
);

const hasWindows = computed(
  () => defaultWindows.value.length > 0 || providerGroups.value.some((g) => g.windows.length > 0)
);

const hasServiceTiers = computed(
  () =>
    Object.keys(defaultServiceTiers.value).length > 0 ||
    providerGroups.value.some((g) => Object.keys(g.serviceTiers).length > 0)
);

/** Short labels for extra priced dimensions (cached, reasoning, overlays, ...). */
const activeDims = computed(() => {
  const dims = new Set<string>();
  for (const f of COST_FIELDS) {
    if (f.dimKey && costRange(props.model, f.key) != null) dims.add(t(f.dimKey));
  }
  if (hasTiers.value) dims.add(t("models.pricingTiersSection"));
  if (hasWindows.value) dims.add(t("models.pricingWindowsSection"));
  if (hasServiceTiers.value) dims.add(t("models.serviceTiersSection"));
  return [...dims];
});

const extraDimsTitle = computed(() =>
  activeDims.value.length
    ? t("models.morePricingDims", { dims: activeDims.value.join(", ") })
    : undefined
);

const hasAnyPricing = computed(
  () =>
    defaultRows.value.length > 0 ||
    providerGroups.value.length > 0 ||
    hasTiers.value ||
    hasWindows.value ||
    hasServiceTiers.value
);
</script>

<template>
  <Popover v-if="hasAnyPricing">
    <PopoverTrigger as-child>
      <button
        type="button"
        class="rounded-md px-1.5 py-1 -my-1 cursor-pointer transition-colors hover:bg-muted/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring/60"
        :class="field ? 'text-right' : 'w-full text-right'"
      >
        <!-- Single-column mode: one clear mono value -->
        <template v-if="field">
          <span class="text-data text-xs font-medium text-foreground">{{ fieldText ?? "—" }}</span>
          <Tooltip v-if="activeDims.length">
            <TooltipTrigger as-child>
              <span class="ml-0.5 align-super text-[9px] font-medium text-muted-foreground"
                >+{{ activeDims.length }}</span
              >
            </TooltipTrigger>
            <TooltipContent>{{ extraDimsTitle }}</TooltipContent>
          </Tooltip>
        </template>

        <!-- Stacked mode: labeled IN/OUT lines -->
        <div v-else class="text-data text-xs leading-snug">
          <div class="flex items-baseline justify-end gap-1.5">
            <span
              class="text-[10px] font-sans font-medium uppercase tracking-wider text-muted-foreground"
              >{{ t("models.inputShort") }}</span
            >
            <span class="font-medium text-foreground">{{ inputText ?? "—" }}</span>
          </div>
          <div class="flex items-baseline justify-end gap-1.5">
            <span
              class="text-[10px] font-sans font-medium uppercase tracking-wider text-muted-foreground"
              >{{ t("models.outputShort") }}</span
            >
            <span class="font-medium text-foreground">{{ outputText ?? "—" }}</span>
          </div>
          <div class="flex items-baseline justify-end gap-1.5">
            <span
              class="text-[10px] font-sans font-medium uppercase tracking-wider text-muted-foreground"
              >{{ t("models.cachedShort") }}</span
            >
            <span class="font-medium text-foreground">{{ cachedText ?? "—" }}</span>
          </div>
          <Tooltip v-if="activeDims.length">
            <TooltipTrigger as-child>
              <div
                class="mt-0.5 text-[10px] font-sans font-medium uppercase tracking-wider text-muted-foreground"
              >
                +{{ activeDims.length }} {{ t("models.moreDims") }}
              </div>
            </TooltipTrigger>
            <TooltipContent>{{ extraDimsTitle }}</TooltipContent>
          </Tooltip>
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
        <section v-if="defaultRows.length || defaultTiers.length">
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
          <PricingRateList
            v-if="defaultTiers.length"
            :heading="t('models.pricingTiersSection')"
            :rows="tierRateRows(t, defaultTiers)"
          />
          <PricingRateList
            v-if="defaultWindows.length"
            :heading="t('models.pricingWindowsSection')"
            :rows="windowRateRows(t, defaultWindows, locale)"
          />
          <PricingRateList
            v-if="Object.keys(defaultServiceTiers).length"
            :heading="t('models.serviceTiersSection')"
            :rows="serviceTierRateRows(t, defaultServiceTiers)"
          />
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
          <PricingRateList
            v-if="group.tiers.length"
            :heading="t('models.pricingTiersSection')"
            :rows="tierRateRows(t, group.tiers)"
            dense
          />
          <PricingRateList
            v-if="group.windows.length"
            :heading="t('models.pricingWindowsSection')"
            :rows="windowRateRows(t, group.windows, locale)"
            dense
          />
          <PricingRateList
            v-if="Object.keys(group.serviceTiers).length"
            :heading="t('models.serviceTiersSection')"
            :rows="serviceTierRateRows(t, group.serviceTiers)"
            dense
          />
        </section>
      </div>
    </PopoverContent>
  </Popover>

  <span v-else class="text-data text-xs text-muted-foreground">—</span>
</template>
