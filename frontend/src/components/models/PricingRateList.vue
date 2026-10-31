<script setup lang="ts">
import { computed } from "vue";
import type { RateRow } from "@/utils/pricingTiers";

/**
 * Read-only rate rows for one price overlay — context tiers, time windows or
 * service tiers — each row a label (threshold, clock or mode name) and the
 * rates it sets.
 *
 * Shared by the model-default and per-provider sections of the pricing popover,
 * which render identical rows one density step apart.
 */
defineOptions({ name: "PricingRateList" });

const props = withDefaults(
  defineProps<{
    heading: string;
    rows: RateRow[];
    /** Nested sections step the heading down: 10px/mb-1 instead of 11px/mb-1.5. */
    dense?: boolean;
  }>(),
  { dense: false }
);

const sectionClass = computed(() => (props.dense ? "mt-2" : "mt-2.5"));
const headingClass = computed(() => (props.dense ? "mb-1 text-[10px]" : "mb-1.5 text-[11px]"));
</script>

<template>
  <div :class="sectionClass">
    <p class="font-semibold uppercase tracking-wider text-muted-foreground" :class="headingClass">
      {{ heading }}
    </p>
    <div class="divide-y divide-border/40">
      <div
        v-for="row in rows"
        :key="row.key"
        class="flex items-baseline justify-between gap-3 py-1"
      >
        <span class="text-xs text-muted-foreground">{{ row.label }}</span>
        <span class="text-data text-xs text-foreground">{{ row.text }}</span>
      </div>
    </div>
  </div>
</template>
