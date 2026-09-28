<script setup lang="ts">
import { BarChart3 } from "@lucide/vue";
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { Bar } from "vue-chartjs";
import type { TooltipItem } from "chart.js";
import { baseBarChartOptions, useChartTheme } from "@/composables/useChartTheme";
import { formatChartDate, formatNumberWithSuffix, formatTokens } from "@/utils/format";
import type { HourlyUsageBucket } from "@/types/schemas";
import { registerBarChart } from "@/lib/charts";

registerBarChart();

/** Bucket width — drives the axis labels and the tooltip range header. */
type Granularity = "hour" | "day";

interface Props {
  buckets: HourlyUsageBucket[];
  granularity: Granularity;
}

const props = defineProps<Props>();
const { t } = useI18n();
const { themeColors } = useChartTheme();

const isHourly = computed(() => props.granularity === "hour");

/** Per-bucket derived series: cache hit / cache miss / output. */
interface BreakdownRow {
  key: string;
  cacheHit: number;
  cacheMiss: number;
  output: number;
  total: number;
}

const rows = computed<BreakdownRow[]>(() =>
  props.buckets.map((b) => {
    const cacheHit = b.cache_read_tokens + b.cached_prompt_tokens;
    const cacheMiss = Math.max(b.input_tokens - cacheHit, 0);
    const output = b.output_tokens;
    return {
      key: b.bucket,
      cacheHit,
      cacheMiss,
      output,
      total: cacheHit + cacheMiss + output,
    };
  })
);

const totalTokens = computed(() => rows.value.reduce((sum, r) => sum + r.total, 0));

/** Short axis label: "20:00" for hourly buckets, "Mar 1" for daily. */
const axisLabel = (key: string): string => {
  if (isHourly.value) return key.slice(11, 16);
  return formatChartDate(key);
};

/** Tooltip header: "20:00 ~ 21:00" for hourly buckets, full date for daily. */
const rangeLabel = (key: string): string => {
  if (isHourly.value) {
    const hour = key.slice(11, 13);
    return `${hour}:00 ~ ${hour}:59`;
  }
  return new Date(key).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
};

// Cache hit / cache miss / output get three different hues rather than one hue
// at three alphas: adjacent stacked segments have to stay separable, and a
// lightness ramp of a single low-saturation tint washes out in light mode.
const chartData = computed(() => ({
  labels: rows.value.map((r) => axisLabel(r.key)),
  datasets: [
    {
      label: t("home.inputCacheHit"),
      data: rows.value.map((r) => r.cacheHit),
      backgroundColor: themeColors.value.actionBlue,
    },
    {
      label: t("home.inputCacheMiss"),
      data: rows.value.map((r) => r.cacheMiss),
      backgroundColor: themeColors.value.actionViolet,
    },
    {
      label: t("home.outputTokensShort"),
      data: rows.value.map((r) => r.output),
      backgroundColor: themeColors.value.actionAmber,
    },
  ],
}));

const chartOptions = computed(() =>
  baseBarChartOptions(themeColors.value, {
    interactionMode: "index",
    xMaxTicksLimit: isHourly.value ? 8 : 10,
    yPrecision: 0,
    yTickLabel: formatNumberWithSuffix,
    tooltip: {
      // Header mirrors the reference design: bucket range + bucket total.
      title: (items: TooltipItem<"bar">[]) => {
        const first = items[0];
        if (!first) return "";
        const row = rows.value[first.dataIndex];
        if (!row) return "";
        return `${rangeLabel(row.key)}    ${formatTokens(row.total)}`;
      },
      label: (context: TooltipItem<"bar">) => {
        const value = context.raw as number;
        const label = context.dataset?.label ?? "";
        return ` ${label}: ${formatTokens(value)}`;
      },
    },
  })
);

const chartAriaLabel = computed(() =>
  t("home.tokenBreakdownAria", {
    buckets: rows.value.length,
    total: formatTokens(totalTokens.value),
  })
);
</script>

<template>
  <section class="flex flex-col">
    <div class="flex items-center justify-between px-4 sm:px-6 py-3 border-b border-border/60">
      <h2 class="flex items-baseline gap-2 text-sm md:text-base font-semibold text-foreground">
        <BarChart3 class="w-4 h-4 self-center text-action-blue" />
        {{ t("home.tokenBreakdownTitle") }}
        <span class="font-mono text-xs font-normal text-muted-foreground tabular-nums">
          {{ formatTokens(totalTokens) }}
        </span>
      </h2>
    </div>

    <div class="px-4 sm:px-6 pb-4 pt-3">
      <div v-if="rows.length > 0" class="relative w-full h-72 md:h-85">
        <Bar :data="chartData" :options="chartOptions" :aria-label="chartAriaLabel" />
      </div>
      <div v-else class="py-12 text-center text-muted-foreground text-sm">
        {{ t("home.tokenBreakdownEmpty") }}
      </div>
    </div>
  </section>
</template>
