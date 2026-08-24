<script setup lang="ts">
import { Filter, TrendingUp, X } from "@lucide/vue";
import { computed, ref, onMounted, onUnmounted } from "vue";
import { useI18n } from "vue-i18n";
import { Bar } from "vue-chartjs";
import type { TooltipItem } from "chart.js";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { formatCostWithPrecision, formatNumberWithSuffix } from "@/utils/format";
import { createCategoricalColorScale } from "@/utils/colorPalette";
import { registerBarChart } from "@/lib/charts";

registerBarChart();

interface DailyModelUsage {
  model: string;
  requests: number;
  cost: number;
  input_tokens: number;
  output_tokens: number;
  cache_creation_tokens: number;
  cache_read_tokens: number;
  cached_prompt_tokens: number;
}

interface Props {
  dailyUsage: Array<{
    date: string;
    requests: number;
    cost: number;
    input_tokens: number;
    output_tokens: number;
    cache_creation_tokens: number;
    cache_read_tokens: number;
    cached_prompt_tokens: number;
    by_model: DailyModelUsage[];
  }>;
}

const props = defineProps<Props>();
const { t } = useI18n();

const isDarkMode = ref(false);

const updateDarkMode = () => {
  isDarkMode.value = document.documentElement.classList.contains("dark");
};

onMounted(() => {
  updateDarkMode();
  const observer = new MutationObserver(updateDarkMode);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
  onUnmounted(() => observer.disconnect());
});

const themeColors = computed(() => ({
  tooltipBg: isDarkMode.value ? "hsl(220 12% 7% / 0.95)" : "hsl(0 0% 100% / 0.95)",
  tooltipTitle: isDarkMode.value ? "hsl(0 0% 96%)" : "hsl(220 8% 5%)",
  tooltipBody: isDarkMode.value ? "hsl(220 5% 62%)" : "hsl(220 4% 38%)",
  tooltipBorder: isDarkMode.value ? "hsl(220 10% 22%)" : "hsl(220 4% 88%)",
  gridColor: isDarkMode.value ? "hsl(220 10% 22% / 0.3)" : "hsl(220 4% 88% / 0.8)",
  tickColor: isDarkMode.value ? "hsl(220 5% 62%)" : "hsl(220 4% 38%)",
}));

// Chart type state (Requests, Cost, Tokens)
const chartType = ref<"requests" | "cost" | "tokens">("requests");

// Human label for a chart-type value, shared by the aria label and the toggle
// buttons so the two never drift apart.
const chartTypeLabel = (type: (typeof chartType)["value"]): string => {
  switch (type) {
    case "requests":
      return t("home.totalRequests");
    case "cost":
      return t("home.costUsd");
    default:
      return t("home.totalTokens");
  }
};

// Selected models filter (empty = all)
const selectedModels = ref<Set<string>>(new Set());

// Show top N models toggle
const showTopN = ref<number | null>(5);

// Categorical series colors come from the shared scale in @/utils/colorPalette.
// The scale ranks models by usage (allModels order) and hands its 10 prime,
// maximally separated slot colors to the top 10 — matching the chart's hard
// 10-series cap below. A model keeps its color across Top-5/Top-10/filter
// toggles and reloads, and chart series match the hue dots in the filter.
const colorScale = computed(() => createCategoricalColorScale(allModels.value));

// Format date for display
const formatDate = (dateStr: string): string => {
  const date = new Date(dateStr);
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
};

// Get total usage for each model (for ranking)
const modelTotals = computed(() => {
  const totals = new Map<string, number>();
  props.dailyUsage.forEach((day) => {
    day.by_model.forEach((modelData) => {
      const current = totals.get(modelData.model) || 0;
      totals.set(modelData.model, current + modelData.requests);
    });
  });
  return totals;
});

// Get all unique models from the data, sorted by total usage
const allModels = computed(() => {
  const models = Array.from(modelTotals.value.entries());
  models.sort((a, b) => b[1] - a[1]);
  return models.map(([name]) => name);
});

// Top N models
const topNModels = computed(() => {
  if (!showTopN.value) return allModels.value;
  return allModels.value.slice(0, showTopN.value);
});

// Hard cap: the chart never renders more than 10 series, which is exactly
// what the color scale's prime hue slots guarantee distinct colors for.
const MAX_DISPLAY_MODELS = 10;

// Display models (filtered by selection or top N, capped at MAX_DISPLAY_MODELS)
const displayModels = computed(() => {
  if (selectedModels.value.size > 0) {
    return allModels.value.filter((m) => selectedModels.value.has(m)).slice(0, MAX_DISPLAY_MODELS);
  }
  return topNModels.value.slice(0, MAX_DISPLAY_MODELS);
});

// True when the manual selection hit the cap — unchecked rows in the filter
// popover become disabled until the user deselects something.
const selectionAtCap = computed(() => selectedModels.value.size >= MAX_DISPLAY_MODELS);

// Accessible label for screen readers
const chartAriaLabel = computed(() => {
  const typeLabel = chartTypeLabel(chartType.value);
  const modelCount = displayModels.value.length;
  const dateRange =
    props.dailyUsage.length > 0
      ? `${formatDate(props.dailyUsage[0]!.date)} to ${formatDate(props.dailyUsage[props.dailyUsage.length - 1]!.date)}`
      : "";
  return `${typeLabel} chart showing ${modelCount} models from ${dateRange}`;
});

// Toggle model selection (adding is blocked once the cap is reached)
const toggleModel = (model: string) => {
  if (selectedModels.value.has(model)) {
    selectedModels.value.delete(model);
  } else {
    if (selectionAtCap.value) return;
    selectedModels.value.add(model);
  }
  if (selectedModels.value.size > 0) {
    showTopN.value = null;
  }
};

// Select only top N models
const selectTopN = (n: number) => {
  selectedModels.value.clear();
  showTopN.value = n;
};

// Clear all filters
const clearFilters = () => {
  selectedModels.value.clear();
  showTopN.value = 5;
};

// Truncate model name for legend
const truncateModelName = (name: string, maxLength = 20): string => {
  if (name.length <= maxLength) return name;
  return `${name.slice(0, maxLength)}...`;
};

// Prepare chart data based on chart type
const chartData = computed(() => {
  const labels = props.dailyUsage.map((item) => formatDate(item.date));
  const datasets: Array<{
    label: string;
    data: number[];
    backgroundColor: string;
  }> = [];

  displayModels.value.forEach((modelName, _index) => {
    const data = props.dailyUsage.map((day) => {
      const modelData = day.by_model.find((m) => m.model === modelName);
      if (!modelData) return 0;

      switch (chartType.value) {
        case "cost":
          return modelData.cost;
        case "tokens":
          return modelData.input_tokens + modelData.output_tokens;
        default:
          return modelData.requests;
      }
    });

    datasets.push({
      label: truncateModelName(modelName),
      data,
      backgroundColor: colorScale.value(modelName, isDarkMode.value),
    });
  });

  return {
    labels,
    datasets,
  };
});

// Chart options
const chartOptions = computed(() => ({
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      display: true,
      position: "bottom" as const,
      labels: {
        boxWidth: 12,
        padding: 15,
        font: {
          size: 11,
        },
        color: themeColors.value.tickColor,
      },
    },
    tooltip: {
      mode: "index" as const,
      intersect: false,
      backgroundColor: themeColors.value.tooltipBg,
      titleColor: themeColors.value.tooltipTitle,
      bodyColor: themeColors.value.tooltipBody,
      borderColor: themeColors.value.tooltipBorder,
      borderWidth: 1,
      padding: 12,
      cornerRadius: 8,
      displayColors: true,
      boxPadding: 4,
      callbacks: {
        label: (context: TooltipItem<"bar">) => {
          const value = context.raw as number;
          let formattedValue: string;

          switch (chartType.value) {
            case "cost":
              formattedValue = formatCostWithPrecision(value);
              break;
            case "tokens":
              formattedValue = formatNumberWithSuffix(value);
              break;
            default:
              formattedValue = formatNumberWithSuffix(value);
          }

          const label = context.dataset?.label ?? "Unknown";
          return ` ${label}: ${formattedValue}`;
        },
      },
    },
  },
  scales: {
    x: {
      stacked: true,
      grid: {
        display: false,
      },
      ticks: {
        color: themeColors.value.tickColor,
        font: {
          size: 10,
        },
        maxRotation: 0,
        autoSkip: true,
        maxTicksLimit: 7,
      },
    },
    y: {
      stacked: true,
      grid: {
        color: themeColors.value.gridColor,
        drawBorder: false,
      },
      ticks: {
        color: themeColors.value.tickColor,
        font: {
          size: 10,
        },
        // Integer ticks for count series (requests/tokens) so a small
        // total never renders a 0–1 decimal axis that reads as broken. Cost
        // keeps default precision for cents.
        precision: chartType.value === "cost" ? undefined : 0,
        callback: (value: string | number) => {
          const numValue = typeof value === "number" ? value : Number.parseFloat(value);
          if (Number.isNaN(numValue)) return "";

          if (chartType.value === "cost") return `$${numValue}`;
          return formatNumberWithSuffix(numValue);
        },
      },
    },
  },
  interaction: {
    mode: "nearest" as const,
    axis: "x" as const,
    intersect: false,
  },
}));
</script>

<template>
  <section class="flex flex-col border-b border-border/60">
    <!-- Section header row (toolbar-style, hairline beneath) -->
    <div class="flex items-center justify-between px-4 sm:px-6 py-3 border-b border-border/60">
      <h2 class="flex items-center gap-1.5 text-sm md:text-base font-semibold text-foreground">
        <TrendingUp class="w-4 h-4 text-action-amber" />
        {{ t("home.usageTrends") }}
      </h2>

      <div class="flex items-center gap-2">
        <!-- Monochrome segmented chart-type toggle (replaces per-type rainbow tabs) -->
        <div class="inline-flex items-center rounded-md border border-border/60 bg-muted/40 p-0.5">
          <button
            v-for="opt in ['requests', 'cost', 'tokens'] as const"
            :key="opt"
            class="px-2.5 py-1 text-xs font-medium rounded-[6px] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
            :class="
              chartType === opt
                ? 'bg-background text-foreground font-semibold shadow-xs'
                : 'text-muted-foreground hover:text-foreground'
            "
            :aria-pressed="chartType === opt"
            @click="chartType = opt"
          >
            {{ chartTypeLabel(opt) }}
          </button>
        </div>

        <!-- Model filter (unchanged behavior) -->
        <Popover v-if="allModels.length > 0">
          <PopoverTrigger as-child>
            <Button variant="outline" size="sm" class="h-8 gap-1.5 text-xs bg-background">
              <Filter class="w-3.5 h-3.5 text-muted-foreground" />
              <span v-if="selectedModels.size === 0 && showTopN">{{
                t("home.topN", { n: showTopN })
              }}</span>
              <span v-else-if="selectedModels.size > 0"
                >{{ selectedModels.size }} {{ t("models.title") }}</span
              >
              <span v-else>{{ t("common.filterAll") }}</span>
            </Button>
          </PopoverTrigger>
          <PopoverContent class="w-80 p-0" align="end">
            <div class="p-3 border-b border-border/50">
              <div class="flex items-center justify-between gap-2 mb-2">
                <span class="text-sm font-medium">{{ t("logs.model") }}</span>
                <span class="flex items-center gap-2">
                  <span class="text-xs text-muted-foreground">
                    {{ t("home.maxModels", { n: MAX_DISPLAY_MODELS }) }}
                  </span>
                  <Button
                    v-if="selectedModels.size > 0"
                    variant="ghost"
                    size="sm"
                    class="h-6 text-xs"
                    @click="clearFilters"
                  >
                    <X class="w-3.5 h-3.5 mr-1" />
                    {{ t("common.clearFilters") }}
                  </Button>
                </span>
              </div>
              <div class="flex gap-1 flex-wrap">
                <Badge
                  variant="outline"
                  class="cursor-pointer text-xs min-h-8 px-3 flex items-center"
                  :class="
                    showTopN === 5 && selectedModels.size === 0
                      ? 'bg-primary text-primary-foreground border-primary'
                      : ''
                  "
                  @click="selectTopN(5)"
                  >{{ t("home.topN", { n: 5 }) }}</Badge
                >
                <Badge
                  variant="outline"
                  class="cursor-pointer text-xs min-h-8 px-3 flex items-center"
                  :class="
                    showTopN === 10 && selectedModels.size === 0
                      ? 'bg-primary text-primary-foreground border-primary'
                      : ''
                  "
                  @click="selectTopN(10)"
                  >{{ t("home.topN", { n: 10 }) }}</Badge
                >
              </div>
            </div>
            <div class="max-h-60 overflow-y-auto p-2 space-y-1">
              <button
                v-for="model in allModels"
                :key="model"
                class="w-full flex items-center justify-between px-3 py-2 rounded text-sm transition-colors min-h-9"
                :class="
                  !selectedModels.has(model) && selectionAtCap
                    ? 'opacity-40 cursor-not-allowed'
                    : 'hover:bg-muted'
                "
                :disabled="!selectedModels.has(model) && selectionAtCap"
                :aria-pressed="selectedModels.has(model)"
                @click="toggleModel(model)"
              >
                <span class="flex items-center gap-2 truncate mr-2 text-left" :title="model">
                  <span
                    class="w-2.5 h-2.5 rounded-full shrink-0 ring-1 ring-inset ring-muted-foreground/20 dark:ring-muted-foreground/30"
                    aria-hidden="true"
                    :style="{
                      backgroundColor: colorScale(model, isDarkMode),
                    }"
                  />
                  {{ model }}
                </span>
                <div
                  class="w-4 h-4 rounded border flex items-center justify-center shrink-0"
                  :class="
                    selectedModels.has(model)
                      ? 'bg-primary border-primary'
                      : 'border-muted-foreground/30'
                  "
                >
                  <svg
                    v-if="selectedModels.has(model)"
                    class="w-3 h-3 text-primary-foreground"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    stroke-width="3"
                  >
                    <path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7" />
                  </svg>
                </div>
              </button>
            </div>
          </PopoverContent>
        </Popover>
      </div>
    </div>

    <!-- Chart panel (flush, no Card) -->
    <div class="px-4 sm:px-6 pb-4 pt-3">
      <div
        v-if="dailyUsage.length > 0"
        class="relative w-full h-72 md:h-85"
        role="img"
        :aria-label="chartAriaLabel"
      >
        <Bar :data="chartData" :options="chartOptions" />
      </div>
      <div v-else class="py-12 text-center text-muted-foreground text-sm">
        {{ t("logs.noLogs") }}
      </div>
    </div>
  </section>
</template>
