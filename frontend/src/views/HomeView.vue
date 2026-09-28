<script setup lang="ts">
import { getLocalTimeZone, today } from "@internationalized/date";
import { BarChart3, CalendarIcon, Cpu, Layers, Loader2, Settings } from "@lucide/vue";
import { toast } from "vue-sonner";
import { computed, onMounted, ref, watch, type Ref } from "vue";
import { useWindowSize } from "@vueuse/core";

const { width: windowWidth } = useWindowSize();
import { useI18n } from "vue-i18n";
import { useRouter } from "vue-router";
import PageHeader from "@/components/common/PageHeader.vue";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { RangeCalendar } from "@/components/ui/range-calendar";
import { Separator } from "@/components/ui/separator";
import EmptyState from "@/components/common/EmptyState.vue";
import DashboardSkeleton from "@/components/common/DashboardSkeleton.vue";
import { Select, SelectContent, SelectItem, SelectTrigger } from "@/components/ui/select";
import {
  UsageByModel,
  UsageByProvider,
  UsageTrendsChart,
  UsageMetricStrip,
  TokenBreakdownChart,
} from "@/components/usage";
import { logsApi } from "@/services/api/logs";
import { getModelIconUrl, isMonoIcon } from "@/utils/icons";
import type { HourlyUsageBucket, UsageStatsResponse } from "@/types/schemas";
import type { DateRange } from "reka-ui";

const { t } = useI18n();
const router = useRouter();

// Date range state
type PresetType = "24h" | "7d" | "30d" | "90d" | "custom";
const presetSelected = ref<PresetType>("24h");
const now = today(getLocalTimeZone());
const dateRange = ref({
  start: now,
  end: now,
}) as Ref<DateRange>;
const tempDateRange = ref({ start: now, end: now }) as Ref<DateRange>;
const isDatePickerOpen = ref(false);

watch(isDatePickerOpen, (open) => {
  if (open) {
    tempDateRange.value = { start: dateRange.value.start, end: dateRange.value.end };
  }
});

// Computed date range label for display
const dateRangeLabel = computed(() => {
  if (presetSelected.value === "24h") return t("home.last24Hours");
  if (presetSelected.value === "7d") return t("home.last7Days");
  if (presetSelected.value === "30d") return t("home.last30Days");
  if (presetSelected.value === "90d") return t("home.last90Days");
  if (dateRange.value.start && dateRange.value.end) {
    const startStr = formatDateValue(dateRange.value.start);
    const endStr = formatDateValue(dateRange.value.end);
    return `${startStr} - ${endStr}`;
  }
  return t("home.dateRange");
});

// Helper to format DateValue to YYYY-MM-DD string
function formatDateValue(date: { year: number; month: number; day: number } | undefined): string {
  if (!date) return "";
  const year = date.year;
  const month = String(date.month).padStart(2, "0");
  const day = String(date.day).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

// Helper to calculate date range based on preset
function getPresetDates(preset: PresetType) {
  const now = today(getLocalTimeZone());
  let days: number;

  switch (preset) {
    case "24h":
      // "Last 24 hours" at day granularity = today; the hourly token
      // breakdown chart is what makes this preset useful.
      return { start: now, end: now };
    case "7d":
      days = 7;
      break;
    case "30d":
      days = 30;
      break;
    case "90d":
      days = 90;
      break;
    default:
      days = 7;
  }
  return {
    start: now.subtract({ days: days - 1 }),
    end: now,
  };
}

// Usage stats data
const usageStats = ref<UsageStatsResponse | null>(null);
const isLoadingUsage = ref(false);
const usageError = ref<string | null>(null);

// Model drill-down filter. `null` = all models. `availableModels` persists
// across fetches so the dropdown doesn't collapse to the selected model when
// the dashboard data is filtered down to it.
const ALL_MODELS = "__all__";
const selectedModel = ref<string | null>(null);
const availableModels = ref<Array<{ model: string; provider: string }>>([]);

// Icon resolution as in the Chat page model selector: model-name match first,
// provider icon as fallback (usage rows always carry a provider). Built once
// per model list — `getModelIconUrl` walks the well-known icon table, so
// resolving it per option per render is wasted work.
const modelIcons = computed(
  () => new Map(availableModels.value.map((m) => [m.model, getModelIconUrl(m.model, m.provider)]))
);

/** Select options: the "all models" sentinel, then every known model. */
const modelOptions = computed(() => [
  { value: ALL_MODELS, model: null as string | null },
  ...availableModels.value.map((entry) => ({ value: entry.model, model: entry.model })),
]);

// Hourly token breakdown — only fetched for short ranges where an hour axis
// is readable; longer ranges fall back to the daily buckets already present
// in the usage-stats response.
const hourlyBuckets = ref<HourlyUsageBucket[]>([]);

const useHourlyGranularity = computed(() => {
  const start = formatDateValue(dateRange.value.start);
  const end = formatDateValue(dateRange.value.end);
  if (!start || !end) return false;
  const days = (Date.parse(end) - Date.parse(start)) / 86_400_000;
  return days <= 2;
});

const tokenBreakdownBuckets = computed<HourlyUsageBucket[]>(() => {
  if (useHourlyGranularity.value) return hourlyBuckets.value;
  // Daily buckets feed the same chart; only the key and the granularity differ.
  return (usageStats.value?.daily_usage ?? []).map((day) => ({
    bucket: day.date,
    requests: day.requests,
    cost: day.cost,
    input_tokens: day.input_tokens,
    output_tokens: day.output_tokens,
    cache_creation_tokens: day.cache_creation_tokens,
    cache_read_tokens: day.cache_read_tokens,
    cached_prompt_tokens: day.cached_prompt_tokens,
  }));
});

async function fetchHourlyUsage() {
  if (!useHourlyGranularity.value) {
    hourlyBuckets.value = [];
    return;
  }
  try {
    const response = await logsApi.getHourlyUsage({
      start_date: formatDateValue(dateRange.value.start),
      end_date: formatDateValue(dateRange.value.end),
      model: selectedModel.value ?? undefined,
    });
    hourlyBuckets.value = response.buckets;
  } catch (error) {
    // Non-fatal: the breakdown chart degrades to the empty state.
    console.error("Failed to fetch hourly usage:", error);
    hourlyBuckets.value = [];
  }
}

function selectModel(model: string | null) {
  selectedModel.value = model;
  fetchUsageStats();
}

function onModelFilterChange(value: string) {
  // The sentinel only means "all models" while no real model carries that id,
  // so a provider model named exactly ALL_MODELS stays selectable.
  const isSentinel =
    value === ALL_MODELS && !availableModels.value.some((m) => m.model === ALL_MODELS);
  const model = isSentinel ? null : value;
  if (model !== selectedModel.value) {
    selectModel(model);
  }
}

async function fetchUsageStats() {
  try {
    isLoadingUsage.value = true;
    usageError.value = null;

    const start_date = formatDateValue(dateRange.value.start);
    const end_date = formatDateValue(dateRange.value.end);

    const [stats] = await Promise.all([
      logsApi.getUsageStats({
        start_date,
        end_date,
        model: selectedModel.value ?? undefined,
      }),
      fetchHourlyUsage(),
    ]);
    usageStats.value = stats;

    const merged = new Map(availableModels.value.map((m) => [m.model, m.provider]));
    for (const item of stats.by_model) {
      if (!merged.has(item.model)) merged.set(item.model, item.provider);
    }
    availableModels.value = [...merged.entries()]
      .map(([model, provider]) => ({ model, provider }))
      .sort((a, b) => a.model.localeCompare(b.model));
  } catch (error) {
    console.error("Failed to fetch usage stats:", error);
    const message = error instanceof Error ? error.message : t("errors.fetchFailed");
    // Background refresh: when stale data is already on screen, keep the panel
    // and report the failure as a toast instead of nuking the dashboard.
    if (usageStats.value) {
      toast.error(message);
    } else {
      usageError.value = message;
    }
  } finally {
    isLoadingUsage.value = false;
  }
}

// Handle preset selection
function selectPreset(preset: PresetType) {
  presetSelected.value = preset;
  dateRange.value = getPresetDates(preset);
  isDatePickerOpen.value = false;
  fetchUsageStats();
}

// Apply custom date range
function applyCustomRange() {
  if (tempDateRange.value.start && tempDateRange.value.end) {
    dateRange.value = { start: tempDateRange.value.start, end: tempDateRange.value.end };
    presetSelected.value = "custom";
    isDatePickerOpen.value = false;
    fetchUsageStats();
  }
}

onMounted(() => {
  fetchUsageStats();
});
</script>

<template>
  <AppLayout layoutMode="full">
    <template #header>
      <header class="config-header-bar px-4 sm:px-6 py-4">
        <PageHeader
          :title="t('home.usageTitle')"
          :description="t('home.usageOverview')"
          :icon="BarChart3"
        >
          <template #actions>
            <!-- Model filter — same Select pattern + styling as the Chat page
                 model selector (icon chip + mono name, h-8 ghost trigger). -->
            <Select
              :model-value="selectedModel ?? ALL_MODELS"
              @update:model-value="onModelFilterChange($event as string)"
            >
              <SelectTrigger
                :aria-label="t('home.filterByModel')"
                class="w-auto border border-border/60 bg-transparent hover:bg-muted/10 shadow-none focus-visible:ring-1 focus-visible:ring-foreground focus-visible:ring-offset-0 rounded-md h-8 px-2.5 gap-3 transition-colors text-foreground flex items-center min-w-0"
              >
                <div class="flex items-center gap-2 min-w-0">
                  <div
                    class="w-4 h-4 rounded flex items-center justify-center shrink-0 overflow-hidden bg-background/50 border border-border/40"
                  >
                    <img
                      v-if="selectedModel && modelIcons.get(selectedModel)"
                      :src="modelIcons.get(selectedModel)!"
                      alt=""
                      aria-hidden="true"
                      :class="[
                        isMonoIcon(selectedModel) ? 'icon-mono' : null,
                        'w-3.5 h-3.5 object-contain',
                      ]"
                      loading="lazy"
                    />
                    <Layers v-else-if="!selectedModel" class="w-2.5 h-2.5 text-muted-foreground" />
                    <Cpu v-else class="w-2.5 h-2.5 text-muted-foreground" />
                  </div>
                  <!-- Model IDs stay mono (data); the "All Models" UI label uses
                       the body font so it matches the date-range button. -->
                  <span
                    class="truncate min-w-0 max-w-[200px]"
                    :class="selectedModel ? 'font-mono text-[11px] font-medium' : 'text-sm'"
                  >
                    {{ selectedModel ?? t("home.allModels") }}
                  </span>
                </div>
              </SelectTrigger>
              <SelectContent class="min-w-64 max-h-72 rounded-md border border-border/80 shadow-md">
                <SelectItem
                  v-for="option in modelOptions"
                  :key="option.value"
                  :value="option.value"
                  class="rounded-sm"
                >
                  <div class="flex items-center gap-2.5 min-w-0 w-full py-0.5">
                    <div
                      class="w-4 h-4 rounded flex items-center justify-center shrink-0 overflow-hidden bg-background/50 border border-border/40"
                    >
                      <img
                        v-if="option.model && modelIcons.get(option.model)"
                        :src="modelIcons.get(option.model)!"
                        alt=""
                        aria-hidden="true"
                        :class="[
                          isMonoIcon(option.model) ? 'icon-mono' : null,
                          'w-3.5 h-3.5 object-contain',
                        ]"
                        loading="lazy"
                      />
                      <Layers v-else-if="!option.model" class="w-2.5 h-2.5 text-muted-foreground" />
                      <Cpu v-else class="w-2.5 h-2.5 text-muted-foreground" />
                    </div>
                    <!-- Model IDs stay mono (data); the "All Models" label uses
                         the body font so it matches the date-range button. -->
                    <span
                      class="min-w-0 truncate leading-none"
                      :class="
                        option.model
                          ? 'max-w-[40vw] sm:max-w-[250px] text-[11px] font-mono text-foreground'
                          : 'text-sm'
                      "
                      :title="option.model ?? undefined"
                    >
                      {{ option.model ?? t("home.allModels") }}
                    </span>
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
            <Popover v-model:open="isDatePickerOpen">
              <PopoverTrigger as-child>
                <Button
                  variant="outline"
                  class="w-fit justify-start text-left font-normal btn-action bg-background border-border/60 shadow-none backdrop-blur-none hover:bg-accent/60 hover:border-border"
                >
                  <Loader2 v-if="isLoadingUsage && usageStats" class="mr-2 h-4 w-4 animate-spin" />
                  <CalendarIcon v-else class="mr-2 h-4 w-4" />
                  <span>{{ dateRangeLabel }}</span>
                </Button>
              </PopoverTrigger>
              <PopoverContent class="w-[min(92vw,620px)] sm:w-[620px] p-0" align="end">
                <div class="flex flex-col gap-4 p-4">
                  <div class="flex gap-2">
                    <Button
                      :variant="presetSelected === '24h' ? 'default' : 'outline'"
                      size="sm"
                      class="text-xs flex-1"
                      @click="selectPreset('24h')"
                    >
                      {{ t("home.last24Hours") }}
                    </Button>
                    <Button
                      :variant="presetSelected === '7d' ? 'default' : 'outline'"
                      size="sm"
                      class="text-xs flex-1"
                      @click="selectPreset('7d')"
                    >
                      {{ t("home.last7Days") }}
                    </Button>
                    <Button
                      :variant="presetSelected === '30d' ? 'default' : 'outline'"
                      size="sm"
                      class="text-xs flex-1"
                      @click="selectPreset('30d')"
                    >
                      {{ t("home.last30Days") }}
                    </Button>
                    <Button
                      :variant="presetSelected === '90d' ? 'default' : 'outline'"
                      size="sm"
                      class="text-xs flex-1"
                      @click="selectPreset('90d')"
                    >
                      {{ t("home.last90Days") }}
                    </Button>
                  </div>

                  <Separator />

                  <RangeCalendar
                    v-model="tempDateRange"
                    :number-of-months="windowWidth >= 640 ? 2 : 1"
                    class="rounded-md border"
                  />

                  <Button
                    class="w-full"
                    :disabled="!tempDateRange.start || !tempDateRange.end"
                    @click="applyCustomRange"
                  >
                    {{ t("home.apply") }}
                  </Button>
                </div>
              </PopoverContent>
            </Popover>
          </template>
        </PageHeader>
      </header>
    </template>

    <div class="config-content">
      <!-- Loading (initial load only — refreshes keep the panel on screen so
           values crossfade in place instead of the dashboard snapping away).
           The skeleton mirrors the dashboard geometry to avoid a layout jump. -->
      <div v-if="isLoadingUsage && !usageStats" class="h-full overflow-y-auto animate-fade-in">
        <DashboardSkeleton />
      </div>

      <template v-else-if="usageStats">
        <div class="config-scroll stagger-fast">
          <UsageMetricStrip :summary="usageStats.summary" :by-provider="usageStats.by_provider" />

          <div
            class="grid grid-cols-1 lg:grid-cols-2 divide-y lg:divide-y-0 lg:divide-x divide-border/60 items-start border-b border-border/60"
          >
            <TokenBreakdownChart
              :buckets="tokenBreakdownBuckets"
              :granularity="useHourlyGranularity ? 'hour' : 'day'"
            />
            <UsageTrendsChart :daily-usage="usageStats.daily_usage" />
          </div>

          <div
            class="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-border/60 items-start"
          >
            <UsageByProvider :by-provider="usageStats.by_provider" />
            <UsageByModel
              :by-model="usageStats.by_model"
              :selected-model="selectedModel"
              @select="selectModel"
            />
          </div>
        </div>
      </template>

      <!-- Error -->
      <div v-else-if="usageError" class="h-full flex items-center justify-center py-20">
        <EmptyState :text="usageError" :show-retry="true" @retry="fetchUsageStats" />
      </div>

      <!-- Empty / first-run -->
      <div v-else class="h-full flex items-center justify-center py-20">
        <EmptyState
          :icon="BarChart3"
          :text="t('home.emptyStateTitle')"
          :show-cta="true"
          :cta-text="t('home.emptyStateAction')"
          :cta-icon="Settings"
          @click="router.push({ name: 'providers' })"
        >
          <template #description>
            <p class="text-muted-foreground text-sm max-w-md">
              {{ t("home.emptyStateDescription") }}
            </p>
          </template>
        </EmptyState>
      </div>
    </div>
  </AppLayout>
</template>
