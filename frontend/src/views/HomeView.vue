<script setup lang="ts">
import { getLocalTimeZone, today } from "@internationalized/date";
import { BarChart3, CalendarIcon, Loader2, Settings } from "@lucide/vue";
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
import {
  UsageByModel,
  UsageByProvider,
  UsageTrendsChart,
  UsageMetricStrip,
} from "@/components/usage";
import { logsApi } from "@/services/api/logs";
import type { UsageStatsResponse } from "@/types/schemas";
import type { DateRange } from "reka-ui";

const { t } = useI18n();
const router = useRouter();

// Date range state
type PresetType = "7d" | "30d" | "90d" | "custom";
const presetSelected = ref<PresetType>("7d");
const now = today(getLocalTimeZone());
const dateRange = ref({
  start: now.subtract({ days: 6 }),
  end: now,
}) as Ref<DateRange>;
const tempDateRange = ref({ start: now.subtract({ days: 6 }), end: now }) as Ref<DateRange>;
const isDatePickerOpen = ref(false);

watch(isDatePickerOpen, (open) => {
  if (open) {
    tempDateRange.value = { start: dateRange.value.start, end: dateRange.value.end };
  }
});

// Computed date range label for display
const dateRangeLabel = computed(() => {
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

async function fetchUsageStats() {
  try {
    isLoadingUsage.value = true;
    usageError.value = null;

    const start_date = formatDateValue(dateRange.value.start);
    const end_date = formatDateValue(dateRange.value.end);

    usageStats.value = await logsApi.getUsageStats({
      start_date,
      end_date,
    });
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

          <UsageTrendsChart :daily-usage="usageStats.daily_usage" />

          <div
            class="grid grid-cols-1 lg:grid-cols-2 divide-y lg:divide-y-0 lg:divide-x divide-border/60 items-start"
          >
            <UsageByProvider :by-provider="usageStats.by_provider" />
            <UsageByModel :by-model="usageStats.by_model" />
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
