<script setup lang="ts">
import { AlertTriangle, BarChart3, Gauge, Inbox } from "@lucide/vue";
import { computed, defineAsyncComponent, defineComponent, h, ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import { useBudgetDisplay } from "@/composables/useBudgetDisplay";
import SheetHeaderBand from "@/components/common/SheetHeaderBand.vue";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent } from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { apiKeysApi, type ApiKeyRead, type ApiKeyUsageResponse } from "@/services/api/apiKeys";
import { useApiKeyStore } from "@/stores/apiKeys";
import {
  formatCost,
  formatDate,
  formatDuration,
  formatNumberWithSuffix,
  formatPercentage,
} from "@/utils/format";

interface Props {
  apiKey: ApiKeyRead | null;
}

const props = defineProps<Props>();
const open = defineModel<boolean>("open", { default: false });

const { t } = useI18n();
const apiKeyStore = useApiKeyStore();

// The trend chart pulls chart.js into the bundle; loading it only when the
// sheet renders a chart keeps it out of the API keys page's first load. The
// skeleton matches the chart panel height so the sheet does not jump while
// the chunk loads on first open.
const UsageTrendsChart = defineAsyncComponent({
  loader: () => import("@/components/usage/UsageTrendsChart.vue"),
  loadingComponent: defineComponent({
    name: "UsageTrendsChartSkeleton",
    render: () => h(Skeleton, { class: "h-72 md:h-85 rounded-lg" }),
  }),
});

const loading = ref(false);
const usage = ref<ApiKeyUsageResponse | null>(null);
const loadError = ref(false);
const rangeDays = ref(30);
const rangeOptions = [7, 30, 90] as const;

// Sequence guard: rapid range-preset switches fire overlapping requests with
// no cancellation, so only the latest request may commit its result — an
// older response arriving last must not overwrite newer data.
let latestRequestId = 0;

const loadUsage = async () => {
  if (!props.apiKey) return;
  const requestId = ++latestRequestId;
  loading.value = true;
  loadError.value = false;
  try {
    const end = new Date();
    const start = new Date(end.getTime() - rangeDays.value * 24 * 60 * 60 * 1000);
    const result = await apiKeysApi.getKeyUsage(props.apiKey.name, {
      start_date: start.toISOString(),
      end_date: end.toISOString(),
    });
    if (requestId !== latestRequestId) return;
    usage.value = result;
  } catch (e) {
    if (requestId !== latestRequestId) return;
    loadError.value = true;
    console.error("Failed to load API key usage:", e);
  } finally {
    if (requestId === latestRequestId) loading.value = false;
  }
};

watch(
  [open, rangeDays],
  ([isOpen]) => {
    if (isOpen) loadUsage();
  },
  { immediate: true }
);

const summary = computed(() => usage.value?.summary ?? null);
const hasData = computed(() => (summary.value?.total_requests ?? 0) > 0);
// Only the very first load shows skeletons; a range change keeps the previous
// numbers on screen (dimmed) so the layout does not jump.
const showSkeleton = computed(() => loading.value && !usage.value);

const statCards = computed(() => [
  {
    key: "spend",
    label: t("apiKeys.totalSpend"),
    value: formatCost(summary.value?.total_cost),
  },
  {
    key: "requests",
    label: t("apiKeys.requests"),
    value: formatNumberWithSuffix(summary.value?.total_requests ?? 0),
  },
  {
    key: "input",
    label: t("apiKeys.inputTokens"),
    value: formatNumberWithSuffix(summary.value?.total_input_tokens ?? 0),
  },
  {
    key: "output",
    label: t("apiKeys.outputTokens"),
    value: formatNumberWithSuffix(summary.value?.total_output_tokens ?? 0),
  },
  {
    key: "success",
    label: t("apiKeys.successRate"),
    value:
      summary.value && summary.value.total_requests > 0
        ? formatPercentage(summary.value.success_rate)
        : "—",
  },
  {
    key: "latency",
    label: t("apiKeys.avgLatency"),
    value:
      summary.value && summary.value.total_requests > 0
        ? formatDuration(summary.value.avg_response_time_ms)
        : "—",
  },
]);

// --- Budget window ----------------------------------------------------------
// The enforced budget is the spend inside the current UTC window (tracked by
// the spend summary), NOT the selected chart range — keep the two separate so
// the bar always reads as the number the backend rejects requests on.

const spend = computed(() =>
  props.apiKey ? apiKeyStore.spendByKey[props.apiKey.name] : undefined
);

const { hasBudget, budgetSpend, budgetRatio, budgetPeriodLabel, barClass } = useBudgetDisplay(
  {
    budgetUsd: computed(() => props.apiKey?.budget_usd ?? null),
    budgetPeriod: computed(() => props.apiKey?.budget_period ?? spend.value?.budget_period ?? null),
    budgetResetDay: computed(
      () => props.apiKey?.budget_reset_day ?? spend.value?.budget_reset_day ?? null
    ),
    periodSpendUsd: computed(() => spend.value?.period_spend_usd ?? null),
    fallbackSpendUsd: computed(() => summary.value?.total_cost ?? null),
  },
  t
);

// Authoritative window spend when the summary is available; otherwise fall
// back to the selected-range spend (flagged with a note below the bar).
const usingRangeFallback = computed(
  () => hasBudget.value && (spend.value?.period_spend_usd ?? null) === null
);

// The note under the budget bar: window start for periodic budgets, reset
// point for lifetime budgets (a lifetime window opens at the epoch until the
// first manual reset, so only mention the date once it is meaningful).
const isLifetimeBudget = computed(
  () => hasBudget.value && !(props.apiKey?.budget_period ?? spend.value?.budget_period)
);

const windowNote = computed(() => {
  if (usingRangeFallback.value) return t("apiKeys.budgetRangeNote");
  const start = spend.value?.period_start;
  if (isLifetimeBudget.value) {
    return start && new Date(start).getTime() > 0
      ? t("apiKeys.budgetLifetimeNoteReset", { date: formatDate(start) })
      : t("apiKeys.budgetLifetimeNote");
  }
  return t("apiKeys.budgetWindowNote", { date: start ? formatDate(start) : "—" });
});

// --- By-model breakdown -----------------------------------------------------

const modelRows = computed(() => {
  const rows = usage.value?.by_model ?? [];
  const totalCost = rows.reduce((acc, row) => acc + row.cost, 0);
  return rows.map((row) => ({
    ...row,
    share: totalCost > 0 ? (row.cost / totalCost) * 100 : 0,
  }));
});
</script>

<template>
  <Sheet v-model:open="open">
    <SheetContent
      side="right"
      class="w-full sm:max-w-[640px] h-full flex flex-col p-0 gap-0 overflow-hidden border-l border-border/80 bg-card"
    >
      <SheetHeaderBand :icon="BarChart3">
        <template #title>{{ t("apiKeys.usageTitle") }}</template>
        <template #description>{{ apiKey?.name ?? "" }}</template>
      </SheetHeaderBand>

      <div class="flex-1 overflow-y-auto px-4 sm:px-6 py-4 space-y-5">
        <!-- Range selector (matches dashboard preset buttons) -->
        <div class="flex items-center gap-2">
          <Button
            v-for="days in rangeOptions"
            :key="days"
            :variant="rangeDays === days ? 'default' : 'outline'"
            size="sm"
            class="text-xs"
            :aria-pressed="rangeDays === days"
            @click="rangeDays = days"
          >
            {{ t("apiKeys.lastNDays", { n: days }) }}
          </Button>
        </div>

        <!-- First-load skeleton -->
        <div v-if="showSkeleton" class="space-y-4" aria-busy="true">
          <div class="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <Skeleton v-for="i in 6" :key="i" class="h-20 rounded-lg" />
          </div>
          <Skeleton class="h-64 rounded-lg" />
        </div>

        <!-- Load failure: explicit error with a retry path (never a silent
             blank body). Stale data from a previous range stays visible. -->
        <div
          v-else-if="loadError && !usage"
          class="flex flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-border/60 py-14 text-center"
        >
          <AlertTriangle class="w-6 h-6 text-status-warning" aria-hidden="true" />
          <p class="text-sm text-muted-foreground">{{ t("apiKeys.usageLoadError") }}</p>
          <Button variant="outline" size="sm" class="text-xs" @click="loadUsage">
            {{ t("common.retry") }}
          </Button>
        </div>

        <template v-else-if="usage">
          <div
            class="space-y-5 transition-opacity duration-150"
            :class="{ 'opacity-50 pointer-events-none': loading }"
            :aria-busy="loading"
          >
            <!-- Summary stats -->
            <div class="grid grid-cols-2 sm:grid-cols-3 gap-3">
              <div
                v-for="card in statCards"
                :key="card.key"
                class="rounded-lg border border-border/60 bg-muted/30 px-3 py-2.5"
              >
                <div class="text-[11px] uppercase tracking-wide text-muted-foreground">
                  {{ card.label }}
                </div>
                <div class="mt-1 text-lg font-semibold text-data">
                  {{ card.value }}
                </div>
              </div>
            </div>

            <!-- Budget window indicator -->
            <div
              v-if="hasBudget && apiKey"
              class="rounded-lg border border-border/60 bg-muted/30 px-3 py-2.5 space-y-1.5"
            >
              <div class="flex items-center justify-between gap-2 text-xs">
                <span class="flex items-center gap-1.5 text-muted-foreground">
                  <Gauge class="w-3.5 h-3.5" aria-hidden="true" />
                  {{ t("apiKeys.budgetWindow") }}
                  <Badge
                    v-if="budgetPeriodLabel"
                    variant="secondary"
                    class="font-normal text-[11px] px-1.5 py-0"
                  >
                    {{ budgetPeriodLabel }}
                  </Badge>
                </span>
                <span class="text-data">
                  {{ formatCost(budgetSpend) }} / {{ formatCost(apiKey.budget_usd) }}
                </span>
              </div>
              <div
                class="h-1.5 rounded-full bg-muted overflow-hidden"
                role="progressbar"
                :aria-label="t('apiKeys.budgetWindow')"
                :aria-valuenow="Math.round((budgetRatio ?? 0) * 100)"
                aria-valuemin="0"
                aria-valuemax="100"
              >
                <div
                  class="h-full rounded-full transition-all"
                  :class="barClass"
                  :style="{ width: `${(budgetRatio ?? 0) * 100}%` }"
                />
              </div>
              <p class="text-[11px] text-muted-foreground">
                {{ windowNote }}
              </p>
            </div>

            <!-- Per-key rate limit indicator -->
            <div
              v-if="apiKey?.rate_limit_rpm != null"
              class="rounded-lg border border-border/60 bg-muted/30 px-3 py-2.5 flex items-center justify-between gap-2 text-xs"
            >
              <span class="flex items-center gap-1.5 text-muted-foreground">
                <Gauge class="w-3.5 h-3.5" aria-hidden="true" />
                {{ t("apiKeys.rateLimit") }}
              </span>
              <span class="text-data">
                {{ t("apiKeys.rateLimitPerMin", { n: apiKey.rate_limit_rpm }) }}
              </span>
            </div>

            <!-- Empty state -->
            <div
              v-if="!hasData"
              class="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-border/60 py-14 text-center"
            >
              <Inbox class="w-6 h-6 text-muted-foreground/60" aria-hidden="true" />
              <p class="text-sm text-muted-foreground">{{ t("apiKeys.noUsageData") }}</p>
              <p class="text-xs text-muted-foreground/70">{{ t("apiKeys.noUsageDataHint") }}</p>
            </div>

            <template v-else>
              <!-- Daily trend chart -->
              <div class="rounded-lg border border-border/60 p-3">
                <UsageTrendsChart :daily-usage="usage.daily_usage" />
              </div>

              <!-- By-model breakdown -->
              <div v-if="modelRows.length > 0" class="space-y-2">
                <h3 class="text-sm font-medium">{{ t("apiKeys.byModel") }}</h3>
                <div class="rounded-lg border border-border/60 divide-y divide-border/60">
                  <div
                    v-for="row in modelRows"
                    :key="`${row.model}/${row.provider}`"
                    class="flex items-center justify-between gap-3 px-3 py-2 text-sm"
                  >
                    <div class="min-w-0 flex items-center gap-2">
                      <span class="font-mono text-xs truncate">{{ row.model }}</span>
                      <Badge variant="outline" class="font-normal text-[11px] shrink-0">
                        {{ row.provider }}
                      </Badge>
                    </div>
                    <div class="flex items-center gap-3 sm:gap-4 shrink-0 text-data text-xs">
                      <span class="text-muted-foreground hidden sm:inline">
                        {{ formatNumberWithSuffix(row.requests) }} {{ t("apiKeys.requestsShort") }}
                      </span>
                      <span class="text-muted-foreground tabular-nums w-12 text-right">
                        {{ formatPercentage(row.share) }}
                      </span>
                      <span class="tabular-nums">{{ formatCost(row.cost) }}</span>
                    </div>
                  </div>
                </div>
              </div>
            </template>
          </div>
        </template>
      </div>
    </SheetContent>
  </Sheet>
</template>
