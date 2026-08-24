<script setup lang="ts">
import { CheckCircle2, Clock, Cpu, DollarSign, Package, Activity, Zap } from "@lucide/vue";
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import {
  formatCostWithPrecision,
  formatDuration,
  formatNumberWithSuffix,
  formatPercentage,
} from "@/utils/format";
import ValueSwap from "@/components/common/ValueSwap.vue";
import type { UsageByProvider, UsageSummary } from "@/types/schemas";

const props = defineProps<{
  summary: UsageSummary;
  byProvider: UsageByProvider[];
}>();
const { t } = useI18n();

const totalTokens = computed(
  () => props.summary.total_input_tokens + props.summary.total_output_tokens
);

const avgCostPerRequest = computed(() => {
  if (props.summary.total_requests === 0) return "$0.00";
  return formatCostWithPrecision(props.summary.total_cost / props.summary.total_requests, 4);
});

const avgTokensPerRequest = computed(() => {
  if (props.summary.total_requests === 0) return 0;
  return totalTokens.value / props.summary.total_requests;
});

// Token split
const inputPercentage = computed(() =>
  totalTokens.value === 0 ? 0 : (props.summary.total_input_tokens / totalTokens.value) * 100
);
const outputPercentage = computed(() =>
  totalTokens.value === 0 ? 0 : (props.summary.total_output_tokens / totalTokens.value) * 100
);

// Cache hit rate (Anthropic cache reads + OpenAI cached prompts)
// Only count input tokens from providers that actually report cache data.
// Providers like Ollama never return cache tokens, so including their
// input tokens would artificially depress the rate.
const totalCacheHitTokens = computed(
  () => props.summary.total_cache_read_tokens + props.summary.total_cached_prompt_tokens
);
const cacheEligibleInputTokens = computed(() => {
  const reporters = new Set(
    props.byProvider
      .filter(
        (p) => p.cache_read_tokens > 0 || p.cached_prompt_tokens > 0 || p.cache_creation_tokens > 0
      )
      .map((p) => p.provider)
  );
  return props.byProvider.reduce(
    (sum, p) => sum + (reporters.has(p.provider) ? p.input_tokens : 0),
    0
  );
});
const cacheHitRate = computed(() => {
  if (cacheEligibleInputTokens.value === 0) return 0;
  return (totalCacheHitTokens.value / cacheEligibleInputTokens.value) * 100;
});

// Success tier → semantic status tokens. Color is never the sole signal:
// the percentage, the tier label, and the pulse dot carry the same tier.
type SuccessTier = "success" | "warning" | "error";

const successTier = computed<SuccessTier>(() => {
  const r = props.summary.success_rate;
  if (r >= 99) return "success";
  if (r >= 95) return "warning";
  return "error";
});

const tierStyles: Record<
  SuccessTier,
  {
    text: string;
    meter: string;
    badge: string;
    dot: string;
    ping: string;
    label: string;
  }
> = {
  success: {
    text: "text-status-success",
    meter: "bg-status-success",
    badge: "bg-status-success/15 text-status-success border-status-success/30",
    dot: "bg-status-success",
    ping: "bg-status-success/75",
    label: "home.statusOptimal",
  },
  warning: {
    text: "text-status-warning",
    meter: "bg-status-warning",
    badge: "bg-status-warning/15 text-status-warning border-status-warning/30",
    dot: "bg-status-warning",
    ping: "bg-status-warning/75",
    label: "home.statusDegraded",
  },
  error: {
    text: "text-status-error",
    meter: "bg-status-error",
    badge: "bg-status-error/15 text-status-error border-status-error/30",
    dot: "bg-status-error",
    ping: "bg-status-error/75",
    label: "home.statusCritical",
  },
};

const tier = computed(() => tierStyles[successTier.value]);
</script>

<template>
  <section class="flex flex-col">
    <!-- Primary metric strip: 6 flush cells, hairline-separated. No cards. -->
    <div
      class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 divide-x divide-y sm:divide-y-0 lg:divide-y-0 divide-border/60 border-b border-border/60 px-4 sm:px-6"
    >
      <!-- Requests -->
      <div class="px-3 sm:px-4 py-3">
        <div class="flex items-center gap-1.5 text-[11px] text-muted-foreground font-medium">
          <Package class="w-3.5 h-3.5 text-action-blue" />
          {{ t("home.totalRequests") }}
        </div>
        <div
          class="text-base font-mono font-semibold text-foreground mt-1 leading-none tabular-nums"
        >
          <ValueSwap :value="formatNumberWithSuffix(summary.total_requests)" />
        </div>
        <div class="text-[11px] text-muted-foreground mt-1">
          {{ t("home.avgAbbr") }}:
          {{ formatNumberWithSuffix(Math.round(avgTokensPerRequest)) }} tok/req
        </div>
      </div>

      <!-- Cost -->
      <div class="px-3 sm:px-4 py-3">
        <div class="flex items-center gap-1.5 text-[11px] text-muted-foreground font-medium">
          <DollarSign class="w-3.5 h-3.5 text-action-amber" />
          {{ t("home.totalCost") }}
        </div>
        <div
          class="text-base font-mono font-semibold text-foreground mt-1 leading-none tabular-nums"
        >
          <ValueSwap :value="formatCostWithPrecision(summary.total_cost, 2)" />
        </div>
        <div class="text-[11px] text-muted-foreground mt-1">
          {{ t("home.avgAbbr") }}: {{ avgCostPerRequest }}/req
        </div>
      </div>

      <!-- Tokens -->
      <div class="px-3 sm:px-4 py-3">
        <div class="flex items-center gap-1.5 text-[11px] text-muted-foreground font-medium">
          <Activity class="w-3.5 h-3.5 text-action-rose" />
          {{ t("home.totalTokens") }}
        </div>
        <div
          class="text-base font-mono font-semibold text-foreground mt-1 leading-none tabular-nums"
        >
          <ValueSwap :value="formatNumberWithSuffix(totalTokens)" />
        </div>
        <div class="text-[11px] text-muted-foreground mt-1 truncate">
          {{ t("home.inAbbr") }}: {{ formatNumberWithSuffix(summary.total_input_tokens) }} ·
          {{ t("home.outAbbr") }}: {{ formatNumberWithSuffix(summary.total_output_tokens) }}
        </div>
      </div>

      <!-- Success (value + tier chip + status-tinted meter + pulse). The whole
           cell is a link to the proxy error log so a degraded rate is never a
           dead end — operators land on the failing 5xx requests in one click. -->
      <RouterLink
        :to="{ path: '/logs', query: { tab: 'proxy', status: '5xx' } }"
        class="px-3 sm:px-4 py-3 block transition-colors hover:bg-muted/40 focus-visible:bg-muted/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring/60"
        :aria-label="t('home.viewErrors') + ': ' + formatPercentage(summary.success_rate)"
        :title="t('home.viewErrors')"
      >
        <div class="flex items-center gap-1.5 text-[11px] text-muted-foreground font-medium">
          <CheckCircle2 class="w-3.5 h-3.5 text-status-success" />
          {{ t("home.successRate") }}
        </div>
        <div class="flex items-center gap-1.5 mt-1">
          <span class="text-base font-mono font-semibold tabular-nums" :class="tier.text">
            <ValueSwap :value="formatPercentage(summary.success_rate)" />
          </span>
          <span class="text-[11px] px-1.5 py-0.5 rounded-full border" :class="tier.badge">
            {{ t(tier.label) }}
          </span>
        </div>
        <div class="relative h-1 w-full bg-muted rounded-full overflow-hidden mt-1.5">
          <div
            class="h-full rounded-full transition-all duration-500"
            :class="tier.meter"
            :style="{ width: `${summary.success_rate}%` }"
          />
        </div>
        <div class="flex items-center gap-1 mt-1">
          <span class="relative flex h-2 w-2" aria-hidden="true">
            <span
              class="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75"
              :class="tier.ping"
            />
            <span class="relative inline-flex rounded-full h-2 w-2" :class="tier.dot" />
          </span>
          <span class="text-[11px] text-muted-foreground">{{ t("home.viewErrors") }} →</span>
        </div>
      </RouterLink>

      <!-- Latency -->
      <div class="px-3 sm:px-4 py-3">
        <div class="flex items-center gap-1.5 text-[11px] text-muted-foreground font-medium">
          <Clock class="w-3.5 h-3.5 text-action-blue" />
          {{ t("logs.latency") }}
        </div>
        <div
          class="text-base font-mono font-semibold text-foreground mt-1 leading-none tabular-nums"
        >
          <ValueSwap :value="formatDuration(summary.avg_response_time_ms)" />
        </div>
        <div class="text-[11px] text-muted-foreground mt-1">
          TTFT: {{ formatDuration(summary.avg_ttft_ms) }}
        </div>
      </div>

      <!-- Throughput (tps) -->
      <div class="px-3 sm:px-4 py-3">
        <div class="flex items-center gap-1.5 text-[11px] text-muted-foreground font-medium">
          <Cpu class="w-3.5 h-3.5 text-action-amber" />
          {{ t("home.avgThroughput") }}
        </div>
        <div
          class="text-base font-mono font-semibold text-foreground mt-1 leading-none tabular-nums"
        >
          <ValueSwap :value="summary.avg_tokens_per_second.toFixed(1)" />
          <span class="text-xs font-normal text-muted-foreground">tps</span>
        </div>
        <div class="text-[11px] text-muted-foreground mt-1">{{ t("home.throughputCaption") }}</div>
      </div>
    </div>

    <!-- Secondary strip: 2 bar-metrics, flush. No nested boxes. -->
    <div
      class="grid grid-cols-1 sm:grid-cols-2 divide-y sm:divide-y-0 sm:divide-x divide-border/60 border-b border-border/60 px-4 sm:px-6"
    >
      <!-- Token distribution -->
      <div class="px-3 sm:px-4 py-3">
        <div class="flex items-center gap-1.5 text-[11px] text-muted-foreground font-medium">
          <Activity class="w-3.5 h-3.5 text-action-rose" />
          {{ t("home.tokenDistribution") }}
        </div>
        <div class="flex items-center justify-between text-xs mt-1.5">
          <span class="text-muted-foreground" :title="t('home.splitRatioHint')">{{
            t("home.splitRatio")
          }}</span>
          <span class="font-mono text-[11px] text-muted-foreground">
            <span class="text-foreground font-semibold">{{ inputPercentage.toFixed(0) }}%</span>
            {{ t("home.inAbbr") }} /
            <span class="text-foreground font-semibold">{{ outputPercentage.toFixed(0) }}%</span>
            {{ t("home.outAbbr") }}
          </span>
        </div>
        <div class="h-2 w-full rounded-full bg-muted overflow-hidden flex mt-1.5">
          <div
            class="h-full bg-action-blue transition-all duration-500"
            :style="{ width: `${inputPercentage}%` }"
          />
          <div
            class="h-full bg-action-rose transition-all duration-500"
            :style="{ width: `${outputPercentage}%` }"
          />
        </div>
      </div>

      <!-- Cache efficiency -->
      <div class="px-3 sm:px-4 py-3">
        <div class="flex items-center gap-1.5 text-[11px] text-muted-foreground font-medium">
          <Zap class="w-3.5 h-3.5 text-status-success" />
          {{ t("home.cacheEfficiency") }}
        </div>
        <div class="flex items-center justify-between text-xs mt-1.5">
          <span class="text-muted-foreground">{{ t("home.hitRate") }}</span>
          <span class="font-mono text-[11px] text-status-success font-bold"
            >{{ cacheHitRate.toFixed(1) }}%</span
          >
        </div>
        <div class="h-2 w-full rounded-full bg-muted overflow-hidden flex mt-1.5">
          <div
            class="h-full bg-status-success transition-all duration-500"
            :style="{ width: `${cacheHitRate}%` }"
          />
        </div>
        <div class="flex items-center justify-between text-[11px] text-muted-foreground mt-1.5">
          <span>{{ t("home.cacheHits") }}: {{ formatNumberWithSuffix(totalCacheHitTokens) }}</span>
          <span
            >{{ t("home.estimatedSavings") }}:
            <span class="font-mono font-semibold text-status-success">{{
              formatCostWithPrecision(summary.cache_savings_usd)
            }}</span></span
          >
        </div>
      </div>
    </div>
  </section>
</template>
