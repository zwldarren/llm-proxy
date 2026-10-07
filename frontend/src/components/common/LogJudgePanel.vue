<script setup lang="ts">
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { Badge } from "@/components/ui/badge";
import type { RoutingJudgeMetadata } from "@/types/schemas";

/**
 * The routing judge consultation, as recorded on either of the two rows it
 * touches: nested under `routing.judge` on the request the judge served, or as
 * `log_metadata.judge` on the judge call's own row (log_type `judge`). Same
 * verdict, same panel — extracted so the two never drift apart.
 */
const props = defineProps<{
  judge: RoutingJudgeMetadata;
  /** Set on the judge's own row: the request whose routing triggered the call. */
  parentRequestId?: string | null;
  /** Set on the judge's own row: the virtual model that was being resolved. */
  requestedModel?: string | null;
  /** Set on the judge's own row: the model the request was resolved to. */
  resolvedModel?: string | null;
}>();

const { t } = useI18n();

const formatProbability = (value: unknown): string => {
  if (typeof value !== "number") return "\u2014";
  return `${(value * 100).toFixed(1)}%`;
};

// Judge spend is fractions of a cent; a 4-decimal round would print "0".
const formatCost = (value: unknown, source?: string | null): string => {
  if (typeof value !== "number") return "\u2014";
  const formatted = value === 0 ? "0" : value < 0.01 ? value.toExponential(2) : value.toFixed(6);
  return source ? `$${formatted} (${source})` : `$${formatted}`;
};

const hasCallContext = computed(
  () =>
    Boolean(props.parentRequestId) ||
    Boolean(props.requestedModel) ||
    Boolean(props.resolvedModel) ||
    Boolean(props.judge.provider_model_name)
);
</script>

<template>
  <div class="space-y-3">
    <div class="flex items-center gap-2">
      <span class="text-xs text-muted-foreground uppercase tracking-wider font-semibold">
        {{ t("logs.routing.judge") }}
      </span>
      <Badge
        v-if="judge.shadow"
        variant="outline"
        class="text-[11px] h-4 py-0 px-1.5 bg-muted/10 border-border/20 text-foreground uppercase font-mono font-bold"
      >
        {{ t("logs.routing.judgeShadowBadge") }}
      </Badge>
    </div>

    <div class="grid grid-cols-2 sm:grid-cols-4 gap-2">
      <div class="bg-muted/30 px-2.5 py-1.5 rounded border border-border/20">
        <div class="text-[11px] text-muted-foreground">
          {{ t("logs.routing.judgeVerdict") }}
        </div>
        <div class="text-xs font-mono font-medium mt-0.5">
          {{ judge.tier || t("logs.routing.judgeAbstained") }}
        </div>
      </div>
      <div class="bg-muted/30 px-2.5 py-1.5 rounded border border-border/20">
        <div class="text-[11px] text-muted-foreground">
          {{ t("logs.routing.judgeGate") }}
        </div>
        <div class="text-xs font-mono font-medium mt-0.5 truncate" :title="judge.gate ?? undefined">
          {{ judge.gate || "\u2014" }}
        </div>
      </div>
      <div class="bg-muted/30 px-2.5 py-1.5 rounded border border-border/20">
        <div class="text-[11px] text-muted-foreground">
          {{ t("logs.routing.judgeModel") }}
        </div>
        <div
          class="text-xs font-mono font-medium mt-0.5 truncate"
          :title="judge.model ?? undefined"
        >
          {{ judge.model || "\u2014" }}
        </div>
      </div>
      <div class="bg-muted/30 px-2.5 py-1.5 rounded border border-border/20">
        <div class="text-[11px] text-muted-foreground">
          {{ t("logs.routing.judgeLatency") }}
        </div>
        <div class="text-xs font-mono font-medium mt-0.5">
          {{
            typeof judge.latency_ms === "number" ? `${Math.round(judge.latency_ms)} ms` : "\u2014"
          }}
        </div>
      </div>
      <div class="bg-muted/30 px-2.5 py-1.5 rounded border border-border/20">
        <div class="text-[11px] text-muted-foreground">
          {{ t("logs.routing.judgeConfidence") }}
        </div>
        <div class="text-xs font-mono font-medium mt-0.5">
          {{ formatProbability(judge.confidence) }}
        </div>
      </div>
      <div class="bg-muted/30 px-2.5 py-1.5 rounded border border-border/20">
        <div class="text-[11px] text-muted-foreground">
          {{ t("logs.routing.judgeAbstainProb") }}
        </div>
        <div class="text-xs font-mono font-medium mt-0.5">
          {{ formatProbability(judge.abstain_probability) }}
        </div>
      </div>
      <div class="bg-muted/30 px-2.5 py-1.5 rounded border border-border/20">
        <div class="text-[11px] text-muted-foreground">
          {{ t("logs.routing.judgeEscalationProb") }}
        </div>
        <div class="text-xs font-mono font-medium mt-0.5">
          {{ formatProbability(judge.escalation_probability) }}
        </div>
      </div>
      <div class="bg-muted/30 px-2.5 py-1.5 rounded border border-border/20">
        <div class="text-[11px] text-muted-foreground">
          {{ t("logs.routing.judgeCost") }}
        </div>
        <div class="text-xs font-mono font-medium mt-0.5">
          {{ formatCost(judge.cost, judge.cost_source) }}
        </div>
      </div>
    </div>

    <!-- The judge row's own context: which request caused the call, and which
         model that request was resolved to. Absent on the request's own row. -->
    <div v-if="hasCallContext" class="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px]">
      <span v-if="parentRequestId" class="flex items-center gap-1">
        <span class="text-muted-foreground">{{ t("logs.routing.judgeParentRequest") }}</span>
        <code class="font-mono text-foreground/90">{{ parentRequestId }}</code>
      </span>
      <span v-if="requestedModel" class="flex items-center gap-1">
        <span class="text-muted-foreground">{{ t("logs.routing.judgeRequestedModel") }}</span>
        <code class="font-mono text-foreground/90">{{ requestedModel }}</code>
      </span>
      <span v-if="resolvedModel" class="flex items-center gap-1">
        <span class="text-muted-foreground">{{ t("logs.routing.judgeResolvedModel") }}</span>
        <code class="font-mono text-foreground/90">{{ resolvedModel }}</code>
      </span>
      <span v-if="judge.provider_model_name" class="flex items-center gap-1">
        <span class="text-muted-foreground">{{ t("logs.routing.judgeProviderModel") }}</span>
        <code class="font-mono text-foreground/90">{{ judge.provider_model_name }}</code>
      </span>
    </div>

    <p
      v-if="judge.error"
      class="text-xs text-status-error/90 bg-status-error/5 border border-status-error/20 p-2.5 rounded-lg font-mono break-all"
    >
      {{ judge.error }}
    </p>
  </div>
</template>
