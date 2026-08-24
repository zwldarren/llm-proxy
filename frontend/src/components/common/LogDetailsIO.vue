<script setup lang="ts">
import { AlertTriangle, RefreshCw, RotateCcw } from "@lucide/vue";
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import AttemptTimeline from "@/components/common/AttemptTimeline.vue";
import AuditInfoView from "@/components/common/AuditInfoView.vue";
import JsonViewer from "@/components/common/JsonViewer.vue";
import LogIOView from "@/components/common/LogIOView.vue";
import LogMetadataView from "@/components/common/LogMetadataView.vue";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { LogRead, LogRoutingMetadata, RetryAttempt, FallbackAttempt } from "@/types/schemas";

const props = defineProps<{
  log: LogRead;
}>();

const { t } = useI18n();

const isAuditLog = computed(() => props.log.log_type === "audit");
const isMcpLog = computed(() => props.log.log_type === "mcp");
const isWebSearchLog = computed(() => props.log.log_type === "web_search");
const isToolLog = computed(() => isMcpLog.value || isWebSearchLog.value);

// Request type detection
const requestType = computed<string>(() => {
  const rt = props.log.log_metadata?.request_type;
  return typeof rt === "string" ? rt : "chat";
});
const isSpeechRequest = computed(() => requestType.value === "speech");
const isTranscriptionRequest = computed(() => requestType.value === "transcription");
const isTranslationRequest = computed(() => requestType.value === "translation");
const isAudioRequest = computed(
  () => isSpeechRequest.value || isTranscriptionRequest.value || isTranslationRequest.value
);
const isEmbeddingRequest = computed(() => requestType.value === "embedding");
const isImageRequest = computed(
  () => requestType.value === "image_generation" || requestType.value === "image_edit"
);
const isNonChatProxy = computed(
  () => isAudioRequest.value || isEmbeddingRequest.value || isImageRequest.value
);

// Validated retry and fallback attempts with filtering for well-formed entries
// to prevent rendering errors from malformed backend data.
const validRetryAttempts = computed<RetryAttempt[]>(() => {
  const attempts = props.log.log_metadata?.retry_attempts;
  if (!Array.isArray(attempts)) return [];
  return attempts.filter(
    (a): a is RetryAttempt =>
      a !== null && typeof a === "object" && typeof (a as RetryAttempt).provider === "string"
  );
});

const validFallbackAttempts = computed<FallbackAttempt[]>(() => {
  const attempts = props.log.log_metadata?.fallback_attempts;
  if (!Array.isArray(attempts)) return [];
  return attempts.filter(
    (a): a is FallbackAttempt =>
      a !== null && typeof a === "object" && typeof (a as FallbackAttempt).provider === "string"
  );
});

// Routing metadata helpers
interface RoutingScorecard {
  model: string;
  total?: number;
  cost?: number;
  latency?: number;
  reliability?: number;
  predicted_quality?: number;
  quality_alignment?: number;
}

const routingMeta = computed<LogRoutingMetadata | null>(() => {
  if (isAuditLog.value || isToolLog.value) return null;
  const meta = props.log.log_metadata?.routing;
  if (!meta) return null;
  return meta;
});

const routingReasoning = computed(() => {
  const r = routingMeta.value?.reasoning;
  if (r && typeof r === "object" && "text" in r) {
    return (r as Record<string, unknown>).text;
  }
  return r;
});

const routingWeights = computed(() => {
  const value = routingMeta.value?.weights_used;
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
});

const routingSignalVotes = computed(() => {
  const value = routingMeta.value?.signal_votes;
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
});

const routingGuardrailNotes = computed(() => {
  const value = routingMeta.value?.guardrail_notes;
  return Array.isArray(value) ? (value as unknown[]) : undefined;
});

const routingScorecards = computed(() => {
  const value = routingMeta.value?.candidate_scorecards;
  return Array.isArray(value) ? (value as unknown as RoutingScorecard[]) : undefined;
});

const hasRoutingDiagnostics = computed(
  () =>
    (routingWeights.value && Object.keys(routingWeights.value).length > 0) ||
    (routingSignalVotes.value && Object.keys(routingSignalVotes.value).length > 0) ||
    (routingGuardrailNotes.value?.length ?? 0) > 0 ||
    (routingScorecards.value?.length ?? 0) > 0
);

function parseReasoningItem(raw: string): { key: string; value: string; active?: boolean } {
  const active = raw.includes("[active]");
  const cleanRaw = raw.replace("[active]", "").trim();
  const eqIdx = cleanRaw.indexOf("=");
  if (eqIdx !== -1) {
    return {
      key: cleanRaw.substring(0, eqIdx).trim(),
      value: cleanRaw.substring(eqIdx + 1).trim(),
      active,
    };
  }
  return { key: "", value: cleanRaw, active };
}

const parsedReasoning = computed(() => {
  const str = routingReasoning.value;
  if (!str || typeof str !== "string") return [];

  const items: { key: string; value: string; active?: boolean }[] = [];
  let current = "";
  let parenDepth = 0;

  for (let i = 0; i < str.length; i++) {
    const char = str[i];
    if (char === "(" || char === "[") {
      parenDepth++;
      current += char;
    } else if (char === ")" || char === "]") {
      parenDepth--;
      current += char;
    } else if (parenDepth === 0 && (char === "," || char === "|" || char === " ")) {
      if (current.trim()) {
        items.push(parseReasoningItem(current.trim()));
        current = "";
      }
    } else {
      current += char;
    }
  }

  if (current.trim()) {
    items.push(parseReasoningItem(current.trim()));
  }

  return items.filter((item) => item.key || item.value);
});

const formatRoutingValue = (value: unknown): string => {
  if (value === null || value === undefined) return "\u2014";
  if (typeof value === "number") return Number(value.toFixed(4)).toString();
  if (typeof value === "string") return value;
  return JSON.stringify(value);
};

// Expose routing meta for parent to determine tab visibility
defineExpose({
  routingMeta,
  isAuditLog,
  isToolLog,
  isNonChatProxy,
});
</script>

<template>
  <!-- Error Banner -->
  <div
    v-if="log.error_message"
    role="alert"
    class="rounded-md border border-status-error/25 bg-status-error/5 p-3 sm:p-4 space-y-3"
  >
    <h3 class="flex items-center gap-2 text-xs sm:text-sm font-bold text-status-error">
      <AlertTriangle class="w-3.5 h-3.5 sm:w-4 sm:h-4 shrink-0" /> {{ t("logs.error") }}
    </h3>
    <p
      class="text-[11px] sm:text-xs text-foreground/90 font-mono max-h-32 sm:max-h-48 overflow-auto bg-background/50 p-2 sm:p-3 rounded-lg border border-border/20"
    >
      {{ log.error_message }}
    </p>

    <!-- Enhanced Error Details -->
    <div v-if="log.log_metadata?.error_details" class="pt-2 border-t border-status-error/15">
      <h4
        class="text-[11px] sm:text-xs font-bold text-muted-foreground uppercase tracking-wider mb-2"
      >
        {{ t("logs.errorDetails") }}
      </h4>
      <div class="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-2 text-[11px] sm:text-xs">
        <!-- Provider Error Details -->
        <div v-if="log.log_metadata.error_details.provider_name" class="flex items-center gap-2">
          <span class="error-detail-label">Provider:</span>
          <span class="error-detail-value">{{ log.log_metadata.error_details.provider_name }}</span>
        </div>
        <div v-if="log.log_metadata.error_details.error_type" class="flex items-center gap-2">
          <span class="error-detail-label">Error Type:</span>
          <span class="error-detail-value">{{ log.log_metadata.error_details.error_type }}</span>
        </div>
        <div v-if="log.log_metadata.error_details.code" class="flex items-center gap-2">
          <span class="error-detail-label">Error Code:</span>
          <span class="error-detail-value">{{ log.log_metadata.error_details.code }}</span>
        </div>
        <div v-if="log.log_metadata.error_details.param" class="flex items-center gap-2">
          <span class="error-detail-label">Parameter:</span>
          <span class="error-detail-value">{{ log.log_metadata.error_details.param }}</span>
        </div>
        <!-- HTTP Error Details -->
        <div v-if="log.log_metadata.error_details.status_code" class="flex items-center gap-2">
          <span class="error-detail-label">HTTP Status:</span>
          <span class="error-detail-value">{{ log.log_metadata.error_details.status_code }}</span>
        </div>
        <div v-if="log.log_metadata.error_details.method" class="flex items-center gap-2">
          <span class="error-detail-label">Method:</span>
          <span class="error-detail-value">{{ log.log_metadata.error_details.method }}</span>
        </div>
        <div v-if="log.log_metadata.error_details.url" class="flex items-start gap-2 md:col-span-2">
          <span class="error-detail-label">URL:</span>
          <span class="error-detail-value break-all">{{ log.log_metadata.error_details.url }}</span>
        </div>
        <!-- Original Error -->
        <div v-if="log.log_metadata.error_details.original_error" class="mt-2 md:col-span-2">
          <span class="error-detail-label">Original Error:</span>
          <JsonViewer
            :data="log.log_metadata.error_details.original_error"
            class="mt-1"
            max-height="max-h-96"
          />
        </div>
        <!-- Response Body -->
        <div v-if="log.log_metadata.error_details.response_body" class="mt-2 md:col-span-2">
          <span class="error-detail-label">Response Body:</span>
          <div class="mt-1">
            <JsonViewer
              :data="log.log_metadata.error_details.response_body"
              :deep="2"
              max-height="max-h-96"
              class="border border-error-light/15 dark:border-error-dark/20"
            />
          </div>
        </div>
        <!-- Streaming Error Details -->
        <div
          v-if="log.log_metadata.error_details.stream_error_type"
          class="flex items-center gap-2"
        >
          <span class="error-detail-label">Stream Error Type:</span>
          <span class="error-detail-value">{{
            log.log_metadata.error_details.stream_error_type
          }}</span>
        </div>
        <div
          v-if="log.log_metadata.error_details.stream_error_message"
          class="flex items-start gap-2 md:col-span-2"
        >
          <span class="error-detail-label">Stream Error Message:</span>
          <span class="error-detail-value break-all">{{
            log.log_metadata.error_details.stream_error_message
          }}</span>
        </div>
      </div>
    </div>

    <div v-if="log.error_stack_trace" class="pt-2 border-t border-status-error/15">
      <h4
        class="text-[11px] sm:text-xs font-bold text-muted-foreground uppercase tracking-wider mb-2"
      >
        {{ t("logs.stackTrace") }}
      </h4>
      <pre
        class="text-[11px] sm:text-xs font-mono text-foreground/80 bg-background/50 p-2 sm:p-3 rounded-lg border border-border/20 max-h-48 sm:max-h-96 overflow-auto scrollbar-thin"
        >{{ log.error_stack_trace }}</pre>
    </div>
  </div>

  <!-- Fallback Attempts Timeline -->
  <AttemptTimeline
    :attempts="validFallbackAttempts"
    :icon="RotateCcw"
    :icon-animated="true"
    :title="t('logs.fallbackAttempts')"
    success-text="logs.fallbackCount"
    failure-text="logs.fallbackFailed"
    show-more-text="logs.showAllFallbacks"
    :status-code="log.status_code"
  >
    <template #attempt-meta="{ attempt }">
      <span
        v-if="
          (attempt as FallbackAttempt).provider_type &&
          (attempt as FallbackAttempt).provider_type !== (attempt as FallbackAttempt).provider
        "
        class="font-mono text-[11px] text-muted-foreground shrink-0 bg-muted px-1.5 py-0.5 rounded border border-border/30"
      >
        {{ (attempt as FallbackAttempt).provider_type }}
      </span>
    </template>
  </AttemptTimeline>

  <!-- Retry Attempts Timeline (same-provider retries) -->
  <AttemptTimeline
    :attempts="validRetryAttempts"
    :icon="RefreshCw"
    :title="t('logs.retryAttempts')"
    success-text="logs.retryCount"
    failure-text="logs.retryFailed"
    show-more-text="logs.showAllRetries"
    :status-code="log.status_code"
  >
    <template #attempt-meta="{ attempt }">
      <span
        class="font-mono text-[11px] text-muted-foreground shrink-0 bg-muted px-1.5 py-0.5 rounded border border-border/30"
      >
        {{
          t("logs.attemptLabel", {
            n: (attempt as RetryAttempt).attempt,
            total: (attempt as RetryAttempt).total,
          })
        }}
      </span>
    </template>
  </AttemptTimeline>

  <!-- Main Content Tabs -->
  <Tabs :default-value="isAuditLog ? 'audit' : 'io'" :key="log.request_id" class="w-full">
    <TabsList
      class="w-full justify-start border-b border-border/60 rounded-none h-auto p-0 bg-transparent gap-4 sm:gap-6 overflow-x-auto scrollbar-none"
    >
      <TabsTrigger
        v-if="isAuditLog"
        value="audit"
        class="rounded-none border-b border-border/30 data-[state=active]:border-primary data-[state=active]:bg-transparent text-muted-foreground data-[state=active]:text-foreground px-1.5 sm:px-2 py-2.5 sm:py-3 text-xs sm:text-sm font-semibold transition-all shrink-0"
        >{{ t("logs.audit.infoTitle") }}</TabsTrigger
      >
      <TabsTrigger
        value="io"
        class="rounded-none border-b border-border/30 data-[state=active]:border-primary data-[state=active]:bg-transparent text-muted-foreground data-[state=active]:text-foreground px-1.5 sm:px-2 py-2.5 sm:py-3 text-xs sm:text-sm font-semibold transition-all shrink-0"
        >{{ t("logs.io") }}</TabsTrigger
      >
      <TabsTrigger
        v-if="routingMeta"
        value="routing"
        class="rounded-none border-b border-border/30 data-[state=active]:border-primary data-[state=active]:bg-transparent text-muted-foreground data-[state=active]:text-foreground px-1.5 sm:px-2 py-2.5 sm:py-3 text-xs sm:text-sm font-semibold transition-all shrink-0"
        >{{ t("logs.routing.title") }}</TabsTrigger
      >
      <TabsTrigger
        value="metadata"
        class="rounded-none border-b border-border/30 data-[state=active]:border-primary data-[state=active]:bg-transparent text-muted-foreground data-[state=active]:text-foreground px-1.5 sm:px-2 py-2.5 sm:py-3 text-xs sm:text-sm font-semibold transition-all shrink-0"
        >{{ t("logs.metadata") }}</TabsTrigger
      >
    </TabsList>

    <div class="mt-4 sm:mt-6">
      <!-- Audit structured information -->
      <TabsContent v-if="isAuditLog" value="audit" class="m-0 focus-visible:ring-0">
        <AuditInfoView :log="log" />
      </TabsContent>

      <!-- I/O View -->
      <TabsContent value="io" class="m-0 focus-visible:ring-0">
        <LogIOView :log="log" />
      </TabsContent>

      <!-- Routing View -->
      <TabsContent v-if="routingMeta" value="routing" class="m-0 focus-visible:ring-0 space-y-6">
        <!-- Summary Grid -->
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div class="bg-muted/10 border border-border/40 rounded-md p-3 flex flex-col gap-1">
            <span class="text-[11px] text-muted-foreground uppercase tracking-wider font-semibold">
              {{ t("logs.routing.tier") }}
            </span>
            <div class="font-mono text-xs sm:text-sm font-semibold text-foreground">
              {{ routingMeta.tier || "\u2014" }}
            </div>
          </div>
          <div class="bg-muted/10 border border-border/40 rounded-md p-3 flex flex-col gap-1">
            <span class="text-[11px] text-muted-foreground uppercase tracking-wider font-semibold">
              {{ t("logs.routing.confidence") }}
            </span>
            <div class="font-mono text-xs sm:text-sm font-semibold text-foreground">
              {{ formatRoutingValue(routingMeta.confidence) }}
            </div>
          </div>
          <div class="bg-muted/10 border border-border/40 rounded-md p-3 flex flex-col gap-1">
            <span class="text-[11px] text-muted-foreground uppercase tracking-wider font-semibold">
              {{ t("logs.routing.complexity") }}
            </span>
            <div class="font-mono text-xs sm:text-sm font-semibold text-foreground">
              {{ formatRoutingValue(routingMeta.complexity) }}
            </div>
          </div>
          <div class="bg-muted/10 border border-border/40 rounded-md p-3 flex flex-col gap-1">
            <span class="text-[11px] text-muted-foreground uppercase tracking-wider font-semibold">
              {{ t("logs.routing.savings") }}
            </span>
            <div class="font-mono text-xs sm:text-sm font-semibold text-foreground">
              {{ formatRoutingValue(routingMeta.savings) }}
            </div>
          </div>
        </div>

        <!-- Decision Parameters (Parsed Reasoning) -->
        <div v-if="parsedReasoning && parsedReasoning.length > 0" class="space-y-2.5">
          <span class="text-xs text-muted-foreground uppercase tracking-wider font-semibold block">
            {{ t("logs.routing.decisionParameters") }}
          </span>
          <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
            <div
              v-for="item in parsedReasoning"
              :key="item.key || item.value"
              class="relative flex flex-col gap-1.5 p-3 rounded-lg border transition-all duration-200"
              :class="
                item.active
                  ? 'border-foreground/50 bg-muted/10 shadow-sm'
                  : 'border-border/30 bg-muted/10 hover:border-border/50'
              "
            >
              <div class="flex items-center justify-between gap-2">
                <span
                  class="text-[11px] text-muted-foreground font-mono truncate uppercase tracking-wider font-semibold"
                  :title="item.key || 'Value'"
                >
                  {{ item.key || "Value" }}
                </span>
                <Badge
                  v-if="item.active"
                  variant="outline"
                  class="text-[11px] h-4 py-0 px-1.5 bg-muted/10 border-border/20 text-foreground uppercase font-mono font-bold"
                >
                  Active
                </Badge>
              </div>
              <div
                class="text-xs font-mono font-medium truncate text-foreground mt-0.5"
                :title="item.value"
              >
                {{ item.value }}
              </div>
            </div>
          </div>
        </div>

        <!-- Raw Reasoning (if available) -->
        <div v-if="routingReasoning" class="space-y-1.5">
          <span class="text-xs text-muted-foreground uppercase tracking-wider font-semibold block">
            {{ t("logs.routing.rawReasoning") }}
          </span>
          <p
            class="text-xs text-foreground/90 bg-muted/30 p-2.5 rounded-lg border border-border/20 font-mono leading-relaxed break-all"
          >
            {{ routingReasoning }}
          </p>
        </div>

        <!-- Verbose Diagnostics -->
        <template v-if="hasRoutingDiagnostics">
          <div class="border-t border-border/30 pt-5 space-y-5">
            <!-- Weights -->
            <div v-if="routingWeights && Object.keys(routingWeights).length > 0" class="space-y-2">
              <span
                class="text-xs text-muted-foreground uppercase tracking-wider font-semibold block"
              >
                {{ t("logs.routing.weights") }}
              </span>
              <div class="grid grid-cols-2 sm:grid-cols-4 gap-2">
                <div
                  v-for="(weight, key) in routingWeights"
                  :key="key"
                  class="bg-muted/30 px-2.5 py-1.5 rounded border border-border/20"
                >
                  <div class="text-[11px] text-muted-foreground capitalize">{{ key }}</div>
                  <div class="text-xs font-mono font-medium mt-0.5">
                    {{ formatRoutingValue(weight) }}
                  </div>
                </div>
              </div>
            </div>

            <!-- Signal votes -->
            <div
              v-if="routingSignalVotes && Object.keys(routingSignalVotes).length > 0"
              class="space-y-2"
            >
              <span
                class="text-xs text-muted-foreground uppercase tracking-wider font-semibold block"
              >
                {{ t("logs.routing.signalVotes") }}
              </span>
              <div class="grid grid-cols-1 sm:grid-cols-3 gap-2">
                <div
                  v-for="(vote, key) in routingSignalVotes"
                  :key="key"
                  class="bg-muted/30 px-2.5 py-1.5 rounded border border-border/20"
                >
                  <div class="text-[11px] text-muted-foreground capitalize">{{ key }}</div>
                  <div class="text-xs font-mono font-medium mt-0.5">
                    {{ formatRoutingValue(vote) }}
                  </div>
                </div>
              </div>
            </div>

            <!-- Guardrail notes -->
            <div v-if="routingGuardrailNotes && routingGuardrailNotes.length > 0" class="space-y-2">
              <span
                class="text-xs text-muted-foreground uppercase tracking-wider font-semibold block"
              >
                {{ t("logs.routing.guardrails") }}
              </span>
              <ul class="list-disc list-inside text-xs text-foreground/90 space-y-1 pl-1">
                <li v-for="(note, idx) in routingGuardrailNotes" :key="idx">{{ note }}</li>
              </ul>
            </div>

            <!-- Candidate scorecards -->
            <div v-if="routingScorecards && routingScorecards.length > 0" class="space-y-2">
              <span
                class="text-xs text-muted-foreground uppercase tracking-wider font-semibold block"
              >
                {{ t("logs.routing.candidateScores") }}
              </span>
              <div class="overflow-x-auto rounded-lg border border-border/40">
                <Table class="table-modern">
                  <TableHeader>
                    <TableRow>
                      <TableHead class="text-xs">{{ t("logs.routing.model") }}</TableHead>
                      <TableHead class="text-xs text-right">{{
                        t("logs.routing.total")
                      }}</TableHead>
                      <TableHead class="text-xs text-right">{{ t("logs.routing.cost") }}</TableHead>
                      <TableHead class="text-xs text-right">{{
                        t("logs.routing.latency")
                      }}</TableHead>
                      <TableHead class="text-xs text-right">{{
                        t("logs.routing.reliability")
                      }}</TableHead>
                      <TableHead class="text-xs text-right">{{
                        t("logs.routing.quality")
                      }}</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    <TableRow v-for="score in routingScorecards" :key="score.model">
                      <TableCell class="font-mono text-xs">{{ score.model }}</TableCell>
                      <TableCell class="text-xs font-mono text-right">{{
                        formatRoutingValue(score.total)
                      }}</TableCell>
                      <TableCell class="text-xs font-mono text-right">{{
                        formatRoutingValue(score.cost)
                      }}</TableCell>
                      <TableCell class="text-xs font-mono text-right">{{
                        formatRoutingValue(score.latency)
                      }}</TableCell>
                      <TableCell class="text-xs font-mono text-right">{{
                        formatRoutingValue(score.reliability)
                      }}</TableCell>
                      <TableCell class="text-xs font-mono text-right">
                        {{ formatRoutingValue(score.predicted_quality ?? score.quality_alignment) }}
                      </TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </div>
            </div>
          </div>
        </template>

        <p v-else class="text-xs text-muted-foreground italic">
          {{ t("logs.routing.standardOnly") }}
        </p>
      </TabsContent>

      <!-- Metadata View -->
      <TabsContent value="metadata" class="m-0 focus-visible:ring-0">
        <LogMetadataView :log="log" />
      </TabsContent>
    </div>
  </Tabs>
</template>

<script lang="ts">
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
</script>

<style scoped>
@media (prefers-reduced-motion: reduce) {
  .animate-spin-slow {
    animation: none;
  }
}
</style>
