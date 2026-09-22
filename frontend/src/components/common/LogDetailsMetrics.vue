<script setup lang="ts">
import {
  Activity,
  Clock,
  Coins,
  Cpu,
  CornerDownRight,
  Globe,
  Search,
  Server,
  Shield,
  User,
  Wrench,
} from "@lucide/vue";
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import StatusBadge from "@/components/common/StatusBadge.vue";
import MetricCell from "@/components/common/MetricCell.vue";
import { useAuditLabels } from "@/composables/useAuditLabels";
import { Badge } from "@/components/ui/badge";
import type { LogRead } from "@/types/schemas";
import {
  formatCost,
  formatDuration,
  formatTokens,
  getActionFromEndpoint,
  getActor,
} from "@/utils/format";
import { getProviderIconUrl, isMonoProvider } from "@/utils/icons";
import {
  cachedTokens,
  completionTokens,
  costUsd,
  promptTokens,
  ttftMs,
} from "@/utils/logEntryAccessors";

const props = defineProps<{
  log: LogRead;
}>();

const { t } = useI18n();

const isAuditLog = computed(() => props.log.log_type === "audit");
const isMcpLog = computed(() => props.log.log_type === "mcp");
const isWebSearchLog = computed(() => props.log.log_type === "web_search");
const isToolLog = computed(() => isMcpLog.value || isWebSearchLog.value);
const isStreaming = computed(() => props.log.log_metadata?.streaming === true);

// Audit semantic labels (backend-derived, with client-side fallback)
const { formatEventType, formatActionCategory, formatResourceType, formatOutcome, outcomeStatus } =
  useAuditLabels();

// Prefer the backend's curated action classification (action_category +
// resource_type) and fall back to endpoint-derived labels for legacy logs.
const auditActionLabel = computed(() => {
  const cat = props.log.action_category;
  const res = props.log.resource_type;
  if (cat) {
    const verb = formatActionCategory(cat);
    const object = res ? formatResourceType(res) : null;
    return object ? `${verb} ${object}` : verb;
  }
  return getActionFromEndpoint(props.log.method, props.log.endpoint);
});

// Provider model name (actual model sent to provider API)
const providerModelName = computed(() => {
  if (!props.log.log_metadata) return null;
  const val = props.log.log_metadata.provider_model_name;
  return typeof val === "string" ? val : null;
});

// TTFT, token chips and TPS all resolve through logEntryAccessors so the
// metrics panel agrees with the list rows and the response view.
//
// Calculate TPS (tokens per second)
const tps = computed(() => {
  const completion = completionTokens(props.log);
  const responseTime = props.log.response_time_ms;
  if (!completion || !responseTime || responseTime <= 0) return null;
  return Math.round((completion / responseTime) * 1000);
});

// Cache savings
const cacheSavings = computed((): number | null => {
  const val = props.log.log_metadata?.cache_savings_usd;
  return typeof val === "number" ? val : null;
});

// MCP log metadata
const mcpMetadata = computed(() => {
  if (!isMcpLog.value || !props.log.log_metadata) return null;
  const meta = props.log.log_metadata as Record<string, unknown>;
  return {
    server: (meta.mcp_server as string) || props.log.model || "-",
    operation: meta.mcp_operation as string | undefined,
    resourceType: meta.mcp_resource_type as string | undefined,
    resourceName: meta.mcp_resource_name as string | undefined,
  };
});

// Format MCP operation type for display
const formatMcpOperation = (op: string | undefined): string => {
  if (!op) return "-";
  const mapping: Record<string, string> = {
    tool_call: t("logs.toolCall"),
    tool_list: t("logs.toolList"),
    resource_read: t("logs.resourceRead"),
    resource_list: t("logs.resourceList"),
    prompt_get: t("logs.promptGet"),
    prompt_list: t("logs.promptList"),
  };
  return mapping[op] || op;
};

// Web Search log metadata
const webSearchMetadata = computed(() => {
  if (!isWebSearchLog.value || !props.log.log_metadata) return null;
  const meta = props.log.log_metadata as Record<string, unknown>;
  return {
    query: meta.web_search_query as string | undefined,
    status: meta.web_search_status as string | undefined,
    resultCount: (meta.web_search_result_count as number) ?? 0,
    provider: meta.web_search_provider as string | undefined,
    maxUses: meta.web_search_max_uses as number | undefined,
    currentUse: meta.web_search_current_use as number | undefined,
  };
});

// Status code → semantic color (success / warning / error) for the hero number
const statusColorClass = computed(() => {
  const code = props.log.status_code;
  if (code && code >= 200 && code < 300) return "text-status-success";
  if (code && code >= 400 && code < 500) return "text-status-warning";
  if (code) return "text-status-error";
  return "text-muted-foreground";
});
</script>

<template>
  <!-- Editorial spec-sheet: a single framed panel with hairline-divided cells.
       Replaces the former identical-card grid to read as a calm technical readout. -->
  <div class="rounded-lg border border-border/60 overflow-hidden bg-card">
    <div class="grid grid-cols-2 lg:grid-cols-4 gap-px bg-border/40">
      <!-- 1. PROXY LOG METRICS -->
      <template v-if="!isAuditLog && !isToolLog">
        <!-- Cell: Model & Provider -->
        <MetricCell :label="t('logs.model')">
          <template #icon><Cpu class="size-3 text-muted-foreground/50" /></template>
          <div class="flex flex-col gap-1 sm:gap-1.5 mt-1 min-w-0">
            <span
              class="text-xs sm:text-sm font-semibold truncate text-foreground"
              :title="log.model || ''"
            >
              {{ log.model || "—" }}
            </span>
            <div
              v-if="providerModelName && providerModelName !== log.model"
              class="flex items-center gap-1 text-[11px] text-muted-foreground/80 font-mono"
              :title="t('logs.providerModel')"
            >
              <CornerDownRight class="size-3 text-muted-foreground/50 shrink-0" />
              <span class="truncate">{{ providerModelName }}</span>
            </div>
            <div v-if="log.provider" class="flex items-center gap-1.5 mt-0.5">
              <div
                v-if="getProviderIconUrl(log.provider)"
                class="size-4 rounded-full overflow-hidden bg-background border border-border/50 flex items-center justify-center shrink-0"
              >
                <img
                  :src="getProviderIconUrl(log.provider)!"
                  :class="[
                    isMonoProvider(log.provider) ? 'icon-mono' : null,
                    'size-2.5 object-contain',
                  ]"
                />
              </div>
              <span class="text-[11px] text-muted-foreground font-medium capitalize">{{
                log.provider
              }}</span>
            </div>
          </div>
        </MetricCell>

        <!-- Cell: Status & Request -->
        <MetricCell :label="t('logs.status')">
          <template #icon><Activity class="size-3 text-muted-foreground/50" /></template>
          <div class="flex flex-col gap-1 mt-1 min-w-0">
            <div class="flex items-center gap-2">
              <span :class="['text-metric', statusColorClass]">
                {{ log.status_code || "—" }}
              </span>
              <Badge variant="outline" class="font-mono text-[11px] uppercase font-bold py-0">
                {{ log.method }}
              </Badge>
            </div>
            <span
              class="text-[11px] text-muted-foreground truncate font-mono mt-0.5"
              :title="log.endpoint || ''"
            >
              {{ log.endpoint }}
            </span>
          </div>
        </MetricCell>

        <!-- Cell: Duration & Performance -->
        <MetricCell :label="t('logs.duration')">
          <template #icon><Clock class="size-3 text-muted-foreground/50" /></template>
          <div class="flex flex-col gap-1.5 mt-1">
            <span class="text-metric text-foreground">
              {{ formatDuration(log.response_time_ms) }}
            </span>
            <div class="flex items-center gap-1.5 flex-wrap">
              <span
                v-if="ttftMs(log) !== null"
                class="text-[11px] text-muted-foreground font-mono"
                :title="t('logs.ttft')"
              >
                TTFT {{ ttftMs(log) }}ms
              </span>
              <span
                v-if="ttftMs(log) !== null && tps !== null"
                class="text-[11px] text-muted-foreground"
                >·</span
              >
              <span
                v-if="tps !== null"
                class="text-[11px] text-muted-foreground font-mono"
                :title="t('logs.tps')"
              >
                {{ tps }} tok/s
              </span>
              <span
                v-if="isStreaming"
                class="text-[11px] bg-action-blue/10 border border-action-blue/20 text-action-blue font-medium px-1.5 py-0.5 rounded"
              >
                {{ t("logs.streamBadge") }}
              </span>
            </div>
          </div>
        </MetricCell>

        <!-- Cell: Tokens & Cost -->
        <MetricCell :label="t('logs.cost')">
          <template #icon><Coins class="size-3 text-muted-foreground/50" /></template>
          <div class="flex flex-col gap-1.5 mt-1">
            <span class="text-metric text-foreground">
              {{ formatCost(costUsd(log)) }}
            </span>
            <div class="flex items-center gap-1.5 flex-wrap">
              <span
                class="text-[11px] text-muted-foreground font-mono"
                :title="`${t('logs.inputTokens')}: ${formatTokens(promptTokens(log))} | ${t('logs.outputTokens')}: ${formatTokens(completionTokens(log))}`"
              >
                {{ formatTokens(promptTokens(log)) }} → {{ formatTokens(completionTokens(log)) }}
              </span>
              <span
                v-if="cacheSavings !== null"
                class="text-[11px] font-semibold bg-status-success/15 border border-status-success/30 text-status-success px-1.5 py-0.5 rounded"
                :title="`${t('logs.cacheSavings')}: -${formatCost(cacheSavings)}`"
              >
                {{ t("logs.saved") }} {{ formatCost(cacheSavings) }}
              </span>
              <span
                v-if="cachedTokens(log) > 0"
                class="text-[11px] font-semibold bg-action-blue/15 border border-action-blue/30 text-action-blue px-1.5 py-0.5 rounded"
                :title="`${t('logs.cachedTokens')}: ${formatTokens(cachedTokens(log))}`"
              >
                {{ t("logs.cachedTokens") }} {{ formatTokens(cachedTokens(log)) }}
              </span>
            </div>
          </div>
        </MetricCell>
      </template>

      <!-- 2. AUDIT LOG METRICS -->
      <template v-else-if="isAuditLog">
        <!-- Cell: Action -->
        <MetricCell :label="t('logs.action')">
          <template #icon><Shield class="size-3 text-muted-foreground/50" /></template>
          <div class="flex flex-col gap-1.5 mt-1 min-w-0">
            <span
              class="text-xs sm:text-sm font-semibold font-mono truncate text-foreground"
              :title="auditActionLabel"
            >
              {{ auditActionLabel }}
            </span>
            <div class="flex items-center gap-1.5 flex-wrap">
              <Badge variant="outline" class="font-mono text-[11px] uppercase font-bold py-0">
                {{ log.method }}
              </Badge>
              <Badge v-if="log.event_type" variant="secondary" class="text-[11px] font-medium py-0">
                {{ formatEventType(log.event_type) }}
              </Badge>
            </div>
          </div>
        </MetricCell>

        <!-- Cell: Actor -->
        <MetricCell :label="t('logs.actor')">
          <template #icon><User class="size-3 text-muted-foreground/50" /></template>
          <div class="flex flex-col gap-1 mt-1 min-w-0">
            <span
              class="text-xs sm:text-sm font-bold text-foreground truncate"
              :title="getActor(log)"
            >
              {{ getActor(log) }}
            </span>
            <span
              v-if="log.client_ip"
              class="text-[11px] text-muted-foreground font-mono truncate"
              :title="t('logs.ipAddress')"
            >
              {{ log.client_ip }}
            </span>
          </div>
        </MetricCell>

        <!-- Cell: Status -->
        <MetricCell :label="t('logs.status')">
          <template #icon><Activity class="size-3 text-muted-foreground/50" /></template>
          <div class="flex flex-col gap-1 mt-1 min-w-0">
            <span :class="['text-metric', statusColorClass]">
              {{ log.status_code || "—" }}
            </span>
            <StatusBadge
              v-if="log.outcome"
              variant="status"
              :status="outcomeStatus(log.outcome)"
              class="self-start mt-0.5"
            >
              {{ formatOutcome(log.outcome) }}
            </StatusBadge>
            <span
              class="text-[11px] text-muted-foreground truncate font-mono mt-0.5"
              :title="log.endpoint || ''"
            >
              {{ log.endpoint }}
            </span>
          </div>
        </MetricCell>

        <!-- Cell: Latency -->
        <MetricCell :label="t('logs.latency')">
          <template #icon><Clock class="size-3 text-muted-foreground/50" /></template>
          <div class="flex flex-col gap-1 mt-1">
            <span class="text-metric text-foreground">
              {{ formatDuration(log.response_time_ms) }}
            </span>
            <span class="text-[11px] text-muted-foreground font-mono">{{
              t("logs.duration")
            }}</span>
          </div>
        </MetricCell>
      </template>

      <!-- 3. MCP LOG METRICS -->
      <template v-else-if="isMcpLog && mcpMetadata">
        <!-- Cell: Server -->
        <MetricCell :label="t('logs.mcpServer')">
          <template #icon><Server class="size-3 text-muted-foreground/50" /></template>
          <div class="flex flex-col gap-1.5 mt-1 min-w-0">
            <span
              class="text-xs sm:text-sm font-semibold font-mono truncate text-foreground"
              :title="mcpMetadata.server"
            >
              {{ mcpMetadata.server }}
            </span>
            <span class="text-[11px] text-muted-foreground font-mono truncate">
              {{ t("logs.type") }}: {{ mcpMetadata.resourceType || "—" }}
            </span>
          </div>
        </MetricCell>

        <!-- Cell: Operation -->
        <MetricCell :label="t('logs.mcpOperation')">
          <template #icon><Wrench class="size-3 text-muted-foreground/50" /></template>
          <div class="flex flex-col gap-1.5 mt-1 min-w-0">
            <span class="text-xs sm:text-sm font-bold text-foreground truncate">
              {{ formatMcpOperation(mcpMetadata.operation) }}
            </span>
            <span
              class="text-[11px] text-muted-foreground font-mono truncate max-w-full"
              :title="mcpMetadata.resourceName || ''"
            >
              {{ mcpMetadata.resourceName || "—" }}
            </span>
          </div>
        </MetricCell>

        <!-- Cell: Status -->
        <MetricCell :label="t('logs.status')">
          <template #icon><Activity class="size-3 text-muted-foreground/50" /></template>
          <div class="flex flex-col gap-1 mt-1">
            <span :class="['text-metric', statusColorClass]">
              {{ log.status_code || "—" }}
            </span>
            <span class="text-[11px] text-muted-foreground font-mono mt-0.5">
              {{ t("logs.httpCode") }}
            </span>
          </div>
        </MetricCell>

        <!-- Cell: Latency -->
        <MetricCell :label="t('logs.latency')">
          <template #icon><Clock class="size-3 text-muted-foreground/50" /></template>
          <div class="flex flex-col gap-1 mt-1">
            <span class="text-metric text-foreground">
              {{ formatDuration(log.response_time_ms) }}
            </span>
            <span class="text-[11px] text-muted-foreground font-mono">{{
              t("logs.duration")
            }}</span>
          </div>
        </MetricCell>
      </template>

      <!-- 4. WEB SEARCH LOG METRICS -->
      <template v-else-if="isWebSearchLog && webSearchMetadata">
        <!-- Cell: Search Query -->
        <MetricCell :label="t('logs.searchQuery')">
          <template #icon><Search class="size-3 text-muted-foreground/50" /></template>
          <div class="flex flex-col gap-1.5 mt-1 min-w-0">
            <span
              class="text-xs sm:text-sm font-semibold font-mono truncate text-foreground"
              :title="webSearchMetadata.query"
            >
              {{ webSearchMetadata.query || "—" }}
            </span>
            <span class="text-[11px] text-muted-foreground font-mono truncate">
              {{ t("logs.queryString") }}
            </span>
          </div>
        </MetricCell>

        <!-- Cell: Provider -->
        <MetricCell :label="t('logs.webSearchProvider')">
          <template #icon><Globe class="size-3 text-muted-foreground/50" /></template>
          <div class="flex flex-col gap-1.5 mt-1 min-w-0">
            <div class="flex items-center gap-1.5">
              <div
                v-if="getProviderIconUrl(log.provider || 'searxng')"
                class="size-4 rounded-full overflow-hidden bg-background border border-border/50 flex items-center justify-center shrink-0"
              >
                <img
                  :src="getProviderIconUrl(log.provider || 'searxng')!"
                  :class="[
                    isMonoProvider(log.provider || 'searxng') ? 'icon-mono' : null,
                    'size-2.5 object-contain',
                  ]"
                />
              </div>
              <span class="text-xs sm:text-sm font-bold text-foreground capitalize truncate">{{
                log.provider || "searxng"
              }}</span>
            </div>
            <span class="text-[11px] text-muted-foreground font-mono">
              {{ t("logs.webSearchCount") }}: {{ webSearchMetadata.resultCount }}
            </span>
          </div>
        </MetricCell>

        <!-- Cell: Status -->
        <MetricCell :label="t('logs.webSearchStatus')">
          <template #icon><Activity class="size-3 text-muted-foreground/50" /></template>
          <div class="flex flex-col gap-1.5 mt-1 min-w-0">
            <span
              :class="[
                'inline-flex items-center self-start font-mono text-xs sm:text-sm font-bold uppercase py-0.5 px-2 sm:px-2.5 rounded-md border',
                webSearchMetadata.status === 'success'
                  ? 'bg-status-success/15 border-status-success/30 text-status-success'
                  : 'bg-status-error/15 border-status-error/30 text-status-error',
              ]"
            >
              {{ webSearchMetadata.status || "unknown" }}
            </span>
            <span class="text-[11px] text-muted-foreground font-mono">
              {{ t("logs.httpCode") }}: {{ log.status_code || "—" }}
            </span>
          </div>
        </MetricCell>

        <!-- Cell: Latency -->
        <MetricCell :label="t('logs.latency')">
          <template #icon><Clock class="size-3 text-muted-foreground/50" /></template>
          <div class="flex flex-col gap-1 mt-1">
            <span class="text-metric text-foreground">
              {{ formatDuration(log.response_time_ms) }}
            </span>
            <span class="text-[11px] text-muted-foreground font-mono">{{
              t("logs.duration")
            }}</span>
          </div>
        </MetricCell>
      </template>
    </div>
  </div>
</template>
