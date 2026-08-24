<script setup lang="ts">
import {
  ChevronRight,
  Clock,
  Coins,
  Cpu,
  CornerDownRight,
  RotateCcw,
  User,
  Zap,
  type LucideIcon,
} from "@lucide/vue";
import { computed, type ComputedRef } from "vue";
import { useI18n } from "vue-i18n";
import StatusBadge from "@/components/common/StatusBadge.vue";
import { useAuditLabels } from "@/composables/useAuditLabels";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  formatCost,
  formatDate,
  formatDuration,
  getActionFromEndpoint,
  getActor,
  getStatusType,
  formatApiKeyName,
  type LogItem,
} from "@/utils/format";
import { sanitizeHighlightText } from "@/utils/sanitize";
import { getProviderIconUrl, isMonoProvider } from "@/utils/icons";

interface MetricItem {
  icon: LucideIcon;
  label: string;
  value: string;
  mono?: boolean;
  truncate?: boolean;
  provider?: string;
}

interface Props {
  log: LogItem;
  searchQuery?: string;
  tab: "proxy" | "audit" | "mcp" | "websearch";
}

const props = withDefaults(defineProps<Props>(), {
  searchQuery: "",
});

const emit = defineEmits<{
  viewDetails: [log: LogItem];
}>();

const { t } = useI18n();

const getTokenCount = (log: LogItem, type: "prompt" | "completion" | "total") => {
  switch (type) {
    case "prompt":
      return log.prompt_tokens || (log.log_metadata?.prompt_tokens as number) || 0;
    case "completion":
      return log.completion_tokens || (log.log_metadata?.completion_tokens as number) || 0;
    default:
      return log.total_tokens || (log.log_metadata?.total_tokens as number) || 0;
  }
};

const formatTokenBreakdown = (log: LogItem): string => {
  const input = getTokenCount(log, "prompt");
  const output = getTokenCount(log, "completion");

  if (input === 0 && output === 0) return "-";
  if (output === 0 && input > 0) return input.toLocaleString();
  return `${input.toLocaleString()}/${output.toLocaleString()}`;
};

const getMcpOperation = (log: LogItem): string => {
  return (log.log_metadata?.mcp_operation as string | undefined) || "-";
};

const getMcpResourceName = (log: LogItem): string => {
  return (log.log_metadata?.mcp_resource_name as string | undefined) || "-";
};

const getMcpServer = (log: LogItem): string => {
  return (log.log_metadata?.mcp_server as string | undefined) || log.model || "-";
};

const getWebSearchQuery = (log: LogItem): string => {
  return (log.log_metadata?.web_search_query as string | undefined) || "-";
};

const getWebSearchResultCount = (log: LogItem): number => {
  return (log.log_metadata?.web_search_result_count as number | undefined) || 0;
};

const getWebSearchStatus = (log: LogItem): string => {
  return (log.log_metadata?.web_search_status as string | undefined) || "unknown";
};

const getProviderModelName = (log: LogItem): string | null =>
  (log.log_metadata?.provider_model_name as string | undefined) || null;

const getRequestedModel = (log: LogItem): string | null =>
  (log.log_metadata?.routing?.requested_model as string | undefined) || null;

const getResolvedModel = (log: LogItem): string | null =>
  (log.log_metadata?.routing?.resolved_model as string | undefined) || log.model || null;

const getJwtUsername = (log: LogItem): string | null => log.user_identity || null;

// Audit semantic labels (backend-derived, with client-side fallback)
const { formatEventType, formatActionCategory } = useAuditLabels();

// Backend-aware action label for audit rows. Prefers the backend's curated
// action_category and falls back to endpoint-derived labels.
const auditActionLabel = (log: LogItem): string => {
  const cat = log.action_category;
  if (cat) {
    const verb = formatActionCategory(cat);
    const parts = log.endpoint.split("/").filter(Boolean);
    const resource = parts.at(-1) || parts.at(-2) || "unknown";
    const formattedResource =
      resource.charAt(0).toUpperCase() + resource.slice(1).replace(/-/g, " ");
    return `${verb} ${formattedResource}`;
  }
  return getActionFromEndpoint(log.method, log.endpoint);
};

const primaryContent = computed(() => {
  switch (props.tab) {
    case "proxy": {
      const requestedModel = getRequestedModel(props.log);
      const resolvedModel = getResolvedModel(props.log);
      const hasRoutingDisplay = requestedModel && resolvedModel && requestedModel !== resolvedModel;
      return {
        label: t("logs.model"),
        value: requestedModel || props.log.model,
        badge: true,
        subValue: hasRoutingDisplay ? resolvedModel : getProviderModelName(props.log),
      };
    }
    case "audit":
      return {
        label: t("logs.action"),
        value: auditActionLabel(props.log),
        badge: false,
        subValue: props.log.event_type ? formatEventType(props.log.event_type) : null,
      };
    case "mcp":
      return {
        label: t("logs.mcpServer"),
        value: getMcpServer(props.log),
        badge: true,
        subValue: null,
      };
    case "websearch":
      return {
        label: t("logs.webSearchQuery"),
        value: getWebSearchQuery(props.log),
        badge: false,
        subValue: null,
      };
    default:
      return { label: "", value: "", badge: false, subValue: null };
  }
});

const secondaryMetrics: ComputedRef<MetricItem[]> = computed(() => {
  const metrics: MetricItem[] = [];

  switch (props.tab) {
    case "proxy": {
      if (props.log.provider) {
        metrics.push({
          icon: Cpu,
          label: t("logs.provider"),
          value: props.log.provider,
          provider: props.log.provider,
        });
      }
      // Show API key or JWT user identity
      if (props.log.auth_method === "jwt") {
        metrics.push({
          icon: User,
          label: t("logs.user"),
          value: getJwtUsername(props.log) || "Admin",
        });
      } else if (props.log.api_key_name) {
        metrics.push({
          icon: Cpu,
          label: t("logs.apiKey"),
          value: formatApiKeyName(props.log.api_key_name),
          mono: true,
        });
      }
      metrics.push({
        icon: Zap,
        label: t("logs.tokens"),
        value: formatTokenBreakdown(props.log),
        mono: true,
      });
      if (props.log.ttft_ms || props.log.log_metadata?.ttft_ms) {
        metrics.push({
          icon: Clock,
          label: t("logs.ttft"),
          value: formatDuration(
            props.log.ttft_ms ?? (props.log.log_metadata?.ttft_ms as number | undefined)
          ),
          mono: true,
        });
      }
      const costValue = props.log.cost_usd ?? props.log.log_metadata?.cost_usd;
      if (costValue !== undefined && costValue !== null) {
        metrics.push({
          icon: Coins,
          label: t("logs.cost"),
          value: formatCost(costValue as number),
          mono: true,
        });
      }
      break;
    }
    case "audit":
      metrics.push({
        icon: Cpu,
        label: t("logs.actor"),
        value: getActor(props.log),
      });
      if (props.log.client_ip) {
        metrics.push({
          icon: Cpu,
          label: t("logs.ipAddress"),
          value: props.log.client_ip,
          mono: true,
        });
      }
      metrics.push({
        icon: Zap,
        label: t("logs.resource"),
        value: props.log.endpoint,
        mono: true,
        truncate: true,
      });
      break;
    case "mcp":
      metrics.push({
        icon: Cpu,
        label: t("logs.mcpOperation"),
        value: getMcpOperation(props.log),
      });
      metrics.push({
        icon: Zap,
        label: t("logs.mcpResource"),
        value: getMcpResourceName(props.log),
        mono: true,
        truncate: true,
      });
      break;
    case "websearch":
      metrics.push({
        icon: Cpu,
        label: t("logs.webSearchProvider"),
        value: props.log.provider || "searxng",
        provider: props.log.provider || "searxng",
      });
      metrics.push({
        icon: Zap,
        label: t("logs.webSearchCount"),
        value: getWebSearchResultCount(props.log).toString(),
        mono: true,
      });
      break;
  }

  return metrics;
});

const handleViewDetails = () => {
  emit("viewDetails", props.log);
};

// Retry attempts count for the retries badge: same-provider retries +
// fallback (provider-switch) attempts. Both are surfaced in log_metadata.
const totalRetryCount = computed(() => {
  const meta = props.log.log_metadata;
  let count = 0;
  if (Array.isArray(meta?.retry_attempts)) {
    count += meta.retry_attempts.filter(({ retried }: { retried?: boolean }) => retried).length;
  }
  if (Array.isArray(meta?.fallback_attempts)) {
    count += meta.fallback_attempts.length;
  }
  return count;
});
</script>

<template>
  <article
    class="log-list-item group relative flex flex-col gap-2 p-3 rounded-xl border border-border bg-card shadow-xs hover:bg-muted/50 hover:border-border/80 transition-colors duration-200 cursor-pointer"
    role="button"
    tabindex="0"
    :aria-label="`${t('logs.viewDetails')} - ${primaryContent.value || '-'}`"
    @click="handleViewDetails"
    @keydown.enter="handleViewDetails"
    @keydown.space.prevent="handleViewDetails"
  >
    <header class="flex items-start justify-between gap-2">
      <div class="flex items-center gap-2 min-w-0 flex-1">
        <time class="text-code-xs text-muted-foreground whitespace-nowrap shrink-0">
          {{ formatDate(log.timestamp) }}
        </time>
        <StatusBadge
          :variant="tab === 'audit' ? 'http' : 'status'"
          :status="getStatusType(log.status_code)"
          :http-method="tab === 'audit' ? log.method : undefined"
          class="shrink-0"
        >
          <template v-if="tab === 'websearch'">{{ getWebSearchStatus(log) }}</template>
          <template v-else-if="tab === 'audit'">{{ log.method }}</template>
          <template v-else>{{ log.status_code }}</template>
        </StatusBadge>
        <Badge
          v-if="totalRetryCount > 0"
          class="text-xs bg-action-amber/15 text-action-amber border-action-amber/30 shrink-0"
        >
          <RotateCcw class="size-3 me-1.5" />
          {{ t("logs.retries", { count: totalRetryCount }) }}
        </Badge>
      </div>
      <div class="flex items-center gap-1 shrink-0">
        <span
          class="flex items-center gap-1 text-code-xs text-muted-foreground"
          :title="t('logs.duration')"
        >
          <Clock class="size-3.5" />
          <span>{{ formatDuration(log.response_time_ms) }}</span>
        </span>
      </div>
    </header>

    <div class="flex items-center gap-2 min-w-0">
      <span class="text-xs text-muted-foreground shrink-0">{{ primaryContent.label }}:</span>
      <div class="flex flex-col gap-0.5 min-w-0">
        <Badge
          v-if="primaryContent.badge"
          variant="secondary"
          class="font-mono text-xs truncate max-w-full self-start"
        >
          <span v-html="sanitizeHighlightText(primaryContent.value || '-', searchQuery)"></span>
        </Badge>
        <span
          v-else
          class="text-sm font-medium truncate"
          v-html="sanitizeHighlightText(primaryContent.value || '-', searchQuery)"
        ></span>
        <div
          v-if="primaryContent.subValue && primaryContent.subValue !== primaryContent.value"
          class="flex items-center gap-1 text-xs text-muted-foreground/80 font-mono pl-1 mt-1"
        >
          <CornerDownRight class="size-3 text-muted-foreground/50 shrink-0" />
          <span class="truncate">{{ primaryContent.subValue }}</span>
        </div>
      </div>
    </div>

    <div class="grid grid-cols-2 gap-x-4 gap-y-2 overflow-hidden">
      <template v-for="metric in secondaryMetrics" :key="metric.label">
        <div class="flex items-center gap-1.5 min-w-0">
          <div
            v-if="metric.provider && getProviderIconUrl(metric.provider)"
            class="size-4 rounded flex items-center justify-center overflow-hidden bg-background border border-border/50 shrink-0"
          >
            <img
              :src="getProviderIconUrl(metric.provider)!"
              :class="[
                isMonoProvider(metric.provider) ? 'icon-mono' : null,
                'size-3 object-contain',
              ]"
              loading="lazy"
            />
          </div>
          <component :is="metric.icon" v-else class="size-3.5 text-muted-foreground shrink-0" />
          <span class="text-xs text-muted-foreground shrink-0">{{ metric.label }}:</span>
          <span
            :class="[
              'text-xs truncate',
              metric.mono ? 'font-mono text-muted-foreground' : 'font-medium',
            ]"
            :title="metric.value"
          >
            <span v-html="sanitizeHighlightText(metric.value, searchQuery)"></span>
          </span>
        </div>
      </template>
    </div>

    <div
      class="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 group-focus:opacity-100 transition-opacity"
    >
      <Button
        size="icon"
        variant="ghost"
        class="h-11 w-11 min-h-11 min-w-11"
        :aria-label="t('logs.viewDetails')"
        @click.stop="handleViewDetails"
      >
        <ChevronRight class="size-5 text-muted-foreground" />
      </Button>
    </div>
  </article>
</template>
