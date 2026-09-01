<script setup lang="ts">
import { computed, ref } from "vue";
import { useI18n } from "vue-i18n";
import JsonViewer from "@/components/common/JsonViewer.vue";
import LogCollapsibleRow from "@/components/common/LogCollapsibleRow.vue";
import LogRequestView from "@/components/common/LogRequestView.vue";
import LogResponseView from "@/components/common/LogResponseView.vue";
import { Badge } from "@/components/ui/badge";
import type { LogRead } from "@/types/schemas";
import { hasPayload } from "@/utils/logFormat";
import { Brackets, Clock, Code, Database, Globe, Server, Terminal } from "@lucide/vue";

const props = defineProps<{
  log: LogRead;
}>();

const { t } = useI18n();

// Audit raw accordions
const showRawRequestHeaders = ref(false);
const showRawResponseHeaders = ref(false);

// Log type detection
const isMcpLog = computed(() => props.log?.log_type === "mcp");
const isWebSearchLog = computed(() => props.log?.log_type === "web_search");
const isAuditLog = computed(() => props.log?.log_type === "audit");

const hasRequestBody = computed(() => hasPayload(props.log?.request_body));

const hasResponseBody = computed(() => hasPayload(props.log?.response_body));

// MCP log data
const mcpMetadata = computed(() => {
  if (!isMcpLog.value || !props.log?.log_metadata) return null;
  const meta = props.log.log_metadata as Record<string, unknown>;
  return {
    server: (meta.mcp_server as string) || props.log.model || "-",
    operation: meta.mcp_operation as string | undefined,
    resourceType: meta.mcp_resource_type as string | undefined,
    resourceName: meta.mcp_resource_name as string | undefined,
    arguments: meta.mcp_arguments as Record<string, unknown> | undefined,
    resultSummary: meta.mcp_result_summary as Record<string, unknown> | undefined,
  };
});

// Web Search log data
const webSearchMetadata = computed(() => {
  if (!isWebSearchLog.value || !props.log?.log_metadata) return null;
  const meta = props.log.log_metadata as Record<string, unknown>;
  return {
    query: meta.web_search_query as string | undefined,
    status: meta.web_search_status as string | undefined,
    resultCount: meta.web_search_result_count as number | undefined,
    results: meta.web_search_results as Array<Record<string, unknown>> | undefined,
    provider: meta.web_search_provider as string | undefined,
    maxUses: meta.web_search_max_uses as number | undefined,
    currentUse: meta.web_search_current_use as number | undefined,
  };
});

// Format MCP operation type for display
const formatMcpOperation = (op: string | undefined): string => {
  if (!op) return "";
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
</script>

<template>
  <div v-if="log" class="space-y-6">
    <!-- 0. AUDIT LOG VIEW -->
    <template v-if="isAuditLog">
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start io-column">
        <!-- A. Request Column -->
        <div class="space-y-4">
          <div class="flex items-center justify-between pb-2 border-b border-border/60">
            <h3 class="font-semibold text-base text-foreground flex items-center gap-2">
              <Database class="size-4 text-muted-foreground" />
              <span>{{ t("logs.request") }}</span>
            </h3>
          </div>

          <!-- Request parameters/Headers/Body -->
          <div class="space-y-4">
            <!-- Headers Accordion -->
            <LogCollapsibleRow
              :open="showRawRequestHeaders"
              chevron-position="end"
              button-class="w-full flex items-center justify-between p-3.5 text-xs font-semibold text-muted-foreground hover:bg-muted/20 transition-colors uppercase tracking-wider cursor-pointer max-sm:min-h-11"
              chevron-class="size-4 text-muted-foreground/60"
              content-class="p-3 border-t border-border/25"
              @toggle="showRawRequestHeaders = !showRawRequestHeaders"
            >
              <template #header>
                <Database class="size-3.5" />
                {{ t("logs.requestHeaders") }}
              </template>
              <JsonViewer :data="log.request_headers" maxHeight="max-h-60" flat />
            </LogCollapsibleRow>

            <!-- Request Body View -->
            <div class="space-y-2">
              <h4 class="text-xs uppercase font-bold text-muted-foreground flex items-center gap-2">
                <Code class="size-3.5" />
                {{ t("logs.requestBody") }}
              </h4>
              <div
                v-if="hasRequestBody"
                class="border border-border/40 rounded-lg overflow-hidden bg-muted/10 p-3"
              >
                <JsonViewer :data="log.request_body" maxHeight="max-h-[500px]" :deep="3" flat />
              </div>
              <div
                v-else
                class="text-xs text-muted-foreground italic bg-muted/5 border border-border/40 p-4 rounded-lg text-center"
              >
                {{ t("logs.noRequestBody") }}
              </div>
            </div>
          </div>
        </div>

        <!-- B. Response Column -->
        <div class="space-y-4">
          <div class="flex items-center justify-between pb-2 border-b border-border/60">
            <h3 class="font-semibold text-base text-foreground flex items-center gap-2">
              <Terminal class="size-4 text-muted-foreground" />
              <span>{{ t("logs.response") }}</span>
            </h3>
          </div>

          <!-- Response parameters/Headers/Body -->
          <div class="space-y-4">
            <!-- Headers Accordion -->
            <LogCollapsibleRow
              :open="showRawResponseHeaders"
              chevron-position="end"
              button-class="w-full flex items-center justify-between p-3.5 text-xs font-semibold text-muted-foreground hover:bg-muted/20 transition-colors uppercase tracking-wider cursor-pointer max-sm:min-h-11"
              chevron-class="size-4 text-muted-foreground/60"
              content-class="p-3 border-t border-border/25"
              @toggle="showRawResponseHeaders = !showRawResponseHeaders"
            >
              <template #header>
                <Database class="size-3.5" />
                {{ t("logs.responseHeaders") }}
              </template>
              <JsonViewer :data="log.response_headers" maxHeight="max-h-60" flat />
            </LogCollapsibleRow>

            <!-- Response Body View -->
            <div class="space-y-2">
              <h4 class="text-xs uppercase font-bold text-muted-foreground flex items-center gap-2">
                <Brackets class="size-3.5" />
                {{ t("logs.responseBody") }}
              </h4>
              <div
                v-if="hasResponseBody"
                class="border border-border/40 rounded-lg overflow-hidden bg-muted/10 p-3"
              >
                <JsonViewer :data="log.response_body" maxHeight="max-h-[500px]" :deep="3" flat />
              </div>
              <div
                v-else
                class="text-xs text-muted-foreground italic bg-muted/5 border border-border/40 p-4 rounded-lg text-center"
              >
                {{ t("logs.noResponseBody") }}
              </div>
            </div>
          </div>
        </div>
      </div>
    </template>

    <!-- 1. MCP LOG VIEW -->
    <template v-else-if="isMcpLog && mcpMetadata">
      <div class="space-y-4">
        <div class="flex items-center gap-2 pb-2 border-b border-border/60">
          <div
            class="p-2.5 bg-muted/20 border border-border/40 rounded-lg shrink-0 group-hover:border-foreground/30 transition-colors"
          >
            <Server class="size-5 text-foreground/80" />
          </div>
          <h3 class="font-semibold text-lg flex items-center gap-2">
            <span>{{ t("logs.mcpServer") }}</span>
          </h3>
          <Badge variant="secondary" class="font-mono text-xs">MCP</Badge>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div class="bg-muted/10 border border-border/40 rounded-xl p-4 flex flex-col gap-1.5">
            <span class="text-[11px] text-muted-foreground uppercase font-bold tracking-wider"
              >Server</span
            >
            <span class="text-sm font-semibold font-mono text-foreground/90 truncate">{{
              mcpMetadata.server || "-"
            }}</span>
          </div>
          <div class="bg-muted/10 border border-border/40 rounded-xl p-4 flex flex-col gap-1.5">
            <span class="text-[11px] text-muted-foreground uppercase font-bold tracking-wider"
              >Operation</span
            >
            <span class="text-sm font-semibold text-foreground/90">{{
              formatMcpOperation(mcpMetadata.operation) || "-"
            }}</span>
          </div>
          <div class="bg-muted/10 border border-border/40 rounded-xl p-4 flex flex-col gap-1.5">
            <span class="text-[11px] text-muted-foreground uppercase font-bold tracking-wider"
              >Resource Type</span
            >
            <span class="text-sm font-semibold font-mono text-foreground/90 truncate">{{
              mcpMetadata.resourceType || "-"
            }}</span>
          </div>
          <div class="bg-muted/10 border border-border/40 rounded-xl p-4 flex flex-col gap-1.5">
            <span class="text-[11px] text-muted-foreground uppercase font-bold tracking-wider"
              >Resource Name</span
            >
            <span class="text-sm font-semibold font-mono text-foreground/90 truncate">{{
              mcpMetadata.resourceName || "-"
            }}</span>
          </div>
        </div>

        <div class="grid grid-cols-1 gap-6 mt-4">
          <!-- Arguments -->
          <JsonViewer
            v-if="mcpMetadata.arguments && Object.keys(mcpMetadata.arguments).length > 0"
            :data="mcpMetadata.arguments"
            :label="t('logs.mcpArguments')"
            max-height="max-h-[300px]"
          />

          <!-- Result Summary -->
          <JsonViewer
            v-if="mcpMetadata.resultSummary && Object.keys(mcpMetadata.resultSummary).length > 0"
            :data="mcpMetadata.resultSummary"
            :label="t('logs.mcpResult')"
            max-height="max-h-[500px]"
          />
        </div>
      </div>
    </template>

    <!-- 2. WEB SEARCH LOG VIEW -->
    <template v-else-if="isWebSearchLog && webSearchMetadata">
      <div class="space-y-4">
        <div class="flex items-center gap-2 pb-2 border-b border-border/60">
          <div
            class="p-2.5 bg-muted/20 border border-border/40 rounded-lg shrink-0 group-hover:border-foreground/30 transition-colors"
          >
            <Globe class="size-5 text-foreground/80" />
          </div>
          <h3 class="font-semibold text-lg flex items-center gap-2">
            <span>{{ t("logs.searchQuery") }}</span>
          </h3>
          <Badge variant="secondary" class="font-mono text-xs">Web Search</Badge>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div class="bg-muted/10 border border-border/40 rounded-xl p-4 flex flex-col gap-1.5">
            <span class="text-[11px] text-muted-foreground uppercase font-bold tracking-wider"
              >Status</span
            >
            <Badge
              :variant="webSearchMetadata.status === 'success' ? 'default' : 'destructive'"
              class="self-start font-mono text-xs shadow-xs"
            >
              {{ webSearchMetadata.status || "-" }}
            </Badge>
          </div>
          <div class="bg-muted/10 border border-border/40 rounded-xl p-4 flex flex-col gap-1.5">
            <span class="text-[11px] text-muted-foreground uppercase font-bold tracking-wider"
              >Provider</span
            >
            <span class="text-sm font-semibold font-mono text-foreground/90 capitalize">{{
              webSearchMetadata.provider || "-"
            }}</span>
          </div>
          <div class="bg-muted/10 border border-border/40 rounded-xl p-4 flex flex-col gap-1.5">
            <span class="text-[11px] text-muted-foreground uppercase font-bold tracking-wider"
              >Result Count</span
            >
            <span class="text-sm font-semibold font-mono text-foreground/90">{{
              webSearchMetadata.resultCount ?? 0
            }}</span>
          </div>
        </div>

        <div
          v-if="webSearchMetadata.query"
          class="bg-muted/5 border border-border/40 rounded-xl p-4"
        >
          <span class="text-[11px] text-muted-foreground uppercase font-bold tracking-wider"
            >Query</span
          >
          <p class="font-mono text-sm font-semibold mt-1 text-foreground/90">
            {{ webSearchMetadata.query }}
          </p>
        </div>

        <!-- Usage Info -->
        <div
          v-if="webSearchMetadata.maxUses !== undefined"
          class="flex items-center gap-2 text-xs text-muted-foreground bg-muted/5 border border-border/30 px-3 py-2 rounded-lg self-start inline-flex"
        >
          <Clock class="size-3.5" />
          <span>{{ t("logs.webSearchProvider") }} usage:</span>
          <span class="font-mono font-bold"
            >{{ webSearchMetadata.currentUse ?? 0 }} / {{ webSearchMetadata.maxUses }}</span
          >
        </div>

        <!-- Search Results -->
        <JsonViewer
          v-if="webSearchMetadata.results && webSearchMetadata.results.length > 0"
          :data="webSearchMetadata.results"
          :label="t('logs.webSearchResults')"
          max-height="max-h-[500px]"
        />
      </div>
    </template>

    <!-- 3. STANDARD REQUEST/RESPONSE VIEW (chat, images, embeddings, audio) -->
    <template v-else>
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start io-column">
        <LogRequestView :log="log" />
        <LogResponseView :log="log" />
      </div>
    </template>
  </div>
</template>

<style scoped>
/* CSS containment for independent layout regions */
.io-column {
  contain: content;
}
</style>
