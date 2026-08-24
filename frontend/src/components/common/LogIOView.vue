<script setup lang="ts">
import { computed, ref, onMounted, nextTick } from "vue";
import { useI18n } from "vue-i18n";
import JsonViewer from "@/components/common/JsonViewer.vue";
import LogImagePreview from "@/components/common/LogImagePreview.vue";
import LogMessageItem from "@/components/common/LogMessageItem.vue";
import LogToolCallCard from "@/components/common/LogToolCallCard.vue";
import { Badge } from "@/components/ui/badge";
import type { LogRead } from "@/types/schemas";
import { isStreamResponse } from "@/utils/sse";
import { formatTokens } from "@/utils/format";
import { formatMessageContent } from "@/utils/logFormat";
import {
  isResponsesStreamResponse,
  parseLogResponse,
  type ParsedResponse,
} from "@/utils/logResponseParser";
import {
  Brain,
  Brackets,
  ChevronDown,
  Clock,
  Code,
  Coins,
  Database,
  Globe,
  Loader2,
  Maximize2,
  Server,
  Settings,
  Shield,
  Terminal,
  Wrench,
} from "@lucide/vue";

const props = defineProps<{
  log: LogRead;
}>();

const { t } = useI18n();

interface ParsedMessage {
  role: string;
  content: unknown;
  tool_calls?: Array<{ id?: string; function?: { name?: string; arguments?: string } }>;
}

// Collapsible raw state toggles
const showRawRequestHeaders = ref(false);
const showRawRequestBody = ref(false);
const showRawResponseHeaders = ref(false);
const showRawResponseBody = ref(false);

// Safe-parse request/response bodies if they arrive as strings
const parsedRequestBody = computed(() => {
  const body = props.log.request_body;
  if (typeof body === "string") {
    try {
      return JSON.parse(body);
    } catch {
      return null;
    }
  }
  return body;
});

const parsedResponseBody = computed(() => {
  const body = props.log.response_body;
  if (typeof body === "string") {
    try {
      return JSON.parse(body);
    } catch {
      return null;
    }
  }
  return body;
});

// 10MB max before skipping expensive stream parsing
const STREAM_SIZE_LIMIT = 10_000_000;

// Inbound request type from metadata (chat | image_generation | image_edit |
// embedding | speech | transcription | translation | ...). Disambiguates shapes
// that share fields (e.g. audio text vs chat both can carry a `text` field).
const requestType = computed<string>(() => {
  const rt = props.log?.log_metadata?.request_type;
  return typeof rt === "string" ? rt : "chat";
});

// Cheap stream detection across all supported SSE dialects (OpenAI Chat,
// Anthropic, OpenAI Responses). Only inspects leading characters.
const isResponseStream = computed(() => {
  const body = props.log.response_body;
  if (typeof body !== "string") return false;
  if (body.length > STREAM_SIZE_LIMIT) return false;
  return isStreamResponse(body) || isResponsesStreamResponse(body);
});

// Deferred expensive stream parsing — runs after mount to avoid blocking the
// initial render of the details sheet.
const shouldParseStream = ref(false);

onMounted(() => {
  nextTick(() => {
    shouldParseStream.value = true;
  });
});

const EMPTY_PARSED: ParsedResponse = {
  protocol: "unknown",
  content: "",
  reasoning: "",
  toolCalls: [],
  toolResults: [],
  images: [],
  hasData: false,
};

// Unified, protocol-agnostic parse of the response body. Non-streaming bodies
// are parsed synchronously (cheap). Streaming bodies wait for shouldParseStream
// so the sheet renders first and the parse runs off the critical path.
const parsedResponse = computed<ParsedResponse>(() => {
  if (isResponseStream.value) {
    if (!shouldParseStream.value) return EMPTY_PARSED;
    return parseLogResponse(props.log.response_body, requestType.value);
  }
  const parsed = parsedResponseBody.value;
  if (parsed === null) return EMPTY_PARSED;
  return parseLogResponse(parsed, requestType.value);
});

// Whether the response is an image-generation / image-edit payload.
const isImageResponse = computed(() => parsedResponse.value.protocol === "image");

// Match tool calls with their results (by id / tool_use_id) for paired display.
const toolCallsWithResults = computed(() =>
  parsedResponse.value.toolCalls.map((call) => {
    const result = parsedResponse.value.toolResults.find(
      (r) => r.callId === call.id || r.toolUseId === call.id
    );
    return { call, result };
  })
);

// Image request detection (guards base64 in request bodies).
const isImageRequest = computed(
  () => requestType.value === "image_generation" || requestType.value === "image_edit"
);

// Sanitize a body for image requests: replaces oversized string values (base64
// image/mask data) with a size placeholder so the JSON viewer never lays out
// megabytes of text and freeze the page.
function sanitizeLargeStrings(data: unknown, maxLen = 200): unknown {
  if (typeof data === "string") {
    return data.length > maxLen ? `[truncated, ${data.length} chars]` : data;
  }
  if (Array.isArray(data)) return data.map((d) => sanitizeLargeStrings(d, maxLen));
  if (typeof data === "object" && data !== null) {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(data)) out[k] = sanitizeLargeStrings(v, maxLen);
    return out;
  }
  return data;
}

const safeRequestBody = computed(() =>
  isImageRequest.value ? sanitizeLargeStrings(props.log.request_body) : props.log.request_body
);

// Parse a tool result output string as JSON for structured display.
function parseResultOutput(output: string): unknown {
  try {
    return JSON.parse(output);
  } catch {
    return output;
  }
}

// Human-readable byte size for audio / raw-byte markers.
function formatBytesLabel(bytes: number): string {
  if (bytes <= 0) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

// Log type detection
const isMcpLog = computed(() => props.log?.log_type === "mcp");
const isWebSearchLog = computed(() => props.log?.log_type === "web_search");
const isAuditLog = computed(() => props.log?.log_type === "audit");

const hasRequestBody = computed(() => {
  const body = props.log?.request_body;
  if (!body) return false;
  if (typeof body === "object") return Object.keys(body).length > 0;
  if (typeof body === "string") return body.trim().length > 0;
  return true;
});

const hasResponseBody = computed(() => {
  const body = props.log?.response_body;
  if (!body) return false;
  if (typeof body === "object") return Object.keys(body).length > 0;
  if (typeof body === "string") return body.trim().length > 0;
  return true;
});

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

const logUsage = computed(() => {
  if (isAuditLog.value || isMcpLog.value || isWebSearchLog.value) return null;

  const prompt =
    props.log.prompt_tokens ?? (props.log.log_metadata?.prompt_tokens as number | undefined) ?? 0;
  const completion =
    props.log.completion_tokens ??
    (props.log.log_metadata?.completion_tokens as number | undefined) ??
    0;
  const total =
    props.log.total_tokens ?? (props.log.log_metadata?.total_tokens as number | undefined) ?? 0;

  const cacheRead =
    props.log.cache_read_input_tokens ??
    (props.log.log_metadata?.cache_read_input_tokens as number | undefined) ??
    0;
  const cachedPrompt =
    props.log.cached_prompt_tokens ??
    (props.log.log_metadata?.cached_prompt_tokens as number | undefined) ??
    0;
  const cacheCreation =
    props.log.cache_creation_input_tokens ??
    (props.log.log_metadata?.cache_creation_input_tokens as number | undefined) ??
    0;

  return {
    prompt_tokens: prompt,
    completion_tokens: completion,
    total_tokens: total,
    cache_read_tokens: cacheRead,
    cached_prompt_tokens: cachedPrompt,
    cache_creation_tokens: cacheCreation,
    has_usage: prompt > 0 || completion > 0 || total > 0,
  };
});

// Request parsing
const parsedRequest = computed(() => {
  const body = parsedRequestBody.value;
  if (typeof body !== "object" || body === null) return null;
  const b = body as Record<string, unknown>;

  // Try to extract messages
  let messages: ParsedMessage[] = [];
  if (Array.isArray(b.messages)) {
    messages = b.messages as ParsedMessage[];
  }

  // Try to extract system prompt (Anthropic top-level system parameter, or first OpenAI system message)
  let systemPrompt = (b.system as string) || "";
  let displayMessages = messages;
  const firstMsg = messages[0];
  if (!systemPrompt && firstMsg && firstMsg.role === "system") {
    systemPrompt = formatMessageContent(firstMsg.content);
    displayMessages = messages.slice(1);
  }

  return {
    messages: displayMessages,
    systemPrompt,
    model: b.model || props.log.model || "",
    temperature: b.temperature,
    maxTokens: b.max_tokens ?? b.max_completion_tokens,
    stream: b.stream,
    topP: b.top_p,
    hasParams:
      b.model !== undefined ||
      b.temperature !== undefined ||
      b.max_tokens !== undefined ||
      b.stream !== undefined,
  };
});

// Image-generation / image-edit request parameters (prompt, n, size, quality,
// response_format, etc.) — shown instead of the generic "non-standard payload"
// note for image requests.
const imageRequestParams = computed(() => {
  if (!isImageRequest.value) return null;
  const body = parsedRequestBody.value;
  if (typeof body !== "object" || body === null || Array.isArray(body)) return null;
  const b = body as Record<string, unknown>;
  return {
    prompt: typeof b.prompt === "string" ? (b.prompt as string) : undefined,
    model: b.model,
    n: b.n,
    size: b.size,
    quality: b.quality,
    style: b.style,
    responseFormat: b.response_format,
    outputFormat: b.output_format,
    background: b.background,
  };
});
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
            <div
              class="border border-border/40 rounded-xl overflow-hidden bg-muted/5 transition-all"
            >
              <button
                @click="showRawRequestHeaders = !showRawRequestHeaders"
                :aria-expanded="showRawRequestHeaders"
                class="w-full flex items-center justify-between p-3.5 text-xs font-semibold text-muted-foreground hover:bg-muted/10 transition-colors uppercase tracking-wider cursor-pointer max-sm:min-h-11"
              >
                <span class="flex items-center gap-2">
                  <Database class="size-3.5" />
                  {{ t("logs.requestHeaders") }}
                </span>
                <ChevronDown
                  class="size-4 transition-transform duration-200 text-muted-foreground/60"
                  :class="{ 'rotate-180': showRawRequestHeaders }"
                />
              </button>
              <div v-if="showRawRequestHeaders" class="p-3 border-t border-border/40 bg-card">
                <JsonViewer :data="log.request_headers" maxHeight="max-h-60" />
              </div>
            </div>

            <!-- Request Body View -->
            <div class="space-y-2">
              <h4
                class="text-xs uppercase font-bold text-muted-foreground flex items-center gap-2 pl-1"
              >
                <Code class="size-3.5" />
                {{ t("logs.requestBody") }}
              </h4>
              <div
                v-if="hasRequestBody"
                class="border border-border/40 rounded-xl overflow-hidden bg-card"
              >
                <JsonViewer :data="log.request_body" maxHeight="max-h-[500px]" :deep="3" />
              </div>
              <div
                v-else
                class="text-xs text-muted-foreground italic bg-muted/5 border border-border/40 p-4 rounded-xl text-center shadow-xs"
              >
                No request body payload
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
            <div
              class="border border-border/40 rounded-xl overflow-hidden bg-muted/5 transition-all"
            >
              <button
                @click="showRawResponseHeaders = !showRawResponseHeaders"
                :aria-expanded="showRawResponseHeaders"
                class="w-full flex items-center justify-between p-3.5 text-xs font-semibold text-muted-foreground hover:bg-muted/10 transition-colors uppercase tracking-wider cursor-pointer max-sm:min-h-11"
              >
                <span class="flex items-center gap-2">
                  <Database class="size-3.5" />
                  {{ t("logs.responseHeaders") }}
                </span>
                <ChevronDown
                  class="size-4 transition-transform duration-200 text-muted-foreground/60"
                  :class="{ 'rotate-180': showRawResponseHeaders }"
                />
              </button>
              <div v-if="showRawResponseHeaders" class="p-3 border-t border-border/40 bg-card">
                <JsonViewer :data="log.response_headers" maxHeight="max-h-60" />
              </div>
            </div>

            <!-- Response Body View -->
            <div class="space-y-2">
              <h4
                class="text-xs uppercase font-bold text-muted-foreground flex items-center gap-2 pl-1"
              >
                <Brackets class="size-3.5" />
                {{ t("logs.responseBody") }}
              </h4>
              <div
                v-if="hasResponseBody"
                class="border border-border/40 rounded-xl overflow-hidden bg-card"
              >
                <JsonViewer :data="log.response_body" maxHeight="max-h-[500px]" :deep="3" />
              </div>
              <div
                v-else
                class="text-xs text-muted-foreground italic bg-muted/5 border border-border/40 p-4 rounded-xl text-center shadow-xs"
              >
                No response body payload
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

    <!-- 3. STANDARD REQUEST/RESPONSE VIEW -->
    <template v-else>
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start io-column">
        <!-- A. Request Column -->
        <div class="space-y-4">
          <div class="flex items-center justify-between pb-2 border-b border-border/60">
            <h3 class="font-semibold text-base text-foreground flex items-center gap-2">
              <Database class="size-4 text-muted-foreground" />
              <span>{{ t("logs.request") }}</span>
            </h3>
          </div>

          <!-- Request parameters card -->
          <div
            v-if="parsedRequest && parsedRequest.hasParams"
            class="bg-muted/10 border border-border/40 rounded-xl p-3.5 space-y-2.5 shadow-xs"
          >
            <div
              class="flex items-center justify-between text-xs text-muted-foreground font-bold uppercase tracking-wider pb-1.5 border-b border-border/20"
            >
              <span>{{ t("logs.parameters") }}</span>
              <Settings class="size-3.5 text-muted-foreground/60" />
            </div>
            <div class="grid grid-cols-2 sm:grid-cols-3 gap-2.5 pt-1">
              <div v-if="parsedRequest.model" class="flex flex-col gap-0.5">
                <span class="text-[11px] text-muted-foreground uppercase font-medium">{{
                  t("logs.model")
                }}</span>
                <span class="text-xs font-semibold font-mono truncate text-foreground/80">{{
                  parsedRequest.model
                }}</span>
              </div>
              <div v-if="parsedRequest.temperature !== undefined" class="flex flex-col gap-0.5">
                <span class="text-[11px] text-muted-foreground uppercase font-medium">{{
                  t("logs.temperature")
                }}</span>
                <span class="text-xs font-semibold font-mono text-foreground/80">{{
                  parsedRequest.temperature
                }}</span>
              </div>
              <div v-if="parsedRequest.maxTokens !== undefined" class="flex flex-col gap-0.5">
                <span class="text-[11px] text-muted-foreground uppercase font-medium">{{
                  t("logs.maxTokens")
                }}</span>
                <span class="text-xs font-semibold font-mono text-foreground/80">{{
                  parsedRequest.maxTokens
                }}</span>
              </div>
              <div v-if="parsedRequest.stream !== undefined" class="flex flex-col gap-0.5">
                <span class="text-[11px] text-muted-foreground uppercase font-medium">{{
                  t("logs.stream")
                }}</span>
                <span class="text-xs font-semibold font-mono text-foreground/80">{{
                  parsedRequest.stream ? "true" : "false"
                }}</span>
              </div>
            </div>
          </div>

          <!-- System prompt card if any -->
          <div
            v-if="parsedRequest && parsedRequest.systemPrompt"
            class="bg-action-amber/5 border border-action-amber/20 rounded-xl p-3.5 space-y-2 shadow-xs"
          >
            <div
              class="flex items-center gap-1.5 text-xs text-action-amber font-bold uppercase tracking-wider"
            >
              <Shield class="size-3.5" />
              <span>{{ t("logs.systemInstructions") }}</span>
            </div>
            <div
              class="text-xs font-mono text-foreground/90 whitespace-pre-wrap leading-relaxed max-h-48 overflow-y-auto bg-card/50 p-2.5 rounded border border-action-amber/10 scrollbar-thin break-all"
            >
              {{ parsedRequest.systemPrompt }}
            </div>
          </div>

          <!-- Messages conversation history (Scrollable if very long) -->
          <div
            v-if="parsedRequest && parsedRequest.messages && parsedRequest.messages.length > 0"
            class="space-y-3"
          >
            <h4 class="text-xs font-bold text-muted-foreground uppercase tracking-wider pl-1">
              {{ t("logs.conversationMessages") }}
            </h4>
            <div
              class="space-y-3 max-h-[480px] overflow-y-auto pr-2 scrollbar-thin"
              style="contain: content"
            >
              <LogMessageItem
                v-for="(msg, index) in parsedRequest.messages"
                :key="index"
                :msg="msg"
              />
            </div>
          </div>

          <!-- Image request parameters (prompt / n / size / quality / ...) -->
          <div
            v-if="imageRequestParams"
            class="bg-action-blue/5 border border-action-blue/20 rounded-xl p-3.5 space-y-3 shadow-xs"
          >
            <div
              class="flex items-center gap-1.5 text-xs text-action-blue font-bold uppercase tracking-wider"
            >
              <Maximize2 class="size-3.5" />
              <span>{{ t("logs.imageRequestParams") }}</span>
            </div>
            <!-- Prompt -->
            <div v-if="imageRequestParams.prompt" class="space-y-1">
              <span class="text-[11px] text-muted-foreground uppercase font-medium">
                {{ t("logs.prompt") }}
              </span>
              <p
                class="text-xs font-mono text-foreground/90 whitespace-pre-wrap break-all leading-relaxed"
              >
                {{ imageRequestParams.prompt }}
              </p>
            </div>
            <!-- Param grid -->
            <div class="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
              <div v-if="imageRequestParams.model" class="flex flex-col gap-0.5">
                <span class="text-[11px] text-muted-foreground uppercase font-medium">{{
                  t("logs.model")
                }}</span>
                <span class="text-xs font-semibold font-mono truncate text-foreground/80">{{
                  imageRequestParams.model
                }}</span>
              </div>
              <div v-if="imageRequestParams.size" class="flex flex-col gap-0.5">
                <span class="text-[11px] text-muted-foreground uppercase font-medium">Size</span>
                <span class="text-xs font-semibold text-foreground/80">{{
                  imageRequestParams.size
                }}</span>
              </div>
              <div v-if="imageRequestParams.n !== undefined" class="flex flex-col gap-0.5">
                <span class="text-[11px] text-muted-foreground uppercase font-medium">N</span>
                <span class="text-xs font-semibold text-foreground/80">{{
                  imageRequestParams.n
                }}</span>
              </div>
              <div v-if="imageRequestParams.quality" class="flex flex-col gap-0.5">
                <span class="text-[11px] text-muted-foreground uppercase font-medium">Quality</span>
                <span class="text-xs font-semibold text-foreground/80 capitalize">{{
                  imageRequestParams.quality
                }}</span>
              </div>
              <div v-if="imageRequestParams.style" class="flex flex-col gap-0.5">
                <span class="text-[11px] text-muted-foreground uppercase font-medium">Style</span>
                <span class="text-xs font-semibold text-foreground/80">{{
                  imageRequestParams.style
                }}</span>
              </div>
              <div v-if="imageRequestParams.responseFormat" class="flex flex-col gap-0.5">
                <span class="text-[11px] text-muted-foreground uppercase font-medium">Format</span>
                <span class="text-xs font-semibold text-foreground/80">{{
                  imageRequestParams.responseFormat
                }}</span>
              </div>
            </div>
          </div>

          <!-- Non-standard request fallback -->
          <div
            v-else-if="!parsedRequest || parsedRequest.messages.length === 0"
            class="text-xs text-muted-foreground italic bg-muted/5 border border-border/40 p-4 rounded-xl text-center shadow-xs"
          >
            {{ t("logs.nonStandardPayload") }}
          </div>

          <!-- Raw Accordions -->
          <div class="space-y-3 pt-2">
            <!-- Headers -->
            <div
              class="border border-border/40 rounded-xl overflow-hidden bg-muted/5 transition-all"
            >
              <button
                @click="showRawRequestHeaders = !showRawRequestHeaders"
                :aria-expanded="showRawRequestHeaders"
                aria-controls="panel-raw-req-headers"
                class="w-full flex items-center justify-between p-3.5 text-xs font-semibold text-muted-foreground hover:bg-muted/10 transition-colors uppercase tracking-wider cursor-pointer max-sm:min-h-11"
              >
                <span class="flex items-center gap-2">
                  <Database class="size-3.5" />
                  {{ t("logs.requestHeaders") }}
                </span>
                <ChevronDown
                  class="size-4 transition-transform duration-200 text-muted-foreground/60"
                  :class="{ 'rotate-180': showRawRequestHeaders }"
                />
              </button>
              <div
                v-if="showRawRequestHeaders"
                id="panel-raw-req-headers"
                class="p-3 border-t border-border/40 bg-card"
              >
                <JsonViewer :data="log.request_headers" maxHeight="max-h-48" />
              </div>
            </div>

            <!-- Body -->
            <div
              class="border border-border/40 rounded-xl overflow-hidden bg-muted/5 transition-all"
            >
              <button
                @click="showRawRequestBody = !showRawRequestBody"
                :aria-expanded="showRawRequestBody"
                aria-controls="panel-raw-req-body"
                class="w-full flex items-center justify-between p-3.5 text-xs font-semibold text-muted-foreground hover:bg-muted/10 transition-colors uppercase tracking-wider cursor-pointer max-sm:min-h-11"
              >
                <span class="flex items-center gap-2">
                  <Code class="size-3.5" />
                  {{ t("logs.requestBody") }}
                </span>
                <ChevronDown
                  class="size-4 transition-transform duration-200 text-muted-foreground/60"
                  :class="{ 'rotate-180': showRawRequestBody }"
                />
              </button>
              <div
                v-if="showRawRequestBody"
                id="panel-raw-req-body"
                class="p-3 border-t border-border/40 bg-card"
              >
                <JsonViewer :data="safeRequestBody" maxHeight="max-h-96" />
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
            <Badge
              variant="outline"
              class="font-mono text-[11px] uppercase font-bold py-0.5 border-border/80 shadow-xs"
              :class="
                isResponseStream
                  ? 'bg-action-blue/10 border-action-blue/20 text-action-blue'
                  : 'bg-status-success/15 border-status-success/30 text-status-success'
              "
            >
              {{ isResponseStream ? t("logs.streamBadge") : t("logs.jsonBadge") }}
            </Badge>
          </div>

          <!-- RESPONSE METRICS (Tokens & Usage) -->
          <div
            v-if="logUsage && logUsage.has_usage"
            class="bg-muted/10 border border-border/40 rounded-xl p-3.5 space-y-2.5 shadow-xs"
          >
            <div
              class="flex items-center justify-between text-xs text-muted-foreground font-bold uppercase tracking-wider pb-1.5 border-b border-border/20"
            >
              <span>{{ t("logs.tokenUsage") }}</span>
              <Coins class="size-3.5 text-muted-foreground/60" />
            </div>
            <div class="grid grid-cols-3 gap-2.5 pt-1 text-center font-mono">
              <div class="flex flex-col bg-card/65 p-2 rounded-lg border border-border/10">
                <span class="text-[11px] text-muted-foreground uppercase font-semibold">{{
                  t("logs.promptTokens")
                }}</span>
                <span class="text-xs font-bold text-foreground/80 mt-0.5">{{
                  formatTokens(logUsage.prompt_tokens)
                }}</span>
              </div>
              <div class="flex flex-col bg-card/65 p-2 rounded-lg border border-border/10">
                <span class="text-[11px] text-muted-foreground uppercase font-semibold">{{
                  t("logs.completionTokens")
                }}</span>
                <span class="text-xs font-bold text-foreground/80 mt-0.5">{{
                  formatTokens(logUsage.completion_tokens)
                }}</span>
              </div>
              <div class="flex flex-col bg-card/65 p-2 rounded-lg border border-border/10">
                <span class="text-[11px] text-muted-foreground uppercase font-semibold">{{
                  t("logs.totalTokens")
                }}</span>
                <span class="text-xs font-bold text-foreground/80 mt-0.5">{{
                  formatTokens(logUsage.total_tokens)
                }}</span>
              </div>
            </div>
            <!-- Cache Details -->
            <div
              v-if="
                logUsage.cached_prompt_tokens > 0 ||
                logUsage.cache_read_tokens > 0 ||
                logUsage.cache_creation_tokens > 0
              "
              class="flex flex-wrap gap-x-4 gap-y-2 pt-2 border-t border-border/25 text-xs font-mono"
            >
              <div v-if="logUsage.cached_prompt_tokens > 0" class="flex items-center gap-1">
                <span class="text-muted-foreground text-[11px]">{{ t("logs.cachedTokens") }}:</span>
                <span class="font-bold text-action-blue">{{
                  formatTokens(logUsage.cached_prompt_tokens)
                }}</span>
              </div>
              <div v-if="logUsage.cache_read_tokens > 0" class="flex items-center gap-1">
                <span class="text-muted-foreground text-[11px]"
                  >{{ t("logs.cacheReadTokens") }}:</span
                >
                <span class="font-bold text-action-blue">{{
                  formatTokens(logUsage.cache_read_tokens)
                }}</span>
              </div>
              <div v-if="logUsage.cache_creation_tokens > 0" class="flex items-center gap-1">
                <span class="text-muted-foreground text-[11px]"
                  >{{ t("logs.cacheCreationTokens") }}:</span
                >
                <span class="font-bold text-muted-foreground">{{
                  formatTokens(logUsage.cache_creation_tokens)
                }}</span>
              </div>
            </div>
          </div>

          <!-- Stream parsing loading state -->
          <div
            v-if="isResponseStream && !shouldParseStream"
            class="flex items-center gap-2 p-4 rounded-xl border border-border/40 bg-muted/5 text-sm text-muted-foreground"
          >
            <Loader2 class="w-4 h-4 animate-spin shrink-0" />
            <span>{{ t("logs.parsingStream") }}</span>
          </div>

          <!-- Backend sampled out the full body (only a sentinel was stored) -->
          <div
            v-else-if="parsedResponse.isSampledOut"
            class="text-xs text-muted-foreground italic bg-muted/5 border border-border/40 p-4 rounded-xl text-center shadow-xs"
          >
            {{ t("logs.bodySampledOut") }}
          </div>

          <!-- Image generation / image edit: dedicated preview (avoids base64 freeze) -->
          <LogImagePreview
            v-else-if="isImageResponse"
            :images="parsedResponse.images"
            :output-format="parsedResponse.images[0]?.outputFormat || undefined"
          />

          <!-- Embeddings -->
          <div
            v-else-if="parsedResponse.protocol === 'embedding' && parsedResponse.embeddings"
            class="space-y-3"
          >
            <h4
              class="text-xs uppercase font-bold text-muted-foreground flex items-center gap-2 pl-1"
            >
              <Database class="size-3.5" />
              {{ t("logs.embeddings") }}
              <span class="text-muted-foreground/60 font-mono normal-case tracking-normal">
                ({{ parsedResponse.embeddings.length }})
              </span>
            </h4>
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div
                v-for="emb in parsedResponse.embeddings"
                :key="emb.index"
                class="bg-muted/10 border border-border/40 rounded-lg p-3 space-y-1.5"
              >
                <div class="flex items-center justify-between">
                  <span
                    class="text-[11px] font-mono text-muted-foreground uppercase tracking-wider"
                  >
                    #{{ emb.index }}
                  </span>
                  <span class="text-[11px] font-mono text-muted-foreground/70">
                    {{ t("logs.dimensions") }}: {{ emb.fullLength }}
                  </span>
                </div>
                <div
                  class="font-mono text-[11px] text-muted-foreground/80 break-all leading-relaxed"
                >
                  [{{ emb.vectorPreview.map((v) => v.toFixed(4)).join(", ")
                  }}{{ emb.fullLength > emb.vectorPreview.length ? ", …" : "" }}]
                </div>
              </div>
            </div>
          </div>

          <!-- Audio raw bytes (speech / text-format transcription/translation) -->
          <div
            v-else-if="parsedResponse.protocol === 'audio-raw'"
            class="flex items-center gap-3 p-4 rounded-xl border border-border/40 bg-muted/5 text-sm"
          >
            <div class="p-2.5 rounded-lg bg-muted/20 border border-border/40 shrink-0">
              <Terminal class="size-5 text-foreground/80" />
            </div>
            <div class="flex flex-col gap-0.5">
              <span class="text-xs font-semibold text-foreground/90">{{
                t("logs.audioOutput")
              }}</span>
              <span class="text-[11px] text-muted-foreground font-mono">
                {{ t("logs.audioRawSize") }}:
                {{ formatBytesLabel(parsedResponse.audioRaw?.size ?? 0) }}
              </span>
            </div>
          </div>

          <!-- Audio text (transcription / translation json family) -->
          <div v-else-if="parsedResponse.protocol === 'audio-text'" class="space-y-2">
            <h4
              class="text-xs uppercase font-bold text-muted-foreground flex items-center gap-2 pl-1"
            >
              <Terminal class="size-3.5" />
              {{ t("logs.transcription") }}
            </h4>
            <div class="code-container p-4 max-h-125 shadow-xs">
              <div
                class="text-xs font-mono whitespace-pre-wrap text-foreground/90 leading-relaxed scrollbar-thin break-all"
              >
                {{ parsedResponse.audioText || parsedResponse.content }}
              </div>
            </div>
          </div>

          <!-- Unified chat / responses / anthropic / stream view -->
          <div v-else-if="parsedResponse.hasData" class="space-y-4">
            <!-- Reasoning / Thinking -->
            <div v-if="parsedResponse.reasoning" class="space-y-2">
              <h4
                class="text-xs uppercase font-bold text-action-violet flex items-center gap-2 pl-1"
              >
                <Brain class="size-3.5" />
                {{ t("logs.thinkingContent") }}
              </h4>
              <div
                class="code-container bg-action-violet/5 border-action-violet/20 p-4 max-h-75 shadow-xs"
              >
                <div
                  class="text-xs font-mono whitespace-pre-wrap text-action-violet leading-relaxed scrollbar-thin break-all"
                >
                  {{ parsedResponse.reasoning }}
                </div>
              </div>
            </div>

            <!-- Images embedded in content (e.g. Anthropic image blocks) -->
            <LogImagePreview
              v-if="parsedResponse.images.length > 0 && !isImageResponse"
              :images="parsedResponse.images"
            />

            <!-- Tool Calls + matched results -->
            <div v-if="toolCallsWithResults.length > 0" class="space-y-2.5">
              <h4
                class="text-xs uppercase font-bold text-action-amber flex items-center gap-2 pl-1"
              >
                <Wrench class="size-3.5" />
                {{ t("chat.toolCalls") }}
                <span class="text-muted-foreground/60 font-mono normal-case tracking-normal">
                  ({{ toolCallsWithResults.length }})
                </span>
              </h4>
              <div class="space-y-2.5">
                <LogToolCallCard
                  v-for="(item, index) in toolCallsWithResults"
                  :key="item.call.id || index"
                  :call="item.call"
                  :result="item.result"
                  :index="index"
                />
              </div>
            </div>

            <!-- Orphan tool results (no matching call) -->
            <div
              v-if="parsedResponse.toolResults.length > 0 && toolCallsWithResults.length === 0"
              class="space-y-2.5"
            >
              <h4 class="text-xs uppercase font-bold text-action-blue flex items-center gap-2 pl-1">
                <Wrench class="size-3.5" />
                {{ t("logs.toolResults") }}
              </h4>
              <div class="space-y-2.5">
                <div
                  v-for="(res, index) in parsedResponse.toolResults"
                  :key="index"
                  class="p-3 bg-action-blue/5 rounded-lg border border-action-blue/20"
                >
                  <div class="text-[11px] font-mono text-muted-foreground mb-1">
                    {{ res.toolUseId || res.callId || "#" + (index + 1) }}
                  </div>
                  <JsonViewer
                    :data="parseResultOutput(res.output)"
                    :deep="2"
                    max-height="max-h-72"
                  />
                </div>
              </div>
            </div>

            <!-- Reconstructed text content -->
            <div v-if="parsedResponse.content" class="space-y-2">
              <h4
                class="text-xs uppercase font-bold text-muted-foreground flex items-center gap-2 pl-1"
              >
                <Code class="size-3.5" />
                {{ t("logs.reconstructedContent") }}
              </h4>
              <div class="code-container p-4 max-h-125 shadow-xs">
                <div
                  class="text-xs font-mono whitespace-pre-wrap text-foreground/90 leading-relaxed scrollbar-thin break-all"
                >
                  {{ parsedResponse.content }}
                </div>
              </div>
            </div>
          </div>

          <!-- Fallback: unparseable / empty response body -->
          <div v-else class="space-y-2">
            <h4
              class="text-xs uppercase font-bold text-muted-foreground flex items-center gap-2 pl-1"
            >
              <Brackets class="size-3.5" />
              {{ t("logs.responseBody") }}
            </h4>
            <div
              class="text-xs text-muted-foreground italic bg-muted/5 border border-border/40 p-4 rounded-xl text-center shadow-xs"
            >
              {{ t("logs.unparseableResponse") }}
            </div>
          </div>

          <!-- Response Raw Accordions (Unified Body/Stream block at the bottom) -->
          <div class="space-y-3 pt-2">
            <!-- Headers -->
            <div
              class="border border-border/40 rounded-xl overflow-hidden bg-muted/5 transition-all"
            >
              <button
                @click="showRawResponseHeaders = !showRawResponseHeaders"
                :aria-expanded="showRawResponseHeaders"
                aria-controls="panel-raw-resp-headers"
                class="w-full flex items-center justify-between p-3.5 text-xs font-semibold text-muted-foreground hover:bg-muted/10 transition-colors uppercase tracking-wider cursor-pointer"
              >
                <span class="flex items-center gap-2">
                  <Database class="size-3.5" />
                  {{ t("logs.responseHeaders") }}
                </span>
                <ChevronDown
                  class="size-4 transition-transform duration-200 text-muted-foreground/60"
                  :class="{ 'rotate-180': showRawResponseHeaders }"
                />
              </button>
              <div
                v-if="showRawResponseHeaders"
                id="panel-raw-resp-headers"
                class="p-3 border-t border-border/40 bg-card"
              >
                <JsonViewer :data="log.response_headers" maxHeight="max-h-48" />
              </div>
            </div>

            <!-- Body / Stream Accordion (Single entry point for raw response stream/body data) -->
            <div
              v-if="hasResponseBody"
              class="border border-border/40 rounded-xl overflow-hidden bg-muted/5 transition-all"
            >
              <button
                @click="showRawResponseBody = !showRawResponseBody"
                :aria-expanded="showRawResponseBody"
                aria-controls="panel-raw-resp-body"
                class="w-full flex items-center justify-between p-3.5 text-xs font-semibold text-muted-foreground hover:bg-muted/10 transition-colors uppercase tracking-wider cursor-pointer"
              >
                <span class="flex items-center gap-2">
                  <Code class="size-3.5" />
                  <span>{{ isResponseStream ? t("logs.rawStream") : t("logs.responseBody") }}</span>
                </span>
                <ChevronDown
                  class="size-4 transition-transform duration-200 text-muted-foreground/60"
                  :class="{ 'rotate-180': showRawResponseBody }"
                />
              </button>
              <div
                v-if="showRawResponseBody"
                id="panel-raw-resp-body"
                class="p-3 border-t border-border/40 bg-card"
              >
                <div
                  v-if="isResponseStream"
                  class="code-container bg-muted/30 p-3.5 max-h-96 overflow-auto border-0"
                >
                  <pre
                    class="text-[11px] font-mono text-foreground/80 whitespace-pre-wrap break-all scrollbar-thin"
                    >{{ log.response_body }}</pre>
                </div>
                <!-- Image responses: never feed base64 into the JSON viewer (freeze) -->
                <div
                  v-else-if="isImageResponse"
                  class="text-xs text-muted-foreground p-3 space-y-1"
                >
                  <p>{{ t("logs.imageRawOmitted", { count: parsedResponse.images.length }) }}</p>
                  <p class="font-mono text-[11px] text-muted-foreground/70">
                    {{ t("logs.viewImagesAbove") }}
                  </p>
                </div>
                <JsonViewer v-else :data="log.response_body" maxHeight="max-h-96" />
              </div>
            </div>
          </div>
        </div>
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
