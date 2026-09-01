<script setup lang="ts">
/**
 * Response side of the Logs I/O view.
 *
 * Renders the parsed response as an ORDERED output timeline (reasoning,
 * text, tool calls, tool results, images in emission order) — the same
 * clarity as reading the raw stream, but structured and collapsible.
 * Response-level metadata (finish/stop reason, id, model, stream event
 * count) is surfaced as badges. Raw headers/body/stream live behind the
 * Formatted/Raw view-mode toggle (LogViewModeToggle), never mixed into
 * the formatted flow.
 */
import { Brackets, Code, Database, Gauge, Loader2, Terminal } from "@lucide/vue";
import { computed, nextTick, onMounted, ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import JsonViewer from "@/components/common/JsonViewer.vue";
import LogCollapsibleRow from "@/components/common/LogCollapsibleRow.vue";
import LogExpandControls from "@/components/common/LogExpandControls.vue";
import LogImagePreview from "@/components/common/LogImagePreview.vue";
import LogRawSection from "@/components/common/LogRawSection.vue";
import LogTextBlock from "@/components/common/LogTextBlock.vue";
import LogToolCallCard from "@/components/common/LogToolCallCard.vue";
import LogValueChip from "@/components/common/LogValueChip.vue";
import LogViewModeToggle from "@/components/common/LogViewModeToggle.vue";
import { Badge } from "@/components/ui/badge";
import type { LogRead } from "@/types/schemas";
import { formatTokens } from "@/utils/format";
import {
  MAX_INLINE_BYTES,
  firstLinePreview,
  formatBytes,
  formatCharCount,
  hasPayload,
  isOversizedPayload,
  parseResultOutput,
} from "@/utils/logFormat";
import {
  isResponsesStreamResponse,
  parseLogResponse,
  type ParsedResponse,
  type ResponseItem,
} from "@/utils/logResponseParser";
import { isStreamResponse } from "@/utils/sse";

const props = defineProps<{
  log: LogRead;
}>();

const { t } = useI18n();

// --- View mode ---------------------------------------------------------------
// Formatted (structured timeline) and Raw (headers + body/stream) are fully
// separate modes; raw never sits below the formatted content.
const viewMode = ref<"formatted" | "raw">("formatted");

// Switching logs always lands back on the formatted view.
watch(
  () => props.log.request_id,
  () => {
    viewMode.value = "formatted";
  }
);

// --- Response parsing ---------------------------------------------------------
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
  items: [],
  meta: {},
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

function resultFor(callId: string | undefined) {
  if (!callId) return undefined;
  return parsedResponse.value.toolResults.find(
    (r) => r.callId === callId || r.toolUseId === callId
  );
}

// --- Usage --------------------------------------------------------------------
const logUsage = computed(() => {
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

// --- Stats strip (usage + meta as compact tags) ------------------------------
interface StatsTag {
  label: string;
  value: string;
}

const statsTags = computed<StatsTag[]>(() => {
  const tags: StatsTag[] = [];
  const usage = logUsage.value;
  if (usage.has_usage) {
    tags.push(
      { label: t("logs.promptTokens"), value: formatTokens(usage.prompt_tokens) },
      { label: t("logs.completionTokens"), value: formatTokens(usage.completion_tokens) },
      { label: t("logs.totalTokens"), value: formatTokens(usage.total_tokens) }
    );
    if (usage.cached_prompt_tokens > 0)
      tags.push({ label: t("logs.cachedTokens"), value: formatTokens(usage.cached_prompt_tokens) });
    if (usage.cache_read_tokens > 0)
      tags.push({ label: t("logs.cacheReadTokens"), value: formatTokens(usage.cache_read_tokens) });
    if (usage.cache_creation_tokens > 0)
      tags.push({
        label: t("logs.cacheCreationTokens"),
        value: formatTokens(usage.cache_creation_tokens),
      });
  }

  const meta = parsedResponse.value.meta;
  if (meta.model) tags.push({ label: t("logs.model"), value: meta.model });
  if (meta.finishReason) tags.push({ label: t("logs.finishReason"), value: meta.finishReason });
  if (meta.stopReason) tags.push({ label: t("logs.stopReason"), value: meta.stopReason });
  if (meta.stopSequence) tags.push({ label: t("logs.stopSequence"), value: meta.stopSequence });
  if (meta.status && meta.status !== "completed")
    tags.push({ label: t("logs.status"), value: meta.status });
  if (meta.serviceTier) tags.push({ label: t("logs.serviceTier"), value: meta.serviceTier });
  if (meta.eventCount) tags.push({ label: t("logs.streamEvents"), value: String(meta.eventCount) });
  if (meta.id) tags.push({ label: t("logs.responseId"), value: meta.id });
  return tags;
});

// --- Ordered output items --------------------------------------------------------
// Collapse defaults: text blocks stay open unless very long; reasoning
// collapses sooner — it dominates long agentic traces.
const TEXT_COLLAPSE_THRESHOLD = 2000;
const REASONING_COLLAPSE_THRESHOLD = 800;

const itemExpanded = ref<Record<number, boolean>>({});

function defaultItemExpanded(item: ResponseItem): boolean {
  if (item.kind === "text") return item.text.length <= TEXT_COLLAPSE_THRESHOLD;
  if (item.kind === "reasoning") return item.text.length <= REASONING_COLLAPSE_THRESHOLD;
  return true;
}

function initItemState() {
  const next: Record<number, boolean> = {};
  parsedResponse.value.items.forEach((item, i) => {
    next[i] = defaultItemExpanded(item);
  });
  itemExpanded.value = next;
}

// (Re)init when the parsed response arrives/changes (deferred stream parse).
watch(() => parsedResponse.value, initItemState, { immediate: true });

function isItemExpanded(i: number): boolean {
  return itemExpanded.value[i] ?? true;
}

function toggleItem(i: number) {
  itemExpanded.value = { ...itemExpanded.value, [i]: !isItemExpanded(i) };
}

function setAllTextItems(expanded: boolean) {
  const next = { ...itemExpanded.value };
  parsedResponse.value.items.forEach((item, i) => {
    if (item.kind === "text" || item.kind === "reasoning") next[i] = expanded;
  });
  itemExpanded.value = next;
}

// --- Helpers --------------------------------------------------------------------

const hasResponseBody = computed(() => hasPayload(props.log?.response_body));
const hasResponseHeaders = computed(() => hasPayload(props.log?.response_headers));

// Raw body accordion: a multi-MB JSON *string* would freeze vue-json-pretty,
// so oversized strings render as (sliced) text instead.
const oversizeRawBodyText = computed((): string | null =>
  isOversizedPayload(props.log.response_body) ? props.log.response_body : null
);
// isResponseStream ⇒ the body is a string (checked inside that computed).
const streamBodyText = computed((): string =>
  isResponseStream.value && typeof props.log.response_body === "string"
    ? props.log.response_body
    : ""
);
const isToolResultOversized = (output: string): boolean => output.length > MAX_INLINE_BYTES;
</script>

<template>
  <div class="space-y-4">
    <div class="flex items-center justify-between gap-3 pb-2 border-b border-border/60">
      <h3 class="font-semibold text-base text-foreground flex items-center gap-2">
        <Terminal class="size-4 text-muted-foreground" />
        <span>{{ t("logs.response") }}</span>
      </h3>
      <div class="flex items-center gap-2.5">
        <Badge
          variant="outline"
          class="font-mono text-[11px] uppercase font-bold py-0.5 rounded-full border-border/80"
          :class="
            isResponseStream
              ? 'bg-action-blue/10 border-action-blue/20 text-action-blue'
              : 'bg-status-success/15 border-status-success/30 text-status-success'
          "
        >
          {{ isResponseStream ? t("logs.streamBadge") : t("logs.jsonBadge") }}
        </Badge>
        <LogViewModeToggle v-model="viewMode" />
      </div>
    </div>

    <template v-if="viewMode === 'formatted'">
      <!-- Stats strip: token usage + response meta as compact tags -->
      <div v-if="statsTags.length > 0" class="space-y-2">
        <div class="flex items-center gap-1.5">
          <Gauge class="size-3.5 text-muted-foreground/70" />
          <span class="text-xs text-muted-foreground font-bold uppercase tracking-wider">
            {{ t("logs.responseMeta") }}
          </span>
        </div>
        <div class="flex flex-wrap items-center gap-1.5">
          <LogValueChip
            v-for="tag in statsTags"
            :key="tag.label"
            :label="tag.label"
            :value="tag.value"
            :title="`${tag.label}: ${tag.value}`"
          />
        </div>
      </div>

      <!-- Stream parsing loading state -->
      <div
        v-if="isResponseStream && !shouldParseStream"
        class="flex items-center gap-2 p-4 rounded-lg border border-border/40 bg-muted/5 text-sm text-muted-foreground"
      >
        <Loader2 class="w-4 h-4 animate-spin shrink-0" />
        <span>{{ t("logs.parsingStream") }}</span>
      </div>

      <!-- Backend sampled out the full body (only a sentinel was stored) -->
      <div
        v-else-if="parsedResponse.isSampledOut"
        class="text-xs text-muted-foreground italic bg-muted/5 border border-border/40 p-4 rounded-lg text-center"
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
        <h4 class="text-xs uppercase font-bold text-muted-foreground flex items-center gap-2">
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
              <span class="text-[11px] font-mono text-muted-foreground uppercase tracking-wider">
                #{{ emb.index }}
              </span>
              <span class="text-[11px] font-mono text-muted-foreground/70">
                {{ t("logs.dimensions") }}: {{ emb.fullLength }}
              </span>
            </div>
            <div class="font-mono text-[11px] text-muted-foreground/80 break-all leading-relaxed">
              [{{ emb.vectorPreview.map((v) => v.toFixed(4)).join(", ")
              }}{{ emb.fullLength > emb.vectorPreview.length ? ", …" : "" }}]
            </div>
          </div>
        </div>
      </div>

      <!-- Audio raw bytes (speech / text-format transcription/translation) -->
      <div
        v-else-if="parsedResponse.protocol === 'audio-raw'"
        class="flex items-center gap-3 p-4 rounded-lg border border-border/40 bg-muted/5 text-sm"
      >
        <div class="p-2.5 rounded-lg bg-muted/20 border border-border/40 shrink-0">
          <Terminal class="size-5 text-foreground/80" />
        </div>
        <div class="flex flex-col gap-0.5">
          <span class="text-xs font-semibold text-foreground/90">{{ t("logs.audioOutput") }}</span>
          <span class="text-[11px] text-muted-foreground font-mono">
            {{ t("logs.audioRawSize") }}:
            {{ formatBytes(parsedResponse.audioRaw?.size ?? 0) || "—" }}
          </span>
        </div>
      </div>

      <!-- Audio text (transcription / translation json family) -->
      <div v-else-if="parsedResponse.protocol === 'audio-text'" class="space-y-2">
        <h4 class="text-xs uppercase font-bold text-muted-foreground flex items-center gap-2">
          <Terminal class="size-3.5" />
          {{ t("logs.transcription") }}
        </h4>
        <div class="code-container p-4 max-h-125 shadow-xs">
          <LogTextBlock
            :text="parsedResponse.audioText || parsedResponse.content"
            class="text-xs font-mono text-foreground/90 scrollbar-thin"
          />
        </div>
      </div>

      <!-- Ordered output timeline (chat / responses / anthropic / streams) -->
      <div v-else-if="parsedResponse.hasData && parsedResponse.items.length > 0" class="space-y-2">
        <div class="flex items-center gap-2">
          <h4 class="text-xs font-bold text-muted-foreground uppercase tracking-wider">
            {{ t("logs.outputItems") }}
          </h4>
          <span class="text-[11px] font-mono text-muted-foreground/60 tabular-nums">
            ({{ parsedResponse.items.length }})
          </span>
          <LogExpandControls @expand="setAllTextItems(true)" @collapse="setAllTextItems(false)" />
        </div>

        <div class="divide-y divide-border/30 border-y border-border/40">
          <template v-for="(item, i) in parsedResponse.items" :key="i">
            <!-- Reasoning block -->
            <LogCollapsibleRow
              v-if="item.kind === 'reasoning'"
              :open="isItemExpanded(i)"
              card-class="bg-transparent"
              button-class="w-full flex items-center gap-2 px-3 py-2.5 text-left hover:bg-muted/20 transition-colors cursor-pointer"
              chevron-class="text-muted-foreground/60"
              content-class="px-3 pb-3 pt-0.5 border-t border-border/25"
              @toggle="toggleItem(i)"
            >
              <template #header>
                <span class="text-[11px] font-bold text-action-violet uppercase tracking-wider">
                  {{ t("logs.thinkingContent") }}
                </span>
                <span
                  v-if="!isItemExpanded(i) && item.text"
                  class="text-[11px] font-mono text-muted-foreground/60 truncate min-w-0 flex-1"
                >
                  {{ firstLinePreview(item.text) }}
                </span>
                <span v-else class="flex-1" />
                <span class="text-[11px] font-mono text-muted-foreground/60 shrink-0 tabular-nums">
                  {{ formatCharCount(item.text.length) }}
                </span>
              </template>
              <div v-if="item.redacted" class="text-[11px] font-mono text-action-violet/70 italic">
                {{ t("logs.redactedContent") }}
              </div>
              <LogTextBlock
                v-else
                :text="item.text"
                class="text-xs font-mono text-foreground/90 max-h-96 overflow-y-auto scrollbar-thin"
              />
            </LogCollapsibleRow>

            <!-- Text block -->
            <LogCollapsibleRow
              v-else-if="item.kind === 'text'"
              :open="isItemExpanded(i)"
              card-class="bg-transparent"
              button-class="w-full flex items-center gap-2 px-3 py-2.5 text-left hover:bg-muted/20 transition-colors cursor-pointer"
              content-class="px-3 pb-3 pt-0.5 border-t border-border/25"
              @toggle="toggleItem(i)"
            >
              <template #header>
                <span class="text-[11px] font-bold text-muted-foreground uppercase tracking-wider">
                  {{ t("logs.textContent") }}
                </span>
                <span
                  v-if="!isItemExpanded(i)"
                  class="text-[11px] font-mono text-muted-foreground/60 truncate min-w-0 flex-1"
                >
                  {{ firstLinePreview(item.text) }}
                </span>
                <span v-else class="flex-1" />
                <span class="text-[11px] font-mono text-muted-foreground/60 shrink-0 tabular-nums">
                  {{ formatCharCount(item.text.length) }}
                </span>
              </template>
              <LogTextBlock
                :text="item.text"
                class="text-xs font-mono text-foreground/90 max-h-[500px] overflow-y-auto scrollbar-thin"
              />
            </LogCollapsibleRow>

            <!-- Tool call (with matched result) -->
            <LogToolCallCard
              v-else-if="item.kind === 'tool_call'"
              :call="item.call"
              :result="resultFor(item.call.id)"
              :index="i"
              bare
            />

            <!-- Tool result -->
            <div v-else-if="item.kind === 'tool_result'" class="p-3">
              <div class="flex items-center gap-2 mb-1.5 min-w-0">
                <span
                  class="text-[11px] font-bold text-action-blue uppercase tracking-wider shrink-0"
                >
                  {{ t("logs.toolResult") }}
                </span>
                <span class="text-[11px] font-mono text-muted-foreground/60 truncate min-w-0">
                  {{ item.result.toolUseId || item.result.callId || "#" + (i + 1) }}
                </span>
              </div>
              <!-- Oversized outputs are sliced as text (see hasData timeline above) -->
              <LogTextBlock
                v-if="isToolResultOversized(item.result.output)"
                :text="item.result.output"
                class="text-xs font-mono text-foreground/90 max-h-72 overflow-y-auto scrollbar-thin"
              />
              <JsonViewer
                v-else
                :data="parseResultOutput(item.result.output)"
                :deep="2"
                max-height="max-h-72"
                flat
              />
            </div>

            <!-- Image block -->
            <LogImagePreview v-else-if="item.kind === 'image'" :images="[item.image]" />
          </template>
        </div>
      </div>

      <!-- Fallback: parsed data without ordered items (defensive) -->
      <div v-else-if="parsedResponse.hasData" class="space-y-4">
        <div v-if="parsedResponse.reasoning" class="space-y-2">
          <h4 class="text-xs uppercase font-bold text-action-violet flex items-center gap-2">
            {{ t("logs.thinkingContent") }}
          </h4>
          <div class="code-container p-4 max-h-75 shadow-xs">
            <LogTextBlock
              :text="parsedResponse.reasoning"
              class="text-xs font-mono text-foreground/90 scrollbar-thin"
            />
          </div>
        </div>

        <LogImagePreview
          v-if="parsedResponse.images.length > 0 && !isImageResponse"
          :images="parsedResponse.images"
        />

        <div v-if="toolCallsWithResults.length > 0" class="space-y-2.5">
          <h4 class="text-xs uppercase font-bold text-action-amber flex items-center gap-2">
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

        <div v-if="parsedResponse.content" class="space-y-2">
          <h4 class="text-xs uppercase font-bold text-muted-foreground flex items-center gap-2">
            <Code class="size-3.5" />
            {{ t("logs.reconstructedContent") }}
          </h4>
          <div class="code-container p-4 max-h-125 shadow-xs">
            <LogTextBlock
              :text="parsedResponse.content"
              class="text-xs font-mono text-foreground/90 scrollbar-thin"
            />
          </div>
        </div>
      </div>

      <!-- Fallback: unparseable / empty response body -->
      <div v-else class="space-y-2">
        <h4 class="text-xs uppercase font-bold text-muted-foreground flex items-center gap-2">
          <Brackets class="size-3.5" />
          {{ t("logs.responseBody") }}
        </h4>
        <div
          class="text-xs text-muted-foreground italic bg-muted/5 border border-border/40 p-4 rounded-lg text-center"
        >
          {{ t("logs.unparseableResponse") }}
        </div>
      </div>
    </template>

    <!-- Raw mode: headers + body/stream shown directly, no accordions -->
    <template v-else>
      <LogRawSection
        :icon="Database"
        :title="t('logs.responseHeaders')"
        :has-content="hasResponseHeaders"
      >
        <JsonViewer :data="log.response_headers" maxHeight="max-h-72" flat />
        <template #empty>{{ t("logs.noHeaders") }}</template>
      </LogRawSection>

      <LogRawSection
        :icon="Code"
        :title="isResponseStream ? t('logs.rawStream') : t('logs.responseBody')"
        :has-content="hasResponseBody"
        :padded="!(streamBodyText || oversizeRawBodyText || isImageResponse)"
      >
        <!-- Streams render as raw text (never a JSON tree) -->
        <LogTextBlock
          v-if="streamBodyText"
          :text="streamBodyText"
          class="text-[11px] font-mono text-foreground/80 max-h-[500px] overflow-y-auto p-3 scrollbar-thin"
        />
        <!-- Image responses: never feed base64 into the JSON viewer (freeze) -->
        <div v-else-if="isImageResponse" class="text-xs text-muted-foreground p-3 space-y-1">
          <p>{{ t("logs.imageRawOmitted", { count: parsedResponse.images.length }) }}</p>
          <p class="font-mono text-[11px] text-muted-foreground/70">
            {{ t("logs.viewImagesAbove") }}
          </p>
        </div>
        <!-- Oversized JSON strings render as sliced text instead of a full tree -->
        <LogTextBlock
          v-else-if="oversizeRawBodyText"
          :text="oversizeRawBodyText"
          class="text-[11px] font-mono text-foreground/80 max-h-[500px] overflow-y-auto p-3 scrollbar-thin"
        />
        <JsonViewer v-else :data="log.response_body" maxHeight="max-h-[500px]" flat />
        <template #empty>{{ t("logs.noResponseBody") }}</template>
      </LogRawSection>
    </template>
  </div>
</template>
