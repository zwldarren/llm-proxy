<script setup lang="ts">
/**
 * Unified tool call / tool result card for the Logs I/O view.
 *
 * Renders one tool call (name, id, arguments) and optionally its result
 * (output) using a small JSON viewer for structured arguments/output. This
 * replaces the scattered per-protocol inline <pre> blocks and gives tool
 * calls/results a consistent, parseable presentation across OpenAI Chat,
 * OpenAI Responses, and Anthropic.
 *
 * Large argument/result payloads are size-guarded: anything beyond
 * MAX_INLINE_BYTES is collapsed by default and only expanded on demand, so
 * a multi-MB tool result never freezes the page.
 */
import { ArrowRight, ChevronDown, Search, Wrench } from "@lucide/vue";
import { computed, ref } from "vue";
import { useI18n } from "vue-i18n";
import JsonViewer from "@/components/common/JsonViewer.vue";
import { Badge } from "@/components/ui/badge";
import type { ToolCallInfo, ToolResultInfo } from "@/utils/logResponseParser";
import { parseToolArgs } from "@/utils/logFormat";

const props = withDefaults(
  defineProps<{
    call: ToolCallInfo;
    /** Matching result, if any (matched by call id / tool use id) */
    result?: ToolResultInfo;
    /** Sequence index for display */
    index?: number;
    /** Start expanded? Default: collapsed */
    defaultExpanded?: boolean;
  }>(),
  { defaultExpanded: false }
);

const { t } = useI18n();

const MAX_INLINE_BYTES = 256 * 1024; // 256 KB

const expanded = ref(props.defaultExpanded);

const kindMeta = computed(() => {
  switch (props.call.kind) {
    case "web_search":
      return { icon: Search, label: "Web Search", tint: "action-blue" };
    case "tool_search":
      return { icon: Search, label: "Tool Search", tint: "action-blue" };
    case "custom":
      return { icon: Wrench, label: "Custom", tint: "action-amber" };
    case "server_tool_use":
      return { icon: Wrench, label: "Server Tool", tint: "action-amber" };
    default:
      return { icon: Wrench, label: "Function", tint: "action-amber" };
  }
});

const parsedArguments = computed(() => {
  if (props.call.parsedArguments) return props.call.parsedArguments;
  return parseToolArgs(props.call.arguments);
});

const hasArguments = computed(() => {
  const args = props.call.arguments;
  if (!args) return false;
  const parsed = parsedArguments.value;
  return Object.keys(parsed).length > 0;
});

// Size estimate for the raw arguments string
const argsByteSize = computed(() => {
  if (!props.call.arguments) return 0;
  // arguments is JSON text; approximate byte size = string length for ASCII,
  // but use a safer UTF-8 estimate via TextEncoder when available.
  try {
    return new TextEncoder().encode(props.call.arguments).length;
  } catch {
    return props.call.arguments.length;
  }
});

const isOversized = computed(() => argsByteSize.value > MAX_INLINE_BYTES);

// Result handling
const resultByteSize = computed(() => {
  if (!props.result?.output) return 0;
  try {
    return new TextEncoder().encode(props.result.output).length;
  } catch {
    return props.result.output.length;
  }
});

const resultIsOversized = computed(() => resultByteSize.value > MAX_INLINE_BYTES);

const parsedResult = computed<unknown>(() => {
  const out = props.result?.output;
  if (!out) return undefined;
  // Try to parse as JSON for structured display; fall back to raw string.
  try {
    return JSON.parse(out);
  } catch {
    return out;
  }
});

const hasResult = computed(() => Boolean(props.result && props.result.output));

function formatBytes(bytes: number): string {
  if (bytes <= 0) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function toggle() {
  expanded.value = !expanded.value;
}

// For web_search, extract query from parsed args for inline display
const searchQuery = computed(() => {
  if (props.call.kind !== "web_search") return undefined;
  const args = parsedArguments.value;
  const q = (args as Record<string, unknown>)?.query;
  return typeof q === "string" ? q : undefined;
});
</script>

<template>
  <div
    class="rounded-lg border bg-action-amber/5 border-action-amber/20 overflow-hidden transition-colors"
    :class="expanded ? 'hover:border-action-amber/35' : ''"
  >
    <!-- Header row -->
    <button
      type="button"
      class="w-full flex items-center gap-2 p-3 text-left hover:bg-action-amber/5 transition-colors cursor-pointer"
      :aria-expanded="expanded"
      @click="toggle"
    >
      <ChevronDown
        class="size-3.5 text-muted-foreground/60 transition-transform duration-200 shrink-0"
        :class="{ 'rotate-180': expanded }"
      />
      <component :is="kindMeta.icon" class="size-3.5 text-action-amber shrink-0" />
      <span class="text-xs font-bold font-mono text-action-amber truncate">
        {{ call.name || t("logs.unknownTool") }}
      </span>
      <span
        v-if="call.id"
        class="text-[11px] font-mono text-muted-foreground bg-muted px-1.5 py-0.5 rounded border border-border/30 shrink-0 max-w-[40%] truncate"
        :title="call.id"
      >
        {{ call.id }}
      </span>
      <Badge
        v-if="call.status && call.status !== 'completed'"
        variant="outline"
        class="text-[11px] h-4 py-0 px-1.5 shrink-0"
      >
        {{ call.status }}
      </Badge>
      <span
        v-if="index !== undefined"
        class="ml-auto text-[11px] font-mono text-muted-foreground/50 shrink-0"
      >
        #{{ index + 1 }}
      </span>
    </button>

    <!-- Body -->
    <div v-if="expanded" class="px-3 pb-3 pt-1 space-y-2.5 border-t border-action-amber/10">
      <!-- Inline search query for web_search -->
      <div v-if="searchQuery" class="text-xs font-mono text-muted-foreground/90 pt-2">
        <span class="text-muted-foreground/60">query: </span>
        <span class="text-foreground/90">"{{ searchQuery }}"</span>
      </div>

      <!-- Arguments -->
      <div v-if="hasArguments" class="space-y-1.5">
        <div class="flex items-center justify-between">
          <span class="text-[11px] font-bold text-muted-foreground uppercase tracking-wider">
            {{ t("logs.toolArguments") }}
          </span>
          <span v-if="isOversized" class="text-[11px] font-mono text-status-warning">
            {{ formatBytes(argsByteSize) }}
          </span>
        </div>
        <JsonViewer
          :data="parsedArguments"
          :deep="2"
          :max-height="isOversized ? 'max-h-[400px]' : 'max-h-72'"
        />
      </div>
      <div
        v-else-if="call.arguments && !hasArguments"
        class="text-[11px] text-muted-foreground italic pt-1"
      >
        {{ call.arguments || t("logs.noArguments") }}
      </div>

      <!-- Result -->
      <div v-if="hasResult && result" class="space-y-1.5 pt-2 border-t border-action-amber/10">
        <div class="flex items-center justify-between">
          <span
            class="text-[11px] font-bold text-muted-foreground uppercase tracking-wider flex items-center gap-1.5"
          >
            <ArrowRight class="size-3" />
            {{ t("logs.toolResult") }}
          </span>
          <div class="flex items-center gap-2">
            <Badge
              v-if="result.isError"
              variant="outline"
              class="text-[11px] h-4 py-0 px-1.5 border-status-error/30 text-status-error"
            >
              error
            </Badge>
            <span v-if="resultIsOversized" class="text-[11px] font-mono text-status-warning">
              {{ formatBytes(resultByteSize) }}
            </span>
          </div>
        </div>
        <JsonViewer
          :data="parsedResult"
          :deep="2"
          :max-height="resultIsOversized ? 'max-h-[400px]' : 'max-h-72'"
        />
      </div>
    </div>
  </div>
</template>
