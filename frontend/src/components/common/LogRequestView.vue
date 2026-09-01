<script setup lang="ts">
/**
 * Request side of the Logs I/O view.
 *
 * Faithful structured rendering of the stored request body:
 *  - EVERY scalar parameter in a grid (labels are exact JSON keys),
 *    complex params under an "advanced" collapsible,
 *  - the offered tool definitions (LogRequestTools),
 *  - the system prompt (collapsible, char count),
 *  - the conversation messages as typed, collapsible blocks with a
 *    global expand/collapse control,
 *  - raw headers/body live behind the Formatted/Raw view-mode toggle
 *    (LogViewModeToggle), never mixed into the formatted flow.
 */
import { Braces, ChevronDown, Code, Database, Settings, Shield } from "@lucide/vue";
import { computed, ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import JsonViewer from "@/components/common/JsonViewer.vue";
import LogCollapsibleRow from "@/components/common/LogCollapsibleRow.vue";
import LogExpandControls from "@/components/common/LogExpandControls.vue";
import LogMessageItem from "@/components/common/LogMessageItem.vue";
import LogRequestTools from "@/components/common/LogRequestTools.vue";
import LogRawSection from "@/components/common/LogRawSection.vue";
import LogTextBlock from "@/components/common/LogTextBlock.vue";
import LogValueChip from "@/components/common/LogValueChip.vue";
import LogViewModeToggle from "@/components/common/LogViewModeToggle.vue";
import { Badge } from "@/components/ui/badge";
import type { LogRead } from "@/types/schemas";
import { formatCharCount, hasPayload, isOversizedPayload } from "@/utils/logFormat";
import { parseLogRequest } from "@/utils/logRequestParser";

const props = defineProps<{
  log: LogRead;
}>();

const { t } = useI18n();

// --- View mode -------------------------------------------------------------
// Formatted (structured rendering) and Raw (headers + body) are fully
// separate modes; raw never sits below the formatted content.
const viewMode = ref<"formatted" | "raw">("formatted");

// --- Body parsing ----------------------------------------------------------
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

const parsed = computed(() => parseLogRequest(parsedRequestBody.value));

// Image request detection (guards base64 in raw request bodies).
const requestType = computed<string>(() => {
  const rt = props.log?.log_metadata?.request_type;
  return typeof rt === "string" ? rt : "chat";
});
const isImageRequest = computed(
  () => requestType.value === "image_generation" || requestType.value === "image_edit"
);

// Sanitize a body for image requests: replaces oversized string values (base64
// image/mask data) with a size placeholder so the JSON viewer never lays out
// megabytes of text and freezes the page.
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

// --- Parameters ------------------------------------------------------------

// Long values (prompts, cache keys, stop sequences...) are truncated inside
// the tag; the full value is in the tooltip and the raw body accordion.
const PARAM_VALUE_TRUNCATE = 56;

function truncateParam(value: string): string {
  return value.length > PARAM_VALUE_TRUNCATE
    ? value.slice(0, PARAM_VALUE_TRUNCATE - 1) + "\u2026"
    : value;
}

const objectParamsAsRecord = computed(() => {
  const out: Record<string, unknown> = {};
  for (const p of parsed.value?.objectParams ?? []) out[p.key] = p.value;
  return out;
});

const showAdvancedParams = ref(false);

// --- System prompt -----------------------------------------------------------
const SYSTEM_COLLAPSE_THRESHOLD = 1500;
const systemExpanded = ref(true);

const systemCharLabel = computed(() => formatCharCount(parsed.value?.systemPrompt.length ?? 0));

// --- Messages ----------------------------------------------------------------
const MESSAGE_COLLAPSE_THRESHOLD = 600;
const messageExpanded = ref<boolean[]>([]);

/**
 * Reset every per-log collapse state (messages, system prompt, advanced
 * params) when a different log is opened.
 */
function initCollapseState() {
  viewMode.value = "formatted";
  const messages = parsed.value?.messages ?? [];
  messageExpanded.value = messages.map(
    (m, i) => m.charCount <= MESSAGE_COLLAPSE_THRESHOLD || i === messages.length - 1
  );
  systemExpanded.value = (parsed.value?.systemPrompt.length ?? 0) <= SYSTEM_COLLAPSE_THRESHOLD;
  showAdvancedParams.value = false;
}

function setMessageExpanded(i: number, value: boolean) {
  messageExpanded.value[i] = value;
}

function expandAllMessages() {
  messageExpanded.value = messageExpanded.value.map(() => true);
}

function collapseAllMessages() {
  messageExpanded.value = messageExpanded.value.map(() => false);
}

// Re-init collapse state when a different log is opened.
watch(() => props.log.request_id, initCollapseState, { immediate: true });

const totalMessageChars = computed(() =>
  (parsed.value?.messages ?? []).reduce((sum, m) => sum + m.charCount, 0)
);

const totalCharsLabel = computed(() => formatCharCount(totalMessageChars.value));

// --- Raw mode payload guards -------------------------------------------------
const hasRequestHeaders = computed(() => hasPayload(props.log?.request_headers));
const hasRequestBody = computed(() => hasPayload(props.log?.request_body));

// Raw request body: multi-MB JSON strings would freeze the viewer — render
// them as (sliced) text instead.
const oversizeRequestBodyText = computed((): string | null =>
  isOversizedPayload(safeRequestBody.value) ? safeRequestBody.value : null
);
</script>

<template>
  <div class="space-y-4">
    <div class="flex items-center justify-between gap-3 pb-2 border-b border-border/60">
      <h3 class="font-semibold text-base text-foreground flex items-center gap-2">
        <Database class="size-4 text-muted-foreground" />
        <span>{{ t("logs.request") }}</span>
      </h3>
      <div class="flex items-center gap-2.5">
        <Badge
          v-if="parsed && parsed.protocol !== 'unknown'"
          variant="outline"
          class="font-mono text-[11px] uppercase font-bold tracking-wider text-muted-foreground bg-muted/40 border-border/50 py-0.5 rounded-full"
        >
          {{ parsed.protocol }}
        </Badge>
        <LogViewModeToggle v-model="viewMode" />
      </div>
    </div>

    <template v-if="viewMode === 'formatted'">
      <!-- Parameters: every scalar param as a compact tag strip -->
      <div
        v-if="parsed && (parsed.scalarParams.length > 0 || parsed.objectParams.length > 0)"
        class="space-y-2"
      >
        <div class="flex items-center gap-1.5">
          <Settings class="size-3.5 text-muted-foreground/70" />
          <span class="text-xs text-muted-foreground font-bold uppercase tracking-wider">
            {{ t("logs.parameters") }}
          </span>
        </div>
        <div class="flex flex-wrap gap-1.5">
          <LogValueChip
            v-for="p in parsed.scalarParams"
            :key="p.key"
            :label="p.key"
            :value="truncateParam(p.value)"
            :title="`${p.key}: ${p.value}`"
          />

          <!-- Advanced (complex) params toggle, rendered as a chip-shaped button -->
          <button
            v-if="parsed.objectParams.length > 0"
            type="button"
            class="inline-flex items-center gap-1.5 text-[11px] font-mono bg-muted/50 rounded-full px-2.5 py-0.5 text-muted-foreground hover:text-foreground hover:bg-muted/80 transition-colors cursor-pointer"
            :aria-expanded="showAdvancedParams"
            @click="showAdvancedParams = !showAdvancedParams"
          >
            <Braces class="size-3 shrink-0" />
            <span>{{ t("logs.advancedParams") }} ({{ parsed.objectParams.length }})</span>
            <ChevronDown
              class="size-3 transition-transform duration-200 text-muted-foreground/60"
              :class="{ 'rotate-180': showAdvancedParams }"
            />
          </button>
        </div>
        <div v-if="showAdvancedParams && parsed.objectParams.length > 0">
          <JsonViewer :data="objectParamsAsRecord" :deep="2" max-height="max-h-80" />
        </div>
      </div>

      <!-- Offered tools -->
      <LogRequestTools
        v-if="parsed && parsed.tools.length > 0"
        :tools="parsed.tools"
        :tool-choice="parsed.toolChoice"
      />

      <!-- System prompt -->
      <LogCollapsibleRow
        v-if="parsed && parsed.systemPrompt"
        :open="systemExpanded"
        button-class="w-full flex items-center gap-2 px-3 py-2.5 text-left hover:bg-muted/20 transition-colors cursor-pointer"
        chevron-class="text-muted-foreground/60"
        content-class="px-3 pb-3 pt-0.5 border-t border-border/25"
        @toggle="systemExpanded = !systemExpanded"
      >
        <template #header>
          <Shield class="size-3.5 text-action-amber shrink-0" />
          <span class="text-[11px] text-action-amber font-bold uppercase tracking-wider">
            {{ t("logs.systemInstructions") }}
          </span>
          <span class="ml-auto text-[11px] font-mono text-muted-foreground/60 tabular-nums">
            {{ systemCharLabel }} {{ t("logs.charsUnit") }}
          </span>
        </template>
        <LogTextBlock
          :text="parsed.systemPrompt"
          class="text-xs font-mono text-foreground/90 max-h-72 overflow-y-auto scrollbar-thin"
        />
      </LogCollapsibleRow>

      <!-- Messages -->
      <div v-if="parsed && parsed.messages.length > 0" class="space-y-2">
        <div class="flex items-center gap-2">
          <h4 class="text-xs font-bold text-muted-foreground uppercase tracking-wider">
            {{ t("logs.conversationMessages") }}
          </h4>
          <span class="text-[11px] font-mono text-muted-foreground/60 tabular-nums">
            ({{ parsed.messages.length }} · {{ totalCharsLabel }} {{ t("logs.charsUnit") }})
          </span>
          <LogExpandControls @expand="expandAllMessages" @collapse="collapseAllMessages" />
        </div>

        <div class="divide-y divide-border/30 border-y border-border/40" style="contain: content">
          <LogMessageItem
            v-for="(msg, index) in parsed.messages"
            :key="index"
            :msg="msg"
            :index="index"
            :expanded="messageExpanded[index] ?? false"
            @update:expanded="setMessageExpanded(index, $event)"
          />
        </div>
      </div>

      <!-- Non-chat-like payload note (only when nothing else was parsed) -->
      <div
        v-if="
          parsed &&
          !parsed.isChatLike &&
          parsed.scalarParams.length === 0 &&
          parsed.objectParams.length === 0
        "
        class="text-xs text-muted-foreground italic bg-muted/5 border border-border/40 p-3 rounded-lg text-center"
      >
        {{ t("logs.nonStandardPayload") }}
      </div>

      <!-- Body could not be parsed at all — point to Raw mode -->
      <div
        v-if="!parsed"
        class="text-xs text-muted-foreground italic bg-muted/5 border border-border/40 p-3 rounded-lg text-center"
      >
        {{ t("logs.unparseableRequest") }}
      </div>
    </template>

    <!-- Raw mode: headers + body shown directly, no accordions -->
    <template v-else>
      <LogRawSection
        :icon="Database"
        :title="t('logs.requestHeaders')"
        :has-content="hasRequestHeaders"
      >
        <JsonViewer :data="log.request_headers" maxHeight="max-h-72" flat />
        <template #empty>{{ t("logs.noHeaders") }}</template>
      </LogRawSection>

      <LogRawSection
        :icon="Code"
        :title="t('logs.requestBody')"
        :has-content="!!oversizeRequestBodyText || hasRequestBody"
        :padded="!oversizeRequestBodyText"
      >
        <LogTextBlock
          v-if="oversizeRequestBodyText"
          :text="oversizeRequestBodyText"
          class="text-[11px] font-mono text-foreground/80 max-h-[500px] overflow-y-auto p-3 scrollbar-thin"
        />
        <JsonViewer v-else :data="safeRequestBody" maxHeight="max-h-[500px]" flat />
        <template #empty>{{ t("logs.noRequestBody") }}</template>
      </LogRawSection>
    </template>
  </div>
</template>
