<script setup lang="ts">
/**
 * One conversation message in the Logs I/O request view.
 *
 * Renders the normalized typed blocks produced by logRequestParser (text,
 * thinking, tool_call, tool_result, image/audio/file chips) instead of
 * flattening everything into a single text blob — so tool history and
 * multimodal parts stay visible, and base64 payloads never hit the DOM.
 *
 * The item is collapsible (controlled by the parent so "expand/collapse all"
 * works): collapsed shows a one-line preview with block-type chips and a
 * character count.
 *
 * Visual contract: the item renders as a BARE row — the messages list
 * container supplies the hairline dividers (One-Console pattern, same as the
 * tools list), so a row never draws its own border/radius. The header is
 * text-only: role is a plain tinted label (no icon, no pill tag). Inner
 * blocks are plain text or BORDERLESS tonal washes — a bordered box inside
 * the row would be a card-in-card.
 */
import { Brain, File, ImageIcon, Link, Volume2, Wrench } from "@lucide/vue";
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import JsonViewer from "@/components/common/JsonViewer.vue";
import LogCollapsibleRow from "@/components/common/LogCollapsibleRow.vue";
import LogTextBlock from "@/components/common/LogTextBlock.vue";
import { Badge } from "@/components/ui/badge";
import {
  firstLinePreview,
  formatBytes,
  formatCharCount,
  isOversizedPayload,
  parseResultOutput,
} from "@/utils/logFormat";
import type { MessageBlock, ParsedLogMessage } from "@/utils/logRequestParser";

const props = defineProps<{
  msg: ParsedLogMessage;
  index: number;
  expanded: boolean;
}>();

const emit = defineEmits<(e: "update:expanded", value: boolean) => void>();

const { t } = useI18n();

/** Role text tint only — no icon, no pill tag. The role label is plain text
 * (uppercase, tinted) so the header stays quiet; color is never the sole
 * signal, the text label carries it. */
const roleText = computed(() => {
  switch (props.msg.role) {
    case "user":
      return "text-action-blue";
    case "assistant":
    case "reasoning":
      return "text-action-violet";
    case "tool":
      return "text-action-blue";
    case "system":
    case "developer":
      return "text-action-amber";
    default:
      return "text-muted-foreground";
  }
});

/** Role label in "User:" form — capitalized + colon, no tag, no icon. */
const roleLabel = computed(() => {
  const role = props.msg.role;
  return role ? role.charAt(0).toUpperCase() + role.slice(1) + ":" : "";
});

const preview = computed(() => firstLinePreview(props.msg.plainText));

const charCountLabel = computed(() => formatCharCount(props.msg.charCount));

function blockKey(block: MessageBlock, i: number): string {
  return `${block.kind}-${i}`;
}

function toggle() {
  emit("update:expanded", !props.expanded);
}
</script>

<template>
  <LogCollapsibleRow
    :open="expanded"
    card-class="bg-transparent"
    button-class="w-full flex items-center gap-2 px-3 py-2.5 text-left cursor-pointer hover:bg-muted/20 transition-colors"
    content-class="border-t border-border/25"
    @toggle="toggle"
  >
    <template #header>
      <span class="text-[11px] font-bold shrink-0" :class="roleText">
        {{ roleLabel }}
      </span>

      <!-- Participant / tool-call linkage -->
      <span
        v-if="msg.name"
        class="text-[11px] font-mono text-muted-foreground truncate max-w-32"
        :title="msg.name"
      >
        {{ msg.name }}
      </span>
      <span
        v-if="msg.toolCallId"
        class="text-[11px] font-mono text-muted-foreground bg-muted/60 px-1.5 py-0.5 rounded-full truncate max-w-40"
        :title="msg.toolCallId"
      >
        → {{ msg.toolCallId }}
      </span>

      <!-- Collapsed: one-line preview -->
      <template v-if="!expanded">
        <span class="text-[11px] font-mono text-muted-foreground/60 truncate min-w-0 flex-1">
          {{ preview }}
        </span>
      </template>
      <span v-else class="flex-1" />

      <span class="text-[11px] font-mono text-muted-foreground/50 shrink-0 tabular-nums">
        {{ charCountLabel }} · #{{ index + 1 }}
      </span>
    </template>

    <!-- Body -->
    <div class="px-3 pb-3 pt-1 space-y-2.5" style="contain: content">
      <template v-for="(block, i) in msg.blocks" :key="blockKey(block, i)">
        <!-- Text (sliced at 100k chars; copy-all/expand for the tail) -->
        <LogTextBlock
          v-if="block.kind === 'text'"
          :text="block.text"
          class="text-xs font-mono text-foreground/90"
        />

        <!-- Thinking (borderless tonal wash — no card-in-card) -->
        <div v-else-if="block.kind === 'thinking'" class="rounded-md bg-muted/40 px-2.5 py-2">
          <div
            class="flex items-center gap-1.5 text-[11px] font-bold text-action-violet uppercase tracking-wider mb-1"
          >
            <Brain class="size-3" />
            {{ t("logs.thinkingContent") }}
          </div>
          <div v-if="block.redacted" class="text-[11px] font-mono text-action-violet/70 italic">
            {{ t("logs.redactedContent") }}
          </div>
          <LogTextBlock
            v-else
            :text="block.text"
            class="text-xs font-mono text-foreground/90 max-h-72 overflow-y-auto scrollbar-thin"
          />
        </div>

        <!-- Tool call -->
        <div
          v-else-if="block.kind === 'tool_call'"
          class="rounded-md bg-muted/40 px-2.5 py-2 space-y-1.5"
        >
          <div class="flex items-center gap-1.5 flex-wrap">
            <Wrench class="size-3 text-action-amber shrink-0" />
            <span class="text-xs font-bold font-mono text-action-amber">{{ block.name }}</span>
            <span
              v-if="block.id"
              class="text-[11px] font-mono text-muted-foreground bg-muted/60 px-1.5 py-0.5 rounded-full truncate max-w-48"
              :title="block.id"
            >
              {{ block.id }}
            </span>
          </div>
          <JsonViewer
            v-if="Object.keys(block.parsedArguments).length > 0"
            :data="block.parsedArguments"
            :deep="2"
            max-height="max-h-60"
            flat
          />
          <div v-else class="text-[11px] text-muted-foreground italic">
            {{ block.arguments || t("logs.noArguments") }}
          </div>
        </div>

        <!-- Tool result -->
        <div
          v-else-if="block.kind === 'tool_result'"
          class="rounded-md bg-muted/40 px-2.5 py-2 space-y-1.5"
        >
          <div class="flex items-center gap-1.5 flex-wrap">
            <span class="text-[11px] font-bold text-action-blue uppercase tracking-wider">
              {{ t("logs.toolResult") }}
            </span>
            <Badge
              v-if="block.isError"
              variant="outline"
              class="text-[11px] h-4 py-0 px-1.5 border-status-error/30 text-status-error"
            >
              {{ t("logs.toolResultError") }}
            </Badge>
            <span
              v-if="block.id"
              class="text-[11px] font-mono text-muted-foreground bg-muted/60 px-1.5 py-0.5 rounded-full truncate max-w-48"
              :title="block.id"
            >
              {{ block.id }}
            </span>
          </div>
          <!-- Oversized outputs are sliced as text — a full JSON tree is
               thousands of DOM nodes and freezes the details sheet. -->
          <LogTextBlock
            v-if="block.output && isOversizedPayload(block.output)"
            :text="block.output"
            class="text-xs font-mono text-foreground/90 max-h-60 overflow-y-auto scrollbar-thin"
          />
          <JsonViewer
            v-else-if="block.output"
            :data="parseResultOutput(block.output)"
            :deep="2"
            max-height="max-h-60"
            flat
          />
        </div>

        <!-- Image chip (base64 never inlined) -->
        <div
          v-else-if="block.kind === 'image'"
          class="flex items-center gap-2 rounded-md bg-muted/30 px-2.5 py-2 text-[11px] font-mono text-muted-foreground"
        >
          <ImageIcon class="size-3.5 shrink-0 text-muted-foreground/70" />
          <span v-if="block.url" class="flex items-center gap-1 min-w-0">
            <Link class="size-3 shrink-0" />
            <span class="truncate" :title="block.url">{{ block.url }}</span>
          </span>
          <span v-else>{{ block.mediaType || t("logs.mediaImage") }}</span>
          <span v-if="block.bytes" class="text-muted-foreground/60"
            >~{{ formatBytes(block.bytes) }}</span
          >
          <span v-if="block.detail" class="text-muted-foreground/60">· {{ block.detail }}</span>
        </div>

        <!-- Audio chip -->
        <div
          v-else-if="block.kind === 'audio'"
          class="flex items-center gap-2 rounded-md bg-muted/30 px-2.5 py-2 text-[11px] font-mono text-muted-foreground"
        >
          <Volume2 class="size-3.5 shrink-0 text-muted-foreground/70" />
          <span>{{ block.format || t("logs.mediaAudio") }}</span>
          <span v-if="block.bytes" class="text-muted-foreground/60"
            >~{{ formatBytes(block.bytes) }}</span
          >
        </div>

        <!-- File chip -->
        <div
          v-else-if="block.kind === 'file'"
          class="flex items-center gap-2 rounded-md bg-muted/30 px-2.5 py-2 text-[11px] font-mono text-muted-foreground"
        >
          <File class="size-3.5 shrink-0 text-muted-foreground/70" />
          <span class="truncate" :title="block.name">{{ block.name || t("logs.mediaFile") }}</span>
          <span v-if="block.bytes" class="text-muted-foreground/60"
            >~{{ formatBytes(block.bytes) }}</span
          >
        </div>

        <!-- Unknown block: structured fallback -->
        <div v-else class="rounded-md bg-muted/40 px-2.5 py-2 space-y-1.5">
          <span class="text-[11px] font-mono text-muted-foreground uppercase tracking-wider">
            {{ block.label }}
          </span>
          <JsonViewer :data="block.raw" :deep="2" max-height="max-h-48" flat />
        </div>
      </template>

      <!-- Empty message -->
      <div v-if="msg.blocks.length === 0" class="text-[11px] text-muted-foreground italic">
        {{ t("logs.emptyMessage") }}
      </div>
    </div>
  </LogCollapsibleRow>
</template>
