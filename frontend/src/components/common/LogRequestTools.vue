<script setup lang="ts">
/**
 * Available-tools list for the Logs I/O request view.
 *
 * Parses the request's tool definitions (OpenAI function/custom, Anthropic
 * input_schema tools + server tools, Responses built-ins like web_search /
 * code_interpreter / mcp) into a scannable list: name + kind + one-line
 * description, with the full JSON schema one click away. A filter input
 * appears when the tool count is large.
 */
import { ChevronDown, Search, Wrench } from "@lucide/vue";
import { computed, ref } from "vue";
import { useI18n } from "vue-i18n";
import JsonViewer from "@/components/common/JsonViewer.vue";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import type { RequestToolInfo } from "@/utils/logRequestParser";

const props = defineProps<{
  tools: RequestToolInfo[];
  toolChoice?: string;
}>();

const { t } = useI18n();

const filter = ref("");
const expanded = ref<Set<number>>(new Set());

const filtered = computed(() => {
  const q = filter.value.trim().toLowerCase();
  if (!q) return props.tools.map((tool, i) => ({ tool, i }));
  return props.tools
    .map((tool, i) => ({ tool, i }))
    .filter(
      ({ tool }) =>
        tool.name.toLowerCase().includes(q) ||
        tool.kind.toLowerCase().includes(q) ||
        (tool.description ?? "").toLowerCase().includes(q)
    );
});

const showFilter = computed(() => props.tools.length > 6);

function kindBadgeClass(kind: string): string {
  // Function/custom tools are amber; server-side/built-in tools are blue.
  if (kind === "function" || kind === "custom") {
    return "bg-action-amber/10 border-action-amber/20 text-action-amber";
  }
  return "bg-action-blue/10 border-action-blue/20 text-action-blue";
}

function toggle(i: number) {
  const next = new Set(expanded.value);
  if (next.has(i)) next.delete(i);
  else next.add(i);
  expanded.value = next;
}

function expandAll() {
  expanded.value = new Set(props.tools.map((_, i) => i));
}

function collapseAll() {
  expanded.value = new Set();
}
</script>

<template>
  <div class="space-y-2">
    <!-- Section header (label + controls live outside the list, hairline rows below) -->
    <div class="flex items-center gap-2 flex-wrap">
      <Wrench class="size-3.5 text-muted-foreground shrink-0" />
      <span class="text-xs font-bold text-muted-foreground uppercase tracking-wider">
        {{ t("logs.toolsOffered") }}
      </span>
      <span class="text-[11px] font-mono text-muted-foreground/70">({{ tools.length }})</span>

      <Badge
        v-if="toolChoice"
        variant="outline"
        class="font-mono text-[11px] py-0 px-1.5 rounded-full border-border/60 text-muted-foreground"
        :title="t('logs.toolChoice')"
      >
        {{ t("logs.toolChoice") }}: {{ toolChoice }}
      </Badge>

      <div class="ml-auto flex items-center gap-1">
        <div v-if="showFilter" class="relative">
          <Search
            class="size-3 absolute left-2 top-1/2 -translate-y-1/2 text-muted-foreground/60 pointer-events-none"
          />
          <Input
            v-model="filter"
            type="text"
            :aria-label="t('logs.filterTools')"
            :placeholder="t('logs.filterTools')"
            class="h-7 max-sm:h-7 w-36 sm:w-44 pl-7 pr-2 text-[11px] md:text-[11px] font-mono rounded-md"
          />
        </div>
        <LogExpandControls @expand="expandAll" @collapse="collapseAll" />
      </div>
    </div>

    <!-- Tool rows: hairline list, no enclosing card -->
    <div class="divide-y divide-border/30 border-y border-border/40">
      <div v-for="{ tool, i } in filtered" :key="i">
        <button
          type="button"
          class="w-full flex items-center gap-2 px-3 py-2.5 text-left hover:bg-muted/20 transition-colors cursor-pointer"
          :aria-expanded="expanded.has(i)"
          @click="toggle(i)"
        >
          <ChevronDown
            class="size-3.5 shrink-0 text-muted-foreground/60 transition-transform duration-200"
            :class="{ 'rotate-180': expanded.has(i) }"
          />
          <span class="text-xs font-bold font-mono text-foreground/90 truncate">{{
            tool.name
          }}</span>
          <span
            class="text-[11px] font-mono px-1.5 py-0.5 rounded-full border shrink-0"
            :class="kindBadgeClass(tool.kind)"
          >
            {{ tool.kind }}
          </span>
          <span
            v-if="!expanded.has(i) && (tool.description || tool.summary)"
            class="text-[11px] text-muted-foreground/60 truncate min-w-0 flex-1 hidden sm:block"
          >
            {{ tool.description || tool.summary }}
          </span>
        </button>

        <div v-if="expanded.has(i)" class="px-3 pb-3 pt-1 space-y-2 border-t border-border/25">
          <p
            v-if="tool.description"
            class="text-xs text-foreground/80 leading-relaxed whitespace-pre-wrap"
          >
            {{ tool.description }}
          </p>
          <p v-if="tool.summary" class="text-[11px] font-mono text-muted-foreground break-all">
            {{ tool.summary }}
          </p>
          <div v-if="tool.schema !== undefined" class="space-y-1">
            <span class="text-[11px] font-bold text-muted-foreground uppercase tracking-wider">
              {{ t("logs.toolSchema") }}
            </span>
            <JsonViewer :data="tool.schema" :deep="2" max-height="max-h-72" flat />
          </div>
          <div v-else-if="tool.raw !== undefined && !tool.summary" class="space-y-1">
            <JsonViewer :data="tool.raw" :deep="2" max-height="max-h-48" flat />
          </div>
        </div>
      </div>

      <div
        v-if="filtered.length === 0"
        class="px-3 py-4 text-center text-xs text-muted-foreground italic"
      >
        {{ t("logs.noToolsMatch") }}
      </div>
    </div>
  </div>
</template>
