<script setup lang="ts">
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { useMcpServerRow } from "@/composables/useMcpServerRow";
import type { McpServerCapabilities, McpServerRead, McpServerStatus } from "@/types/schemas";

const props = defineProps<{
  server: McpServerRead;
  status?: McpServerStatus;
  isLoading?: boolean;
  serverCapabilities?: McpServerCapabilities;
  capabilitiesFailed?: boolean;
}>();

const emit = defineEmits<{
  open: [tab: "tools" | "prompts" | "resources"];
}>();

const { t } = useI18n();
const { toolsCount, promptsCount, resourcesCount, showCapabilities } = useMcpServerRow(
  () => props.server,
  () => props.status,
  () => props.serverCapabilities
);

const isEmpty = computed(
  () => toolsCount.value === 0 && promptsCount.value === 0 && resourcesCount.value === 0
);
</script>

<template>
  <div v-if="showCapabilities" class="flex items-center gap-1.5 flex-wrap">
    <button
      v-if="toolsCount > 0"
      class="inline-flex items-center gap-1 rounded-full border border-border/60 bg-background/55 px-2 py-0 text-[11px] font-mono transition-colors hover:bg-accent max-sm:min-h-11 max-sm:px-3"
      :disabled="isLoading"
      @click.stop="emit('open', 'tools')"
    >
      {{ toolsCount }} {{ t("mcpServers.tools") }}
    </button>
    <button
      v-if="promptsCount > 0"
      class="inline-flex items-center gap-1 rounded-full border border-border/60 bg-background/55 px-2 py-0 text-[11px] font-mono transition-colors hover:bg-accent max-sm:min-h-11 max-sm:px-3"
      :disabled="isLoading"
      @click.stop="emit('open', 'prompts')"
    >
      {{ promptsCount }} {{ t("mcpServers.prompts") }}
    </button>
    <button
      v-if="resourcesCount > 0"
      class="inline-flex items-center gap-1 rounded-full border border-border/60 bg-background/55 px-2 py-0 text-[11px] font-mono transition-colors hover:bg-accent max-sm:min-h-11 max-sm:px-3"
      :disabled="isLoading"
      @click.stop="emit('open', 'resources')"
    >
      {{ resourcesCount }} {{ t("mcpServers.resources") }}
    </button>
    <span v-if="capabilitiesFailed" class="text-[11px] text-destructive/70 italic">{{
      t("mcpServers.capabilitiesFailed")
    }}</span>
    <span v-else-if="isEmpty" class="text-[11px] text-muted-foreground italic">—</span>
  </div>
</template>
