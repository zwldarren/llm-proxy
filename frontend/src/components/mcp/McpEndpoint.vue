<script setup lang="ts">
import { Check, Globe, Loader2 } from "@lucide/vue";
import { useI18n } from "vue-i18n";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useMcpServerRow } from "@/composables/useMcpServerRow";
import type { McpServerCapabilities, McpServerRead, McpServerStatus } from "@/types/schemas";

const props = defineProps<{
  server: McpServerRead;
  status?: McpServerStatus;
  isLoading?: boolean;
  serverCapabilities?: McpServerCapabilities;
  capabilitiesFailed?: boolean;
}>();

const { t } = useI18n();
const { fullProxyUrl, copied, copyProxyUrl, commandDisplay, capabilitiesFetched } = useMcpServerRow(
  () => props.server,
  () => props.status,
  () => props.serverCapabilities
);
</script>

<template>
  <!-- Proxy URL (copy on click) or the closest available fallback -->
  <Tooltip v-if="fullProxyUrl">
    <TooltipTrigger as-child>
      <button
        class="flex items-center gap-1.5 min-w-0 max-w-full text-muted-foreground hover:text-foreground transition-colors"
        :disabled="isLoading"
        @click.stop="copyProxyUrl"
      >
        <component
          :is="copied ? Check : Globe"
          class="w-3.5 h-3.5 shrink-0 transition-all duration-300"
          :class="copied ? 'text-status-success' : 'text-muted-foreground/70'"
        />
        <span class="truncate min-w-0 font-mono text-xs">{{ fullProxyUrl }}</span>
      </button>
    </TooltipTrigger>
    <TooltipContent class="break-all max-w-xs">{{
      copied ? t("common.copied") : fullProxyUrl
    }}</TooltipContent>
  </Tooltip>
  <span
    v-else-if="commandDisplay"
    class="block truncate font-mono text-xs text-muted-foreground"
    :title="commandDisplay"
  >
    {{ commandDisplay }}
  </span>
  <span v-else-if="capabilitiesFailed" class="text-xs text-destructive/70 italic">
    {{ t("mcpServers.capabilitiesFailed") }}
  </span>
  <span
    v-else-if="status?.status === 'running' && !status?.error_message && !capabilitiesFetched"
    class="flex items-center gap-2 text-xs text-muted-foreground"
  >
    <Loader2 class="h-3 w-3 animate-spin text-primary shrink-0" />
    <span>{{ t("mcpServers.loadingCapabilities") }}</span>
  </span>
  <span v-else class="text-xs text-muted-foreground">—</span>
</template>
