<script setup lang="ts">
import { Loader2 } from "@lucide/vue";
import { useI18n } from "vue-i18n";
import McpCapabilityChips from "@/components/mcp/McpCapabilityChips.vue";
import McpEndpoint from "@/components/mcp/McpEndpoint.vue";
import McpRowActions from "@/components/mcp/McpRowActions.vue";
import McpStatusIndicator from "@/components/mcp/McpStatusIndicator.vue";
import McpTypeBadge from "@/components/mcp/McpTypeBadge.vue";
import McpTypeTile from "@/components/mcp/McpTypeTile.vue";
import { useMcpServerRow } from "@/composables/useMcpServerRow";
import { Switch } from "@/components/ui/switch";
import type { McpServerCapabilities, McpServerRead, McpServerStatus } from "@/types/schemas";

interface Props {
  server: McpServerRead;
  status?: McpServerStatus;
  isLoading?: boolean;
  serverCapabilities?: McpServerCapabilities;
  capabilitiesFailed?: boolean;
}

const props = defineProps<Props>();

const emit = defineEmits<{
  edit: [];
  delete: [];
  toggle: [enabled: boolean];
  openCapabilities: [tab: "tools" | "prompts" | "resources"];
}>();

const { t } = useI18n();

const { isEnabled } = useMcpServerRow(
  () => props.server,
  () => props.status,
  () => props.serverCapabilities
);
</script>

<template>
  <article
    class="group px-4 sm:px-6 py-2.5 border-b border-border transition-colors duration-150 hover:bg-muted/50"
    :class="isLoading && 'opacity-75 pointer-events-none'"
  >
    <div class="flex items-center gap-3">
      <!-- Type icon -->
      <McpTypeTile :server="server" />

      <!-- Name + type + endpoint + capabilities -->
      <div class="flex-1 min-w-0">
        <div class="flex items-center gap-2 min-w-0">
          <h3 class="text-sm font-medium text-foreground truncate" :title="server.name">
            {{ server.name }}
          </h3>
          <Loader2 v-if="isLoading" class="w-3.5 h-3.5 animate-spin text-primary shrink-0" />
          <McpTypeBadge :server="server" compact />
        </div>

        <div class="mt-1 flex items-center gap-2 min-w-0 flex-wrap">
          <McpEndpoint
            :server="server"
            :status="status"
            :is-loading="isLoading"
            :server-capabilities="serverCapabilities"
            :capabilities-failed="capabilitiesFailed"
          />

          <McpCapabilityChips
            :server="server"
            :status="status"
            :is-loading="isLoading"
            :server-capabilities="serverCapabilities"
            :capabilities-failed="capabilitiesFailed"
            @open="emit('openCapabilities', $event)"
          />
        </div>

        <p
          v-if="status?.error_message"
          class="text-[11px] text-destructive/80 line-clamp-1 mt-1"
          :title="status.error_message"
        >
          {{ status.error_message }}
        </p>
      </div>

      <!-- Status -->
      <div class="hidden sm:flex items-center gap-2 shrink-0 w-24">
        <McpStatusIndicator :status="status" />
      </div>

      <!-- Enabled switch -->
      <div class="flex items-center shrink-0">
        <Switch
          :disabled="isLoading"
          :model-value="isEnabled"
          :aria-label="t('mcpServers.enabled')"
          @update:model-value="emit('toggle', $event)"
        />
      </div>

      <!-- Actions -->
      <div
        class="flex items-center justify-end gap-1 shrink-0 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 sm:group-focus-within:opacity-100 transition-opacity"
      >
        <McpRowActions :is-loading="isLoading" @edit="emit('edit')" @delete="emit('delete')" />
      </div>
    </div>
  </article>
</template>
