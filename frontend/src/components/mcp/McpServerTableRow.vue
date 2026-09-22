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
import { TableCell, TableRow } from "@/components/ui/table";
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

const { isEnabled, showCapabilities } = useMcpServerRow(
  () => props.server,
  () => props.status,
  () => props.serverCapabilities
);
</script>

<template>
  <TableRow class="group" :class="isLoading && 'opacity-75 pointer-events-none'">
    <!-- Type icon -->
    <TableCell class="w-12">
      <McpTypeTile :server="server" />
    </TableCell>

    <!-- Name -->
    <TableCell class="font-medium">
      <div class="flex items-center gap-2 min-w-0">
        <span class="truncate" :title="server.name">{{ server.name }}</span>
        <Loader2 v-if="isLoading" class="w-3.5 h-3.5 animate-spin text-primary shrink-0" />
      </div>
      <p
        v-if="status?.error_message"
        class="text-[11px] text-destructive/80 line-clamp-1 mt-0.5"
        :title="status.error_message"
      >
        {{ status.error_message }}
      </p>
    </TableCell>

    <!-- Type badge -->
    <TableCell class="w-32">
      <McpTypeBadge :server="server" />
    </TableCell>

    <!-- Endpoint (proxy URL or command) -->
    <TableCell class="w-64 overflow-hidden">
      <McpEndpoint
        :server="server"
        :status="status"
        :is-loading="isLoading"
        :server-capabilities="serverCapabilities"
        :capabilities-failed="capabilitiesFailed"
      />
    </TableCell>

    <!-- Status -->
    <TableCell class="w-32">
      <div class="flex items-center gap-2">
        <McpStatusIndicator :status="status" />
      </div>
    </TableCell>

    <!-- Enabled switch -->
    <TableCell class="w-20">
      <div class="flex items-center justify-center">
        <Switch
          :disabled="isLoading"
          :model-value="isEnabled"
          :aria-label="t('mcpServers.enabled')"
          @update:model-value="emit('toggle', $event)"
        />
      </div>
    </TableCell>

    <!-- Capabilities -->
    <TableCell class="w-44">
      <McpCapabilityChips
        :server="server"
        :status="status"
        :is-loading="isLoading"
        :server-capabilities="serverCapabilities"
        :capabilities-failed="capabilitiesFailed"
        @open="emit('openCapabilities', $event)"
      />
      <span v-if="!showCapabilities" class="text-xs text-muted-foreground">—</span>
    </TableCell>

    <!-- Actions -->
    <TableCell class="w-24 text-right">
      <div
        class="flex items-center justify-end gap-1 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 sm:group-focus-within:opacity-100 transition-opacity"
      >
        <McpRowActions :is-loading="isLoading" @edit="emit('edit')" @delete="emit('delete')" />
      </div>
    </TableCell>
  </TableRow>
</template>
