<script setup lang="ts">
import { Check, Edit, Globe, Loader2, Trash2 } from "@lucide/vue";
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { TableCell, TableRow } from "@/components/ui/table";
import type { McpServerCapabilities, McpServerRead, McpServerStatus } from "@/types/schemas";
import { cn } from "@/lib/utils";
import { useMcpServerMeta } from "@/composables/useMcpServerMeta";

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

const { statusConfig, typeConfig, isEnabled, fullProxyUrl, copied, copyProxyUrl } =
  useMcpServerMeta(
    () => props.server,
    () => props.status,
    t
  );

const capabilitiesFetched = computed(() => !!props.serverCapabilities);

const toolsCount = computed(() => props.serverCapabilities?.tools.length ?? 0);
const promptsCount = computed(() => props.serverCapabilities?.prompts.length ?? 0);
const resourcesCount = computed(() => props.serverCapabilities?.resources.length ?? 0);

const showCapabilities = computed(
  () => props.status?.status === "running" && capabilitiesFetched.value
);

// Secondary endpoint display for stdio servers without a proxy URL
const commandDisplay = computed(() => {
  if (fullProxyUrl.value) return null;
  if (props.server.type !== "stdio") return null;
  const cmd = props.server.command || "";
  const args = Array.isArray(props.server.args) ? props.server.args.join(" ") : "";
  return [cmd, args].filter(Boolean).join(" ") || null;
});
</script>

<template>
  <TableRow class="group" :class="isLoading && 'opacity-75 pointer-events-none'">
    <!-- Type icon -->
    <TableCell class="w-12">
      <div
        :class="
          cn(
            'w-9 h-9 rounded-lg flex items-center justify-center shrink-0 border',
            typeConfig.bg,
            typeConfig.border
          )
        "
      >
        <component :is="typeConfig.icon" :class="cn('w-4.5 h-4.5', typeConfig.iconColor)" />
      </div>
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
      <Badge
        variant="secondary"
        :class="
          cn(
            'px-2 py-0 text-[11px] font-semibold uppercase tracking-wider border',
            typeConfig.bg,
            typeConfig.text,
            typeConfig.border
          )
        "
      >
        {{ typeConfig.label }}
      </Badge>
    </TableCell>

    <!-- Endpoint (proxy URL or command) -->
    <TableCell class="w-64 overflow-hidden">
      <button
        v-if="fullProxyUrl"
        class="flex items-center gap-1.5 max-w-full text-muted-foreground hover:text-foreground transition-colors"
        :disabled="isLoading"
        :title="fullProxyUrl"
        @click.stop="copyProxyUrl"
      >
        <component
          :is="copied ? Check : Globe"
          class="w-3.5 h-3.5 shrink-0 transition-all duration-300"
          :class="copied ? 'text-status-success' : 'text-muted-foreground/70'"
        />
        <span class="truncate min-w-0 font-mono text-xs">{{ fullProxyUrl }}</span>
      </button>
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
        class="flex items-center gap-2 text-xs text-muted-foreground/60"
      >
        <Loader2 class="h-3 w-3 animate-spin text-primary shrink-0" />
        <span>{{ t("mcpServers.loadingCapabilities") }}</span>
      </span>
      <span v-else class="text-xs text-muted-foreground/50">—</span>
    </TableCell>

    <!-- Status -->
    <TableCell class="w-32">
      <div class="flex items-center gap-2">
        <span
          :class="
            cn(
              'w-2 h-2 rounded-full ring-2 transition-all duration-300 shrink-0',
              statusConfig.bg,
              statusConfig.ring,
              statusConfig.pulse && 'animate-pulse'
            )
          "
        />
        <span :class="cn('text-xs font-medium', statusConfig.color)">
          {{ statusConfig.label }}
        </span>
      </div>
    </TableCell>

    <!-- Enabled switch -->
    <TableCell class="w-20">
      <div class="flex items-center justify-center">
        <Switch
          :disabled="isLoading"
          :model-value="isEnabled"
          @update:model-value="emit('toggle', $event)"
        />
      </div>
    </TableCell>

    <!-- Capabilities -->
    <TableCell class="w-44">
      <div v-if="showCapabilities" class="flex items-center gap-1.5 flex-wrap">
        <button
          v-if="toolsCount > 0"
          class="inline-flex items-center gap-1 rounded-full border border-border/60 bg-background/55 px-2 py-0 text-[11px] font-mono transition-colors hover:bg-accent"
          :disabled="isLoading"
          @click.stop="emit('openCapabilities', 'tools')"
        >
          {{ toolsCount }} {{ t("mcpServers.tools") }}
        </button>
        <button
          v-if="promptsCount > 0"
          class="inline-flex items-center gap-1 rounded-full border border-border/60 bg-background/55 px-2 py-0 text-[11px] font-mono transition-colors hover:bg-accent"
          :disabled="isLoading"
          @click.stop="emit('openCapabilities', 'prompts')"
        >
          {{ promptsCount }} {{ t("mcpServers.prompts") }}
        </button>
        <button
          v-if="resourcesCount > 0"
          class="inline-flex items-center gap-1 rounded-full border border-border/60 bg-background/55 px-2 py-0 text-[11px] font-mono transition-colors hover:bg-accent"
          :disabled="isLoading"
          @click.stop="emit('openCapabilities', 'resources')"
        >
          {{ resourcesCount }} {{ t("mcpServers.resources") }}
        </button>
        <span v-if="capabilitiesFailed" class="text-[11px] text-destructive/70 italic">{{
          t("mcpServers.capabilitiesFailed")
        }}</span>
        <span
          v-else-if="toolsCount === 0 && promptsCount === 0 && resourcesCount === 0"
          class="text-[11px] text-muted-foreground/60 italic"
          >—</span
        >
      </div>
      <span v-else class="text-xs text-muted-foreground/50">—</span>
    </TableCell>

    <!-- Actions -->
    <TableCell class="w-24 text-right">
      <div
        class="flex items-center justify-end gap-1 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 sm:group-focus-within:opacity-100 transition-opacity"
      >
        <Button
          variant="ghost"
          size="icon"
          class="h-9 w-9"
          :disabled="isLoading"
          @click.stop="emit('edit')"
        >
          <Edit class="w-4 h-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          class="h-9 w-9 text-destructive hover:text-destructive hover:bg-destructive/10"
          :disabled="isLoading"
          @click.stop="emit('delete')"
        >
          <Trash2 class="w-4 h-4" />
        </Button>
      </div>
    </TableCell>
  </TableRow>
</template>
