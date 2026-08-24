<script setup lang="ts">
import { Check, Edit, Globe, Loader2, Trash2 } from "@lucide/vue";
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
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
  <article
    class="group px-4 sm:px-6 py-2.5 border-b border-border transition-colors duration-150 hover:bg-muted/50"
    :class="isLoading && 'opacity-75 pointer-events-none'"
  >
    <div class="flex items-center gap-3">
      <!-- Type icon -->
      <div
        :class="
          cn(
            'w-9 h-9 rounded-lg flex items-center justify-center shrink-0 border',
            typeConfig.bg,
            typeConfig.border
          )
        "
      >
        <component :is="typeConfig.icon" :class="cn('w-4 h-4', typeConfig.iconColor)" />
      </div>

      <!-- Name + type + endpoint + capabilities -->
      <div class="flex-1 min-w-0">
        <div class="flex items-center gap-2 min-w-0">
          <h3 class="text-sm font-medium text-foreground truncate" :title="server.name">
            {{ server.name }}
          </h3>
          <Loader2 v-if="isLoading" class="w-3.5 h-3.5 animate-spin text-primary shrink-0" />
          <Badge
            variant="secondary"
            :class="
              cn(
                'px-1.5 py-0 text-[11px] font-semibold uppercase tracking-wider border shrink-0',
                typeConfig.bg,
                typeConfig.text,
                typeConfig.border
              )
            "
          >
            {{ typeConfig.label }}
          </Badge>
        </div>

        <div class="mt-1 flex items-center gap-2 min-w-0 flex-wrap">
          <!-- Endpoint (proxy URL or command) -->
          <button
            v-if="fullProxyUrl"
            class="flex items-center gap-1.5 min-w-0 max-w-full text-muted-foreground hover:text-foreground transition-colors"
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
            class="truncate font-mono text-xs text-muted-foreground"
            :title="commandDisplay"
          >
            {{ commandDisplay }}
          </span>
          <span v-else-if="capabilitiesFailed" class="text-xs text-destructive/70 italic">
            {{ t("mcpServers.capabilitiesFailed") }}
          </span>
          <span
            v-else-if="
              status?.status === 'running' && !status?.error_message && !capabilitiesFetched
            "
            class="flex items-center gap-2 text-xs text-muted-foreground/60"
          >
            <Loader2 class="h-3 w-3 animate-spin text-primary shrink-0" />
            <span>{{ t("mcpServers.loadingCapabilities") }}</span>
          </span>
          <span v-else class="text-xs text-muted-foreground/50">—</span>

          <!-- Capability chips -->
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
          </div>
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
        <Button
          variant="ghost"
          size="icon"
          class="h-9 w-9"
          :disabled="isLoading"
          :aria-label="t('common.edit')"
          @click.stop="emit('edit')"
        >
          <Edit class="w-4 h-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          class="h-9 w-9 text-destructive hover:text-destructive hover:bg-destructive/10"
          :disabled="isLoading"
          :aria-label="t('common.delete')"
          @click.stop="emit('delete')"
        >
          <Trash2 class="w-4 h-4" />
        </Button>
      </div>
    </div>
  </article>
</template>
