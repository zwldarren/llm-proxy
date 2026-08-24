<script setup lang="ts">
import { useClipboard } from "@vueuse/core";
import { Activity, Check, Clock, Copy, Globe, Shield, Wrench } from "@lucide/vue";
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import LogDetailsIO from "@/components/common/LogDetailsIO.vue";
import LogDetailsMetrics from "@/components/common/LogDetailsMetrics.vue";
import { Badge } from "@/components/ui/badge";
import { Sheet, SheetContent } from "@/components/ui/sheet";
import type { LogRead } from "@/types/schemas";
import { formatDate } from "@/utils/format";

const props = defineProps<{
  log: LogRead | null;
  open: boolean;
}>();

const emit = defineEmits<(e: "update:open", value: boolean) => void>();

const { t } = useI18n();
const { copy, copied: copiedId } = useClipboard({ legacy: true });

const isAuditLog = computed(() => props.log?.log_type === "audit");
const isMcpLog = computed(() => props.log?.log_type === "mcp");
const isWebSearchLog = computed(() => props.log?.log_type === "web_search");
const isToolLog = computed(() => isMcpLog.value || isWebSearchLog.value);

const copyId = async () => {
  if (!props.log?.request_id) return;
  await copy(props.log.request_id);
};
</script>

<template>
  <Sheet :open="open" @update:open="emit('update:open', $event)">
    <SheetContent
      class="w-full sm:max-w-[550px] md:max-w-[700px] lg:max-w-[850px] xl:max-w-[1000px] h-full flex flex-col p-0 gap-0 overflow-hidden border-l border-border/80 bg-card transition-colors duration-300 pb-[env(safe-area-inset-bottom\,0px)]"
    >
      <!-- Header -->
      <div
        class="px-4 sm:px-6 py-4 sm:py-5 border-b border-border/60 bg-muted/10 shrink-0 relative pr-16 flex flex-col gap-3 sm:gap-3.5"
      >
        <div class="flex items-start sm:items-center gap-2.5 sm:gap-3">
          <div
            class="p-2 sm:p-2.5 rounded-md bg-muted border border-border/40 text-muted-foreground shrink-0 flex items-center justify-center"
          >
            <Activity v-if="!isAuditLog && !isToolLog" class="size-4 sm:size-5" />
            <Shield v-else-if="isAuditLog" class="size-4 sm:size-5" />
            <Wrench v-else-if="isMcpLog" class="size-4 sm:size-5" />
            <Globe v-else-if="isWebSearchLog" class="size-4 sm:size-5" />
          </div>

          <div class="flex flex-col min-w-0">
            <h2
              class="text-sm sm:text-base font-semibold text-foreground flex flex-wrap items-center gap-x-2 gap-y-1"
            >
              <span>{{ t("logs.viewDetails") }}</span>
              <span
                class="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider bg-muted/70 px-2 py-0.5 rounded border border-border/40"
              >
                {{
                  isAuditLog
                    ? t("nav.auditLogs")
                    : isMcpLog
                      ? t("nav.mcpLogs")
                      : isWebSearchLog
                        ? t("nav.webSearchLogs")
                        : t("nav.proxyLogs")
                }}
              </span>
            </h2>
            <div class="flex items-center gap-2 text-xs text-muted-foreground mt-1">
              <Clock class="size-3.5 shrink-0" />
              <span class="font-mono truncate">{{ formatDate(log?.timestamp) }}</span>
            </div>
          </div>
        </div>

        <!-- Request ID and HTTP status -->
        <div class="flex flex-wrap items-center gap-2">
          <span class="text-xs text-muted-foreground font-medium shrink-0">ID:</span>
          <Badge
            variant="outline"
            class="font-mono text-xs bg-background text-muted-foreground py-1 sm:py-0.5 px-2.5 flex items-center gap-2 hover:border-muted-foreground/30 transition-colors border-border/70 min-h-8 sm:min-h-0 truncate max-w-full"
          >
            <span class="truncate">{{ log?.request_id }}</span>
            <button
              @click="copyId"
              class="hover:text-foreground transition-colors cursor-pointer shrink-0 flex items-center justify-center min-h-8 min-w-8 sm:min-h-6 sm:min-w-6 p-1 sm:p-1"
              :title="copiedId ? t('common.copied') : t('common.copy')"
            >
              <Check v-if="copiedId" class="w-3.5 h-3.5 text-status-success" />
              <Copy v-else class="w-3.5 h-3.5" />
            </button>
          </Badge>

          <Badge
            v-if="log?.status_code"
            :class="[
              'text-xs font-semibold py-0.5 px-2.5 font-mono border',
              log.status_code >= 200 && log.status_code < 300
                ? 'bg-status-success/15 border-status-success/30 text-status-success'
                : log.status_code >= 400 && log.status_code < 500
                  ? 'bg-status-warning/15 border-status-warning/30 text-status-warning'
                  : 'bg-status-error/15 border-status-error/30 text-status-error',
            ]"
          >
            HTTP {{ log.status_code }}
          </Badge>
        </div>
      </div>

      <!-- Scrollable content area -->
      <div v-if="log" class="flex-1 overflow-y-auto p-4 sm:p-6 space-y-4 sm:space-y-6">
        <!-- Metrics spec sheet -->
        <LogDetailsMetrics :log="log" />

        <!-- Main Content: I/O tabs component -->
        <LogDetailsIO :log="log" />
      </div>
    </SheetContent>
  </Sheet>
</template>
