<script setup lang="ts">
/**
 * Run inspector — the drawer a tray specimen opens, shared by Chat and Images.
 * Shows the exact wire payload plus real run telemetry. A failed run is held
 * here as an inspectable tableau: status, error, and the payload that caused
 * it stay frozen until dismissed. Runs restored from a previous session carry
 * telemetry only — their payload was never persisted.
 */
import { Activity, AlertTriangle, X } from "@lucide/vue";
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import JsonViewer from "@/components/common/JsonViewer.vue";
import StatusBadge from "@/components/common/StatusBadge.vue";
import { Button } from "@/components/ui/button";
import type { ChatRunStatus } from "@/types/runs";
import { formatClock, formatLatency } from "@/utils/runs";

const props = withDefaults(
  defineProps<{
    open: boolean;
    /** 1-based position of the run in the tray. */
    runNumber: number;
    status: ChatRunStatus;
    endpoint: string;
    model: string;
    startedAt: number;
    latencyMs?: number;
    responseChars?: number;
    errorMessage?: string;
    /** Null when the run was restored from a previous session (stubs persist
     * telemetry only, never payloads). */
    payload: Record<string, unknown> | null;
    /** Additional telemetry rows (e.g. image size / quality / count). */
    extraRows?: { label: string; value: string }[];
  }>(),
  { extraRows: () => [] }
);

const emit = defineEmits<{
  close: [];
}>();

const { t } = useI18n();

const close = () => emit("close");

const statusVariant = computed(() => {
  switch (props.status) {
    case "ok":
      return "success" as const;
    case "error":
      return "error" as const;
    case "stopped":
      return "warning" as const;
    default:
      return "unknown" as const;
  }
});

const statusLabel = computed(() => {
  switch (props.status) {
    case "ok":
      return t("playground.statusOk");
    case "error":
      return t("playground.statusError");
    case "stopped":
      return t("playground.statusStopped");
    default:
      return t("playground.statusStreaming");
  }
});
</script>

<template>
  <!-- Inspector overlay (mobile) -->
  <transition
    enter-active-class="settings-overlay-enter-active"
    leave-active-class="settings-overlay-leave-active"
  >
    <div
      v-if="open"
      class="fixed inset-0 overlay-light backdrop-blur-[2px] z-40 lg:hidden"
      @click="close"
    />
  </transition>

  <!-- Inspector panel -->
  <transition
    enter-active-class="settings-panel-enter-active"
    leave-active-class="settings-panel-leave-active"
  >
    <aside
      v-if="open"
      class="fixed right-0 top-0 bottom-0 w-80 z-40 lg:relative lg:z-auto shrink-0 border-l border-border/50 bg-card overflow-hidden will-change-transform"
      :aria-label="t('playground.inspector')"
    >
      <div class="h-full flex flex-col">
        <!-- Panel header -->
        <div class="p-4 border-b border-border/50 lg:pt-4 pt-[calc(env(safe-area-inset-top)+1rem)]">
          <div class="flex items-center justify-between gap-2">
            <div class="flex items-center gap-2 min-w-0">
              <div class="icon-container p-1.5">
                <Activity class="w-4 h-4 text-primary" />
              </div>
              <h3 class="font-semibold text-sm font-mono truncate">
                {{ t("playground.runNumber", { n: String(runNumber).padStart(2, "0") }) }}
              </h3>
              <StatusBadge variant="status" :status="statusVariant">{{ statusLabel }}</StatusBadge>
            </div>
            <Button variant="ghost" size="icon" class="h-10 w-10 shrink-0" @click="close">
              <X class="w-4 h-4" />
            </Button>
          </div>
        </div>

        <!-- Inspector content -->
        <div class="flex-1 overflow-y-auto p-4 space-y-5">
          <!-- Telemetry: every value in IBM Plex Mono, tabular -->
          <dl class="space-y-2.5">
            <div class="flex items-center justify-between gap-3">
              <dt class="text-xs text-muted-foreground shrink-0">
                {{ t("playground.endpoint") }}
              </dt>
              <dd class="flex items-center gap-1.5 min-w-0">
                <StatusBadge variant="http" http-method="POST">POST</StatusBadge>
                <span class="text-code-xs text-foreground truncate" :title="endpoint">
                  {{ endpoint }}
                </span>
              </dd>
            </div>
            <div class="flex items-center justify-between gap-3">
              <dt class="text-xs text-muted-foreground">{{ t("playground.model") }}</dt>
              <dd class="text-code-xs text-foreground truncate" :title="model">
                {{ model }}
              </dd>
            </div>
            <div class="flex items-center justify-between gap-3">
              <dt class="text-xs text-muted-foreground">{{ t("playground.started") }}</dt>
              <dd class="text-data-xs text-foreground">
                {{ formatClock(startedAt) }}
              </dd>
            </div>
            <div class="flex items-center justify-between gap-3">
              <dt class="text-xs text-muted-foreground">{{ t("playground.duration") }}</dt>
              <dd class="text-data-xs text-foreground">
                {{ formatLatency(latencyMs) }}
              </dd>
            </div>
            <div v-if="responseChars !== undefined" class="flex items-center justify-between gap-3">
              <dt class="text-xs text-muted-foreground">{{ t("playground.responseSize") }}</dt>
              <dd class="text-data-xs text-foreground">
                {{ t("playground.chars", { n: responseChars }) }}
              </dd>
            </div>
            <div
              v-for="row in extraRows"
              :key="row.label"
              class="flex items-center justify-between gap-3"
            >
              <dt class="text-xs text-muted-foreground">{{ row.label }}</dt>
              <dd class="text-code-xs text-foreground truncate" :title="row.value">
                {{ row.value }}
              </dd>
            </div>
          </dl>

          <!-- Held failure: the exact error, frozen with its payload below -->
          <div
            v-if="status === 'error' && errorMessage"
            class="rounded-lg border border-status-error/30 bg-status-error/5 p-3 space-y-1.5"
            role="alert"
          >
            <div class="flex items-center gap-2 text-status-error">
              <AlertTriangle class="w-3.5 h-3.5 shrink-0" />
              <span class="text-xs font-medium">{{ t("playground.runFailed") }}</span>
            </div>
            <p class="text-code-xs text-status-error/90 break-words leading-relaxed">
              {{ errorMessage }}
            </p>
          </div>

          <!-- Request payload -->
          <div class="space-y-2">
            <span class="text-xs font-medium text-muted-foreground">
              {{ t("playground.requestPayload") }}
            </span>
            <JsonViewer v-if="payload" :data="payload" :deep="4" max-height="none" />
            <p v-else class="text-code-xs text-muted-foreground/70">
              {{ t("playground.payloadNotRetained") }}
            </p>
          </div>
        </div>
      </div>
    </aside>
  </transition>
</template>
