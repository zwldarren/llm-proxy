<script setup lang="ts">
import { useClipboard } from "@vueuse/core";
import { Fingerprint, Link2, Server, Shield, User } from "@lucide/vue";
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useAuditLabels } from "@/composables/useAuditLabels";
import StatusBadge from "@/components/common/StatusBadge.vue";
import { getActor } from "@/utils/format";
import type { LogRead } from "@/types/schemas";

const props = defineProps<{
  log: LogRead;
}>();

const { t } = useI18n();
const { copy, copied: copiedField } = useClipboard({ legacy: true });
const {
  formatEventType,
  formatActionCategory,
  formatResourceType,
  formatAuthMethod,
  formatOutcome,
  outcomeStatus,
} = useAuditLabels();

const hasIntegrity = computed(
  () => props.log.sequence_number != null || props.log.content_hash != null
);

// Truncate a long hash/id for display while preserving the full value for copy/tooltip.
const truncate = (value: string | null | undefined, head = 10, tail = 8): string => {
  if (!value) return "-";
  if (value.length <= head + tail + 1) return value;
  return `${value.slice(0, head)}…${value.slice(-tail)}`;
};
</script>

<template>
  <div class="space-y-6">
    <div class="flex items-center gap-2.5 pb-2.5 border-b border-border/60">
      <div
        class="p-2 rounded-lg min-h-8 min-w-8 bg-muted/60 border border-border/40 text-muted-foreground flex items-center justify-center"
        aria-hidden="true"
      >
        <Shield class="size-4" />
      </div>
      <h3 class="font-semibold text-base text-foreground">{{ t("logs.audit.infoTitle") }}</h3>
    </div>

    <!-- What: Event classification -->
    <section class="space-y-2.5">
      <h4
        class="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground"
      >
        <Link2 class="size-3" /> {{ t("logs.audit.what") }}
      </h4>
      <dl class="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-2.5 text-xs">
        <div class="flex items-center justify-between gap-2 min-w-0">
          <dt class="text-muted-foreground shrink-0">{{ t("logs.audit.eventType") }}</dt>
          <dd class="font-medium text-foreground text-right truncate">
            {{ formatEventType(log.event_type) }}
          </dd>
        </div>
        <div class="flex items-center justify-between gap-2 min-w-0">
          <dt class="text-muted-foreground shrink-0">{{ t("logs.audit.actionCategory") }}</dt>
          <dd class="font-medium text-foreground text-right truncate">
            {{ formatActionCategory(log.action_category) }}
          </dd>
        </div>
        <div class="flex items-center justify-between gap-2 min-w-0">
          <dt class="text-muted-foreground shrink-0">{{ t("logs.audit.resourceType") }}</dt>
          <dd class="font-medium text-foreground text-right truncate">
            {{ formatResourceType(log.resource_type) }}
          </dd>
        </div>
        <div class="flex items-center justify-between gap-2 min-w-0">
          <dt class="text-muted-foreground shrink-0">{{ t("logs.audit.resourceId") }}</dt>
          <dd class="font-mono text-foreground text-right truncate" :title="log.resource_id ?? ''">
            {{ log.resource_id || "—" }}
          </dd>
        </div>
        <div class="flex items-center justify-between gap-2 sm:col-span-2 min-w-0">
          <dt class="text-muted-foreground shrink-0">{{ t("logs.audit.outcome") }}</dt>
          <dd class="flex items-center gap-2">
            <StatusBadge v-if="log.outcome" variant="status" :status="outcomeStatus(log.outcome)">
              {{ formatOutcome(log.outcome) }}
            </StatusBadge>
            <span v-else class="text-muted-foreground">—</span>
          </dd>
        </div>
      </dl>
    </section>

    <!-- Who: Actor & identity -->
    <section class="space-y-2.5">
      <h4
        class="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground"
      >
        <User class="size-3" /> {{ t("logs.audit.who") }}
      </h4>
      <dl class="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-2.5 text-xs">
        <div class="flex items-center justify-between gap-2 min-w-0">
          <dt class="text-muted-foreground shrink-0">{{ t("logs.actor") }}</dt>
          <dd class="font-medium text-foreground text-right truncate" :title="getActor(log)">
            {{ getActor(log) }}
          </dd>
        </div>
        <div class="flex items-center justify-between gap-2 min-w-0">
          <dt class="text-muted-foreground shrink-0">{{ t("logs.audit.authMethod") }}</dt>
          <dd class="text-right">
            <Badge v-if="log.auth_method" variant="outline" class="font-mono text-[11px]">
              {{ formatAuthMethod(log.auth_method) }}
            </Badge>
            <span v-else class="text-muted-foreground">—</span>
          </dd>
        </div>
        <div class="flex items-center justify-between gap-2 min-w-0">
          <dt class="text-muted-foreground shrink-0">{{ t("logs.ipAddress") }}</dt>
          <dd class="font-mono text-foreground text-right truncate">{{ log.client_ip || "—" }}</dd>
        </div>
        <div class="flex items-center justify-between gap-2 min-w-0">
          <dt class="text-muted-foreground shrink-0">{{ t("logs.audit.sessionId") }}</dt>
          <dd class="font-mono text-foreground text-right truncate" :title="log.session_id ?? ''">
            {{ truncate(log.session_id) }}
          </dd>
        </div>
        <div class="flex items-start justify-between gap-2 sm:col-span-2 min-w-0">
          <dt class="text-muted-foreground shrink-0 pt-0.5">{{ t("logs.audit.userAgent") }}</dt>
          <dd
            class="font-mono text-muted-foreground text-right text-[11px] break-all min-w-0 flex-1"
            :title="log.user_agent ?? ''"
          >
            {{ log.user_agent || "—" }}
          </dd>
        </div>
      </dl>
    </section>

    <!-- Where: Service -->
    <section class="space-y-2.5">
      <h4
        class="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground"
      >
        <Server class="size-3" /> {{ t("logs.audit.where") }}
      </h4>
      <dl class="grid grid-cols-1 sm:grid-cols-3 gap-x-4 gap-y-2.5 text-xs">
        <div class="flex items-center justify-between gap-2 min-w-0">
          <dt class="text-muted-foreground shrink-0">{{ t("logs.audit.serviceName") }}</dt>
          <dd class="font-medium text-foreground text-right truncate">
            {{ log.service_name || "—" }}
          </dd>
        </div>
        <div class="flex items-center justify-between gap-2 min-w-0">
          <dt class="text-muted-foreground shrink-0">{{ t("logs.audit.serverHostname") }}</dt>
          <dd class="font-mono text-foreground text-right truncate">
            {{ log.server_hostname || "—" }}
          </dd>
        </div>
        <div class="flex items-center justify-between gap-2 min-w-0">
          <dt class="text-muted-foreground shrink-0">{{ t("logs.audit.serviceVersion") }}</dt>
          <dd class="font-mono text-foreground text-right truncate">
            {{ log.service_version || "—" }}
          </dd>
        </div>
      </dl>
    </section>

    <!-- Integrity: hash chain -->
    <section v-if="hasIntegrity" class="space-y-2.5">
      <h4
        class="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground"
      >
        <Fingerprint class="size-3" /> {{ t("logs.audit.integrity") }}
      </h4>
      <dl class="space-y-2 text-xs">
        <div class="flex items-center justify-between gap-2 min-w-0">
          <dt class="text-muted-foreground shrink-0">{{ t("logs.audit.sequenceNumber") }}</dt>
          <dd class="font-mono text-foreground">#{{ log.sequence_number ?? "—" }}</dd>
        </div>
        <div class="flex items-center justify-between gap-2 min-w-0">
          <dt class="text-muted-foreground shrink-0">{{ t("logs.audit.contentHash") }}</dt>
          <dd class="flex items-center gap-1.5 min-w-0">
            <span class="font-mono text-muted-foreground truncate" :title="log.content_hash ?? ''">
              {{ truncate(log.content_hash, 12, 10) }}
            </span>
            <Button
              v-if="log.content_hash"
              variant="ghost"
              size="icon"
              class="h-6 w-6 shrink-0"
              :aria-label="t('common.copy')"
              @click="copy(log.content_hash)"
            >
              <span v-if="copiedField" class="text-status-success text-[11px]">✓</span>
              <span v-else class="text-[11px]">⧉</span>
            </Button>
          </dd>
        </div>
        <div class="flex items-center justify-between gap-2 min-w-0">
          <dt class="text-muted-foreground shrink-0">{{ t("logs.audit.previousHash") }}</dt>
          <dd class="flex items-center gap-1.5 min-w-0">
            <span class="font-mono text-muted-foreground truncate" :title="log.previous_hash ?? ''">
              {{ truncate(log.previous_hash, 12, 10) }}
            </span>
            <Button
              v-if="log.previous_hash"
              variant="ghost"
              size="icon"
              class="h-6 w-6 shrink-0"
              :aria-label="t('common.copy')"
              @click="copy(log.previous_hash)"
            >
              <span v-if="copiedField" class="text-status-success text-[11px]">✓</span>
              <span v-else class="text-[11px]">⧉</span>
            </Button>
          </dd>
        </div>
      </dl>
    </section>
  </div>
</template>
