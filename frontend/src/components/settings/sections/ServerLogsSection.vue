<script setup lang="ts">
import { FileText, Server, Trash2 } from "@lucide/vue";
import { useI18n } from "vue-i18n";
import { SettingsItem, SettingsSection } from "@/components/settings";
import NumberStepper from "@/components/settings/NumberStepper.vue";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Switch } from "@/components/ui/switch";
import type { AutoSaveState } from "@/composables/useSettingAutoSave";
import { useAutoSaveRefs } from "@/composables/useAutoSaveRefs";
import { BYTES_PER_MB, DEFAULT_LOGGING } from "@/constants/defaults";
import type { LoggingConfig } from "@/types/schemas";

const props = defineProps<{
  autoSave: AutoSaveState<LoggingConfig>;
  isAdmin: boolean;
}>();

const emit = defineEmits<{
  (e: "cleanup"): void;
}>();

const { state, pending, error } = useAutoSaveRefs(props.autoSave);

const { t } = useI18n();
</script>

<template>
  <SettingsSection
    :error="error"
    :title="t('settings.serverLogs')"
    :icon="Server"
    :description="t('settings.serverLogsDescription')"
  >
    <section v-if="isAdmin" aria-labelledby="log-capture-heading" class="divide-y divide-border/45">
      <h3
        id="log-capture-heading"
        class="px-5 pb-3 pt-4 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground"
      >
        {{ t("settings.logCaptureGroup") }}
      </h3>

      <SettingsItem
        :icon="FileText"
        :title="t('settings.logInputOutput')"
        :description="t('settings.logInputOutputDescription')"
        :loading="pending"
      >
        <template #action>
          <Switch v-model="state.log_input_output" :aria-label="t('settings.logInputOutput')" />
        </template>
      </SettingsItem>

      <SettingsItem
        :title="t('settings.logRawStream')"
        :description="t('settings.logRawStreamDescription')"
        :loading="pending"
      >
        <template #action>
          <Switch v-model="state.log_raw_stream" :aria-label="t('settings.logRawStream')" />
        </template>
      </SettingsItem>

      <SettingsItem
        :title="t('settings.maskSensitiveData')"
        :description="t('settings.maskSensitiveDataDescription')"
        :loading="pending"
      >
        <template #action>
          <Switch
            v-model="state.mask_sensitive_data"
            :aria-label="t('settings.maskSensitiveData')"
          />
        </template>
      </SettingsItem>

      <SettingsItem
        :title="t('settings.sensitiveKeys')"
        :description="t('settings.sensitiveKeysDescription')"
        :loading="pending"
      >
        <template #action>
          <Input
            v-model="state.sensitive_keys"
            :aria-label="t('settings.sensitiveKeys')"
            :placeholder="t('settings.sensitiveKeysPlaceholder')"
            class="font-mono text-sm w-64 bg-background"
          />
        </template>
      </SettingsItem>
    </section>

    <section aria-labelledby="log-retention-heading" class="divide-y divide-border/45">
      <h3
        id="log-retention-heading"
        class="px-5 pb-3 pt-4 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground"
      >
        {{ t("settings.logRetentionGroup") }}
      </h3>

      <SettingsItem
        v-if="isAdmin"
        :title="t('settings.logRetentionDays')"
        :description="t('settings.setZeroToKeepIndefinitely')"
        :loading="pending"
      >
        <template #action>
          <NumberStepper
            :model-value="state.log_retention_days"
            :min="0"
            :suffix="t('settings.days')"
            :aria-label="t('settings.logRetentionDays')"
            @update:model-value="state.log_retention_days = $event ?? 0"
          />
        </template>
      </SettingsItem>

      <SettingsItem
        v-if="isAdmin"
        :title="t('settings.batchRetentionDays')"
        :description="t('settings.batchRetentionDaysDescription')"
        :loading="pending"
      >
        <template #action>
          <NumberStepper
            :model-value="state.batch_retention_days"
            :min="0"
            :suffix="t('settings.days')"
            :aria-label="t('settings.batchRetentionDays')"
            @update:model-value="state.batch_retention_days = $event ?? 0"
          />
        </template>
      </SettingsItem>

      <SettingsItem
        :icon="Trash2"
        :title="t('settings.cleanOldLogs')"
        :description="t('settings.cleanOldLogsDescription')"
      >
        <template #action>
          <Button
            variant="destructive"
            size="sm"
            class="h-auto min-h-9 max-w-full whitespace-normal py-2 text-center max-sm:min-h-11"
            @click="emit('cleanup')"
          >
            <Trash2 class="size-4" />
            {{ t("settings.cleanupLogs") }}
          </Button>
        </template>
      </SettingsItem>
    </section>

    <section v-if="isAdmin" aria-labelledby="log-audit-heading" class="divide-y divide-border/45">
      <h3
        id="log-audit-heading"
        class="px-5 pb-3 pt-4 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground"
      >
        {{ t("settings.logAuditGroup") }}
      </h3>

      <SettingsItem
        :title="t('settings.samplingRate')"
        :description="t('settings.samplingRateDescription')"
        :loading="pending"
      >
        <template #action>
          <NumberInput
            :model-value="state.sampling_rate"
            min="0"
            max="1"
            step="0.1"
            :aria-label="t('settings.samplingRate')"
            class="font-mono text-sm w-28 bg-background"
            @update:model-value="state.sampling_rate = $event ?? 1"
          />
        </template>
      </SettingsItem>

      <SettingsItem
        :title="t('settings.maxLoggedBodySize')"
        :description="t('settings.maxLoggedBodySizeDescription')"
        :loading="pending"
      >
        <template #action>
          <NumberStepper
            :model-value="Math.round((state.max_logged_body_bytes / BYTES_PER_MB) * 10) / 10"
            :min="0"
            :step="1"
            :suffix="t('security.megabytes')"
            :aria-label="t('settings.maxLoggedBodySize')"
            @update:model-value="
              state.max_logged_body_bytes =
                ($event ?? DEFAULT_LOGGING.max_logged_body_bytes / BYTES_PER_MB) * BYTES_PER_MB
            "
          />
        </template>
      </SettingsItem>

      <SettingsItem
        :title="t('settings.auditSamplingRate')"
        :description="t('settings.auditSamplingRateDescription')"
        :loading="pending"
      >
        <template #action>
          <NumberInput
            v-model="state.audit_sampling_rate"
            min="0"
            max="1"
            step="0.1"
            :aria-label="t('settings.auditSamplingRate')"
            :placeholder="t('settings.inheritPlaceholder')"
            class="font-mono text-sm w-28 bg-background"
          />
        </template>
      </SettingsItem>

      <SettingsItem
        :title="t('settings.auditRetentionDays')"
        :description="t('settings.auditRetentionDaysDescription')"
        :loading="pending"
      >
        <template #action>
          <NumberInput
            v-model="state.audit_retention_days"
            min="0"
            :aria-label="t('settings.auditRetentionDays')"
            :placeholder="t('settings.inheritPlaceholder')"
            class="font-mono text-sm w-28 bg-background"
          />
        </template>
      </SettingsItem>
    </section>
  </SettingsSection>
</template>
