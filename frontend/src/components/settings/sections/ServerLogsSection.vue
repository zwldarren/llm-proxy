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
    :title="t('settings.serverLogs')"
    :icon="Server"
    :description="t('settings.serverLogsDescription')"
  >
    <SettingsItem
      v-if="isAdmin"
      :icon="FileText"
      :title="t('settings.logInputOutput')"
      :description="t('settings.logInputOutputDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <Switch v-model="state.log_input_output" />
      </template>
    </SettingsItem>

    <SettingsItem
      v-if="isAdmin"
      :title="t('settings.logRetentionDays')"
      :description="t('settings.setZeroToKeepIndefinitely')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <NumberStepper
          :model-value="state.log_retention_days"
          :min="0"
          :suffix="t('settings.days')"
          @update:model-value="state.log_retention_days = $event ?? 0"
        />
      </template>
    </SettingsItem>

    <SettingsItem
      v-if="isAdmin"
      :title="t('settings.maskSensitiveData')"
      :description="t('settings.maskSensitiveDataDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <Switch v-model="state.mask_sensitive_data" />
      </template>
    </SettingsItem>

    <SettingsItem
      v-if="isAdmin"
      :title="t('settings.samplingRate')"
      :description="t('settings.samplingRateDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <NumberInput
          :model-value="state.sampling_rate"
          min="0"
          max="1"
          step="0.1"
          class="font-mono text-sm w-28 bg-background"
          @update:model-value="state.sampling_rate = $event ?? 1"
        />
      </template>
    </SettingsItem>

    <SettingsItem
      v-if="isAdmin"
      :title="t('settings.auditSamplingRate')"
      :description="t('settings.auditSamplingRateDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <NumberInput
          v-model="state.audit_sampling_rate"
          min="0"
          max="1"
          step="0.1"
          :placeholder="t('settings.inheritPlaceholder')"
          class="font-mono text-sm w-28 bg-background"
        />
      </template>
    </SettingsItem>

    <SettingsItem
      v-if="isAdmin"
      :title="t('settings.auditRetentionDays')"
      :description="t('settings.auditRetentionDaysDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <NumberInput
          v-model="state.audit_retention_days"
          min="0"
          :placeholder="t('settings.inheritPlaceholder')"
          class="font-mono text-sm w-28 bg-background"
        />
      </template>
    </SettingsItem>

    <SettingsItem
      v-if="isAdmin"
      :title="t('settings.sensitiveKeys')"
      :description="t('settings.sensitiveKeysDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <Input
          v-model="state.sensitive_keys"
          :placeholder="t('settings.sensitiveKeysPlaceholder')"
          class="font-mono text-sm w-64 bg-background"
        />
      </template>
    </SettingsItem>

    <SettingsItem
      :icon="Trash2"
      :title="t('settings.cleanOldLogs')"
      :description="t('settings.cleanOldLogsDescription')"
    >
      <template #action>
        <Button variant="destructive" size="sm" @click="emit('cleanup')">
          <Trash2 class="size-4 mr-2" />
          {{ t("settings.cleanupLogs") }}
        </Button>
      </template>
    </SettingsItem>
  </SettingsSection>
</template>
