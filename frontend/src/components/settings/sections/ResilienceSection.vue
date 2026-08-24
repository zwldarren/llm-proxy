<script setup lang="ts">
import { RefreshCw } from "@lucide/vue";
import { useI18n } from "vue-i18n";
import { SettingsItem, SettingsSection } from "@/components/settings";
import NumberStepper from "@/components/settings/NumberStepper.vue";
import type { AutoSaveState } from "@/composables/useSettingAutoSave";
import { useAutoSaveRefs } from "@/composables/useAutoSaveRefs";
import type { ResilienceConfig } from "@/types/schemas";

const props = defineProps<{
  autoSave: AutoSaveState<ResilienceConfig>;
}>();

const { state, pending, error } = useAutoSaveRefs(props.autoSave);

const { t } = useI18n();
</script>

<template>
  <SettingsSection
    :title="t('resilience.title')"
    :icon="RefreshCw"
    :description="t('resilience.description')"
  >
    <SettingsItem
      :title="t('resilience.maxRetries')"
      :description="t('resilience.maxRetriesDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <NumberStepper
          :model-value="state.max_retries"
          :min="0"
          :max="10"
          @update:model-value="state.max_retries = $event ?? 0"
        />
      </template>
    </SettingsItem>

    <SettingsItem
      :title="t('resilience.maxFallbackAttempts')"
      :description="t('resilience.maxFallbackAttemptsDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <NumberStepper
          :model-value="state.max_fallback_attempts"
          :min="1"
          :max="20"
          @update:model-value="state.max_fallback_attempts = $event ?? 1"
        />
      </template>
    </SettingsItem>
  </SettingsSection>
</template>
