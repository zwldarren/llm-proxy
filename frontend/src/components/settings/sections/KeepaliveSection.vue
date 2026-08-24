<script setup lang="ts">
import { HeartPulse } from "@lucide/vue";
import { useI18n } from "vue-i18n";
import { SettingsItem, SettingsSection } from "@/components/settings";
import NumberStepper from "@/components/settings/NumberStepper.vue";
import { Switch } from "@/components/ui/switch";
import type { AutoSaveState } from "@/composables/useSettingAutoSave";
import { useAutoSaveRefs } from "@/composables/useAutoSaveRefs";
import type { KeepaliveConfig } from "@/types/schemas";
import { DEFAULT_KEEPALIVE } from "@/constants/defaults";

const props = defineProps<{
  autoSave: AutoSaveState<KeepaliveConfig>;
}>();

const { state, pending, error } = useAutoSaveRefs(props.autoSave);

const { t } = useI18n();
</script>

<template>
  <SettingsSection
    :title="t('keepalive.title')"
    :icon="HeartPulse"
    :description="t('keepalive.description')"
  >
    <SettingsItem
      :title="t('keepalive.enabled')"
      :description="t('keepalive.enabledDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <Switch v-model="state.enabled" />
      </template>
    </SettingsItem>

    <SettingsItem
      v-if="state.enabled"
      :title="t('keepalive.graceSeconds')"
      :description="t('keepalive.graceSecondsDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <NumberStepper
          :model-value="state.grace_seconds"
          :min="1"
          :suffix="t('keepalive.seconds')"
          @update:model-value="state.grace_seconds = $event ?? DEFAULT_KEEPALIVE.grace_seconds"
        />
      </template>
    </SettingsItem>

    <SettingsItem
      v-if="state.enabled"
      :title="t('keepalive.intervalSeconds')"
      :description="t('keepalive.intervalSecondsDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <NumberStepper
          :model-value="state.interval_seconds"
          :min="1"
          :suffix="t('keepalive.seconds')"
          @update:model-value="
            state.interval_seconds = $event ?? DEFAULT_KEEPALIVE.interval_seconds
          "
        />
      </template>
    </SettingsItem>
  </SettingsSection>
</template>
