<script setup lang="ts">
import { ShieldCheck } from "@lucide/vue";
import { useI18n } from "vue-i18n";
import { SettingsItem, SettingsSection } from "@/components/settings";
import NumberStepper from "@/components/settings/NumberStepper.vue";
import { Switch } from "@/components/ui/switch";
import { Alert, AlertDescription } from "@/components/ui/alert";
import type { AutoSaveState } from "@/composables/useSettingAutoSave";
import { useAutoSaveRefs } from "@/composables/useAutoSaveRefs";
import type { SecurityConfig } from "@/types/schemas";
import { DEFAULT_SECURITY } from "@/constants/defaults";

const props = defineProps<{
  autoSave: AutoSaveState<SecurityConfig>;
}>();

const { state, pending, error } = useAutoSaveRefs(props.autoSave);

const { t } = useI18n();
</script>

<template>
  <SettingsSection
    :title="t('security.title')"
    :icon="ShieldCheck"
    :description="t('security.description')"
  >
    <SettingsItem
      :title="t('security.maxFailedLoginAttempts')"
      :description="t('security.maxFailedLoginAttemptsDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <NumberStepper
          :model-value="state.max_failed_login_attempts"
          :min="1"
          @update:model-value="
            state.max_failed_login_attempts = $event ?? DEFAULT_SECURITY.max_failed_login_attempts
          "
        />
      </template>
    </SettingsItem>

    <SettingsItem
      :title="t('security.lockoutDurationSeconds')"
      :description="t('security.lockoutDurationSecondsDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <NumberStepper
          :model-value="state.lockout_duration_seconds"
          :min="1"
          :suffix="t('security.seconds')"
          @update:model-value="
            state.lockout_duration_seconds = $event ?? DEFAULT_SECURITY.lockout_duration_seconds
          "
        />
      </template>
    </SettingsItem>

    <SettingsItem
      :title="t('security.maxFailedApiKeyAttempts')"
      :description="t('security.maxFailedApiKeyAttemptsDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <NumberStepper
          :model-value="state.max_failed_api_key_attempts"
          :min="1"
          @update:model-value="
            state.max_failed_api_key_attempts =
              $event ?? DEFAULT_SECURITY.max_failed_api_key_attempts
          "
        />
      </template>
    </SettingsItem>

    <SettingsItem
      :title="t('security.apiKeyLockoutDurationSeconds')"
      :description="t('security.apiKeyLockoutDurationSecondsDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <NumberStepper
          :model-value="state.api_key_lockout_duration_seconds"
          :min="1"
          :suffix="t('security.seconds')"
          @update:model-value="
            state.api_key_lockout_duration_seconds =
              $event ?? DEFAULT_SECURITY.api_key_lockout_duration_seconds
          "
        />
      </template>
    </SettingsItem>

    <SettingsItem
      :title="t('security.authFailureDelayMs')"
      :description="t('security.authFailureDelayMsDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <NumberStepper
          :model-value="state.auth_failure_delay_ms"
          :min="0"
          :step="50"
          :suffix="t('security.ms')"
          @update:model-value="
            state.auth_failure_delay_ms = $event ?? DEFAULT_SECURITY.auth_failure_delay_ms
          "
        />
      </template>
    </SettingsItem>

    <SettingsItem
      :title="t('security.rateLimitDisabled')"
      :description="t('security.rateLimitDisabledDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <Switch v-model="state.rate_limit_disabled" />
      </template>
    </SettingsItem>

    <Alert v-if="state.rate_limit_disabled" variant="destructive" class="mt-2">
      <AlertDescription>
        {{ t("security.rateLimitDisabledWarning") }}
      </AlertDescription>
    </Alert>

    <SettingsItem
      :title="t('security.redisRateLimitFailClosed')"
      :description="t('security.redisRateLimitFailClosedDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <Switch v-model="state.redis_rate_limit_fail_closed" />
      </template>
    </SettingsItem>

    <SettingsItem
      :title="t('security.hstsEnabled')"
      :description="t('security.hstsEnabledDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <Switch v-model="state.hsts_enabled" />
      </template>
    </SettingsItem>

    <SettingsItem
      v-if="state.hsts_enabled"
      :title="t('security.hstsMaxAge')"
      :description="t('security.hstsMaxAgeDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <NumberStepper
          :model-value="state.hsts_max_age"
          :min="0"
          :step="86400"
          :suffix="t('security.seconds')"
          @update:model-value="state.hsts_max_age = $event ?? DEFAULT_SECURITY.hsts_max_age"
        />
      </template>
    </SettingsItem>

    <SettingsItem
      :title="t('security.maxRequestBodySize')"
      :description="t('security.maxRequestBodySizeDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <NumberStepper
          :model-value="Math.floor(state.max_request_body_size_bytes / (1024 * 1024))"
          :min="1"
          suffix="MB"
          @update:model-value="
            state.max_request_body_size_bytes =
              ($event ?? DEFAULT_SECURITY.max_request_body_size_bytes / (1024 * 1024)) * 1024 * 1024
          "
        />
      </template>
    </SettingsItem>
  </SettingsSection>
</template>
