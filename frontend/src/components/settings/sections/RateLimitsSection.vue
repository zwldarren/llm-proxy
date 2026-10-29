<script setup lang="ts">
import { Gauge } from "@lucide/vue";
import { computed, ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import { SettingsItem, SettingsSection } from "@/components/settings";
import { Input } from "@/components/ui/input";
import type { AutoSaveState } from "@/composables/useSettingAutoSave";
import { useAutoSaveRefs } from "@/composables/useAutoSaveRefs";
import type { RateLimitsConfig } from "@/types/schemas";

const props = defineProps<{
  autoSave: AutoSaveState<RateLimitsConfig>;
}>();

const { state, pending, error } = useAutoSaveRefs(props.autoSave);

const { t } = useI18n();

const BUCKETS = [
  "auth.login",
  "auth.setup",
  "auth.setup_status",
  "auth.capabilities",
  "auth.mfa",
  "auth.step_up",
  "auth.invite_accept",
  "auth.invite_status",
  "auth.passkey",
  "auth.oauth",
  "auth.email_verify",
  "auth.password_reset",
  "auth.password_reset_confirm",
  "account.sensitive",
] as const;

const bucketLabels: Record<(typeof BUCKETS)[number], { title: string; description: string }> = {
  "auth.login": {
    title: "rateLimits.login",
    description: "rateLimits.loginDescription",
  },
  "auth.setup": {
    title: "rateLimits.setup",
    description: "rateLimits.setupDescription",
  },
  "auth.setup_status": {
    title: "rateLimits.setupStatus",
    description: "rateLimits.setupStatusDescription",
  },
  "auth.capabilities": {
    title: "rateLimits.capabilities",
    description: "rateLimits.capabilitiesDescription",
  },
  "auth.mfa": {
    title: "rateLimits.mfa",
    description: "rateLimits.mfaDescription",
  },
  "auth.step_up": {
    title: "rateLimits.stepUp",
    description: "rateLimits.stepUpDescription",
  },
  "auth.invite_accept": {
    title: "rateLimits.inviteAccept",
    description: "rateLimits.inviteAcceptDescription",
  },
  "auth.invite_status": {
    title: "rateLimits.inviteStatus",
    description: "rateLimits.inviteStatusDescription",
  },
  "auth.passkey": {
    title: "rateLimits.passkey",
    description: "rateLimits.passkeyDescription",
  },
  "auth.oauth": {
    title: "rateLimits.oauth",
    description: "rateLimits.oauthDescription",
  },
  "auth.email_verify": {
    title: "rateLimits.emailVerify",
    description: "rateLimits.emailVerifyDescription",
  },
  "auth.password_reset": {
    title: "rateLimits.passwordReset",
    description: "rateLimits.passwordResetDescription",
  },
  "auth.password_reset_confirm": {
    title: "rateLimits.passwordResetConfirm",
    description: "rateLimits.passwordResetConfirmDescription",
  },
  "account.sensitive": {
    title: "rateLimits.accountSensitive",
    description: "rateLimits.accountSensitiveDescription",
  },
};

// The backend rejects zero (it would make every request fail), so the UI must
// accept exactly what the API accepts: a positive count and a valid period.
const LIMIT_PATTERN = /^[1-9]\d*\/(second|minute|hour|day|\d+second|\d+minute|\d+hour)$/;

// Local drafts let the user type freely while only valid values are committed
// to state (and thus persisted). Invalid values are shown in the input but
// never saved to the backend.
const drafts = ref<Record<string, string>>({});

watch(
  () => state.value.limits,
  (limits) => {
    drafts.value = { ...limits };
  },
  { immediate: true, deep: true }
);

function onLimitInput(bucket: string, value: string | number | null) {
  const v = String(value ?? "").trim();
  drafts.value[bucket] = v;
  if (LIMIT_PATTERN.test(v)) {
    state.value.limits[bucket] = v;
  }
}

const invalidBuckets = computed(() =>
  BUCKETS.filter((bucket) => {
    const value = drafts.value[bucket] ?? "";
    return !value || !LIMIT_PATTERN.test(value.trim());
  })
);
</script>

<template>
  <SettingsSection
    :title="t('rateLimits.title')"
    :icon="Gauge"
    :description="t('rateLimits.description')"
  >
    <SettingsItem
      v-for="bucket in BUCKETS"
      :key="bucket"
      :title="t(bucketLabels[bucket].title)"
      :description="t(bucketLabels[bucket].description)"
      :loading="pending"
      :error="invalidBuckets.includes(bucket) ? t('rateLimits.invalidFormat') : error"
    >
      <template #action>
        <Input
          :model-value="drafts[bucket] ?? state.limits[bucket] ?? ''"
          :aria-label="t(bucketLabels[bucket].title)"
          :placeholder="t('rateLimits.placeholder')"
          class="font-mono text-sm w-32 bg-background"
          @update:model-value="onLimitInput(bucket, $event)"
        />
      </template>
    </SettingsItem>
  </SettingsSection>
</template>
