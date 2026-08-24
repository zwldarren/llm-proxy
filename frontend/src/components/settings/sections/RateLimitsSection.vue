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

const BUCKETS = ["auth.login", "auth.setup", "auth.setup_status"] as const;

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
};

// Allow 0 as a valid limit value (e.g. "0/minute") to block a bucket entirely.
const LIMIT_PATTERN = /^\d+\/(second|minute|hour|day|\d+second|\d+minute|\d+hour)$/;

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
          :placeholder="t('rateLimits.placeholder')"
          class="font-mono text-sm w-32 bg-background"
          @update:model-value="onLimitInput(bucket, $event)"
        />
      </template>
    </SettingsItem>
  </SettingsSection>
</template>
