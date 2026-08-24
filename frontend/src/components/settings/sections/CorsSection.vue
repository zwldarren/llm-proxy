<script setup lang="ts">
import { Globe } from "@lucide/vue";
import { useI18n } from "vue-i18n";
import { SettingsItem, SettingsSection } from "@/components/settings";
import { TagsInputSimple } from "@/components/ui/tags-input";
import type { AutoSaveState } from "@/composables/useSettingAutoSave";
import { useAutoSaveRefs } from "@/composables/useAutoSaveRefs";
import type { CorsConfig } from "@/types/schemas";

const props = defineProps<{
  autoSave: AutoSaveState<CorsConfig>;
}>();

const { state, pending, error } = useAutoSaveRefs(props.autoSave);

const { t } = useI18n();
</script>

<template>
  <SettingsSection :title="t('cors.title')" :icon="Globe" :description="t('cors.description')">
    <SettingsItem
      :title="t('cors.allowedOrigins')"
      :description="t('cors.allowedOriginsDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <TagsInputSimple
          v-model="state.origins"
          :placeholder="t('cors.originPlaceholder')"
          class="max-w-md"
        />
      </template>
    </SettingsItem>
  </SettingsSection>
</template>
