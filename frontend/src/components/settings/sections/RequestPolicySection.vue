<script setup lang="ts">
import { Settings } from "@lucide/vue";
import { useI18n } from "vue-i18n";
import { SettingsItem, SettingsSection } from "@/components/settings";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { AutoSaveState } from "@/composables/useSettingAutoSave";
import { useAutoSaveRefs } from "@/composables/useAutoSaveRefs";
import type { RequestPolicyConfig } from "@/types/schemas";

const props = defineProps<{
  autoSave: AutoSaveState<RequestPolicyConfig>;
}>();

const { state, pending, error } = useAutoSaveRefs(props.autoSave);

const { t } = useI18n();
</script>

<template>
  <SettingsSection
    :title="t('requestPolicy.title')"
    :icon="Settings"
    :description="t('requestPolicy.description')"
  >
    <SettingsItem
      :title="t('requestPolicy.unknownFieldsPolicy')"
      :description="t('requestPolicy.unknownFieldsPolicyDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <Select v-model="state.unknown_fields_policy">
          <SelectTrigger class="w-44">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="passthrough">
              {{ t("requestPolicy.unknownFieldsPolicyPassthrough") }}
            </SelectItem>
            <SelectItem value="ignore">
              {{ t("requestPolicy.unknownFieldsPolicyIgnore") }}
            </SelectItem>
            <SelectItem value="error">
              {{ t("requestPolicy.unknownFieldsPolicyError") }}
            </SelectItem>
          </SelectContent>
        </Select>
      </template>
    </SettingsItem>

    <SettingsItem
      :title="t('requestPolicy.unsupportedBlockPolicy')"
      :description="t('requestPolicy.unsupportedBlockPolicyDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <Select v-model="state.unsupported_block_policy">
          <SelectTrigger class="w-44">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="drop">
              {{ t("requestPolicy.unsupportedBlockPolicyDrop") }}
            </SelectItem>
            <SelectItem value="degrade">
              {{ t("requestPolicy.unsupportedBlockPolicyDegrade") }}
            </SelectItem>
            <SelectItem value="error">
              {{ t("requestPolicy.unsupportedBlockPolicyError") }}
            </SelectItem>
          </SelectContent>
        </Select>
      </template>
    </SettingsItem>
  </SettingsSection>
</template>
