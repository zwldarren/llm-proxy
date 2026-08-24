<script setup lang="ts">
import { Palette } from "@lucide/vue";
import { useI18n } from "vue-i18n";
import { SettingsItem, SettingsSection } from "@/components/settings";
import ThemeSelector from "@/components/settings/ThemeSelector.vue";
import LanguageSelector from "@/components/settings/LanguageSelector.vue";

const { t } = useI18n();
</script>

<template>
  <SettingsSection
    :title="t('settings.preference')"
    :icon="Palette"
    :description="t('settings.preferenceDescription')"
  >
    <SettingsItem :title="t('settings.theme')" :description="t('settings.themeDescription')">
      <template #action>
        <ThemeSelector />
      </template>
    </SettingsItem>

    <SettingsItem :title="t('settings.language')" :description="t('settings.languageDescription')">
      <template #action>
        <LanguageSelector />
      </template>
    </SettingsItem>
  </SettingsSection>
</template>
