<script setup lang="ts">
import { Palette } from "@lucide/vue";
import { useI18n } from "vue-i18n";
import PanelSection from "@/components/common/PanelSection.vue";
import ThemeSelector from "@/components/settings/ThemeSelector.vue";
import LanguageSelector from "@/components/settings/LanguageSelector.vue";

const { t } = useI18n();
</script>

<template>
  <!-- Personal, browser-stored preferences: they belong to the account, not to
       the instance configuration on /config/settings. -->
  <PanelSection
    id="appearance"
    :title="t('account.appearance')"
    :description="t('account.appearanceDescription')"
    :icon="Palette"
  >
    <dl class="divide-y divide-border/50 text-sm">
      <div class="flex flex-wrap items-center justify-between gap-3 py-3">
        <dt class="min-w-0">
          <span class="block text-sm font-medium text-foreground">{{ t("settings.theme") }}</span>
          <span class="mt-0.5 block text-xs text-muted-foreground">
            {{ t("settings.themeDescription") }}
          </span>
        </dt>
        <dd>
          <ThemeSelector />
        </dd>
      </div>

      <div class="flex flex-wrap items-center justify-between gap-3 py-3">
        <dt class="min-w-0">
          <span class="block text-sm font-medium text-foreground">
            {{ t("settings.language") }}
          </span>
          <span class="mt-0.5 block text-xs text-muted-foreground">
            {{ t("settings.languageDescription") }}
          </span>
        </dt>
        <dd>
          <LanguageSelector />
        </dd>
      </div>
    </dl>
  </PanelSection>
</template>
