<script setup lang="ts">
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { useRoute } from "vue-router";
import { IdCard } from "@lucide/vue";
import AppLayout from "@/components/layout/AppLayout.vue";
import PageHeader from "@/components/common/PageHeader.vue";
import { SegmentedControl, SegmentedControlItem } from "@/components/ui/segmented-control";

const { t } = useI18n();
const route = useRoute();

const isProfile = computed(() => route.name === "account");
const isPreferences = computed(() => route.name === "preferences");
const isSecurity = computed(() => route.name === "security");
</script>

<template>
  <AppLayout>
    <!-- One account workspace: the routes are deep-linkable tabs, not
         separate destinations. max-w-4xl gives the identity form, the
         preference panels and the credential panels the same reading measure
         without leaving the column pinned to the left edge on wide screens. -->
    <div class="mx-auto w-full max-w-4xl">
      <PageHeader
        :title="t('account.title')"
        :description="t('account.description')"
        :icon="IdCard"
      >
        <template #actions>
          <SegmentedControl kind="nav" :aria-label="t('account.tabsLabel')">
            <SegmentedControlItem kind="nav" as-child :active="isProfile">
              <RouterLink to="/account">{{ t("account.tabProfile") }}</RouterLink>
            </SegmentedControlItem>
            <SegmentedControlItem kind="nav" as-child :active="isPreferences">
              <RouterLink to="/account/preferences">
                {{ t("account.tabPreferences") }}
              </RouterLink>
            </SegmentedControlItem>
            <SegmentedControlItem kind="nav" as-child :active="isSecurity">
              <RouterLink to="/account/security">{{ t("account.tabSecurity") }}</RouterLink>
            </SegmentedControlItem>
          </SegmentedControl>
        </template>
      </PageHeader>

      <div class="mt-6">
        <slot />
      </div>
    </div>
  </AppLayout>
</template>
