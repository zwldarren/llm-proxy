<script setup lang="ts">
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { useRoute } from "vue-router";
import { IdCard } from "@lucide/vue";
import AppLayout from "@/components/layout/AppLayout.vue";
import PageHeader from "@/components/common/PageHeader.vue";

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
          <nav
            class="flex items-center rounded-lg border border-border/40 bg-muted/40 p-0.5"
            :aria-label="t('account.tabsLabel')"
          >
            <RouterLink
              to="/account"
              class="rounded-md px-2.5 py-1 text-xs font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50 max-md:min-h-11 max-md:px-4"
              :class="
                isProfile
                  ? 'bg-background text-foreground shadow-xs'
                  : 'text-muted-foreground hover:text-foreground'
              "
              :aria-current="isProfile ? 'page' : undefined"
            >
              {{ t("account.tabProfile") }}
            </RouterLink>
            <RouterLink
              to="/account/preferences"
              class="rounded-md px-2.5 py-1 text-xs font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50 max-md:min-h-11 max-md:px-4"
              :class="
                isPreferences
                  ? 'bg-background text-foreground shadow-xs'
                  : 'text-muted-foreground hover:text-foreground'
              "
              :aria-current="isPreferences ? 'page' : undefined"
            >
              {{ t("account.tabPreferences") }}
            </RouterLink>
            <RouterLink
              to="/account/security"
              class="rounded-md px-2.5 py-1 text-xs font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50 max-md:min-h-11 max-md:px-4"
              :class="
                isSecurity
                  ? 'bg-background text-foreground shadow-xs'
                  : 'text-muted-foreground hover:text-foreground'
              "
              :aria-current="isSecurity ? 'page' : undefined"
            >
              {{ t("account.tabSecurity") }}
            </RouterLink>
          </nav>
        </template>
      </PageHeader>

      <div class="mt-6">
        <slot />
      </div>
    </div>
  </AppLayout>
</template>
