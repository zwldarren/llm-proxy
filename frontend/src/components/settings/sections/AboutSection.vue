<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useI18n } from "vue-i18n";
import { toast } from "vue-sonner";
import { ArrowUpRight, Info, RefreshCw } from "@lucide/vue";
import { SettingsItem, SettingsSection } from "@/components/settings";
import StatusBadge from "@/components/common/StatusBadge.vue";
import { Button } from "@/components/ui/button";
import { useSystemStore } from "@/stores/system";

const RELEASES_URL = "https://github.com/zwldarren/llm-proxy/releases";

const { t, locale } = useI18n();
const systemStore = useSystemStore();

const checking = ref(false);
/** Set when a manual check throws or the backend reports check_failed. */
const checkError = ref(false);

const info = computed(() => systemStore.info);

const lastChecked = computed(() => {
  const checkedAt = info.value?.checked_at;
  if (!checkedAt) return null;
  const date = new Date(checkedAt);
  if (Number.isNaN(date.getTime())) return null;
  return date.toLocaleString(locale.value);
});

// The store dedupes against AppLayout's app-load fetch; this only re-fetches
// when that earlier call failed (info still null).
onMounted(() => {
  systemStore.fetchSystemInfo().catch(() => {});
});

async function checkForUpdates() {
  if (checking.value) return;
  checking.value = true;
  checkError.value = false;
  try {
    const result = await systemStore.fetchSystemInfo(true);
    if (result.check_failed) {
      // The upstream check could not be performed — inline message + retry.
      checkError.value = true;
    } else if (result.update_available && result.latest_version) {
      toast.success(t("about.updateAvailable", { version: result.latest_version }));
    } else {
      toast.success(t("about.upToDate"));
    }
  } catch {
    checkError.value = true;
  } finally {
    checking.value = false;
  }
}
</script>

<template>
  <SettingsSection :title="t('about.title')" :icon="Info" :description="t('about.description')">
    <SettingsItem :title="t('about.currentVersion')">
      <template #action>
        <a
          v-if="info"
          :href="RELEASES_URL"
          target="_blank"
          rel="noopener noreferrer"
          :aria-label="t('about.viewReleases')"
          class="inline-flex items-center gap-1 text-sm text-muted-foreground transition-colors duration-200 hover:text-foreground"
        >
          <span class="text-data">v{{ info.version }}</span>
          <ArrowUpRight class="size-3.5" aria-hidden="true" />
        </a>
        <span v-else class="text-data text-sm">—</span>
      </template>
    </SettingsItem>

    <SettingsItem v-if="info?.latest_version" :title="t('about.latestVersion')">
      <template #action>
        <a
          :href="RELEASES_URL"
          target="_blank"
          rel="noopener noreferrer"
          :aria-label="t('about.viewReleases')"
          class="inline-flex items-center gap-1 text-sm text-muted-foreground transition-colors duration-200 hover:text-foreground"
        >
          <span class="text-data">v{{ info.latest_version }}</span>
          <ArrowUpRight class="size-3.5" aria-hidden="true" />
        </a>
      </template>
    </SettingsItem>

    <SettingsItem v-if="systemStore.updateAvailable" :title="t('about.status')">
      <template #action>
        <StatusBadge variant="status" status="warning">
          <span class="size-1.5 rounded-full bg-status-warning" aria-hidden="true"></span>
          {{ t("about.updateAvailable", { version: info!.latest_version }) }}
        </StatusBadge>
      </template>
    </SettingsItem>

    <SettingsItem v-if="lastChecked" :title="t('about.lastChecked')">
      <template #action>
        <span class="text-data text-xs text-muted-foreground">{{ lastChecked }}</span>
      </template>
    </SettingsItem>

    <SettingsItem
      v-if="info?.update_check_enabled"
      :title="t('about.checkForUpdates')"
      :description="t('about.checkForUpdatesDescription')"
      :error="checkError ? t('about.checkFailed') : null"
    >
      <template #action>
        <Button variant="outline" size="sm" :disabled="checking" @click="checkForUpdates">
          <RefreshCw :class="['size-3.5', checking && 'animate-spin']" aria-hidden="true" />
          {{
            checking
              ? t("about.checking")
              : checkError
                ? t("about.retry")
                : t("about.checkForUpdates")
          }}
        </Button>
      </template>
    </SettingsItem>

    <SettingsItem
      v-else-if="info"
      :title="t('about.checkForUpdates')"
      :description="t('about.updateChecksDisabled')"
    />
  </SettingsSection>
</template>
