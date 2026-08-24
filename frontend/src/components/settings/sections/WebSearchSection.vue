<script setup lang="ts">
import { Search } from "@lucide/vue";
import { watch } from "vue";
import { useI18n } from "vue-i18n";
import { SettingsItem, SettingsSection } from "@/components/settings";
import CollapsiblePanel from "@/components/common/CollapsiblePanel.vue";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { NumberInput } from "@/components/ui/number-input";
import type { AutoSaveState } from "@/composables/useSettingAutoSave";
import { useAutoSaveRefs } from "@/composables/useAutoSaveRefs";
import type { WebSearchConfig } from "@/types/schemas";

const props = defineProps<{
  autoSave: AutoSaveState<WebSearchConfig>;
}>();

const { t } = useI18n();

const { state, pending, error } = useAutoSaveRefs(props.autoSave);

// Lazily create the per-provider config object the first time a provider is
// selected, so the form below has a defined object to bind against.
watch(
  () => state.value.provider,
  (provider) => {
    if (provider === "searxng" && !state.value.searxng) {
      state.value.searxng = {
        url: "",
        timeout: 30,
        max_results: 10,
      };
    }
    if (provider === "ollama" && !state.value.ollama) {
      state.value.ollama = {
        api_key: "",
        base_url: "https://ollama.com",
        timeout: 30,
        max_results: 10,
      };
    }
  }
);
</script>

<template>
  <SettingsSection
    :title="t('settings.webSearch')"
    :icon="Search"
    :description="t('settings.webSearchDescription')"
  >
    <SettingsItem
      :title="t('settings.enableWebSearch')"
      :description="t('settings.enableWebSearchDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <Switch v-model="state.enabled" />
      </template>
    </SettingsItem>

    <CollapsiblePanel :open="state.enabled" bordered>
      <SettingsItem :title="t('settings.searchProvider')">
        <template #action>
          <Select
            :model-value="state.provider"
            @update:model-value="
              (v) => {
                if (v === 'searxng' || v === 'ollama') state.provider = v;
              }
            "
          >
            <SelectTrigger class="w-[180px] h-9">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="searxng">{{ t("settings.searxng.name") }}</SelectItem>
              <SelectItem value="ollama">{{ t("settings.ollama.name") }}</SelectItem>
            </SelectContent>
          </Select>
        </template>
      </SettingsItem>

      <div
        v-if="state.provider === 'searxng' && state.searxng"
        class="border-t border-border/40 bg-muted/20 p-5 space-y-4"
      >
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <!-- URL -->
          <div class="md:col-span-2 space-y-1.5">
            <Label class="text-xs font-semibold text-muted-foreground">{{
              t("settings.searxng.url")
            }}</Label>
            <Input
              v-model="state.searxng.url"
              :placeholder="t('settings.searxng.urlPlaceholder')"
              class="font-mono text-sm w-full bg-background"
            />
          </div>

          <!-- API Key -->
          <div class="md:col-span-2 space-y-1.5">
            <Label class="text-xs font-semibold text-muted-foreground">{{
              t("settings.searxng.apiKey")
            }}</Label>
            <Input
              v-model="state.searxng.api_key"
              type="password"
              :placeholder="t('settings.searxng.apiKeyPlaceholder')"
              class="font-mono text-sm w-full bg-background"
            />
          </div>

          <!-- Username -->
          <div class="space-y-1.5">
            <Label class="text-xs font-semibold text-muted-foreground">{{
              t("settings.searxng.basicAuthUsername")
            }}</Label>
            <Input
              v-model="state.searxng.basic_auth_username"
              :placeholder="t('settings.searxng.basicAuthUsernamePlaceholder')"
              class="font-mono text-sm w-full bg-background"
            />
          </div>

          <!-- Password -->
          <div class="space-y-1.5">
            <Label class="text-xs font-semibold text-muted-foreground">{{
              t("settings.searxng.basicAuthPassword")
            }}</Label>
            <Input
              v-model="state.searxng.basic_auth_password"
              type="password"
              :placeholder="t('settings.searxng.basicAuthPasswordPlaceholder')"
              class="font-mono text-sm w-full bg-background"
            />
          </div>

          <!-- Timeout -->
          <div class="space-y-1.5">
            <Label class="text-xs font-semibold text-muted-foreground">
              {{ t("settings.searxng.timeout") }}
            </Label>
            <NumberInput
              v-model.number="state.searxng.timeout"
              class="font-mono text-sm w-full bg-background"
            />
          </div>

          <!-- Max Results -->
          <div class="space-y-1.5">
            <Label class="text-xs font-semibold text-muted-foreground">{{
              t("settings.searxng.maxResults")
            }}</Label>
            <NumberInput
              v-model.number="state.searxng.max_results"
              min="1"
              max="20"
              class="font-mono text-sm w-full bg-background"
            />
          </div>
        </div>
      </div>

      <div
        v-if="state.provider === 'ollama' && state.ollama"
        class="border-t border-border/40 bg-muted/20 p-5 space-y-4"
      >
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <!-- API Key -->
          <div class="md:col-span-2 space-y-1.5">
            <Label class="text-xs font-semibold text-muted-foreground">{{
              t("settings.ollama.apiKey")
            }}</Label>
            <Input
              v-model="state.ollama.api_key"
              type="password"
              :placeholder="t('settings.ollama.apiKeyPlaceholder')"
              class="font-mono text-sm w-full bg-background"
            />
          </div>

          <!-- Base URL -->
          <div class="md:col-span-2 space-y-1.5">
            <Label class="text-xs font-semibold text-muted-foreground">{{
              t("settings.ollama.baseUrl")
            }}</Label>
            <Input
              v-model="state.ollama.base_url"
              :placeholder="t('settings.ollama.baseUrlPlaceholder')"
              class="font-mono text-sm w-full bg-background"
            />
          </div>

          <!-- Timeout -->
          <div class="space-y-1.5">
            <Label class="text-xs font-semibold text-muted-foreground">
              {{ t("settings.ollama.timeout") }}
            </Label>
            <NumberInput
              v-model.number="state.ollama.timeout"
              class="font-mono text-sm w-full bg-background"
            />
          </div>

          <!-- Max Results -->
          <div class="space-y-1.5">
            <Label class="text-xs font-semibold text-muted-foreground">{{
              t("settings.ollama.maxResults")
            }}</Label>
            <NumberInput
              v-model.number="state.ollama.max_results"
              min="1"
              max="10"
              class="font-mono text-sm w-full bg-background"
            />
          </div>
        </div>
      </div>
    </CollapsiblePanel>
  </SettingsSection>
</template>
