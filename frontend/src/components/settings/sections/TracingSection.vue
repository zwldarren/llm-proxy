<script setup lang="ts">
import { Edit, Plus, Radio, Trash2 } from "@lucide/vue";
import { computed, ref } from "vue";
import { useI18n } from "vue-i18n";
import StatusBadge from "@/components/common/StatusBadge.vue";
import CollapsiblePanel from "@/components/common/CollapsiblePanel.vue";
import { SettingsItem, SettingsSection } from "@/components/settings";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Switch } from "@/components/ui/switch";
import type { AutoSaveState } from "@/composables/useSettingAutoSave";
import { useAutoSaveRefs } from "@/composables/useAutoSaveRefs";
import {
  configuredProviderCount,
  type TracingProviderEditor,
} from "@/composables/useTracingProviders";
import type { TracingConfig } from "@/types/schemas";

const props = defineProps<{
  autoSave: AutoSaveState<TracingConfig>;
  editor: TracingProviderEditor;
}>();

const emit = defineEmits<{
  (e: "editProvider", id: string): void;
}>();

const { t } = useI18n();

const { state, pending, error } = useAutoSaveRefs(props.autoSave);

const isAddDropdownOpen = ref(false);

const configuredCount = computed(() =>
  configuredProviderCount(state.value.providers, props.editor.providerTypes.value)
);

const hasProviders = computed(() => state.value.providers.length > 0);
</script>

<template>
  <SettingsSection :title="t('nav.tracing')" :icon="Radio" :description="t('tracing.description')">
    <template #actions>
      <StatusBadge v-if="state.enabled && configuredCount > 0" variant="status" status="success">
        {{ t("tracing.status.active") }}&nbsp;·&nbsp;<span class="font-mono tabular-nums">{{
          configuredCount
        }}</span>
      </StatusBadge>
    </template>

    <SettingsItem
      :title="t('tracing.enableTracing')"
      :description="t('tracing.enableTracingDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <Switch v-model="state.enabled" />
      </template>
    </SettingsItem>

    <CollapsiblePanel :open="state.enabled" bordered>
      <div class="overflow-hidden divide-y divide-border/40">
        <!-- Empty state (matches the dashed-panel style used elsewhere) -->
        <div v-if="!hasProviders" class="p-5.5">
          <div
            class="flex flex-col items-center text-center gap-3 py-8 px-4 border border-dashed border-border/60 rounded-lg bg-muted/20"
          >
            <div
              class="size-9 rounded-lg bg-muted/50 border border-border/40 flex items-center justify-center"
            >
              <Radio class="size-4 text-muted-foreground/70" />
            </div>
            <div class="space-y-1">
              <h4 class="text-sm font-medium text-foreground">
                {{ t("tracing.emptyState.title") }}
              </h4>
              <p class="text-xs text-muted-foreground max-w-sm leading-relaxed">
                {{ t("tracing.emptyState.description") }}
              </p>
            </div>
          </div>
        </div>

        <!-- Provider rows — same rhythm as SettingsItem rows; edit in side sheet -->
        <div
          v-for="provider in state.providers"
          :key="provider.id"
          class="px-5.5 py-3.5 flex items-center gap-3.5 hover:bg-muted/12 transition-colors duration-200"
        >
          <div class="flex-1 min-w-0">
            <div
              class="text-sm font-semibold tracking-tight truncate"
              :class="provider.enabled ? 'text-foreground' : 'text-muted-foreground'"
            >
              {{ editor.providerTypeLabel(provider.provider) }}
              <span v-if="provider.name" class="font-normal text-muted-foreground">
                · {{ provider.name }}
              </span>
            </div>
          </div>

          <div class="flex items-center gap-1 shrink-0">
            <Switch
              :model-value="provider.enabled"
              @update:model-value="
                provider.id && editor.updateProviderEnabled(provider.id!, Boolean($event))
              "
            />
            <Button
              variant="ghost"
              size="icon"
              class="size-8 text-muted-foreground hover:text-foreground"
              :aria-label="t('common.edit')"
              @click="provider.id && emit('editProvider', provider.id!)"
            >
              <Edit class="size-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              class="size-8 text-destructive hover:text-destructive hover:bg-destructive/10"
              :aria-label="t('common.delete')"
              @click="provider.id && editor.removeProvider(provider.id)"
            >
              <Trash2 class="size-4" />
            </Button>
          </div>
        </div>

        <!-- Add provider (single type → direct button, multiple → menu) -->
        <div class="px-5.5 py-3.5 flex items-center gap-3">
          <Popover v-if="editor.providerTypes.value.length > 1" v-model:open="isAddDropdownOpen">
            <PopoverTrigger as-child>
              <Button variant="outline" size="sm">
                <Plus class="size-4 mr-1.5" />
                {{ t("tracing.selectProviderType") }}
              </Button>
            </PopoverTrigger>
            <PopoverContent class="w-32 p-1" align="start">
              <button
                v-for="type in editor.providerTypes.value"
                :key="type.name"
                class="w-full text-left px-2.5 py-1.5 text-sm font-medium rounded-sm hover:bg-accent hover:text-accent-foreground transition-colors cursor-pointer"
                @click="
                  editor.addProvider(type.name);
                  isAddDropdownOpen = false;
                "
              >
                {{ editor.providerTypeLabel(type.name) }}
              </button>
            </PopoverContent>
          </Popover>
          <Button v-else variant="outline" size="sm" @click="editor.addProvider('langfuse')">
            <Plus class="size-4 mr-1.5" />
            {{ t("tracing.addProvider") }}
          </Button>
        </div>
      </div>
    </CollapsiblePanel>
  </SettingsSection>
</template>
