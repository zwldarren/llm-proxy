<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import AccountShell from "@/components/layout/AccountShell.vue";
import AppearancePanel from "@/components/account/AppearancePanel.vue";
import TracingPanel from "@/components/account/TracingPanel.vue";
import TracingProviderSheet from "@/components/settings/sections/TracingProviderSheet.vue";
import { useSettingAutoSave } from "@/composables/useSettingAutoSave";
import { useTracingProviderEditor } from "@/composables/useTracingProviders";
import { meTracingApi } from "@/services/api/config";
import { DEFAULT_TRACING } from "@/constants/defaults";
import type { TracingConfig } from "@/types/schemas";

// ── Personal tracing ────────────────────────────────────────────────
// Per-user, self-service, and therefore a personal setting: it belongs to the
// account workspace, not to the instance-wide configuration screen.
const tracingAutoSave = useSettingAutoSave<TracingConfig>(DEFAULT_TRACING, (val) => {
  const saveData = tracing.buildSavePayload(val);
  return meTracingApi.updateTracing(saveData).then((r) => r.config);
});

const tracing = useTracingProviderEditor(tracingAutoSave.state);

const editingProviderId = ref<string | null>(null);
const editingProvider = computed(
  () => tracingAutoSave.state.value.providers.find((p) => p.id === editingProviderId.value) ?? null
);

function openProviderEditor(id: string) {
  editingProviderId.value = id;
}

function closeProviderEditor() {
  editingProviderId.value = null;
}

async function loadPersonalTracing() {
  try {
    await tracing.loadProviderTypes(meTracingApi);
  } catch (err) {
    console.error("Failed to load tracing provider types:", err);
  }
  try {
    const res = await meTracingApi.getTracing();
    tracingAutoSave.initialize({
      enabled: res.config.enabled ?? false,
      providers: tracing.prepareInitialProviders(res.config.providers ?? []),
    });
  } catch (err) {
    console.error("Failed to load personal tracing config:", err);
  }
}

onMounted(() => {
  void loadPersonalTracing();
});
</script>

<template>
  <AccountShell>
    <div class="space-y-6 pb-16">
      <AppearancePanel />
      <TracingPanel
        :auto-save="tracingAutoSave"
        :editor="tracing"
        @edit-provider="openProviderEditor"
      />
    </div>

    <!-- Tracing provider editor (side sheet) -->
    <TracingProviderSheet
      :provider="editingProvider"
      :editor="tracing"
      @close="closeProviderEditor"
    />
  </AccountShell>
</template>
