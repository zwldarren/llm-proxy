<script setup lang="ts">
import { onErrorCaptured, ref, watchEffect } from "vue";
import i18n from "@/i18n";
import { Toaster } from "@/components/ui/sonner";
import { useI18n } from "vue-i18n";

const { t } = useI18n();

watchEffect(() => {
  document.documentElement.lang = i18n.global.locale.value;
});

const keepAliveNames = [
  "LogsView",
  "ProvidersView",
  "ModelsView",
  "ApiKeysView",
  "ChatView",
  "ImagesView",
];

const error = ref<Error | null>(null);

onErrorCaptured((err, instance, info) => {
  console.error("Global error captured:", err, info);
  error.value = err;
  return false;
});

const handleRetry = () => {
  error.value = null;
  window.location.reload();
};
</script>

<template>
  <main role="main">
    <!-- Error Boundary -->
    <div
      v-if="error"
      class="fixed inset-0 z-[100] flex items-center justify-center bg-background/95"
    >
      <div class="text-center max-w-md px-6">
        <div class="text-destructive text-6xl mb-4" role="img" aria-label="Warning">⚠️</div>
        <h2 class="text-2xl font-semibold text-foreground mb-2">
          {{ t("errors.somethingWrong") }}
        </h2>
        <p class="text-muted-foreground mb-6">{{ error.message || t("errors.unexpectedError") }}</p>
        <button
          type="button"
          @click="handleRetry"
          class="px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
        >
          {{ t("common.reload") }}
        </button>
      </div>
    </div>
    <RouterView v-else v-slot="{ Component }">
      <KeepAlive :include="keepAliveNames" :max="5">
        <component :is="Component" />
      </KeepAlive>
    </RouterView>
  </main>
  <Toaster position="top-right" :expand="false" />
</template>
