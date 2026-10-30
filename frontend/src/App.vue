<script setup lang="ts">
import { onErrorCaptured, ref, watch, watchEffect } from "vue";
import { useRoute, useRouter } from "vue-router";
import i18n from "@/i18n";
import AppErrorState from "@/components/common/AppErrorState.vue";
import StepUpDialog from "@/components/common/StepUpDialog.vue";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";

const route = useRoute();
const router = useRouter();

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
/** Vue's label for where it caught the error ("render function", ...). */
const errorInfo = ref<string | null>(null);
/** Component that threw, when Vue could name one — the fastest way to locate it. */
const errorComponent = ref<string | null>(null);

function resetError(): void {
  error.value = null;
  errorInfo.value = null;
  errorComponent.value = null;
}

onErrorCaptured((err, instance, info) => {
  console.error("Global error captured:", err, info);
  error.value = err instanceof Error ? err : new Error(String(err));
  errorInfo.value = info;
  errorComponent.value = instance?.$options.name ?? null;
  return false;
});

/**
 * A render error belongs to the screen that raised it.
 *
 * The boundary replaces `RouterView`, so the operator cannot navigate away by
 * hand — but other code can (a 401 redirecting to /login, a store funnelling
 * into the forced password change). Once the route moves on, the broken
 * component is gone with the previous screen, and holding the boundary up
 * would hide the page they were actually sent to.
 */
watch(
  () => route.fullPath,
  () => {
    if (error.value) resetError();
  }
);

async function goHome(): Promise<void> {
  resetError();
  await router.replace("/").catch(() => {});
}

function reload(): void {
  window.location.reload();
}
</script>

<template>
  <TooltipProvider>
    <!-- App-shell wrapper, not a landmark: the single <main> landmark lives in
         SidebarInset (#main-content); auth views render their own <main>. -->
    <div>
      <!-- Error boundary: one broken screen must not brick the whole console. -->
      <AppErrorState
        v-if="error"
        :error="error"
        :info="errorInfo"
        :component-name="errorComponent"
        @home="goHome"
        @reload="reload"
      />
      <RouterView v-else v-slot="{ Component }">
        <KeepAlive :include="keepAliveNames" :max="5">
          <component :is="Component" />
        </KeepAlive>
      </RouterView>
    </div>
    <StepUpDialog />
    <Toaster position="top-right" :expand="false" />
  </TooltipProvider>
</template>
