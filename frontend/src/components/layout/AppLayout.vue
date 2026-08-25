<script setup lang="ts">
import { onActivated, onMounted, ref } from "vue";
import { useI18n } from "vue-i18n";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import MobileSidebarTrigger from "./MobileSidebarTrigger.vue";
import NavSidebar from "./NavSidebar.vue";
import CommandPalette from "./CommandPalette.vue";
import { useAuthStore } from "@/stores/auth";
import { useSystemStore } from "@/stores/system";

const { t } = useI18n();
const authStore = useAuthStore();
const systemStore = useSystemStore();

interface Props {
  layoutMode?: "default" | "full";
}

withDefaults(defineProps<Props>(), {
  layoutMode: "default",
});

/* Page entrance — a coordinated fade+rise for the whole page block (header +
 * body). The same config-page-reveal animation drives both the default and full
 * layouts, so every page animates consistently. Re-triggered on every
 * activation so revisiting a kept-alive page always animates instead of
 * snapping in. The class is removed again on animationend to drop the transform
 * (avoids a lingering containing block for sticky/fixed children). */
const revealEl = ref<HTMLElement | null>(null);

const REVEAL_DURATION = 500; // must match or exceed CSS animation duration
let revealTimer: ReturnType<typeof setTimeout> | null = null;

const triggerReveal = () => {
  const el = revealEl.value;
  if (!el) return;
  if (revealTimer) return; // already scheduled
  el.classList.remove("config-page-reveal");
  // Cancel any running animations so re-adding the class restarts the CSS animation.
  el.getAnimations().forEach((a) => a.cancel());
  el.classList.add("config-page-reveal");
  // Safety cleanup in case animationend never fires
  revealTimer = setTimeout(() => {
    revealTimer = null;
    el.classList.remove("config-page-reveal");
  }, REVEAL_DURATION + 50);
};

const onRevealEnd = (event: AnimationEvent) => {
  // Only act on the wrapper's own animation, not bubbled child events.
  if (event.target !== revealEl.value) return;
  if (revealTimer) {
    clearTimeout(revealTimer);
    revealTimer = null;
  }
  revealEl.value?.classList.remove("config-page-reveal");
};

// onMounted fires once. onActivated also fires on the first mount
// (for kept-alive pages). The debounce in triggerReveal naturally
// deduplicates calls within the same tick, so we can call both safely.
onMounted(() => {
  triggerReveal();
  // Version/update-check info for the sidebar + settings. Fire-and-forget:
  // never block render, never toast on this automatic call. The endpoint is
  // admin-only, so non-admins skip it (they would get a guaranteed 403).
  if (authStore.isAdmin) {
    systemStore.fetchSystemInfo().catch(() => {});
  }
});
onActivated(triggerReveal);
</script>

<template>
  <SidebarProvider
    class="group/sidebar-wrapper flex h-screen overflow-hidden bg-background font-sans text-foreground"
  >
    <NavSidebar />

    <!-- Global Cmd/Ctrl+K command palette (routes + recent logs) -->
    <CommandPalette />

    <SidebarInset
      id="main-content"
      tabindex="-1"
      class="bg-muted/5 relative flex flex-1 flex-col overflow-hidden"
    >
      <!-- Skip to main content link for keyboard users -->
      <a
        href="#main-content"
        class="sr-only focus:not-sr-only focus:fixed focus:top-4 focus:left-4 focus:z-100 focus:px-4 focus:py-2 focus:bg-primary focus:text-primary-foreground focus:rounded-lg focus:shadow-lg focus:outline-none focus:ring-2 focus:ring-ring"
      >
        {{ t("a11y.skipToContent") }}
      </a>

      <!-- Mobile header -->
      <div
        class="md:hidden flex-none border-b border-border bg-background z-10 h-16 flex items-center px-4 gap-3"
      >
        <MobileSidebarTrigger />
        <p class="text-lg font-semibold truncate">{{ $t("home.title") }}</p>
      </div>

      <!-- Standard layout with scroll and padding. The inner wrapper carries the
           coordinated config-page-reveal entrance animation (see script) — the
           same one used by the full layout — so every page animates consistently
           and re-animates on revisit. -->
      <div v-if="layoutMode === 'default'" class="flex-1 overflow-y-auto">
        <div ref="revealEl" class="w-full px-4 sm:px-6 py-5 sm:py-6" @animationend="onRevealEnd">
          <div
            class="flex flex-col sm:flex-row justify-between items-start gap-4 sm:gap-0 mb-4 sm:mb-5"
          >
            <div class="flex-1 w-full">
              <slot name="header" />
            </div>
          </div>
          <slot />
        </div>
      </div>

      <!-- Full layout (no padding, handled by child). The wrapper carries the
           coordinated config-page-reveal entrance animation (see script). -->
      <div
        v-else
        ref="revealEl"
        class="flex-1 flex flex-col h-full overflow-hidden"
        @animationend="onRevealEnd"
      >
        <slot name="header" />
        <slot />
      </div>
    </SidebarInset>
  </SidebarProvider>
</template>
