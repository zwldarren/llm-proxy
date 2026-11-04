<script setup lang="ts">
import type { SidebarProps } from ".";
import { computed } from "vue";
import { cn } from "@/lib/utils";
import { Sheet, SheetContent, SheetHeader } from "@/components/ui/sheet";
import { SIDEBAR_WIDTH_MOBILE, useSidebar } from "./utils";

defineOptions({
  inheritAttrs: false,
});

const props = withDefaults(defineProps<SidebarProps>(), {
  side: "left",
  variant: "sidebar",
  collapsible: "offcanvas",
});

const { isMobile, state, openMobile, setOpenMobile } = useSidebar();

// The rail slides on transform; the desktop gap snaps to the new width once.
// Animating width/left/right forced layout on every frame and moved the whole
// inset with it. Keeping the fixed panel at full width and shifting it left
// lets the content stay put while only the visible edge moves.
const collapsedWidth = computed(() =>
  props.variant === "floating" || props.variant === "inset"
    ? "calc(var(--sidebar-width-icon) + 1rem + 2px)"
    : "var(--sidebar-width-icon)"
);

const collapsedTransform = computed(() =>
  props.side === "left"
    ? "group-data-[collapsible=icon]:translate-x-[calc(var(--sidebar-collapsed-width)-var(--sidebar-width))]"
    : "group-data-[collapsible=icon]:translate-x-[calc(var(--sidebar-width)-var(--sidebar-collapsed-width))]"
);

// On the left, the content travels back the same distance the rail slides out
// so icons stay put while the panel edge moves; the width itself snaps once.
const collapsedContentTransform = computed(() =>
  props.side === "left"
    ? "group-data-[collapsible=icon]:translate-x-[calc(var(--sidebar-width)-var(--sidebar-collapsed-width))]"
    : ""
);
</script>

<template>
  <div
    v-if="collapsible === 'none'"
    data-slot="sidebar"
    :class="
      cn('bg-sidebar text-sidebar-foreground flex h-full w-(--sidebar-width) flex-col', props.class)
    "
    v-bind="$attrs"
  >
    <slot />
  </div>

  <Sheet v-else-if="isMobile" :open="openMobile" v-bind="$attrs" @update:open="setOpenMobile">
    <SheetContent
      data-sidebar="sidebar"
      data-slot="sidebar"
      data-mobile="true"
      :side="side"
      size="none"
      variant="sidebar"
      :show-close-button="false"
      :style="{
        '--sidebar-width': SIDEBAR_WIDTH_MOBILE,
      }"
    >
      <SheetHeader variant="sr-only" title="Sidebar" description="Displays the mobile sidebar." />
      <div class="flex h-full w-full flex-col">
        <slot />
      </div>
    </SheetContent>
  </Sheet>

  <div
    v-else
    class="group peer text-sidebar-foreground hidden md:block"
    data-slot="sidebar"
    :data-state="state"
    :data-collapsible="state === 'collapsed' ? collapsible : ''"
    :data-variant="variant"
    :data-side="side"
  >
    <!-- This is what handles the sidebar gap on desktop  -->
    <div
      :class="
        cn(
          'relative w-(--sidebar-width) bg-transparent',
          'group-data-[collapsible=offcanvas]:w-0',
          'group-data-[side=right]:rotate-180',
          variant === 'floating' || variant === 'inset'
            ? 'group-data-[collapsible=icon]:w-[calc(var(--sidebar-width-icon)+(--spacing(4)))]'
            : 'group-data-[collapsible=icon]:w-(--sidebar-width-icon)'
        )
      "
    />
    <div
      :style="{ '--sidebar-collapsed-width': collapsedWidth }"
      :class="
        cn(
          // Above page chrome: a sticky page header also carries z-10 and, being
          // later in the DOM, would paint over the edge controls that straddle
          // the rail.
          'fixed inset-y-0 z-30 hidden h-svh w-(--sidebar-width) transition-transform duration-200 ease-linear md:flex',
          side === 'left'
            ? 'left-0 group-data-[collapsible=offcanvas]:-translate-x-full'
            : 'right-0 group-data-[collapsible=offcanvas]:translate-x-full',
          collapsedTransform,
          // Adjust the padding for floating and inset variants.
          variant === 'floating' || variant === 'inset'
            ? 'p-2'
            : 'bg-sidebar group-data-[side=left]:border-r group-data-[side=right]:border-l',
          props.class
        )
      "
      v-bind="$attrs"
    >
      <!-- Keep edge controls outside the clipped content layer so a control
           straddling the rail edge is never cut in half mid-transition. -->
      <slot name="edge" />
      <div class="h-full w-full overflow-hidden">
        <div
          data-sidebar="sidebar"
          role="complementary"
          :class="
            cn(
              'bg-sidebar group-data-[variant=floating]:border-sidebar-border relative flex h-full w-full flex-col transition-transform duration-200 ease-linear group-data-[variant=floating]:rounded-lg group-data-[variant=floating]:border group-data-[variant=floating]:shadow-sm group-data-[collapsible=icon]:w-[var(--sidebar-collapsed-width)]',
              collapsedContentTransform
            )
          "
        >
          <!-- The rail is a landmark so the brand header and the account footer
               (which sit outside the inner <nav>) are reachable when navigating
               by landmark; the <nav> inside remains the navigation landmark. -->
          <slot />
        </div>
      </div>
    </div>
  </div>
</template>
