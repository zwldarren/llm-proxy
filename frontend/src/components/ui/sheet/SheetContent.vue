<script setup lang="ts">
import type { DialogContentEmits, DialogContentProps } from "reka-ui";
import type { HTMLAttributes } from "vue";
import { reactiveOmit } from "@vueuse/core";
import { computed } from "vue";
import { useForwardPropsEmits } from "reka-ui";
import { cn } from "@/lib/utils";
import DialogSurface from "../dialog/DialogSurface.vue";

interface SheetContentProps extends DialogContentProps {
  class?: HTMLAttributes["class"];
  side?: "top" | "right" | "bottom" | "left";
  size?: "sm" | "md" | "lg" | "xl" | "2xl" | "none";
  variant?: "default" | "sidebar";
  showCloseButton?: boolean;
  layer?: "base" | "top";
}

defineOptions({
  inheritAttrs: false,
});

const props = withDefaults(defineProps<SheetContentProps>(), {
  side: "right",
  size: "md",
  variant: "default",
  showCloseButton: true,
  layer: "base",
});
const emits = defineEmits<DialogContentEmits>();

const delegatedProps = reactiveOmit(
  props,
  "class",
  "side",
  "size",
  "variant",
  "showCloseButton",
  "layer"
);
const forwarded = useForwardPropsEmits(delegatedProps, emits);

const BASE_CLASS =
  "fixed flex flex-col gap-0 overflow-hidden border-border/80 bg-card p-0 text-card-foreground transition ease-in-out pb-[env(safe-area-inset-bottom,0px)] data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:duration-300 data-[state=open]:duration-500";

const SIDES = {
  right:
    "inset-y-0 right-0 h-full border-l data-[state=closed]:slide-out-to-right data-[state=open]:slide-in-from-right",
  left: "inset-y-0 left-0 h-full border-r data-[state=closed]:slide-out-to-left data-[state=open]:slide-in-from-left",
  top: "inset-x-0 top-0 h-auto border-b data-[state=closed]:slide-out-to-top data-[state=open]:slide-in-from-top",
  bottom:
    "inset-x-0 bottom-0 h-auto border-t data-[state=closed]:slide-out-to-bottom data-[state=open]:slide-in-from-bottom",
} as const;

const SIZES = {
  sm: "w-full sm:max-w-[440px]",
  md: "w-full sm:max-w-[480px]",
  lg: "w-full sm:max-w-[520px] lg:max-w-[600px]",
  xl: "w-full sm:max-w-[640px] lg:max-w-[760px]",
  "2xl": "w-full sm:max-w-[550px] md:max-w-[700px] lg:max-w-[850px] xl:max-w-[1000px]",
  none: "",
} as const;

const classes = computed(() => {
  const horizontal = props.side === "right" || props.side === "left";
  const sidebar = props.variant === "sidebar";
  return cn(
    BASE_CLASS,
    SIDES[props.side],
    horizontal && !sidebar ? SIZES[props.size] : SIZES.none,
    sidebar && "w-(--sidebar-width) bg-sidebar text-sidebar-foreground",
    props.class
  );
});
</script>

<template>
  <DialogSurface
    v-bind="{ ...$attrs, ...forwarded }"
    data-slot="sheet-content"
    :content-class="classes"
    :show-close-button="showCloseButton"
    :layer="layer"
  >
    <slot />
  </DialogSurface>
</template>
