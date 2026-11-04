<script setup lang="ts">
import type { DialogContentEmits, DialogContentProps } from "reka-ui";
import type { HTMLAttributes } from "vue";
import { reactiveOmit } from "@vueuse/core";
import { computed } from "vue";
import { useForwardPropsEmits } from "reka-ui";
import { cn } from "@/lib/utils";
import DialogSurface from "./DialogSurface.vue";

defineOptions({
  inheritAttrs: false,
});

const props = withDefaults(
  defineProps<
    DialogContentProps & {
      class?: HTMLAttributes["class"];
      showCloseButton?: boolean;
      size?: "sm" | "md" | "lg" | "xl" | "screen" | "none";
      layout?: "padded" | "flush" | "none";
      variant?: "modal" | "palette" | "media";
      layer?: "base" | "top";
      scroll?: boolean;
    }
  >(),
  {
    showCloseButton: true,
    size: "md",
    layout: "padded",
    variant: "modal",
    layer: "base",
    scroll: false,
  }
);
const emits = defineEmits<DialogContentEmits>();

const delegatedProps = reactiveOmit(
  props,
  "class",
  "showCloseButton",
  "size",
  "layout",
  "variant",
  "layer",
  "scroll"
);

const forwarded = useForwardPropsEmits(delegatedProps, emits);

const BASE_CLASS =
  "fixed grid w-full border border-border/60 bg-card text-card-foreground duration-200 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95";

const VARIANTS = {
  modal: "top-[50%] left-[50%] max-w-[calc(100%-2rem)] translate-x-[-50%] translate-y-[-50%]",
  palette: "top-[20%] left-[50%] max-w-[calc(100%-2rem)] translate-x-[-50%] block",
  media:
    "inset-0 flex h-full w-full max-w-none items-center justify-center rounded-none border-0 bg-background/95 p-4 backdrop-blur-sm",
} as const;

// DESIGN.md pins the dialog measures: sm 28rem, md 32rem, lg 42rem, xl 36rem
// (the command palette's width) and screen. `xl` is deliberately not wider than
// `lg`; pick the size whose measure you want, not its position in the scale.
const SIZES = {
  sm: "sm:max-w-md",
  md: "sm:max-w-lg",
  lg: "sm:max-w-2xl",
  xl: "sm:max-w-xl",
  screen: "h-[86vh] w-[96vw] max-w-[96vw] flex flex-col sm:max-w-[1500px]",
  none: "",
} as const;

const LAYOUTS = {
  padded: "gap-4 rounded-xl p-6",
  flush: "gap-0 overflow-hidden rounded-xl p-0",
  none: "",
} as const;

const classes = computed(() =>
  cn(
    BASE_CLASS,
    VARIANTS[props.variant],
    props.variant === "media" ? "" : SIZES[props.size],
    props.variant === "media" ? "" : LAYOUTS[props.layout],
    props.scroll && "max-h-[85vh] flex flex-col",
    props.class
  )
);
</script>

<template>
  <DialogSurface
    v-bind="{ ...$attrs, ...forwarded }"
    data-slot="dialog-content"
    :content-class="classes"
    :show-close-button="showCloseButton"
    :layer="layer"
  >
    <slot />
  </DialogSurface>
</template>
