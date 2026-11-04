<script setup lang="ts">
import type { DialogContentEmits, DialogContentProps } from "reka-ui";
import type { HTMLAttributes } from "vue";
import { X } from "@lucide/vue";
import { reactiveOmit } from "@vueuse/core";
import {
  DialogClose,
  DialogContent as DialogContentPrimitive,
  DialogPortal,
  useForwardPropsEmits,
} from "reka-ui";
import { useI18n } from "vue-i18n";
import { cn } from "@/lib/utils";
import DialogOverlay from "./DialogOverlay.vue";
import { vInertBackground } from "./inertBackground";

defineOptions({
  inheritAttrs: false,
});

const props = withDefaults(
  defineProps<
    DialogContentProps & {
      contentClass?: HTMLAttributes["class"];
      closeButtonClass?: HTMLAttributes["class"];
      showCloseButton?: boolean;
      layer?: "base" | "top";
    }
  >(),
  {
    showCloseButton: true,
    layer: "base",
  }
);
const emits = defineEmits<DialogContentEmits>();
const { t } = useI18n();

const delegatedProps = reactiveOmit(
  props,
  "contentClass",
  "closeButtonClass",
  "showCloseButton",
  "layer"
);
const forwarded = useForwardPropsEmits(delegatedProps, emits);

// The scrim and the panel must share one stacking layer, otherwise the scrim
// (z-50) paints over a panel that declares no z-index at all and the opened
// surface reads as "the whole page just went dark". Owning the layer here keeps
// every caller (dialog, sheet, nested top layer) on the same footing.
const LAYERS = {
  base: "z-50",
  top: "z-[100]",
} as const;

const CLOSE_BUTTON_CLASS =
  "ring-offset-background focus:ring-ring data-[state=open]:bg-accent data-[state=open]:text-muted-foreground absolute top-4 right-4 rounded-xs p-2 max-sm:p-3.5 opacity-70 transition-opacity hover:opacity-100 focus:ring-2 focus:ring-offset-2 focus:outline-hidden disabled:pointer-events-none touch-target touch-target-square [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4";
</script>

<template>
  <DialogPortal>
    <DialogOverlay :class="LAYERS[layer]" />
    <DialogContentPrimitive
      v-inert-background
      v-bind="{ ...$attrs, ...forwarded }"
      aria-modal="true"
      :class="cn(LAYERS[layer], contentClass)"
    >
      <slot />

      <DialogClose
        v-if="showCloseButton"
        data-slot="dialog-close"
        :class="cn(CLOSE_BUTTON_CLASS, closeButtonClass)"
      >
        <X />
        <span class="sr-only">{{ t("common.close") }}</span>
      </DialogClose>
    </DialogContentPrimitive>
  </DialogPortal>
</template>
