<script setup lang="ts">
import type { AlertDialogContentEmits, AlertDialogContentProps } from "reka-ui";
import type { HTMLAttributes } from "vue";
import { reactiveOmit } from "@vueuse/core";
import {
  AlertDialogContent,
  AlertDialogOverlay,
  AlertDialogPortal,
  useForwardPropsEmits,
} from "reka-ui";
import { cn } from "@/lib/utils";
import { SURFACE_LAYERS, type SurfaceLayer } from "../dialog/layers";

defineOptions({
  inheritAttrs: false,
});

/**
 * A confirm is always an interruption, so it defaults to the top layer rather
 * than the base one: sharing a layer with the surface it is guarding leaves it
 * underneath whenever the view declares it first (ModelsView declared the
 * discard guard above the sheet it belongs to). See `layers.ts`.
 */
const props = withDefaults(
  defineProps<
    AlertDialogContentProps & { class?: HTMLAttributes["class"]; layer?: SurfaceLayer }
  >(),
  { layer: "top" }
);
const emits = defineEmits<AlertDialogContentEmits>();

const delegatedProps = reactiveOmit(props, "class", "layer");

const forwarded = useForwardPropsEmits(delegatedProps, emits);
</script>

<template>
  <AlertDialogPortal>
    <AlertDialogOverlay
      data-slot="alert-dialog-overlay"
      :class="
        cn(
          'data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 fixed inset-0 bg-black/80',
          SURFACE_LAYERS[layer]
        )
      "
    />
    <AlertDialogContent
      data-slot="alert-dialog-content"
      v-bind="{ ...$attrs, ...forwarded }"
      :class="
        cn(
          'bg-background data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 fixed top-[50%] left-[50%] grid w-full max-w-[calc(100%-2rem)] translate-x-[-50%] translate-y-[-50%] gap-4 rounded-md border p-6 shadow-lg duration-200 sm:max-w-lg pointer-events-auto',
          SURFACE_LAYERS[layer],
          props.class
        )
      "
    >
      <slot />
    </AlertDialogContent>
  </AlertDialogPortal>
</template>
