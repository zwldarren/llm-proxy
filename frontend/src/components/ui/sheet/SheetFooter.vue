<script setup lang="ts">
import type { HTMLAttributes } from "vue";
import DialogFooter from "../dialog/DialogFooter.vue";

// Sheet footers always use the dialog's bar treatment; delegating keeps the
// hairline, gutter and spacing in one place so the two cannot drift.
const props = defineProps<{ class?: HTMLAttributes["class"] }>();
</script>

<template>
  <DialogFooter data-slot="sheet-footer" variant="bar" :class="props.class">
    <slot />
  </DialogFooter>
</template>
