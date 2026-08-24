<script setup lang="ts">
import { TableCell } from "@/components/ui/table";

interface Props {
  class?: string;
}

withDefaults(defineProps<Props>(), {
  class: "",
});
</script>

<template>
  <TableCell class="w-24 text-right whitespace-nowrap" :class="class">
    <slot />
  </TableCell>
</template>
