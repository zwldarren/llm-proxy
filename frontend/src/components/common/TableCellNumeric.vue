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
  <TableCell class="text-data text-right" :class="class">
    <slot />
  </TableCell>
</template>
