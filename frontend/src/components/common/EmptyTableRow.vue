<script setup lang="ts">
import { TableCell, TableRow } from "@/components/ui/table";
import EmptyFilterResults from "@/components/common/EmptyFilterResults.vue";

defineProps<{ colspan: number }>();
const emit = defineEmits<{ clear: [] }>();
</script>

<template>
  <!-- The table-row shell around the shared "nothing matches" block. The copy,
       the clear button and its label live in EmptyFilterResults, which is what
       the card list renders on the same page, so the two layouts cannot drift
       apart in what they say. -->
  <TableRow class="hover:bg-transparent hover:border-l-transparent">
    <TableCell :colspan="colspan" class="p-0">
      <EmptyFilterResults @clear="emit('clear')" />
    </TableCell>
  </TableRow>
</template>
