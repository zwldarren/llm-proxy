<script setup lang="ts">
import { useI18n } from "vue-i18n";
import { Button } from "@/components/ui/button";
import { TableCell, TableRow } from "@/components/ui/table";

defineProps<{ colspan: number }>();
const emit = defineEmits<{ clear: [] }>();
const { t } = useI18n();
</script>

<template>
  <TableRow class="hover:bg-transparent hover:border-l-transparent">
    <TableCell :colspan="colspan" class="text-center py-12 text-muted-foreground">
      <p class="mb-4 text-sm">{{ t("common.noMatchingResults") }}</p>
      <Button variant="outline" size="sm" @click="emit('clear')">
        {{ t("common.clearFilters") }}
      </Button>
    </TableCell>
  </TableRow>
</template>
