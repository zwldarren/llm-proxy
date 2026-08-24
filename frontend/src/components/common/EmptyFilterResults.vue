<script setup lang="ts">
import { Button } from "@/components/ui/button";
import { useI18n } from "vue-i18n";

defineEmits<{
  clear: [];
}>();

const { t } = useI18n();
</script>

<template>
  <div class="px-4 sm:px-6 py-12 text-center">
    <p class="mb-4 text-sm text-muted-foreground">{{ t("common.noMatchingResults") }}</p>
    <Button variant="outline" size="sm" @click="$emit('clear')">
      {{ t("common.clearFilters") }}
    </Button>
  </div>
</template>
