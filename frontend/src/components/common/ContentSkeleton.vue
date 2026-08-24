<script setup lang="ts">
import { Skeleton } from "@/components/ui/skeleton";

interface Props {
  rows?: number;
}

const props = withDefaults(defineProps<Props>(), {
  rows: 6,
});
</script>

<template>
  <div class="w-full max-w-5xl space-y-3 px-6 py-6" aria-busy="true" role="status">
    <div class="flex items-center gap-3 pb-2">
      <Skeleton class="h-8 w-1/4 max-w-[12rem]" />
      <div class="flex-1" />
      <Skeleton class="h-8 w-20" />
    </div>
    <Skeleton v-for="i in props.rows" :key="i" class="h-12 w-full" />
    <span class="sr-only">Loading...</span>
  </div>
</template>
