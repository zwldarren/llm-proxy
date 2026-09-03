<script setup lang="ts">
import { useI18n } from "vue-i18n";
import { Skeleton } from "@/components/ui/skeleton";
import { useSkeletonRows } from "@/composables/useSkeletonRows";

interface Props {
  /** Number of list rows to render */
  rows?: number;
}

const props = withDefaults(defineProps<Props>(), {
  rows: 8,
});

const { t } = useI18n();

// Deterministic width pattern; per-row pulse delay gives a quiet cascade.
const nameWidths = ["w-44", "w-36", "w-52", "w-40", "w-48", "w-32", "w-44", "w-40"];
const subWidths = ["w-28", "w-24", "w-32", "w-20", "w-28", "w-24", "w-32", "w-28"];

const rowList = useSkeletonRows(props.rows, nameWidths, subWidths);
</script>

<template>
  <!-- Mirrors config-list rows (PlazaModelListItem / McpServerListItem):
       icon tile, two text lines, status pill, chevron. -->
  <div class="w-full" role="status" aria-busy="true" :aria-label="t('common.loading')">
    <div
      v-for="row in rowList"
      :key="row.key"
      class="flex items-center gap-3 border-b border-border/60 px-4 sm:px-6 py-3 last:border-b-0"
    >
      <Skeleton class="size-9 shrink-0 rounded-lg" :style="{ animationDelay: row.delay }" />
      <div class="flex-1 min-w-0 space-y-1.5">
        <Skeleton class="h-3.5" :class="row.name" :style="{ animationDelay: row.delay }" />
        <Skeleton class="h-3" :class="row.detail" :style="{ animationDelay: row.delay }" />
      </div>
      <Skeleton
        class="h-5 w-16 shrink-0 rounded-full hidden sm:block"
        :style="{ animationDelay: row.delay }"
      />
      <Skeleton class="size-4 shrink-0 rounded-sm" :style="{ animationDelay: row.delay }" />
    </div>
  </div>
</template>
