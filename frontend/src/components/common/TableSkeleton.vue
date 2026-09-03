<script setup lang="ts">
import { useI18n } from "vue-i18n";
import { Skeleton } from "@/components/ui/skeleton";
import { useSkeletonRows } from "@/composables/useSkeletonRows";

interface Props {
  /** Number of body rows to render */
  rows?: number;
  /** Whether rows lead with a square icon cell (provider/model/mcp tables) */
  icon?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  rows: 8,
  icon: true,
});

const { t } = useI18n();

// Deterministic width patterns so rows read as real records, not one stamped
// mold. Per-row pulse delay gives a quiet cascade instead of a sync blink.
const nameWidths = ["w-40", "w-32", "w-44", "w-36", "w-48", "w-28", "w-40", "w-36"];
const detailWidths = ["w-56", "w-44", "w-64", "w-40", "w-52", "w-48", "w-60", "w-36"];

const rowList = useSkeletonRows(props.rows, nameWidths, detailWidths);
</script>

<template>
  <!-- Mirrors the console table: sticky thead replica (bg-muted/50 + border-b-2)
       over hairline-separated rows with icon, name, badge, detail, actions. -->
  <div class="w-full" role="status" aria-busy="true" :aria-label="t('common.loading')">
    <div class="flex items-center gap-4 border-b-2 border-border/70 bg-muted/50 px-4 sm:px-6 py-3">
      <div v-if="props.icon" class="w-10 shrink-0" aria-hidden="true" />
      <Skeleton class="h-3 w-16" />
      <Skeleton class="h-3 w-14 hidden sm:block" />
      <Skeleton class="h-3 w-20 hidden md:block flex-1 max-w-40" />
      <div class="flex-1" />
      <Skeleton class="h-3 w-12" />
    </div>

    <div
      v-for="row in rowList"
      :key="row.key"
      class="flex items-center gap-4 border-b border-border/60 px-4 sm:px-6 py-2 last:border-b-0"
    >
      <Skeleton
        v-if="props.icon"
        class="size-9 shrink-0 rounded-lg"
        :style="{ animationDelay: row.delay }"
      />
      <div class="flex-1 min-w-0">
        <Skeleton class="h-4" :class="row.name" :style="{ animationDelay: row.delay }" />
      </div>
      <Skeleton
        class="h-5 w-16 shrink-0 rounded-full hidden sm:block"
        :style="{ animationDelay: row.delay }"
      />
      <Skeleton
        class="h-3.5 hidden md:block"
        :class="row.detail"
        :style="{ animationDelay: row.delay }"
      />
      <div class="flex shrink-0 items-center justify-end gap-1 w-20">
        <Skeleton class="size-8 rounded-md" :style="{ animationDelay: row.delay }" />
        <Skeleton
          class="size-8 rounded-md hidden sm:block"
          :style="{ animationDelay: row.delay }"
        />
      </div>
    </div>
  </div>
</template>
