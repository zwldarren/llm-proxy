<script setup lang="ts">
import { useI18n } from "vue-i18n";
import { Skeleton } from "@/components/ui/skeleton";

const { t } = useI18n();

// Deterministic widths so the strip cells read as distinct metrics.
const valueWidths = ["w-16", "w-20", "w-14", "w-18", "w-16", "w-12"];
const captionWidths = ["w-24", "w-20", "w-28", "w-24", "w-20", "w-16"];
const listRowWidths = ["w-3/4", "w-2/3", "w-4/5", "w-1/2"];

// Top-to-bottom pulse cascade: each section continues the wave from the
// previous one, so the skeleton reads as a single flowing surface.
const delay = (step: number) => `${step * 70}ms`;
const stripDelay = (i: number) => delay(i - 1);
const meterDelay = (i: number) => delay(6 + i - 1);
const chartDelay = delay(8);
const breakdownDelay = (col: number, i: number) => delay(9 + (col - 1) * 4 + (i - 1));
</script>

<template>
  <!-- Mirrors the dashboard geometry: metric strip -> secondary meters ->
       trends chart -> two flush breakdown columns. -->
  <div class="w-full" role="status" aria-busy="true" :aria-label="t('common.loading')">
    <!-- Primary metric strip: 6 flush cells, hairline-separated -->
    <div
      class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 divide-x divide-y sm:divide-y-0 lg:divide-y-0 divide-border/60 border-b border-border/60 px-4 sm:px-6"
    >
      <div v-for="i in 6" :key="i" class="px-3 sm:px-4 py-3">
        <Skeleton class="h-3 w-16" :style="{ animationDelay: stripDelay(i) }" />
        <Skeleton
          class="h-5 mt-2"
          :class="valueWidths[i - 1]"
          :style="{ animationDelay: stripDelay(i) }"
        />
        <Skeleton
          class="h-3 mt-2"
          :class="captionWidths[i - 1]"
          :style="{ animationDelay: stripDelay(i) }"
        />
      </div>
    </div>

    <!-- Secondary strip: 2 meter cells -->
    <div
      class="grid grid-cols-1 sm:grid-cols-2 divide-y sm:divide-y-0 sm:divide-x divide-border/60 border-b border-border/60 px-4 sm:px-6"
    >
      <div v-for="i in 2" :key="i" class="px-3 sm:px-4 py-3">
        <div class="flex items-center justify-between">
          <Skeleton class="h-3 w-24" :style="{ animationDelay: meterDelay(i) }" />
          <Skeleton class="h-3 w-16" :style="{ animationDelay: meterDelay(i) }" />
        </div>
        <Skeleton
          class="h-1 w-full rounded-full mt-2.5"
          :style="{ animationDelay: meterDelay(i) }"
        />
      </div>
    </div>

    <!-- Trends chart: header row + chart panel -->
    <div class="border-b border-border/60">
      <div class="flex items-center justify-between px-4 sm:px-6 py-3 border-b border-border/60">
        <Skeleton class="h-4 w-32" :style="{ animationDelay: chartDelay }" />
        <Skeleton class="h-7 w-24 rounded-md" :style="{ animationDelay: chartDelay }" />
      </div>
      <div class="px-4 sm:px-6 pb-4 pt-3">
        <Skeleton class="h-72 md:h-85 w-full rounded-lg" :style="{ animationDelay: chartDelay }" />
      </div>
    </div>

    <!-- Breakdown columns: provider / model lists with meters -->
    <div
      class="grid grid-cols-1 lg:grid-cols-2 divide-y lg:divide-y-0 lg:divide-x divide-border/60 items-start"
    >
      <div v-for="col in 2" :key="col" class="px-4 sm:px-6 py-4 space-y-4">
        <Skeleton class="h-4 w-28" :style="{ animationDelay: breakdownDelay(col, 0) }" />
        <div v-for="i in 4" :key="i" class="space-y-1.5">
          <div class="flex items-center justify-between">
            <Skeleton class="h-3 w-24" :style="{ animationDelay: breakdownDelay(col, i) }" />
            <Skeleton class="h-3 w-12" :style="{ animationDelay: breakdownDelay(col, i) }" />
          </div>
          <Skeleton
            class="h-1.5 rounded-full"
            :class="listRowWidths[i - 1]"
            :style="{ animationDelay: breakdownDelay(col, i) }"
          />
        </div>
      </div>
    </div>
  </div>
</template>
