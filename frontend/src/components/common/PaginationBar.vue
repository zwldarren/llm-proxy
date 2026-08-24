<script setup lang="ts">
import { ChevronFirst, ChevronLast, ChevronLeft, ChevronRight } from "@lucide/vue";
import { computed, ref, watch } from "vue";
import type { HTMLAttributes } from "vue";
import { useI18n } from "vue-i18n";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import {
  Pagination,
  PaginationContent,
  PaginationEllipsis,
  PaginationFirst,
  PaginationItem,
  PaginationLast,
  PaginationNext,
  PaginationPrevious,
} from "@/components/ui/pagination";

interface Props {
  currentPage: number;
  totalPages: number;
  disabled?: boolean;
  totalItems?: number;
  itemsPerPage?: number;
  class?: HTMLAttributes["class"];
}

const props = withDefaults(defineProps<Props>(), {
  currentPage: 1,
  totalPages: 1,
  disabled: false,
  totalItems: 0,
  itemsPerPage: 20,
});

const emit = defineEmits<{
  prev: [];
  next: [];
  first: [];
  last: [];
  "go-to-page": [page: number];
}>();

const { t } = useI18n();

const pageInput = ref(props.currentPage.toString());

watch(
  () => props.currentPage,
  (newPage) => {
    pageInput.value = newPage.toString();
  }
);

const canGoPrev = computed(() => props.currentPage > 1);
const canGoNext = computed(() => props.currentPage < props.totalPages);

// Handle page changes from reka-ui pagination component
// This handles all navigation: first, prev, page numbers, next, last
const handlePageChange = (page: number) => {
  if (page !== props.currentPage) {
    emit("go-to-page", page);
  }
};

const handlePageInputKeydown = (event: KeyboardEvent) => {
  if (event.key === "Enter") {
    goToInputPage();
  }
};

const goToInputPage = () => {
  const page = Number.parseInt(pageInput.value, 10);
  if (!Number.isNaN(page) && page >= 1 && page <= props.totalPages) {
    if (page !== props.currentPage) {
      emit("go-to-page", page);
    }
  } else {
    pageInput.value = props.currentPage.toString();
  }
};

// Navigation helpers that emit to parent
const goToPrev = () => {
  if (canGoPrev.value) {
    emit("go-to-page", props.currentPage - 1);
  }
};

const goToNext = () => {
  if (canGoNext.value) {
    emit("go-to-page", props.currentPage + 1);
  }
};

const goToFirst = () => {
  if (canGoPrev.value) {
    emit("go-to-page", 1);
  }
};

const goToLast = () => {
  if (canGoNext.value) {
    emit("go-to-page", props.totalPages);
  }
};

const rangeText = computed(() => {
  const startItem = (props.currentPage - 1) * props.itemsPerPage + 1;
  const endItem = Math.min(props.currentPage * props.itemsPerPage, props.totalItems);
  return props.totalItems > 0 ? `${startItem}-${endItem}` : "0";
});
</script>

<template>
  <div
    :class="
      cn(
        'border-t border-border/40 bg-card/50 px-6 py-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between',
        props.class
      )
    "
  >
    <!-- Left: Pagination Info -->
    <div class="text-sm text-muted-foreground font-medium hidden sm:block">
      <span v-if="totalItems > 0">
        {{ t("common.showingResults", { count: rangeText, total: totalItems }) }}
      </span>
      <span v-else>
        {{ t("common.noResults") }}
      </span>
    </div>

    <!-- Right: Pagination controls & page input -->
    <div class="flex flex-col sm:flex-row items-center justify-center gap-4 sm:gap-6">
      <Pagination
        :page="currentPage"
        :total="totalPages"
        :items-per-page="1"
        :disabled="disabled"
        class="w-auto mx-0"
        @update:page="handlePageChange"
      >
        <!-- Mobile: Simplified pagination -->
        <div class="flex items-center justify-between gap-4 sm:hidden w-full px-2">
          <button
            type="button"
            variant="outline"
            size="sm"
            class="flex-1 max-w-[120px] h-9 text-xs border border-input bg-background hover:bg-accent hover:text-accent-foreground rounded-md disabled:opacity-50 disabled:pointer-events-none flex items-center justify-center gap-1 transition-colors"
            :disabled="disabled || !canGoPrev"
            @click="goToPrev"
          >
            <ChevronLeft class="w-4 h-4" />
            {{ t("common.previous") }}
          </button>
          <span class="text-xs text-muted-foreground font-semibold text-center">
            {{ currentPage }} / {{ totalPages }}
          </span>
          <button
            type="button"
            variant="outline"
            size="sm"
            class="flex-1 max-w-[120px] h-9 text-xs border border-input bg-background hover:bg-accent hover:text-accent-foreground rounded-md disabled:opacity-50 disabled:pointer-events-none flex items-center justify-center gap-1 transition-colors"
            :disabled="disabled || !canGoNext"
            @click="goToNext"
          >
            {{ t("common.next") }}
            <ChevronRight class="w-4 h-4" />
          </button>
        </div>

        <!-- Desktop: Full pagination with Shadcn-Vue -->
        <PaginationContent v-slot="{ items }" class="hidden sm:flex items-center gap-1.5">
          <PaginationFirst
            :disabled="disabled || !canGoPrev"
            size="icon-sm"
            class="border border-border/80 bg-background/80 hover:bg-accent/70 hover:text-accent-foreground rounded-md transition-colors duration-200"
            :title="t('common.firstPage')"
            @click="goToFirst"
          >
            <ChevronFirst />
          </PaginationFirst>

          <PaginationPrevious
            :disabled="disabled || !canGoPrev"
            size="icon-sm"
            class="border border-border/80 bg-background/80 hover:bg-accent/70 hover:text-accent-foreground rounded-md transition-colors duration-200"
            :title="t('common.previous')"
            @click="goToPrev"
          >
            <ChevronLeft />
          </PaginationPrevious>

          <template v-for="(item, index) in items" :key="index">
            <PaginationItem
              v-if="item.type === 'page'"
              :value="item.value"
              :isActive="item.value === currentPage"
              size="icon-sm"
              class="border rounded-md transition-colors duration-200"
              :class="
                cn(
                  item.value === currentPage
                    ? 'bg-primary/10 text-primary border-primary/30 font-semibold shadow-xs'
                    : 'border-border/80 bg-background/80 text-muted-foreground hover:bg-accent/70 hover:text-accent-foreground'
                )
              "
            >
              {{ item.value }}
            </PaginationItem>
            <PaginationEllipsis
              v-else
              :index="index"
              class="size-9 text-muted-foreground flex items-center justify-center"
            />
          </template>

          <PaginationNext
            :disabled="disabled || !canGoNext"
            size="icon-sm"
            class="border border-border/80 bg-background/80 hover:bg-accent/70 hover:text-accent-foreground rounded-md transition-colors duration-200"
            :title="t('common.next')"
            @click="goToNext"
          >
            <ChevronRight />
          </PaginationNext>

          <PaginationLast
            :disabled="disabled || !canGoNext"
            size="icon-sm"
            class="border border-border/80 bg-background/80 hover:bg-accent/70 hover:text-accent-foreground rounded-md transition-colors duration-200"
            :title="t('common.lastPage')"
            @click="goToLast"
          >
            <ChevronLast />
          </PaginationLast>
        </PaginationContent>
      </Pagination>

      <!-- Page input (desktop only) -->
      <div
        v-if="totalPages > 1"
        class="hidden sm:flex items-center gap-2 text-sm text-muted-foreground"
      >
        <span class="text-xs">{{ t("common.goToPage") }}</span>
        <Input
          v-model="pageInput"
          type="text"
          inputmode="numeric"
          class="h-8 w-12 text-center p-0 text-xs font-semibold focus-visible:ring-primary/30 bg-muted/40 border-border/60 hover:bg-muted/60 transition-colors rounded-md"
          :disabled="disabled"
          :aria-label="t('common.goToPage') + ' (' + t('common.of') + ' ' + totalPages + ')'"
          @keydown="handlePageInputKeydown"
          @blur="goToInputPage"
        />
        <span class="text-xs">{{ t("common.of") }} {{ totalPages }}</span>
      </div>
    </div>
  </div>
</template>
