<script setup lang="ts">
import { ArrowDown, ChevronLeft, ChevronRight, Cpu } from "@lucide/vue";
import { computed, ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { formatCostWithPrecision, formatNumberWithSuffix } from "@/utils/format";

interface Props {
  byModel: Array<{
    model: string;
    provider: string;
    requests: number;
    cost: number;
  }>;
}

const props = defineProps<Props>();
const { t } = useI18n();

// Pagination
const itemsPerPage = ref(10);
const currentPage = ref(1);

type SortBy = "requests" | "cost" | "name";
const sortBy = ref<SortBy>("requests");

// Sort models
const sortedModels = computed(() => {
  const result = [...props.byModel];

  if (sortBy.value === "cost") {
    result.sort((a, b) => b.cost - a.cost || b.requests - a.requests);
    return result;
  }

  if (sortBy.value === "name") {
    result.sort((a, b) => a.model.localeCompare(b.model));
    return result;
  }

  result.sort((a, b) => b.requests - a.requests || b.cost - a.cost);
  return result;
});

// Paginated results
const paginatedModels = computed(() => {
  const start = (currentPage.value - 1) * itemsPerPage.value;
  const end = start + itemsPerPage.value;
  return sortedModels.value.slice(start, end);
});

const totalPages = computed(() => Math.ceil(sortedModels.value.length / itemsPerPage.value));

const totalRequests = computed(() => {
  return sortedModels.value.reduce((sum, item) => sum + item.requests, 0);
});

watch([sortBy, itemsPerPage], () => {
  currentPage.value = 1;
});

watch(sortedModels, (items) => {
  const pageCount = Math.max(1, Math.ceil(items.length / itemsPerPage.value));
  if (currentPage.value > pageCount) {
    currentPage.value = pageCount;
  }
});

// Calculate share percentage
const getShare = (requests: number): string => {
  if (totalRequests.value === 0) return "0.0";
  return ((requests / totalRequests.value) * 100).toFixed(1);
};

// Per-model stable hue dot — same model = same hue across sorts and pagination,

// Truncate long model names
const truncateModelName = (name: string, maxLength = 28): string => {
  if (name.length <= maxLength) return name;
  return `${name.slice(0, maxLength)}...`;
};
</script>

<template>
  <section class="flex flex-col">
    <div class="flex items-center justify-between px-4 sm:px-6 py-3 border-b border-border/60">
      <h2 class="flex items-center gap-1.5 text-sm md:text-base font-semibold text-foreground">
        <Cpu class="w-4 h-4 text-action-amber" />
        {{ t("home.byModel") }}
      </h2>
    </div>

    <div class="flex-1 min-h-0">
      <!-- Interactive Column Headers (Table Style Click-to-Sort) -->
      <div
        v-if="sortedModels.length > 0"
        class="flex items-center justify-between px-4 sm:px-6 py-1.5 border-b border-border/50 bg-muted/5 text-xs text-muted-foreground font-medium select-none"
      >
        <div class="flex items-center gap-2 flex-1 min-w-0">
          <span>#</span>
          <button
            class="hover:text-foreground transition-colors flex items-center gap-0.5 cursor-pointer focus:outline-none truncate"
            :class="{ 'text-foreground font-semibold': sortBy === 'name' }"
            @click="sortBy = 'name'"
          >
            {{ t("logs.model") }}
            <ArrowDown v-if="sortBy === 'name'" class="w-3.5 h-3.5 text-action-amber" />
          </button>
        </div>

        <div class="flex items-center gap-6 shrink-0 font-sans">
          <button
            class="hover:text-foreground transition-colors flex items-center gap-0.5 cursor-pointer focus:outline-none"
            :class="{ 'text-foreground font-semibold': sortBy === 'requests' }"
            @click="sortBy = 'requests'"
          >
            {{ t("home.totalRequests") }}
            <ArrowDown v-if="sortBy === 'requests'" class="w-3.5 h-3.5 text-action-amber" />
          </button>
          <button
            class="hover:text-foreground transition-colors flex items-center gap-0.5 cursor-pointer focus:outline-none w-20 justify-end"
            :class="{ 'text-foreground font-semibold': sortBy === 'cost' }"
            @click="sortBy = 'cost'"
          >
            {{ t("home.costUsd") }}
            <ArrowDown v-if="sortBy === 'cost'" class="w-3.5 h-3.5 text-action-amber" />
          </button>
        </div>
      </div>

      <ScrollArea class="h-90 w-full">
        <div v-if="sortedModels.length > 0" class="px-4 sm:px-6 py-1 space-y-0.5">
          <TooltipProvider>
            <div
              v-for="(item, index) in paginatedModels"
              :key="`${item.provider}-${item.model}`"
              class="group relative rounded-lg py-1.5 px-2 transition-colors hover:bg-muted/50 border border-transparent hover:border-border/40"
            >
              <div class="flex items-center gap-3 mb-1">
                <div class="flex items-start gap-2 flex-1 min-w-0">
                  <span
                    class="inline-flex items-center justify-center h-5 min-w-5 rounded border border-border/60 bg-background text-[11px] font-semibold text-muted-foreground shrink-0 mt-0.5"
                  >
                    {{ (currentPage - 1) * itemsPerPage + index + 1 }}
                  </span>
                  <div class="min-w-0 flex-1">
                    <div class="flex items-center gap-1.5 min-w-0">
                      <Tooltip>
                        <TooltipTrigger as-child>
                          <span class="text-sm truncate font-medium cursor-default flex-1 min-w-0">
                            {{ truncateModelName(item.model) }}
                          </span>
                        </TooltipTrigger>
                        <TooltipContent class="font-mono text-xs max-w-xs">
                          <p>{{ item.model }}</p>
                        </TooltipContent>
                      </Tooltip>
                    </div>
                  </div>
                </div>

                <div class="flex items-center gap-6 text-right shrink-0">
                  <div>
                    <div class="font-mono text-[13px] font-medium text-foreground">
                      {{ formatNumberWithSuffix(item.requests) }}
                    </div>
                  </div>
                  <div class="w-20">
                    <div class="font-mono text-[13px] font-semibold text-action-amber">
                      {{ formatCostWithPrecision(item.cost, 2) }}
                    </div>
                  </div>
                </div>
              </div>

              <div
                class="mb-0.5 flex items-center justify-between text-[11px] text-muted-foreground"
              >
                <span>{{ getShare(item.requests) }}% {{ t("home.ofTotalRequests") }}</span>
                <span>{{ item.provider }}</span>
              </div>

              <div class="relative h-1.5 w-full rounded-full bg-muted overflow-hidden">
                <Tooltip>
                  <TooltipTrigger as-child>
                    <div
                      class="h-full rounded-full transition-all duration-500 bg-foreground/70"
                      :style="{ width: `${Math.max((item.requests / totalRequests) * 100, 2)}%` }"
                    />
                  </TooltipTrigger>
                  <TooltipContent side="bottom" class="font-mono text-xs">
                    {{ getShare(item.requests) }}%
                    {{ t("home.ofTotalRequests") }}
                  </TooltipContent>
                </Tooltip>
              </div>
            </div>
          </TooltipProvider>

          <!-- Pagination -->
          <Separator v-if="totalPages > 1" class="mt-2" />
          <div v-if="totalPages > 1" class="flex items-center justify-between pt-2 pb-1 px-2">
            <span class="text-xs text-muted-foreground">
              {{
                t("common.showingResults", {
                  count: paginatedModels.length,
                  total: sortedModels.length,
                })
              }}
            </span>
            <div class="flex items-center gap-1">
              <Button
                variant="ghost"
                size="sm"
                class="h-7 px-2"
                :disabled="currentPage === 1"
                :aria-label="t('common.previous')"
                @click="currentPage--"
              >
                <ChevronLeft class="w-3.5 h-3.5" aria-hidden="true" />
              </Button>
              <span class="text-xs text-muted-foreground px-2">
                {{ currentPage }} / {{ totalPages }}
              </span>
              <Button
                variant="ghost"
                size="sm"
                class="h-7 px-2"
                :disabled="currentPage === totalPages"
                :aria-label="t('common.next')"
                @click="currentPage++"
              >
                <ChevronRight class="w-3.5 h-3.5" aria-hidden="true" />
              </Button>
            </div>
          </div>
        </div>
        <div v-else class="py-12 text-center text-muted-foreground text-sm">
          {{ t("logs.noLogs") }}
        </div>
      </ScrollArea>
    </div>
  </section>
</template>
