<script setup lang="ts">
import { ArrowDown, Database } from "@lucide/vue";
import { computed, ref } from "vue";
import { useI18n } from "vue-i18n";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { formatCostWithPrecision, formatNumberWithSuffix } from "@/utils/format";

interface Props {
  byProvider: Array<{
    provider: string;
    requests: number;
    cost: number;
    input_tokens: number;
    output_tokens: number;
  }>;
}

const props = defineProps<Props>();
const { t } = useI18n();

type SortBy = "requests" | "cost";
const sortBy = ref<SortBy>("requests");

const sortedProviders = computed(() => {
  const result = [...props.byProvider];

  if (sortBy.value === "cost") {
    result.sort((a, b) => b.cost - a.cost || b.requests - a.requests);
    return result;
  }

  result.sort((a, b) => b.requests - a.requests || b.cost - a.cost);
  return result;
});

const totalRequests = computed(() => {
  return sortedProviders.value.reduce((sum, item) => sum + item.requests, 0);
});

// Calculate share percentage
const getShare = (requests: number): string => {
  if (totalRequests.value === 0) return "0.0";
  return ((requests / totalRequests.value) * 100).toFixed(1);
};

// Per-provider stable hue dot — same provider = same hue across sorts and
// pagination, matching the trends chart. Progress bars are monochrome
</script>

<template>
  <section class="flex flex-col">
    <div class="flex items-center justify-between px-4 sm:px-6 py-3 border-b border-border/60">
      <h2 class="flex items-center gap-1.5 text-sm md:text-base font-semibold text-foreground">
        <Database class="w-4 h-4 text-action-blue" />
        {{ t("home.byProvider") }}
      </h2>
    </div>

    <div class="flex-1 min-h-0">
      <!-- Interactive Column Headers (Table Style Click-to-Sort) -->
      <div
        v-if="sortedProviders.length > 0"
        class="flex items-center justify-between px-4 sm:px-6 py-1.5 border-b border-border/50 bg-muted/5 text-xs text-muted-foreground font-medium select-none"
      >
        <div class="flex items-center gap-2 flex-1">
          <span>#</span>
          <span>{{ t("labels.provider") }}</span>
        </div>
        <div class="flex items-center gap-6 shrink-0 font-sans">
          <button
            class="hover:text-foreground transition-colors flex items-center gap-0.5 cursor-pointer focus:outline-none"
            :class="{ 'text-foreground font-semibold': sortBy === 'requests' }"
            @click="sortBy = 'requests'"
          >
            {{ t("home.totalRequests") }}
            <ArrowDown v-if="sortBy === 'requests'" class="w-3.5 h-3.5 text-action-blue" />
          </button>
          <button
            class="hover:text-foreground transition-colors flex items-center gap-0.5 cursor-pointer focus:outline-none w-20 justify-end"
            :class="{ 'text-foreground font-semibold': sortBy === 'cost' }"
            @click="sortBy = 'cost'"
          >
            {{ t("home.costUsd") }}
            <ArrowDown v-if="sortBy === 'cost'" class="w-3.5 h-3.5 text-action-blue" />
          </button>
        </div>
      </div>

      <ScrollArea class="h-85 w-full">
        <div v-if="sortedProviders.length > 0" class="px-4 sm:px-6 py-1 space-y-0.5">
          <TooltipProvider>
            <div
              v-for="(item, index) in sortedProviders"
              :key="item.provider"
              class="group relative rounded-lg py-1.5 px-2 transition-colors hover:bg-muted/50 border border-transparent hover:border-border/40"
            >
              <div class="flex items-center gap-3 mb-1">
                <div class="flex items-center gap-2 flex-1 min-w-0">
                  <span
                    class="inline-flex items-center justify-center h-5 min-w-5 rounded border border-border/60 bg-background text-[11px] font-semibold text-muted-foreground"
                  >
                    {{ index + 1 }}
                  </span>
                  <span class="font-medium text-sm capitalize truncate">
                    {{ item.provider }}
                  </span>
                </div>

                <div class="flex items-center gap-6 text-right shrink-0">
                  <div>
                    <div class="font-mono text-[13px] font-medium text-foreground">
                      {{ formatNumberWithSuffix(item.requests) }}
                    </div>
                  </div>
                  <div class="w-20">
                    <div class="font-mono text-[13px] font-semibold text-action-blue">
                      {{ formatCostWithPrecision(item.cost, 2) }}
                    </div>
                  </div>
                </div>
              </div>

              <div
                class="mb-0.5 flex items-center justify-between text-[11px] text-muted-foreground"
              >
                <span>{{ getShare(item.requests) }}% {{ t("home.ofTotalRequests") }}</span>
                <span
                  >{{ formatNumberWithSuffix(item.input_tokens + item.output_tokens) }}
                  {{ t("logs.totalTokens") }}</span
                >
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
                    <div class="space-y-1">
                      <p>
                        {{ getShare(item.requests) }}%
                        {{ t("home.ofTotalRequests") }}
                      </p>
                      <p class="text-muted-foreground">
                        {{ formatNumberWithSuffix(item.input_tokens + item.output_tokens) }}
                        {{ t("logs.totalTokens") }}
                      </p>
                    </div>
                  </TooltipContent>
                </Tooltip>
              </div>
            </div>
          </TooltipProvider>
        </div>
        <div v-else class="py-12 text-center text-muted-foreground text-sm">
          {{ t("logs.noLogs") }}
        </div>
      </ScrollArea>
    </div>
  </section>
</template>
