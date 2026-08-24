<script setup lang="ts">
import { Calendar as CalendarIcon, X } from "@lucide/vue";
import { computed, onMounted, onUnmounted, ref, watch, type Ref } from "vue";
import { useI18n } from "vue-i18n";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { RangeCalendar } from "@/components/ui/range-calendar";
import { useWindowSize } from "@vueuse/core";
import { CalendarDate, parseDate } from "@internationalized/date";
import type { DateRange } from "reka-ui";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { configApi } from "@/services/api/config";
import { apiKeysApi, type ApiKeyRead } from "@/services/api/apiKeys";
import type { LogFilter } from "@/types/schemas";

const props = defineProps<{
  filters: LogFilter;
  activeTab: "proxy" | "audit" | "mcp" | "websearch";
}>();

const emit = defineEmits<{
  (e: "update:filters", value: LogFilter): void;
}>();

const { t } = useI18n();

type StatusFilter = "all" | "2xx" | "4xx" | "5xx";

const getLocalDateString = (isoString: string | undefined): string => {
  if (!isoString) return "";
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return "";
    const year = d.getFullYear();
    const month = String(d.getMonth() + 1).padStart(2, "0");
    const day = String(d.getDate()).padStart(2, "0");
    return `${year}-${month}-${day}`;
  } catch {
    return "";
  }
};

const getInitialStatusFilter = (filters: LogFilter): StatusFilter => {
  const from = filters.status_code_from;
  const to = filters.status_code_to;
  if (from === 200 && to === 299) return "2xx";
  if (from === 400 && to === 499) return "4xx";
  if (from === 500 && to === 599) return "5xx";
  return "all";
};

// Individual refs for each filter field
const searchText = ref(props.filters.search ?? "");
const modelText = ref(props.filters.model ?? "");
const providerText = ref(props.filters.provider ?? "");
const userText = ref(props.filters.user ?? "");
const apiKeyText = ref(props.filters.api_key ?? "");
const startDateInput = ref(getLocalDateString(props.filters.start_date));
const endDateInput = ref(getLocalDateString(props.filters.end_date));

const { width: windowWidth } = useWindowSize();
const tempDateRange = ref({ start: undefined, end: undefined }) as Ref<DateRange>;
const isDatePickerOpen = ref(false);

// Helper to parse date string to CalendarDate
const parseToCalendarDate = (dateStr: string): CalendarDate | undefined => {
  if (!dateStr) return undefined;
  try {
    return parseDate(dateStr);
  } catch {
    return undefined;
  }
};

// Helper to format CalendarDate to date string YYYY-MM-DD
const formatCalendarDate = (
  date: { year: number; month: number; day: number } | undefined
): string => {
  if (!date) return "";
  const year = date.year;
  const month = String(date.month).padStart(2, "0");
  const day = String(date.day).padStart(2, "0");
  return `${year}-${month}-${day}`;
};

// Watch parent filter changes to sync tempDateRange
watch(
  [startDateInput, endDateInput],
  ([start, end]) => {
    tempDateRange.value = {
      start: start ? parseToCalendarDate(start) : undefined,
      end: end ? parseToCalendarDate(end) : undefined,
    };
  },
  { immediate: true }
);

// Apply button handler
const applyCustomRange = () => {
  if (tempDateRange.value.start && tempDateRange.value.end) {
    startDateInput.value = formatCalendarDate(tempDateRange.value.start);
    endDateInput.value = formatCalendarDate(tempDateRange.value.end);
    isDatePickerOpen.value = false;
  }
};

// Display label for Date Range Button
const dateRangeText = computed(() => {
  if (startDateInput.value && endDateInput.value) {
    return `${startDateInput.value} — ${endDateInput.value}`;
  }
  if (startDateInput.value) return `${t("logs.from")} ${startDateInput.value}`;
  if (endDateInput.value) return `${t("logs.until")} ${endDateInput.value}`;
  return t("logs.dateRange");
});

const modelsList = ref<string[]>([]);
const providersList = ref<string[]>([]);
const apiKeysList = ref<ApiKeyRead[]>([]);

onMounted(async () => {
  try {
    const [models, providers] = await Promise.all([
      configApi.getModels(),
      configApi.getProviders(),
    ]);
    modelsList.value = Array.from(new Set(models.map((m) => m.name))).sort();
    const uniqueProviders = new Set<string>();
    providers.forEach((p) => {
      if (p.name) uniqueProviders.add(p.name);
      if (p.type) uniqueProviders.add(p.type);
    });
    providersList.value = Array.from(uniqueProviders).sort();
  } catch (error) {
    console.error("Failed to load models/providers list for filters:", error);
    // Leave lists empty - user can still type in the search field
    modelsList.value = [];
    providersList.value = [];
  }
  // API keys load independently: a non-admin sees only their own keys, and a
  // failure here must not blank the model/provider lists above.
  try {
    apiKeysList.value = await apiKeysApi.getApiKeys();
  } catch (error) {
    console.error("Failed to load API keys list for filters:", error);
    apiKeysList.value = [];
  }
});

const selectedModel = computed({
  get: () => modelText.value || "all",
  set: (val) => {
    modelText.value = val === "all" ? "" : val;
  },
});

const selectedProvider = computed({
  get: () => providerText.value || "all",
  set: (val) => {
    providerText.value = val === "all" ? "" : val;
  },
});

const selectedApiKey = computed({
  get: () => apiKeyText.value || "all",
  set: (val) => {
    apiKeyText.value = val === "all" ? "" : val;
  },
});

// Status filter chips
const statusFilter = ref<StatusFilter>(getInitialStatusFilter(props.filters));

const statusFilterOptions = [
  {
    label: "logs.filterAll",
    isI18n: true,
    value: "all" as const,
    activeClass: "bg-primary/10 text-primary border-primary/20",
  },
  {
    label: "2xx",
    isI18n: false,
    value: "2xx" as const,
    activeClass: "bg-status-success/15 text-status-success border-status-success/30",
  },
  {
    label: "4xx",
    isI18n: false,
    value: "4xx" as const,
    activeClass: "bg-status-warning/15 text-status-warning border-status-warning/30",
  },
  {
    label: "5xx",
    isI18n: false,
    value: "5xx" as const,
    activeClass: "bg-status-error/15 text-status-error border-status-error/30",
  },
];

// Build filter payload from current local state
const buildFilter = (): LogFilter => {
  const filter: LogFilter = {};

  filter.search = searchText.value || undefined;
  filter.model = modelText.value || undefined;
  filter.provider = providerText.value || undefined;
  filter.user = userText.value || undefined;
  filter.api_key = apiKeyText.value || undefined;

  if (statusFilter.value === "2xx") {
    filter.status_code_from = 200;
    filter.status_code_to = 299;
  } else if (statusFilter.value === "4xx") {
    filter.status_code_from = 400;
    filter.status_code_to = 499;
  } else if (statusFilter.value === "5xx") {
    filter.status_code_from = 500;
    filter.status_code_to = 599;
  } else {
    filter.status_code_from = undefined;
    filter.status_code_to = undefined;
  }

  filter.start_date = startDateInput.value
    ? new Date(startDateInput.value).toISOString()
    : undefined;

  if (endDateInput.value) {
    const end = new Date(endDateInput.value);
    // Make end date inclusive of the entire day (up to 23:59:59.999)
    end.setHours(23, 59, 59, 999);
    filter.end_date = end.toISOString();
  } else {
    filter.end_date = undefined;
  }

  return filter;
};

// Compare helper to avoid redundant emissions when synchronized from props
const isFilterEqual = (a: LogFilter, b: LogFilter) => {
  const keys: (keyof LogFilter)[] = [
    "search",
    "model",
    "provider",
    "user",
    "api_key",
    "status_code",
    "status_code_from",
    "status_code_to",
    "start_date",
    "end_date",
  ];
  return keys.every((k) => String(a[k] ?? "") === String(b[k] ?? ""));
};

// Debounce timer for text inputs
let debounceTimer: ReturnType<typeof setTimeout> | null = null;

const emitFilterUpdate = () => {
  const nextFilters = buildFilter();
  if (!isFilterEqual(nextFilters, props.filters)) {
    emit("update:filters", nextFilters);
  }
};

const emitDebounced = () => {
  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(emitFilterUpdate, 300);
};

const emitImmediate = () => {
  if (debounceTimer) clearTimeout(debounceTimer);
  emitFilterUpdate();
};

// Enter key in search triggers immediate search
const onSearchKeydown = (e: KeyboardEvent) => {
  if (e.key === "Enter") emitImmediate();
};

// Watch text fields with debounce
watch(searchText, emitDebounced);
watch(modelText, emitDebounced);
watch(providerText, emitDebounced);
watch(userText, emitDebounced);

// Selects, status chips and date fields trigger immediately (deliberate action)
watch(apiKeyText, emitImmediate);
watch(statusFilter, emitImmediate);
watch([startDateInput, endDateInput], emitImmediate);

// Sync from parent when filters are changed outside this panel (e.g. the
// API-key badge click in the table, tab-switch resets, deep links). Watching
// individual fields is required: the parent mutates one shared reactive
// object, so watching the object reference itself never fires.
watch(
  () => [
    props.filters.search,
    props.filters.model,
    props.filters.provider,
    props.filters.user,
    props.filters.api_key,
    props.filters.status_code,
    props.filters.status_code_from,
    props.filters.status_code_to,
    props.filters.start_date,
    props.filters.end_date,
  ],
  () => {
    const newFilters = props.filters;
    if (newFilters.search !== searchText.value) searchText.value = newFilters.search ?? "";
    if (newFilters.model !== modelText.value) modelText.value = newFilters.model ?? "";
    if (newFilters.provider !== providerText.value) providerText.value = newFilters.provider ?? "";
    if (newFilters.user !== userText.value) userText.value = newFilters.user ?? "";
    if (newFilters.api_key !== apiKeyText.value) apiKeyText.value = newFilters.api_key ?? "";

    // Sync status filter
    if (newFilters.status_code_from !== undefined || newFilters.status_code_to !== undefined) {
      const from = newFilters.status_code_from;
      const to = newFilters.status_code_to;
      if (from === 200 && to === 299) statusFilter.value = "2xx";
      else if (from === 400 && to === 499) statusFilter.value = "4xx";
      else if (from === 500 && to === 599) statusFilter.value = "5xx";
      else statusFilter.value = "all";
    } else if (newFilters.status_code === undefined) {
      statusFilter.value = "all";
    }

    const nextStart = getLocalDateString(newFilters.start_date);
    if (nextStart !== startDateInput.value) {
      startDateInput.value = nextStart;
    }
    const nextEnd = getLocalDateString(newFilters.end_date);
    if (nextEnd !== endDateInput.value) {
      endDateInput.value = nextEnd;
    }
  }
);

const hasDateRange = computed(() => startDateInput.value || endDateInput.value);

const clearDateRange = () => {
  startDateInput.value = "";
  endDateInput.value = "";
  tempDateRange.value = { start: undefined, end: undefined };
};

// Count active filters
const activeFilterCount = computed(() => {
  let count = 0;
  if (searchText.value?.trim()) count++;
  if (statusFilter.value !== "all") count++;
  if (modelText.value?.trim()) count++;
  if (providerText.value?.trim()) count++;
  if (userText.value?.trim()) count++;
  if (apiKeyText.value?.trim()) count++;
  if (startDateInput.value) count++;
  if (endDateInput.value) count++;
  return count;
});

// Clear all advanced filters inside the panel
const clearAllFilters = () => {
  searchText.value = "";
  modelText.value = "";
  providerText.value = "";
  userText.value = "";
  apiKeyText.value = "";
  statusFilter.value = "all";
  clearDateRange();
  // Emit the update so parent knows filters were cleared
  emit("update:filters", buildFilter());
};

defineExpose({ activeFilterCount, clearAllFilters });

onUnmounted(() => {
  if (debounceTimer) {
    clearTimeout(debounceTimer);
    emitFilterUpdate();
    debounceTimer = null;
  }
});
</script>

<template>
  <div class="w-full space-y-3 sm:space-y-4 filters-container">
    <!-- Active filters bar -->
    <div
      v-if="activeFilterCount > 0"
      class="flex items-center justify-between pb-2 border-b border-border/40"
    >
      <span class="text-xs font-medium text-muted-foreground">
        {{ t("logs.activeFilters", { count: activeFilterCount }) }}
      </span>
      <Button
        variant="ghost"
        size="sm"
        class="h-9 text-xs hover:bg-muted/50"
        @click="clearAllFilters"
      >
        {{ t("common.clearFilters") }}
      </Button>
    </div>
    <div
      class="filters-grid gap-3 sm:gap-4"
      :class="activeTab === 'proxy' ? 'filters-grid-6col' : 'filters-grid-4col'"
    >
      <!-- Search -->
      <div class="flex flex-col">
        <label for="log-search" class="text-xs font-medium text-muted-foreground mb-1.5">{{
          t("logs.search")
        }}</label>
        <Input
          id="log-search"
          v-model="searchText"
          class="w-full min-w-0 h-10 border-border/60"
          :placeholder="t('logs.search')"
          aria-label="Search logs"
          data-log-search
          @keydown="onSearchKeydown"
        />
      </div>

      <!-- Status chips -->
      <div class="flex flex-col">
        <span id="status-filter-label" class="text-xs font-medium text-muted-foreground mb-1.5">{{
          t("logs.status")
        }}</span>
        <div
          class="status-chips-scroll flex gap-1.5 w-full"
          role="group"
          aria-labelledby="status-filter-label"
        >
          <button
            v-for="option in statusFilterOptions"
            :key="option.value"
            @click="statusFilter = option.value"
            class="flex-1 h-10 text-xs font-medium rounded-md transition-colors cursor-pointer border flex items-center justify-center motion-safe:animate-in motion-safe:fade-in duration-200 whitespace-nowrap"
            :class="[
              statusFilter === option.value
                ? option.activeClass
                : 'bg-background border-border/60 text-muted-foreground hover:bg-muted/50 hover:text-foreground',
            ]"
            :aria-pressed="statusFilter === option.value"
          >
            {{ option.isI18n ? t(option.label) : option.label }}
          </button>
        </div>
      </div>

      <!-- Date range -->
      <div class="flex flex-col">
        <label class="text-xs font-medium text-muted-foreground mb-1.5">{{
          t("logs.dateRange")
        }}</label>

        <Popover v-model:open="isDatePickerOpen">
          <PopoverTrigger as-child>
            <Button
              variant="outline"
              class="w-full justify-start text-left font-normal px-3 h-10 border-border/60 transition-colors"
              :class="{ 'border-foreground text-foreground font-medium': hasDateRange }"
              :aria-label="t('logs.dateRange')"
            >
              <CalendarIcon
                class="mr-2 h-4 w-4 shrink-0 text-muted-foreground/80"
                aria-hidden="true"
              />
              <span class="truncate flex-1 text-xs font-mono">{{ dateRangeText }}</span>
              <X
                v-if="hasDateRange"
                class="ml-2 h-4 w-4 opacity-60 hover:opacity-100 shrink-0 cursor-pointer"
                @click.stop="clearDateRange"
                :aria-label="t('common.clearFilters')"
              />
            </Button>
          </PopoverTrigger>
          <PopoverContent class="w-auto p-0" align="start">
            <div class="flex flex-col gap-3 p-3">
              <RangeCalendar
                v-model="tempDateRange"
                :number-of-months="windowWidth >= 640 ? 2 : 1"
                class="rounded-md border border-border/40"
              />
              <div class="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  class="flex-1 text-xs cursor-pointer"
                  @click="clearDateRange"
                  :disabled="!hasDateRange && !tempDateRange.start && !tempDateRange.end"
                >
                  {{ t("common.clearFilters") }}
                </Button>
                <Button
                  size="sm"
                  class="flex-1 text-xs cursor-pointer"
                  :disabled="!tempDateRange.start || !tempDateRange.end"
                  @click="applyCustomRange"
                >
                  {{ t("home.apply") }}
                </Button>
              </div>
            </div>
          </PopoverContent>
        </Popover>
      </div>

      <!-- Proxy-specific filters -->
      <template v-if="activeTab === 'proxy'">
        <div class="flex flex-col">
          <label for="filter-model" class="text-xs font-medium text-muted-foreground mb-1.5">{{
            t("logs.model")
          }}</label>
          <Select v-model="selectedModel">
            <SelectTrigger id="filter-model" class="w-full h-10 border-border/60 cursor-pointer">
              <SelectValue :placeholder="t('logs.filterAll')" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">{{ t("logs.filterAll") }}</SelectItem>
              <SelectItem v-for="model in modelsList" :key="model" :value="model">
                <span class="font-mono text-xs">{{ model }}</span>
              </SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div class="flex flex-col">
          <label for="filter-provider" class="text-xs font-medium text-muted-foreground mb-1.5">{{
            t("logs.provider")
          }}</label>
          <Select v-model="selectedProvider">
            <SelectTrigger id="filter-provider" class="w-full h-10 border-border/60 cursor-pointer">
              <SelectValue :placeholder="t('logs.filterAll')" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">{{ t("logs.filterAll") }}</SelectItem>
              <SelectItem v-for="provider in providersList" :key="provider" :value="provider">
                <span class="font-mono text-xs">{{ provider }}</span>
              </SelectItem>
            </SelectContent>
          </Select>
        </div>
        <!-- API key filter (proxy tab only: endpoint logs are reliably
             key-bound, while audit/mcp/websearch entries may carry a null
             api_key_name and would silently vanish under a key filter) -->
        <div class="flex flex-col">
          <label for="filter-api-key" class="text-xs font-medium text-muted-foreground mb-1.5">{{
            t("logs.apiKeyFilter")
          }}</label>
          <Select v-model="selectedApiKey">
            <SelectTrigger id="filter-api-key" class="w-full h-10 border-border/60 cursor-pointer">
              <SelectValue :placeholder="t('logs.filterAll')" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">{{ t("logs.filterAll") }}</SelectItem>
              <SelectItem v-for="key in apiKeysList" :key="key.name" :value="key.name">
                <span class="font-mono text-xs">{{ key.name }}</span>
              </SelectItem>
            </SelectContent>
          </Select>
        </div>
      </template>

      <!-- Audit-specific filters -->
      <template v-else>
        <div class="flex flex-col">
          <label for="filter-user" class="text-xs font-medium text-muted-foreground mb-1.5">{{
            t("logs.user")
          }}</label>
          <Input
            id="filter-user"
            v-model="userText"
            placeholder="admin"
            class="w-full h-10 border-border/60"
            aria-label="Filter by user"
          />
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
/* ========================================
 * Container Query Grid
 * Adapts column count to container width
 * ======================================== */
.filters-container {
  container-type: inline-size;
}

.filters-grid {
  display: grid;
  grid-template-columns: 1fr;
}

/* Narrow: 2 columns */
@container (min-width: 480px) {
  .filters-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

/* Medium: 3 columns */
@container (min-width: 700px) {
  .filters-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

/* Wide: 4 columns */
@container (min-width: 900px) {
  .filters-grid {
    grid-template-columns: repeat(4, 1fr);
  }
}

/* Extra wide: 4 columns (audit/mcp/websearch tabs have 4 filter fields) */
@container (min-width: 1100px) {
  .filters-grid-4col {
    grid-template-columns: repeat(4, 1fr);
  }
}

/* Widest: 6 columns (proxy tab has 6 filter fields) */
@container (min-width: 1250px) {
  .filters-grid-6col {
    grid-template-columns: repeat(6, 1fr);
  }
}

/* ========================================
 * Scrollable Status Chips
 * Prevents wrapping on narrow containers
 * ======================================== */
.status-chips-scroll {
  overflow-x: auto;
  flex-wrap: nowrap;
  -webkit-overflow-scrolling: touch;
  scrollbar-width: none;
  /* Fade hint on the right edge when scrollable */
  mask-image: linear-gradient(to right, black calc(100% - 24px), transparent);
  -webkit-mask-image: linear-gradient(to right, black calc(100% - 24px), transparent);
}

.status-chips-scroll::-webkit-scrollbar {
  display: none;
}

/* ========================================
 * Touch Target Adjustments
 * Larger targets on touch devices
 * ======================================== */
@media (pointer: coarse) {
  .filters-grid :deep(input),
  .filters-grid :deep(button),
  .filters-grid :deep([role="combobox"]) {
    min-height: 44px;
  }
}
</style>
