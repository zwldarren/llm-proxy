<script setup lang="ts">
import {
  Activity,
  ArrowDown,
  ArrowUp,
  CornerDownRight,
  Eye,
  Filter,
  Globe,
  ScrollText,
  Shield,
  ThumbsUp,
  Wrench,
} from "@lucide/vue";
import {
  computed,
  defineAsyncComponent,
  markRaw,
  nextTick,
  onActivated,
  onDeactivated,
  onMounted,
  reactive,
  ref,
  shallowRef,
  watch,
} from "vue";
import { useI18n } from "vue-i18n";
import { useRoute, useRouter } from "vue-router";
import { toast } from "vue-sonner";
import { useAuthStore } from "@/stores/auth";
import EmptyState from "@/components/common/EmptyState.vue";
import AppLayout from "@/components/layout/AppLayout.vue";
import LogListItem from "@/components/common/LogListItem.vue";
import PaginationBar from "@/components/common/PaginationBar.vue";
import RefreshButton from "@/components/common/RefreshButton.vue";
import StatusBadge from "@/components/common/StatusBadge.vue";
import TableCellActions from "@/components/common/TableCellActions.vue";
import TableCellCode from "@/components/common/TableCellCode.vue";
import TableCellNumeric from "@/components/common/TableCellNumeric.vue";
import TableCellTimestamp from "@/components/common/TableCellTimestamp.vue";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import LoadingState from "@/components/common/LoadingState.vue";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useAutoRefresh } from "@/composables/useAutoRefresh";
import { useAuditLabels } from "@/composables/useAuditLabels";
import { useDebounceFn, useEventListener, useMediaQuery } from "@vueuse/core";
import { logsApi } from "@/services/api/logs";
import { meApi, type FeedbackSignal } from "@/services/api/me";
import { HttpError } from "@/services/http";
import type { LogFilter, LogListItem as LogListItemType, LogRead } from "@/types/schemas";
import {
  formatCost,
  formatDate,
  formatDuration,
  getActionFromEndpoint,
  getActor,
  getStatusType,
  formatApiKeyName,
} from "@/utils/format";
import { sanitizeHighlightText } from "@/utils/sanitize";
import { getProviderIconUrl, isMonoProvider } from "@/utils/icons";

const LogDetailsSheet = defineAsyncComponent(
  () => import("@/components/common/LogDetailsSheet.vue")
);
const LogsFilters = defineAsyncComponent(() => import("@/components/common/LogsFilters.vue"));
const AuditIntegrityDialog = defineAsyncComponent(
  () => import("@/components/common/AuditIntegrityDialog.vue")
);

defineOptions({ name: "LogsView" });

const { t } = useI18n();
const route = useRoute();
const router = useRouter();
const authStore = useAuthStore();

const isDesktop = useMediaQuery("(min-width: 1024px)");

// Log tab type
type LogTab = "proxy" | "audit" | "mcp" | "websearch";

const getInitialTab = (): LogTab => {
  const tabParam = route.query.tab as string;
  if (
    tabParam === "audit" ||
    tabParam === "proxy" ||
    tabParam === "mcp" ||
    tabParam === "websearch"
  )
    return tabParam;
  return "proxy";
};

const activeTab = ref<LogTab>(getInitialTab());

// Use shallowRef to avoid deep reactivity overhead on large log structures
const logs = shallowRef<LogListItemType[]>([]);
const totalLogs = ref(0);
const currentPage = ref(1);
const pageSize = ref(20);
const isInitialLoad = ref(true);
const isFetching = ref(false);
const isLoadingDetail = ref(false);
const isSwitchingTab = ref(false);
const consecutiveFailures = ref(0);

const cachedLatestTimestamp = ref<number | null>(null);

let fetchSequence = 0;

interface PageCacheEntry {
  items: LogListItemType[];
  total: number;
  timestamp: number;
}

const MAX_CACHED_PAGES = 20;

const pageCache = reactive<Record<LogTab, Map<number, PageCacheEntry>>>({
  proxy: markRaw(new Map()),
  audit: markRaw(new Map()),
  mcp: markRaw(new Map()),
  websearch: markRaw(new Map()),
});

const isComponentActive = ref(true);

const filters = reactive<LogFilter>({
  status_code: undefined,
  model: "",
  provider: "",
  user: "",
  api_key: "",
  endpoint: "",
  start_date: undefined,
  end_date: undefined,
});

const showFilters = ref(false);
const logsFiltersRef = ref<InstanceType<typeof LogsFilters> | null>(null);
const selectedLog = ref<LogRead | null>(null);
const showDetailDialog = ref(false);
const showIntegrityDialog = ref(false);

const { autoRefresh, startAutoRefresh, stopAutoRefresh, toggleAutoRefresh } = useAutoRefresh({
  interval: 5000,
  shouldPauseRefresh: () => showDetailDialog.value,
  onRefreshStateChange: () => {
    checkAndRefresh();
  },
});

const checkAndRefresh = async () => {
  try {
    const stats = await logsApi.getLogStats({
      log_type: logType.value,
      start_date: filters.start_date,
      end_date: filters.end_date,
    });

    if (
      cachedLatestTimestamp.value === null ||
      stats.latest_timestamp !== cachedLatestTimestamp.value ||
      stats.total !== totalLogs.value
    ) {
      cachedLatestTimestamp.value = stats.latest_timestamp;
      await fetchLogs(false);
    }
  } catch {
    await fetchLogs(false);
  }
};

const logType = computed(() => {
  switch (activeTab.value) {
    case "proxy":
      return "endpoint";
    case "audit":
      return "audit";
    case "mcp":
      return "mcp";
    case "websearch":
      return "web_search";
    default:
      return "endpoint";
  }
});

const activeFilterCount = computed(() => {
  let count = 0;
  if (filters.search?.trim()) count++;
  if (
    filters.status_code_from !== undefined ||
    filters.status_code_to !== undefined ||
    filters.status_code !== undefined
  )
    count++;
  if (filters.model?.trim()) count++;
  if (filters.provider?.trim()) count++;
  if (filters.user?.trim()) count++;
  if (filters.api_key?.trim()) count++;
  if (filters.start_date) count++;
  if (filters.end_date) count++;
  return count;
});

const cleanupPageCache = (tab: LogTab) => {
  const cache = pageCache[tab];
  if (cache.size <= MAX_CACHED_PAGES) return;
  const entries = [...cache.entries()].sort((a, b) => a[1].timestamp - b[1].timestamp);
  for (const [p] of entries.slice(0, cache.size - MAX_CACHED_PAGES)) {
    cache.delete(p);
  }
};

const invalidatePageCache = (tab?: LogTab) => {
  if (tab) {
    pageCache[tab].clear();
  } else {
    for (const cache of Object.values(pageCache)) {
      cache.clear();
    }
  }
};

const performFetch = async (
  filters: LogFilter,
  page: number
): Promise<{ items: LogListItemType[]; total: number }> => {
  const effectiveFilters = {
    ...filters,
    log_type: logType.value,
    endpoint: undefined,
    page,
    page_size: pageSize.value,
  };
  return logsApi.getLogs(effectiveFilters);
};

const applyPageCache = (tab: LogTab, page: number, entry: PageCacheEntry) => {
  logs.value = entry.items;
  totalLogs.value = entry.total;
  isInitialLoad.value = false;
  pageCache[tab].set(page, entry);
  cleanupPageCache(tab);
};

const fetchLogs = async (showLoading = true) => {
  const currentSequence = ++fetchSequence;
  const tab = activeTab.value;
  const page = currentPage.value;

  const cached = pageCache[tab].get(page);
  if (cached) {
    logs.value = cached.items;
    totalLogs.value = cached.total;
    isInitialLoad.value = false;
    isFetching.value = showLoading;
  } else if (showLoading) {
    isFetching.value = true;
  }

  try {
    const res = await performFetch(filters, page);

    if (currentSequence !== fetchSequence) return;

    consecutiveFailures.value = 0;

    if (res.items.length > 0) {
      cachedLatestTimestamp.value = res.items[0]!.timestamp;
    }

    const entry: PageCacheEntry = {
      items: res.items,
      total: res.total,
      timestamp: Date.now(),
    };
    applyPageCache(tab, page, entry);
  } catch {
    if (currentSequence === fetchSequence) {
      consecutiveFailures.value++;

      if (showLoading) {
        toast.error(t("errors.fetchFailed"));
      } else if (consecutiveFailures.value >= 3) {
        toast.warning(t("errors.fetchFailedBackground"));
        consecutiveFailures.value = 0;
      }
    }
  } finally {
    if (currentSequence === fetchSequence) {
      isFetching.value = false;
      isInitialLoad.value = false;
    }
  }

  if (currentSequence === fetchSequence) {
    schedulePrefetch(tab, page);
  }
};

const prefetchPage = async (tab: LogTab, page: number) => {
  if (page < 1 || pageCache[tab].has(page)) return;
  try {
    const res = await performFetch(filters, page);
    const entry: PageCacheEntry = {
      items: res.items,
      total: res.total,
      timestamp: Date.now(),
    };
    pageCache[tab].set(page, entry);
    cleanupPageCache(tab);
  } catch {
    // Silent fail for prefetch
  }
};

const schedulePrefetch = (tab: LogTab, page: number) => {
  setTimeout(() => {
    if (activeTab.value !== tab) return;
    void prefetchPage(tab, page - 1);
    void prefetchPage(tab, page + 1);
  }, 200);
};

const debouncedFetchLogs = useDebounceFn(() => {
  fetchLogs(true);
}, 300);

watch(
  [
    () => filters.search,
    () => filters.status_code,
    () => filters.status_code_from,
    () => filters.status_code_to,
    () => filters.start_date,
    () => filters.end_date,
  ],
  () => {
    if (isSwitchingTab.value) return;
    currentPage.value = 1;
    invalidatePageCache();
    debouncedFetchLogs();
  }
);

watch(
  [() => filters.model, () => filters.provider, () => filters.user, () => filters.api_key],
  () => {
    if (isSwitchingTab.value) return;
    currentPage.value = 1;
    invalidatePageCache(activeTab.value);
    debouncedFetchLogs();
  }
);

watch([currentPage, pageSize], () => {
  if (isSwitchingTab.value) return;
  fetchLogs(true);
});

// Reset tab to proxy when admin status changes (audit tab may disappear)
watch(
  () => authStore.isAdmin,
  (isAdmin) => {
    if (!isAdmin && activeTab.value === "audit") {
      activeTab.value = "proxy";
    }
  }
);

// Tab changes
watch(activeTab, async (newTab) => {
  isSwitchingTab.value = true;
  currentPage.value = 1;
  router.replace({ query: { ...route.query, tab: newTab } });

  if (newTab !== "proxy") {
    filters.model = "";
    filters.provider = "";
    // The API key filter is proxy-tab only: non-endpoint logs may carry a
    // null api_key_name, and a stale key filter would silently hide them.
    filters.api_key = "";
  } else {
    filters.user = "";
  }

  const cached = pageCache[newTab].get(1);
  if (cached) {
    logs.value = cached.items;
    totalLogs.value = cached.total;
    isInitialLoad.value = false;
    await fetchLogs(false);
  } else {
    isInitialLoad.value = true;
    await fetchLogs(true);
  }

  await nextTick();
  isSwitchingTab.value = false;
});

// Sync activeTab from route query tab param (handles browser back/forward and external link clicks)
watch(
  () => route.query.tab,
  (newTab) => {
    const tab = (newTab as LogTab) || "proxy";
    if (
      (tab === "proxy" || tab === "audit" || tab === "mcp" || tab === "websearch") &&
      tab !== activeTab.value
    ) {
      activeTab.value = tab;
    }
  }
);

const hasAnyActiveFilter = computed(() => {
  return Boolean(
    filters.search?.trim() ||
    filters.status_code !== undefined ||
    filters.model?.trim() ||
    filters.provider?.trim() ||
    filters.user?.trim() ||
    filters.api_key?.trim() ||
    filters.start_date ||
    filters.end_date
  );
});

const manualRefresh = () => {
  fetchLogs(true);
};

const STATUS_RANGES: Record<string, { from: number; to: number }> = {
  "2xx": { from: 200, to: 299 },
  "4xx": { from: 400, to: 499 },
  "5xx": { from: 500, to: 599 },
};

const applyStatusFromQuery = (status: unknown): boolean => {
  if (typeof status !== "string") return false;
  const range = STATUS_RANGES[status];
  if (!range) return false;
  filters.status_code_from = range.from;
  filters.status_code_to = range.to;
  showFilters.value = true;
  return true;
};

onMounted(() => {
  if (isComponentActive.value) {
    // Apply a dashboard deep-link (e.g. ?status=5xx) before the first fetch so
    // the initial result set already reflects the error filter.
    applyStatusFromQuery(route.query.status);
    fetchLogs(true);
    startAutoRefresh();
  }
});

// React to dashboard deep-links arriving while the logs view is already mounted
// (KeepAlive re-activation or in-app navigation).
watch(
  () => route.query.status,
  (status) => {
    if (status !== undefined) applyStatusFromQuery(status);
  }
);

onActivated(() => {
  isComponentActive.value = true;
  fetchLogs(false);
  if (autoRefresh.value) {
    startAutoRefresh();
  }
});

onDeactivated(() => {
  isComponentActive.value = false;
  showDetailDialog.value = false;
  stopAutoRefresh();
});
watch(showDetailDialog, (isOpen) => {
  if (!isOpen && autoRefresh.value) {
    startAutoRefresh();
  }
});

const handleViewDetails = async (log: LogListItemType) => {
  if (isLoadingDetail.value) return;
  isLoadingDetail.value = true;
  try {
    const fullLog = await logsApi.getLog(log.request_id);
    selectedLog.value = fullLog;
    showDetailDialog.value = true;
  } catch {
    // Error handled by UI
  } finally {
    isLoadingDetail.value = false;
  }
};

// Power-user keyboard layer for the logs table:
//   `/` focuses the search field (opening the filters panel if collapsed)
//   `j` / `k` move row focus down/up; Enter (already bound on rows) opens detail.
// Ignored while typing in any input/textarea/select so it never fights the
// user. Rows carry `data-log-row` so we can move focus without per-row refs.
const isTypingTarget = (el: EventTarget | null): boolean => {
  if (!(el instanceof HTMLElement)) return false;
  const tag = el.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable;
};

const focusLogSearch = () => {
  showFilters.value = true;
  // The filters panel animates in; wait a tick for the input to mount.
  requestAnimationFrame(() => {
    document.querySelector<HTMLInputElement>("[data-log-search]")?.focus();
  });
};

const moveRowFocus = (direction: 1 | -1) => {
  const rows = Array.from(document.querySelectorAll<HTMLElement>('[data-log-row][tabindex="0"]'));
  if (rows.length === 0) return;
  const current = rows.findIndex((r) => r === document.activeElement);
  const nextIndex =
    current === -1 ? 0 : Math.min(rows.length - 1, Math.max(0, current + direction));
  rows[nextIndex]?.focus();
};

useEventListener("keydown", (e: KeyboardEvent) => {
  if (showDetailDialog.value) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (isTypingTarget(e.target)) return;
  if (e.key === "/") {
    e.preventDefault();
    focusLogSearch();
  } else if (e.key === "j" || e.key === "ArrowDown") {
    // ArrowDown is only hijacked when a row is already focused (so it doesn't
    // break native page scroll for non-keyboard users).
    if (e.key === "j" || document.activeElement?.closest("[data-log-row]")) {
      e.preventDefault();
      moveRowFocus(1);
    }
  } else if (e.key === "k" || e.key === "ArrowUp") {
    if (e.key === "k" || document.activeElement?.closest("[data-log-row]")) {
      e.preventDefault();
      moveRowFocus(-1);
    }
  }
});

// Pagination handler - unified via go-to-page event
const totalPages = computed(() => Math.ceil(totalLogs.value / pageSize.value));

const goToPage = (page: number) => {
  currentPage.value = page;
  fetchLogs(true);
};

const getProviderModelName = (log: LogListItemType): string | null =>
  (log.log_metadata?.provider_model_name as string | undefined) || null;

// --- Smart-routing feedback ---------------------------------------------

// Recorded feedback signals keyed by request_id. Rows resolved by smart
// routing show three rating buttons until feedback is recorded, then a
// static indicator of the recorded signal.
const feedbackByRequestId = ref<Record<string, FeedbackSignal>>({});
// Non-reactive bookkeeping: IDs already queried (so page refreshes don't
// refetch) and IDs with an in-flight submission.
const feedbackQueriedIds = new Set<string>();
const feedbackSubmittingIds = new Set<string>();

const FEEDBACK_ICONS = { ok: ThumbsUp, weak: ArrowUp, strong: ArrowDown } as const;
const FEEDBACK_LABEL_KEYS = {
  ok: "logs.feedbackOk",
  weak: "logs.feedbackWeak",
  strong: "logs.feedbackStrong",
} as const;
const FEEDBACK_BADGE_CLASS = {
  ok: "text-status-success bg-status-success/15",
  weak: "text-status-error bg-status-error/15",
  strong: "text-status-warning bg-status-warning/15",
} as const;

const getResolvedModel = (log: LogListItemType): string | null =>
  (log.log_metadata?.routing?.resolved_model as string | undefined) || null;

// Batch-load feedback state for the routed rows of the current page.
watch(logs, async (items) => {
  if (activeTab.value !== "proxy") return;
  const ids = items
    .filter((log) => getResolvedModel(log) !== null)
    .map((log) => log.request_id)
    .filter((id) => !feedbackQueriedIds.has(id));
  if (ids.length === 0) return;
  ids.forEach((id) => feedbackQueriedIds.add(id));
  try {
    const recorded = await meApi.getFeedback(ids);
    if (Object.keys(recorded).length > 0) {
      feedbackByRequestId.value = { ...feedbackByRequestId.value, ...recorded };
    }
  } catch {
    // Allow a later retry when the state fetch fails.
    ids.forEach((id) => feedbackQueriedIds.delete(id));
  }
});

const submitFeedback = async (log: LogListItemType, signal: FeedbackSignal) => {
  const id = log.request_id;
  if (feedbackByRequestId.value[id] || feedbackSubmittingIds.has(id)) return;
  feedbackSubmittingIds.add(id);
  try {
    await meApi.submitFeedback(id, signal);
    feedbackByRequestId.value = { ...feedbackByRequestId.value, [id]: signal };
    toast.success(t("logs.feedbackRecorded"));
  } catch (error) {
    if (error instanceof HttpError && error.status === 409) {
      // Already recorded (possibly by another admin); pull the stored signal.
      try {
        const recorded = await meApi.getFeedback([id]);
        const stored = recorded[id];
        if (stored) feedbackByRequestId.value = { ...feedbackByRequestId.value, [id]: stored };
      } catch {
        // Keep the row interactive if the state refresh fails.
      }
    } else {
      toast.error(t("logs.feedbackFailed"));
    }
  } finally {
    feedbackSubmittingIds.delete(id);
  }
};

// Token helpers
const getTokenCount = (log: LogListItemType, type: "prompt" | "completion" | "total"): number => {
  const meta = log.log_metadata as Record<string, unknown> | undefined;
  const key =
    type === "prompt"
      ? "prompt_tokens"
      : type === "completion"
        ? "completion_tokens"
        : "total_tokens";
  const directValue = (log as unknown as Record<string, unknown>)[key] as number | undefined;
  return directValue || (meta?.[key] as number) || 0;
};

const formatTokenBreakdown = (log: LogListItemType): string => {
  const input = getTokenCount(log, "prompt");
  const output = getTokenCount(log, "completion");
  const total = getTokenCount(log, "total");
  if (input === 0 && output === 0 && total === 0) return "-";
  if (output === 0 && input > 0) return input.toLocaleString();
  return `${input.toLocaleString()} / ${output.toLocaleString()}`;
};

const getTokenTooltip = (log: LogListItemType): string => {
  const input = getTokenCount(log, "prompt");
  const output = getTokenCount(log, "completion");
  if (output === 0 && input > 0) return `Input: ${input.toLocaleString()}`;
  return `Input: ${input.toLocaleString()} | Output: ${output.toLocaleString()}`;
};

const formatTTFT = (log: LogListItemType): string => {
  const ttft = (log.log_metadata?.ttft_ms as number | undefined) ?? log.ttft_ms;
  return formatDurationOrFailed(log, ttft);
};

// A 5xx row that records 0/1ms or no timing didn't really complete a round
// trip — render "failed" instead of a misleading "0ms" / "-".
const isFailedRequest = (log: LogListItemType): boolean => (log.status_code ?? 0) >= 500;

const formatDurationOrFailed = (log: LogListItemType, ms: number | null | undefined): string => {
  if (isFailedRequest(log) && (ms == null || ms <= 1)) return t("logs.failed");
  return formatDuration(ms);
};

const getTimingTooltip = (log: LogListItemType): string => {
  const ttft = (log.log_metadata?.ttft_ms as number | undefined) ?? log.ttft_ms;
  const duration = log.response_time_ms;
  const isStreaming = log.log_metadata?.streaming as boolean | undefined;
  if (isStreaming) {
    return `TTFT: ${formatDurationOrFailed(log, ttft)} | ${t("logs.duration")}: ${formatDurationOrFailed(log, duration)}`;
  }
  return `${t("logs.duration")}: ${formatDurationOrFailed(log, duration)}`;
};

// TabsTrigger class for tab navigation - extracted to avoid repetition
const tabTriggerClass =
  "relative h-10 rounded-none border-b border-border/30 bg-transparent px-1 pb-3 pt-2 text-xs font-medium text-muted-foreground shadow-none transition-colors data-[state=active]:border-foreground data-[state=active]:bg-transparent data-[state=active]:text-foreground data-[state=active]:shadow-none dark:data-[state=active]:text-foreground gap-1.5";

// MCP helpers
const getMcpOperation = (log: LogListItemType): string =>
  (log.log_metadata?.mcp_operation as string | undefined) || "-";
const getMcpResourceName = (log: LogListItemType): string =>
  (log.log_metadata?.mcp_resource_name as string | undefined) || "-";
const getMcpServer = (log: LogListItemType): string =>
  (log.log_metadata?.mcp_server as string | undefined) || log.model || "-";

const getWebSearchQuery = (log: LogListItemType): string =>
  (log.log_metadata?.web_search_query as string | undefined) || "-";
const getWebSearchResultCount = (log: LogListItemType): number =>
  (log.log_metadata?.web_search_result_count as number | undefined) || 0;
const getWebSearchStatus = (log: LogListItemType): string =>
  (log.log_metadata?.web_search_status as string | undefined) || "unknown";

// Audit semantic labels for the audit list/table.
const { formatEventType, formatActionCategory } = useAuditLabels();

// Backend-aware action label for audit list rows. Prefers the backend's
// curated action_category (verb) and falls back to endpoint-derived labels
// for legacy logs without classification.
const auditListAction = (log: LogListItemType): string => {
  const cat = log.action_category;
  if (cat) {
    const verb = formatActionCategory(cat);
    // resource_type is not part of the list payload, so derive the object
    // from the endpoint path.
    const parts = log.endpoint.split("/").filter(Boolean);
    const resource = parts.at(-1) || parts.at(-2) || "unknown";
    const formattedResource =
      resource.charAt(0).toUpperCase() + resource.slice(1).replace(/-/g, " ");
    return `${verb} ${formattedResource}`;
  }
  return getActionFromEndpoint(log.method, log.endpoint);
};
</script>

<template>
  <AppLayout layoutMode="full">
    <template #header>
      <header class="config-header-bar flex flex-col pt-4 px-4 sm:px-6">
        <div class="flex items-center justify-between w-full pb-3">
          <div class="flex items-center gap-3 min-w-0">
            <div
              class="flex items-center justify-center h-8 w-8 rounded-md bg-muted text-foreground border border-border/50 shrink-0"
            >
              <ScrollText class="w-4 h-4 text-foreground/80" />
            </div>
            <div class="space-y-0.5">
              <h1
                class="brand-heading text-xl sm:text-2xl text-foreground leading-tight flex items-center gap-2"
              >
                <span>{{ t("logs.title") }}</span>
                <!-- Pulsing Live Badge -->
                <div
                  v-if="autoRefresh"
                  class="flex items-center gap-1.5 px-2 py-0.5 bg-status-success/10 text-status-success border border-status-success/20 rounded-full text-[11px] font-medium"
                >
                  <span class="relative flex h-1.5 w-1.5">
                    <span
                      class="animate-ping absolute inline-flex h-full w-full rounded-full bg-status-success opacity-75"
                    ></span>
                    <span
                      class="relative inline-flex rounded-full h-1.5 w-1.5 bg-status-success"
                    ></span>
                  </span>
                  <span>{{ t("logs.live") }}</span>
                </div>
              </h1>
              <p class="text-muted-foreground text-xs max-w-2xl hidden sm:block truncate">
                {{
                  activeTab === "proxy"
                    ? t("logs.proxyDescription")
                    : activeTab === "audit"
                      ? t("logs.auditDescription")
                      : activeTab === "mcp"
                        ? t("logs.mcpDescription")
                        : t("logs.webSearchDescription")
                }}
              </p>
            </div>
          </div>

          <div class="flex items-center gap-2">
            <Button
              v-if="activeTab === 'audit' && authStore.isAdmin"
              variant="secondary"
              size="sm"
              class="h-9 transition-colors"
              @click="showIntegrityDialog = true"
            >
              <Shield class="w-4 h-4 sm:mr-2" />
              <span class="hidden sm:inline">{{ t("logs.audit.verifyIntegrity") }}</span>
            </Button>
            <RefreshButton
              :is-loading="isFetching"
              :is-auto-refresh="autoRefresh"
              tooltip-prefix="logs"
              @refresh="manualRefresh"
              @toggle-auto-refresh="toggleAutoRefresh"
            />
            <Button
              variant="secondary"
              size="sm"
              class="h-9 transition-colors"
              @click="showFilters = !showFilters"
              :class="{
                'bg-foreground text-background hover:bg-foreground/90 hover:text-background':
                  showFilters,
              }"
              :aria-expanded="showFilters"
            >
              <Filter class="w-4 h-4 sm:mr-2" />
              <span class="hidden sm:inline">{{ t("logs.filters") }}</span>
              <Badge
                v-if="activeFilterCount"
                variant="default"
                class="ml-1.5 h-5 min-w-5 px-1.5 text-xs"
              >
                {{ activeFilterCount }}
              </Badge>
            </Button>
          </div>
        </div>

        <!-- Inline Navigation Tabs aligned with the bottom border of the header -->
        <div class="-mb-[1px] flex items-center overflow-x-auto scrollbar-none">
          <Tabs v-model="activeTab" class="w-auto">
            <TabsList class="h-10 p-0 bg-transparent flex gap-6 rounded-none border-b-0">
              <TabsTrigger value="proxy" :class="tabTriggerClass">
                <Activity class="w-3.5 h-3.5" />
                <span>{{ t("nav.proxyLogs") }}</span>
              </TabsTrigger>
              <TabsTrigger v-if="authStore.isAdmin" value="audit" :class="tabTriggerClass">
                <Shield class="w-3.5 h-3.5" />
                <span>{{ t("nav.auditLogs") }}</span>
              </TabsTrigger>
              <TabsTrigger value="mcp" :class="tabTriggerClass">
                <Wrench class="w-3.5 h-3.5" />
                <span>{{ t("nav.mcpLogs") }}</span>
              </TabsTrigger>
              <TabsTrigger value="websearch" :class="tabTriggerClass">
                <Globe class="w-3.5 h-3.5" />
                <span>{{ t("nav.webSearchLogs") }}</span>
              </TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
      </header>
    </template>

    <div class="flex-1 flex flex-col overflow-hidden">
      <!-- Collapsible Advanced Filters panel -->
      <div
        v-if="showFilters"
        class="flex-none border-b border-border/60 bg-muted/15 backdrop-blur-sm px-4 sm:px-6 py-4 animate-in slide-in-from-top-2 duration-200"
      >
        <LogsFilters
          ref="logsFiltersRef"
          :filters="filters"
          :active-tab="activeTab"
          @update:filters="(newFilters) => Object.assign(filters, newFilters)"
        />
      </div>

      <!-- Main Content Area: Scrollable Table Container -->
      <div class="flex-1 overflow-hidden flex flex-col relative">
        <!-- Loading bar for non-initial fetches -->
        <div
          v-if="isFetching && !isInitialLoad"
          class="h-0.5 bg-primary/30 overflow-hidden flex-none"
        >
          <div class="h-full bg-primary/60 animate-loading-bar" />
        </div>

        <!-- Initial loading spinner -->
        <div
          v-if="isInitialLoad && isFetching"
          class="flex-1 flex flex-col items-center justify-center"
        >
          <LoadingState mode="spinner" />
        </div>

        <!-- Empty state -->
        <EmptyState
          v-else-if="logs.length === 0 && !isFetching"
          :text="hasAnyActiveFilter ? t('common.noMatchingResults') : t('logs.noLogs')"
          :icon="
            hasAnyActiveFilter
              ? Filter
              : activeTab === 'proxy'
                ? Activity
                : activeTab === 'audit'
                  ? Shield
                  : activeTab === 'mcp'
                    ? Wrench
                    : Globe
          "
          class="flex-1 animate-in fade-in duration-300"
          role="status"
          aria-live="polite"
        />

        <!-- Scrollable data display -->
        <div v-else class="flex-1 overflow-auto">
          <!-- Desktop view -->
          <template v-if="isDesktop">
            <!-- Proxy Logs Table -->
            <Table
              v-if="activeTab === 'proxy'"
              class="table-modern"
              container-class="border-0 bg-transparent rounded-none"
            >
              <TableHeader>
                <TableRow class="hover:bg-transparent hover:border-l-transparent">
                  <TableHead class="table-head-cell">{{ t("logs.timestamp") }}</TableHead>
                  <TableHead class="text-xs hidden md:table-cell">{{ t("logs.apiKey") }}</TableHead>
                  <TableHead class="text-xs">{{ t("logs.model") }}</TableHead>
                  <TableHead class="text-xs hidden sm:table-cell">{{
                    t("logs.provider")
                  }}</TableHead>
                  <TableHead>{{ t("logs.status") }}</TableHead>
                  <TableHead class="table-cell-mono hidden md:table-cell">{{
                    t("logs.ttft")
                  }}</TableHead>
                  <TableHead class="table-cell-mono">{{ t("logs.duration") }}</TableHead>
                  <TableHead class="table-cell-mono text-right hidden sm:table-cell">{{
                    t("logs.tokens")
                  }}</TableHead>
                  <TableHead class="table-cell-mono text-right hidden md:table-cell">{{
                    t("logs.cost")
                  }}</TableHead>
                  <TableHead class="w-36"></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody class="row-stagger">
                <TableRow
                  v-for="log in logs"
                  :key="log.request_id"
                  :data-log-row="log.request_id"
                  :aria-label="`${t('logs.viewDetails')} — ${log.model || log.endpoint || log.request_id} (${log.status_code ?? '—'})`"
                  @click="handleViewDetails(log)"
                  class="table-row-hover relative"
                  tabindex="0"
                  role="button"
                  @keydown.enter="handleViewDetails(log)"
                  @keydown.space.prevent="handleViewDetails(log)"
                >
                  <TableCellTimestamp>{{ formatDate(log.timestamp) }}</TableCellTimestamp>
                  <TableCell class="text-xs hidden md:table-cell">
                    <template v-if="log.auth_method === 'jwt'">
                      <Badge variant="outline" class="font-mono">
                        {{ log.user_identity || t("team.admin") }}
                      </Badge>
                    </template>
                    <template v-else-if="log.api_key_name">
                      <Badge variant="outline" class="font-mono">{{
                        formatApiKeyName(log.api_key_name)
                      }}</Badge>
                    </template>
                    <template v-else>
                      <span class="text-muted-foreground">-</span>
                    </template>
                  </TableCell>
                  <TableCell class="text-xs overflow-hidden">
                    <div class="flex flex-col gap-1 overflow-hidden">
                      <Badge variant="secondary" class="font-mono self-start max-w-full truncate">
                        <!-- eslint-disable-next-line vue/no-v-html -->
                        <span
                          v-html="sanitizeHighlightText(log.model, filters.search || '')"
                        ></span>
                      </Badge>
                      <div
                        v-if="getProviderModelName(log) && getProviderModelName(log) !== log.model"
                        class="flex items-center gap-1 text-[11px] text-muted-foreground/80 font-mono pl-1"
                        :title="t('logs.providerModel')"
                      >
                        <CornerDownRight class="w-3 h-3 text-muted-foreground/50 shrink-0" />
                        <span class="truncate max-w-[30%]">{{ getProviderModelName(log) }}</span>
                      </div>
                    </div>
                  </TableCell>
                  <TableCell class="text-xs hidden sm:table-cell">
                    <div class="flex items-center gap-1.5">
                      <div
                        v-if="getProviderIconUrl(log.provider || '')"
                        class="w-4 h-4 rounded flex items-center justify-center overflow-hidden bg-background border border-border/50 shrink-0"
                      >
                        <img
                          :src="getProviderIconUrl(log.provider || '')!"
                          :class="[
                            isMonoProvider(log.provider || '') ? 'icon-mono' : null,
                            'w-3 h-3 object-contain',
                          ]"
                          loading="lazy"
                        />
                      </div>
                      <!-- eslint-disable-next-line vue/no-v-html -->
                      <span
                        class="text-muted-foreground font-medium"
                        v-html="sanitizeHighlightText(log.provider, filters.search || '')"
                      ></span>
                    </div>
                  </TableCell>
                  <TableCell>
                    <StatusBadge variant="status" :status="getStatusType(log.status_code)">
                      {{ log.status_code }}
                    </StatusBadge>
                  </TableCell>
                  <TableCellCode class="hidden md:table-cell" :title="getTimingTooltip(log)">
                    {{ formatTTFT(log) }}
                  </TableCellCode>
                  <TableCellCode :title="getTimingTooltip(log)">
                    {{ formatDurationOrFailed(log, log.response_time_ms) }}
                  </TableCellCode>
                  <TableCellNumeric class="hidden sm:table-cell">
                    <span :title="getTokenTooltip(log)">{{ formatTokenBreakdown(log) }}</span>
                  </TableCellNumeric>
                  <TableCellNumeric class="hidden md:table-cell">
                    {{
                      formatCost(log.cost_usd ?? (log.log_metadata?.cost_usd as number | undefined))
                    }}
                  </TableCellNumeric>
                  <TableCellActions>
                    <div class="flex items-center justify-end gap-0.5">
                      <!-- Smart-routing feedback: only rows resolved from a
                           virtual model (auto/fast/best) are rateable. -->
                      <div
                        v-if="getResolvedModel(log)"
                        class="flex items-center gap-0.5"
                        role="group"
                        :aria-label="t('logs.feedbackLabel')"
                      >
                        <span
                          v-if="feedbackByRequestId[log.request_id]"
                          class="flex h-8 w-8 items-center justify-center rounded-md"
                          :class="FEEDBACK_BADGE_CLASS[feedbackByRequestId[log.request_id]!]"
                          :title="t(FEEDBACK_LABEL_KEYS[feedbackByRequestId[log.request_id]!])"
                          :aria-label="t(FEEDBACK_LABEL_KEYS[feedbackByRequestId[log.request_id]!])"
                        >
                          <component
                            :is="FEEDBACK_ICONS[feedbackByRequestId[log.request_id]!]"
                            class="size-4"
                          />
                        </span>
                        <template v-else>
                          <Button
                            v-for="signal in ['ok', 'weak', 'strong'] as const"
                            :key="signal"
                            size="icon"
                            variant="ghost"
                            class="h-8 w-8 text-muted-foreground/60 hover:text-foreground"
                            :title="t(FEEDBACK_LABEL_KEYS[signal])"
                            :aria-label="t(FEEDBACK_LABEL_KEYS[signal])"
                            :disabled="feedbackSubmittingIds.has(log.request_id)"
                            @click.stop="submitFeedback(log, signal)"
                            @keydown.stop
                          >
                            <component :is="FEEDBACK_ICONS[signal]" class="size-4" />
                          </Button>
                        </template>
                      </div>
                      <Button
                        size="icon"
                        variant="ghost"
                        class="h-11 w-11 min-h-11 min-w-11"
                        :aria-label="`${t('logs.viewDetails')} - ${log.model || log.request_id}`"
                        :disabled="isLoadingDetail"
                        @click.stop="handleViewDetails(log)"
                      >
                        <Eye class="size-5 text-muted-foreground" />
                      </Button>
                    </div>
                  </TableCellActions>
                </TableRow>
              </TableBody>
            </Table>

            <!-- Audit Logs Table -->
            <Table
              v-else-if="activeTab === 'audit'"
              class="table-modern"
              container-class="border-0 bg-transparent rounded-none"
            >
              <TableHeader>
                <TableRow class="hover:bg-transparent hover:border-l-transparent">
                  <TableHead class="table-head-cell">{{ t("logs.timestamp") }}</TableHead>
                  <TableHead class="text-xs">{{ t("logs.action") }}</TableHead>
                  <TableHead class="text-xs hidden sm:table-cell">{{ t("logs.actor") }}</TableHead>
                  <TableHead class="text-xs hidden xl:table-cell">{{
                    t("logs.ipAddress")
                  }}</TableHead>
                  <TableHead class="table-cell-mono truncate max-w-60 hidden md:table-cell">{{
                    t("logs.resource")
                  }}</TableHead>
                  <TableHead>{{ t("logs.result") }}</TableHead>
                  <TableHead class="table-cell-mono hidden lg:table-cell">{{
                    t("logs.latency")
                  }}</TableHead>
                  <TableHead class="w-16"></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody class="row-stagger">
                <TableRow
                  v-for="log in logs"
                  :key="log.request_id"
                  :data-log-row="log.request_id"
                  :aria-label="`${t('logs.viewDetails')} — ${log.model || log.endpoint || log.request_id} (${log.status_code ?? '—'})`"
                  @click="handleViewDetails(log)"
                  class="table-row-hover relative"
                  tabindex="0"
                  role="button"
                  @keydown.enter="handleViewDetails(log)"
                  @keydown.space.prevent="handleViewDetails(log)"
                >
                  <TableCellTimestamp>{{ formatDate(log.timestamp) }}</TableCellTimestamp>
                  <TableCell>
                    <div class="flex items-center gap-2">
                      <StatusBadge variant="http" :http-method="log.method">{{
                        log.method
                      }}</StatusBadge>
                      <span class="text-xs text-muted-foreground">{{ auditListAction(log) }}</span>
                      <Badge
                        v-if="log.event_type"
                        variant="secondary"
                        class="text-[11px] font-medium"
                      >
                        {{ formatEventType(log.event_type) }}
                      </Badge>
                    </div>
                  </TableCell>
                  <TableCell class="text-xs hidden sm:table-cell">
                    <!-- eslint-disable-next-line vue/no-v-html -->
                    <span
                      class="font-medium"
                      v-html="sanitizeHighlightText(getActor(log), filters.search || '')"
                    ></span>
                  </TableCell>
                  <TableCell class="text-xs hidden xl:table-cell">
                    <span class="font-mono text-muted-foreground">{{ log.client_ip || "—" }}</span>
                  </TableCell>
                  <TableCellCode class="truncate max-w-60 hidden md:table-cell overflow-hidden">
                    <span :title="log.endpoint" class="break-all">{{ log.endpoint }}</span>
                  </TableCellCode>
                  <TableCell>
                    <StatusBadge variant="status" :status="getStatusType(log.status_code)">
                      {{ log.status_code }}
                    </StatusBadge>
                  </TableCell>
                  <TableCellCode class="hidden lg:table-cell">
                    {{ formatDurationOrFailed(log, log.response_time_ms) }}
                  </TableCellCode>
                  <TableCellActions>
                    <Button
                      size="icon"
                      variant="ghost"
                      class="h-11 w-11 min-h-11 min-w-11"
                      :aria-label="`${t('logs.viewDetails')} - ${log.endpoint || log.request_id}`"
                      :disabled="isLoadingDetail"
                      @click.stop="handleViewDetails(log)"
                    >
                      <Eye class="size-5 text-muted-foreground" />
                    </Button>
                  </TableCellActions>
                </TableRow>
              </TableBody>
            </Table>

            <!-- MCP Logs Table -->
            <Table
              v-else-if="activeTab === 'mcp'"
              class="table-modern"
              container-class="border-0 bg-transparent rounded-none"
            >
              <TableHeader>
                <TableRow class="hover:bg-transparent hover:border-l-transparent">
                  <TableHead class="table-head-cell">{{ t("logs.timestamp") }}</TableHead>
                  <TableHead class="text-xs">{{ t("logs.mcpServer") }}</TableHead>
                  <TableHead class="text-xs hidden sm:table-cell">{{
                    t("logs.mcpOperation")
                  }}</TableHead>
                  <TableHead class="table-cell-mono hidden md:table-cell">{{
                    t("logs.mcpResource")
                  }}</TableHead>
                  <TableHead>{{ t("logs.status") }}</TableHead>
                  <TableHead class="table-cell-mono hidden lg:table-cell">{{
                    t("logs.duration")
                  }}</TableHead>
                  <TableHead class="w-16"></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody class="row-stagger">
                <TableRow
                  v-for="log in logs"
                  :key="log.request_id"
                  :data-log-row="log.request_id"
                  :aria-label="`${t('logs.viewDetails')} — ${log.model || log.endpoint || log.request_id} (${log.status_code ?? '—'})`"
                  @click="handleViewDetails(log)"
                  class="table-row-hover relative"
                  tabindex="0"
                  role="button"
                  @keydown.enter="handleViewDetails(log)"
                  @keydown.space.prevent="handleViewDetails(log)"
                >
                  <TableCellTimestamp>{{ formatDate(log.timestamp) }}</TableCellTimestamp>
                  <TableCell class="text-xs">
                    <Badge variant="secondary" class="font-mono">
                      <!-- eslint-disable-next-line vue/no-v-html -->
                      <span
                        v-html="sanitizeHighlightText(getMcpServer(log), filters.search || '')"
                      ></span>
                    </Badge>
                  </TableCell>
                  <TableCell class="text-xs hidden sm:table-cell">
                    <span class="text-muted-foreground">{{ getMcpOperation(log) }}</span>
                  </TableCell>
                  <TableCellCode class="hidden md:table-cell overflow-hidden">
                    <span :title="getMcpResourceName(log)" class="break-all">{{
                      getMcpResourceName(log)
                    }}</span>
                  </TableCellCode>
                  <TableCell>
                    <StatusBadge variant="status" :status="getStatusType(log.status_code)">
                      {{ log.status_code }}
                    </StatusBadge>
                  </TableCell>
                  <TableCellCode class="hidden lg:table-cell">
                    {{ formatDurationOrFailed(log, log.response_time_ms) }}
                  </TableCellCode>
                  <TableCellActions>
                    <Button
                      size="icon"
                      variant="ghost"
                      class="h-11 w-11 min-h-11 min-w-11"
                      :aria-label="`${t('logs.viewDetails')} - ${getMcpServer(log) || log.request_id}`"
                      :disabled="isLoadingDetail"
                      @click.stop="handleViewDetails(log)"
                    >
                      <Eye class="size-5 text-muted-foreground" />
                    </Button>
                  </TableCellActions>
                </TableRow>
              </TableBody>
            </Table>

            <!-- Web Search Logs Table -->
            <Table
              v-else-if="activeTab === 'websearch'"
              class="table-modern"
              container-class="border-0 bg-transparent rounded-none"
            >
              <TableHeader>
                <TableRow class="hover:bg-transparent hover:border-l-transparent">
                  <TableHead class="table-head-cell">{{ t("logs.timestamp") }}</TableHead>
                  <TableHead class="text-xs">{{ t("logs.webSearchQuery") }}</TableHead>
                  <TableHead class="text-xs hidden sm:table-cell">{{
                    t("logs.webSearchProvider")
                  }}</TableHead>
                  <TableHead class="table-cell-mono hidden md:table-cell">{{
                    t("logs.webSearchCount")
                  }}</TableHead>
                  <TableHead>{{ t("logs.status") }}</TableHead>
                  <TableHead class="table-cell-mono hidden lg:table-cell">{{
                    t("logs.duration")
                  }}</TableHead>
                  <TableHead class="w-16"></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody class="row-stagger">
                <TableRow
                  v-for="log in logs"
                  :key="log.request_id"
                  :data-log-row="log.request_id"
                  :aria-label="`${t('logs.viewDetails')} — ${log.model || log.endpoint || log.request_id} (${log.status_code ?? '—'})`"
                  @click="handleViewDetails(log)"
                  class="table-row-hover relative"
                  tabindex="0"
                  role="button"
                  @keydown.enter="handleViewDetails(log)"
                  @keydown.space.prevent="handleViewDetails(log)"
                >
                  <TableCellTimestamp>{{ formatDate(log.timestamp) }}</TableCellTimestamp>
                  <TableCell class="text-xs overflow-hidden">
                    <!-- eslint-disable-next-line vue/no-v-html -->
                    <span
                      class="font-mono text-muted-foreground break-all line-clamp-2"
                      v-html="sanitizeHighlightText(getWebSearchQuery(log), filters.search || '')"
                    ></span>
                  </TableCell>
                  <TableCell class="text-xs hidden sm:table-cell">
                    <div class="flex items-center gap-1.5">
                      <div
                        class="w-4 h-4 rounded flex items-center justify-center overflow-hidden bg-white/90 border border-border/50 shrink-0"
                      >
                        <img
                          v-if="getProviderIconUrl(log.provider || '')"
                          :src="getProviderIconUrl(log.provider || '')!"
                          :class="[
                            isMonoProvider(log.provider || '') ? 'icon-mono' : null,
                            'w-3 h-3 object-contain',
                          ]"
                          loading="lazy"
                        />
                        <Globe v-else class="w-2.5 h-2.5 text-muted-foreground" />
                      </div>
                      <Badge variant="outline" class="font-mono">{{
                        log.provider || "searxng"
                      }}</Badge>
                    </div>
                  </TableCell>
                  <TableCellNumeric class="hidden md:table-cell">
                    {{ getWebSearchResultCount(log) }}
                  </TableCellNumeric>
                  <TableCell>
                    <StatusBadge variant="status" :status="getStatusType(log.status_code)">
                      {{ getWebSearchStatus(log) }}
                    </StatusBadge>
                  </TableCell>
                  <TableCellCode class="hidden lg:table-cell">
                    {{ formatDurationOrFailed(log, log.response_time_ms) }}
                  </TableCellCode>
                  <TableCellActions>
                    <Button
                      size="icon"
                      variant="ghost"
                      class="h-11 w-11 min-h-11 min-w-11"
                      :aria-label="`${t('logs.viewDetails')} - ${getWebSearchQuery(log) || log.request_id}`"
                      :disabled="isLoadingDetail"
                      @click.stop="handleViewDetails(log)"
                    >
                      <Eye class="size-5 text-muted-foreground" />
                    </Button>
                  </TableCellActions>
                </TableRow>
              </TableBody>
            </Table>
          </template>

          <!-- Mobile view -->
          <template v-else>
            <div class="flex flex-col gap-2 p-3">
              <LogListItem
                v-for="log in logs"
                :key="log.request_id"
                :log="log"
                :search-query="filters.search || ''"
                :tab="activeTab"
                @view-details="handleViewDetails"
              />
            </div>
          </template>
        </div>
      </div>

      <!-- Fixed Footer with Pagination (Always Visible) -->
      <footer
        class="h-14 flex-none border-t border-border/60 bg-muted/15 px-4 sm:px-6 flex items-center justify-between z-10"
      >
        <PaginationBar
          :current-page="currentPage"
          :total-pages="totalPages"
          :total-items="totalLogs"
          :items-per-page="pageSize"
          :disabled="isFetching"
          class="w-full border-0 bg-transparent p-0 shadow-none"
          @go-to-page="goToPage"
        />
      </footer>
    </div>

    <LogDetailsSheet v-model:open="showDetailDialog" :log="selectedLog" />
    <AuditIntegrityDialog v-model:open="showIntegrityDialog" />
  </AppLayout>
</template>

<style scoped>
@keyframes loading-bar-slide {
  0% {
    transform: translateX(-100%);
  }
  100% {
    transform: translateX(400%);
  }
}

.animate-loading-bar {
  width: 25%;
  animation: loading-bar-slide 1.2s ease-in-out infinite;
}

@media (prefers-reduced-motion: reduce) {
  .animate-loading-bar {
    animation: none;
  }
}
</style>
