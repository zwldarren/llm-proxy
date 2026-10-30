<script setup lang="ts">
import { AlertCircle, Check, ChevronDown, ListTree, Settings2, Trash2 } from "@lucide/vue";

import { computed, nextTick, onMounted, onUnmounted, ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import { useRoute, useRouter } from "vue-router";
import { toast } from "vue-sonner";
import AppLayout from "@/components/layout/AppLayout.vue";
import LoadingState from "@/components/common/LoadingState.vue";
import PageHeader from "@/components/common/PageHeader.vue";
import NumberStepper from "@/components/settings/NumberStepper.vue";
import ServerLogsSection from "@/components/settings/sections/ServerLogsSection.vue";
import WebSearchSection from "@/components/settings/sections/WebSearchSection.vue";
import RequestPolicySection from "@/components/settings/sections/RequestPolicySection.vue";
import SmartRoutingSection from "@/components/settings/sections/SmartRoutingSection.vue";
import ProviderSelectionSection from "@/components/settings/sections/ProviderSelectionSection.vue";
import ResilienceSection from "@/components/settings/sections/ResilienceSection.vue";
import SecuritySection from "@/components/settings/sections/SecuritySection.vue";
import KeepaliveSection from "@/components/settings/sections/KeepaliveSection.vue";
import RateLimitsSection from "@/components/settings/sections/RateLimitsSection.vue";
import CorsSection from "@/components/settings/sections/CorsSection.vue";
import CircuitBreakerSection from "@/components/settings/sections/CircuitBreakerSection.vue";
import McpSecuritySection from "@/components/settings/sections/McpSecuritySection.vue";
import AboutSection from "@/components/settings/sections/AboutSection.vue";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

import { useSettingAutoSave } from "@/composables/useSettingAutoSave";
import { useSettingsStore } from "@/stores/settings";
import { logsApi } from "@/services/api/logs";
import { scrollToAnchor } from "@/utils/scroll";
import {
  configApi,
  mcpSecurityApi,
  requestPolicyApi,
  smartRoutingApi,
  webSearchApi,
  resilienceApi,
  securityApi,
  keepaliveApi,
  providerSelectionApi,
  rateLimitsApi,
  corsApi,
  circuitBreakerApi,
} from "@/services/api/config";
import type {
  CorsConfig,
  KeepaliveConfig,
  LoggingConfig,
  McpSecurityPolicyConfig,
  RateLimitsConfig,
  RequestPolicyConfig,
  ProviderSelectionConfig,
  SecurityConfig,
  SmartRoutingConfig,
  WebSearchConfig,
  ResilienceConfig,
  CircuitBreakerListResponse,
} from "@/types/schemas";
import {
  DEFAULT_LOGGING,
  DEFAULT_WEB_SEARCH,
  DEFAULT_SMART_ROUTING,
  DEFAULT_REQUEST_POLICY,
  DEFAULT_MCP_SECURITY,
  DEFAULT_RESILIENCE,
  DEFAULT_SECURITY,
  DEFAULT_KEEPALIVE,
  DEFAULT_PROVIDER_SELECTION,
  DEFAULT_RATE_LIMITS,
  DEFAULT_CORS,
} from "@/constants/defaults";

const { t } = useI18n();
const settingsStore = useSettingsStore();
const route = useRoute();
const router = useRouter();

type SettingsSectionId = "general" | "advanced";

/* The ?tab= query is shared URL state — the kept-alive Logs view also reads
 * a `tab` key, so an unrelated value (e.g. ?tab=proxy) can land here. Treat
 * anything that is not a real settings tab as General instead of blanking
 * the page. */
const activeSection = computed<SettingsSectionId>(() =>
  route.query.tab === "advanced" ? "advanced" : "general"
);

const switchTab = (tab: SettingsSectionId) => {
  router.push({ path: route.path, query: { tab } });
  // Each tab is a fresh context — restart from the top rather than landing
  // mid-scroll at whatever offset the other tab happened to be at.
  scrollEl.value?.scrollTo({ top: 0, behavior: "instant" });
};

/* Tab-switch reveal — the general/advanced sections use v-show to preserve
 * form state, so a key-based transition would remount and lose it. Instead we
 * re-trigger the same config-page-reveal rise on the content column: no
 * remount, and the context switch still reads as a deliberate transition. */
const contentEl = ref<HTMLElement | null>(null);

watch(activeSection, async () => {
  await nextTick();
  const el = contentEl.value;
  if (!el) return;
  el.classList.remove("config-page-reveal");
  el.getAnimations().forEach((a) => a.cancel());
  el.classList.add("config-page-reveal");
});

const onContentRevealEnd = (event: AnimationEvent) => {
  const el = contentEl.value;
  if (!el) return;
  // Only act on the wrapper's own animation, not bubbled child events.
  if (event.target !== el) return;
  el.classList.remove("config-page-reveal");
};

/* Deep links into a specific settings card — e.g. the sidebar version badge
 * pointing at /config/settings#about. The target is not guaranteed to be
 * visible when the route changes (admin configs load behind a spinner on a
 * cold visit, and the v-show tab sections render as display:none until their
 * tab activates), so the scroll helper retries until the target has a box. */
let cancelHashScroll: (() => void) | null = null;

function scrollToHash() {
  cancelHashScroll?.();
  cancelHashScroll = null;
  const id = route.hash.replace("#", "");
  if (!id) return;
  cancelHashScroll = scrollToAnchor(id, { highlight: true });
}

watch(
  () => route.fullPath,
  () => scrollToHash(),
  { immediate: true }
);

/* In-page section menu — each tab stacks many cards, so a compact grouped
 * menu in the page header provides jumps without becoming another content
 * band. Advanced groups its sections by topic; General stays flat. */
interface SettingsSectionMeta {
  id: string;
  labelKey: string;
}

interface SettingsGroupMeta {
  id: string;
  tab: SettingsSectionId;
  /** Nav-group label; null means the items are listed without a heading. */
  labelKey: string | null;
  items: SettingsSectionMeta[];
}

const SECTION_GROUPS: SettingsGroupMeta[] = [
  {
    id: "general",
    tab: "general",
    labelKey: null,
    items: [
      { id: "server", labelKey: "settings.serverLogs" },
      { id: "webSearch", labelKey: "settings.webSearch" },
      { id: "mcpSecurity", labelKey: "mcpSecurity.title" },
      { id: "about", labelKey: "about.title" },
    ],
  },
  {
    id: "routing",
    tab: "advanced",
    labelKey: "settings.groupRouting",
    items: [
      { id: "requestPolicy", labelKey: "requestPolicy.title" },
      { id: "smartRouting", labelKey: "smartRouting.title" },
      { id: "providerSelection", labelKey: "providerSelection.title" },
    ],
  },
  {
    id: "resilience",
    tab: "advanced",
    labelKey: "settings.groupResilience",
    items: [
      { id: "retryFallback", labelKey: "resilience.title" },
      { id: "circuitBreaker", labelKey: "circuitBreaker.title" },
      { id: "keepalive", labelKey: "keepalive.title" },
    ],
  },
  {
    id: "access",
    tab: "advanced",
    labelKey: "settings.groupAccess",
    items: [
      { id: "security", labelKey: "security.title" },
      { id: "rateLimits", labelKey: "rateLimits.title" },
      { id: "cors", labelKey: "cors.title" },
    ],
  },
];

const navGroups = computed(() =>
  SECTION_GROUPS.filter((group) => group.tab === activeSection.value)
);

/** Flat reading order for the scrollspy and the end-of-scroll case. */
const navItems = computed(() => navGroups.value.flatMap((group) => group.items));

const activeAnchor = ref("");

/* The trigger carries the section the reading position is in, so the header
 * answers "where am I" without the operator opening the menu first. With no
 * scrollspy hit yet it falls back to the first section of the tab. */
const activeSectionLabel = computed(() => {
  const item = navItems.value.find((entry) => entry.id === activeAnchor.value);
  return item ? t(item.labelKey) : "";
});

const scrollEl = ref<HTMLElement | null>(null);
let anchorFrame = 0;

/* Reading position = the section nearest the line 15% down the scrollport.
 * An intersection band cannot mark a short final section: it can never reach
 * the top of the scrollport, so scrolling to the end of a tab left the last
 * entry unhighlighted. The end-of-scroll case is therefore explicit. */
function updateActiveAnchor() {
  const root = scrollEl.value;
  const items = navItems.value;
  if (!root || items.length === 0) {
    activeAnchor.value = "";
    return;
  }
  if (root.scrollTop + root.clientHeight >= root.scrollHeight - 2) {
    activeAnchor.value = items[items.length - 1].id;
    return;
  }
  const rootTop = root.getBoundingClientRect().top;
  const line = root.clientHeight * 0.15;
  let current = items[0].id;
  let currentTop = -Infinity;
  for (const item of items) {
    const el = document.getElementById(item.id);
    // getClientRects() is empty for display:none subtrees — the v-show'd
    // sections of the inactive tab.
    if (!el || el.getClientRects().length === 0) continue;
    const top = el.getBoundingClientRect().top - rootTop;
    if (top <= line && top > currentTop) {
      current = item.id;
      currentTop = top;
    }
  }
  activeAnchor.value = current;
}

/* Scroll events arrive in bursts; one geometry read per frame is enough. */
function scheduleActiveAnchor() {
  if (anchorFrame) return;
  anchorFrame = requestAnimationFrame(() => {
    anchorFrame = 0;
    updateActiveAnchor();
  });
}

watch(
  activeSection,
  async () => {
    await nextTick();
    updateActiveAnchor();
  },
  { immediate: true }
);

function jumpToSection(id: string) {
  activeAnchor.value = id;
  if (route.hash === `#${id}`) {
    // Hash unchanged → no route event fires; scroll directly.
    scrollToAnchor(id, { highlight: true });
    return;
  }
  router.replace({ query: route.query, hash: `#${id}` });
}

// Show loading spinner only when we have no cached data at all.
// A failed first load gets an explicit retry state instead of a permanent spinner.
const settingsLoadFailed = ref(false);
const showLoadingSpinner = computed(() => !settingsLoadFailed.value && !settingsStore.hasCache());

async function loadSettings() {
  settingsLoadFailed.value = false;
  try {
    await settingsStore.fetchAll();
    initializeAutoSaveFromStore();
  } catch (e) {
    console.error("Failed to fetch settings", e);
    settingsLoadFailed.value = true;
  }
}

// First visit renders the settings cards only after fetchAll() resolves; honour a
// hash that was set before the spinner cleared once the content column mounts.
watch(showLoadingSpinner, (loading) => {
  if (!loading)
    nextTick(() => {
      scrollToHash();
      updateActiveAnchor();
    });
});

// ── Auto-save instances ─────────────────────────────────────────────
// Initialize with defaults - will be updated after data loads

const webSearchAutoSave = useSettingAutoSave<WebSearchConfig>(DEFAULT_WEB_SEARCH, (val) =>
  webSearchApi.updateConfig(val).then((r) => {
    settingsStore.updateWebSearchCache(r);
    return r;
  })
);

const loggingAutoSave = useSettingAutoSave<LoggingConfig>(DEFAULT_LOGGING, (val) =>
  configApi.updateLoggingConfig(val).then((r) => {
    settingsStore.updateLoggingCache(r);
    return r;
  })
);

const smartRoutingAutoSave = useSettingAutoSave<SmartRoutingConfig>(DEFAULT_SMART_ROUTING, (val) =>
  smartRoutingApi.updateConfig(val).then((r) => {
    settingsStore.updateSmartRoutingCache(r);
    return r;
  })
);

const providerSelectionAutoSave = useSettingAutoSave<ProviderSelectionConfig>(
  DEFAULT_PROVIDER_SELECTION,
  (val) =>
    providerSelectionApi.updateConfig(val).then((r) => {
      settingsStore.updateProviderSelectionCache(r);
      return r;
    })
);

const requestPolicyAutoSave = useSettingAutoSave<RequestPolicyConfig>(
  DEFAULT_REQUEST_POLICY,
  (val) =>
    requestPolicyApi.updateConfig(val).then((r) => {
      settingsStore.updateRequestPolicyCache(r);
      return r;
    })
);

const mcpSecurityAutoSave = useSettingAutoSave<McpSecurityPolicyConfig>(
  DEFAULT_MCP_SECURITY,
  (val) =>
    mcpSecurityApi.updateConfig(val).then((r) => {
      settingsStore.updateMcpSecurityCache(r);
      return r;
    })
);

const resilienceAutoSave = useSettingAutoSave<ResilienceConfig>(DEFAULT_RESILIENCE, (val) =>
  resilienceApi.updateConfig(val).then((r) => {
    settingsStore.updateResilienceCache(r);
    return r;
  })
);

const securityAutoSave = useSettingAutoSave<SecurityConfig>(DEFAULT_SECURITY, (val) =>
  securityApi.updateConfig(val).then((r) => {
    settingsStore.updateSecurityCache(r);
    return r;
  })
);

const keepaliveAutoSave = useSettingAutoSave<KeepaliveConfig>(DEFAULT_KEEPALIVE, (val) =>
  keepaliveApi.updateConfig(val).then((r) => {
    settingsStore.updateKeepaliveCache(r);
    return r;
  })
);

const rateLimitsAutoSave = useSettingAutoSave<RateLimitsConfig>(DEFAULT_RATE_LIMITS, (val) =>
  rateLimitsApi.updateConfig({ limits: val.limits }).then((r) => {
    settingsStore.updateRateLimitsCache(r);
    return r;
  })
);

const corsAutoSave = useSettingAutoSave<CorsConfig>(DEFAULT_CORS, (val) =>
  corsApi.updateConfig({ origins: val.origins }).then((r) => {
    settingsStore.updateCorsCache(r);
    return r;
  })
);

// ── Circuit Breaker States & Monitoring ─────────────────────────────
const circuitStates = ref<CircuitBreakerListResponse | null>(null);
const loadingCircuits = ref(false);
const resettingCircuits = ref(false);

async function loadCircuitStates() {
  if (loadingCircuits.value) return; // Guard against concurrent polling
  loadingCircuits.value = true;
  try {
    const statesRes = await circuitBreakerApi.listStates();
    circuitStates.value = statesRes;
    settingsStore.updateCircuitBreakerCache(statesRes);
  } catch (e) {
    console.error("Failed to load circuit breaker states:", e);
  } finally {
    loadingCircuits.value = false;
  }
}

async function resetAllCircuits() {
  resettingCircuits.value = true;
  try {
    await circuitBreakerApi.resetAll();
    toast.success(t("circuitBreaker.resetSuccess"));
    await loadCircuitStates();
  } catch (e) {
    console.error("Failed to reset circuit breakers:", e);
    toast.error(t("circuitBreaker.resetError"));
  } finally {
    resettingCircuits.value = false;
  }
}

async function resetOneCircuit(key: string) {
  try {
    await circuitBreakerApi.resetOne(key);
    toast.success(t("circuitBreaker.resetOneSuccess", { key }));
    await loadCircuitStates();
  } catch (e) {
    console.error("Failed to reset circuit breaker:", e);
    toast.error(t("circuitBreaker.resetError"));
  }
}

// ── MCP security reset ──────────────────────────────────────────────

const showMcpResetDialog = ref(false);

async function confirmMcpResetDefaults() {
  mcpSecurityAutoSave.initialize({
    require_key_mcp_permissions: DEFAULT_MCP_SECURITY.require_key_mcp_permissions,
    allowed_commands: [],
    blocked_commands: [...DEFAULT_MCP_SECURITY.blocked_commands],
    allowed_env_keys: [],
    blocked_env_keys: [...DEFAULT_MCP_SECURITY.blocked_env_keys],
    blocked_url_hosts: [],
    blocked_url_ips: [...DEFAULT_MCP_SECURITY.blocked_url_ips],
  });
  const saved = await mcpSecurityAutoSave.save();
  showMcpResetDialog.value = false;
  if (saved) toast.success(t("mcpSecurity.resetSuccess"));
}

// ── Cleanup dialog ─────────────────────────────────────────────────

const showCleanupDialog = ref(false);
const cleanupDays = ref<number | null>(null);
const isCleaningLogs = ref(false);
const deletedLogsCount = ref<number | null>(null);

const openCleanupDialog = () => {
  cleanupDays.value = null;
  deletedLogsCount.value = null;
  showCleanupDialog.value = true;
};

const confirmCleanupLogs = async () => {
  if (cleanupDays.value === null || cleanupDays.value < 1) return;
  isCleaningLogs.value = true;
  try {
    const response = await logsApi.deleteOldLogs(cleanupDays.value);
    deletedLogsCount.value = response.deleted;
    toast.success(t("settings.logsCleaned"), {
      description: t("settings.logsCleanedCount", { count: response.deleted }),
    });
    showCleanupDialog.value = false;
  } catch (e) {
    console.error("Failed to cleanup logs", e);
    toast.error(t("common.error"), {
      description: t("settings.logsCleanupFailed"),
    });
  } finally {
    isCleaningLogs.value = false;
  }
};

// ── Fetch on mount (cache-first: show cached data immediately, refresh in background) ──

// Initialize the auto-save instances from the store's current config. Reads
// happen before any background refresh so the displayed values reflect the
// cache (or the freshly fetched data on first visit).
function initializeAutoSaveFromStore() {
  const logging = settingsStore.loggingConfig;
  const webSearch = settingsStore.webSearchConfig;
  const smartRouting = settingsStore.smartRoutingConfig;
  const providerSelection = settingsStore.providerSelectionConfig;
  const requestPolicy = settingsStore.requestPolicyConfig;
  const mcpSecurity = settingsStore.mcpSecurityConfig;
  const security = settingsStore.securityConfig;
  const keepalive = settingsStore.keepaliveConfig;
  const rateLimits = settingsStore.rateLimitsConfig;
  const cors = settingsStore.corsConfig;
  const resilience = settingsStore.resilienceConfig;

  // Tracing is loaded per-user via loadPersonalTracing() (not from the shared
  // settings store), so it is intentionally not initialized here.
  if (logging) {
    loggingAutoSave.initialize(logging);
  }
  if (webSearch) {
    webSearchAutoSave.initialize(webSearch);
  }
  if (smartRouting) {
    smartRoutingAutoSave.initialize(smartRouting);
  }
  if (providerSelection) {
    providerSelectionAutoSave.initialize(providerSelection);
  }
  if (requestPolicy) {
    requestPolicyAutoSave.initialize(requestPolicy);
  }
  if (mcpSecurity) {
    mcpSecurityAutoSave.initialize(mcpSecurity);
  }
  if (security) {
    securityAutoSave.initialize(security);
  }
  if (keepalive) {
    keepaliveAutoSave.initialize(keepalive);
  }
  if (rateLimits) {
    rateLimitsAutoSave.initialize(rateLimits);
  }
  if (cors) {
    corsAutoSave.initialize(cors);
  }
  if (resilience) {
    resilienceAutoSave.initialize(resilience);
  }
}

let pollInterval: ReturnType<typeof setInterval> | null = null;

onMounted(async () => {
  // Start polling only when the advanced section is active
  watch(
    () => activeSection.value === "advanced",
    (isActive) => {
      if (isActive) {
        loadCircuitStates();
        pollInterval = setInterval(loadCircuitStates, 10000);
      } else if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
      }
    },
    { immediate: true }
  );

  // If we have cache, initialize auto-save with cached data and skip loading spinner
  if (settingsStore.hasCache()) {
    initializeAutoSaveFromStore();
    // Background refresh - updates cache, auto-save won't trigger because we update cache
    settingsStore.fetchAll();
    return;
  }

  // First visit: show loading spinner, fetch from API
  await loadSettings();
});

onUnmounted(() => {
  if (pollInterval) {
    clearInterval(pollInterval);
  }
  if (anchorFrame) cancelAnimationFrame(anchorFrame);
  cancelHashScroll?.();
});
</script>

<template>
  <AppLayout layoutMode="full">
    <template #header>
      <header class="config-header-bar px-4 sm:px-6 py-4">
        <!-- The page header stays on the shared console gutter; only the form
             column below uses a narrower reading measure. -->
        <div class="w-full">
          <PageHeader
            :title="t('settings.title')"
            :description="t('settings.description')"
            :icon="Settings2"
          >
            <template #actions>
              <div
                class="flex items-center bg-muted/60 p-[3px] rounded-lg border border-border/55"
                :aria-label="t('settings.title')"
                role="group"
              >
                <button
                  type="button"
                  :aria-pressed="activeSection === 'general'"
                  class="max-md:min-h-11 max-md:px-4 px-3 py-1.5 text-xs rounded-md border transition-all duration-200 cursor-pointer focus-visible-ring"
                  :class="
                    activeSection === 'general'
                      ? 'bg-accent text-foreground font-semibold border-border/70 shadow-xs'
                      : 'border-transparent text-muted-foreground font-medium hover:bg-accent/40 hover:text-foreground'
                  "
                  @click="switchTab('general')"
                >
                  {{ t("nav.general") }}
                </button>
                <button
                  type="button"
                  :aria-pressed="activeSection === 'advanced'"
                  class="max-md:min-h-11 max-md:px-4 px-3 py-1.5 text-xs rounded-md border transition-all duration-200 cursor-pointer focus-visible-ring"
                  :class="
                    activeSection === 'advanced'
                      ? 'bg-accent text-foreground font-semibold border-border/70 shadow-xs'
                      : 'border-transparent text-muted-foreground font-medium hover:bg-accent/40 hover:text-foreground'
                  "
                  @click="switchTab('advanced')"
                >
                  {{ t("nav.advanced") }}
                </button>
              </div>

              <DropdownMenu v-if="navItems.length > 1" :modal="false">
                <DropdownMenuTrigger as-child>
                  <!-- The trigger names the section being read rather than the
                       menu itself: "Sections" alone told the operator nothing
                       about their position on a ten-section page. The label
                       keeps the accessible name complete, so the visible text
                       stays a substring of it (WCAG 2.5.3). The width follows
                       the label rather than being pinned to the widest section
                       name: a fixed frame would leave a dead 60–140px inside
                       the hit area and under the focus ring for most sections. -->
                  <Button
                    variant="ghost"
                    size="sm"
                    class="max-w-full gap-1.5 px-2.5 text-xs font-medium text-muted-foreground hover:text-foreground"
                    :title="t('settings.sections')"
                    :aria-label="`${t('settings.sections')}: ${activeSectionLabel}`"
                  >
                    <ListTree class="size-4 shrink-0" />
                    <span class="min-w-0 truncate text-left text-foreground">
                      {{ activeSectionLabel || t("settings.sections") }}
                    </span>
                    <ChevronDown class="size-3.5 shrink-0" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" class="w-64 sm:w-72">
                  <DropdownMenuGroup v-for="group in navGroups" :key="group.id">
                    <DropdownMenuLabel
                      v-if="group.labelKey"
                      class="pt-2 pb-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground"
                    >
                      {{ t(group.labelKey) }}
                    </DropdownMenuLabel>
                    <DropdownMenuItem
                      v-for="item in group.items"
                      :key="item.id"
                      :aria-current="activeAnchor === item.id ? 'location' : undefined"
                      :class="
                        activeAnchor === item.id
                          ? 'bg-accent/60 text-accent-foreground font-medium'
                          : ''
                      "
                      @select="jumpToSection(item.id)"
                    >
                      <span class="flex-1">{{ t(item.labelKey) }}</span>
                      <Check v-if="activeAnchor === item.id" class="size-4" />
                    </DropdownMenuItem>
                  </DropdownMenuGroup>
                </DropdownMenuContent>
              </DropdownMenu>
            </template>
          </PageHeader>
        </div>
      </header>
    </template>

    <div
      ref="scrollEl"
      class="flex-1 overflow-y-auto px-4 sm:px-6 py-6 relative"
      @scroll.passive="scheduleActiveAnchor"
    >
      <div
        v-if="settingsLoadFailed"
        class="mx-auto flex min-h-[40vh] max-w-md flex-col items-center justify-center gap-4 text-center"
        role="alert"
      >
        <AlertCircle class="size-8 text-destructive" aria-hidden="true" />
        <p class="text-sm text-muted-foreground">{{ t("settings.fetchFailed") }}</p>
        <Button variant="outline" @click="loadSettings">{{ t("common.retry") }}</Button>
      </div>

      <LoadingState v-else-if="showLoadingSpinner" :show-text="false" />

      <div v-else ref="contentEl" class="w-full" @animationend="onContentRevealEnd">
        <!-- Content Area — each anchored section clears the viewport edge. -->
        <div class="mx-auto w-full max-w-4xl space-y-8 pb-20">
          <!-- General tab -->
          <div v-show="activeSection === 'general'" id="server" class="space-y-6 mt-0 scroll-mt-6">
            <ServerLogsSection
              :auto-save="loggingAutoSave"
              :is-admin="true"
              @cleanup="openCleanupDialog"
            />
          </div>

          <div
            v-show="activeSection === 'general'"
            id="webSearch"
            class="space-y-6 mt-0 scroll-mt-6"
          >
            <WebSearchSection :auto-save="webSearchAutoSave" />
          </div>

          <div
            v-show="activeSection === 'general'"
            id="mcpSecurity"
            class="space-y-6 mt-0 scroll-mt-6"
          >
            <McpSecuritySection
              :auto-save="mcpSecurityAutoSave"
              @reset-defaults="showMcpResetDialog = true"
            />
          </div>

          <div v-show="activeSection === 'general'" id="about" class="space-y-6 mt-0 scroll-mt-6">
            <AboutSection />
          </div>

          <!-- Advanced tab -->
          <div
            v-show="activeSection === 'advanced'"
            id="requestPolicy"
            class="space-y-6 mt-0 scroll-mt-6"
          >
            <RequestPolicySection :auto-save="requestPolicyAutoSave" />
          </div>

          <div
            v-show="activeSection === 'advanced'"
            id="smartRouting"
            class="space-y-6 mt-0 scroll-mt-6"
          >
            <SmartRoutingSection
              :auto-save="smartRoutingAutoSave"
              :logging-auto-save="loggingAutoSave"
            />
          </div>

          <div
            v-show="activeSection === 'advanced'"
            id="providerSelection"
            class="space-y-6 mt-0 scroll-mt-6"
          >
            <ProviderSelectionSection :auto-save="providerSelectionAutoSave" />
          </div>

          <div
            v-show="activeSection === 'advanced'"
            id="retryFallback"
            class="space-y-6 mt-0 scroll-mt-6"
          >
            <ResilienceSection :auto-save="resilienceAutoSave" />
          </div>

          <div
            v-show="activeSection === 'advanced'"
            id="circuitBreaker"
            class="space-y-6 mt-0 scroll-mt-6"
          >
            <CircuitBreakerSection
              :auto-save="resilienceAutoSave"
              :circuit-states="circuitStates"
              :loading="loadingCircuits"
              :resetting="resettingCircuits"
              @refresh="loadCircuitStates"
              @reset-all="resetAllCircuits"
              @reset-one="resetOneCircuit"
            />
          </div>

          <div
            v-show="activeSection === 'advanced'"
            id="keepalive"
            class="space-y-6 mt-0 scroll-mt-6"
          >
            <KeepaliveSection :auto-save="keepaliveAutoSave" />
          </div>

          <div
            v-show="activeSection === 'advanced'"
            id="security"
            class="space-y-6 mt-0 scroll-mt-6"
          >
            <SecuritySection :auto-save="securityAutoSave" />
          </div>

          <div
            v-show="activeSection === 'advanced'"
            id="rateLimits"
            class="space-y-6 mt-0 scroll-mt-6"
          >
            <RateLimitsSection :auto-save="rateLimitsAutoSave" />
          </div>

          <div v-show="activeSection === 'advanced'" id="cors" class="space-y-6 mt-0 scroll-mt-6">
            <CorsSection :auto-save="corsAutoSave" />
          </div>
        </div>
      </div>
    </div>

    <!-- MCP Security Reset Dialog -->
    <AlertDialog v-model:open="showMcpResetDialog">
      <AlertDialogContent class="sm:max-w-md">
        <AlertDialogHeader>
          <AlertDialogTitle>{{ t("mcpSecurity.resetDefaultsTitle") }}</AlertDialogTitle>
          <AlertDialogDescription>
            {{ t("mcpSecurity.resetDefaultsDescription") }}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>{{ t("common.cancel") }}</AlertDialogCancel>
          <Button variant="destructive" @click="confirmMcpResetDefaults">
            {{ t("mcpSecurity.resetDefaults") }}
          </Button>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>

    <!-- Cleanup Dialog -->
    <AlertDialog v-model:open="showCleanupDialog">
      <AlertDialogContent class="sm:max-w-md">
        <AlertDialogHeader>
          <AlertDialogTitle>{{ t("settings.cleanupLogsTitle") }}</AlertDialogTitle>
          <AlertDialogDescription>
            {{ t("settings.cleanupLogsDescription") }}
          </AlertDialogDescription>
        </AlertDialogHeader>

        <div class="flex flex-col gap-4 py-4">
          <div class="flex flex-col gap-2">
            <Label for="cleanup-days" class="text-sm font-medium">
              {{ t("settings.deleteLogsOlderThan") }}
            </Label>
            <NumberStepper
              id="cleanup-days"
              v-model="cleanupDays"
              :min="1"
              :suffix="t('settings.days')"
              placeholder="30"
              class="w-40"
            />
            <p v-if="deletedLogsCount !== null" class="text-sm text-status-success">
              {{ t("settings.logsCleanedCount", { count: deletedLogsCount }) }}
            </p>
          </div>
        </div>

        <AlertDialogFooter>
          <AlertDialogCancel>{{ t("common.cancel") }}</AlertDialogCancel>
          <Button
            variant="destructive"
            :loading="isCleaningLogs"
            :disabled="isCleaningLogs || !cleanupDays || cleanupDays < 1"
            @click="confirmCleanupLogs"
          >
            <Trash2 data-icon="inline-start" />
            {{ t("settings.confirmCleanup") }}
          </Button>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  </AppLayout>
</template>
