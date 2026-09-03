<script setup lang="ts">
import { Trash2 } from "@lucide/vue";

import { computed, nextTick, onMounted, onUnmounted, ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import { useRoute, useRouter } from "vue-router";
import { toast } from "vue-sonner";
import AppLayout from "@/components/layout/AppLayout.vue";
import LoadingState from "@/components/common/LoadingState.vue";
import NumberStepper from "@/components/settings/NumberStepper.vue";
import PreferenceSection from "@/components/settings/sections/PreferenceSection.vue";
import ServerLogsSection from "@/components/settings/sections/ServerLogsSection.vue";
import WebSearchSection from "@/components/settings/sections/WebSearchSection.vue";
import TracingSection from "@/components/settings/sections/TracingSection.vue";
import TracingProviderSheet from "@/components/settings/sections/TracingProviderSheet.vue";
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
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

import { useSettingAutoSave } from "@/composables/useSettingAutoSave";
import { useAuthStore } from "@/stores/auth";
import { useTracingProviderEditor } from "@/composables/useTracingProviders";
import { useSettingsStore } from "@/stores/settings";
import { logsApi } from "@/services/api/logs";
import {
  configApi,
  meTracingApi,
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
  TracingConfig,
  WebSearchConfig,
  ResilienceConfig,
  CircuitBreakerListResponse,
} from "@/types/schemas";
import {
  DEFAULT_TRACING,
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
const authStore = useAuthStore();
const route = useRoute();
const router = useRouter();

type SettingsSectionId = "general" | "advanced";

const activeSection = computed<SettingsSectionId>(
  () => (route.query.tab as SettingsSectionId) || "general"
);

const switchTab = (tab: SettingsSectionId) => {
  router.push({ path: route.path, query: { tab } });
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

watch(
  () => route.fullPath,
  () => {
    if (route.hash) {
      // Use setTimeout to ensure any Vue transitions or DOM updates (v-if) are fully complete
      setTimeout(() => {
        const id = route.hash.replace("#", "");
        const el = document.getElementById(id);
        if (el) {
          el.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      }, 100);
    }
  },
  { immediate: true }
);

// Show loading spinner only when we have no cached data at all
// Non-admin users don't fetch configs, so skip the spinner
const showLoadingSpinner = computed(() => authStore.isAdmin && !settingsStore.hasCache());

// ── Auto-save instances ─────────────────────────────────────────────
// Initialize with defaults - will be updated after data loads

const tracingAutoSave = useSettingAutoSave<TracingConfig>(DEFAULT_TRACING, (val) => {
  const saveData = tracing.buildSavePayload(val);
  return meTracingApi.updateTracing(saveData).then((r) => r.config);
});

const tracing = useTracingProviderEditor(tracingAutoSave.state);

// ── Tracing provider editor (sheet) ───────────────────────────────
// Editing a provider happens in a focused side sheet rather than inline, so
// the settings page stays compact no matter how many providers are added.
const editingProviderId = ref<string | null>(null);
const editingProvider = computed(
  () => tracingAutoSave.state.value.providers.find((p) => p.id === editingProviderId.value) ?? null
);

function openProviderEditor(id: string) {
  editingProviderId.value = id;
}

function closeProviderEditor() {
  editingProviderId.value = null;
}

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

function confirmMcpResetDefaults() {
  mcpSecurityAutoSave.initialize({
    require_key_mcp_permissions: DEFAULT_MCP_SECURITY.require_key_mcp_permissions,
    allowed_commands: [],
    blocked_commands: [...DEFAULT_MCP_SECURITY.blocked_commands],
    allowed_env_keys: [],
    blocked_env_keys: [...DEFAULT_MCP_SECURITY.blocked_env_keys],
    blocked_url_hosts: [],
    blocked_url_ips: [...DEFAULT_MCP_SECURITY.blocked_url_ips],
  });
  mcpSecurityAutoSave.save();
  showMcpResetDialog.value = false;
  toast.success(t("mcpSecurity.resetSuccess"));
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

/** Fetch the current user's personal tracing config + supported providers and
 * initialize the tracing auto-save from it. Used by non-admin (viewer) users. */
async function loadPersonalTracing() {
  try {
    await tracing.loadProviderTypes(meTracingApi);
  } catch (err) {
    console.error("Failed to load tracing provider types:", err);
  }
  try {
    const res = await meTracingApi.getTracing();
    tracingAutoSave.initialize({
      enabled: res.config.enabled ?? false,
      providers: tracing.prepareInitialProviders(res.config.providers ?? []),
    });
  } catch (err) {
    console.error("Failed to load personal tracing config:", err);
  }
}

let pollInterval: ReturnType<typeof setInterval> | null = null;

onMounted(async () => {
  // Non-admin users only fetch logging config (no admin-only settings).
  if (!authStore.isAdmin) {
    initializeAutoSaveFromStore();
    await loadPersonalTracing();
    return;
  }

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
    await loadPersonalTracing();
    return;
  }

  // First visit: show loading spinner, fetch from API
  try {
    await settingsStore.fetchAll();
    initializeAutoSaveFromStore();
  } catch (e) {
    console.error("Failed to fetch settings", e);
    toast.error(t("common.error"), {
      description: t("settings.fetchFailed"),
    });
  }
  // Tracing is per-user for everyone (admin included).
  await loadPersonalTracing();
});

onUnmounted(() => {
  if (pollInterval) {
    clearInterval(pollInterval);
  }
});
</script>

<template>
  <AppLayout layoutMode="full">
    <template #header>
      <div
        class="flex-none px-4 sm:px-6 py-3 border-b border-border/40 bg-background/95 backdrop-blur-sm z-10 flex items-center gap-3.5"
      >
        <h1 class="text-base font-semibold text-foreground tracking-tight">
          {{ t("nav.settings") }}
        </h1>
        <div class="h-4 w-px bg-border/60"></div>
        <div class="flex items-center bg-muted/40 p-0.5 rounded-lg border border-border/40">
          <button
            type="button"
            class="px-2.5 py-1 text-xs font-medium rounded-md transition-all duration-200 cursor-pointer"
            :class="
              activeSection === 'general'
                ? 'bg-background text-foreground shadow-xs'
                : 'text-muted-foreground hover:text-foreground'
            "
            @click="switchTab('general')"
          >
            {{ t("nav.general") }}
          </button>
          <button
            type="button"
            class="px-2.5 py-1 text-xs font-medium rounded-md transition-all duration-200 cursor-pointer"
            :class="
              activeSection === 'advanced'
                ? 'bg-background text-foreground shadow-xs'
                : 'text-muted-foreground hover:text-foreground'
            "
            @click="switchTab('advanced')"
          >
            {{ t("nav.advanced") }}
          </button>
        </div>
      </div>
    </template>

    <div class="flex-1 overflow-y-auto px-4 sm:px-6 py-6 relative">
      <LoadingState v-if="showLoadingSpinner" :show-text="false" />

      <div
        v-else
        ref="contentEl"
        class="max-w-5xl w-full flex flex-col gap-6"
        @animationend="onContentRevealEnd"
      >
        <!-- Content Area -->
        <div class="flex-1 min-w-0 space-y-8">
          <!-- General tab -->
          <div
            v-show="activeSection === 'general'"
            id="interface"
            class="space-y-6 mt-0 scroll-mt-20"
          >
            <PreferenceSection />
          </div>

          <div v-show="activeSection === 'general'" id="server" class="space-y-6 mt-0 scroll-mt-20">
            <ServerLogsSection
              :auto-save="loggingAutoSave"
              :is-admin="authStore.isAdmin"
              @cleanup="openCleanupDialog"
            />
          </div>

          <div
            v-if="authStore.isAdmin"
            v-show="activeSection === 'general'"
            id="webSearch"
            class="space-y-6 mt-0 scroll-mt-20"
          >
            <WebSearchSection :auto-save="webSearchAutoSave" />
          </div>

          <div
            v-show="activeSection === 'general'"
            id="tracing"
            class="space-y-6 mt-0 scroll-mt-20"
          >
            <TracingSection
              :auto-save="tracingAutoSave"
              :editor="tracing"
              @edit-provider="openProviderEditor"
            />
          </div>

          <div
            v-if="authStore.isAdmin"
            v-show="activeSection === 'general'"
            id="about"
            class="space-y-6 mt-0 scroll-mt-20"
          >
            <AboutSection />
          </div>

          <!-- Advanced tab -->
          <div
            v-if="authStore.isAdmin"
            v-show="activeSection === 'advanced'"
            id="requestPolicy"
            class="space-y-6 mt-0 scroll-mt-20"
          >
            <RequestPolicySection :auto-save="requestPolicyAutoSave" />
          </div>

          <div
            v-if="authStore.isAdmin"
            v-show="activeSection === 'advanced'"
            id="smartRouting"
            class="space-y-6 mt-0 scroll-mt-20"
          >
            <SmartRoutingSection
              :auto-save="smartRoutingAutoSave"
              :logging-auto-save="loggingAutoSave"
            />
          </div>

          <div
            v-if="authStore.isAdmin"
            v-show="activeSection === 'advanced'"
            id="providerSelection"
            class="space-y-6 mt-0 scroll-mt-20"
          >
            <ProviderSelectionSection :auto-save="providerSelectionAutoSave" />
          </div>

          <div
            v-if="authStore.isAdmin"
            v-show="activeSection === 'advanced'"
            id="retryFallback"
            class="space-y-6 mt-0 scroll-mt-20"
          >
            <ResilienceSection :auto-save="resilienceAutoSave" />
          </div>

          <div
            v-if="authStore.isAdmin"
            v-show="activeSection === 'advanced'"
            id="circuitBreaker"
            class="space-y-6 mt-0 scroll-mt-20"
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
            v-if="authStore.isAdmin"
            v-show="activeSection === 'advanced'"
            id="security"
            class="space-y-6 mt-0 scroll-mt-20"
          >
            <SecuritySection :auto-save="securityAutoSave" />
          </div>

          <div
            v-if="authStore.isAdmin"
            v-show="activeSection === 'advanced'"
            id="keepalive"
            class="space-y-6 mt-0 scroll-mt-20"
          >
            <KeepaliveSection :auto-save="keepaliveAutoSave" />
          </div>

          <div
            v-if="authStore.isAdmin"
            v-show="activeSection === 'advanced'"
            id="rateLimits"
            class="space-y-6 mt-0 scroll-mt-20"
          >
            <RateLimitsSection :auto-save="rateLimitsAutoSave" />
          </div>

          <div
            v-if="authStore.isAdmin"
            v-show="activeSection === 'advanced'"
            id="cors"
            class="space-y-6 mt-0 scroll-mt-20"
          >
            <CorsSection :auto-save="corsAutoSave" />
          </div>

          <div
            v-if="authStore.isAdmin"
            v-show="activeSection === 'advanced'"
            id="mcpSecurity"
            class="space-y-6 mt-0 scroll-mt-20"
          >
            <McpSecuritySection
              :auto-save="mcpSecurityAutoSave"
              @reset-defaults="showMcpResetDialog = true"
            />
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

    <!-- Tracing provider editor (side sheet) -->
    <TracingProviderSheet
      :provider="editingProvider"
      :editor="tracing"
      @close="closeProviderEditor"
    />
  </AppLayout>
</template>
