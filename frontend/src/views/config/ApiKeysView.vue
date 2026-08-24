<script setup lang="ts">
import { useClipboard } from "@vueuse/core";
import {
  BarChart3,
  Check,
  Copy,
  Edit,
  Key,
  KeyRound,
  Pencil,
  Plus,
  RotateCcw,
  Server,
  Trash2,
} from "@lucide/vue";
import { computed, onMounted, ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import { toast } from "vue-sonner";
import AccessScopePicker from "@/components/common/AccessScopePicker.vue";
import AccountBudgetBanner from "@/components/common/AccountBudgetBanner.vue";
import ConfirmDialog from "@/components/common/ConfirmDialog.vue";
import DateTimePicker from "@/components/common/DateTimePicker.vue";
import EmptyState from "@/components/common/EmptyState.vue";
import EmptyFilterResults from "@/components/common/EmptyFilterResults.vue";
import FilterBar from "@/components/common/FilterBar.vue";
import ContentSkeleton from "@/components/common/ContentSkeleton.vue";
import PageHeader from "@/components/common/PageHeader.vue";
import EmptyTableRow from "@/components/common/EmptyTableRow.vue";
import SortableHead from "@/components/common/SortableHead.vue";
import TableCellActions from "@/components/common/TableCellActions.vue";
import TableCellName from "@/components/common/TableCellName.vue";
import ApiKeyListItem from "@/components/common/ApiKeyListItem.vue";
import ApiKeySpendCell from "@/components/common/ApiKeySpendCell.vue";
import RateLimitBadge from "@/components/common/RateLimitBadge.vue";
import RateLimitField from "@/components/common/RateLimitField.vue";
import ApiKeyUsageSheet from "@/components/config/ApiKeyUsageSheet.vue";
import ViewToggle from "@/components/common/ViewToggle.vue";
import AppLayout from "@/components/layout/AppLayout.vue";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Sheet, SheetContent } from "@/components/ui/sheet";
import SheetHeaderBand from "@/components/common/SheetHeaderBand.vue";
import { Switch } from "@/components/ui/switch";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useErrorHandler } from "@/composables/useErrorHandler";
import { useTableFilter } from "@/composables/useTableFilter";
import { useViewMode } from "@/composables/useViewMode";
import { getApiKeyStatus, isApiKeyExpired } from "@/utils/apiKeys";
import {
  type ApiKeyCreate,
  type ApiKeyRead,
  type ApiKeyResponse,
  type ApiKeySpendSummary,
  type ApiKeyUpdate,
  type BudgetPeriod,
} from "@/services/api/apiKeys";
import { configApi } from "@/services/api/config";
import { meApi, type MeBudget } from "@/services/api/me";
import { useAuthStore } from "@/stores/auth";
import { useApiKeyStore } from "@/stores/apiKeys";
import { STORAGE_KEYS } from "@/constants/storageKeys";
import { formatDate } from "@/utils/format";

defineOptions({ name: "ApiKeysView" });

const { t } = useI18n();
const { handleSaveError, handleDeleteError } = useErrorHandler();
const { copy, copied: copiedKey } = useClipboard({ legacy: true });
const authStore = useAuthStore();
const apiKeyStore = useApiKeyStore();

const modelNames = ref<string[]>([]);
const mcpServerNames = ref<string[]>([]);

async function fetchModelNames() {
  try {
    modelNames.value = await configApi.getModelNames();
  } catch (e) {
    console.error("Failed to fetch model names:", e);
  }
}

async function fetchMcpServerNames() {
  try {
    mcpServerNames.value = await configApi.getMcpServerNames();
  } catch (e) {
    console.error("Failed to fetch MCP server names:", e);
  }
}

const showCreateDialog = ref(false);
const showKeyRevealDialog = ref(false);
const showEditDialog = ref(false);
const showDeleteDialog = ref(false);
const createdKey = ref<ApiKeyResponse | null>(null);
const editingKey = ref<ApiKeyRead | null>(null);
const editKeyName = ref("");
const editIsActive = ref(true);
// Scope state: null = unrestricted (allow all), [] = deny all, list = allowlist.
const editKeyModels = ref<string[] | null>(null);
const editKeyMcpServers = ref<string[] | null>(null);
const deletingKeyName = ref("");
const isSaving = ref(false);

const apiKeys = computed(() => apiKeyStore.apiKeys);
const spendByKey = computed(() => apiKeyStore.spendByKey);
const availableModels = computed(() => modelNames.value.map((name) => ({ id: name, name })));
const isLoading = computed(() => apiKeyStore.loading && !apiKeyStore.ready);

// Account-level budget envelope (admin-set). Informational: the enforced
// check runs server-side at request time; null when no budget is configured.
const accountBudget = ref<MeBudget | null>(null);

async function fetchAccountBudget() {
  try {
    accountBudget.value = await meApi.getBudget();
  } catch (e) {
    console.error("Failed to fetch account budget:", e);
  }
}

const {
  searchQuery,
  filteredItems: filteredApiKeys,
  clearFilters: clearBaseFilters,
} = useTableFilter(apiKeys, {
  searchFields: ["name"],
});

const viewMode = useViewMode(STORAGE_KEYS.API_KEYS_VIEW_MODE);

type SortField = "name" | "created" | "lastUsed";
const sortField = ref<SortField>("name");
const sortDir = ref<"asc" | "desc">("asc");

// Date columns default to newest-first; name defaults to A->Z.
const defaultDirFor = (field: SortField): "asc" | "desc" =>
  field === "created" || field === "lastUsed" ? "desc" : "asc";

function onSort(field: string) {
  if (field === sortField.value) {
    sortDir.value = sortDir.value === "asc" ? "desc" : "asc";
  } else {
    sortField.value = field as SortField;
    sortDir.value = defaultDirFor(field as SortField);
  }
}

const clearFilters = () => {
  clearBaseFilters();
  sortField.value = "name";
  sortDir.value = "asc";
};

const sortedApiKeys = computed(() => {
  const items = [...filteredApiKeys.value];
  const dir: 1 | -1 = sortDir.value === "asc" ? 1 : -1;

  // null/empty timestamps sort last regardless of direction.
  const compareDate = (
    av: string | null | undefined,
    bv: string | null | undefined,
    dir: 1 | -1
  ): number => {
    if (!av && !bv) return 0;
    if (!av) return 1;
    if (!bv) return -1;
    return av.localeCompare(bv) * dir;
  };

  switch (sortField.value) {
    case "created":
      items.sort((a, b) => compareDate(a.created_at, b.created_at, dir));
      break;
    case "lastUsed":
      items.sort((a, b) => compareDate(a.last_used_at, b.last_used_at, dir));
      break;
    case "name":
    default:
      items.sort((a, b) => a.name.localeCompare(b.name) * dir);
      break;
  }
  return items;
});

// --- Status / expiry / spend helpers ---------------------------------------

function spendFor(key: ApiKeyRead): ApiKeySpendSummary | undefined {
  return spendByKey.value[key.name];
}

// --- Usage detail sheet -----------------------------------------------------

const showUsageSheet = ref(false);
const usageKey = ref<ApiKeyRead | null>(null);

const openUsageSheet = (key: ApiKeyRead) => {
  usageKey.value = key;
  showUsageSheet.value = true;
};

// --- Create / edit form state ------------------------------------------------

/** "none" = lifetime budget: the cap applies to cumulative spend until reset. */
type BudgetPeriodChoice = BudgetPeriod | "none";

const periodOrNull = (choice: BudgetPeriodChoice): BudgetPeriod | null =>
  choice === "none" ? null : choice;

const newKeyName = ref("");
// Scope state: null = unrestricted (allow all), [] = deny all, list = allowlist.
const newKeyModels = ref<string[] | null>(null);
const newKeyMcpServers = ref<string[] | null>(null);
const newKeyNeverExpires = ref(true);
const newKeyExpiresAt = ref<string | null>(null); // ISO 8601 (UTC)
const newKeyBudgetUsd = ref<number | string | null>(null);
const newKeyBudgetPeriod = ref<BudgetPeriodChoice>("monthly");
const newKeyBudgetResetDay = ref(1);
const newKeyRateLimitRpm = ref<number | string | null>(null);

const editNeverExpires = ref(true);
const editExpiresAt = ref<string | null>(null);
const editBudgetUsd = ref<number | string | null>(null);
const editBudgetPeriod = ref<BudgetPeriodChoice>("monthly");
const editBudgetResetDay = ref(1);
const editRateLimitRpm = ref<number | string | null>(null);

/** Normalize an optional number input: empty input maps to null ("unset"). */
function normalizeOptionalNumber(value: number | string | null): number | null {
  if (value === null || value === "") return null;
  const n = typeof value === "string" ? Number(value) : value;
  return Number.isFinite(n) ? n : null;
}

const newKeyBudget = computed(() => normalizeOptionalNumber(newKeyBudgetUsd.value));
const editBudget = computed(() => normalizeOptionalNumber(editBudgetUsd.value));

const newKeyRateLimit = computed(() => normalizeOptionalNumber(newKeyRateLimitRpm.value));
const editRateLimit = computed(() => normalizeOptionalNumber(editRateLimitRpm.value));

/** Per-key rate limits must be positive integers (requests/minute). */
function isRateLimitInvalid(rateLimit: number | null): boolean {
  return rateLimit !== null && (!Number.isInteger(rateLimit) || rateLimit < 1);
}

/** Default expiry suggestion when the user disables "never expires": +30 days. */
function defaultExpiryIso(): string {
  return new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString();
}

// Prefill a sensible expiry when the toggle is switched off, so the form never
// sits in the ambiguous "dated but empty" state.
watch(newKeyNeverExpires, (never) => {
  if (!never && !newKeyExpiresAt.value) newKeyExpiresAt.value = defaultExpiryIso();
});
watch(editNeverExpires, (never) => {
  if (!never && !editExpiresAt.value) editExpiresAt.value = defaultExpiryIso();
});

const availableMcpServers = computed(() =>
  mcpServerNames.value.map((name) => ({ id: name, name }))
);

const openCreateDialog = () => {
  newKeyName.value = "";
  fetchModelNames();
  if (authStore.isAdmin) {
    fetchMcpServerNames();
  }
  newKeyModels.value = null;
  newKeyMcpServers.value = null;
  newKeyNeverExpires.value = true;
  newKeyExpiresAt.value = null;
  newKeyBudgetUsd.value = null;
  newKeyBudgetPeriod.value = "monthly";
  newKeyBudgetResetDay.value = 1;
  newKeyRateLimitRpm.value = null;
  showCreateDialog.value = true;
};

const createApiKey = async () => {
  if (!newKeyName.value.trim()) {
    toast.error(t("errors.validation.nameRequired"));
    return;
  }
  if (!newKeyNeverExpires.value && !newKeyExpiresAt.value) {
    toast.error(t("apiKeys.expiresRequired"));
    return;
  }
  const budgetUsd = newKeyBudget.value;
  if (budgetUsd !== null && budgetUsd <= 0) {
    toast.error(t("apiKeys.budgetMustBePositive"));
    return;
  }
  const rateLimit = newKeyRateLimit.value;
  if (authStore.isAdmin && isRateLimitInvalid(rateLimit)) {
    toast.error(t("apiKeys.rateLimitInvalid"));
    return;
  }

  isSaving.value = true;
  try {
    // undefined (from null) omits the field -> unrestricted; an explicit []
    // is preserved so "deny all" remains expressible.
    const payload: ApiKeyCreate = {
      name: newKeyName.value,
      allowed_models: newKeyModels.value ?? undefined,
      allowed_mcp_servers: newKeyMcpServers.value ?? undefined,
    };
    if (!newKeyNeverExpires.value && newKeyExpiresAt.value) {
      payload.expires_at = newKeyExpiresAt.value;
    }
    if (budgetUsd !== null) {
      payload.budget_usd = budgetUsd;
      payload.budget_period = periodOrNull(newKeyBudgetPeriod.value);
      // A reset day is only meaningful for monthly windows; the 1st is the
      // implicit default and need not be stored.
      if (newKeyBudgetPeriod.value === "monthly" && newKeyBudgetResetDay.value !== 1) {
        payload.budget_reset_day = newKeyBudgetResetDay.value;
      }
    }
    // Rate limiting is admin-only; viewers never send the field.
    if (authStore.isAdmin && rateLimit !== null) {
      payload.rate_limit_rpm = rateLimit;
    }
    const res = await apiKeyStore.createApiKey(payload);
    createdKey.value = res;
    showCreateDialog.value = false;
    showKeyRevealDialog.value = true;
  } catch (e) {
    handleSaveError(e);
  } finally {
    isSaving.value = false;
  }
};

const copyKey = async () => {
  if (!createdKey.value) return;
  await copy(createdKey.value.key);
};

const closeKeyRevealDialog = () => {
  showKeyRevealDialog.value = false;
  createdKey.value = null;
};

const openDeleteDialog = (keyName: string) => {
  deletingKeyName.value = keyName;
  showDeleteDialog.value = true;
};

const deleteApiKey = async () => {
  if (!deletingKeyName.value) return;

  isSaving.value = true;
  try {
    await apiKeyStore.deleteApiKey(deletingKeyName.value);
    toast.success(t("apiKeys.deleteSuccess"));
    showDeleteDialog.value = false;
    deletingKeyName.value = "";
  } catch (e) {
    handleDeleteError(e);
  } finally {
    isSaving.value = false;
  }
};

const openEditDialog = (key: ApiKeyRead) => {
  editingKey.value = key;
  editKeyName.value = key.name;
  editIsActive.value = key.is_active;
  // Keys have two effective states (allow-all vs. allowlist): an empty stored
  // list is normalized to null so it displays as "Allow all models".
  editKeyModels.value = key.allowed_models?.length ? [...key.allowed_models] : null;
  editKeyMcpServers.value = key.allowed_mcp_servers?.length ? [...key.allowed_mcp_servers] : null;
  editNeverExpires.value = key.expires_at === null;
  // Normalize to the same ISO shape DateTimePicker emits, so the unchanged
  // comparison in updateApiKey() compares instants, not string formatting.
  editExpiresAt.value = key.expires_at ? new Date(key.expires_at).toISOString() : null;
  editBudgetUsd.value = key.budget_usd;
  editBudgetPeriod.value = key.budget_period ?? "none";
  editBudgetResetDay.value = key.budget_reset_day ?? 1;
  editRateLimitRpm.value = key.rate_limit_rpm;
  fetchModelNames();
  if (authStore.isAdmin) {
    fetchMcpServerNames();
  }
  showEditDialog.value = true;
};

const updateApiKey = async () => {
  if (!editingKey.value) return;
  if (!editKeyName.value.trim()) {
    toast.error(t("errors.validation.nameRequired"));
    return;
  }
  if (!editNeverExpires.value && !editExpiresAt.value) {
    toast.error(t("apiKeys.expiresRequired"));
    return;
  }
  const original = editingKey.value;
  const budgetUsd = editBudget.value;
  if (budgetUsd !== null && budgetUsd <= 0) {
    toast.error(t("apiKeys.budgetMustBePositive"));
    return;
  }
  // Key-level budgets are fully self-service (raise, lower, clear, re-window):
  // spend is ultimately bounded by the admin-set account-level budget.
  const rateLimit = editRateLimit.value;
  if (authStore.isAdmin && isRateLimitInvalid(rateLimit)) {
    toast.error(t("apiKeys.rateLimitInvalid"));
    return;
  }

  isSaving.value = true;
  try {
    const data: ApiKeyUpdate = {};
    if (editKeyName.value !== original.name) {
      data.name = editKeyName.value;
    }
    if (editIsActive.value !== original.is_active) {
      data.is_active = editIsActive.value;
    }
    // Empty selection means "no restriction" (null = allow all), not deny-all.
    data.allowed_models = editKeyModels.value?.length ? editKeyModels.value : null;
    data.allowed_mcp_servers = editKeyMcpServers.value?.length ? editKeyMcpServers.value : null;

    // Expiry: only send when the effective value changed. Clearing sends an
    // explicit null (the backend treats it as "remove the expiry").
    const newExpiresIso = editNeverExpires.value ? null : editExpiresAt.value;
    const origExpiresIso = original.expires_at ? new Date(original.expires_at).toISOString() : null;
    if (newExpiresIso !== origExpiresIso) {
      data.expires_at = newExpiresIso;
    }

    // Budget: send all window fields together when any changed (the backend
    // validates the combination). Clearing the budget sends explicit nulls.
    const newPeriod = periodOrNull(editBudgetPeriod.value);
    const newResetDay =
      budgetUsd !== null && newPeriod === "monthly" && editBudgetResetDay.value !== 1
        ? editBudgetResetDay.value
        : null;
    const budgetChanged =
      budgetUsd !== original.budget_usd ||
      (budgetUsd !== null && newPeriod !== original.budget_period) ||
      (budgetUsd !== null && newResetDay !== (original.budget_reset_day ?? null));
    if (budgetChanged) {
      data.budget_usd = budgetUsd;
      data.budget_period = budgetUsd === null ? null : newPeriod;
      data.budget_reset_day = newResetDay;
    }

    // Rate limiting is admin-only; an explicit null clears the limit.
    if (authStore.isAdmin && rateLimit !== original.rate_limit_rpm) {
      data.rate_limit_rpm = rateLimit;
    }

    await apiKeyStore.updateApiKey(original.name, data);
    toast.success(t("apiKeys.editSuccess"));
    showEditDialog.value = false;
    editingKey.value = null;
    apiKeyStore.fetchSpendSummary(true);
    fetchAccountBudget();
  } catch (e) {
    handleSaveError(e);
  } finally {
    isSaving.value = false;
  }
};

const resetBudget = async () => {
  if (!editingKey.value) return;
  isSaving.value = true;
  try {
    await apiKeyStore.resetBudget(editingKey.value.name);
    toast.success(t("apiKeys.resetBudgetSuccess"));
    showEditDialog.value = false;
    editingKey.value = null;
  } catch (e) {
    handleSaveError(e);
  } finally {
    isSaving.value = false;
  }
};

onMounted(() => {
  apiKeyStore.fetchApiKeys();
  apiKeyStore.fetchSpendSummary();
  fetchAccountBudget();
});
</script>

<template>
  <AppLayout layoutMode="full">
    <template #header>
      <header class="config-header-bar px-4 sm:px-6 py-4">
        <PageHeader
          :title="authStore.isAdmin ? t('apiKeys.title') : t('apiKeys.myTitle')"
          :description="t('apiKeys.description')"
          :icon="KeyRound"
        >
          <template #actions>
            <Button @click="openCreateDialog" class="btn-action">
              <Plus class="w-4 h-4 mr-2" />
              {{ t("apiKeys.create") }}
            </Button>
          </template>
        </PageHeader>
      </header>
    </template>

    <!-- Toolbar band (flush) -->
    <div v-if="apiKeys.length > 0 || isLoading" class="config-toolbar px-4 sm:px-6 py-3">
      <FilterBar
        v-model:search-query="searchQuery"
        :search-placeholder="t('common.searchPlaceholder')"
        :result-count="sortedApiKeys.length"
        :total-count="apiKeys.length"
        @clear-filters="clearFilters"
      >
        <ViewToggle v-model="viewMode" />
      </FilterBar>
    </div>

    <!-- Account-level budget envelope (only when the admin set one) -->
    <div v-if="accountBudget?.budget_usd != null" class="px-4 sm:px-6 pt-3">
      <AccountBudgetBanner :budget="accountBudget" />
    </div>

    <!-- Content area -->
    <div class="config-content">
      <div
        v-if="isLoading && apiKeys.length === 0"
        class="h-full flex items-start justify-center animate-fade-in px-6"
      >
        <ContentSkeleton />
      </div>
      <div
        v-else-if="apiKeys.length === 0"
        class="h-full flex items-center justify-center animate-fade-in px-6"
      >
        <EmptyState
          :icon="Key"
          :text="t('apiKeys.noKeys')"
          :show-cta="true"
          :cta-text="t('apiKeys.createFirst')"
          @click="openCreateDialog"
        />
      </div>
      <template v-else>
        <!-- Table view (default) -->
        <Table
          v-if="viewMode === 'table'"
          class="table-modern"
          container-class="h-full border-0 bg-transparent rounded-none overflow-x-auto"
        >
          <TableHeader class="config-thead">
            <TableRow class="bg-transparent hover:bg-transparent hover:border-l-transparent">
              <SortableHead
                :label="t('apiKeys.name')"
                sort-key="name"
                :active-field="sortField"
                :active-dir="sortDir"
                @sort="onSort"
              />
              <TableHead>{{ t("apiKeys.status") }}</TableHead>
              <TableHead>{{ t("apiKeys.models") }}</TableHead>
              <TableHead class="hidden lg:table-cell">{{ t("apiKeys.mcpServers") }}</TableHead>
              <TableHead>{{ t("apiKeys.expires") }}</TableHead>
              <TableHead>{{ t("apiKeys.spend") }}</TableHead>
              <SortableHead
                :label="t('apiKeys.created')"
                sort-key="created"
                :active-field="sortField"
                :active-dir="sortDir"
                class="hidden 2xl:table-cell"
                @sort="onSort"
              />
              <SortableHead
                :label="t('apiKeys.lastUsed')"
                sort-key="lastUsed"
                :active-field="sortField"
                :active-dir="sortDir"
                class="hidden md:table-cell"
                @sort="onSort"
              />
              <TableHead class="w-24 text-right">
                <span class="sr-only">{{ t("common.actions") }}</span>
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody class="row-stagger">
            <TableRow v-for="key in sortedApiKeys" :key="key.name">
              <TableCellName>
                <button
                  type="button"
                  class="hover:underline underline-offset-4 cursor-pointer text-left"
                  :title="t('apiKeys.viewUsage')"
                  @click="openUsageSheet(key)"
                >
                  {{ key.name }}
                </button>
              </TableCellName>
              <TableCell>
                <div class="flex flex-wrap items-center gap-1.5">
                  <Badge
                    v-if="getApiKeyStatus(key) === 'active'"
                    variant="secondary"
                    class="font-medium text-[11px]"
                  >
                    {{ t("apiKeys.active") }}
                  </Badge>
                  <Badge
                    v-else-if="getApiKeyStatus(key) === 'expired'"
                    variant="outline"
                    class="font-medium text-[11px] border-status-warning/60 text-status-warning"
                  >
                    {{ t("apiKeys.expired") }}
                  </Badge>
                  <Badge v-else variant="destructive" class="font-medium text-[11px]">
                    {{ t("apiKeys.disabled") }}
                  </Badge>
                  <RateLimitBadge v-if="key.rate_limit_rpm != null" :rpm="key.rate_limit_rpm" />
                </div>
              </TableCell>
              <TableCell>
                <div class="flex flex-wrap gap-1.5">
                  <Badge
                    v-if="!key.allowed_models || key.allowed_models.length === 0"
                    variant="secondary"
                    class="font-normal"
                  >
                    {{ t("apiKeys.allModels") }}
                  </Badge>
                  <template v-else>
                    <Badge
                      v-for="model in key.allowed_models.slice(0, 3)"
                      :key="model"
                      variant="outline"
                      class="font-mono text-[11px]"
                    >
                      {{ model }}
                    </Badge>
                    <Badge
                      v-if="key.allowed_models.length > 3"
                      variant="outline"
                      class="font-normal text-muted-foreground"
                    >
                      +{{ key.allowed_models.length - 3 }}
                    </Badge>
                  </template>
                </div>
              </TableCell>
              <TableCell class="hidden lg:table-cell">
                <div class="flex flex-wrap gap-1.5">
                  <Badge
                    v-if="!key.allowed_mcp_servers || key.allowed_mcp_servers.length === 0"
                    variant="secondary"
                    class="font-normal"
                  >
                    {{ t("apiKeys.allMcpServers") }}
                  </Badge>
                  <template v-else>
                    <Badge
                      v-for="server in key.allowed_mcp_servers.slice(0, 3)"
                      :key="server"
                      variant="outline"
                      class="text-[11px]"
                    >
                      <Server class="w-3 h-3 mr-1" />
                      {{ server }}
                    </Badge>
                    <Badge
                      v-if="key.allowed_mcp_servers.length > 3"
                      variant="outline"
                      class="font-normal text-muted-foreground"
                    >
                      +{{ key.allowed_mcp_servers.length - 3 }}
                    </Badge>
                  </template>
                </div>
              </TableCell>
              <TableCell class="text-data text-muted-foreground">
                <span :class="{ 'text-status-warning': isApiKeyExpired(key.expires_at) }">
                  {{ key.expires_at ? formatDate(key.expires_at) : t("apiKeys.never") }}
                </span>
              </TableCell>
              <TableCell>
                <ApiKeySpendCell :spend="spendFor(key)" />
              </TableCell>
              <TableCell class="text-data text-muted-foreground hidden 2xl:table-cell">{{
                formatDate(key.created_at)
              }}</TableCell>
              <TableCell class="text-data text-muted-foreground hidden md:table-cell">{{
                key.last_used_at ? formatDate(key.last_used_at) : t("apiKeys.never")
              }}</TableCell>
              <TableCellActions>
                <div class="flex items-center justify-end gap-1">
                  <Button
                    variant="ghost"
                    size="icon"
                    class="h-9 w-9"
                    :aria-label="t('apiKeys.viewUsage')"
                    @click="openUsageSheet(key)"
                  >
                    <BarChart3 class="w-4 h-4 icon-btn-muted" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    class="h-9 w-9"
                    :disabled="isSaving"
                    :aria-label="t('common.edit')"
                    @click="openEditDialog(key)"
                  >
                    <Edit class="w-4 h-4 icon-btn-muted" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    class="h-9 w-9 text-destructive hover:text-destructive hover:bg-destructive/10"
                    :disabled="isSaving"
                    :aria-label="t('common.delete')"
                    @click="openDeleteDialog(key.name)"
                  >
                    <Trash2 class="w-4 h-4" />
                  </Button>
                </div>
              </TableCellActions>
            </TableRow>
            <EmptyTableRow v-if="sortedApiKeys.length === 0" :colspan="9" @clear="clearFilters" />
          </TableBody>
        </Table>

        <!-- List view -->
        <div v-else class="config-scroll">
          <EmptyFilterResults v-if="sortedApiKeys.length === 0" @clear="clearFilters" />
          <div v-if="sortedApiKeys.length > 0" class="config-list list-stagger">
            <ApiKeyListItem
              v-for="key in sortedApiKeys"
              :key="key.name"
              :api-key="key"
              :spend="spendFor(key)"
              :is-loading="isSaving"
              @edit="openEditDialog(key)"
              @delete="openDeleteDialog(key.name)"
              @view-usage="openUsageSheet(key)"
            />
          </div>
        </div>
      </template>
    </div>

    <Sheet v-model:open="showCreateDialog">
      <SheetContent
        side="right"
        class="w-full sm:max-w-[480px] h-full flex flex-col p-0 gap-0 overflow-hidden border-l border-border/80 bg-card"
      >
        <!-- Header band -->
        <SheetHeaderBand :icon="KeyRound">
          <template #title>{{ t("apiKeys.create") }}</template>
          <template #description>{{ t("apiKeys.createDescription") }}</template>
        </SheetHeaderBand>
        <div class="flex-1 overflow-y-auto px-4 sm:px-6 py-4 space-y-4">
          <div class="space-y-2">
            <Label for="name">{{ t("apiKeys.name") }}</Label>
            <Input id="name" v-model="newKeyName" :placeholder="t('apiKeys.namePlaceholder')" />
          </div>
          <div class="space-y-2">
            <Label>{{ t("apiKeys.allowedModels") }}</Label>
            <AccessScopePicker
              v-model="newKeyModels"
              :items="availableModels"
              :all-label="t('apiKeys.allModels')"
              :all-help-text="t('apiKeys.modelsHelp')"
              :search-placeholder="t('apiKeys.searchModels')"
              :empty-available-text="t('apiKeys.noModelsAvailable')"
              :empty-search-text="t('apiKeys.noModelsFound')"
              :empty-text="t('apiKeys.emptyModelsInfo')"
              mono
            />
          </div>
          <!-- MCP Servers (admin only) -->
          <div v-if="authStore.isAdmin" class="space-y-2">
            <Label>{{ t("apiKeys.allowedMcpServers") }}</Label>
            <AccessScopePicker
              v-model="newKeyMcpServers"
              :items="availableMcpServers"
              :all-label="t('apiKeys.allMcpServers')"
              :all-help-text="t('apiKeys.mcpServersHelp')"
              :search-placeholder="t('apiKeys.searchMcpServers')"
              :empty-available-text="t('apiKeys.noMcpServersAvailable')"
              :empty-search-text="t('apiKeys.noMcpServersFound')"
              :empty-text="t('apiKeys.emptyMcpServersInfo')"
              :icon="Server"
            />
          </div>
          <!-- Expiry -->
          <div class="space-y-2">
            <div class="flex items-center justify-between">
              <Label for="new-never-expires">{{ t("apiKeys.neverExpires") }}</Label>
              <Switch id="new-never-expires" v-model="newKeyNeverExpires" />
            </div>
            <template v-if="!newKeyNeverExpires">
              <DateTimePicker
                v-model="newKeyExpiresAt"
                :placeholder="t('apiKeys.expiresPlaceholder')"
                :time-label="t('apiKeys.expiresTimeLabel')"
                :clear-label="t('common.clear')"
              />
              <p class="text-xs text-muted-foreground">{{ t("apiKeys.expiresHelp") }}</p>
            </template>
          </div>
          <!-- Budget -->
          <div class="space-y-2">
            <Label for="new-budget-usd">{{ t("apiKeys.budget") }}</Label>
            <div class="flex items-center gap-2">
              <div class="relative flex-1">
                <span
                  class="absolute left-3 top-1/2 -translate-y-1/2 text-sm text-muted-foreground pointer-events-none"
                  aria-hidden="true"
                  >$</span
                >
                <Input
                  id="new-budget-usd"
                  v-model.number="newKeyBudgetUsd"
                  type="number"
                  min="0"
                  step="any"
                  :placeholder="t('apiKeys.budgetPlaceholder')"
                  class="pl-7"
                />
              </div>
              <Select v-model="newKeyBudgetPeriod" :disabled="newKeyBudget === null">
                <SelectTrigger class="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">{{ t("apiKeys.periodNone") }}</SelectItem>
                  <SelectItem value="daily">{{ t("apiKeys.periodDaily") }}</SelectItem>
                  <SelectItem value="weekly">{{ t("apiKeys.periodWeekly") }}</SelectItem>
                  <SelectItem value="monthly">{{ t("apiKeys.periodMonthly") }}</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <!-- Monthly reset day -->
            <div
              v-if="newKeyBudgetPeriod === 'monthly' && newKeyBudget !== null"
              class="flex items-center gap-2 text-xs text-muted-foreground"
            >
              <span>{{ t("apiKeys.resetDayPrefix") }}</span>
              <Select v-model="newKeyBudgetResetDay">
                <SelectTrigger class="w-18 h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent class="max-h-56">
                  <SelectItem v-for="day in 31" :key="day" :value="day">
                    {{ day }}
                  </SelectItem>
                </SelectContent>
              </Select>
              <span>{{ t("apiKeys.resetDaySuffix") }}</span>
            </div>
            <p class="text-xs text-muted-foreground">
              {{
                newKeyBudgetPeriod === "none"
                  ? t("apiKeys.budgetHelpNone")
                  : t("apiKeys.budgetHelp")
              }}
            </p>
          </div>
          <!-- Rate limit (admin only) -->
          <RateLimitField
            v-if="authStore.isAdmin"
            id="new-rate-limit"
            v-model="newKeyRateLimitRpm"
          />
        </div>
        <div
          class="flex items-center justify-end gap-2 px-4 sm:px-6 py-4 border-t border-border/60 bg-muted/10 shrink-0"
        >
          <Button variant="outline" @click="showCreateDialog = false">{{
            t("common.cancel")
          }}</Button>
          <Button @click="createApiKey" :disabled="isSaving">
            <Key class="w-4 h-4 mr-2" />
            {{ t("apiKeys.generate") }}
          </Button>
        </div>
      </SheetContent>
    </Sheet>

    <Sheet v-model:open="showEditDialog">
      <SheetContent
        side="right"
        class="w-full sm:max-w-[480px] h-full flex flex-col p-0 gap-0 overflow-hidden border-l border-border/80 bg-card"
      >
        <!-- Header band -->
        <SheetHeaderBand :icon="Pencil">
          <template #title>{{ t("apiKeys.edit") }}</template>
          <template #description>{{ editingKey?.name || t("apiKeys.editDescription") }}</template>
        </SheetHeaderBand>
        <div class="flex-1 overflow-y-auto px-4 sm:px-6 py-4 space-y-4">
          <div class="space-y-2">
            <Label for="edit-name">{{ t("apiKeys.name") }}</Label>
            <Input
              id="edit-name"
              v-model="editKeyName"
              :placeholder="t('apiKeys.namePlaceholder')"
            />
          </div>
          <!-- Deliberate enable/disable control (replaces the old in-row toggle) -->
          <div
            class="flex items-center justify-between gap-4 rounded-lg border border-border/60 bg-muted/30 px-3 py-2.5"
          >
            <div class="space-y-0.5 min-w-0">
              <Label for="edit-is-active" class="text-sm font-medium">
                {{ t("apiKeys.keyEnabled") }}
              </Label>
              <p class="text-xs text-muted-foreground">{{ t("apiKeys.keyEnabledHelp") }}</p>
            </div>
            <Switch id="edit-is-active" v-model="editIsActive" class="shrink-0" />
          </div>
          <div class="space-y-2">
            <Label>{{ t("apiKeys.allowedModels") }}</Label>
            <AccessScopePicker
              v-model="editKeyModels"
              :items="availableModels"
              :all-label="t('apiKeys.allModels')"
              :all-help-text="t('apiKeys.modelsHelp')"
              :search-placeholder="t('apiKeys.searchModels')"
              :empty-available-text="t('apiKeys.noModelsAvailable')"
              :empty-search-text="t('apiKeys.noModelsFound')"
              :empty-text="t('apiKeys.emptyModelsInfo')"
              mono
            />
          </div>
          <!-- MCP Servers (admin only) -->
          <div v-if="authStore.isAdmin" class="space-y-2">
            <Label>{{ t("apiKeys.allowedMcpServers") }}</Label>
            <AccessScopePicker
              v-model="editKeyMcpServers"
              :items="availableMcpServers"
              :all-label="t('apiKeys.allMcpServers')"
              :all-help-text="t('apiKeys.mcpServersHelp')"
              :search-placeholder="t('apiKeys.searchMcpServers')"
              :empty-available-text="t('apiKeys.noMcpServersAvailable')"
              :empty-search-text="t('apiKeys.noMcpServersFound')"
              :empty-text="t('apiKeys.emptyMcpServersInfo')"
              :icon="Server"
            />
          </div>
          <!-- Expiry -->
          <div class="space-y-2">
            <div class="flex items-center justify-between">
              <Label for="edit-never-expires">{{ t("apiKeys.neverExpires") }}</Label>
              <Switch id="edit-never-expires" v-model="editNeverExpires" />
            </div>
            <template v-if="!editNeverExpires">
              <DateTimePicker
                v-model="editExpiresAt"
                :placeholder="t('apiKeys.expiresPlaceholder')"
                :time-label="t('apiKeys.expiresTimeLabel')"
                :clear-label="t('common.clear')"
              />
              <p class="text-xs text-muted-foreground">{{ t("apiKeys.expiresHelp") }}</p>
            </template>
          </div>
          <!-- Budget -->
          <div class="space-y-2">
            <Label for="edit-budget-usd">{{ t("apiKeys.budget") }}</Label>
            <div class="flex items-center gap-2">
              <div class="relative flex-1">
                <span
                  class="absolute left-3 top-1/2 -translate-y-1/2 text-sm text-muted-foreground pointer-events-none"
                  aria-hidden="true"
                  >$</span
                >
                <Input
                  id="edit-budget-usd"
                  v-model.number="editBudgetUsd"
                  type="number"
                  min="0"
                  step="any"
                  :placeholder="t('apiKeys.budgetPlaceholder')"
                  class="pl-7"
                />
              </div>
              <Select v-model="editBudgetPeriod" :disabled="editBudget === null">
                <SelectTrigger class="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">{{ t("apiKeys.periodNone") }}</SelectItem>
                  <SelectItem value="daily">{{ t("apiKeys.periodDaily") }}</SelectItem>
                  <SelectItem value="weekly">{{ t("apiKeys.periodWeekly") }}</SelectItem>
                  <SelectItem value="monthly">{{ t("apiKeys.periodMonthly") }}</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <!-- Monthly reset day -->
            <div
              v-if="editBudgetPeriod === 'monthly' && editBudget !== null"
              class="flex items-center gap-2 text-xs text-muted-foreground"
            >
              <span>{{ t("apiKeys.resetDayPrefix") }}</span>
              <Select v-model="editBudgetResetDay">
                <SelectTrigger class="w-18 h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent class="max-h-56">
                  <SelectItem v-for="day in 31" :key="day" :value="day">
                    {{ day }}
                  </SelectItem>
                </SelectContent>
              </Select>
              <span>{{ t("apiKeys.resetDaySuffix") }}</span>
            </div>
            <p class="text-xs text-muted-foreground">
              {{
                editBudgetPeriod === "none" ? t("apiKeys.budgetHelpNone") : t("apiKeys.budgetHelp")
              }}
            </p>
            <!-- Budget reset (owner self-service; only when a budget is configured) -->
            <div
              v-if="editingKey?.budget_usd != null"
              class="flex items-center justify-between rounded-lg border border-border/60 bg-muted/30 px-3 py-2"
            >
              <span class="text-xs text-muted-foreground">
                {{ t("apiKeys.resetBudgetHint") }}
              </span>
              <Button variant="outline" size="sm" :disabled="isSaving" @click="resetBudget">
                <RotateCcw class="w-3.5 h-3.5 mr-1.5" />
                {{ t("apiKeys.resetBudget") }}
              </Button>
            </div>
          </div>
          <!-- Rate limit (admin only) -->
          <RateLimitField
            v-if="authStore.isAdmin"
            id="edit-rate-limit"
            v-model="editRateLimitRpm"
          />
        </div>
        <div
          class="flex items-center justify-end gap-2 px-4 sm:px-6 py-4 border-t border-border/60 bg-muted/10 shrink-0"
        >
          <Button variant="outline" @click="showEditDialog = false">{{
            t("common.cancel")
          }}</Button>
          <Button @click="updateApiKey" :disabled="isSaving">
            <Pencil class="w-4 h-4 mr-2" />
            {{ t("apiKeys.save") }}
          </Button>
        </div>
      </SheetContent>
    </Sheet>

    <Dialog v-model:open="showKeyRevealDialog">
      <DialogContent class="brand-panel max-w-md">
        <DialogHeader>
          <DialogTitle class="flex items-center gap-2">
            <Key class="w-5 h-5 text-status-success" />
            {{ t("apiKeys.createdTitle") }}
          </DialogTitle>
          <DialogDescription>{{ t("apiKeys.createdDescription") }}</DialogDescription>
        </DialogHeader>
        <div class="py-4">
          <div class="code-container p-4">
            <div class="break-all font-mono text-sm select-all">{{ createdKey?.key }}</div>
          </div>
        </div>
        <DialogFooter class="sm:justify-end gap-2">
          <Button @click="copyKey" variant="outline" class="w-full sm:w-auto">
            <Check v-if="copiedKey" class="w-4 h-4 mr-2 text-status-success" />
            <Copy v-else class="w-4 h-4 mr-2" />
            {{ copiedKey ? t("apiKeys.copied") : t("apiKeys.copyKey") }}
          </Button>
          <Button @click="closeKeyRevealDialog" class="w-full sm:w-auto">
            {{ t("apiKeys.iHaveSaved") }}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>

    <ConfirmDialog
      v-model:open="showDeleteDialog"
      :title="t('apiKeys.deleteTitle')"
      :description="t('apiKeys.deleteDescription', { name: deletingKeyName })"
      :confirm-text="t('apiKeys.delete')"
      :cancel-text="t('common.cancel')"
      :loading="isLoading"
      @confirm="deleteApiKey"
    />

    <ApiKeyUsageSheet v-model:open="showUsageSheet" :api-key="usageKey" />
  </AppLayout>
</template>
