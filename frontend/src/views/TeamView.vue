<script setup lang="ts">
import {
  Boxes,
  Check,
  KeyRound,
  Loader2,
  Plus,
  RotateCcw,
  Trash2,
  UserCheck,
  UserPen,
  Users,
  UserX,
  Wallet,
} from "@lucide/vue";
import { computed, onMounted, ref } from "vue";
import { useI18n } from "vue-i18n";
import { toast } from "vue-sonner";
import { useAuthStore } from "@/stores/auth";
import { useModelStore } from "@/stores/models";
import { getErrorMessage } from "@/utils/error";
import { passwordRequirementsText, validatePasswordStrength } from "@/utils/password";
import AppLayout from "@/components/layout/AppLayout.vue";
import AccessScopePicker from "@/components/common/AccessScopePicker.vue";
import ConfirmDialog from "@/components/common/ConfirmDialog.vue";
import EmptyState from "@/components/common/EmptyState.vue";
import EmptyTableRow from "@/components/common/EmptyTableRow.vue";
import FilterBar from "@/components/common/FilterBar.vue";
import LoadingState from "@/components/common/LoadingState.vue";
import MemberBudgetCell from "@/components/common/MemberBudgetCell.vue";
import PageHeader from "@/components/common/PageHeader.vue";
import SortableHead from "@/components/common/SortableHead.vue";
import TableCellActions from "@/components/common/TableCellActions.vue";
import TableCellName from "@/components/common/TableCellName.vue";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Sheet, SheetContent } from "@/components/ui/sheet";
import SheetHeaderBand from "@/components/common/SheetHeaderBand.vue";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useTableFilter } from "@/composables/useTableFilter";
import {
  teamApi,
  type TeamBudgetPeriod,
  type TeamMember,
  type TeamMemberBudgetUpdate,
  type TeamRole,
} from "@/services/api/team";
import { formatCost, formatDate } from "@/utils/format";

defineOptions({ name: "TeamView" });

const { t } = useI18n();
const authStore = useAuthStore();
const modelStore = useModelStore();

// --- State ---
const members = ref<TeamMember[]>([]);
const isLoading = ref(true);
const loadError = ref<string | null>(null);

// Create member dialog
const showCreateDialog = ref(false);
const createUsername = ref("");
const createPassword = ref("");
const createConfirmPassword = ref("");
const createError = ref<string | null>(null);
const isCreating = ref(false);

// Reset password dialog
const showResetDialog = ref(false);
const resettingMember = ref<TeamMember | null>(null);
const resetPassword = ref("");
const resetConfirmPassword = ref("");
const resetError = ref<string | null>(null);
const isResetting = ref(false);

// Rename username dialog
const showRenameDialog = ref(false);
const renamingMember = ref<TeamMember | null>(null);
const renameUsername = ref("");
const renameError = ref<string | null>(null);
const isRenaming = ref(false);

// Delete confirmation dialog
const showDeleteDialog = ref(false);
const deletingMember = ref<TeamMember | null>(null);
const isDeleting = ref(false);

// Role change confirmation dialog (the role cell dropdown picks the new role,
// the dialog confirms it because the member's sessions are revoked on change)
const showRoleDialog = ref(false);
const roleTarget = ref<TeamMember | null>(null);
const roleNewValue = ref<TeamRole>("viewer");
const isUpdatingRole = ref(false);

// Deactivate/reactivate confirmation dialogs
const showDeactivateDialog = ref(false);
const showReactivateDialog = ref(false);
const togglingMember = ref<TeamMember | null>(null);
const isTogglingActive = ref(false);

// Manage allowed models dialog (admin sets a member's per-user model allowlist).
// memberModels mirrors the API semantics: null = unrestricted, [] = deny all.
const showModelsDialog = ref(false);
const editingMember = ref<TeamMember | null>(null);
const memberModels = ref<string[] | null>(null);
const isSavingModels = ref(false);

// Manage account budget dialog (admin sets a member's account-level spend cap,
// enforced across all of the member's keys; key-level budgets stay self-service).
type BudgetPeriodChoice = TeamBudgetPeriod | "none"; // "none" = lifetime cap
const showBudgetDialog = ref(false);
const budgetMember = ref<TeamMember | null>(null);
const budgetUsd = ref<number | string | null>(null);
const budgetPeriod = ref<BudgetPeriodChoice>("monthly");
const budgetResetDay = ref(1);
const isSavingBudget = ref(false);
const isResettingBudget = ref(false);

/** Normalize an optional number input: empty input maps to null ("no limit"). */
const normalizedBudgetUsd = computed(() => {
  const value = budgetUsd.value;
  if (value === null || value === "") return null;
  const n = typeof value === "string" ? Number(value) : value;
  return Number.isFinite(n) ? n : null;
});

// --- Table Filters & Sorting ---
type SortField = "username" | "role" | "status" | "created";
const sortField = ref<SortField>("username");
const sortDir = ref<"asc" | "desc">("asc");

const {
  searchQuery,
  filteredItems: filteredMembers,
  clearFilters: clearBaseFilters,
} = useTableFilter<TeamMember>(members, {
  searchFields: ["username"],
});

const defaultDirFor = (field: SortField): "asc" | "desc" => (field === "created" ? "desc" : "asc");

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
  sortField.value = "username";
  sortDir.value = "asc";
};

// null/empty timestamps sort last regardless of direction.
function compareDate(
  av: string | null | undefined,
  bv: string | null | undefined,
  dir: 1 | -1
): number {
  if (!av && !bv) return 0;
  if (!av) return 1;
  if (!bv) return -1;
  return av.localeCompare(bv) * dir;
}

const sortedMembers = computed(() => {
  const items = [...filteredMembers.value];
  const dir: 1 | -1 = sortDir.value === "asc" ? 1 : -1;
  switch (sortField.value) {
    case "created":
      items.sort((a, b) => compareDate(a.created_at, b.created_at, dir));
      break;
    case "role":
      items.sort((a, b) => a.role.localeCompare(b.role) * dir);
      break;
    case "status":
      items.sort((a, b) => {
        const statusA = a.is_active ? 1 : 0;
        const statusB = b.is_active ? 1 : 0;
        return (statusA - statusB) * dir;
      });
      break;
    case "username":
    default:
      items.sort((a, b) => a.username.localeCompare(b.username) * dir);
      break;
  }
  return items;
});

// --- Model allowlist helpers ---
const availableModels = computed(() =>
  modelStore.models.map((m) => ({ id: m.name, name: m.name }))
);

// --- Role / status helpers ---
// Own-row actions (role change, deactivate) are disabled: the backend refuses
// them anyway, and hiding the affordance avoids a pointless error roundtrip.
const isSelf = (member: TeamMember) => member.username === authStore.username;

const roleLabel = (role: TeamRole) => (role === "admin" ? t("team.admin") : t("team.viewer"));

const ROLE_OPTIONS: TeamRole[] = ["admin", "viewer"];

/** Patch one member in place from an API response (role/active/flag changes). */
function patchMember(updated: TeamMember) {
  const idx = members.value.findIndex((m) => m.id === updated.id);
  if (idx !== -1) {
    members.value[idx] = updated;
  }
}

// --- Methods ---
async function loadMembers() {
  isLoading.value = true;
  loadError.value = null;
  try {
    members.value = await teamApi.listMembers();
  } catch {
    loadError.value = t("errors.fetchFailed");
  } finally {
    isLoading.value = false;
  }
}

function openCreateDialog() {
  createUsername.value = "";
  createPassword.value = "";
  createConfirmPassword.value = "";
  createError.value = null;
  showCreateDialog.value = true;
}

async function handleCreate() {
  createError.value = null;

  const username = createUsername.value.trim();
  if (!username) {
    createError.value = t("errors.validation.nameRequired");
    return;
  }
  if (username.length > 64 || !/^[a-zA-Z0-9_-]+$/.test(username)) {
    createError.value = t("team.usernameInvalid");
    return;
  }
  if (createPassword.value.length < 8) {
    createError.value = t("auth.passwordTooShort");
    return;
  }
  if (createPassword.value !== createConfirmPassword.value) {
    createError.value = t("team.passwordMismatch");
    return;
  }
  const strengthError = validatePasswordStrength(createPassword.value);
  if (strengthError) {
    createError.value = strengthError;
    return;
  }

  isCreating.value = true;
  try {
    await teamApi.createMember({
      username,
      password: createPassword.value,
    });
    showCreateDialog.value = false;
    // The admin chose the initial password, so the member is flagged to
    // replace it on first login.
    toast.success(t("team.createSuccess"), {
      description: t("team.memberMustSetPassword"),
    });
    await loadMembers();
  } catch (err: unknown) {
    createError.value = getErrorMessage(err) || t("errors.saveFailed");
  } finally {
    isCreating.value = false;
  }
}

function openResetDialog(member: TeamMember) {
  resettingMember.value = member;
  resetPassword.value = "";
  resetConfirmPassword.value = "";
  resetError.value = null;
  showResetDialog.value = true;
}

async function handleReset() {
  if (!resettingMember.value) return;
  resetError.value = null;

  if (resetPassword.value.length < 8) {
    resetError.value = t("auth.passwordTooShort");
    return;
  }
  if (resetPassword.value !== resetConfirmPassword.value) {
    resetError.value = t("team.passwordMismatch");
    return;
  }
  const strengthError = validatePasswordStrength(resetPassword.value);
  if (strengthError) {
    resetError.value = strengthError;
    return;
  }

  isResetting.value = true;
  try {
    await teamApi.resetPassword(resettingMember.value.id, resetPassword.value);
    showResetDialog.value = false;
    // An admin reset flags the member for a forced password change and
    // revokes their sessions; the table reflects it after the reload.
    toast.success(t("team.passwordResetSuccess"), {
      description: t("team.memberMustSetPassword"),
    });
    await loadMembers();
  } catch (err: unknown) {
    resetError.value = getErrorMessage(err) || t("errors.saveFailed");
  } finally {
    isResetting.value = false;
  }
}

function openRenameDialog(member: TeamMember) {
  renamingMember.value = member;
  renameUsername.value = "";
  renameError.value = null;
  showRenameDialog.value = true;
}

async function handleRename() {
  if (!renamingMember.value) return;
  renameError.value = null;

  const username = renameUsername.value.trim();
  if (!username) {
    renameError.value = t("errors.validation.nameRequired");
    return;
  }
  if (username.length > 64 || !/^[a-zA-Z0-9_-]+$/.test(username)) {
    renameError.value = t("team.usernameInvalid");
    return;
  }
  if (username.toLowerCase() === renamingMember.value.username.toLowerCase()) {
    renameError.value = t("team.usernameUnchanged");
    return;
  }

  isRenaming.value = true;
  try {
    const updated = await teamApi.updateUsername(renamingMember.value.id, username);
    const idx = members.value.findIndex((m) => m.id === renamingMember.value!.id);
    if (idx !== -1) {
      members.value[idx] = { ...members.value[idx], username: updated.username };
    }
    // Self-rename: the old JWT's `sub` is stale, so swap in the fresh token.
    if (updated.access_token) {
      authStore.setToken(updated.access_token);
    }
    showRenameDialog.value = false;
    toast.success(t("team.renameSuccess", { username: updated.username }));
  } catch (err: unknown) {
    const httpErr = err as { status?: number };
    if (httpErr.status === 409) {
      renameError.value = t("team.usernameTaken");
    } else {
      renameError.value = getErrorMessage(err) || t("errors.saveFailed");
    }
  } finally {
    isRenaming.value = false;
  }
}

function openDeleteDialog(member: TeamMember) {
  deletingMember.value = member;
  showDeleteDialog.value = true;
}

async function handleDelete() {
  if (!deletingMember.value) return;

  // Prevent self-deletion to avoid account lockout
  if (deletingMember.value.username === authStore.username) {
    toast.error(t("team.cannotDeleteSelf"));
    showDeleteDialog.value = false;
    return;
  }

  isDeleting.value = true;
  try {
    await teamApi.deleteMember(deletingMember.value.id);
    showDeleteDialog.value = false;
    toast.success(t("team.deleteSuccess"));
    await loadMembers();
  } catch (err: unknown) {
    toast.error(getErrorMessage(err) || t("errors.deleteFailed"));
  } finally {
    isDeleting.value = false;
  }
}

function openRoleDialog(member: TeamMember, role: TeamRole) {
  if (member.role === role) return;
  roleTarget.value = member;
  roleNewValue.value = role;
  showRoleDialog.value = true;
}

async function handleRoleChange() {
  if (!roleTarget.value) return;

  isUpdatingRole.value = true;
  try {
    // On a real change the backend revokes the member's sessions, so they
    // must log in again at the new privilege level.
    const updated = await teamApi.updateRole(roleTarget.value.id, roleNewValue.value);
    patchMember(updated);
    showRoleDialog.value = false;
    toast.success(
      t("team.roleUpdated", { username: updated.username, role: roleLabel(updated.role) })
    );
  } catch (err: unknown) {
    // 400s (own role, last active admin) carry a descriptive backend message.
    toast.error(getErrorMessage(err) || t("errors.saveFailed"));
  } finally {
    isUpdatingRole.value = false;
  }
}

function openDeactivateDialog(member: TeamMember) {
  togglingMember.value = member;
  showDeactivateDialog.value = true;
}

function openReactivateDialog(member: TeamMember) {
  togglingMember.value = member;
  showReactivateDialog.value = true;
}

async function handleSetActive(activate: boolean) {
  if (!togglingMember.value) return;

  isTogglingActive.value = true;
  try {
    const updated = activate
      ? await teamApi.reactivateMember(togglingMember.value.id)
      : await teamApi.deactivateMember(togglingMember.value.id);
    patchMember(updated);
    if (activate) {
      showReactivateDialog.value = false;
    } else {
      showDeactivateDialog.value = false;
    }
    toast.success(
      t(activate ? "team.reactivateSuccess" : "team.deactivateSuccess", {
        username: updated.username,
      })
    );
  } catch (err: unknown) {
    toast.error(getErrorMessage(err) || t("errors.saveFailed"));
  } finally {
    isTogglingActive.value = false;
  }
}

async function openModelsDialog(member: TeamMember) {
  editingMember.value = member;
  memberModels.value = member.allowed_models ? [...member.allowed_models] : null;
  await modelStore.fetchModels();
  showModelsDialog.value = true;
}

function openBudgetDialog(member: TeamMember) {
  budgetMember.value = member;
  budgetUsd.value = member.budget_usd;
  budgetPeriod.value = member.budget_period ?? "none";
  budgetResetDay.value = member.budget_reset_day ?? 1;
  showBudgetDialog.value = true;
}

async function saveMemberBudget() {
  if (!budgetMember.value) return;
  const amount = normalizedBudgetUsd.value;
  if (amount !== null && amount <= 0) {
    toast.error(t("team.budgetMustBePositive"));
    return;
  }
  isSavingBudget.value = true;
  try {
    // The dialog edits the full effective state, so send all fields: an empty
    // amount clears the budget (and its window) server-side; otherwise the
    // period/reset day are written together with the cap.
    const payload: TeamMemberBudgetUpdate =
      amount === null
        ? { budget_usd: null }
        : {
            budget_usd: amount,
            budget_period: budgetPeriod.value === "none" ? null : budgetPeriod.value,
            // The 1st is the implicit default and need not be stored; a reset
            // day is only meaningful for monthly windows.
            budget_reset_day:
              budgetPeriod.value === "monthly" && budgetResetDay.value !== 1
                ? budgetResetDay.value
                : null,
          };
    const updated = await teamApi.updateMemberBudget(budgetMember.value.id, payload);
    patchMember(updated);
    showBudgetDialog.value = false;
    toast.success(t("team.budgetSaved", { username: updated.username }));
  } catch (err: unknown) {
    toast.error(getErrorMessage(err) || t("errors.saveFailed"));
  } finally {
    isSavingBudget.value = false;
  }
}

async function resetMemberBudgetWindow() {
  if (!budgetMember.value) return;
  isResettingBudget.value = true;
  try {
    const updated = await teamApi.resetMemberBudget(budgetMember.value.id);
    patchMember(updated);
    budgetMember.value = updated;
    toast.success(t("team.resetSpendSuccess", { username: updated.username }));
  } catch (err: unknown) {
    toast.error(getErrorMessage(err) || t("errors.saveFailed"));
  } finally {
    isResettingBudget.value = false;
  }
}

async function saveMemberModels() {
  if (!editingMember.value) return;
  isSavingModels.value = true;
  try {
    // null = unrestricted (allow all), [] = deny all, non-empty = restrict.
    const updated = await teamApi.updateMemberModels(editingMember.value.id, memberModels.value);
    const idx = members.value.findIndex((m) => m.id === editingMember.value!.id);
    if (idx !== -1) {
      members.value[idx] = {
        ...members.value[idx],
        allowed_models: updated.allowed_models,
      };
    }
    showModelsDialog.value = false;
    toast.success(t("team.modelsUpdated"));
  } catch (err: unknown) {
    toast.error(getErrorMessage(err) || t("errors.saveFailed"));
  } finally {
    isSavingModels.value = false;
  }
}

onMounted(loadMembers);
</script>

<template>
  <AppLayout layoutMode="full">
    <template #header>
      <header class="config-header-bar px-4 sm:px-6 py-4">
        <PageHeader :title="t('team.title')" :description="t('team.description')" :icon="Users">
          <template #actions>
            <Button @click="openCreateDialog" class="btn-action">
              <Plus class="h-4 w-4 mr-2" />
              {{ t("team.addMember") }}
            </Button>
          </template>
        </PageHeader>
      </header>
    </template>

    <!-- Toolbar band (flush) -->
    <div v-if="members.length > 0 || isLoading" class="config-toolbar px-4 sm:px-6 py-3">
      <FilterBar
        v-model:search-query="searchQuery"
        :search-placeholder="t('common.searchPlaceholder')"
        @clear-filters="clearFilters"
      />
    </div>

    <!-- Content area -->
    <div class="config-content">
      <!-- Loading state -->
      <div
        v-if="isLoading && members.length === 0"
        class="h-full flex items-center justify-center animate-fade-in px-6"
      >
        <LoadingState />
      </div>

      <!-- Error state -->
      <div
        v-else-if="loadError && members.length === 0"
        class="h-full flex items-center justify-center animate-fade-in px-6"
      >
        <EmptyState :text="loadError" :show-retry="true" @retry="loadMembers" />
      </div>

      <!-- Empty state (no members at all) -->
      <div
        v-else-if="members.length === 0"
        class="h-full flex items-center justify-center animate-fade-in px-6"
      >
        <EmptyState
          :icon="Users"
          :text="t('team.noMembers')"
          :show-cta="true"
          :cta-text="t('team.addMember')"
          @click="openCreateDialog"
        />
      </div>

      <!-- Members table -->
      <template v-else>
        <Table
          class="table-modern"
          container-class="h-full border-0 bg-transparent rounded-none overflow-x-auto"
        >
          <TableHeader class="config-thead">
            <TableRow class="bg-transparent hover:bg-transparent hover:border-l-transparent">
              <SortableHead
                :label="t('team.username')"
                sort-key="username"
                :active-field="sortField"
                :active-dir="sortDir"
                @sort="onSort"
              />
              <SortableHead
                :label="t('team.role')"
                sort-key="role"
                :active-field="sortField"
                :active-dir="sortDir"
                @sort="onSort"
              />
              <SortableHead
                :label="t('team.status')"
                sort-key="status"
                :active-field="sortField"
                :active-dir="sortDir"
                @sort="onSort"
              />
              <TableHead class="hidden lg:table-cell">{{ t("team.allowedModels") }}</TableHead>
              <TableHead class="hidden xl:table-cell">{{ t("team.budget") }}</TableHead>
              <SortableHead
                class="hidden md:table-cell"
                :label="t('apiKeys.created')"
                sort-key="created"
                :active-field="sortField"
                :active-dir="sortDir"
                @sort="onSort"
              />
              <TableHead class="w-24 text-right"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody class="row-stagger">
            <TableRow v-for="member in sortedMembers" :key="member.id" class="group">
              <TableCellName>{{ member.username }}</TableCellName>
              <TableCell>
                <!-- Role change: dropdown on the badge (hidden on the current
                     admin's own row), confirmed by dialog below. -->
                <DropdownMenu v-if="!isSelf(member)">
                  <DropdownMenuTrigger as-child>
                    <button
                      type="button"
                      class="cursor-pointer rounded-full transition-colors hover:opacity-80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                      :title="t('team.changeRole')"
                      :aria-label="t('team.changeRole')"
                    >
                      <Badge :variant="member.role === 'admin' ? 'default' : 'secondary'">
                        {{ roleLabel(member.role) }}
                      </Badge>
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="start">
                    <DropdownMenuItem
                      v-for="option in ROLE_OPTIONS"
                      :key="option"
                      :disabled="member.role === option"
                      class="cursor-pointer"
                      @click="openRoleDialog(member, option)"
                    >
                      <Check
                        class="mr-2 h-4 w-4"
                        :class="member.role === option ? 'opacity-100' : 'opacity-0'"
                      />
                      <span>{{ roleLabel(option) }}</span>
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
                <Badge v-else :variant="member.role === 'admin' ? 'default' : 'secondary'">
                  {{ roleLabel(member.role) }}
                </Badge>
              </TableCell>
              <TableCell>
                <div class="flex flex-wrap items-center gap-1.5">
                  <Badge :variant="member.is_active ? 'default' : 'secondary'">
                    <Check v-if="member.is_active" class="mr-1 h-3 w-3" />
                    {{ member.is_active ? t("team.active") : t("team.inactive") }}
                  </Badge>
                  <Badge
                    v-if="member.must_change_password"
                    variant="outline"
                    class="border-status-warning/60 text-status-warning font-normal"
                    :title="t('team.memberMustSetPassword')"
                  >
                    {{ t("team.passwordChangePending") }}
                  </Badge>
                </div>
              </TableCell>
              <TableCell class="hidden lg:table-cell">
                <!-- Clicking the scope summary opens the manage dialog directly -->
                <component
                  :is="member.role !== 'admin' ? 'button' : 'div'"
                  :type="member.role !== 'admin' ? 'button' : undefined"
                  :title="member.role !== 'admin' ? t('team.manageModels') : undefined"
                  class="-m-1 flex flex-wrap items-center gap-1.5 rounded-md p-1 text-left transition-colors"
                  :class="{ 'cursor-pointer hover:bg-accent/60': member.role !== 'admin' }"
                  @click="member.role !== 'admin' && openModelsDialog(member)"
                >
                  <Badge
                    v-if="member.allowed_models == null"
                    variant="secondary"
                    class="font-normal"
                  >
                    {{ t("team.allModels") }}
                  </Badge>
                  <Badge
                    v-else-if="member.allowed_models.length === 0"
                    variant="outline"
                    class="border-status-warning/40 text-status-warning font-normal"
                  >
                    {{ t("team.noModels") }}
                  </Badge>
                  <template v-else>
                    <Badge
                      v-for="model in member.allowed_models.slice(0, 3)"
                      :key="model"
                      variant="outline"
                      class="font-mono text-[11px]"
                    >
                      {{ model }}
                    </Badge>
                    <Badge
                      v-if="member.allowed_models.length > 3"
                      variant="outline"
                      class="font-normal text-muted-foreground"
                    >
                      +{{ member.allowed_models.length - 3 }}
                    </Badge>
                  </template>
                </component>
              </TableCell>
              <!-- Account budget: click-through to the manage-budget sheet -->
              <TableCell class="hidden xl:table-cell">
                <button
                  type="button"
                  :title="t('team.manageBudget')"
                  :aria-label="t('team.manageBudget')"
                  class="-m-1 rounded-md p-1 text-left transition-colors cursor-pointer hover:bg-accent/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  @click="openBudgetDialog(member)"
                >
                  <MemberBudgetCell :member="member" />
                </button>
              </TableCell>
              <TableCell class="hidden md:table-cell text-data text-muted-foreground">
                {{ formatDate(member.created_at) }}
              </TableCell>
              <TableCellActions>
                <div
                  class="flex items-center justify-end gap-1 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 sm:group-focus-within:opacity-100 transition-opacity"
                >
                  <Button
                    v-if="member.role !== 'admin'"
                    variant="ghost"
                    size="icon"
                    class="h-9 w-9"
                    :title="t('team.manageModels')"
                    :aria-label="t('team.manageModels')"
                    @click="openModelsDialog(member)"
                  >
                    <Boxes class="h-4 w-4 icon-btn-muted" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    class="h-9 w-9"
                    :title="t('team.manageBudget')"
                    :aria-label="t('team.manageBudget')"
                    @click="openBudgetDialog(member)"
                  >
                    <Wallet class="h-4 w-4 icon-btn-muted" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    class="h-9 w-9"
                    :title="t('team.renameUsername')"
                    :aria-label="t('team.renameUsername')"
                    @click="openRenameDialog(member)"
                  >
                    <UserPen class="h-4 w-4 icon-btn-muted" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    class="h-9 w-9"
                    :title="t('team.resetPassword')"
                    :aria-label="t('team.resetPassword')"
                    @click="openResetDialog(member)"
                  >
                    <KeyRound class="h-4 w-4 icon-btn-muted" />
                  </Button>
                  <Button
                    v-if="member.is_active"
                    variant="ghost"
                    size="icon"
                    class="h-9 w-9"
                    :disabled="isSelf(member)"
                    :title="t('team.deactivateMember')"
                    :aria-label="t('team.deactivateMember')"
                    @click="openDeactivateDialog(member)"
                  >
                    <UserX class="h-4 w-4 icon-btn-muted" />
                  </Button>
                  <Button
                    v-else
                    variant="ghost"
                    size="icon"
                    class="h-9 w-9"
                    :title="t('team.reactivateMember')"
                    :aria-label="t('team.reactivateMember')"
                    @click="openReactivateDialog(member)"
                  >
                    <UserCheck class="h-4 w-4 icon-btn-muted" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    class="h-9 w-9 text-destructive hover:text-destructive hover:bg-destructive/10"
                    :title="t('team.deleteMember')"
                    :aria-label="t('team.deleteMember')"
                    @click="openDeleteDialog(member)"
                  >
                    <Trash2 class="h-4 w-4" />
                  </Button>
                </div>
              </TableCellActions>
            </TableRow>
            <EmptyTableRow v-if="sortedMembers.length === 0" :colspan="7" @clear="clearFilters" />
          </TableBody>
        </Table>
      </template>
    </div>

    <!-- Create Member Sheet (slides in from the right) -->
    <Sheet v-model:open="showCreateDialog">
      <SheetContent
        side="right"
        class="w-full sm:max-w-[440px] h-full flex flex-col p-0 gap-0 overflow-hidden border-l border-border/80 bg-card"
      >
        <SheetHeaderBand :icon="Users">
          <template #title>{{ t("team.createMember") }}</template>
          <template #description>{{ t("team.description") }}</template>
        </SheetHeaderBand>
        <div class="flex-1 overflow-y-auto px-4 sm:px-6 py-4 space-y-4">
          <div class="space-y-2">
            <Label for="create-username">{{ t("team.username") }}</Label>
            <Input
              id="create-username"
              v-model="createUsername"
              :placeholder="t('auth.username')"
              :disabled="isCreating"
              @keydown.enter="handleCreate"
            />
          </div>
          <div class="space-y-2">
            <Label for="create-password">{{ t("team.password") }}</Label>
            <Input
              id="create-password"
              v-model="createPassword"
              type="password"
              :disabled="isCreating"
              @keydown.enter="handleCreate"
            />
            <p v-if="createError" class="text-destructive text-xs">{{ createError }}</p>
            <p v-else-if="createPassword" class="text-xs text-muted-foreground">
              {{ passwordRequirementsText() }}
            </p>
          </div>
          <div class="space-y-2">
            <Label for="create-confirm-password">{{ t("team.confirmPassword") }}</Label>
            <Input
              id="create-confirm-password"
              v-model="createConfirmPassword"
              type="password"
              :disabled="isCreating"
              @keydown.enter="handleCreate"
            />
          </div>
        </div>
        <div
          class="flex items-center justify-end gap-2 px-4 sm:px-6 py-4 border-t border-border/60 bg-muted/10 shrink-0"
        >
          <Button variant="outline" :disabled="isCreating" @click="showCreateDialog = false">
            {{ t("common.cancel") }}
          </Button>
          <Button :disabled="isCreating" @click="handleCreate">
            <Loader2 v-if="isCreating" class="h-4 w-4 animate-spin" />
            {{ t("team.addMember") }}
          </Button>
        </div>
      </SheetContent>
    </Sheet>

    <!-- Rename Username Sheet (slides in from the right) -->
    <Sheet v-model:open="showRenameDialog">
      <SheetContent
        side="right"
        class="w-full sm:max-w-[440px] h-full flex flex-col p-0 gap-0 overflow-hidden border-l border-border/80 bg-card"
      >
        <SheetHeaderBand :icon="UserPen">
          <template #title>{{ t("team.renameUsername") }}</template>
          <template #description>
            {{ t("team.renameUsernameDescription") }} —
            <span class="font-medium">{{ renamingMember?.username }}</span>
          </template>
        </SheetHeaderBand>
        <div class="flex-1 overflow-y-auto px-4 sm:px-6 py-4 space-y-4">
          <div class="space-y-2">
            <Label for="rename-username">{{ t("team.newUsername") }}</Label>
            <Input
              id="rename-username"
              v-model="renameUsername"
              :placeholder="renamingMember?.username ?? ''"
              :disabled="isRenaming"
              @keydown.enter="handleRename"
            />
            <p v-if="renameError" class="text-destructive text-xs">{{ renameError }}</p>
            <p v-else class="text-xs text-muted-foreground">
              {{ t("team.usernameFormatHint") }}
            </p>
          </div>
        </div>
        <div
          class="flex items-center justify-end gap-2 px-4 sm:px-6 py-4 border-t border-border/60 bg-muted/10 shrink-0"
        >
          <Button variant="outline" :disabled="isRenaming" @click="showRenameDialog = false">
            {{ t("common.cancel") }}
          </Button>
          <Button :disabled="isRenaming" @click="handleRename">
            <Loader2 v-if="isRenaming" class="h-4 w-4 animate-spin" />
            {{ t("team.renameUsername") }}
          </Button>
        </div>
      </SheetContent>
    </Sheet>

    <!-- Reset Password Sheet (slides in from the right) -->
    <Sheet v-model:open="showResetDialog">
      <SheetContent
        side="right"
        class="w-full sm:max-w-[440px] h-full flex flex-col p-0 gap-0 overflow-hidden border-l border-border/80 bg-card"
      >
        <SheetHeaderBand :icon="KeyRound">
          <template #title>{{ t("team.resetPassword") }}</template>
          <template #description>
            {{ t("team.resetPassword") }} —
            <span class="font-medium">{{ resettingMember?.username }}</span>
          </template>
        </SheetHeaderBand>
        <div class="flex-1 overflow-y-auto px-4 sm:px-6 py-4 space-y-4">
          <div class="space-y-2">
            <Label for="reset-password">{{ t("team.newPassword") }}</Label>
            <Input
              id="reset-password"
              v-model="resetPassword"
              type="password"
              :disabled="isResetting"
              @keydown.enter="handleReset"
            />
            <p v-if="resetError" class="text-destructive text-xs">{{ resetError }}</p>
            <p v-else-if="resetPassword" class="text-xs text-muted-foreground">
              {{ passwordRequirementsText() }}
            </p>
          </div>
          <div class="space-y-2">
            <Label for="reset-confirm-password">{{ t("team.confirmPassword") }}</Label>
            <Input
              id="reset-confirm-password"
              v-model="resetConfirmPassword"
              type="password"
              :disabled="isResetting"
              @keydown.enter="handleReset"
            />
          </div>
        </div>
        <div
          class="flex items-center justify-end gap-2 px-4 sm:px-6 py-4 border-t border-border/60 bg-muted/10 shrink-0"
        >
          <Button variant="outline" :disabled="isResetting" @click="showResetDialog = false">
            {{ t("common.cancel") }}
          </Button>
          <Button :disabled="isResetting" @click="handleReset">
            <Loader2 v-if="isResetting" class="h-4 w-4 animate-spin" />
            {{ t("team.resetPassword") }}
          </Button>
        </div>
      </SheetContent>
    </Sheet>

    <!-- Manage Allowed Models Sheet (slides in from the right) -->
    <Sheet v-model:open="showModelsDialog">
      <SheetContent
        side="right"
        class="w-full sm:max-w-[480px] h-full flex flex-col p-0 gap-0 overflow-hidden border-l border-border/80 bg-card"
      >
        <SheetHeaderBand :icon="Boxes">
          <template #title>{{ t("team.manageModels") }}</template>
          <template #description>
            {{ t("team.manageModelsDescription") }} —
            <span class="font-medium">{{ editingMember?.username }}</span>
          </template>
        </SheetHeaderBand>
        <div class="flex-1 overflow-y-auto px-4 sm:px-6 py-4 space-y-4">
          <AccessScopePicker
            v-model="memberModels"
            :items="availableModels"
            :all-label="t('team.allModels')"
            :all-help-text="t('team.allowAllModelsHelp')"
            :search-placeholder="t('team.searchModels')"
            :empty-available-text="t('team.noModelsAvailable')"
            :empty-search-text="t('team.noModelsFound')"
            :empty-text="t('team.denyAllWarning')"
            empty-meaning="deny"
            mono
          />
        </div>
        <div
          class="flex items-center justify-end gap-2 px-4 sm:px-6 py-4 border-t border-border/60 bg-muted/10 shrink-0"
        >
          <Button variant="outline" :disabled="isSavingModels" @click="showModelsDialog = false">
            {{ t("common.cancel") }}
          </Button>
          <Button :disabled="isSavingModels" @click="saveMemberModels">
            <Loader2 v-if="isSavingModels" class="h-4 w-4 animate-spin" />
            {{ t("common.save") }}
          </Button>
        </div>
      </SheetContent>
    </Sheet>

    <!-- Manage Account Budget Sheet (slides in from the right) -->
    <Sheet v-model:open="showBudgetDialog">
      <SheetContent
        side="right"
        class="w-full sm:max-w-[440px] h-full flex flex-col p-0 gap-0 overflow-hidden border-l border-border/80 bg-card"
      >
        <SheetHeaderBand :icon="Wallet">
          <template #title>{{ t("team.manageBudget") }}</template>
          <template #description>
            {{ t("team.manageBudgetDescription") }} —
            <span class="font-medium">{{ budgetMember?.username }}</span>
          </template>
        </SheetHeaderBand>
        <div class="flex-1 overflow-y-auto px-4 sm:px-6 py-4 space-y-4">
          <div class="space-y-2">
            <Label for="member-budget-usd">{{ t("apiKeys.budget") }}</Label>
            <div class="flex items-center gap-2">
              <div class="relative flex-1">
                <span
                  class="absolute left-3 top-1/2 -translate-y-1/2 text-sm text-muted-foreground pointer-events-none"
                  aria-hidden="true"
                  >$</span
                >
                <Input
                  id="member-budget-usd"
                  v-model.number="budgetUsd"
                  type="number"
                  min="0"
                  step="any"
                  :placeholder="t('apiKeys.budgetPlaceholder')"
                  class="pl-7"
                  :disabled="isSavingBudget"
                />
              </div>
              <Select v-model="budgetPeriod" :disabled="normalizedBudgetUsd === null">
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
              v-if="budgetPeriod === 'monthly' && normalizedBudgetUsd !== null"
              class="flex items-center gap-2 text-xs text-muted-foreground"
            >
              <span>{{ t("apiKeys.resetDayPrefix") }}</span>
              <Select v-model="budgetResetDay">
                <SelectTrigger class="w-18 h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent class="max-h-56">
                  <SelectItem v-for="day in 31" :key="day" :value="day">{{ day }}</SelectItem>
                </SelectContent>
              </Select>
              <span>{{ t("apiKeys.resetDaySuffix") }}</span>
            </div>
            <p class="text-xs text-muted-foreground">
              {{ budgetPeriod === "none" ? t("team.budgetHelpNone") : t("team.budgetHelp") }}
            </p>
            <p class="text-xs text-muted-foreground">{{ t("team.budgetEmptyHint") }}</p>
          </div>

          <!-- Current window spend + manual reset (only when a budget exists) -->
          <div
            v-if="budgetMember?.budget_usd != null"
            class="flex items-center justify-between rounded-lg border border-border/60 bg-muted/30 px-3 py-2"
          >
            <span class="text-xs text-muted-foreground">
              {{ t("team.currentWindowSpend") }}:
              <span class="font-medium text-foreground">
                {{ formatCost(budgetMember.budget_spend_usd) }}
              </span>
              — {{ t("team.resetSpendHint") }}
            </span>
            <Button
              variant="outline"
              size="sm"
              :disabled="isSavingBudget || isResettingBudget"
              @click="resetMemberBudgetWindow"
            >
              <Loader2 v-if="isResettingBudget" class="h-3.5 w-3.5 mr-1.5 animate-spin" />
              <RotateCcw v-else class="w-3.5 h-3.5 mr-1.5" />
              {{ t("team.resetSpend") }}
            </Button>
          </div>
        </div>
        <div
          class="flex items-center justify-end gap-2 px-4 sm:px-6 py-4 border-t border-border/60 bg-muted/10 shrink-0"
        >
          <Button variant="outline" :disabled="isSavingBudget" @click="showBudgetDialog = false">
            {{ t("common.cancel") }}
          </Button>
          <Button :disabled="isSavingBudget" @click="saveMemberBudget">
            <Loader2 v-if="isSavingBudget" class="h-4 w-4 animate-spin" />
            {{ t("common.save") }}
          </Button>
        </div>
      </SheetContent>
    </Sheet>

    <!-- Delete Confirmation Dialog -->
    <ConfirmDialog
      v-model:open="showDeleteDialog"
      :title="t('team.deleteMember')"
      :description="
        deletingMember
          ? t(deletingMember.role === 'admin' ? 'team.deleteAdminConfirm' : 'team.deleteConfirm', {
              username: deletingMember.username,
            })
          : ''
      "
      :confirm-text="t('team.deleteMember')"
      variant="destructive"
      :loading="isDeleting"
      @confirm="handleDelete"
    />

    <!-- Role Change Confirmation Dialog -->
    <ConfirmDialog
      v-model:open="showRoleDialog"
      :title="t('team.changeRole')"
      :description="
        roleTarget
          ? t('team.changeRoleConfirm', {
              username: roleTarget.username,
              role: roleLabel(roleNewValue),
            })
          : ''
      "
      :confirm-text="t('team.changeRole')"
      variant="default"
      :loading="isUpdatingRole"
      @confirm="handleRoleChange"
    />

    <!-- Deactivate Confirmation Dialog -->
    <ConfirmDialog
      v-model:open="showDeactivateDialog"
      :title="t('team.deactivateMember')"
      :description="
        togglingMember ? t('team.deactivateConfirm', { username: togglingMember.username }) : ''
      "
      :confirm-text="t('team.deactivateMember')"
      variant="destructive"
      :loading="isTogglingActive"
      @confirm="handleSetActive(false)"
    />

    <!-- Reactivate Confirmation Dialog -->
    <ConfirmDialog
      v-model:open="showReactivateDialog"
      :title="t('team.reactivateMember')"
      :description="
        togglingMember ? t('team.reactivateConfirm', { username: togglingMember.username }) : ''
      "
      :confirm-text="t('team.reactivateMember')"
      variant="default"
      :loading="isTogglingActive"
      @confirm="handleSetActive(true)"
    />
  </AppLayout>
</template>
