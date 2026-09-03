<script setup lang="ts">
import { Plus, Zap } from "@lucide/vue";
import { computed, onMounted, ref } from "vue";
import { useI18n } from "vue-i18n";
import { toast } from "vue-sonner";
import ConfirmDialog from "@/components/common/ConfirmDialog.vue";
import EmptyState from "@/components/common/EmptyState.vue";
import FilterBar from "@/components/common/FilterBar.vue";
import TableSkeleton from "@/components/common/TableSkeleton.vue";
import PageHeader from "@/components/common/PageHeader.vue";
import ViewToggle from "@/components/common/ViewToggle.vue";
import McpServerList from "@/components/mcp/McpServerList.vue";
import McpServerTable from "@/components/mcp/McpServerTable.vue";
import { Button } from "@/components/ui/button";
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
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { useErrorHandler } from "@/composables/useErrorHandler";
import { useJsonField } from "@/composables/useJsonField";
import { useTableFilter } from "@/composables/useTableFilter";
import { useViewMode } from "@/composables/useViewMode";
import { useMcpServerStore } from "@/stores/mcpServers";
import { STORAGE_KEYS } from "@/constants/storageKeys";
import type {
  McpServerCreate,
  McpServerRead,
  McpServerStatus,
  McpServerUpdate,
} from "@/types/schemas";

const { t } = useI18n();
const { handleSaveError, handleDeleteError } = useErrorHandler();
const { parseJsonField, initJsonField } = useJsonField();
const mcpStore = useMcpServerStore();

const serverStatuses = ref<Record<string, McpServerStatus>>({});
const showCreateDialog = ref(false);

const servers = computed(() => mcpStore.mcpServers);
const isLoading = computed(() => mcpStore.loading && !mcpStore.ready);
const isSaving = ref(false);

const loadingServers = ref<Set<string>>(new Set());
const showDeleteDialog = ref(false);
const deletingServerName = ref("");
const isEditing = ref(false);
const editingServerName = ref("");

const serverTypes = computed(() => [
  { label: t("mcpServers.stdio"), value: "stdio" },
  { label: t("mcpServers.streamableHttp"), value: "streamableHttp" },
]);

const {
  searchQuery,
  typeFilter,
  filteredItems: filteredServers,
  clearFilters: clearBaseFilters,
} = useTableFilter(servers, {
  searchFields: ["name", "type", "proxy_url"],
  typeField: "type",
});

function clearFilters() {
  clearBaseFilters();
  sortField.value = "name";
  sortDir.value = "asc";
}

const viewMode = useViewMode(STORAGE_KEYS.MCP_SERVERS_VIEW_MODE);

type SortField = "name" | "type" | "status";
const sortField = ref<SortField>("name");
const sortDir = ref<"asc" | "desc">("asc");

function onSort(field: string) {
  if (field === sortField.value) {
    sortDir.value = sortDir.value === "asc" ? "desc" : "asc";
  } else {
    sortField.value = field as SortField;
    sortDir.value = "asc";
  }
}

const statusRank = (status: string): number =>
  status === "running" ? 0 : status === "stopped" ? 1 : 2;

const sortedServers = computed(() => {
  const items = [...(filteredServers.value as unknown as McpServerRead[])];
  const dir: 1 | -1 = sortDir.value === "asc" ? 1 : -1;
  switch (sortField.value) {
    case "status":
      items.sort((a, b) => {
        const ra = statusRank(serverStatuses.value[a.name]?.status || "stopped");
        const rb = statusRank(serverStatuses.value[b.name]?.status || "stopped");
        return (ra - rb || a.name.localeCompare(b.name)) * dir;
      });
      break;
    case "type":
      items.sort(
        (a, b) =>
          ((a.type ?? "").localeCompare(b.type ?? "") ||
            (a.name ?? "").localeCompare(b.name ?? "")) * dir
      );
      break;
    case "name":
    default:
      items.sort((a, b) => a.name.localeCompare(b.name) * dir);
      break;
  }
  return items;
});

const newServer = ref<McpServerCreate>({
  name: "",
  type: "stdio",
  command: "",
  args: [],
  base_url: "",
  env: {},
  enabled: true,
});

const argsList = ref<string[]>([]);
const envJson = ref("{}");

async function fetchServerCapabilities(serverName: string) {
  try {
    const status = await mcpStore.getMcpServerStatus(serverName);
    serverStatuses.value[serverName] = status;
    if (status.status === "running") {
      try {
        const capabilities = await mcpStore.getMcpServerCapabilities(serverName);
        mcpStore.setCapabilities(serverName, capabilities);
        mcpStore.setCapabilitiesFailed(serverName, false);
      } catch {
        mcpStore.setCapabilities(serverName, {
          tools: [],
          prompts: [],
          resources: [],
        });
        mcpStore.setCapabilitiesFailed(serverName, true);
      }
    } else {
      mcpStore.removeCapabilities(serverName);
    }
  } catch {
    serverStatuses.value[serverName] = {
      name: serverName,
      status: "stopped",
      proxy_url: null,
      error_message: null,
    };
    mcpStore.removeCapabilities(serverName);
  }
}

const fetchServers = async () => {
  await mcpStore.fetchMcpServers();
  for (const server of servers.value) {
    await fetchServerCapabilities(server.name);
  }
};

const openCreateDialog = () => {
  isEditing.value = false;
  editingServerName.value = "";
  newServer.value = {
    name: "",
    type: "stdio",
    command: "",
    args: [],
    base_url: "",
    env: {},
    enabled: true,
  };
  argsList.value = [];
  envJson.value = "{}";
  showCreateDialog.value = true;
};

const openEditDialog = (server: McpServerRead) => {
  isEditing.value = true;
  editingServerName.value = server.name;
  newServer.value = {
    name: server.name,
    type: server.type,
    command: server.command || "",
    args: server.args || [],
    base_url: server.base_url || "",
    env: server.env || {},
    enabled: server.enabled,
  };
  argsList.value = [...(server.args || [])];
  envJson.value = initJsonField(server.env || {});
  showCreateDialog.value = true;
};

const saveServer = async () => {
  // Client-side validation
  if (!newServer.value.name?.trim()) {
    toast.error(t("errors.validation.nameRequired"));
    return;
  }

  if (newServer.value.type === "stdio") {
    const command = newServer.value.command?.trim();
    if (!command) {
      toast.error(t("mcpServers.commandRequired"));
      return;
    }
  }

  if (newServer.value.type === "streamableHttp" && !newServer.value.base_url?.trim()) {
    toast.error(t("mcpServers.baseUrlRequired"));
    return;
  }

  isSaving.value = true;
  try {
    const serverData: McpServerCreate = { ...newServer.value };

    // Process args
    serverData.args = argsList.value.filter((arg) => arg.trim());

    // Process env vars
    const parsedEnv = parseJsonField(envJson.value, {
      errorTitle: t("mcpServers.invalidJsonError"),
      errorDescription: t("mcpServers.invalidJsonDescription"),
    });
    if (parsedEnv === undefined) {
      isSaving.value = false;
      return;
    }
    if (parsedEnv) {
      serverData.env = Object.fromEntries(
        Object.entries(parsedEnv).map(([k, v]) => [k, String(v)])
      ) as Record<string, string>;
    } else {
      serverData.env = {};
    }

    if (isEditing.value) {
      const updateData: McpServerUpdate = {
        type: serverData.type,
        command: serverData.command,
        args: serverData.args,
        base_url: serverData.base_url,
        env: serverData.env,
        enabled: serverData.enabled,
      };
      await mcpStore.updateMcpServer(editingServerName.value, updateData);
    } else {
      await mcpStore.createMcpServer(serverData);
    }

    showCreateDialog.value = false;
    toast.success(t("common.success"), {
      description: isEditing.value ? t("mcpServers.updateSuccess") : t("mcpServers.createSuccess"),
    });
  } catch (e) {
    handleSaveError(e);
  } finally {
    isSaving.value = false;
  }
};

const openDeleteDialog = (name: string) => {
  deletingServerName.value = name;
  showDeleteDialog.value = true;
};

const confirmDelete = async () => {
  const name = deletingServerName.value;
  showDeleteDialog.value = false;
  try {
    await mcpStore.deleteMcpServer(name);
    toast.success(t("common.success"), {
      description: t("mcpServers.deleteSuccess"),
    });
  } catch (e) {
    handleDeleteError(e);
  }
};

const toggleServerEnabled = async (server: McpServerRead, nextEnabled: boolean) => {
  // Set per-server loading state
  loadingServers.value.add(server.name);

  try {
    const updateData: McpServerUpdate = { enabled: nextEnabled };
    await mcpStore.updateMcpServer(server.name, updateData);

    // Fetch updated status in background without showing skeleton
    await fetchServerCapabilities(server.name);

    toast.success(t("common.success"), {
      description: nextEnabled ? t("mcpServers.enabledSuccess") : t("mcpServers.disabledSuccess"),
    });
  } catch (e) {
    handleSaveError(e);
  } finally {
    loadingServers.value.delete(server.name);
  }
};

onMounted(() => fetchServers());
</script>

<template>
  <AppLayout layoutMode="full">
    <template #header>
      <header class="config-header-bar px-4 sm:px-6 py-4">
        <PageHeader
          :title="t('mcpServers.title')"
          :description="t('mcpServers.description')"
          :icon="Zap"
        >
          <template #actions>
            <Button @click="openCreateDialog" class="btn-action">
              <Plus class="w-4 h-4 mr-2" />
              {{ t("mcpServers.addServer") }}
            </Button>
          </template>
        </PageHeader>
      </header>
    </template>

    <!-- Toolbar band (flush) -->
    <div v-if="servers.length > 0 || isLoading" class="config-toolbar px-4 sm:px-6 py-3">
      <FilterBar
        :search-placeholder="t('common.searchPlaceholder')"
        :type-filter-options="serverTypes"
        :type-filter-label="t('mcpServers.type')"
        @search="searchQuery = $event"
        @type-filter="typeFilter = $event"
        @clear-filters="clearFilters"
      >
        <ViewToggle v-model="viewMode" />
      </FilterBar>
    </div>

    <!-- Content area -->
    <div class="config-content">
      <div v-if="isLoading && servers.length === 0" class="h-full animate-fade-in">
        <TableSkeleton />
      </div>
      <div
        v-else-if="servers.length === 0"
        class="h-full flex items-center justify-center animate-fade-in px-6"
      >
        <EmptyState
          :text="t('mcpServers.noServers')"
          :show-cta="true"
          :cta-text="t('mcpServers.addServer')"
          @click="openCreateDialog"
        />
      </div>
      <template v-else>
        <!-- Table view (default) -->
        <McpServerTable
          v-if="viewMode === 'table'"
          :servers="sortedServers"
          :statuses="serverStatuses"
          :loading-servers="loadingServers"
          :server-capabilities="mcpStore.mcpServerCapabilities"
          :capabilities-failed="mcpStore.capabilitiesFailed"
          :sort-field="sortField"
          :sort-dir="sortDir"
          @edit="openEditDialog"
          @delete="openDeleteDialog"
          @toggle="toggleServerEnabled"
          @clear-filters="clearFilters"
          @sort="onSort"
        />

        <!-- List view -->
        <div v-else class="config-scroll">
          <McpServerList
            :servers="sortedServers"
            :statuses="serverStatuses"
            :loading-servers="loadingServers"
            :server-capabilities="mcpStore.mcpServerCapabilities"
            :capabilities-failed="mcpStore.capabilitiesFailed"
            @edit="openEditDialog"
            @delete="openDeleteDialog"
            @toggle="toggleServerEnabled"
            @clear-filters="clearFilters"
          />
        </div>
      </template>
    </div>

    <!-- Delete Confirmation Dialog -->
    <ConfirmDialog
      v-model:open="showDeleteDialog"
      :title="t('dialogs.confirmDeleteTitle')"
      :description="t('dialogs.confirmDelete', { name: deletingServerName })"
      :confirm-text="t('common.delete')"
      :cancel-text="t('common.cancel')"
      :loading="isSaving"
      @confirm="confirmDelete"
    />

    <!-- Create/Edit Sheet (slides in from the right) -->
    <Sheet v-model:open="showCreateDialog">
      <SheetContent
        side="right"
        class="w-full sm:max-w-[520px] h-full flex flex-col p-0 gap-0 overflow-hidden border-l border-border/80 bg-card"
      >
        <!-- Header band -->
        <SheetHeaderBand :icon="Zap">
          <template #title>
            {{ isEditing ? t("mcpServers.editServer") : t("mcpServers.addServer") }}
          </template>
          <template #description>
            {{ isEditing ? newServer.name : t("mcpServers.description") }}
          </template>
        </SheetHeaderBand>

        <div class="flex-1 overflow-y-auto px-4 sm:px-6 py-4 space-y-4">
          <div class="grid gap-2">
            <Label for="name"
              >{{ t("mcpServers.name") }} <span class="text-destructive">*</span></Label
            >
            <Input
              id="name"
              v-model="newServer.name"
              :placeholder="t('placeholders.serverName')"
              :disabled="isEditing"
            />
          </div>

          <div class="grid gap-2">
            <Label for="type"
              >{{ t("mcpServers.type") }} <span class="text-destructive">*</span></Label
            >
            <Select v-model="newServer.type">
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem
                  v-for="typeOption in serverTypes"
                  :key="typeOption.value"
                  :value="typeOption.value"
                >
                  {{ typeOption.label }}
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          <!-- stdio-specific fields -->
          <div v-if="newServer.type === 'stdio'" class="space-y-4">
            <div class="grid gap-2">
              <Label for="command"
                >{{ t("mcpServers.command") }} <span class="text-destructive">*</span></Label
              >
              <Input
                id="command"
                v-model="newServer.command"
                :placeholder="t('placeholders.command')"
              />
              <p class="text-[11px] text-muted-foreground">{{ t("mcpServers.commandHelp") }}</p>
            </div>

            <div class="grid gap-2">
              <Label>{{ t("mcpServers.args") }}</Label>
              <div class="space-y-2">
                <Input
                  v-for="(arg, index) in argsList"
                  :key="index"
                  v-model="argsList[index]"
                  :placeholder="`arg ${index + 1}`"
                  class="font-mono"
                />
                <Button
                  variant="outline"
                  size="sm"
                  class="w-full border-dashed text-xs"
                  @click="argsList.push('')"
                >
                  <Plus class="w-3 h-3 mr-2" />
                  {{ t("common.add") }}
                </Button>
              </div>
            </div>

            <div class="grid gap-2">
              <Label for="env">{{ t("mcpServers.env") }} (JSON)</Label>
              <Textarea
                id="env"
                v-model="envJson"
                placeholder='{"KEY": "value"}'
                class="code-textarea min-h-24"
              />
              <p class="text-[11px] text-muted-foreground">{{ t("mcpServers.envHelp") }}</p>
            </div>
          </div>

          <!-- streamableHttp-specific fields -->
          <div v-if="newServer.type === 'streamableHttp'" class="space-y-4">
            <div class="grid gap-2">
              <Label for="baseurl"
                >{{ t("mcpServers.baseUrl") }} <span class="text-destructive">*</span></Label
              >
              <Input
                id="baseurl"
                v-model="newServer.base_url"
                :placeholder="t('placeholders.baseUrl')"
              />
              <p class="text-[11px] text-muted-foreground">{{ t("mcpServers.baseUrlHelp") }}</p>
            </div>
          </div>

          <!-- Common fields -->
          <div class="flex items-center justify-between">
            <Label for="enabled">{{ t("mcpServers.enabled") }}</Label>
            <Switch id="enabled" v-model="newServer.enabled" />
          </div>
        </div>

        <div
          class="flex items-center justify-end gap-2 px-4 sm:px-6 py-4 border-t border-border/60 bg-muted/10 shrink-0"
        >
          <Button variant="outline" @click="showCreateDialog = false">{{
            t("common.cancel")
          }}</Button>
          <Button @click="saveServer" :disabled="isSaving">{{ t("common.save") }}</Button>
        </div>
      </SheetContent>
    </Sheet>
  </AppLayout>
</template>
