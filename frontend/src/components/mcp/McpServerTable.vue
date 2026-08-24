<script setup lang="ts">
import { ref } from "vue";
import { useI18n } from "vue-i18n";
import EmptyTableRow from "@/components/common/EmptyTableRow.vue";
import SortableHead from "@/components/common/SortableHead.vue";
import { Table, TableBody, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { McpServerCapabilities, McpServerRead, McpServerStatus } from "@/types/schemas";
import McpCapabilitiesDialog from "./McpCapabilitiesDialog.vue";
import McpServerTableRow from "./McpServerTableRow.vue";

interface Props {
  servers: McpServerRead[];
  statuses: Record<string, McpServerStatus>;
  loadingServers?: Set<string>;
  serverCapabilities?: Record<string, McpServerCapabilities>;
  capabilitiesFailed?: Record<string, boolean>;
  sortField?: string;
  sortDir?: "asc" | "desc";
}

const props = defineProps<Props>();

const emit = defineEmits<{
  edit: [server: McpServerRead];
  delete: [name: string];
  toggle: [server: McpServerRead, enabled: boolean];
  clearFilters: [];
  sort: [field: string];
}>();

const { t } = useI18n();

const isServerLoading = (name: string) => props.loadingServers?.has(name) ?? false;

// Single shared capabilities dialog driven by the selected row
const showCapabilitiesDialog = ref(false);
const selectedServerName = ref("");
const selectedCapabilities = ref<McpServerCapabilities | undefined>(undefined);
const selectedCapabilitiesLoading = ref(false);
const initialTab = ref<"tools" | "prompts" | "resources">("tools");

const openCapabilities = (serverName: string, tab: "tools" | "prompts" | "resources") => {
  selectedServerName.value = serverName;
  selectedCapabilities.value = props.serverCapabilities?.[serverName];
  selectedCapabilitiesLoading.value = !selectedCapabilities.value;
  initialTab.value = tab;
  showCapabilitiesDialog.value = true;
};

const handleEdit = (server: McpServerRead) => emit("edit", server);
const handleDelete = (name: string) => emit("delete", name);
const handleToggle = (server: McpServerRead, enabled: boolean) => emit("toggle", server, enabled);
const onSort = (field: string) => emit("sort", field);
const onClear = () => emit("clearFilters");
</script>

<template>
  <Table class="table-modern" container-class="h-full border-0 bg-transparent rounded-none">
    <TableHeader class="config-thead">
      <TableRow class="bg-transparent hover:bg-transparent hover:border-l-transparent">
        <TableHead class="w-12"></TableHead>
        <SortableHead
          :label="t('mcpServers.name')"
          sort-key="name"
          :active-field="sortField"
          :active-dir="sortDir"
          @sort="onSort"
        />
        <SortableHead
          :label="t('mcpServers.type')"
          sort-key="type"
          width-class="w-32"
          :active-field="sortField"
          :active-dir="sortDir"
          @sort="onSort"
        />
        <TableHead>{{ t("mcpServers.proxyUrl") }}</TableHead>
        <SortableHead
          :label="t('common.status')"
          sort-key="status"
          width-class="w-32"
          :active-field="sortField"
          :active-dir="sortDir"
          @sort="onSort"
        />
        <TableHead class="w-20 text-center">{{ t("mcpServers.enabled") }}</TableHead>
        <TableHead class="w-44">{{ t("mcpServers.capabilities") }}</TableHead>
        <TableHead class="w-24 text-right">{{ t("common.actions") }}</TableHead>
      </TableRow>
    </TableHeader>
    <TableBody class="row-stagger">
      <McpServerTableRow
        v-for="server in servers"
        :key="server.name"
        :server="server"
        :status="statuses[server.name]"
        :is-loading="isServerLoading(server.name)"
        :server-capabilities="serverCapabilities?.[server.name]"
        :capabilities-failed="capabilitiesFailed?.[server.name]"
        @edit="handleEdit(server)"
        @delete="handleDelete(server.name)"
        @toggle="(enabled) => handleToggle(server, enabled)"
        @open-capabilities="(tab) => openCapabilities(server.name, tab)"
      />
      <EmptyTableRow v-if="servers.length === 0" :colspan="8" @clear="onClear" />
    </TableBody>
  </Table>

  <McpCapabilitiesDialog
    v-model:open="showCapabilitiesDialog"
    :server-name="selectedServerName"
    :capabilities="selectedCapabilities"
    :initial-tab="initialTab"
    :is-loading="selectedCapabilitiesLoading"
    :capabilities-failed="capabilitiesFailed?.[selectedServerName] ?? false"
  />
</template>
