<script setup lang="ts">
import { ref } from "vue";
import { useI18n } from "vue-i18n";
import { Button } from "@/components/ui/button";
import type { McpServerCapabilities, McpServerRead, McpServerStatus } from "@/types/schemas";
import McpCapabilitiesDialog from "./McpCapabilitiesDialog.vue";
import McpServerListItem from "./McpServerListItem.vue";

interface Props {
  servers: McpServerRead[];
  statuses: Record<string, McpServerStatus>;
  loadingServers?: Set<string>;
  serverCapabilities?: Record<string, McpServerCapabilities>;
  capabilitiesFailed?: Record<string, boolean>;
}

const props = defineProps<Props>();

const emit = defineEmits<{
  edit: [server: McpServerRead];
  delete: [name: string];
  toggle: [server: McpServerRead, enabled: boolean];
  clearFilters: [];
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
const onClear = () => emit("clearFilters");
</script>

<template>
  <div v-if="servers.length === 0" class="px-4 sm:px-6 py-12 text-center">
    <p class="mb-4 text-sm text-muted-foreground">{{ t("common.noMatchingResults") }}</p>
    <Button variant="outline" size="sm" @click="onClear">
      {{ t("common.clearFilters") }}
    </Button>
  </div>
  <div v-else class="config-list list-stagger">
    <McpServerListItem
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
  </div>

  <McpCapabilitiesDialog
    v-model:open="showCapabilitiesDialog"
    :server-name="selectedServerName"
    :capabilities="selectedCapabilities"
    :initial-tab="initialTab"
    :is-loading="selectedCapabilitiesLoading"
    :capabilities-failed="capabilitiesFailed?.[selectedServerName] ?? false"
  />
</template>
