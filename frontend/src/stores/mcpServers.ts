import { defineStore } from "pinia";
import { ref } from "vue";
import { configApi } from "@/services/api/config";
import { createResourceStore } from "@/composables/useResourceStore";
import type {
  McpServerCapabilities,
  McpServerCreate,
  McpServerRead,
  McpServerStatus,
  McpServerUpdate,
} from "@/types/schemas";

export const useMcpServerStore = defineStore("mcpServers", () => {
  const store = createResourceStore<McpServerRead, McpServerCreate, McpServerUpdate>({
    name: "MCP server",
    fetchFn: () => configApi.getMcpServers(),
    createFn: (data) => configApi.createMcpServer(data),
    updateFn: (name, data) => configApi.updateMcpServer(name, data),
    deleteFn: (name) => configApi.deleteMcpServer(name),
  });

  // Per-server capability map, keyed by server name. The factory does not
  // cover this: capabilities are fetched lazily per server, not as a list.
  const mcpServerCapabilities = ref<Record<string, McpServerCapabilities>>({});
  const capabilitiesFailed = ref<Record<string, boolean>>({});

  async function deleteMcpServer(name: string): Promise<void> {
    await store.deleteItem(name);
    removeCapabilities(name);
  }

  async function getMcpServerStatus(name: string): Promise<McpServerStatus> {
    return configApi.getMcpServerStatus(name);
  }

  async function getMcpServerCapabilities(name: string): Promise<McpServerCapabilities> {
    return configApi.getMcpServerCapabilities(name);
  }

  function setCapabilities(name: string, capabilities: McpServerCapabilities): void {
    mcpServerCapabilities.value[name] = capabilities;
  }

  function setCapabilitiesFailed(name: string, failed: boolean): void {
    capabilitiesFailed.value[name] = failed;
  }

  function removeCapabilities(name: string): void {
    delete mcpServerCapabilities.value[name];
    delete capabilitiesFailed.value[name];
  }

  function reset(): void {
    store.reset();
    mcpServerCapabilities.value = {};
    capabilitiesFailed.value = {};
  }

  return {
    mcpServers: store.items,
    loading: store.loading,
    loaded: store.loaded,
    ready: store.ready,
    mcpServerCapabilities,
    capabilitiesFailed,
    fetchMcpServers: store.fetchItems,
    prefetch: store.prefetch,
    createMcpServer: store.createItem,
    updateMcpServer: store.updateItem,
    deleteMcpServer,
    getMcpServerStatus,
    getMcpServerCapabilities,
    setCapabilities,
    setCapabilitiesFailed,
    removeCapabilities,
    reset,
  };
});
