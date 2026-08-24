import { defineStore } from "pinia";
import { shallowRef, ref, computed } from "vue";
import { configApi } from "@/services/api/config";
import type {
  McpServerCapabilities,
  McpServerCreate,
  McpServerRead,
  McpServerStatus,
  McpServerUpdate,
} from "@/types/schemas";

export const useMcpServerStore = defineStore("mcpServers", () => {
  const mcpServers = shallowRef<McpServerRead[]>([]);
  const loading = ref(false);
  const loaded = ref(false);
  const error = ref<string | null>(null);
  const mcpServerCapabilities = ref<Record<string, McpServerCapabilities>>({});
  const capabilitiesFailed = ref<Record<string, boolean>>({});

  const ready = computed(() => loaded.value);

  async function fetchMcpServers(force = false): Promise<McpServerRead[]> {
    if (!force && loaded.value && mcpServers.value.length > 0) {
      return mcpServers.value;
    }
    if (loading.value) return mcpServers.value;
    loading.value = true;
    try {
      const res = await configApi.getMcpServers();
      mcpServers.value = res;
      loaded.value = true;
      error.value = null;
      return res;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to fetch MCP servers";
      error.value = errorMsg;
      console.error("Failed to fetch MCP servers:", err);
      throw err;
    } finally {
      loading.value = false;
    }
  }

  function prefetch(): void {
    if (!loaded.value) {
      fetchMcpServers().catch((err) => {
        const errorMsg = err instanceof Error ? err.message : "MCP server prefetch failed";
        error.value = errorMsg;
      });
    }
  }

  async function createMcpServer(data: McpServerCreate): Promise<McpServerRead> {
    error.value = null;
    try {
      const res = await configApi.createMcpServer(data);
      await fetchMcpServers(true);
      return res;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to create MCP server";
      error.value = errorMsg;
      console.error("Failed to create MCP server:", err);
      throw err;
    }
  }

  async function updateMcpServer(name: string, data: McpServerUpdate): Promise<McpServerRead> {
    error.value = null;
    try {
      const res = await configApi.updateMcpServer(name, data);
      await fetchMcpServers(true);
      return res;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to update MCP server";
      error.value = errorMsg;
      console.error("Failed to update MCP server:", err);
      throw err;
    }
  }

  async function deleteMcpServer(name: string): Promise<void> {
    error.value = null;
    try {
      await configApi.deleteMcpServer(name);
      removeCapabilities(name);
      await fetchMcpServers(true);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to delete MCP server";
      error.value = errorMsg;
      console.error("Failed to delete MCP server:", err);
      throw err;
    }
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
    mcpServers.value = [];
    loading.value = false;
    loaded.value = false;
    mcpServerCapabilities.value = {};
    capabilitiesFailed.value = {};
  }

  return {
    mcpServers,
    loading,
    loaded,
    ready,
    mcpServerCapabilities,
    capabilitiesFailed,
    fetchMcpServers,
    prefetch,
    createMcpServer,
    updateMcpServer,
    deleteMcpServer,
    getMcpServerStatus,
    getMcpServerCapabilities,
    setCapabilities,
    setCapabilitiesFailed,
    removeCapabilities,
    reset,
  };
});
