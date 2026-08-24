import { copyToClipboard } from "@/utils/clipboard";
import { Terminal, Webhook } from "@lucide/vue";
import {
  computed,
  onScopeDispose,
  ref,
  toValue,
  type ComputedRef,
  type MaybeRefOrGetter,
} from "vue";
import type { Component } from "vue";
import type { McpServerRead, McpServerStatus } from "@/types/schemas";

/**
 * Shared presentation metadata for an MCP server, used by both the
 * {@link McpServerListItem} and the {@link McpServerTable} so the status/type
 * visuals and the proxy-URL copy affordance stay consistent.
 */

interface McpStatusConfig {
  label: string;
  /** text color class for the status label */
  color: string;
  /** background class for the status dot */
  bg: string;
  /** ring class for the status dot */
  ring: string;
  /** whether the running dot should pulse */
  pulse: boolean;
}

interface McpTypeConfig {
  icon: Component;
  label: string;
  /** icon container background */
  bg: string;
  /** badge text color */
  text: string;
  /** badge border color */
  border: string;
  /** icon color */
  iconColor: string;
}

type TFn = (key: string, ...args: unknown[]) => string;

export function useMcpServerMeta(
  server: MaybeRefOrGetter<McpServerRead>,
  status: MaybeRefOrGetter<McpServerStatus | undefined>,
  t: TFn
) {
  const statusConfig: ComputedRef<McpStatusConfig> = computed(() => {
    const current = toValue(status)?.status || "stopped";
    switch (current) {
      case "running":
        return {
          label: t("mcpServers.statusRunning"),
          color: "text-status-success",
          bg: "bg-status-success",
          ring: "ring-status-success/20",
          pulse: true,
        };
      case "error":
        return {
          label: t("mcpServers.statusError"),
          color: "text-destructive",
          bg: "bg-destructive",
          ring: "ring-destructive/20",
          pulse: false,
        };
      default:
        return {
          label: t("mcpServers.statusStopped"),
          color: "text-muted-foreground",
          bg: "bg-muted-foreground",
          ring: "ring-muted-foreground/20",
          pulse: false,
        };
    }
  });

  const typeConfig: ComputedRef<McpTypeConfig> = computed(() => {
    const srv = toValue(server);
    switch (srv.type) {
      case "stdio":
        return {
          icon: Terminal,
          label: t("mcpServers.stdio"),
          bg: "bg-primary/10",
          text: "text-primary",
          border: "border-primary/25",
          iconColor: "text-primary",
        };
      case "streamableHttp":
        return {
          icon: Webhook,
          label: t("mcpServers.streamableHttp"),
          bg: "bg-action-violet/10",
          text: "text-action-violet",
          border: "border-action-violet/25",
          iconColor: "text-action-violet",
        };
      default:
        return {
          icon: Terminal,
          label: t("mcpServers.unknownType", { type: srv.type }),
          bg: "bg-muted",
          text: "text-muted-foreground",
          border: "border-border",
          iconColor: "text-muted-foreground",
        };
    }
  });

  const isEnabled = computed(() => toValue(server).enabled !== false);

  const fullProxyUrl = computed(() => {
    const srv = toValue(server);
    if (!srv.proxy_url) return null;
    return `${window.location.origin}${srv.proxy_url}`;
  });

  const copied = ref(false);
  let copyTimeout: ReturnType<typeof setTimeout> | null = null;

  onScopeDispose(() => {
    if (copyTimeout) clearTimeout(copyTimeout);
  });

  async function copyProxyUrl() {
    const url = fullProxyUrl.value;
    if (!url) return;
    try {
      await copyToClipboard(url);
    } catch {
      copied.value = false;
      return;
    }
    copied.value = true;
    copyTimeout = setTimeout(() => {
      copied.value = false;
      copyTimeout = null;
    }, 2000);
  }

  return {
    statusConfig,
    typeConfig,
    isEnabled,
    fullProxyUrl,
    copied,
    copyProxyUrl,
  };
}
