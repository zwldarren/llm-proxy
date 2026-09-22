import { computed, toValue, type MaybeRefOrGetter } from "vue";
import { useI18n } from "vue-i18n";
import { useMcpServerMeta } from "@/composables/useMcpServerMeta";
import type { McpServerCapabilities, McpServerRead, McpServerStatus } from "@/types/schemas";

/**
 * Shared row model for the MCP server components (list item + table row).
 *
 * Both shapes render the same server — status/type presentation, the
 * proxy-URL copy affordance, the capability counts and the endpoint
 * fallbacks — so the derived state lives here once instead of being
 * copy-pasted into each template's script block.
 */
export function useMcpServerRow(
  server: MaybeRefOrGetter<McpServerRead>,
  status: MaybeRefOrGetter<McpServerStatus | undefined>,
  capabilities: MaybeRefOrGetter<McpServerCapabilities | undefined>
) {
  const { t } = useI18n();
  const meta = useMcpServerMeta(server, status, t);

  const capabilitiesFetched = computed(() => !!toValue(capabilities));
  const toolsCount = computed(() => toValue(capabilities)?.tools.length ?? 0);
  const promptsCount = computed(() => toValue(capabilities)?.prompts.length ?? 0);
  const resourcesCount = computed(() => toValue(capabilities)?.resources.length ?? 0);

  const showCapabilities = computed(
    () => toValue(status)?.status === "running" && capabilitiesFetched.value
  );

  /**
   * Secondary endpoint display: the launch command for stdio servers that have
   * no proxy URL to show.
   */
  const commandDisplay = computed(() => {
    if (meta.fullProxyUrl.value) return null;
    const srv = toValue(server);
    if (srv.type !== "stdio") return null;
    const cmd = srv.command || "";
    const args = Array.isArray(srv.args) ? srv.args.join(" ") : "";
    return [cmd, args].filter(Boolean).join(" ") || null;
  });

  return {
    ...meta,
    capabilitiesFetched,
    toolsCount,
    promptsCount,
    resourcesCount,
    showCapabilities,
    commandDisplay,
  };
}
