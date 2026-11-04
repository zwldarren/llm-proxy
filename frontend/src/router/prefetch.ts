import { useAuthStore } from "@/stores/auth";

type Importer = () => Promise<unknown>;

/**
 * Warm a route's view first, then the lazy children it mounts. The children
 * only start once the parent resolves, so navigating after the prefetch does
 * not leave the first paint waiting on a second sequential request.
 */
async function loadWithChildren(view: Importer, children: readonly Importer[] = []): Promise<void> {
  await view();
  await Promise.all(children.map((load) => load()));
}

/**
 * Every path with a warm-up recipe. The registries below are typed against this
 * union, so a renamed route path fails the build instead of silently
 * no-opping at runtime.
 */
export const ROUTE_PATHS = [
  "/",
  "/chat",
  "/images",
  "/logs",
  "/models",
  "/config/providers",
  "/config/api-keys",
  "/config/mcp-servers",
  "/config/settings",
  "/config/console-auth",
  "/team",
  "/account",
  "/account/preferences",
  "/account/security",
] as const;

type PrefetchPath = (typeof ROUTE_PATHS)[number];

interface IdlePrefetchRoute {
  name: string;
  path: PrefetchPath;
  adminOnly: boolean;
}

/** Routes worth warming when the browser is idle, in navigation order. */
export const IDLE_PREFETCH_ROUTES = [
  { name: "home", path: "/", adminOnly: false },
  { name: "chat", path: "/chat", adminOnly: false },
  { name: "images", path: "/images", adminOnly: false },
  { name: "logs", path: "/logs", adminOnly: false },
  { name: "models", path: "/models", adminOnly: false },
  { name: "providers", path: "/config/providers", adminOnly: true },
  { name: "apiKeys", path: "/config/api-keys", adminOnly: false },
  { name: "mcpServers", path: "/config/mcp-servers", adminOnly: true },
  { name: "settings", path: "/config/settings", adminOnly: true },
  { name: "team", path: "/team", adminOnly: true },
] as const satisfies readonly IdlePrefetchRoute[];

type ModelsRole = "admin" | "member";

const modelsRole = (): ModelsRole => (useAuthStore().isAdmin ? "admin" : "member");

/**
 * `/models` resolves to one of two views by role. Both the chunk prefetch and
 * the data warm-up read this table, so the role split is stated once.
 */
const MODELS_BY_ROLE: Record<ModelsRole, { chunk: Importer; warm: () => Promise<void> }> = {
  admin: {
    chunk: () => import("@/views/config/ModelsView.vue"),
    warm: async () => {
      const { useModelStore } = await import("@/stores/models");
      useModelStore().prefetch();
    },
  },
  member: {
    chunk: () => import("@/views/ModelPlazaView.vue"),
    warm: async () => {
      const { useCatalogStore } = await import("@/stores/catalog");
      useCatalogStore().prefetch();
    },
  },
};

const ROUTE_IMPORTERS: Record<PrefetchPath, Importer> = {
  // HomeView imports the usage panels statically, so the view chunk already
  // carries them; prefetching the children separately would be a no-op.
  "/": () => import("@/views/HomeView.vue"),
  "/logs": () =>
    loadWithChildren(
      () => import("@/views/LogsView.vue"),
      [
        () => import("@/components/common/LogDetailsSheet.vue"),
        () => import("@/components/common/LogsFilters.vue"),
        () => import("@/components/common/AuditIntegrityDialog.vue"),
      ]
    ),
  "/models": async () => {
    await import("@/views/ModelsView.vue");
    await MODELS_BY_ROLE[modelsRole()].chunk();
  },
  "/config/providers": () => import("@/views/config/ProvidersView.vue"),
  "/config/api-keys": () => import("@/views/config/ApiKeysView.vue"),
  "/config/mcp-servers": () => import("@/views/config/McpServersView.vue"),
  "/config/settings": () => import("@/views/config/SettingsView.vue"),
  "/config/console-auth": () => import("@/views/config/ConsoleAuthView.vue"),
  "/chat": () => import("@/views/ChatView.vue"),
  "/images": () => import("@/views/ImagesView.vue"),
  "/account": () => import("@/views/AccountView.vue"),
  "/account/preferences": () => import("@/views/PreferencesView.vue"),
  "/account/security": () => import("@/views/SecurityView.vue"),
  "/team": () => import("@/views/TeamView.vue"),
};

const ROUTE_DATA_WARMERS: Partial<Record<PrefetchPath, () => Promise<void>>> = {
  "/config/providers": async () => {
    const { useProviderStore } = await import("@/stores/providers");
    useProviderStore().prefetch();
  },
  "/config/api-keys": async () => {
    const { useApiKeyStore } = await import("@/stores/apiKeys");
    useApiKeyStore().prefetch();
  },
  "/config/mcp-servers": async () => {
    const { useMcpServerStore } = await import("@/stores/mcpServers");
    useMcpServerStore().prefetch();
  },
  "/models": () => MODELS_BY_ROLE[modelsRole()].warm(),
};

const inFlight = new Map<string, Promise<unknown>>();

/** Prefetch only the route's view chunk chain, without touching its data API. */
export function prefetchRouteChunk(href: string): Promise<unknown> {
  const importer = ROUTE_IMPORTERS[href as PrefetchPath];
  if (!importer) return Promise.resolve();

  const existing = inFlight.get(href);
  if (existing) return existing;

  const pending = importer()
    .catch(() => {
      // Prefetching is best-effort; navigation will retry the import.
    })
    .finally(() => inFlight.delete(href));
  inFlight.set(href, pending);
  return pending;
}

/** Warm both the route's view chain and the data it renders on first paint. */
export function prefetchRoute(href: string): void {
  // Keep store modules off the startup graph until the operator signals intent.
  void ROUTE_DATA_WARMERS[href as PrefetchPath]?.().catch(() => {});
  void prefetchRouteChunk(href);
}
