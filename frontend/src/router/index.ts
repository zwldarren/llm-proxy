import { createRouter, createWebHistory } from "vue-router";
import { useAuthStore } from "@/stores/auth";

// Augment RouteMeta for type-safe meta fields
declare module "vue-router" {
  interface RouteMeta {
    requiresAuth?: boolean;
    adminOnly?: boolean;
  }
}

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/login",
      name: "login",
      component: () => import("@/views/LoginView.vue"),
    },
    {
      path: "/setup",
      name: "setup",
      component: () => import("@/views/SetupView.vue"),
    },
    {
      // Forced password change: reachable only while authenticated with the
      // `mustChangePassword` flag set; the guard funnels all other
      // authenticated navigation here until the password is replaced.
      path: "/force-change-password",
      name: "forceChangePassword",
      component: () => import("@/views/ForceChangePasswordView.vue"),
      meta: { requiresAuth: true },
    },
    {
      path: "/",
      name: "home",
      component: () => import("@/views/HomeView.vue"),
      meta: { requiresAuth: true },
    },
    {
      path: "/chat",
      name: "chat",
      component: () => import("@/views/ChatView.vue"),
      meta: { requiresAuth: true },
    },
    {
      path: "/images",
      name: "images",
      component: () => import("@/views/ImagesView.vue"),
      meta: { requiresAuth: true },
    },
    {
      path: "/logs",
      name: "logs",
      component: () => import("@/views/LogsView.vue"),
      meta: { requiresAuth: true },
    },
    {
      path: "/models",
      name: "modelPlaza",
      component: () => import("@/views/ModelPlazaView.vue"),
      meta: { requiresAuth: true },
    },
    {
      path: "/config/providers",
      name: "providers",
      component: () => import("@/views/config/ProvidersView.vue"),
      meta: { requiresAuth: true, adminOnly: true },
    },
    {
      path: "/config/models",
      name: "models",
      component: () => import("@/views/config/ModelsView.vue"),
      meta: { requiresAuth: true, adminOnly: true },
    },
    {
      path: "/config/settings",
      name: "settings",
      component: () => import("@/views/config/SettingsView.vue"),
      meta: { requiresAuth: true },
    },

    {
      path: "/config/api-keys",
      name: "apiKeys",
      component: () => import("@/views/config/ApiKeysView.vue"),
      meta: { requiresAuth: true },
    },
    {
      path: "/config/mcp-servers",
      name: "mcpServers",
      component: () => import("@/views/config/McpServersView.vue"),
      meta: { requiresAuth: true, adminOnly: true },
    },
    {
      path: "/team",
      name: "team",
      component: () => import("@/views/TeamView.vue"),
      meta: { requiresAuth: true, adminOnly: true },
    },
  ],
});

export default router;

router.beforeEach((to, _from) => {
  const authStore = useAuthStore();

  // If setup status is still loading (null), allow navigation to proceed
  // without redirecting - avoids race conditions while setup status resolves
  if (authStore.needsSetup === null) {
    return;
  }

  // Force the first-run admin setup screen when no admin account exists yet.
  if (authStore.needsSetup === true && to.name !== "setup") {
    return { path: "/setup" };
  }
  // Once setup is complete, do not allow revisiting the setup screen.
  if (authStore.needsSetup === false && to.name === "setup") {
    return authStore.isAuthenticated ? "/" : "/login";
  }
  if (to.meta.requiresAuth && !authStore.isAuthenticated) {
    return {
      path: "/login",
      query: { redirect: to.fullPath },
    };
  }

  // Forced password change: while the flag is set, every authenticated area
  // redirects to the dedicated screen until the user sets a new password.
  if (authStore.isAuthenticated && authStore.mustChangePassword) {
    if (to.name !== "forceChangePassword" && to.meta.requiresAuth) {
      return { name: "forceChangePassword" };
    }
  } else if (to.name === "forceChangePassword") {
    // No pending forced change: the screen is not directly visitable.
    return { path: "/" };
  }

  if (to.meta.adminOnly && !authStore.isAdmin) {
    return { path: "/" };
  }
});

// Smart preloading based on current route - sequentially prefetch other main pages when idle
type ViewName =
  | "HomeView"
  | "ChatView"
  | "ImagesView"
  | "LogsView"
  | "ModelPlazaView"
  | "config/ProvidersView"
  | "config/ModelsView"
  | "config/ApiKeysView"
  | "config/McpServersView"
  | "config/SettingsView"
  | "TeamView";

const VIEW_IMPORTS: Record<ViewName, () => Promise<unknown>> = {
  HomeView: () => import("@/views/HomeView.vue"),
  ChatView: () => import("@/views/ChatView.vue"),
  ImagesView: () => import("@/views/ImagesView.vue"),
  LogsView: () => import("@/views/LogsView.vue"),
  ModelPlazaView: () => import("@/views/ModelPlazaView.vue"),
  "config/ProvidersView": () => import("@/views/config/ProvidersView.vue"),
  "config/ModelsView": () => import("@/views/config/ModelsView.vue"),
  "config/ApiKeysView": () => import("@/views/config/ApiKeysView.vue"),
  "config/McpServersView": () => import("@/views/config/McpServersView.vue"),
  "config/SettingsView": () => import("@/views/config/SettingsView.vue"),
  TeamView: () => import("@/views/TeamView.vue"),
} satisfies Record<ViewName, () => Promise<unknown>>;

router.afterEach((to) => {
  const authStore = useAuthStore();
  const allViews = [
    { name: "home", view: "HomeView" },
    { name: "chat", view: "ChatView" },
    { name: "images", view: "ImagesView" },
    { name: "logs", view: "LogsView" },
    { name: "modelPlaza", view: "ModelPlazaView" },
    { name: "providers", view: "config/ProvidersView" },
    { name: "models", view: "config/ModelsView" },
    { name: "apiKeys", view: "config/ApiKeysView" },
    { name: "mcpServers", view: "config/McpServersView" },
    { name: "settings", view: "config/SettingsView" },
    { name: "team", view: "TeamView" },
  ];

  // Define which routes are admin-only
  const adminRoutes = new Set(["providers", "models", "mcpServers", "team", "circuitBreaker"]);

  // Filter out the page we just navigated to, and admin-only pages for non-admins
  const viewsToPrefetch = allViews.filter(
    (v) => v.name !== to.name && (!adminRoutes.has(v.name as string) || authStore.isAdmin)
  );

  // Prefetch them sequentially with staggered timeouts when browser is idle
  viewsToPrefetch.forEach((v, index) => {
    const importer = VIEW_IMPORTS[v.view];
    if (importer) {
      requestIdleCallback(() => importer(), { timeout: 2000 + index * 1000 });
    }
  });
});
