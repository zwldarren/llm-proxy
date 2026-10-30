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
      path: "/invite/:token",
      name: "invite",
      component: () => import("@/views/InviteAcceptView.vue"),
    },
    {
      path: "/oauth/complete",
      name: "oauthComplete",
      component: () => import("@/views/OAuthCompleteView.vue"),
    },
    {
      path: "/verify-email",
      name: "verifyEmail",
      component: () => import("@/views/VerifyEmailView.vue"),
    },
    {
      // Where the emailed reset link lands. The token in the query string is
      // the credential, so the route must stay reachable while signed out.
      path: "/reset-password",
      name: "resetPassword",
      component: () => import("@/views/ResetPasswordView.vue"),
    },
    {
      path: "/forgot-password",
      name: "forgotPassword",
      component: () => import("@/views/ForgotPasswordView.vue"),
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
      // Single models entry point: admins get the management UI, other
      // authenticated users get the read-only browse view (see ModelsView).
      path: "/models",
      name: "models",
      component: () => import("@/views/ModelsView.vue"),
      meta: { requiresAuth: true },
    },
    {
      path: "/config/providers",
      name: "providers",
      component: () => import("@/views/config/ProvidersView.vue"),
      meta: { requiresAuth: true, adminOnly: true },
    },
    {
      // Legacy admin URL kept as a redirect so old bookmarks keep working.
      path: "/config/models",
      redirect: { name: "models" },
    },
    {
      path: "/config/settings",
      name: "settings",
      component: () => import("@/views/config/SettingsView.vue"),
      // Instance-wide configuration: theme, language and personal tracing moved
      // to the account workspace, so nothing on this page is a member's.
      meta: { requiresAuth: true, adminOnly: true },
    },
    {
      path: "/config/console-auth",
      name: "consoleAuth",
      component: () => import("@/views/config/ConsoleAuthView.vue"),
      meta: { requiresAuth: true, adminOnly: true },
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
    {
      path: "/account",
      name: "account",
      component: () => import("@/views/AccountView.vue"),
      meta: { requiresAuth: true },
    },
    {
      path: "/account/security",
      name: "security",
      component: () => import("@/views/SecurityView.vue"),
      meta: { requiresAuth: true },
    },
    {
      // Personal preferences (theme, language, per-user tracing) are an
      // account tab: they are nobody else's configuration.
      path: "/account/preferences",
      name: "preferences",
      component: () => import("@/views/PreferencesView.vue"),
      meta: { requiresAuth: true },
    },
    {
      path: "/:pathMatch(.*)*",
      name: "notFound",
      component: () => import("@/views/NotFoundView.vue"),
    },
  ],
});

export default router;

router.beforeEach(async (to, _from) => {
  const authStore = useAuthStore();

  // Resolve the first-run setup status before deciding anything: the check is
  // memoized, so the initial navigation waits for the API round-trip while
  // later navigations resolve immediately.
  await authStore.ensureSetupStatus();

  // The status probe failed: the app cannot know whether the first-run setup
  // screen or the login form is correct. SetupView shows the retry state.
  if (authStore.setupStatusError && to.name !== "setup") {
    return { path: "/setup" };
  }
  // Force the first-run admin setup screen when no admin account exists yet.
  if (authStore.needsSetup === true && to.name !== "setup") {
    return { path: "/setup" };
  }
  // Resolve the HttpOnly cookie's identity before any auth decision. The probe
  // is memoized; a login/logout/401 replaces the memoized result.
  if (authStore.needsSetup === false) {
    await authStore.ensureSession();
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
  // (including /login and unknown paths) redirects to the dedicated screen
  // until the user sets a new password.
  if (authStore.isAuthenticated && authStore.mustChangePassword) {
    if (to.name !== "forceChangePassword") {
      return { name: "forceChangePassword" };
    }
  } else if (to.name === "forceChangePassword") {
    // No pending forced change: the screen is not directly visitable.
    return { path: "/" };
  }

  // /login is meaningless with a live session; send the operator back to the
  // console instead of showing a second sign-in form.
  if (to.name === "login" && authStore.isAuthenticated) {
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
  | "ModelsView"
  | "config/ProvidersView"
  | "config/ApiKeysView"
  | "config/McpServersView"
  | "config/SettingsView"
  | "TeamView";

const VIEW_IMPORTS: Record<ViewName, () => Promise<unknown>> = {
  HomeView: () => import("@/views/HomeView.vue"),
  ChatView: () => import("@/views/ChatView.vue"),
  ImagesView: () => import("@/views/ImagesView.vue"),
  LogsView: () => import("@/views/LogsView.vue"),
  ModelsView: async () => {
    // /models renders a role-switched dispatcher that lazy-loads its inner
    // view; warm the role-appropriate inner chunk too, or first paint waits
    // on a second sequential fetch.
    const authStore = useAuthStore();
    await Promise.all([
      import("@/views/ModelsView.vue"),
      authStore.isAdmin
        ? import("@/views/config/ModelsView.vue")
        : import("@/views/ModelPlazaView.vue"),
    ]);
  },
  "config/ProvidersView": () => import("@/views/config/ProvidersView.vue"),
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
    { name: "models", view: "ModelsView" },
    { name: "providers", view: "config/ProvidersView" },
    { name: "apiKeys", view: "config/ApiKeysView" },
    { name: "mcpServers", view: "config/McpServersView" },
    { name: "settings", view: "config/SettingsView" },
    { name: "team", view: "TeamView" },
  ];

  // Define which routes are admin-only
  const ADMIN_ROUTES: Record<string, true> = {
    providers: true,
    mcpServers: true,
    settings: true,
    team: true,
    circuitBreaker: true,
  };

  // Filter out the page we just navigated to, and admin-only pages for non-admins
  const viewsToPrefetch = allViews.filter(
    (v) =>
      v.name !== to.name && (!Object.hasOwn(ADMIN_ROUTES, v.name as string) || authStore.isAdmin)
  );

  // Prefetch them sequentially with staggered timeouts when browser is idle
  viewsToPrefetch.forEach((v, index) => {
    const importer = VIEW_IMPORTS[v.view];
    if (importer) {
      schedulePrefetch(importer, 2000 + index * 1000);
    }
  });
});

/**
 * Run a prefetch when the browser is idle, falling back to a plain timer where
 * `requestIdleCallback` is unavailable (Safari < 17, non-browser test envs).
 */
function schedulePrefetch(importer: () => Promise<unknown>, timeoutMs: number): void {
  if (typeof requestIdleCallback === "function") {
    requestIdleCallback(() => void importer(), { timeout: timeoutMs });
    return;
  }
  setTimeout(() => void importer(), timeoutMs);
}
