import { createRouter, createWebHistory } from "vue-router";
import { useAuthStore } from "@/stores/auth";
import { IDLE_PREFETCH_ROUTES, prefetchRouteChunk } from "./prefetch";

// Augment RouteMeta for type-safe meta fields
declare module "vue-router" {
  interface RouteMeta {
    requiresAuth?: boolean;
    adminOnly?: boolean;
    /** i18n key for the browser tab title. */
    titleKey?: string;
  }
}

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/login",
      name: "login",
      component: () => import("@/views/LoginView.vue"),
      meta: { titleKey: "auth.loginTitle" },
    },
    {
      path: "/setup",
      name: "setup",
      component: () => import("@/views/SetupView.vue"),
      meta: { titleKey: "auth.setupTitle" },
    },
    {
      path: "/invite/:token",
      name: "invite",
      component: () => import("@/views/InviteAcceptView.vue"),
      meta: { titleKey: "auth.inviteTitle" },
    },
    {
      path: "/oauth/complete",
      name: "oauthComplete",
      component: () => import("@/views/OAuthCompleteView.vue"),
      meta: { titleKey: "auth.oauthTitle" },
    },
    {
      path: "/verify-email",
      name: "verifyEmail",
      component: () => import("@/views/VerifyEmailView.vue"),
      meta: { titleKey: "accountSecurity.emailVerificationTitle" },
    },
    {
      // Where the emailed reset link lands. The token in the query string is
      // the credential, so the route must stay reachable while signed out.
      path: "/reset-password",
      name: "resetPassword",
      component: () => import("@/views/ResetPasswordView.vue"),
      meta: { titleKey: "auth.resetTitle" },
    },
    {
      path: "/forgot-password",
      name: "forgotPassword",
      component: () => import("@/views/ForgotPasswordView.vue"),
      meta: { titleKey: "auth.forgotTitle" },
    },
    {
      // Forced password change: reachable only while authenticated with the
      // `mustChangePassword` flag set; the guard funnels all other
      // authenticated navigation here until the password is replaced.
      path: "/force-change-password",
      name: "forceChangePassword",
      component: () => import("@/views/ForceChangePasswordView.vue"),
      meta: { requiresAuth: true, titleKey: "auth.forcedPasswordTitle" },
    },
    {
      path: "/",
      name: "home",
      component: () => import("@/views/HomeView.vue"),
      meta: { requiresAuth: true, titleKey: "nav.usage" },
    },
    {
      path: "/chat",
      name: "chat",
      component: () => import("@/views/ChatView.vue"),
      meta: { requiresAuth: true, titleKey: "nav.chat" },
    },
    {
      path: "/images",
      name: "images",
      component: () => import("@/views/ImagesView.vue"),
      meta: { requiresAuth: true, titleKey: "nav.images" },
    },
    {
      path: "/logs",
      name: "logs",
      component: () => import("@/views/LogsView.vue"),
      meta: { requiresAuth: true, titleKey: "nav.logs" },
    },
    {
      // Single models entry point: admins get the management UI, other
      // authenticated users get the read-only browse view (see ModelsView).
      path: "/models",
      name: "models",
      component: () => import("@/views/ModelsView.vue"),
      meta: { requiresAuth: true, titleKey: "nav.models" },
    },
    {
      path: "/config/providers",
      name: "providers",
      component: () => import("@/views/config/ProvidersView.vue"),
      meta: { requiresAuth: true, adminOnly: true, titleKey: "nav.providers" },
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
      meta: { requiresAuth: true, adminOnly: true, titleKey: "nav.settings" },
    },
    {
      path: "/config/console-auth",
      name: "consoleAuth",
      component: () => import("@/views/config/ConsoleAuthView.vue"),
      meta: { requiresAuth: true, adminOnly: true, titleKey: "nav.consoleAuth" },
    },

    {
      path: "/config/api-keys",
      name: "apiKeys",
      component: () => import("@/views/config/ApiKeysView.vue"),
      meta: { requiresAuth: true, titleKey: "nav.apiKeys" },
    },
    {
      path: "/config/mcp-servers",
      name: "mcpServers",
      component: () => import("@/views/config/McpServersView.vue"),
      meta: { requiresAuth: true, adminOnly: true, titleKey: "nav.mcpServers" },
    },
    {
      path: "/team",
      name: "team",
      component: () => import("@/views/TeamView.vue"),
      meta: { requiresAuth: true, adminOnly: true, titleKey: "nav.team" },
    },
    {
      path: "/account",
      name: "account",
      component: () => import("@/views/AccountView.vue"),
      meta: { requiresAuth: true, titleKey: "nav.account" },
    },
    {
      path: "/account/security",
      name: "security",
      component: () => import("@/views/SecurityView.vue"),
      meta: { requiresAuth: true, titleKey: "nav.security" },
    },
    {
      // Personal preferences (theme, language, per-user tracing) are an
      // account tab: they are nobody else's configuration.
      path: "/account/preferences",
      name: "preferences",
      component: () => import("@/views/PreferencesView.vue"),
      meta: { requiresAuth: true, titleKey: "nav.preferences" },
    },
    {
      path: "/:pathMatch(.*)*",
      name: "notFound",
      component: () => import("@/views/NotFoundView.vue"),
      meta: { titleKey: "notFound.title" },
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

router.afterEach((to) => {
  const authStore = useAuthStore();

  // Filter out the page we just navigated to, and admin-only pages for non-admins.
  const routesToPrefetch = IDLE_PREFETCH_ROUTES.filter(
    (route) => route.name !== to.name && (!route.adminOnly || authStore.isAdmin)
  );

  // Stagger the idle work. Each route's own prefetch chains its lazy children.
  routesToPrefetch.forEach((route, index) => {
    schedulePrefetch(() => prefetchRouteChunk(route.path), 2000 + index * 1000);
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
