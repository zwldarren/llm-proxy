<script setup lang="ts">
import type { Component, ComputedRef } from "vue";
import { computed, nextTick, ref, watch } from "vue";
import {
  BarChart3,
  ChevronLeft,
  ChevronRight,
  Cpu,
  Database,
  ImageIcon,
  Key,
  KeyRound,
  LogOut,
  MessageSquare,
  ScrollText,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  UserRound,
  Users,
  X,
  ChevronsUpDown,
} from "@lucide/vue";
import { useI18n } from "vue-i18n";
import { useRoute, useRouter } from "vue-router";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarMenuSubItem,
  SidebarMenuSubButton,
  SidebarSeparator,
  useSidebar,
} from "@/components/ui/sidebar";
import McpIcon from "@/components/common/McpIcon.vue";
import LogoIcon from "@/components/common/LogoIcon.vue";
import { scrollToAnchor } from "@/utils/scroll";
import { useStorage } from "@vueuse/core";
import { STORAGE_KEYS } from "@/constants/storageKeys";
import { useAuthStore } from "@/stores/auth";
import { useSystemStore } from "@/stores/system";
import { useProviderStore } from "@/stores/providers";
import { useModelStore } from "@/stores/models";
import { useCatalogStore } from "@/stores/catalog";
import { useApiKeyStore } from "@/stores/apiKeys";
import { useMcpServerStore } from "@/stores/mcpServers";

const route = useRoute();
const router = useRouter();
const { t } = useI18n();
const authStore = useAuthStore();
const providerStore = useProviderStore();
const modelStore = useModelStore();
const catalogStore = useCatalogStore();
const apiKeyStore = useApiKeyStore();
const mcpServerStore = useMcpServerStore();
const { state, isMobile, openMobile, toggleSidebar } = useSidebar();
const systemStore = useSystemStore();
const navigationEl = ref<HTMLElement | null>(null);

const userInitials = computed(() => {
  if (!authStore.username) return "U";
  return authStore.username.slice(0, 2).toUpperCase();
});

const isCollapsed: ComputedRef<boolean> = computed(
  () => state.value === "collapsed" && !isMobile.value
);

const toggleLabel = computed(() => {
  if (isMobile.value) return t("nav.closeMenu");
  return isCollapsed.value ? t("nav.expandSidebar") : t("nav.collapseSidebar");
});

const ToggleIcon = computed(() => {
  if (isMobile.value) return X;
  return isCollapsed.value ? ChevronRight : ChevronLeft;
});

type NavItem = {
  name: string;
  href?: string;
  icon?: Component;
  subItems?: Array<{
    name: string;
    href: string;
  }>;
};

type NavSection = {
  title: string;
  items: NavItem[];
};

// Role-aware placement for the merged /models page: viewers browse it as a
// catalog, while admins find model management where every other admin tool
// lives — under Configuration. Same route either way; the view switches by
// role inside the route component.
const memberSections: ComputedRef<NavSection[]> = computed(() => {
  const sections: NavSection[] = [
    {
      title: "nav.overview",
      items: [
        { name: "nav.usage", href: "/", icon: BarChart3 },
        { name: "nav.logs", href: "/logs", icon: ScrollText },
      ],
    },
  ];
  if (!authStore.isAdmin) {
    sections.push({
      title: "nav.catalog",
      items: [{ name: "nav.models", href: "/models", icon: Cpu }],
    });
  }
  sections.push({
    title: "nav.playground",
    items: [
      { name: "nav.chat", href: "/chat", icon: MessageSquare },
      { name: "nav.images", href: "/images", icon: ImageIcon },
    ],
  });
  return sections;
});

const expandedGroups = useStorage<Record<string, boolean>>(STORAGE_KEYS.SIDEBAR_EXPANDED_GROUPS, {
  "nav.general": true,
  "nav.advanced": true,
});

// Access section — who may call the proxy: every member manages their own keys;
// member management and the instance's sign-in policy are admin work.
const accessSections: ComputedRef<NavSection[]> = computed(() => {
  const items: NavItem[] = [{ name: "nav.apiKeys", href: "/config/api-keys", icon: Key }];

  if (authStore.isAdmin) {
    items.push(
      { name: "team.title", href: "/team", icon: Users },
      { name: "nav.consoleAuth", href: "/config/console-auth", icon: KeyRound }
    );
  }

  return [{ title: "nav.access", items }];
});

// Configuration section — what the instance routes to, plus its settings page.
// The whole section is admin-only: the settings page is instance-wide policy,
// and the personal preferences that used to live there moved to the account
// workspace.
const configSections: ComputedRef<NavSection[]> = computed(() => {
  if (!authStore.isAdmin) return [];

  return [
    {
      title: "nav.config",
      items: [
        { name: "nav.providers", href: "/config/providers", icon: Database },
        // /models renders the management UI for admins (see memberSections).
        { name: "nav.models", href: "/models", icon: Cpu },
        { name: "nav.mcpServers", href: "/config/mcp-servers", icon: McpIcon },
        // Settings is instance-wide, so it sits with the other admin resources
        // instead of repeating its own name as a one-item section label.
        { name: "nav.settings", href: "/config/settings", icon: Settings },
      ],
    },
  ];
});

// Sections visible based on user role
const visibleSections = computed(() => {
  return [...memberSections.value, ...accessSections.value, ...configSections.value];
});

// Ensure any dynamic/new collapsible groups default to true
watch(
  visibleSections,
  (sections) => {
    sections.forEach((section) => {
      section.items.forEach((item) => {
        if (item.subItems && expandedGroups.value[item.name] === undefined) {
          expandedGroups.value[item.name] = true;
        }
      });
    });
  },
  { immediate: true }
);

// Long role-aware menus can exceed a phone's viewport. Reveal the current
// route when the drawer opens instead of leaving it clipped below the footer.
watch([isMobile, openMobile], async ([mobile, drawerOpen]) => {
  if (!mobile || !drawerOpen) return;
  await nextTick();
  navigationEl.value
    ?.querySelector<HTMLElement>('[data-active="true"]')
    ?.scrollIntoView({ block: "nearest" });
});

function isCurrent(href: string): boolean {
  if (href === "/") return route.path === "/";
  // Settings is a single sidebar item; the in-page tabs track their own
  // active state. Any /config/settings route highlights the sidebar item.
  if (href.startsWith("/config/settings")) {
    return route.path === "/config/settings";
  }
  return route.path.startsWith(href.split("?")[0]);
}

function closeMobileSidebar() {
  if (isMobile.value && openMobile.value) toggleSidebar();
}

function navigateTo(href: string) {
  closeMobileSidebar();
  void router.push(href);
}

const handleLogout = async () => {
  closeMobileSidebar();
  await authStore.logout();
  await router.push("/login");
};

const PREFETCH_ROUTES: Record<string, () => void> = {
  "/config/providers": () => {
    providerStore.prefetch();
    import("@/views/config/ProvidersView.vue");
  },
  "/config/api-keys": () => {
    apiKeyStore.prefetch();
    import("@/views/config/ApiKeysView.vue");
  },
  "/config/mcp-servers": () => {
    mcpServerStore.prefetch();
    import("@/views/config/McpServersView.vue");
  },
  "/logs": () => import("@/views/LogsView.vue"),
  "/models": () => {
    // The route lazy-loads a role-appropriate inner view, so warm that chunk
    // too — warming only the dispatcher leaves first paint waiting on a
    // second fetch. The admin store warms only for admins (the config API
    // rejects viewers).
    import("@/views/ModelsView.vue");
    if (authStore.isAdmin) {
      modelStore.prefetch();
      import("@/views/config/ModelsView.vue");
    } else {
      catalogStore.prefetch();
      import("@/views/ModelPlazaView.vue");
    }
  },
  "/chat": () => import("@/views/ChatView.vue"),
  "/images": () => import("@/views/ImagesView.vue"),
  "/": () => import("@/views/HomeView.vue"),
  "/config/settings": () => import("@/views/config/SettingsView.vue"),
  "/config/console-auth": () => import("@/views/config/ConsoleAuthView.vue"),
  "/account": () => import("@/views/AccountView.vue"),
  "/account/preferences": () => import("@/views/PreferencesView.vue"),
  "/account/security": () => import("@/views/SecurityView.vue"),
  "/team": () => import("@/views/TeamView.vue"),
};

const prefetchRoute = (href: string) => {
  PREFETCH_ROUTES[href]?.();
};

/** The badge is a RouterLink to the About card, but a click that does not change
 * the URL (already on Settings, same hash) never fires the view's hash watcher.
 * Scroll directly in that case so the jump always happens. The helper retries
 * while the target is still display:none (e.g. the Advanced tab is active). */
const handleVersionBadgeClick = () => {
  closeMobileSidebar();
  if (route.name !== "settings") return;
  scrollToAnchor("about", { highlight: true });
};
</script>

<template>
  <Sidebar collapsible="icon" class="border-r border-sidebar-border">
    <!-- Brand header -->
    <SidebarHeader class="h-16 border-b border-sidebar-border justify-center relative">
      <div
        class="flex h-full items-center w-full transition-opacity duration-200"
        :class="isCollapsed ? 'justify-center px-0' : 'justify-between pl-2.5 pr-7'"
      >
        <div class="flex items-center gap-2 min-w-0">
          <RouterLink
            to="/"
            class="flex min-w-0 items-center gap-2 rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring"
            :aria-label="t('home.title')"
            :title="isCollapsed ? t('home.title') : undefined"
            @click="closeMobileSidebar"
            @mouseenter="prefetchRoute('/')"
          >
            <span
              class="flex size-8 shrink-0 items-center justify-center rounded-md border border-sidebar-border/60 bg-sidebar-accent/45 text-sidebar-foreground"
            >
              <LogoIcon class="size-5" />
            </span>
            <span
              v-if="!isCollapsed"
              class="brand-heading text-lg font-semibold tracking-tight shrink-0 animate-in fade-in duration-200"
            >
              {{ t("home.title") }}
            </span>
          </RouterLink>
          <!-- Instance version badge — links to the About section in Settings.
               Renders nothing while the info is loading or when the fetch failed. -->
          <RouterLink
            v-if="!isCollapsed && systemStore.info"
            to="/config/settings#about"
            :title="t('about.title')"
            @click="handleVersionBadgeClick"
            class="flex min-w-0 max-w-36 shrink items-center gap-1 overflow-hidden rounded-md border border-sidebar-border/60 px-1.5 py-0.5 font-mono text-[10px] leading-none text-sidebar-foreground/70 transition-colors duration-200 hover:border-sidebar-border hover:text-sidebar-foreground animate-in fade-in duration-200"
          >
            <span class="truncate" :title="`v${systemStore.info.version}`">
              v{{ systemStore.info.version }}
            </span>
            <template v-if="systemStore.updateAvailable">
              <span role="status" class="flex items-center gap-1">
                <span
                  class="size-1.5 shrink-0 rounded-full bg-status-warning"
                  aria-hidden="true"
                ></span>
                <span class="sr-only">
                  {{ t("about.updateAvailable", { version: systemStore.info.latest_version }) }}
                </span>
              </span>
            </template>
          </RouterLink>
        </div>

        <!-- Mobile close button -->
        <Button
          v-if="isMobile"
          variant="ghost"
          size="icon"
          class="shrink-0"
          :aria-label="t('nav.closeMenu')"
          :title="t('nav.closeMenu')"
          @click="toggleSidebar"
        >
          <X class="size-5" />
        </Button>
      </div>

      <!-- Floating Expand/Collapse Toggle Button for Desktop -->
      <button
        v-if="!isMobile"
        type="button"
        class="absolute top-5 right-0 z-50 hidden md:flex h-6 w-6 translate-x-1/2 items-center justify-center rounded-full border border-sidebar-border bg-background text-sidebar-foreground shadow-xs hover:bg-sidebar-accent hover:text-sidebar-accent-foreground focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-sidebar-ring transition-transform duration-200 cursor-pointer [@media(hover:none)]:h-11 [@media(hover:none)]:w-11"
        :aria-label="toggleLabel"
        :title="toggleLabel"
        @click="toggleSidebar"
      >
        <component :is="ToggleIcon" class="size-3.5" />
      </button>
    </SidebarHeader>

    <!-- Navigation -->
    <SidebarContent class="py-4">
      <!-- Navigation sections -->
      <nav ref="navigationEl" :aria-label="t('nav.mainNavigation')" class="flex flex-col gap-2">
        <SidebarGroup v-for="section in visibleSections" :key="section.title">
          <SidebarGroupLabel
            class="text-[11px] font-semibold tracking-[0.14em] uppercase text-sidebar-foreground/70"
          >
            {{ t(section.title) }}
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu class="gap-0.5">
              <SidebarMenuItem v-for="item in section.items" :key="item.name">
                <template v-if="!item.subItems">
                  <SidebarMenuButton
                    as-child
                    :is-active="isCurrent(item.href!)"
                    :tooltip="t(item.name)"
                    class="max-md:h-11"
                  >
                    <RouterLink
                      :to="item.href!"
                      :aria-current="isCurrent(item.href!) ? 'page' : undefined"
                      @click="closeMobileSidebar"
                      @mouseenter="prefetchRoute(item.href!)"
                    >
                      <component :is="item.icon" />
                      <span>{{ t(item.name) }}</span>
                    </RouterLink>
                  </SidebarMenuButton>
                </template>
                <template v-else>
                  <Collapsible v-model:open="expandedGroups[item.name]" class="group/collapsible">
                    <CollapsibleTrigger as-child>
                      <SidebarMenuButton class="max-md:h-11" :tooltip="t(item.name)">
                        <component :is="item.icon" v-if="item.icon" />
                        <span>{{ t(item.name) }}</span>
                        <ChevronRight
                          class="ml-auto transition-transform group-data-[state=open]/collapsible:rotate-90"
                        />
                      </SidebarMenuButton>
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                      <SidebarMenuSub>
                        <SidebarMenuSubItem v-for="sub in item.subItems" :key="sub.name">
                          <SidebarMenuSubButton
                            as-child
                            :is-active="isCurrent(sub.href)"
                            class="max-md:h-11"
                          >
                            <RouterLink
                              :to="sub.href"
                              :aria-current="isCurrent(sub.href) ? 'page' : undefined"
                              @click="closeMobileSidebar"
                              @mouseenter="prefetchRoute(sub.href)"
                            >
                              <span>{{ t(sub.name) }}</span>
                            </RouterLink>
                          </SidebarMenuSubButton>
                        </SidebarMenuSubItem>
                      </SidebarMenuSub>
                    </CollapsibleContent>
                  </Collapsible>
                </template>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </nav>
    </SidebarContent>

    <!-- Footer: user info + profile + logout merged -->
    <SidebarFooter v-if="authStore.isAuthenticated" class="p-2">
      <SidebarSeparator class="mb-2" />
      <SidebarMenu>
        <SidebarMenuItem>
          <DropdownMenu>
            <DropdownMenuTrigger as-child>
              <SidebarMenuButton
                size="lg"
                :tooltip="t('nav.accountMenu')"
                :aria-label="t('nav.accountMenu')"
                class="w-full justify-start gap-2.5 px-2 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground transition-colors duration-200"
              >
                <div
                  class="flex size-8 items-center justify-center rounded-md bg-muted text-xs font-semibold uppercase text-muted-foreground border border-sidebar-border/30 shrink-0 select-none"
                >
                  {{ userInitials }}
                </div>
                <div
                  v-if="!isCollapsed"
                  class="flex flex-1 flex-col text-left leading-tight min-w-0"
                >
                  <span class="truncate text-xs font-semibold text-sidebar-foreground">{{
                    authStore.username
                  }}</span>
                  <span class="truncate text-[11px] text-sidebar-foreground/70 font-medium">
                    {{ authStore.isAdmin ? t("team.admin") : t("team.viewer") }}
                  </span>
                </div>
                <ChevronsUpDown
                  v-if="!isCollapsed"
                  class="ml-auto size-3.5 text-sidebar-foreground/40 shrink-0"
                />
              </SidebarMenuButton>
            </DropdownMenuTrigger>
            <DropdownMenuContent
              :side="isCollapsed ? 'right' : 'top'"
              align="end"
              :side-offset="8"
              class="w-56 animate-in slide-in-from-bottom-2 duration-150"
            >
              <DropdownMenuLabel class="p-0 font-normal">
                <div class="flex items-center gap-2.5 px-2 py-1.5 text-left text-xs">
                  <div
                    class="flex size-8 items-center justify-center rounded-md bg-muted text-xs font-semibold uppercase text-muted-foreground border border-sidebar-border/30 shrink-0 select-none"
                  >
                    {{ userInitials }}
                  </div>
                  <div class="flex flex-1 flex-col text-left leading-tight min-w-0">
                    <span class="truncate font-semibold text-foreground">{{
                      authStore.username
                    }}</span>
                    <span class="truncate text-[11px] text-muted-foreground">
                      {{ authStore.isAdmin ? t("team.admin") : t("team.viewer") }}
                    </span>
                  </div>
                </div>
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuGroup>
                <DropdownMenuItem @click="navigateTo('/account')" class="cursor-pointer">
                  <UserRound class="mr-2 size-4 text-muted-foreground" />
                  <span>{{ t("nav.account") }}</span>
                </DropdownMenuItem>
                <DropdownMenuItem
                  @click="navigateTo('/account/preferences')"
                  class="cursor-pointer"
                >
                  <SlidersHorizontal class="mr-2 size-4 text-muted-foreground" />
                  <span>{{ t("nav.preferences") }}</span>
                </DropdownMenuItem>
                <DropdownMenuItem @click="navigateTo('/account/security')" class="cursor-pointer">
                  <ShieldCheck class="mr-2 size-4 text-muted-foreground" />
                  <span>{{ t("nav.security") }}</span>
                </DropdownMenuItem>
              </DropdownMenuGroup>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                @click="handleLogout"
                class="cursor-pointer text-destructive focus:bg-destructive/10 focus:text-destructive transition-colors duration-150"
              >
                <LogOut class="mr-2 size-4" />
                <span>{{ t("auth.logout") }}</span>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </SidebarMenuItem>
      </SidebarMenu>
    </SidebarFooter>
  </Sidebar>
</template>
