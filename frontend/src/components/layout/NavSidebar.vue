<script setup lang="ts">
import type { Component, ComputedRef } from "vue";
import { computed, ref, watch } from "vue";
import {
  BarChart3,
  Boxes,
  ChevronLeft,
  ChevronRight,
  Cpu,
  Database,
  ImageIcon,
  Key,
  KeyRound,
  Loader2,
  LogOut,
  MessageSquare,
  ScrollText,
  Settings,
  UserPen,
  Users,
  X,
  ChevronsUpDown,
} from "@lucide/vue";
import { useI18n } from "vue-i18n";
import { useRoute, useRouter } from "vue-router";
import { toast } from "vue-sonner";
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
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
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
  SidebarRail,
  SidebarSeparator,
  useSidebar,
} from "@/components/ui/sidebar";
import McpIcon from "@/components/common/McpIcon.vue";
import { useChangePassword } from "@/composables/useChangePassword";
import { meApi } from "@/services/api/me";
import { getErrorMessage } from "@/utils/error";
import { useStorage } from "@vueuse/core";
import { STORAGE_KEYS } from "@/constants/storageKeys";
import { useAuthStore } from "@/stores/auth";
import { useSystemStore } from "@/stores/system";
import { useProviderStore } from "@/stores/providers";
import { useModelStore } from "@/stores/models";
import { useApiKeyStore } from "@/stores/apiKeys";
import { useMcpServerStore } from "@/stores/mcpServers";

const route = useRoute();
const router = useRouter();
const { t } = useI18n();
const authStore = useAuthStore();
const providerStore = useProviderStore();
const modelStore = useModelStore();
const apiKeyStore = useApiKeyStore();
const mcpServerStore = useMcpServerStore();
const { state, isMobile, toggleSidebar } = useSidebar();
const systemStore = useSystemStore();

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

const memberSections: NavSection[] = [
  {
    title: "nav.overview",
    items: [
      { name: "nav.usage", href: "/", icon: BarChart3 },
      { name: "nav.logs", href: "/logs", icon: ScrollText },
    ],
  },
  {
    title: "nav.catalog",
    items: [{ name: "nav.modelPlaza", href: "/models", icon: Boxes }],
  },
  {
    title: "nav.playground",
    items: [
      { name: "nav.chat", href: "/chat", icon: MessageSquare },
      { name: "nav.images", href: "/images", icon: ImageIcon },
    ],
  },
];

const expandedGroups = useStorage<Record<string, boolean>>(STORAGE_KEYS.SIDEBAR_EXPANDED_GROUPS, {
  "nav.general": true,
  "nav.advanced": true,
});

// Configuration section — visible to all users, with role-based item filtering
const configSections: ComputedRef<NavSection[]> = computed(() => {
  const items: NavItem[] = [];

  // Items accessible to all authenticated users
  items.push({ name: "nav.apiKeys", href: "/config/api-keys", icon: Key });
  // Future: add more non-admin config items here, e.g.:
  // items.push({ name: "nav.availableModels", href: "/config/models", icon: Cpu });

  // Admin-only items
  if (authStore.isAdmin) {
    items.push(
      { name: "nav.providers", href: "/config/providers", icon: Database },
      { name: "nav.models", href: "/config/models", icon: Cpu },
      { name: "nav.mcpServers", href: "/config/mcp-servers", icon: McpIcon },
      { name: "team.title", href: "/team", icon: Users }
    );
  }

  return [{ title: "nav.config", items }];
});

// Settings section — a single sidebar item. The in-page General/Advanced tabs
// (and section headings) handle the rest, so the sidebar no longer mirrors
// every settings subsection. This removes the triple-navigation and the
// anchor-label/heading-label mismatch (e.g. "Server Configuration" vs the
// "Log Management" section heading).
const settingsSections: ComputedRef<NavSection[]> = computed(() => {
  return [
    {
      title: "nav.settings",
      items: [{ name: "nav.settings", href: "/config/settings", icon: Settings }],
    },
  ];
});

// Sections visible based on user role
const visibleSections = computed(() => {
  return [...memberSections, ...configSections.value, ...settingsSections.value];
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

// Change password dialog
const showPasswordDialog = ref(false);
const {
  currentPassword: passwordCurrent,
  newPassword: passwordNew,
  confirmPassword: passwordConfirm,
  error: passwordError,
  loading: isChangingPassword,
  passwordHint,
  reset: resetPasswordForm,
  submit: submitPasswordForm,
} = useChangePassword();

function openPasswordDialog() {
  resetPasswordForm();
  showPasswordDialog.value = true;
}

async function handleChangePassword() {
  if (await submitPasswordForm()) {
    showPasswordDialog.value = false;
    toast.success(t("profile.passwordChanged"));
  }
}

// Change username dialog
const showUsernameDialog = ref(false);
const usernameNew = ref("");
const usernamePassword = ref("");
const usernameError = ref<string | null>(null);
const isChangingUsername = ref(false);

function openUsernameDialog() {
  usernameNew.value = "";
  usernamePassword.value = "";
  usernameError.value = null;
  showUsernameDialog.value = true;
}

async function handleChangeUsername() {
  usernameError.value = null;
  const newName = usernameNew.value.trim();

  if (!newName || newName.length > 64 || !/^[a-zA-Z0-9_-]+$/.test(newName)) {
    usernameError.value = t("profile.usernameInvalid");
    return;
  }
  if (newName.toLowerCase() === (authStore.username ?? "").toLowerCase()) {
    usernameError.value = t("profile.usernameUnchanged");
    return;
  }
  if (!usernamePassword.value) {
    usernameError.value = t("profile.passwordRequired");
    return;
  }

  isChangingUsername.value = true;
  try {
    const res = await meApi.changeUsername(usernamePassword.value, newName);
    // Swap in the fresh JWT: the old token's `sub` still names the previous
    // username and would stop resolving after the rename.
    authStore.setToken(res.access_token);
    showUsernameDialog.value = false;
    toast.success(t("profile.usernameChanged", { username: res.username }));
  } catch (err: unknown) {
    const httpErr = err as { status?: number };
    if (httpErr.status === 401 || httpErr.status === 403) {
      usernameError.value = t("profile.wrongPassword");
    } else if (httpErr.status === 409) {
      usernameError.value = t("profile.usernameTaken");
    } else {
      const msg = getErrorMessage(err);
      usernameError.value = msg || t("errors.saveFailed");
    }
  } finally {
    isChangingUsername.value = false;
  }
}

function isCurrent(href: string): boolean {
  if (href === "/") return route.path === "/";
  // Settings is a single sidebar item; the in-page tabs track their own
  // active state. Any /config/settings route highlights the sidebar item.
  if (href.startsWith("/config/settings")) {
    return route.path === "/config/settings";
  }
  return route.path.startsWith(href.split("?")[0]);
}

const handleLogout = () => {
  authStore.logout();
  router.push("/login");
};

const PREFETCH_ROUTES: Record<string, () => void> = {
  "/config/providers": () => {
    providerStore.prefetch();
    import("@/views/config/ProvidersView.vue");
  },
  "/config/models": () => {
    modelStore.prefetch();
    import("@/views/config/ModelsView.vue");
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
  "/models": () => import("@/views/ModelPlazaView.vue"),
  "/chat": () => import("@/views/ChatView.vue"),
  "/images": () => import("@/views/ImagesView.vue"),
  "/": () => import("@/views/HomeView.vue"),
  "/config/settings": () => import("@/views/config/SettingsView.vue"),
  "/team": () => import("@/views/TeamView.vue"),
};

const prefetchRoute = (href: string) => {
  PREFETCH_ROUTES[href]?.();
};
</script>

<template>
  <Sidebar collapsible="icon" class="border-r border-sidebar-border">
    <!-- Brand header -->
    <SidebarHeader class="h-16 border-b border-sidebar-border justify-center relative">
      <div
        class="flex h-full items-center w-full transition-opacity duration-200"
        :class="isCollapsed ? 'justify-center px-0' : 'justify-between px-4'"
      >
        <div class="flex items-center gap-2 min-w-0">
          <span
            v-if="!isCollapsed"
            class="brand-heading text-lg font-semibold tracking-tight truncate animate-in fade-in duration-200"
          >
            {{ t("home.title") }}
          </span>
          <!-- Instance version badge — links to the About section in Settings.
               Renders nothing while the info is loading or when the fetch failed. -->
          <RouterLink
            v-if="!isCollapsed && systemStore.info"
            to="/config/settings"
            :title="t('about.title')"
            class="flex shrink-0 items-center gap-1 rounded-md border border-sidebar-border/60 px-1.5 py-0.5 font-mono text-[10px] leading-none text-sidebar-foreground/50 transition-colors duration-200 hover:border-sidebar-border hover:text-sidebar-foreground animate-in fade-in duration-200"
          >
            <span>v{{ systemStore.info.version }}</span>
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
        class="absolute top-5 right-0 z-50 hidden md:flex h-6 w-6 translate-x-1/2 items-center justify-center rounded-full border border-sidebar-border bg-background text-sidebar-foreground shadow-xs hover:bg-sidebar-accent hover:text-sidebar-accent-foreground focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-sidebar-ring transition-transform duration-200 cursor-pointer"
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
      <SidebarGroup v-for="section in visibleSections" :key="section.title">
        <SidebarGroupLabel
          class="text-[11px] font-semibold tracking-[0.14em] uppercase text-sidebar-foreground/50"
        >
          {{ t(section.title) }}
        </SidebarGroupLabel>
        <SidebarGroupContent>
          <SidebarMenu class="gap-0.5">
            <SidebarMenuItem v-for="item in section.items" :key="item.name">
              <template v-if="!item.subItems">
                <SidebarMenuButton as-child :is-active="isCurrent(item.href!)">
                  <RouterLink
                    :to="item.href!"
                    :aria-current="isCurrent(item.href!) ? 'page' : undefined"
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
                    <SidebarMenuButton>
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
                        <SidebarMenuSubButton as-child :is-active="isCurrent(sub.href)">
                          <RouterLink
                            :to="sub.href"
                            :aria-current="isCurrent(sub.href) ? 'page' : undefined"
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
                  <span class="truncate text-[11px] text-sidebar-foreground/50 font-medium">
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
                <DropdownMenuItem @click="openUsernameDialog" class="cursor-pointer">
                  <UserPen class="mr-2 size-4 text-muted-foreground" />
                  <span>{{ t("profile.changeUsername") }}</span>
                </DropdownMenuItem>
                <DropdownMenuItem @click="openPasswordDialog" class="cursor-pointer">
                  <KeyRound class="mr-2 size-4 text-muted-foreground" />
                  <span>{{ t("profile.changePassword") }}</span>
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

    <SidebarRail />

    <!-- Change Username Dialog -->
    <Dialog v-model:open="showUsernameDialog">
      <DialogContent class="brand-panel">
        <DialogHeader>
          <DialogTitle>{{ t("profile.changeUsername") }}</DialogTitle>
          <DialogDescription>{{ t("profile.changeUsernameDescription") }}</DialogDescription>
        </DialogHeader>
        <div class="flex flex-col gap-4 py-2">
          <div class="flex flex-col gap-2">
            <Label for="sidebar-new-username">{{ t("profile.newUsername") }}</Label>
            <Input
              id="sidebar-new-username"
              v-model="usernameNew"
              :placeholder="authStore.username ?? ''"
              :disabled="isChangingUsername"
            />
            <p class="text-xs text-muted-foreground">{{ t("profile.usernameFormatHint") }}</p>
          </div>
          <div class="flex flex-col gap-2">
            <Label for="sidebar-username-password">{{ t("profile.currentPassword") }}</Label>
            <Input
              id="sidebar-username-password"
              v-model="usernamePassword"
              type="password"
              :disabled="isChangingUsername"
              @keydown.enter="handleChangeUsername"
            />
          </div>
          <p v-if="usernameError" class="text-destructive text-xs">{{ usernameError }}</p>
        </div>
        <DialogFooter>
          <Button
            variant="outline"
            :disabled="isChangingUsername"
            @click="showUsernameDialog = false"
          >
            {{ t("common.cancel") }}
          </Button>
          <Button :disabled="isChangingUsername" @click="handleChangeUsername">
            <Loader2 v-if="isChangingUsername" class="h-4 w-4 animate-spin" />
            {{ t("profile.changeUsername") }}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>

    <!-- Change Password Dialog -->
    <Dialog v-model:open="showPasswordDialog">
      <DialogContent class="brand-panel">
        <DialogHeader>
          <DialogTitle>{{ t("profile.changePassword") }}</DialogTitle>
          <DialogDescription>{{ t("profile.changePasswordDescription") }}</DialogDescription>
        </DialogHeader>
        <div class="flex flex-col gap-4 py-2">
          <div class="flex flex-col gap-2">
            <Label for="sidebar-current-password">{{ t("profile.currentPassword") }}</Label>
            <Input
              id="sidebar-current-password"
              v-model="passwordCurrent"
              type="password"
              :disabled="isChangingPassword"
            />
          </div>
          <div class="flex flex-col gap-2">
            <Label for="sidebar-new-password">{{ t("profile.newPassword") }}</Label>
            <Input
              id="sidebar-new-password"
              v-model="passwordNew"
              type="password"
              :disabled="isChangingPassword"
            />
            <p v-if="passwordError" class="text-destructive text-xs">{{ passwordError }}</p>
            <p v-else-if="passwordNew" class="text-xs text-muted-foreground">{{ passwordHint }}</p>
          </div>
          <div class="flex flex-col gap-2">
            <Label for="sidebar-confirm-password">{{ t("profile.confirmPassword") }}</Label>
            <Input
              id="sidebar-confirm-password"
              v-model="passwordConfirm"
              type="password"
              :disabled="isChangingPassword"
              @keydown.enter="handleChangePassword"
            />
          </div>
        </div>
        <DialogFooter>
          <Button
            variant="outline"
            :disabled="isChangingPassword"
            @click="showPasswordDialog = false"
          >
            {{ t("common.cancel") }}
          </Button>
          <Button :disabled="isChangingPassword" @click="handleChangePassword">
            <Loader2 v-if="isChangingPassword" class="h-4 w-4 animate-spin" />
            {{ t("profile.changePassword") }}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  </Sidebar>
</template>
