<script setup lang="ts">
import { computed, nextTick, ref, shallowRef, watch } from "vue";
import { useMagicKeys, whenever } from "@vueuse/core";
import { useI18n } from "vue-i18n";
import { useRouter } from "vue-router";
import { Search } from "@lucide/vue";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { logsApi } from "@/services/api/logs";
import { useAuthStore } from "@/stores/auth";
import type { LogListItem } from "@/types/schemas";

const { t } = useI18n();
const router = useRouter();
const authStore = useAuthStore();

const open = ref(false);
const query = ref("");
const activeIndex = ref(0);
const inputEl = ref<HTMLInputElement | null>(null);

// Navigation commands — mirrors the sidebar's primary destinations, filtered
// by admin role. Each carries its router target so Enter can navigate.
interface Command {
  id: string;
  label: string;
  group: string;
  hint?: string;
  run: () => void;
}

const navCommands = computed<Command[]>(() => {
  const base: Command[] = [
    {
      id: "nav-home",
      label: t("home.usageTitle"),
      group: t("nav.overview"),
      run: () => router.push("/"),
    },
    {
      id: "nav-logs",
      label: t("nav.logs"),
      group: t("nav.overview"),
      run: () => router.push("/logs"),
    },
    {
      id: "nav-models",
      label: t("nav.modelPlaza"),
      group: t("nav.catalog"),
      run: () => router.push("/models"),
    },
    {
      id: "nav-chat",
      label: t("nav.chat"),
      group: t("nav.playground"),
      run: () => router.push("/chat"),
    },
    {
      id: "nav-images",
      label: t("nav.images"),
      group: t("nav.playground"),
      run: () => router.push("/images"),
    },
    {
      id: "nav-apikeys",
      label: t("nav.apiKeys"),
      group: t("nav.config"),
      run: () => router.push("/config/api-keys"),
    },
    {
      id: "nav-settings",
      label: t("nav.settings"),
      group: t("nav.settings"),
      run: () => router.push("/config/settings"),
    },
  ];
  if (authStore.isAdmin) {
    base.push(
      {
        id: "nav-providers",
        label: t("nav.providers"),
        group: t("nav.config"),
        run: () => router.push("/config/providers"),
      },
      {
        id: "nav-models-cfg",
        label: t("nav.models"),
        group: t("nav.config"),
        run: () => router.push("/config/models"),
      },
      {
        id: "nav-mcp",
        label: t("nav.mcpServers"),
        group: t("nav.config"),
        run: () => router.push("/config/mcp-servers"),
      },
      {
        id: "nav-team",
        label: t("team.title"),
        group: t("nav.config"),
        run: () => router.push("/team"),
      }
    );
  }
  return base;
});

// Recent logs fetched lazily when the palette opens. Bounded to a handful so the
// palette stays a keyboard accelerator, not a data table.
const recentLogs = shallowRef<LogListItem[]>([]);
const loadingLogs = ref(false);

async function loadRecentLogs() {
  loadingLogs.value = true;
  try {
    const res = await logsApi.getLogs({ page: 1, page_size: 6, log_type: "endpoint" });
    recentLogs.value = res.items ?? [];
  } catch {
    recentLogs.value = [];
  } finally {
    loadingLogs.value = false;
  }
}

const logCommands = computed<Command[]>(() =>
  recentLogs.value.map((log) => ({
    id: `log-${log.request_id}`,
    label: log.model || log.provider || log.request_id,
    group: t("nav.logs"),
    hint: `${log.status_code} · ${log.request_id}`,
    run: () => router.push({ path: "/logs", query: { tab: "proxy" } }),
  }))
);

const filtered = computed<Command[]>(() => {
  const q = query.value.trim().toLowerCase();
  const all = [...navCommands.value, ...logCommands.value];
  if (!q) return all;
  return all.filter(
    (c) =>
      c.label.toLowerCase().includes(q) ||
      c.group.toLowerCase().includes(q) ||
      (c.hint?.toLowerCase().includes(q) ?? false)
  );
});

// Reset state whenever the dialog opens/closes.
watch(open, (isOpen) => {
  if (isOpen) {
    query.value = "";
    activeIndex.value = 0;
    void loadRecentLogs();
    void nextTick(() => inputEl.value?.focus());
  }
});

watch(query, () => {
  activeIndex.value = 0;
});

const move = (delta: number) => {
  const n = filtered.value.length;
  if (n === 0) return;
  activeIndex.value = (activeIndex.value + delta + n) % n;
  // Scroll the active row into view.
  const el = document.querySelector<HTMLElement>(`[data-cmd-index="${activeIndex.value}"]`);
  el?.scrollIntoView({ block: "nearest" });
};

const runActive = () => {
  const cmd = filtered.value[activeIndex.value];
  if (!cmd) return;
  open.value = false;
  cmd.run();
};

const onKeydown = (e: KeyboardEvent) => {
  if (e.key === "ArrowDown") {
    e.preventDefault();
    move(1);
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    move(-1);
  } else if (e.key === "Enter") {
    e.preventDefault();
    runActive();
  }
};

// Global Cmd/Ctrl+K toggles the palette. useMagicKeys gives us a reactive
// flag without racing manual addEventListener cleanup.
const keys = useMagicKeys();
whenever(keys["ctrl_k"], () => {
  open.value = !open.value;
});
whenever(keys["meta_k"], () => {
  open.value = !open.value;
});
</script>

<template>
  <Dialog v-model:open="open">
    <DialogContent
      class="sm:max-w-xl p-0 gap-0 overflow-hidden top-[20%] translate-y-0 block"
      @keydown="onKeydown"
    >
      <DialogTitle class="sr-only">{{ t("nav.commandPalette") }}</DialogTitle>
      <DialogDescription class="sr-only">{{
        t("nav.commandPaletteDescription")
      }}</DialogDescription>

      <!-- Search field -->
      <div class="flex items-center gap-2.5 px-3.5 h-13 border-b border-border/60">
        <Search class="size-4 text-muted-foreground shrink-0" />
        <input
          ref="inputEl"
          v-model="query"
          type="text"
          :placeholder="t('nav.commandPalettePlaceholder')"
          class="flex-1 bg-transparent outline-none text-sm text-foreground placeholder:text-muted-foreground/70"
          autocomplete="off"
          spellcheck="false"
        />
        <kbd
          class="hidden sm:inline-flex h-5 select-none items-center gap-0.5 rounded border border-border/60 bg-muted/40 px-1.5 font-mono text-[11px] text-muted-foreground"
        >
          {{ t("common.esc") }}
        </kbd>
      </div>

      <!-- Results -->
      <div class="max-h-80 overflow-y-auto py-1.5">
        <div
          v-if="filtered.length === 0"
          class="px-4 py-8 text-center text-sm text-muted-foreground"
        >
          {{ t("common.noResults") }}
        </div>
        <template v-else>
          <button
            v-for="(cmd, idx) in filtered"
            :key="cmd.id"
            type="button"
            :data-cmd-index="idx"
            class="flex w-full items-center gap-3 px-3.5 py-2 text-left text-sm transition-colors outline-none"
            :class="
              idx === activeIndex
                ? 'bg-muted/60 text-foreground'
                : 'text-foreground/80 hover:bg-muted/40'
            "
            @mousemove="activeIndex = idx"
            @click="runActive"
          >
            <span class="flex-1 min-w-0">
              <span class="block truncate font-medium">{{ cmd.label }}</span>
              <span
                v-if="cmd.hint"
                class="block truncate font-mono text-[11px] text-muted-foreground"
              >
                {{ cmd.hint }}
              </span>
            </span>
            <span class="text-[11px] text-muted-foreground/70 shrink-0">{{ cmd.group }}</span>
          </button>
        </template>
        <div v-if="loadingLogs" class="px-4 py-2 text-[11px] text-muted-foreground">
          {{ t("common.loading") }}…
        </div>
      </div>

      <!-- Footer hint -->
      <div
        class="flex items-center justify-between px-3.5 h-9 border-t border-border/60 bg-muted/20 text-[11px] text-muted-foreground"
      >
        <span class="flex items-center gap-2">
          <kbd class="font-mono">↑↓</kbd>
          <span>{{ t("nav.commandPaletteNav") }}</span>
        </span>
        <span class="flex items-center gap-2">
          <kbd class="font-mono">↵</kbd>
          <span>{{ t("common.select") }}</span>
        </span>
      </div>
    </DialogContent>
  </Dialog>
</template>
