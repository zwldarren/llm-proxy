<script setup lang="ts">
import { copyToClipboard } from "@/utils/clipboard";
import { ref, computed, watch } from "vue";
import { useI18n } from "vue-i18n";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Search,
  Copy,
  Check,
  Wrench,
  MessageSquare,
  FolderOpen,
  X,
  Loader2,
  AlertCircle,
} from "@lucide/vue";
import type { McpServerCapabilities } from "@/types/schemas";

interface Props {
  open: boolean;
  serverName: string;
  capabilities?: McpServerCapabilities;
  initialTab?: "tools" | "prompts" | "resources";
  isLoading?: boolean;
  capabilitiesFailed?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  initialTab: "tools",
  isLoading: false,
  capabilitiesFailed: false,
});

const emit = defineEmits<{
  "update:open": [value: boolean];
}>();

const { t } = useI18n();

const activeTab = ref<"tools" | "prompts" | "resources">("tools");
const searchQuery = ref("");
const copiedName = ref<string | null>(null);

// Reset tab & search when dialog opens
watch(
  () => props.open,
  (newVal) => {
    if (newVal) {
      activeTab.value = props.initialTab;
      searchQuery.value = "";
    }
  }
);

watch(
  () => props.initialTab,
  (newVal) => {
    activeTab.value = newVal;
  }
);

const tools = computed(() => props.capabilities?.tools || []);
const prompts = computed(() => props.capabilities?.prompts || []);
const resources = computed(() => props.capabilities?.resources || []);

const filteredTools = computed(() => {
  const query = searchQuery.value.toLowerCase().trim();
  if (!query) return tools.value;
  return tools.value.filter(
    (t) =>
      t.name.toLowerCase().includes(query) ||
      (t.description && t.description.toLowerCase().includes(query))
  );
});

const filteredPrompts = computed(() => {
  const query = searchQuery.value.toLowerCase().trim();
  if (!query) return prompts.value;
  return prompts.value.filter(
    (p) =>
      p.name.toLowerCase().includes(query) ||
      (p.description && p.description.toLowerCase().includes(query))
  );
});

const filteredResources = computed(() => {
  const query = searchQuery.value.toLowerCase().trim();
  if (!query) return resources.value;
  return resources.value.filter(
    (r) =>
      r.name.toLowerCase().includes(query) ||
      (r.description && r.description.toLowerCase().includes(query))
  );
});

const handleCopy = async (text: string) => {
  try {
    await copyToClipboard(text);
    copiedName.value = text;
    setTimeout(() => {
      if (copiedName.value === text) {
        copiedName.value = null;
      }
    }, 1500);
  } catch (err) {
    console.error("Failed to copy", err);
  }
};
</script>

<template>
  <Dialog :open="open" @update:open="emit('update:open', $event)">
    <DialogContent
      class="brand-panel sm:max-w-2xl max-h-[85vh] flex flex-col p-6 gap-4"
      :show-close-button="true"
    >
      <DialogHeader class="pb-2 border-b">
        <div class="flex items-center gap-3">
          <div class="p-2 bg-primary/10 rounded-lg text-primary shrink-0">
            <Wrench v-if="activeTab === 'tools'" class="w-5 h-5" />
            <MessageSquare v-else-if="activeTab === 'prompts'" class="w-5 h-5" />
            <FolderOpen v-else class="w-5 h-5" />
          </div>
          <div class="min-w-0">
            <DialogTitle class="text-xl font-bold flex items-center gap-2 truncate">
              {{ serverName }}
            </DialogTitle>
            <DialogDescription class="text-xs text-muted-foreground mt-0.5">
              {{ t("mcpServers.allCapabilities") }}
            </DialogDescription>
          </div>
        </div>
      </DialogHeader>

      <!-- Search and Filters -->
      <div class="relative w-full shrink-0 flex items-center">
        <Search class="absolute left-3 h-4 w-4 text-muted-foreground z-10 pointer-events-none" />
        <Input
          v-model="searchQuery"
          :placeholder="t('mcpServers.searchCapabilities')"
          class="pl-9 pr-8 h-9 w-full"
        />
        <Button
          v-if="searchQuery"
          variant="ghost"
          size="icon"
          class="absolute right-1.5 h-6 w-6 text-muted-foreground hover:text-foreground z-10"
          @click="searchQuery = ''"
        >
          <X class="w-3 h-3" />
        </Button>
      </div>

      <!-- Tabs -->
      <Tabs v-model="activeTab" class="flex-1 flex flex-col min-h-0">
        <TabsList class="grid grid-cols-3 mb-4 shrink-0">
          <TabsTrigger
            value="tools"
            class="flex items-center justify-center gap-1.5 py-1.5 text-xs"
          >
            <Wrench class="w-3.5 h-3.5" />
            <span>{{ t("mcpServers.tools") }}</span>
            <Badge
              variant="secondary"
              class="ml-1 px-1.5 py-0 text-[11px] font-normal rounded-full bg-muted/60"
            >
              {{ tools.length }}
            </Badge>
          </TabsTrigger>
          <TabsTrigger
            value="prompts"
            class="flex items-center justify-center gap-1.5 py-1.5 text-xs"
          >
            <MessageSquare class="w-3.5 h-3.5" />
            <span>{{ t("mcpServers.prompts") }}</span>
            <Badge
              variant="secondary"
              class="ml-1 px-1.5 py-0 text-[11px] font-normal rounded-full bg-muted/60"
            >
              {{ prompts.length }}
            </Badge>
          </TabsTrigger>
          <TabsTrigger
            value="resources"
            class="flex items-center justify-center gap-1.5 py-1.5 text-xs"
          >
            <FolderOpen class="w-3.5 h-3.5" />
            <span>{{ t("mcpServers.resources") }}</span>
            <Badge
              variant="secondary"
              class="ml-1 px-1.5 py-0 text-[11px] font-normal rounded-full bg-muted/60"
            >
              {{ resources.length }}
            </Badge>
          </TabsTrigger>
        </TabsList>

        <!-- Scrollable content area -->
        <div class="flex-1 min-h-0 relative border rounded-lg bg-muted/10 p-2">
          <!-- Loading state for the whole content -->
          <div
            v-if="isLoading"
            class="h-[360px] flex flex-col items-center justify-center text-muted-foreground gap-3 bg-card/50 backdrop-blur-[0.5px] rounded-md"
          >
            <Loader2 class="w-8 h-8 animate-spin text-primary" />
            <span class="text-xs font-medium">{{ t("mcpServers.loadingCapabilities") }}</span>
          </div>

          <!-- Error state -->
          <div
            v-else-if="capabilitiesFailed"
            class="h-[360px] flex flex-col items-center justify-center text-muted-foreground gap-3 bg-card/50 backdrop-blur-[0.5px] rounded-md"
          >
            <AlertCircle class="w-8 h-8 text-destructive" />
            <span class="text-xs font-medium text-destructive">{{
              t("mcpServers.capabilitiesFailed")
            }}</span>
          </div>

          <template v-else>
            <!-- Tools Tab -->
            <TabsContent value="tools" class="h-full m-0 outline-none">
              <ScrollArea class="h-[360px] pr-1">
                <div v-if="filteredTools.length > 0" class="space-y-2 p-1">
                  <div
                    v-for="tool in filteredTools"
                    :key="tool.name"
                    class="p-3 bg-card hover:bg-accent/40 border rounded-lg transition-colors duration-150 group flex flex-col gap-1.5 relative shadow-xs"
                  >
                    <div class="flex items-center justify-between gap-4">
                      <span
                        class="font-mono text-xs font-semibold text-action-blue bg-action-blue/10 border border-action-blue/20 px-2 py-0.5 rounded truncate max-w-full"
                      >
                        {{ tool.name }}
                      </span>
                      <Button
                        variant="ghost"
                        size="icon"
                        class="h-7 w-7 opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity shrink-0"
                        :title="t('mcpServers.copyName')"
                        @click.stop="handleCopy(tool.name)"
                      >
                        <component
                          :is="copiedName === tool.name ? Check : Copy"
                          class="w-3.5 h-3.5"
                          :class="
                            copiedName === tool.name
                              ? 'text-status-success'
                              : 'text-muted-foreground'
                          "
                        />
                      </Button>
                    </div>
                    <p
                      class="text-xs text-muted-foreground leading-relaxed pr-6 whitespace-pre-wrap"
                    >
                      {{ tool.description || t("mcpServers.noDescription") }}
                    </p>
                  </div>
                </div>
                <div
                  v-else
                  class="h-[250px] flex flex-col items-center justify-center text-muted-foreground gap-2"
                >
                  <Wrench class="w-8 h-8 opacity-25" />
                  <span class="text-xs font-medium">{{
                    searchQuery ? t("mcpServers.noCapabilitiesFound") : t("mcpServers.noTools")
                  }}</span>
                </div>
              </ScrollArea>
            </TabsContent>

            <!-- Prompts Tab -->
            <TabsContent value="prompts" class="h-full m-0 outline-none">
              <ScrollArea class="h-[360px] pr-1">
                <div v-if="filteredPrompts.length > 0" class="space-y-2 p-1">
                  <div
                    v-for="prompt in filteredPrompts"
                    :key="prompt.name"
                    class="p-3 bg-card hover:bg-accent/40 border rounded-lg transition-colors duration-150 group flex flex-col gap-1.5 relative shadow-xs"
                  >
                    <div class="flex items-center justify-between gap-4">
                      <span
                        class="font-mono text-xs font-semibold text-action-violet bg-action-violet/10 border border-action-violet/20 px-2 py-0.5 rounded truncate max-w-full"
                      >
                        {{ prompt.name }}
                      </span>
                      <Button
                        variant="ghost"
                        size="icon"
                        class="h-7 w-7 opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity shrink-0"
                        :title="t('mcpServers.copyName')"
                        @click.stop="handleCopy(prompt.name)"
                      >
                        <component
                          :is="copiedName === prompt.name ? Check : Copy"
                          class="w-3.5 h-3.5"
                          :class="
                            copiedName === prompt.name
                              ? 'text-status-success'
                              : 'text-muted-foreground'
                          "
                        />
                      </Button>
                    </div>
                    <p
                      class="text-xs text-muted-foreground leading-relaxed pr-6 whitespace-pre-wrap"
                    >
                      {{ prompt.description || t("mcpServers.noDescription") }}
                    </p>
                  </div>
                </div>
                <div
                  v-else
                  class="h-[250px] flex flex-col items-center justify-center text-muted-foreground gap-2"
                >
                  <MessageSquare class="w-8 h-8 opacity-25" />
                  <span class="text-xs font-medium">{{
                    searchQuery ? t("mcpServers.noCapabilitiesFound") : t("mcpServers.noPrompts")
                  }}</span>
                </div>
              </ScrollArea>
            </TabsContent>

            <!-- Resources Tab -->
            <TabsContent value="resources" class="h-full m-0 outline-none">
              <ScrollArea class="h-[360px] pr-1">
                <div v-if="filteredResources.length > 0" class="space-y-2 p-1">
                  <div
                    v-for="resource in filteredResources"
                    :key="resource.name"
                    class="p-3 bg-card hover:bg-accent/40 border rounded-lg transition-colors duration-150 group flex flex-col gap-1.5 relative shadow-xs"
                  >
                    <div class="flex items-center justify-between gap-4">
                      <span
                        class="font-mono text-xs font-semibold text-action-amber bg-action-amber/10 border border-action-amber/20 px-2 py-0.5 rounded truncate max-w-full"
                      >
                        {{ resource.name }}
                      </span>
                      <Button
                        variant="ghost"
                        size="icon"
                        class="h-7 w-7 opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity shrink-0"
                        :title="t('mcpServers.copyName')"
                        @click.stop="handleCopy(resource.name)"
                      >
                        <component
                          :is="copiedName === resource.name ? Check : Copy"
                          class="w-3.5 h-3.5"
                          :class="
                            copiedName === resource.name
                              ? 'text-status-success'
                              : 'text-muted-foreground'
                          "
                        />
                      </Button>
                    </div>
                    <p
                      class="text-xs text-muted-foreground leading-relaxed pr-6 whitespace-pre-wrap"
                    >
                      {{ resource.description || t("mcpServers.noDescription") }}
                    </p>
                  </div>
                </div>
                <div
                  v-else
                  class="h-[250px] flex flex-col items-center justify-center text-muted-foreground gap-2"
                >
                  <FolderOpen class="w-8 h-8 opacity-25" />
                  <span class="text-xs font-medium">{{
                    searchQuery ? t("mcpServers.noCapabilitiesFound") : t("mcpServers.noResources")
                  }}</span>
                </div>
              </ScrollArea>
            </TabsContent>
          </template>
        </div>
      </Tabs>
    </DialogContent>
  </Dialog>
</template>
