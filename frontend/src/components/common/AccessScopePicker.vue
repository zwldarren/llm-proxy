<script setup lang="ts">
import type { Component } from "vue";
import { Info, Search, TriangleAlert, X } from "@lucide/vue";
import { computed, ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

/**
 * Allowlist editor with explicit scope semantics:
 * `null` = unrestricted ("All"), `[]` = deny-all, non-empty = allowlist.
 * Keeps selection visible while toggling scopes and never hides the list
 * behind a dismiss-on-select popover.
 */
interface AccessScopeItem {
  id: string;
  name: string;
}

const props = withDefaults(
  defineProps<{
    modelValue: string[] | null;
    items: AccessScopeItem[];
    /** Label for the unrestricted option, e.g. "All models". */
    allLabel: string;
    /** Muted help text shown while in unrestricted (All) mode. */
    allHelpText?: string;
    searchPlaceholder: string;
    /** Shown when `items` is empty (nothing configured server-side). */
    emptyAvailableText: string;
    /** Shown when the search query matches nothing. */
    emptySearchText: string;
    /** Shown in custom mode with an empty selection. */
    emptyText: string;
    /**
     * What an empty custom selection means: "all" = same as unrestricted
     * (info style), "deny" = deny-all footgun (warning style).
     */
    emptyMeaning?: "all" | "deny";
    /** Optional leading icon for chips and rows (e.g. Server). */
    icon?: Component;
    /** Render item names in the data/mono face (for technical identifiers). */
    mono?: boolean;
    disabled?: boolean;
  }>(),
  {
    icon: undefined,
    mono: false,
    disabled: false,
    emptyMeaning: "all",
    allHelpText: undefined,
  }
);

const emit = defineEmits<{
  "update:modelValue": [value: string[] | null];
}>();

const { t } = useI18n();

const searchQuery = ref("");

// Remember the last non-empty custom selection so toggling All -> Custom
// restores it instead of dropping the user into a deny-all state.
const lastSelection = ref<string[]>([]);
watch(
  () => props.modelValue,
  (value) => {
    if (Array.isArray(value) && value.length > 0) {
      lastSelection.value = [...value];
    }
  },
  { immediate: true }
);

const mode = computed<"all" | "custom">(() => (props.modelValue === null ? "all" : "custom"));

function onModeChange(value: unknown) {
  // Reka emits "" / null when the active item is clicked again — keep the mode sticky.
  if (value !== "all" && value !== "custom") return;
  if (value === mode.value) return;
  emit("update:modelValue", value === "all" ? null : [...lastSelection.value]);
  searchQuery.value = "";
}

const selection = computed<string[]>(() => props.modelValue ?? []);
const selectionSet = computed(() => new Set(selection.value));

const filteredItems = computed(() => {
  const query = searchQuery.value.trim().toLowerCase();
  if (!query) return props.items;
  return props.items.filter((item) => item.name.toLowerCase().includes(query));
});

const allFilteredSelected = computed(
  () =>
    filteredItems.value.length > 0 && filteredItems.value.every((i) => selectionSet.value.has(i.id))
);

function toggleItem(id: string) {
  const next = selectionSet.value.has(id)
    ? selection.value.filter((s) => s !== id)
    : [...selection.value, id];
  emit("update:modelValue", next);
}

function selectAllFiltered() {
  const next = new Set(selection.value);
  for (const item of filteredItems.value) next.add(item.id);
  emit("update:modelValue", [...next]);
}

function clearSelection() {
  emit("update:modelValue", []);
}

function itemName(id: string): string {
  return props.items.find((i) => i.id === id)?.name ?? id;
}

// Prune stale IDs when the available items change
watch(
  () => props.items,
  (items) => {
    if (!Array.isArray(props.modelValue)) return;
    const validIds = new Set(items.map((i) => i.id));
    const pruned = props.modelValue.filter((id) => validIds.has(id));
    if (pruned.length !== props.modelValue.length) {
      emit("update:modelValue", pruned);
    }
  }
);
</script>

<template>
  <div class="space-y-2.5">
    <!-- Scope toggle: All (unrestricted) vs Custom (allowlist) -->
    <ToggleGroup
      :model-value="mode"
      type="single"
      :spacing="1"
      :disabled="disabled"
      class="grid w-full grid-cols-2 rounded-lg border border-border/60 bg-muted/40 p-[3px]"
      @update:model-value="onModeChange"
    >
      <ToggleGroupItem
        value="all"
        :aria-label="allLabel"
        class="h-8 rounded-md px-3 text-xs font-medium text-muted-foreground transition-colors data-[state=on]:bg-background data-[state=on]:text-foreground data-[state=on]:shadow-sm"
      >
        {{ allLabel }}
      </ToggleGroupItem>
      <ToggleGroupItem
        value="custom"
        :aria-label="t('accessPicker.custom')"
        class="h-8 rounded-md px-3 text-xs font-medium text-muted-foreground transition-colors data-[state=on]:bg-background data-[state=on]:text-foreground data-[state=on]:shadow-sm"
      >
        {{ t("accessPicker.custom") }}
      </ToggleGroupItem>
    </ToggleGroup>

    <!-- Help text for the unrestricted state -->
    <p v-if="mode === 'all' && allHelpText" class="text-xs text-muted-foreground">
      {{ allHelpText }}
    </p>

    <!-- Custom allowlist editor -->
    <template v-if="mode === 'custom'">
      <fieldset :disabled="disabled" class="contents">
        <div class="rounded-lg border border-border/70 bg-background/60">
          <!-- Nothing configured server-side -->
          <p v-if="items.length === 0" class="px-3 py-6 text-center text-sm text-muted-foreground">
            {{ emptyAvailableText }}
          </p>

          <template v-else>
            <!-- Selected chips (dismissible) -->
            <div
              v-if="selection.length > 0"
              class="flex flex-wrap gap-1.5 border-b border-border/60 px-3 py-2.5"
            >
              <Badge
                v-for="id in selection"
                :key="id"
                variant="secondary"
                class="flex items-center gap-1 pr-1 font-normal"
                :class="mono ? 'font-mono text-[11px]' : 'text-xs'"
              >
                <component :is="icon" v-if="icon" class="h-3 w-3 shrink-0" />
                <span class="max-w-48 truncate">{{ itemName(id) }}</span>
                <button
                  type="button"
                  class="ml-0.5 rounded-full p-0.5 transition-colors hover:bg-muted-foreground/20"
                  :aria-label="t('accessPicker.removeSelection', { name: itemName(id) })"
                  @click="toggleItem(id)"
                >
                  <X class="h-3 w-3" />
                </button>
              </Badge>
            </div>

            <!-- Search + quick actions -->
            <div class="flex items-center gap-2 px-3 py-2">
              <div class="relative flex-1">
                <Search
                  class="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground"
                />
                <Input
                  v-model="searchQuery"
                  :placeholder="searchPlaceholder"
                  class="h-8 border-0 bg-transparent pl-8 text-xs shadow-none focus-visible:ring-1"
                />
              </div>
              <Button
                variant="ghost"
                size="sm"
                class="h-7 shrink-0 px-2 text-xs text-muted-foreground hover:text-foreground"
                :disabled="allFilteredSelected"
                @click="selectAllFiltered"
              >
                {{ t("accessPicker.selectAll") }}
              </Button>
              <Button
                variant="ghost"
                size="sm"
                class="h-7 shrink-0 px-2 text-xs text-muted-foreground hover:text-foreground"
                :disabled="selection.length === 0"
                @click="clearSelection"
              >
                {{ t("accessPicker.clear") }}
              </Button>
            </div>

            <!-- Checkbox list (stays open; rows toggle) -->
            <div class="max-h-48 overflow-y-auto border-t border-border/60">
              <p
                v-if="filteredItems.length === 0"
                class="px-3 py-4 text-center text-sm text-muted-foreground"
              >
                {{ emptySearchText }}
              </p>
              <label
                v-for="item in filteredItems"
                :key="item.id"
                class="flex cursor-pointer items-center gap-2.5 px-3 py-2 transition-colors hover:bg-accent/60"
                :class="{ 'bg-accent/40': selectionSet.has(item.id) }"
              >
                <Checkbox
                  :model-value="selectionSet.has(item.id)"
                  :aria-label="item.name"
                  @update:model-value="toggleItem(item.id)"
                />
                <component
                  :is="icon"
                  v-if="icon"
                  class="h-3.5 w-3.5 shrink-0 text-muted-foreground"
                />
                <span class="truncate" :class="mono ? 'font-mono text-xs' : 'text-sm'">
                  {{ item.name }}
                </span>
              </label>
            </div>

            <!-- Selection count -->
            <div class="border-t border-border/60 px-3 py-1.5">
              <span class="text-[11px] tabular-nums text-muted-foreground">
                {{
                  t("accessPicker.selectedCount", { count: selection.length, total: items.length })
                }}
              </span>
            </div>
          </template>
        </div>

        <!-- Empty-selection meaning: deny-all warning vs. allow-all info -->
        <p
          v-if="items.length > 0 && selection.length === 0"
          class="flex items-start gap-1.5 text-xs"
          :class="emptyMeaning === 'deny' ? 'text-status-warning' : 'text-muted-foreground'"
        >
          <TriangleAlert v-if="emptyMeaning === 'deny'" class="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <Info v-else class="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>{{ emptyText }}</span>
        </p>
      </fieldset>
    </template>
  </div>
</template>
