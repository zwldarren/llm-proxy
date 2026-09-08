<script setup lang="ts">
import { RefreshCw, Search } from "@lucide/vue";
import { computed, ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import { toast } from "vue-sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { DialogFooter } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { metadataApi } from "@/services/api/config";
import { useErrorHandler } from "@/composables/useErrorHandler";

import type {
  MetadataUpdateItem,
  ModelMetadataOption,
  SyncMetadataResponse,
  SyncMetadataResult,
} from "@/types/schemas";

/**
 * Review-and-apply panel for models.dev metadata/capability fields, mirroring
 * the pricing panel's state machine: fetch preview → review per-model diffs →
 * apply only the selected rows. Metadata lives on the model (not per provider
 * mapping), so rows are keyed by model name; ambiguous catalog entries are
 * resolved through a per-row source switch.
 */
defineOptions({ name: "CapabilitySyncPanel" });

const props = defineProps<{ open: boolean }>();
const emit = defineEmits<{
  applied: [];
  close: [];
}>();

const { t } = useI18n();
const { handleError } = useErrorHandler();

// ---- Field catalog ----
// Synced fields must mirror the backend `_SYNCED_FIELDS` order; bool labels
// reuse the plaza capability vocabulary, the rest carry their own keys.
type MetadataField = Extract<
  keyof ModelMetadataOption,
  | "supports_images"
  | "attachment"
  | "reasoning"
  | "tool_call"
  | "structured_output"
  | "temperature"
  | "open_weights"
  | "status"
  | "family"
  | "knowledge"
  | "release_date"
  | "context_length"
  | "max_output_tokens"
>;

interface FieldDef {
  field: MetadataField;
  labelKey: string;
  kind: "bool" | "text" | "num";
}

const FIELD_DEFS: FieldDef[] = [
  { field: "supports_images", labelKey: "plaza.capability.vision", kind: "bool" },
  { field: "attachment", labelKey: "plaza.capability.attachment", kind: "bool" },
  { field: "reasoning", labelKey: "plaza.capability.reasoning", kind: "bool" },
  { field: "tool_call", labelKey: "plaza.capability.toolCall", kind: "bool" },
  { field: "structured_output", labelKey: "plaza.capability.structuredOutput", kind: "bool" },
  { field: "temperature", labelKey: "plaza.capability.temperature", kind: "bool" },
  { field: "open_weights", labelKey: "plaza.capability.openWeights", kind: "bool" },
  { field: "status", labelKey: "models.capabilitySync.fields.status", kind: "text" },
  { field: "family", labelKey: "models.capabilitySync.fields.family", kind: "text" },
  { field: "knowledge", labelKey: "models.capabilitySync.fields.knowledge", kind: "text" },
  { field: "release_date", labelKey: "models.capabilitySync.fields.releaseDate", kind: "text" },
  { field: "context_length", labelKey: "models.capabilitySync.fields.contextLength", kind: "num" },
  {
    field: "max_output_tokens",
    labelKey: "models.capabilitySync.fields.maxOutputTokens",
    kind: "num",
  },
];

// ---- State machine ----
type Step = "idle" | "fetching" | "review" | "applying";
type RowStatus = "changed" | "unchanged" | "nodata";
type FilterKey = "changes" | "unchanged" | "nodata";

interface FieldChange {
  field: MetadataField;
  labelKey: string;
  rawOld: boolean | string | number | null;
  rawNew: boolean | string | number | null;
  oldLabel: string;
  newLabel: string;
}

interface ReviewRow {
  modelName: string;
  status: RowStatus;
  // Classification captured at fetch time. Tabs and counts are driven by
  // this, so switching sources never moves rows between tabs mid-review.
  initialStatus: RowStatus;
  selected: boolean;
  sourceKey: string | null;
  sourceOptions: ModelMetadataOption[];
  old: ModelMetadataOption;
  candidate: ModelMetadataOption;
  changes: FieldChange[];
}

const rows = ref<ReviewRow[]>([]);
const activeFilter = ref<FilterKey>("changes");
const searchQuery = ref("");
const step = ref<Step>("idle");
const fetchError = ref<string | null>(null);

// ---- Diff helpers ----
function formatField(def: FieldDef, value: boolean | string | number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  if (def.kind === "bool") return value ? "✓" : "—";
  if (def.kind === "num") return Number(value).toLocaleString("en-US");
  return String(value);
}

function computeChanges(old: ModelMetadataOption, candidate: ModelMetadataOption): FieldChange[] {
  const changes: FieldChange[] = [];
  for (const def of FIELD_DEFS) {
    const newV = candidate[def.field] ?? null;
    // Catalog fields without data never clear stored values.
    if (newV === null) continue;
    const oldV = old[def.field] ?? null;
    if (newV === oldV) continue;
    changes.push({
      field: def.field,
      labelKey: def.labelKey,
      rawOld: oldV,
      rawNew: newV,
      oldLabel: formatField(def, oldV),
      newLabel: formatField(def, newV),
    });
  }
  return changes;
}

function deriveStatus(sourceCount: number, changes: FieldChange[]): RowStatus {
  if (sourceCount === 0) return "nodata";
  if (changes.length === 0) return "unchanged";
  return "changed";
}

/**
 * Rows that only fill in previously-unset values are safe to auto-select:
 * the models.dev-mirrored columns default to false/empty, so "false → true"
 * is a gap fill rather than an overwrite of an admin decision. Rows that
 * overwrite or clear a stored value require explicit opt-in.
 */
function isPureAddition(changes: FieldChange[]): boolean {
  return changes.every((c) => c.rawOld === null || c.rawOld === false);
}

function candidateFromSource(row: SyncMetadataResult, source: string): ModelMetadataOption | null {
  return row.available_sources.find((o) => o.source === source) ?? null;
}

function buildRows(response: SyncMetadataResponse): ReviewRow[] {
  return response.results.map((r) => {
    const sourceKey = r.selected_source ?? null;
    const candidate = (sourceKey && candidateFromSource(r, sourceKey)) || r.old;
    const changes = computeChanges(r.old, candidate);
    const status = deriveStatus(r.available_sources.length, changes);
    return {
      modelName: r.model_name,
      status,
      initialStatus: status,
      selected: status === "changed" && isPureAddition(changes),
      sourceKey,
      sourceOptions: r.available_sources,
      old: r.old,
      candidate,
      changes,
    };
  });
}

// ---- Fetch ----
async function fetchPreview() {
  step.value = "fetching";
  fetchError.value = null;
  try {
    const response = await metadataApi.fetchPreview();
    if (!response.success) {
      fetchError.value = response.error ?? t("models.capabilitySync.fetchFailed");
      step.value = "idle";
      return;
    }
    rows.value = buildRows(response);
    activeFilter.value = "changes";
    step.value = "review";
  } catch (error) {
    fetchError.value = error instanceof Error ? error.message : String(error);
    step.value = "idle";
  }
}

// ---- Source switching ----
function onSourceChange(row: ReviewRow, source: string) {
  const candidate = row.sourceOptions.find((o) => o.source === source);
  if (!candidate) return;
  row.sourceKey = source;
  row.candidate = candidate;
  row.changes = computeChanges(row.old, candidate);
  row.status = deriveStatus(row.sourceOptions.length, row.changes);
}

// ---- Filters & tabs ----
function isActionable(row: ReviewRow): boolean {
  return row.status !== "nodata";
}

function matchesSearch(row: ReviewRow): boolean {
  const q = searchQuery.value.trim().toLowerCase();
  if (!q) return true;
  return row.modelName.toLowerCase().includes(q);
}

// Tab membership and counts use `initialStatus` (the classification at fetch
// time) so switching a source never reshuffles tabs under the user.
const changedRows = computed(() => rows.value.filter((r) => r.initialStatus === "changed"));
const unchangedRows = computed(() => rows.value.filter((r) => r.initialStatus === "unchanged"));
const noDataRows = computed(() => rows.value.filter((r) => r.initialStatus === "nodata"));

const counts = computed(() => ({
  changed: changedRows.value.length,
  unchanged: unchangedRows.value.length,
  nodata: noDataRows.value.length,
}));

const changedCount = computed(() => counts.value.changed);

interface FilterTab {
  key: FilterKey;
  label: string;
  count: number;
  visible: boolean;
}

const filterTabs = computed<FilterTab[]>(() => [
  {
    key: "changes",
    label: t("models.capabilitySync.tabChanges"),
    count: counts.value.changed,
    visible: true,
  },
  {
    key: "unchanged",
    label: t("models.capabilitySync.tabUnchanged"),
    count: counts.value.unchanged,
    visible: true,
  },
  {
    key: "nodata",
    label: t("models.capabilitySync.tabNoData"),
    count: counts.value.nodata,
    visible: counts.value.nodata > 0,
  },
]);

const displayedRows = computed(() => {
  const base =
    activeFilter.value === "changes"
      ? changedRows.value
      : activeFilter.value === "unchanged"
        ? unchangedRows.value
        : noDataRows.value;
  return base.filter(matchesSearch);
});

const visibleActionable = computed(() => displayedRows.value.filter(isActionable));

const emptyStateMessage = computed(() => {
  if (changedCount.value === 0 && activeFilter.value === "changes") {
    return {
      title: t("models.capabilitySync.upToDate"),
      description: t("models.capabilitySync.upToDateDescription"),
    };
  }
  if (activeFilter.value === "unchanged") {
    return {
      title: t("models.capabilitySync.emptyUnchanged"),
      description: t("models.capabilitySync.emptyUnchangedDescription"),
    };
  }
  return {
    title: t("models.capabilitySync.emptyNoData"),
    description: t("models.capabilitySync.emptyNoDataDescription"),
  };
});

// ---- Selection ----
const selectedCount = computed(() => rows.value.filter((r) => r.selected).length);
// Rows that will actually be written when applied: selected AND carrying a
// real diff against the stored values.
const applicableCount = computed(
  () => rows.value.filter((r) => r.selected && r.changes.length > 0).length
);
const selectedWithoutChange = computed(() => selectedCount.value - applicableCount.value);

const allVisibleSelected = computed(
  () => visibleActionable.value.length > 0 && visibleActionable.value.every((r) => r.selected)
);
const someVisibleSelected = computed(() => visibleActionable.value.some((r) => r.selected));

function onSelectAll(checked: boolean | "indeterminate") {
  const value = checked === "indeterminate" ? false : Boolean(checked);
  for (const row of visibleActionable.value) row.selected = value;
}

function clearSelection() {
  for (const row of rows.value) row.selected = false;
}

function selectAllActionable() {
  for (const row of rows.value) {
    if (isActionable(row)) row.selected = true;
  }
}

// ---- Apply ----
function buildUpdates(): MetadataUpdateItem[] {
  const updates: MetadataUpdateItem[] = [];
  for (const row of rows.value) {
    if (!row.selected || row.changes.length === 0) continue;
    const item: MetadataUpdateItem = { model_name: row.modelName };
    const fields = item as unknown as Record<MetadataField, unknown>;
    for (const change of row.changes) {
      // Send the raw candidate value; omitted fields stay untouched.
      fields[change.field] = row.candidate[change.field] ?? null;
    }
    updates.push(item);
  }
  return updates;
}

async function applySelected() {
  const updates = buildUpdates();
  if (updates.length === 0) return;
  step.value = "applying";
  try {
    const response = await metadataApi.applyMetadata({ updates });
    if (response.applied_count > 0) {
      toast.success(t("models.capabilitySync.applied", { count: response.applied_count }));
    }
    if (response.failed_count > 0) {
      toast.error(t("models.capabilitySync.applyFailed", { count: response.failed_count }));
      step.value = "review";
      return;
    }
    emit("applied");
    emit("close");
  } catch (error) {
    handleError(error);
    step.value = "review";
  }
}

// ---- Panel lifecycle ----
// Mounted inside the shell's DialogContent: only exists while the dialog is
// open; `immediate` covers the mount-with-open=true case.
watch(
  () => props.open,
  (open) => {
    if (open) {
      step.value = "idle";
      fetchError.value = null;
      rows.value = [];
      activeFilter.value = "changes";
      searchQuery.value = "";
      // Fetch immediately — the review table is the point of the panel.
      fetchPreview();
    }
  },
  { immediate: true }
);
</script>

<template>
  <div class="flex min-h-0 flex-1 flex-col">
    <!-- Step: idle — only reachable after a fetch error; offer retry -->
    <div
      v-if="step === 'idle'"
      class="flex-1 flex flex-col items-center justify-center gap-4 py-16 text-center px-6"
    >
      <p class="text-sm text-muted-foreground max-w-md">
        {{ t("models.capabilitySync.idleDescription") }}
      </p>
      <p v-if="fetchError" class="text-sm text-status-error">{{ fetchError }}</p>
      <Button class="btn-action" @click="fetchPreview">
        <RefreshCw class="w-4 h-4 mr-2" />
        {{ t("models.capabilitySync.retry") }}
      </Button>
    </div>

    <!-- Step: fetching -->
    <div
      v-else-if="step === 'fetching'"
      class="flex-1 flex items-center justify-center gap-3 py-20"
    >
      <RefreshCw class="w-4 h-4 animate-spin text-muted-foreground" />
      <span class="text-sm text-muted-foreground">{{ t("models.capabilitySync.fetching") }}</span>
    </div>

    <!-- Step: review / applying -->
    <template v-else>
      <!-- Toolbar: filter tabs + search -->
      <div class="px-6 py-3 border-b border-border flex flex-wrap items-center gap-3">
        <div class="flex items-center gap-1 rounded-md bg-muted/40 p-0.5">
          <button
            v-for="tab in filterTabs"
            :key="tab.key"
            v-show="tab.visible"
            type="button"
            class="inline-flex items-center gap-1.5 rounded-[5px] px-2.5 py-1 text-xs font-medium transition-colors"
            :class="
              activeFilter === tab.key
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            "
            @click="activeFilter = tab.key"
          >
            {{ tab.label }}
            <span
              class="rounded-full px-1.5 py-px text-[11px] font-semibold tabular-nums"
              :class="
                activeFilter === tab.key
                  ? 'bg-muted text-muted-foreground'
                  : 'bg-muted/50 text-muted-foreground'
              "
            >
              {{ tab.count }}
            </span>
          </button>
        </div>

        <div class="ml-auto relative">
          <Search
            class="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground"
          />
          <Input
            v-model="searchQuery"
            :placeholder="t('common.searchPlaceholder')"
            class="h-8 w-56 pl-8 text-sm"
          />
        </div>
      </div>

      <!-- Review table -->
      <div class="flex-1 min-h-0 overflow-auto">
        <Table v-if="displayedRows.length > 0" class="w-max min-w-full">
          <TableHeader class="sticky top-0 z-10">
            <TableRow class="hover:bg-transparent hover:border-l-transparent">
              <TableHead class="w-10">
                <Checkbox
                  v-if="visibleActionable.length > 0"
                  :model-value="
                    allVisibleSelected ? true : someVisibleSelected ? 'indeterminate' : false
                  "
                  :aria-label="t('models.capabilitySync.selectAllActionable')"
                  @update:model-value="onSelectAll"
                />
              </TableHead>
              <TableHead class="min-w-[220px]">{{ t("models.capabilitySync.colModel") }}</TableHead>
              <TableHead class="w-40">{{ t("models.capabilitySync.colSource") }}</TableHead>
              <TableHead class="min-w-[320px]">{{
                t("models.capabilitySync.colChanges")
              }}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            <TableRow
              v-for="row in displayedRows"
              :key="row.modelName"
              :data-state="row.selected ? 'selected' : undefined"
            >
              <TableCell>
                <Checkbox
                  v-if="step !== 'applying' && isActionable(row)"
                  v-model="row.selected"
                  :aria-label="row.modelName"
                />
              </TableCell>
              <TableCell>
                <div class="flex items-center gap-2">
                  <div class="min-w-0">
                    <div class="font-medium truncate">{{ row.modelName }}</div>
                    <div v-if="row.status === 'nodata'" class="text-xs text-muted-foreground">
                      {{ t("models.capabilitySync.nodataHint") }}
                    </div>
                  </div>
                  <Badge
                    v-if="row.status === 'changed'"
                    variant="outline"
                    class="shrink-0 text-action-amber border-action-amber/40"
                  >
                    {{ t("models.capabilitySync.badgeChanged") }}
                  </Badge>
                  <Badge
                    v-else-if="row.status === 'nodata'"
                    variant="outline"
                    class="shrink-0 text-muted-foreground border-border"
                  >
                    {{ t("models.capabilitySync.badgeNoData") }}
                  </Badge>
                </div>
              </TableCell>
              <TableCell>
                <Select
                  v-if="step !== 'applying' && isActionable(row) && row.sourceOptions.length > 1"
                  :model-value="row.sourceKey ?? undefined"
                  @update:model-value="(v) => onSourceChange(row, String(v))"
                >
                  <SelectTrigger class="h-8 w-full text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem
                      v-for="opt in row.sourceOptions"
                      :key="opt.source"
                      :value="opt.source"
                    >
                      {{ opt.source }}
                    </SelectItem>
                  </SelectContent>
                </Select>
                <span v-else class="text-data text-muted-foreground">
                  {{ row.sourceKey ?? "—" }}
                </span>
              </TableCell>
              <!-- Changed fields as old → new chips -->
              <TableCell>
                <div class="flex flex-wrap gap-1">
                  <span
                    v-for="change in row.changes"
                    :key="change.field"
                    class="inline-flex items-center gap-1 rounded border border-border px-1.5 py-0.5 text-[11px] text-data"
                  >
                    <span class="text-muted-foreground">{{ t(change.labelKey) }}</span>
                    <span class="text-muted-foreground">{{ change.oldLabel }}</span>
                    <span class="text-muted-foreground">→</span>
                    <span>{{ change.newLabel }}</span>
                  </span>
                  <span v-if="row.changes.length === 0" class="text-muted-foreground text-[11px]">
                    —
                  </span>
                </div>
              </TableCell>
            </TableRow>
          </TableBody>
        </Table>

        <!-- Empty state for the active filter -->
        <div v-else class="flex flex-col items-center gap-2 py-16 text-center px-6">
          <p class="text-sm font-medium">{{ emptyStateMessage.title }}</p>
          <p class="text-sm text-muted-foreground max-w-sm">
            {{ emptyStateMessage.description }}
          </p>
        </div>
      </div>

      <!-- Selection hint + footer -->
      <div class="px-6 py-3 border-t border-border">
        <div class="flex flex-wrap items-center gap-3 text-xs text-muted-foreground mb-3">
          <span class="font-medium text-foreground">
            {{
              t("models.capabilitySync.selectionSummary", {
                selected: selectedCount,
                applicable: applicableCount,
              })
            }}
          </span>
          <span v-if="selectedWithoutChange > 0" class="text-status-warning">
            {{ t("models.capabilitySync.noChangeHint") }}
          </span>
          <template v-if="changedCount > 0">
            <span>·</span>
            <button
              type="button"
              class="underline-offset-2 hover:underline hover:text-foreground transition-colors"
              @click="selectAllActionable"
            >
              {{ t("models.capabilitySync.selectAllActionable") }}
            </button>
            <span>·</span>
            <button
              type="button"
              class="underline-offset-2 hover:underline hover:text-foreground transition-colors"
              @click="clearSelection"
            >
              {{ t("models.capabilitySync.clearSelection") }}
            </button>
          </template>
          <span class="ml-auto">{{ t("models.capabilitySync.reviewHint") }}</span>
        </div>
        <DialogFooter class="gap-2 sm:gap-2">
          <Button variant="outline" :disabled="step === 'applying'" @click="emit('close')">
            {{ t("common.cancel") }}
          </Button>
          <Button
            class="btn-action"
            :disabled="applicableCount === 0 || step === 'applying'"
            @click="applySelected"
          >
            <RefreshCw v-if="step === 'applying'" class="w-4 h-4 mr-2 animate-spin" />
            {{
              step === "applying"
                ? t("models.capabilitySync.applying")
                : t("models.capabilitySync.apply", { count: applicableCount })
            }}
          </Button>
        </DialogFooter>
      </div>
    </template>
  </div>
</template>
