<script setup lang="ts">
import { RefreshCw, Search } from "@lucide/vue";
import { computed, onMounted, ref } from "vue";
import { useI18n } from "vue-i18n";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
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

import type { SyncSectionState } from "./syncSection";
import type {
  MetadataUpdateItem,
  ModelMetadataOption,
  SyncMetadataResponse,
  SyncMetadataResult,
} from "@/types/schemas";

/**
 * Capability/metadata review section of the models.dev sync wizard.
 *
 * Metadata lives on the model (not per provider mapping), so rows are keyed by
 * model name and ambiguous catalog entries are resolved through a per-row
 * source switch. Like the pricing section it owns its fetch, rows and
 * selection, and exposes them through `summary` for the wizard's overview,
 * step gating and single apply.
 */
defineOptions({ name: "CapabilitySyncPanel" });

const { t } = useI18n();

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
type RowStatus = "changed" | "unchanged" | "nodata";
type FilterKey = "changes" | "unchanged" | "nodata";

interface FieldChange {
  field: MetadataField;
  labelKey: string;
  rawOld: boolean | string | number | null;
  rawNew: boolean | string | number | null;
  oldLabel: string;
  newLabel: string;
  /** Nothing stored in this field yet, so the chip shows only the new value. */
  noOld: boolean;
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
const status = ref<"idle" | "loading" | "ready" | "error">("idle");
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
      noOld: oldV === null || oldV === undefined,
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
  status.value = "loading";
  fetchError.value = null;
  try {
    const response = await metadataApi.fetchPreview();
    if (!response.success) {
      fetchError.value = response.error ?? t("models.capabilitySync.fetchFailed");
      status.value = "error";
      return;
    }
    rows.value = buildRows(response);
    activeFilter.value = "changes";
    status.value = "ready";
  } catch (error) {
    fetchError.value = error instanceof Error ? error.message : String(error);
    status.value = "error";
  }
}

// The wizard mounts this panel once and drives nothing else: the preview is
// fetched as soon as the dialog opens, so the overview can show its counts.
onMounted(fetchPreview);

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

interface FilterOption {
  key: FilterKey;
  label: string;
  count: number;
  visible: boolean;
}

const filterOptions = computed<FilterOption[]>(() => [
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

const selectableFilters = computed(() => filterOptions.value.filter((option) => option.visible));

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
// Rows that will actually be written when applied: selected AND carrying a
// real diff against the stored values.
const applicableCount = computed(
  () => rows.value.filter((r) => r.selected && r.changes.length > 0).length
);
// Rows that replace or clear a value you already store are never preselected.
// The dialog reports how many of those are still unchecked, so the preselect
// rule is stated rather than guessed at.
const blockedCount = computed(
  () =>
    rows.value.filter((r) => !r.selected && r.changes.length > 0 && !isPureAddition(r.changes))
      .length
);

const allVisibleSelected = computed(
  () => visibleActionable.value.length > 0 && visibleActionable.value.every((r) => r.selected)
);
const someVisibleSelected = computed(() => visibleActionable.value.some((r) => r.selected));

function onSelectAll(checked: boolean | "indeterminate") {
  const value = checked === "indeterminate" ? false : Boolean(checked);
  for (const row of visibleActionable.value) row.selected = value;
}

// ---- Apply payload & wizard contract ----
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

// Named rows only: the confirmation bar says how many, and these lines say
// which. The changes themselves stay in the table, one click away.
const pending = computed<string[]>(() =>
  rows.value.filter((row) => row.selected && row.changes.length > 0).map((row) => row.modelName)
);

const summary = computed<SyncSectionState>(() => ({
  status: status.value,
  error: fetchError.value,
  changedCount: changedCount.value,
  applicableCount: applicableCount.value,
  blockedCount: blockedCount.value,
  pending: pending.value,
}));

defineExpose({ fetchPreview, buildUpdates, summary });
</script>

<template>
  <div class="flex min-h-0 flex-1 flex-col">
    <!-- Defensive: the wizard only routes here once the preview is ready. -->
    <div
      v-if="status !== 'ready'"
      class="flex flex-1 flex-col items-center justify-center gap-3 px-6 text-center"
    >
      <template v-if="status === 'error'">
        <p class="max-w-md text-sm text-status-error">
          {{ fetchError ?? t("models.capabilitySync.fetchFailed") }}
        </p>
        <Button variant="outline" size="sm" @click="fetchPreview">
          <RefreshCw class="h-3.5 w-3.5" />
          {{ t("common.retry") }}
        </Button>
      </template>
      <template v-else>
        <RefreshCw class="h-4 w-4 animate-spin text-muted-foreground" />
        <span class="text-sm text-muted-foreground">{{ t("models.capabilitySync.fetching") }}</span>
      </template>
    </div>

    <template v-else>
      <!-- Toolbar: bucket filter + search. The filter is a select so it never
           competes visually with the section tabs above it. -->
      <div class="flex flex-wrap items-center gap-3 border-b border-border px-6 py-2.5">
        <div class="flex items-center gap-2">
          <span class="text-xs text-muted-foreground">{{ t("models.sync.filterLabel") }}</span>
          <Select
            :model-value="activeFilter"
            @update:model-value="(v) => (activeFilter = String(v) as FilterKey)"
          >
            <SelectTrigger class="h-8 w-[190px] text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem v-for="option in selectableFilters" :key="option.key" :value="option.key">
                {{ option.label }} · {{ option.count }}
              </SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div class="relative ml-auto">
          <Search
            class="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground"
          />
          <Input
            v-model="searchQuery"
            :placeholder="t('models.sync.searchPlaceholder')"
            class="h-8 w-56 pl-8 text-sm"
          />
        </div>
      </div>

      <!-- Review table -->
      <div class="min-h-0 flex-1 overflow-auto">
        <Table v-if="displayedRows.length > 0" class="w-max min-w-full">
          <TableHeader class="sticky top-0 z-10">
            <TableRow class="hover:border-l-transparent hover:bg-transparent">
              <TableHead class="w-10">
                <Checkbox
                  v-if="visibleActionable.length > 0"
                  :model-value="
                    allVisibleSelected ? true : someVisibleSelected ? 'indeterminate' : false
                  "
                  :aria-label="t('models.capabilitySync.selectAll')"
                  @update:model-value="onSelectAll"
                />
              </TableHead>
              <TableHead class="min-w-[220px]">{{ t("models.capabilitySync.colModel") }}</TableHead>
              <TableHead class="w-40">{{ t("models.capabilitySync.colSource") }}</TableHead>
              <TableHead class="min-w-[320px] max-w-[640px]">{{
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
                  v-if="isActionable(row)"
                  v-model="row.selected"
                  :aria-label="row.modelName"
                />
              </TableCell>
              <TableCell>
                <div class="flex items-center gap-2">
                  <div class="min-w-0">
                    <div class="truncate font-medium">{{ row.modelName }}</div>
                    <div v-if="row.status === 'nodata'" class="text-xs text-muted-foreground">
                      {{ t("models.capabilitySync.nodataHint") }}
                    </div>
                  </div>
                  <Badge
                    v-if="row.status === 'changed'"
                    variant="outline"
                    class="shrink-0"
                    :class="
                      isPureAddition(row.changes)
                        ? 'border-status-success/40 text-status-success'
                        : 'border-action-amber/40 text-action-amber'
                    "
                  >
                    {{
                      isPureAddition(row.changes)
                        ? t("models.capabilitySync.badgeFill")
                        : t("models.capabilitySync.badgeOverwrite")
                    }}
                  </Badge>
                  <Badge
                    v-else-if="row.status === 'nodata'"
                    variant="outline"
                    class="shrink-0 border-border text-muted-foreground"
                  >
                    {{ t("models.capabilitySync.badgeNoData") }}
                  </Badge>
                </div>
              </TableCell>
              <TableCell>
                <Select
                  v-if="isActionable(row) && row.sourceOptions.length > 1"
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
              <!-- Changed fields as old → new chips. A max width keeps long
                   field lists wrapping inside the column instead of pushing the
                   table into horizontal scroll. -->
              <TableCell class="min-w-[320px] max-w-[640px]">
                <div class="flex flex-wrap gap-1">
                  <span
                    v-for="change in row.changes"
                    :key="change.field"
                    class="inline-flex items-center gap-1 rounded border border-border px-1.5 py-0.5 text-[11px] text-data"
                  >
                    <span class="text-muted-foreground">{{ t(change.labelKey) }}</span>
                    <!-- "— → 16,384" is noise: a gap-fill shows only what it sets. -->
                    <template v-if="!change.noOld">
                      <span class="text-muted-foreground">{{ change.oldLabel }}</span>
                      <span class="text-muted-foreground">→</span>
                    </template>
                    <span>{{ change.newLabel }}</span>
                  </span>
                  <span v-if="row.changes.length === 0" class="text-[11px] text-muted-foreground">
                    —
                  </span>
                </div>
              </TableCell>
            </TableRow>
          </TableBody>
        </Table>

        <!-- Empty state for the active filter -->
        <div v-else class="flex flex-col items-center gap-2 px-6 py-16 text-center">
          <p class="text-sm font-medium">{{ emptyStateMessage.title }}</p>
          <p class="max-w-sm text-sm text-muted-foreground">
            {{ emptyStateMessage.description }}
          </p>
        </div>
      </div>
    </template>
  </div>
</template>
