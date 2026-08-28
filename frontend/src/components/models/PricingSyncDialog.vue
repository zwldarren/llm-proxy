<script setup lang="ts">
import { RefreshCw, Search } from "@lucide/vue";
import { computed, ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import { toast } from "vue-sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
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
import { pricingApi } from "@/services/api/config";
import { useErrorHandler } from "@/composables/useErrorHandler";

import type { PricingUpdateItem, SyncPricingResult, SyncPricingResponse } from "@/types/schemas";

defineOptions({ name: "PricingSyncDialog" });

const props = defineProps<{ open: boolean }>();
const emit = defineEmits<{
  "update:open": [value: boolean];
  applied: [];
}>();

const { t } = useI18n();
const { handleError } = useErrorHandler();

// ---- State machine ----
type Step = "idle" | "fetching" | "review" | "applying";
const step = ref<Step>("idle");
const fetchError = ref<string | null>(null);

// ---- Review rows ----
type RowStatus = "new" | "changed" | "unchanged" | "nodata";
type PriceField = "input" | "output" | "cachedRead" | "cachedWrite" | "audioInput" | "audioOutput";
type FilterKey = "changes" | "unchanged" | "nodata";

interface Candidate {
  input: number | null;
  output: number | null;
  cachedRead: number | null;
  cachedWrite: number | null;
  audioInput: number | null;
  audioOutput: number | null;
}

interface ReviewRow {
  key: number; // mapping_id
  result: SyncPricingResult;
  status: RowStatus;
  // Classification captured at fetch time. Tabs and counts are driven by
  // this, so editing a row never makes it jump between tabs mid-review.
  initialStatus: RowStatus;
  selected: boolean;
  sourceKey: string | null;
  old: Candidate;
  candidate: Candidate;
}

const rows = ref<ReviewRow[]>([]);
const activeFilter = ref<FilterKey>("changes");
const searchQuery = ref("");

function candidateFrom(r: SyncPricingResult): Candidate {
  return {
    input: r.new_input_cost ?? null,
    output: r.new_output_cost ?? null,
    cachedRead: r.new_cached_read_cost ?? null,
    cachedWrite: r.new_cached_write_cost ?? null,
    audioInput: r.new_audio_input_cost ?? null,
    audioOutput: r.new_audio_output_cost ?? null,
  };
}

function oldFrom(r: SyncPricingResult): Candidate {
  return {
    input: r.old_input_cost ?? null,
    output: r.old_output_cost ?? null,
    cachedRead: r.old_cached_read_cost ?? null,
    cachedWrite: r.old_cached_write_cost ?? null,
    audioInput: r.old_audio_input_cost ?? null,
    audioOutput: r.old_audio_output_cost ?? null,
  };
}

function candidateFromSource(r: SyncPricingResult, sourceKey: string): Candidate | null {
  const opt = r.available_sources.find((o) => o.source === sourceKey);
  if (!opt) return null;
  return {
    input: opt.input_cost_per_1m ?? null,
    output: opt.output_cost_per_1m ?? null,
    cachedRead: opt.cached_read_cost_per_1m ?? null,
    cachedWrite: opt.cached_write_cost_per_1m ?? null,
    audioInput: opt.audio_input_cost_per_1m ?? null,
    audioOutput: opt.audio_output_cost_per_1m ?? null,
  };
}

function samePrice(a: number | null, b: number | null): boolean {
  if (a === null || b === null) return a === b;
  return Math.abs(a - b) < 1e-9;
}

function isUnchanged(old: Candidate, next: Candidate): boolean {
  return (
    samePrice(old.input, next.input) &&
    samePrice(old.output, next.output) &&
    samePrice(old.cachedRead, next.cachedRead) &&
    samePrice(old.cachedWrite, next.cachedWrite) &&
    samePrice(old.audioInput, next.audioInput) &&
    samePrice(old.audioOutput, next.audioOutput)
  );
}

function deriveStatus(r: SyncPricingResult, old: Candidate, next: Candidate): RowStatus {
  if (!r.selected_source || r.available_sources.length === 0) return "nodata";
  if (isUnchanged(old, next)) return "unchanged";
  if (old.input === null && old.output === null) return "new";
  return "changed";
}

/**
 * Infer which source a mapping's stored pricing came from.
 *
 * The backend has no memory of the source used by a previous sync and defaults
 * to the alphabetically-first source, which makes the Source column show an
 * arbitrary value (and can even flag rows as "changed" against a source whose
 * prices were never saved). When the stored prices exactly match one of the
 * available sources, that is almost certainly where they came from — prefer it
 * over the backend default.
 */
function inferSavedSource(r: SyncPricingResult): string | null {
  if (r.old_input_cost === null && r.old_output_cost === null) return null;
  const matches = r.available_sources.filter(
    (o) =>
      samePrice(o.input_cost_per_1m ?? null, r.old_input_cost ?? null) &&
      samePrice(o.output_cost_per_1m ?? null, r.old_output_cost ?? null)
  );
  if (matches.length === 0) return null;
  if (matches.length === 1) return matches[0].source;
  // Input/output collide across sources; disambiguate on the other dimensions.
  const exact = matches.find(
    (o) =>
      samePrice(o.cached_read_cost_per_1m ?? null, r.old_cached_read_cost ?? null) &&
      samePrice(o.cached_write_cost_per_1m ?? null, r.old_cached_write_cost ?? null) &&
      samePrice(o.audio_input_cost_per_1m ?? null, r.old_audio_input_cost ?? null) &&
      samePrice(o.audio_output_cost_per_1m ?? null, r.old_audio_output_cost ?? null)
  );
  return (exact ?? matches[0]).source;
}

function buildRows(response: SyncPricingResponse): ReviewRow[] {
  return response.results.map((r) => {
    const old = oldFrom(r);
    const savedSource = inferSavedSource(r);
    const sourceKey = savedSource ?? r.selected_source ?? null;
    // Recompute the candidate from the inferred source so the shown prices
    // always belong to the source displayed in the Source column.
    const candidate = (savedSource && candidateFromSource(r, savedSource)) || candidateFrom(r);
    const status = deriveStatus({ ...r, selected_source: sourceKey }, old, candidate);
    return {
      key: r.mapping_id,
      result: r,
      status,
      initialStatus: status,
      // Auto-select only NEW rows: filling gaps is safe, changing existing
      // prices always requires explicit opt-in.
      selected: status === "new",
      sourceKey,
      old,
      candidate,
    };
  });
}

// ---- Fetch ----
async function fetchPreview() {
  step.value = "fetching";
  fetchError.value = null;
  try {
    const response = await pricingApi.fetchPreview();
    if (!response.success) {
      fetchError.value = response.error ?? t("models.pricingSync.fetchFailed");
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
  const next = candidateFromSource(row.result, source);
  if (!next) return;
  row.sourceKey = source;
  row.candidate = next;
  // Recompute status against the new candidate. Selection is left to the
  // user — a row that resolves back to "unchanged" simply contributes no
  // diff (and thus nothing) when applied.
  row.status = deriveStatus({ ...row.result, selected_source: source }, row.old, next);
  editingCell.value = null;
}

// ---- Inline editing ----
const editingCell = ref<{ key: number; field: "input" | "output" } | null>(null);
const editingValue = ref("");

function startEdit(row: ReviewRow, field: "input" | "output") {
  editingCell.value = { key: row.key, field };
  const v = row.candidate[field];
  editingValue.value = v === null ? "" : String(v);
}

function commitEdit(row: ReviewRow) {
  if (!editingCell.value) return;
  const raw = editingValue.value.trim();
  const parsed = raw === "" ? null : Number(raw);
  if (parsed === null || (Number.isFinite(parsed) && parsed >= 0)) {
    row.candidate[editingCell.value.field] = parsed;
    row.status = deriveStatus(
      { ...row.result, selected_source: row.sourceKey },
      row.old,
      row.candidate
    );
  } else {
    // Revert to the current candidate value to signal rejection
    const current = row.candidate[editingCell.value.field];
    editingValue.value = current === null ? "" : String(current);
    return; // keep editingCell open so user can correct
  }
  editingCell.value = null;
}

function cancelEdit() {
  editingCell.value = null;
}

// ---- Filters & tabs ----
function isActionable(row: ReviewRow): boolean {
  // Any row with models.dev data can be selected and edited. "Up to date"
  // rows are intentionally editable too — switch source or override a price
  // to force an update. Only "no data" rows (no source match) are locked.
  return row.status !== "nodata";
}

function rowHasDiff(row: ReviewRow): boolean {
  return (
    !samePrice(row.candidate.input, row.old.input) ||
    !samePrice(row.candidate.output, row.old.output) ||
    !samePrice(row.candidate.cachedRead, row.old.cachedRead) ||
    !samePrice(row.candidate.cachedWrite, row.old.cachedWrite) ||
    !samePrice(row.candidate.audioInput, row.old.audioInput) ||
    !samePrice(row.candidate.audioOutput, row.old.audioOutput)
  );
}

function matchesSearch(row: ReviewRow): boolean {
  const q = searchQuery.value.trim().toLowerCase();
  if (!q) return true;
  return (
    row.result.model_name.toLowerCase().includes(q) ||
    row.result.provider.toLowerCase().includes(q) ||
    row.result.provider_model_name.toLowerCase().includes(q)
  );
}

// Tab membership and tab counts use `initialStatus` (the classification at
// fetch time), so editing/switching a row never moves it out from under the
// user mid-review. Live `status` still drives the badge shown on the row.
const actionableRows = computed(() =>
  rows.value
    .filter((r) => r.initialStatus === "new" || r.initialStatus === "changed")
    .sort((a, b) => (a.initialStatus === b.initialStatus ? 0 : a.initialStatus === "new" ? -1 : 1))
);
const unchangedRows = computed(() => rows.value.filter((r) => r.initialStatus === "unchanged"));
const noDataRows = computed(() => rows.value.filter((r) => r.initialStatus === "nodata"));

const counts = computed(() => ({
  new: rows.value.filter((r) => r.initialStatus === "new").length,
  changed: rows.value.filter((r) => r.initialStatus === "changed").length,
  unchanged: unchangedRows.value.length,
  nodata: noDataRows.value.length,
}));

const actionableCount = computed(() => counts.value.new + counts.value.changed);
const totalActionable = computed(() => rows.value.filter(isActionable).length);

interface FilterTab {
  key: FilterKey;
  label: string;
  count: number;
  visible: boolean;
}

const filterTabs = computed<FilterTab[]>(() => [
  {
    key: "changes",
    label: t("models.pricingSync.tabChanges"),
    count: actionableCount.value,
    visible: true,
  },
  {
    key: "unchanged",
    label: t("models.pricingSync.tabUnchanged"),
    count: counts.value.unchanged,
    visible: true,
  },
  {
    key: "nodata",
    label: t("models.pricingSync.tabNoData"),
    count: counts.value.nodata,
    visible: counts.value.nodata > 0,
  },
]);

const displayedRows = computed(() => {
  const base =
    activeFilter.value === "changes"
      ? actionableRows.value
      : activeFilter.value === "unchanged"
        ? unchangedRows.value
        : noDataRows.value;
  return base.filter(matchesSearch);
});

const visibleActionable = computed(() => displayedRows.value.filter(isActionable));

const emptyStateMessage = computed(() => {
  if (actionableCount.value === 0 && activeFilter.value === "changes") {
    return {
      title: t("models.pricingSync.upToDate"),
      description: t("models.pricingSync.upToDateDescription"),
    };
  }
  if (activeFilter.value === "unchanged") {
    return {
      title: t("models.pricingSync.emptyUnchanged"),
      description: t("models.pricingSync.emptyUnchangedDescription"),
    };
  }
  return {
    title: t("models.pricingSync.emptyNoData"),
    description: t("models.pricingSync.emptyNoDataDescription"),
  };
});

// ---- Selection ----
const selectedCount = computed(() => rows.value.filter((r) => r.selected).length);
// Rows that will actually be written when applied: selected AND carrying a
// real diff vs. the stored values. Selecting an "Up to date" row without
// editing it is harmless but contributes nothing here.
const applicableCount = computed(
  () => rows.value.filter((r) => r.selected && rowHasDiff(r)).length
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
function buildUpdates(): PricingUpdateItem[] {
  const updates: PricingUpdateItem[] = [];
  for (const row of rows.value) {
    if (!row.selected) continue;
    // Diff-driven payload: only send fields whose value actually changes,
    // so untouched dimensions (e.g. manually set cache pricing) are never
    // clobbered by a missing models.dev value.
    const item: PricingUpdateItem = { mapping_id: row.key };
    if (!samePrice(row.candidate.input, row.old.input))
      item.input_cost_per_1m = row.candidate.input;
    if (!samePrice(row.candidate.output, row.old.output))
      item.output_cost_per_1m = row.candidate.output;
    if (!samePrice(row.candidate.cachedRead, row.old.cachedRead))
      item.cached_read_cost_per_1m = row.candidate.cachedRead;
    if (!samePrice(row.candidate.cachedWrite, row.old.cachedWrite))
      item.cached_write_cost_per_1m = row.candidate.cachedWrite;
    if (!samePrice(row.candidate.audioInput, row.old.audioInput))
      item.audio_input_cost_per_1m = row.candidate.audioInput;
    if (!samePrice(row.candidate.audioOutput, row.old.audioOutput))
      item.audio_output_cost_per_1m = row.candidate.audioOutput;
    // Skip rows where the user reverted everything back to current values.
    const hasChange =
      "input_cost_per_1m" in item ||
      "output_cost_per_1m" in item ||
      "cached_read_cost_per_1m" in item ||
      "cached_write_cost_per_1m" in item ||
      "audio_input_cost_per_1m" in item ||
      "audio_output_cost_per_1m" in item;
    if (hasChange) updates.push(item);
  }
  return updates;
}

async function applySelected() {
  const updates = buildUpdates();
  if (updates.length === 0) return;
  step.value = "applying";
  try {
    const response = await pricingApi.applyPricing({ updates });
    if (response.applied_count > 0) {
      toast.success(t("models.pricingSync.applied", { count: response.applied_count }));
    }
    if (response.failed_count > 0) {
      toast.error(t("models.pricingSync.applyFailed", { count: response.failed_count }));
      step.value = "review";
      return;
    }
    emit("applied");
    close();
  } catch (error) {
    handleError(error);
    step.value = "review";
  }
}

// ---- Dialog lifecycle ----
function close() {
  emit("update:open", false);
}

watch(
  () => props.open,
  (open) => {
    if (open) {
      step.value = "idle";
      fetchError.value = null;
      rows.value = [];
      activeFilter.value = "changes";
      searchQuery.value = "";
      editingCell.value = null;
      // Fetch immediately — the review table is the point of the dialog,
      // an extra confirmation click only adds friction.
      fetchPreview();
    }
  }
);

// ---- Formatting ----
function fmt(v: number | null): string {
  if (v === null) return "—";
  return v.toLocaleString("en-US", { maximumFractionDigits: 4 });
}

function pctChange(oldV: number | null, newV: number | null): string | null {
  if (oldV === null || newV === null || oldV === 0) return null;
  const pct = ((newV - oldV) / oldV) * 100;
  if (Math.abs(pct) < 0.05) return null;
  return `${pct > 0 ? "+" : ""}${pct.toFixed(0)}%`;
}

interface ExtraChip {
  label: string;
  oldV: number | null;
  newV: number | null;
}

function extraChips(row: ReviewRow): ExtraChip[] {
  const chips: ExtraChip[] = [];
  const dims: Array<{ label: string; field: Exclude<PriceField, "input" | "output"> }> = [
    { label: "cache r", field: "cachedRead" },
    { label: "cache w", field: "cachedWrite" },
    { label: "audio i", field: "audioInput" },
    { label: "audio o", field: "audioOutput" },
  ];
  for (const d of dims) {
    const newV = row.candidate[d.field];
    const oldV = row.old[d.field];
    if (newV === null && oldV === null) continue;
    chips.push({ label: d.label, oldV, newV });
  }
  return chips;
}

const hasRows = computed(() => displayedRows.value.length > 0);
</script>

<template>
  <Dialog :open="open" @update:open="emit('update:open', $event)">
    <DialogContent class="flex h-[86vh] w-[96vw] max-w-[96vw] flex-col gap-0 p-0 sm:max-w-[1500px]">
      <DialogHeader class="px-6 pt-6 pb-4 border-b border-border">
        <DialogTitle>{{ t("models.pricingSync.title") }}</DialogTitle>
        <DialogDescription>{{ t("models.pricingSync.subtitle") }}</DialogDescription>
      </DialogHeader>

      <!-- Step: idle — only reachable after a fetch error; offer retry -->
      <div
        v-if="step === 'idle'"
        class="flex-1 flex flex-col items-center justify-center gap-4 py-16 text-center px-6"
      >
        <p class="text-sm text-muted-foreground max-w-md">
          {{ t("models.pricingSync.idleDescription") }}
        </p>
        <p v-if="fetchError" class="text-sm text-status-error">{{ fetchError }}</p>
        <Button class="btn-action" @click="fetchPreview">
          <RefreshCw class="w-4 h-4 mr-2" />
          {{ t("models.pricingSync.retry") }}
        </Button>
      </div>

      <!-- Step: fetching -->
      <div
        v-else-if="step === 'fetching'"
        class="flex-1 flex items-center justify-center gap-3 py-20"
      >
        <RefreshCw class="w-4 h-4 animate-spin text-muted-foreground" />
        <span class="text-sm text-muted-foreground">{{ t("models.pricingSync.fetching") }}</span>
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
          <Table v-if="hasRows" class="w-max min-w-full">
            <TableHeader class="sticky top-0 z-10">
              <TableRow class="hover:bg-transparent hover:border-l-transparent">
                <TableHead class="w-10">
                  <Checkbox
                    v-if="visibleActionable.length > 0"
                    :model-value="
                      allVisibleSelected ? true : someVisibleSelected ? 'indeterminate' : false
                    "
                    :aria-label="t('models.pricingSync.selectAll')"
                    @update:model-value="onSelectAll"
                  />
                </TableHead>
                <TableHead class="min-w-[220px]">{{ t("models.pricingSync.colModel") }}</TableHead>
                <TableHead class="w-40">{{ t("models.pricingSync.colSource") }}</TableHead>
                <TableHead class="w-44 text-right">{{
                  t("models.pricingSync.colInput")
                }}</TableHead>
                <TableHead class="w-44 text-right">{{
                  t("models.pricingSync.colOutput")
                }}</TableHead>
                <TableHead class="min-w-[200px]">{{ t("models.pricingSync.colExtra") }}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              <TableRow
                v-for="row in displayedRows"
                :key="row.key"
                :data-state="row.selected ? 'selected' : undefined"
              >
                <TableCell>
                  <Checkbox
                    v-if="step !== 'applying' && isActionable(row)"
                    v-model="row.selected"
                    :aria-label="row.result.model_name"
                  />
                </TableCell>
                <TableCell>
                  <div class="flex items-center gap-2">
                    <div class="min-w-0">
                      <div class="font-medium truncate">{{ row.result.model_name }}</div>
                      <div class="text-xs text-muted-foreground truncate">
                        {{ row.result.provider }} · {{ row.result.provider_model_name }}
                      </div>
                    </div>
                    <Badge
                      v-if="row.status === 'new'"
                      variant="outline"
                      class="shrink-0 text-status-success border-status-success/40"
                    >
                      {{ t("models.pricingSync.badgeNew") }}
                    </Badge>
                    <Badge
                      v-else-if="row.status === 'changed'"
                      variant="outline"
                      class="shrink-0 text-action-amber border-action-amber/40"
                    >
                      {{ t("models.pricingSync.badgeChanged") }}
                    </Badge>
                    <Badge
                      v-else-if="row.status === 'nodata'"
                      variant="outline"
                      class="shrink-0 text-muted-foreground border-border"
                    >
                      {{ t("models.pricingSync.badgeNoData") }}
                    </Badge>
                  </div>
                </TableCell>
                <TableCell>
                  <Select
                    v-if="
                      step !== 'applying' &&
                      isActionable(row) &&
                      row.result.available_sources.length > 1
                    "
                    :model-value="row.sourceKey ?? undefined"
                    @update:model-value="(v) => onSourceChange(row, String(v))"
                  >
                    <SelectTrigger class="h-8 w-full text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem
                        v-for="opt in row.result.available_sources"
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
                <!-- Input price: old → new (click new value to edit) -->
                <TableCell class="text-right">
                  <div class="flex items-center justify-end gap-1.5 text-data">
                    <span class="text-muted-foreground">{{ fmt(row.old.input) }}</span>
                    <span class="text-muted-foreground">→</span>
                    <input
                      v-if="
                        step !== 'applying' &&
                        isActionable(row) &&
                        editingCell?.key === row.key &&
                        editingCell.field === 'input'
                      "
                      v-model="editingValue"
                      type="number"
                      step="any"
                      min="0"
                      class="w-20 h-7 px-1.5 text-right text-data bg-background border border-ring rounded outline-none"
                      autofocus
                      @blur="commitEdit(row)"
                      @keydown.enter="commitEdit(row)"
                      @keydown.esc="cancelEdit"
                    />
                    <button
                      v-else-if="step !== 'applying' && isActionable(row)"
                      type="button"
                      class="hover:bg-muted rounded px-1 -mx-1 cursor-text"
                      :class="
                        samePrice(row.candidate.input, row.old.input)
                          ? ''
                          : row.candidate.input !== null &&
                              row.old.input !== null &&
                              row.candidate.input > row.old.input
                            ? 'text-action-amber'
                            : 'text-status-success'
                      "
                      @click="startEdit(row, 'input')"
                    >
                      {{ fmt(row.candidate.input) }}
                    </button>
                    <span v-else class="text-muted-foreground">{{ fmt(row.old.input) }}</span>
                    <span
                      v-if="isActionable(row) && pctChange(row.old.input, row.candidate.input)"
                      class="text-xs"
                      :class="
                        (row.candidate.input ?? 0) > (row.old.input ?? 0)
                          ? 'text-action-amber'
                          : 'text-status-success'
                      "
                    >
                      {{ pctChange(row.old.input, row.candidate.input) }}
                    </span>
                  </div>
                </TableCell>
                <!-- Output price -->
                <TableCell class="text-right">
                  <div class="flex items-center justify-end gap-1.5 text-data">
                    <span class="text-muted-foreground">{{ fmt(row.old.output) }}</span>
                    <span class="text-muted-foreground">→</span>
                    <input
                      v-if="
                        step !== 'applying' &&
                        isActionable(row) &&
                        editingCell?.key === row.key &&
                        editingCell.field === 'output'
                      "
                      v-model="editingValue"
                      type="number"
                      step="any"
                      min="0"
                      class="w-20 h-7 px-1.5 text-right text-data bg-background border border-ring rounded outline-none"
                      autofocus
                      @blur="commitEdit(row)"
                      @keydown.enter="commitEdit(row)"
                      @keydown.esc="cancelEdit"
                    />
                    <button
                      v-else-if="step !== 'applying' && isActionable(row)"
                      type="button"
                      class="hover:bg-muted rounded px-1 -mx-1 cursor-text"
                      :class="
                        samePrice(row.candidate.output, row.old.output)
                          ? ''
                          : row.candidate.output !== null &&
                              row.old.output !== null &&
                              row.candidate.output > row.old.output
                            ? 'text-action-amber'
                            : 'text-status-success'
                      "
                      @click="startEdit(row, 'output')"
                    >
                      {{ fmt(row.candidate.output) }}
                    </button>
                    <span v-else class="text-muted-foreground">{{ fmt(row.old.output) }}</span>
                    <span
                      v-if="isActionable(row) && pctChange(row.old.output, row.candidate.output)"
                      class="text-xs"
                      :class="
                        (row.candidate.output ?? 0) > (row.old.output ?? 0)
                          ? 'text-action-amber'
                          : 'text-status-success'
                      "
                    >
                      {{ pctChange(row.old.output, row.candidate.output) }}
                    </span>
                  </div>
                </TableCell>
                <!-- Extra dimensions as chips -->
                <TableCell>
                  <div class="flex flex-wrap gap-1">
                    <span
                      v-for="chip in extraChips(row)"
                      :key="chip.label"
                      class="inline-flex items-center gap-1 rounded border border-border px-1.5 py-0.5 text-[11px] text-data"
                      :class="samePrice(chip.oldV, chip.newV) ? 'text-muted-foreground' : ''"
                    >
                      <span class="text-muted-foreground">{{ chip.label }}</span>
                      <template v-if="!samePrice(chip.oldV, chip.newV)">
                        <span class="text-muted-foreground">{{ fmt(chip.oldV) }}</span>
                        <span class="text-muted-foreground">→</span>
                      </template>
                      <span>{{ fmt(chip.newV) }}</span>
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
                t("models.pricingSync.selectionSummary", {
                  selected: selectedCount,
                  applicable: applicableCount,
                })
              }}
            </span>
            <span v-if="selectedWithoutChange > 0" class="text-status-warning">
              {{ t("models.pricingSync.noChangeHint") }}
            </span>
            <template v-if="totalActionable > 0">
              <span>·</span>
              <button
                type="button"
                class="underline-offset-2 hover:underline hover:text-foreground transition-colors"
                @click="selectAllActionable"
              >
                {{ t("models.pricingSync.selectAllActionable") }}
              </button>
              <span>·</span>
              <button
                type="button"
                class="underline-offset-2 hover:underline hover:text-foreground transition-colors"
                @click="clearSelection"
              >
                {{ t("models.pricingSync.clearSelection") }}
              </button>
            </template>
            <span class="ml-auto">{{ t("models.pricingSync.reviewHint") }}</span>
          </div>
          <DialogFooter class="gap-2 sm:gap-2">
            <Button variant="outline" :disabled="step === 'applying'" @click="close">
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
                  ? t("models.pricingSync.applying")
                  : t("models.pricingSync.apply", { count: applicableCount })
              }}
            </Button>
          </DialogFooter>
        </div>
      </template>
    </DialogContent>
  </Dialog>
</template>
