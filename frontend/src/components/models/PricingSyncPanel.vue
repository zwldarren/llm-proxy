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
import { pricingApi } from "@/services/api/config";
import { TIER_RATE_KEYS, tierThresholdLabel } from "@/utils/pricingTiers";

import type { SyncSectionState } from "./syncSection";
import type {
  PricingTier,
  PricingUpdateItem,
  ServiceTierPricing,
  ServiceTierRates,
  SyncPricingResult,
  SyncPricingResponse,
} from "@/types/schemas";

/**
 * Pricing review section of the models.dev sync wizard.
 *
 * Owns its own preview fetch, rows, source switching and selection, and
 * exposes them through `summary`; the wizard reads that to show the overview,
 * gate the steps and collect `buildUpdates()` for the single apply. The panel
 * never writes anything itself.
 */
defineOptions({ name: "PricingSyncPanel" });

const { t } = useI18n();

// ---- Fetch state ----
const status = ref<"idle" | "loading" | "ready" | "error">("idle");
const fetchError = ref<string | null>(null);

// ---- Review rows ----
type RowStatus = "new" | "changed" | "unchanged" | "nodata";
type PriceField = "input" | "output" | "reasoning" | "audioInput" | "audioOutput";
type ExtraField = PriceField | "cachedRead" | "cachedWrite";
type FilterKey = "changes" | "unchanged" | "nodata";

interface Candidate {
  input: number | null;
  output: number | null;
  cachedRead: number | null;
  cachedWrite: number | null;
  reasoning: number | null;
  audioInput: number | null;
  audioOutput: number | null;
  tiers: PricingTier[];
  serviceTiers: ServiceTierPricing;
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

/** Rate dimensions shown as chips, with their short labels. */
const EXTRA_DIMS: { field: ExtraField; labelKey: string }[] = [
  { field: "cachedRead", labelKey: "models.pricingSync.dimCacheRead" },
  { field: "cachedWrite", labelKey: "models.pricingSync.dimCacheWrite" },
  { field: "reasoning", labelKey: "models.pricingSync.dimReasoning" },
  { field: "audioInput", labelKey: "models.pricingSync.dimAudioInput" },
  { field: "audioOutput", labelKey: "models.pricingSync.dimAudioOutput" },
];

/** Dimensions listed in the confirmation step, in billing order. */
const rows = ref<ReviewRow[]>([]);
const activeFilter = ref<FilterKey>("changes");
const searchQuery = ref("");

function candidateFrom(r: SyncPricingResult): Candidate {
  return {
    input: r.new_input_cost ?? null,
    output: r.new_output_cost ?? null,
    cachedRead: r.new_cached_read_cost ?? null,
    cachedWrite: r.new_cached_write_cost ?? null,
    reasoning: r.new_reasoning_cost ?? null,
    audioInput: r.new_audio_input_cost ?? null,
    audioOutput: r.new_audio_output_cost ?? null,
    tiers: r.new_pricing_tiers ?? [],
    serviceTiers: r.new_service_tier_pricing ?? {},
  };
}

function oldFrom(r: SyncPricingResult): Candidate {
  return {
    input: r.old_input_cost ?? null,
    output: r.old_output_cost ?? null,
    cachedRead: r.old_cached_read_cost ?? null,
    cachedWrite: r.old_cached_write_cost ?? null,
    reasoning: r.old_reasoning_cost ?? null,
    audioInput: r.old_audio_input_cost ?? null,
    audioOutput: r.old_audio_output_cost ?? null,
    tiers: r.old_pricing_tiers ?? [],
    serviceTiers: r.old_service_tier_pricing ?? {},
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
    reasoning: opt.reasoning_cost_per_1m ?? null,
    audioInput: opt.audio_input_cost_per_1m ?? null,
    audioOutput: opt.audio_output_cost_per_1m ?? null,
    tiers: opt.tiers ?? [],
    serviceTiers: opt.service_tier_pricing ?? {},
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
    samePrice(old.reasoning, next.reasoning) &&
    samePrice(old.audioInput, next.audioInput) &&
    samePrice(old.audioOutput, next.audioOutput) &&
    tiersEqual(old.tiers, next.tiers) &&
    serviceTiersEqual(old.serviceTiers, next.serviceTiers)
  );
}

/** Order-insensitive equality over every tier dimension, for diff detection. */
function tiersEqual(a: PricingTier[], b: PricingTier[]): boolean {
  if (a.length !== b.length) return false;
  const normalize = (tiers: PricingTier[]) =>
    [...tiers]
      .map((tier) => ({
        threshold: tier.threshold,
        rates: TIER_RATE_KEYS.map((key) => tier[key] ?? null),
      }))
      .sort((x, y) => x.threshold - y.threshold);
  const left = normalize(a);
  const right = normalize(b);
  return left.every(
    (tier, i) =>
      tier.threshold === right[i]!.threshold &&
      tier.rates.every((rate, j) => samePrice(rate, right[i]!.rates[j]!))
  );
}

/** Order-insensitive equality over every service-tier rate, for diff detection. */
function serviceTiersEqual(a: ServiceTierPricing, b: ServiceTierPricing): boolean {
  const left = Object.keys(a).sort();
  const right = Object.keys(b).sort();
  if (left.length !== right.length || left.some((key, i) => key !== right[i])) return false;
  return left.every((key) =>
    TIER_RATE_KEYS.every((field) => samePrice(a[key]?.[field] ?? null, b[key]?.[field] ?? null))
  );
}

function deriveStatus(r: SyncPricingResult, old: Candidate, next: Candidate): RowStatus {
  if (!r.selected_source || r.available_sources.length === 0) return "nodata";
  if (isUnchanged(old, next)) return "unchanged";
  if (
    old.input === null &&
    old.output === null &&
    old.tiers.length === 0 &&
    old.reasoning === null &&
    Object.keys(old.serviceTiers).length === 0
  ) {
    return "new";
  }
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
  status.value = "loading";
  fetchError.value = null;
  try {
    const response = await pricingApi.fetchPreview();
    if (!response.success) {
      fetchError.value = response.error ?? t("models.pricingSync.fetchFailed");
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
    !samePrice(row.candidate.reasoning, row.old.reasoning) ||
    !samePrice(row.candidate.audioInput, row.old.audioInput) ||
    !samePrice(row.candidate.audioOutput, row.old.audioOutput) ||
    !tiersEqual(row.candidate.tiers, row.old.tiers) ||
    !serviceTiersEqual(row.candidate.serviceTiers, row.old.serviceTiers)
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

interface FilterOption {
  key: FilterKey;
  label: string;
  count: number;
  visible: boolean;
}

const filterOptions = computed<FilterOption[]>(() => [
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

const selectableFilters = computed(() => filterOptions.value.filter((option) => option.visible));

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
// Rows that will actually be written when applied: selected AND carrying a
// real diff vs. the stored values. Selecting an "Up to date" row without
// editing it is harmless but contributes nothing here.
const applicableCount = computed(
  () => rows.value.filter((r) => r.selected && rowHasDiff(r)).length
);
// A "changed" row replaces rates you already store, so it is never
// preselected. The dialog reports how many of those are still unchecked; that
// number is the visible half of the preselect rule.
const blockedCount = computed(
  () => rows.value.filter((r) => !r.selected && r.status === "changed" && rowHasDiff(r)).length
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
    if (!samePrice(row.candidate.reasoning, row.old.reasoning))
      item.reasoning_cost_per_1m = row.candidate.reasoning;
    if (!samePrice(row.candidate.audioInput, row.old.audioInput))
      item.audio_input_cost_per_1m = row.candidate.audioInput;
    if (!samePrice(row.candidate.audioOutput, row.old.audioOutput))
      item.audio_output_cost_per_1m = row.candidate.audioOutput;
    if (!tiersEqual(row.candidate.tiers, row.old.tiers)) item.pricing_tiers = row.candidate.tiers;
    if (!serviceTiersEqual(row.candidate.serviceTiers, row.old.serviceTiers))
      item.service_tier_pricing = row.candidate.serviceTiers;
    // Skip rows where the user reverted everything back to current values.
    const hasChange =
      "input_cost_per_1m" in item ||
      "output_cost_per_1m" in item ||
      "cached_read_cost_per_1m" in item ||
      "cached_write_cost_per_1m" in item ||
      "reasoning_cost_per_1m" in item ||
      "audio_input_cost_per_1m" in item ||
      "audio_output_cost_per_1m" in item ||
      "pricing_tiers" in item ||
      "service_tier_pricing" in item;
    if (hasChange) updates.push(item);
  }
  return updates;
}

// Named rows only: the confirmation bar says how many, and these lines say
// which. The diffs themselves stay in the table, one click away.
const pending = computed<string[]>(() =>
  rows.value
    .filter((row) => row.selected && rowHasDiff(row))
    .map((row) => `${row.result.model_name} · ${row.result.provider}`)
);

const summary = computed<SyncSectionState>(() => ({
  status: status.value,
  error: fetchError.value,
  changedCount: actionableCount.value,
  applicableCount: applicableCount.value,
  blockedCount: blockedCount.value,
  pending: pending.value,
}));

defineExpose({ fetchPreview, buildUpdates, summary });

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
  for (const dim of EXTRA_DIMS) {
    const newV = row.candidate[dim.field];
    const oldV = row.old[dim.field];
    if (newV === null && oldV === null) continue;
    chips.push({ label: t(dim.labelKey), oldV, newV });
  }
  return chips;
}

interface TierChip {
  key: number;
  label: string;
  oldText: string | null;
  newText: string;
  changed: boolean;
}

function fmtTierRates(tier: PricingTier | undefined): string | null {
  if (!tier) return null;
  return `${fmt(tier.input_cost_per_1m ?? null)} / ${fmt(tier.output_cost_per_1m ?? null)}`;
}

/** One line per threshold seen on either side, marking changed bands. */
function tierChips(row: ReviewRow): TierChip[] {
  const oldByThreshold = new Map(row.old.tiers.map((tier) => [tier.threshold, tier]));
  const newByThreshold = new Map(row.candidate.tiers.map((tier) => [tier.threshold, tier]));
  const thresholds = [...new Set([...oldByThreshold.keys(), ...newByThreshold.keys()])].sort(
    (a, b) => a - b
  );
  return thresholds.map((threshold) => {
    const oldTier = oldByThreshold.get(threshold);
    const newTier = newByThreshold.get(threshold);
    const changed =
      !oldTier ||
      !newTier ||
      TIER_RATE_KEYS.some((key) => !samePrice(oldTier[key] ?? null, newTier[key] ?? null));
    return {
      key: threshold,
      label: tierThresholdLabel(t, threshold),
      oldText: fmtTierRates(oldTier),
      newText: fmtTierRates(newTier) ?? "—",
      changed,
    };
  });
}

interface ServiceTierChip {
  name: string;
  oldText: string | null;
  newText: string | null;
  changed: boolean;
}

function fmtServiceTierRates(rates: ServiceTierRates | undefined): string {
  return `${fmt(rates?.input_cost_per_1m ?? null)} / ${fmt(rates?.output_cost_per_1m ?? null)}`;
}

/** One chip per mode seen on either side, marking the ones whose rates moved. */
function serviceTierChips(row: ReviewRow): ServiceTierChip[] {
  const names = [
    ...new Set([...Object.keys(row.old.serviceTiers), ...Object.keys(row.candidate.serviceTiers)]),
  ].sort();
  return names.map((name) => {
    const oldRates = row.old.serviceTiers[name];
    const newRates = row.candidate.serviceTiers[name];
    const changed =
      !oldRates ||
      !newRates ||
      TIER_RATE_KEYS.some((field) => !samePrice(oldRates[field] ?? null, newRates[field] ?? null));
    return {
      name,
      oldText: oldRates ? fmtServiceTierRates(oldRates) : null,
      newText: newRates ? fmtServiceTierRates(newRates) : null,
      changed,
    };
  });
}

const hasRows = computed(() => displayedRows.value.length > 0);
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
          {{ fetchError ?? t("models.pricingSync.fetchFailed") }}
        </p>
        <Button variant="outline" size="sm" @click="fetchPreview">
          <RefreshCw class="h-3.5 w-3.5" />
          {{ t("common.retry") }}
        </Button>
      </template>
      <template v-else>
        <RefreshCw class="h-4 w-4 animate-spin text-muted-foreground" />
        <span class="text-sm text-muted-foreground">{{ t("models.pricingSync.fetching") }}</span>
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
        <Table v-if="hasRows" class="w-max min-w-full">
          <TableHeader class="sticky top-0 z-10">
            <TableRow class="hover:border-l-transparent hover:bg-transparent">
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
              <TableHead class="w-44 text-right">{{ t("models.pricingSync.colInput") }}</TableHead>
              <TableHead class="w-44 text-right">{{ t("models.pricingSync.colOutput") }}</TableHead>
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
                  v-if="isActionable(row)"
                  v-model="row.selected"
                  :aria-label="row.result.model_name"
                />
              </TableCell>
              <TableCell>
                <div class="flex items-center gap-2">
                  <div class="min-w-0">
                    <div class="truncate font-medium">{{ row.result.model_name }}</div>
                    <div class="truncate text-xs text-muted-foreground">
                      {{ row.result.provider }} · {{ row.result.provider_model_name }}
                    </div>
                  </div>
                  <Badge
                    v-if="row.status === 'new'"
                    variant="outline"
                    class="shrink-0 border-status-success/40 text-status-success"
                  >
                    {{ t("models.pricingSync.badgeNew") }}
                  </Badge>
                  <Badge
                    v-else-if="row.status === 'changed'"
                    variant="outline"
                    class="shrink-0 border-action-amber/40 text-action-amber"
                  >
                    {{ t("models.pricingSync.badgeOverwrite") }}
                  </Badge>
                  <Badge
                    v-else-if="row.status === 'nodata'"
                    variant="outline"
                    class="shrink-0 border-border text-muted-foreground"
                  >
                    {{ t("models.pricingSync.badgeNoData") }}
                  </Badge>
                </div>
              </TableCell>
              <TableCell>
                <Select
                  v-if="isActionable(row) && row.result.available_sources.length > 1"
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
                  <!-- The arrow only appears when the rate actually moves; an
                       unchanged "3 → 3" is noise in a diff table. -->
                  <template v-if="!samePrice(row.candidate.input, row.old.input)">
                    <span class="text-muted-foreground">{{ fmt(row.old.input) }}</span>
                    <span class="text-muted-foreground">→</span>
                  </template>
                  <input
                    v-if="
                      isActionable(row) &&
                      editingCell?.key === row.key &&
                      editingCell.field === 'input'
                    "
                    v-model="editingValue"
                    type="number"
                    step="any"
                    min="0"
                    class="h-7 w-20 rounded border border-ring bg-background px-1.5 text-right text-data outline-none"
                    autofocus
                    @blur="commitEdit(row)"
                    @keydown.enter="commitEdit(row)"
                    @keydown.esc="cancelEdit"
                  />
                  <button
                    v-else-if="isActionable(row)"
                    type="button"
                    class="-mx-1 cursor-text rounded px-1 hover:bg-muted"
                    :class="
                      samePrice(row.candidate.input, row.old.input)
                        ? 'text-muted-foreground'
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
                  <template v-if="!samePrice(row.candidate.output, row.old.output)">
                    <span class="text-muted-foreground">{{ fmt(row.old.output) }}</span>
                    <span class="text-muted-foreground">→</span>
                  </template>
                  <input
                    v-if="
                      isActionable(row) &&
                      editingCell?.key === row.key &&
                      editingCell.field === 'output'
                    "
                    v-model="editingValue"
                    type="number"
                    step="any"
                    min="0"
                    class="h-7 w-20 rounded border border-ring bg-background px-1.5 text-right text-data outline-none"
                    autofocus
                    @blur="commitEdit(row)"
                    @keydown.enter="commitEdit(row)"
                    @keydown.esc="cancelEdit"
                  />
                  <button
                    v-else-if="isActionable(row)"
                    type="button"
                    class="-mx-1 cursor-text rounded px-1 hover:bg-muted"
                    :class="
                      samePrice(row.candidate.output, row.old.output)
                        ? 'text-muted-foreground'
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
                <div v-if="tierChips(row).length" class="mt-1 space-y-0.5">
                  <div
                    v-for="tier in tierChips(row)"
                    :key="tier.key"
                    class="flex items-center gap-1 text-[11px] text-data"
                  >
                    <span class="text-muted-foreground">
                      {{ t("models.pricingSync.tierChip") }} {{ tier.label }}
                    </span>
                    <template v-if="tier.changed && tier.oldText">
                      <span class="text-muted-foreground">{{ tier.oldText }}</span>
                      <span class="text-muted-foreground">→</span>
                    </template>
                    <span :class="tier.changed ? 'text-action-amber' : ''">{{ tier.newText }}</span>
                  </div>
                </div>
                <div v-if="serviceTierChips(row).length" class="mt-1 space-y-0.5">
                  <div
                    v-for="mode in serviceTierChips(row)"
                    :key="mode.name"
                    class="flex items-center gap-1 text-[11px] text-data"
                  >
                    <span class="text-muted-foreground">
                      {{ t("models.pricingSync.serviceTierChip") }} {{ mode.name }}
                    </span>
                    <template v-if="mode.changed && mode.oldText">
                      <span class="text-muted-foreground">{{ mode.oldText }}</span>
                      <span class="text-muted-foreground">→</span>
                    </template>
                    <span :class="mode.changed ? 'text-action-amber' : ''">
                      {{ mode.newText ?? "—" }}
                    </span>
                  </div>
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
