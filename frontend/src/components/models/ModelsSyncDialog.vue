<script setup lang="ts">
import { ArrowRight, CircleDollarSign, RefreshCw, SlidersHorizontal } from "@lucide/vue";
import { computed, ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import { toast } from "vue-sonner";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { metadataApi, pricingApi } from "@/services/api/config";
import CapabilitySyncPanel from "./CapabilitySyncPanel.vue";
import PricingSyncPanel from "./PricingSyncPanel.vue";
import { EMPTY_SECTION_STATE } from "./syncSection";

import type { SyncSectionKey } from "./syncSection";

/**
 * models.dev sync, on one page.
 *
 * Two independent field sets (pricing per provider mapping, capabilities per
 * model) share one dialog: tab-switched review sections over a single bottom
 * bar, which shows what apply would write and expands in place into the final
 * confirmation. There is one navigation axis (the section tabs) and one write
 * action, so the same counts are never presented twice.
 *
 * The dialog owns the flow and the write; the sections own their rows and
 * expose `summary` / `buildUpdates()` (see ./syncSection). Both stay mounted
 * (v-show, never v-if) so switching sections never refetches and the bottom
 * bar can read selections from a section that is not on screen.
 */
defineOptions({ name: "ModelsSyncDialog" });

defineProps<{ open: boolean }>();
const emit = defineEmits<{
  "update:open": [value: boolean];
  applied: [];
}>();

const { t } = useI18n();

const activeSection = ref<SyncSectionKey>("pricing");
const applying = ref(false);
/** The bottom bar has expanded into "what will be written, confirm?". */
const confirming = ref(false);
const applyError = ref<string | null>(null);

// ---- Section state ----
const pricingRef = ref<InstanceType<typeof PricingSyncPanel> | null>(null);
const capabilitiesRef = ref<InstanceType<typeof CapabilitySyncPanel> | null>(null);

const sections = computed(() => [
  {
    key: "pricing" as const,
    labelKey: "models.sync.modePricing",
    icon: CircleDollarSign,
    state: pricingRef.value?.summary ?? EMPTY_SECTION_STATE,
  },
  {
    key: "capabilities" as const,
    labelKey: "models.sync.modeCapabilities",
    icon: SlidersHorizontal,
    state: capabilitiesRef.value?.summary ?? EMPTY_SECTION_STATE,
  },
]);

const totalApplicable = computed(() =>
  sections.value.reduce((sum, s) => sum + s.state.applicableCount, 0)
);
const blockedTotal = computed(() =>
  sections.value.reduce((sum, s) => sum + s.state.blockedCount, 0)
);
const failedSections = computed(() => sections.value.filter((s) => s.state.status === "error"));
const loading = computed(() => sections.value.some((s) => s.state.status === "loading"));

const pending = computed(() => sections.value.flatMap((s) => s.state.pending));
/** "3 mappings repriced · 1 model updated" — the write, in words. */
const writeSummary = computed(() =>
  sections.value
    .filter((s) => s.state.applicableCount > 0)
    .map((s) =>
      s.key === "pricing"
        ? t(
            "models.sync.parts.pricing",
            { count: s.state.applicableCount },
            s.state.applicableCount
          )
        : t(
            "models.sync.parts.capabilities",
            { count: s.state.applicableCount },
            s.state.applicableCount
          )
    )
    .join(" · ")
);

/** Explains the one rule a reviewer needs: what is ticked by default. */
const ruleHint = computed(() =>
  t(
    activeSection.value === "pricing" ? "models.sync.hint.pricing" : "models.sync.hint.capabilities"
  )
);

// A section that loses its last applicable row cannot stay in the confirm view.
watch(totalApplicable, (count) => {
  if (count === 0) confirming.value = false;
});

function refetch() {
  if (applying.value) return;
  confirming.value = false;
  applyError.value = null;
  void pricingRef.value?.fetchPreview();
  void capabilitiesRef.value?.fetchPreview();
}

function onTabKeydown(event: KeyboardEvent, index: number) {
  if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
  event.preventDefault();
  const next =
    (index + (event.key === "ArrowRight" ? 1 : -1) + sections.value.length) % sections.value.length;
  const key = sections.value[next]?.key;
  if (!key) return;
  activeSection.value = key;
  document.getElementById(`models-sync-tab-${key}`)?.focus();
}

function close() {
  if (applying.value) return;
  emit("update:open", false);
}

/** Escape, the X and the overlay all route here: a write in flight must finish. */
function requestOpen(value: boolean) {
  if (!value && applying.value) return;
  emit("update:open", value);
}

// ---- Apply: one action writes both sections ----
async function applyAll() {
  applying.value = true;
  applyError.value = null;
  const pricingUpdates = pricingRef.value?.buildUpdates() ?? [];
  const metadataUpdates = capabilitiesRef.value?.buildUpdates() ?? [];

  const [pricingOutcome, capabilitiesOutcome] = await Promise.allSettled([
    pricingUpdates.length > 0
      ? pricingApi.applyPricing({ updates: pricingUpdates })
      : Promise.resolve(null),
    metadataUpdates.length > 0
      ? metadataApi.applyMetadata({ updates: metadataUpdates })
      : Promise.resolve(null),
  ]);

  const pricing = pricingOutcome.status === "fulfilled" ? pricingOutcome.value : null;
  const capabilities =
    capabilitiesOutcome.status === "fulfilled" ? capabilitiesOutcome.value : null;
  const pricingApplied = pricing?.applied_count ?? 0;
  const capabilitiesApplied = capabilities?.applied_count ?? 0;
  const failures = [
    ...(pricingOutcome.status === "rejected"
      ? [t("models.sync.result.pricingFailed", { count: pricingUpdates.length })]
      : pricing?.failed_count
        ? [t("models.sync.result.pricingFailed", { count: pricing.failed_count })]
        : []),
    ...(capabilitiesOutcome.status === "rejected"
      ? [t("models.sync.result.capabilitiesFailed", { count: metadataUpdates.length })]
      : capabilities?.failed_count
        ? [t("models.sync.result.capabilitiesFailed", { count: capabilities.failed_count })]
        : []),
  ];

  // Writes that landed are already live, so refresh the model list either way.
  if (pricingApplied > 0 || capabilitiesApplied > 0) emit("applied");
  applying.value = false;

  if (failures.length > 0) {
    applyError.value = t("models.sync.result.errorHint", { failed: failures.join(", ") });
    toast.error(t("models.sync.result.titleFailed"), { description: applyError.value });
    // Stay on the confirmation so the error and the retry are in the same place.
    return;
  }

  confirming.value = false;
  toast.success(t("models.sync.result.titleSuccess"), { description: writeSummary.value });
  emit("update:open", false);
}
</script>

<template>
  <Dialog :open="open" @update:open="requestOpen">
    <DialogContent
      class="flex h-[86vh] w-[96vw] max-w-[96vw] flex-col gap-0 overflow-hidden p-0 sm:max-w-[1500px]"
    >
      <DialogHeader class="border-b border-border px-6 pt-5 pb-4">
        <div class="flex flex-wrap items-start justify-between gap-4 pr-8">
          <div class="space-y-1.5">
            <DialogTitle>{{ t("models.sync.title") }}</DialogTitle>
            <DialogDescription>{{ t("models.sync.subtitle") }}</DialogDescription>
          </div>
          <Button variant="outline" size="sm" :disabled="applying || loading" @click="refetch">
            <RefreshCw class="h-3.5 w-3.5" :class="loading ? 'animate-spin' : ''" />
            {{ t("models.sync.refetch") }}
          </Button>
        </div>
      </DialogHeader>

      <!-- One navigation axis: the section tabs. A section's count is its
           pending diff count, so the tab doubles as the "is there anything
           here" signal the old overview step used to carry. -->
      <div
        class="flex items-center gap-1 border-b border-border px-4"
        role="tablist"
        :aria-label="t('models.sync.title')"
      >
        <button
          v-for="(section, index) in sections"
          :id="`models-sync-tab-${section.key}`"
          :key="section.key"
          type="button"
          role="tab"
          :aria-selected="activeSection === section.key"
          :aria-controls="`models-sync-panel-${section.key}`"
          :tabindex="activeSection === section.key ? 0 : -1"
          class="-mb-px flex items-center gap-2 border-b-2 px-3 py-2.5 text-sm transition-colors"
          :class="
            activeSection === section.key
              ? 'border-primary font-medium text-foreground'
              : 'border-transparent text-muted-foreground hover:text-foreground'
          "
          @click="activeSection = section.key"
          @keydown="onTabKeydown($event, index)"
        >
          <component :is="section.icon" class="h-4 w-4" aria-hidden="true" />
          <span class="flex items-center gap-1.5">
            {{ t(section.labelKey) }}
            <RefreshCw
              v-if="section.state.status === 'loading'"
              class="h-3 w-3 animate-spin text-muted-foreground"
              aria-hidden="true"
            />
            <span
              v-else-if="section.state.status === 'ready'"
              class="rounded-full px-1.5 py-px text-[11px] font-semibold tabular-nums"
              :class="
                section.state.changedCount > 0
                  ? 'bg-muted text-foreground'
                  : 'bg-muted/60 text-muted-foreground'
              "
            >
              {{ section.state.changedCount }}
            </span>
            <span
              v-else-if="section.state.status === 'error'"
              class="text-status-error"
              aria-hidden
            >
              !
            </span>
            <span v-else class="text-muted-foreground" aria-hidden>–</span>
          </span>
          <span class="sr-only">
            {{
              section.state.status === "error"
                ? t("models.sync.tabFailed")
                : section.state.status === "loading"
                  ? t("models.sync.tabLoading")
                  : ""
            }}
          </span>
        </button>
      </div>

      <div class="flex min-h-0 flex-1 flex-col overflow-hidden">
        <div
          v-show="activeSection === 'pricing'"
          id="models-sync-panel-pricing"
          role="tabpanel"
          aria-labelledby="models-sync-tab-pricing"
          class="flex min-h-0 flex-1 flex-col outline-none"
        >
          <PricingSyncPanel ref="pricingRef" />
        </div>
        <div
          v-show="activeSection === 'capabilities'"
          id="models-sync-panel-capabilities"
          role="tabpanel"
          aria-labelledby="models-sync-tab-capabilities"
          class="flex min-h-0 flex-1 flex-col outline-none"
        >
          <CapabilitySyncPanel ref="capabilitiesRef" />
        </div>
      </div>

      <!-- Bottom bar. Same place, same width, two states: what apply would
           write, and — after one click — the final confirm. -->
      <div class="border-t border-border px-6 py-3">
        <div v-if="confirming" class="flex flex-wrap items-center justify-between gap-3">
          <div class="min-w-0 flex-1">
            <p class="text-sm">
              <span class="font-medium">{{ t("models.sync.confirm.willWrite") }}</span>
              <span class="text-muted-foreground"> · {{ writeSummary }}</span>
            </p>
            <p class="truncate text-xs text-muted-foreground" :title="pending.join(', ')">
              {{ pending.join(", ") }}
            </p>
            <p v-if="applyError" class="mt-1 text-xs text-status-error">{{ applyError }}</p>
          </div>
          <div class="flex items-center gap-2">
            <Button variant="outline" :disabled="applying" @click="confirming = false">
              {{ t("common.back") }}
            </Button>
            <Button class="btn-action" :disabled="applying" @click="applyAll">
              <RefreshCw v-if="applying" class="h-4 w-4 animate-spin" aria-hidden="true" />
              {{
                applying
                  ? t("models.sync.applying")
                  : t("models.sync.confirm.apply", { count: totalApplicable }, totalApplicable)
              }}
            </Button>
          </div>
        </div>

        <div v-else class="flex flex-wrap items-center justify-between gap-3">
          <div class="min-w-0 flex-1 space-y-0.5">
            <p class="text-sm">
              {{
                totalApplicable > 0
                  ? t("models.sync.selection", { count: totalApplicable }, totalApplicable)
                  : t("models.sync.selectionNone")
              }}
            </p>
            <p class="truncate text-xs text-muted-foreground">
              <span v-if="blockedTotal > 0" class="text-status-warning">
                {{ t("models.sync.warnBlocked", { count: blockedTotal }) }}
              </span>
              <span
                v-for="section in failedSections"
                :key="section.key"
                class="text-status-error"
                :class="blockedTotal > 0 ? 'ml-2' : ''"
              >
                {{ t("models.sync.warnSectionFailed", { section: t(section.labelKey) }) }}
              </span>
              <span v-if="blockedTotal === 0 && failedSections.length === 0">{{ ruleHint }}</span>
            </p>
          </div>
          <div class="flex items-center gap-2">
            <Button variant="outline" @click="close">{{ t("common.cancel") }}</Button>
            <Button class="btn-action" :disabled="totalApplicable === 0" @click="confirming = true">
              {{ t("models.sync.apply", { count: totalApplicable }, totalApplicable) }}
              <ArrowRight class="h-4 w-4" aria-hidden="true" />
            </Button>
          </div>
        </div>
      </div>
    </DialogContent>
  </Dialog>
</template>
