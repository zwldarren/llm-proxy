<script setup lang="ts">
import { ref } from "vue";
import { useI18n } from "vue-i18n";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import CapabilitySyncPanel from "./CapabilitySyncPanel.vue";
import PricingSyncPanel from "./PricingSyncPanel.vue";

/**
 * Single entry point for syncing from models.dev: a thin shell hosting the
 * pricing review (per provider mapping) and the capability/metadata review
 * (per model) as two sibling panels. Both panels stay mounted (v-show) so
 * in-progress review state survives switching modes; each fetches on open.
 */
defineOptions({ name: "ModelsSyncDialog" });

defineProps<{ open: boolean }>();
const emit = defineEmits<{
  "update:open": [value: boolean];
  applied: [];
}>();

const { t } = useI18n();

type Mode = "pricing" | "capabilities";
const activeMode = ref<Mode>("pricing");

const modes: { key: Mode; labelKey: string }[] = [
  { key: "pricing", labelKey: "models.sync.modePricing" },
  { key: "capabilities", labelKey: "models.sync.modeCapabilities" },
];

// A successful apply refreshes the model store and closes the whole dialog.
function handleApplied() {
  emit("applied");
  emit("update:open", false);
}
</script>

<template>
  <Dialog :open="open" @update:open="emit('update:open', $event)">
    <DialogContent class="flex h-[86vh] w-[96vw] max-w-[96vw] flex-col gap-0 p-0 sm:max-w-[1500px]">
      <DialogHeader class="px-6 pt-6 pb-4 border-b border-border">
        <DialogTitle>{{ t("models.sync.title") }}</DialogTitle>
        <DialogDescription>{{ t("models.sync.subtitle") }}</DialogDescription>
      </DialogHeader>

      <!-- Mode switch: sibling panels stay mounted so review state persists -->
      <div class="px-6 py-3 border-b border-border">
        <div class="inline-flex items-center gap-1 rounded-md bg-muted/40 p-0.5">
          <button
            v-for="mode in modes"
            :key="mode.key"
            type="button"
            class="inline-flex items-center rounded-[5px] px-3 py-1 text-xs font-medium transition-colors"
            :class="
              activeMode === mode.key
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            "
            @click="activeMode = mode.key"
          >
            {{ t(mode.labelKey) }}
          </button>
        </div>
      </div>

      <div v-show="activeMode === 'pricing'" class="flex min-h-0 flex-1 flex-col">
        <PricingSyncPanel
          :open="open"
          @applied="handleApplied"
          @close="emit('update:open', false)"
        />
      </div>
      <div v-show="activeMode === 'capabilities'" class="flex min-h-0 flex-1 flex-col">
        <CapabilitySyncPanel
          :open="open"
          @applied="handleApplied"
          @close="emit('update:open', false)"
        />
      </div>
    </DialogContent>
  </Dialog>
</template>
