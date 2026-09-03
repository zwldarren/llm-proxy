<script setup lang="ts">
import { AlertCircle, CheckCircle2, Fingerprint, Loader2, ShieldAlert } from "@lucide/vue";
import { ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import LoadingState from "@/components/common/LoadingState.vue";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { logsApi } from "@/services/api/logs";
import type { AuditIntegrityError, AuditIntegrityResult } from "@/types/schemas";

const props = defineProps<{
  open: boolean;
}>();

const emit = defineEmits<{
  (e: "update:open", value: boolean): void;
}>();

const { t } = useI18n();

const isVerifying = ref(false);
const error = ref<string | null>(null);
const result = ref<AuditIntegrityResult | null>(null);

const runVerification = async () => {
  isVerifying.value = true;
  error.value = null;
  result.value = null;
  try {
    result.value = await logsApi.verifyIntegrity();
  } catch {
    error.value = t("logs.audit.integrityCheckFailed");
  } finally {
    isVerifying.value = false;
  }
};

// Trigger verification each time the dialog is opened.
watch(
  () => props.open,
  (isOpen) => {
    if (isOpen) {
      void runVerification();
    } else {
      // Reset state on close so the next open re-verifies cleanly.
      result.value = null;
      error.value = null;
    }
  }
);

const close = () => emit("update:open", false);

const errorType = (e: AuditIntegrityError): "warning" | "error" => {
  // "Content hash mismatch" is the most severe (possible tampering).
  return e.error.toLowerCase().includes("content hash") ? "error" : "warning";
};
</script>

<template>
  <Dialog :open="open" @update:open="emit('update:open', $event)">
    <DialogContent class="max-w-2xl gap-0 p-0 overflow-hidden border-border/80 bg-card">
      <!-- Header -->
      <DialogHeader
        class="px-4 sm:px-6 py-4 sm:py-5 border-b border-border/60 bg-muted/10 space-y-0"
      >
        <DialogTitle class="flex items-center gap-2.5 text-base font-semibold text-foreground">
          <div
            class="p-2 rounded-md bg-muted border border-border/40 text-muted-foreground shrink-0 flex items-center justify-center"
          >
            <Fingerprint class="size-4" />
          </div>
          {{ t("logs.audit.verifyIntegrityTitle") }}
        </DialogTitle>
        <DialogDescription class="text-xs text-muted-foreground pt-1">
          {{ t("logs.audit.verifyIntegrityDescription") }}
        </DialogDescription>
      </DialogHeader>

      <!-- Body -->
      <div class="p-4 sm:p-6 min-h-[180px]">
        <!-- Loading -->
        <div
          v-if="isVerifying"
          class="flex flex-col items-center justify-center gap-3 py-10 text-center"
        >
          <LoadingState :show-text="false" />
          <p class="text-xs text-muted-foreground">{{ t("logs.audit.verifying") }}</p>
        </div>

        <!-- Error -->
        <Alert v-else-if="error" variant="destructive">
          <AlertCircle class="size-4" />
          <p class="font-medium leading-none mb-1">{{ t("common.error") }}</p>
          <AlertDescription>{{ error }}</AlertDescription>
        </Alert>

        <!-- Result -->
        <template v-else-if="result">
          <!-- Valid -->
          <Alert v-if="result.valid && result.verified_count > 0" variant="default">
            <CheckCircle2 class="size-4 text-status-success" />
            <p class="font-medium leading-none mb-1 text-foreground">
              {{ t("logs.audit.integrityValid", { count: result.verified_count }) }}
            </p>
            <AlertDescription>
              {{ t("logs.audit.hashIntact") }}
            </AlertDescription>
          </Alert>

          <!-- Empty -->
          <Alert v-else-if="result.valid && result.verified_count === 0" variant="default">
            <AlertCircle class="size-4 text-muted-foreground" />
            <p class="font-medium leading-none text-foreground">
              {{ t("logs.audit.integrityEmpty") }}
            </p>
          </Alert>

          <!-- Invalid -->
          <template v-else>
            <Alert variant="destructive">
              <ShieldAlert class="size-4" />
              <p class="font-medium leading-none mb-1">
                {{ t("logs.audit.integrityInvalid", { count: result.errors.length }) }}
              </p>
              <AlertDescription>
                {{ t("logs.audit.integrityVerified") }}: {{ result.verified_count }}
              </AlertDescription>
            </Alert>

            <!-- Violation list -->
            <ScrollArea class="mt-4 h-[280px] w-full rounded-md border border-border/40">
              <div class="p-3 space-y-2.5">
                <div
                  v-for="(e, idx) in result.errors"
                  :key="idx"
                  class="rounded-md border p-3 bg-background/60"
                  :class="
                    errorType(e) === 'error'
                      ? 'border-status-error/30 bg-status-error/5'
                      : 'border-status-warning/30 bg-status-warning/5'
                  "
                >
                  <div class="flex items-center justify-between gap-2">
                    <span
                      class="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground"
                    >
                      {{ t("logs.audit.integritySequence") }}
                    </span>
                    <span class="font-mono text-xs font-bold text-foreground">
                      #{{ e.sequence ?? "—" }}
                    </span>
                  </div>
                  <p class="mt-1.5 text-xs font-medium text-foreground break-words">
                    {{ e.error }}
                  </p>
                  <div
                    v-if="e.expected || e.actual"
                    class="mt-2 grid grid-cols-1 sm:grid-cols-2 gap-2 text-[11px]"
                  >
                    <div class="min-w-0">
                      <div class="text-muted-foreground font-semibold">
                        {{ t("logs.audit.integrityExpected") }}
                      </div>
                      <div
                        class="font-mono text-muted-foreground break-all bg-muted/40 px-1.5 py-1 rounded mt-0.5"
                      >
                        {{ e.expected || "—" }}
                      </div>
                    </div>
                    <div class="min-w-0">
                      <div class="text-muted-foreground font-semibold">
                        {{ t("logs.audit.integrityActual") }}
                      </div>
                      <div
                        class="font-mono text-status-error/90 break-all bg-muted/40 px-1.5 py-1 rounded mt-0.5"
                      >
                        {{ e.actual || "—" }}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </ScrollArea>
          </template>
        </template>
      </div>

      <!-- Footer -->
      <DialogFooter
        class="px-4 sm:px-6 py-4 border-t border-border/60 bg-muted/10 justify-between sm:justify-between"
      >
        <Button
          variant="ghost"
          size="sm"
          class="text-xs"
          :disabled="isVerifying"
          @click="runVerification"
        >
          <Loader2 v-if="isVerifying" class="size-3.5 mr-1.5 animate-spin" />
          <span>{{ t("common.refresh") }}</span>
        </Button>
        <Button variant="outline" size="sm" class="text-xs" @click="close">
          {{ t("common.close") }}
        </Button>
      </DialogFooter>
    </DialogContent>
  </Dialog>
</template>
