<script setup lang="ts">
import { ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import { Fingerprint, Loader2 } from "@lucide/vue";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { authApi } from "@/services/api/auth";
import { useStepUpStore } from "@/stores/stepUp";
import { getErrorMessage } from "@/utils/error";
import { getPasskey } from "@/utils/webauthn";

const { t } = useI18n();
const stepUp = useStepUpStore();
const password = ref("");
const loading = ref(false);
const error = ref("");

watch(
  () => stepUp.required,
  (open) => {
    if (open) {
      password.value = "";
      error.value = "";
    }
  }
);

function messageFor(e: unknown): string {
  return getErrorMessage(e) || t("common.error");
}

async function verifyWithPasskey() {
  const begin = await authApi.passkeyStepUpBegin();
  const credential = await getPasskey(begin.options);
  await authApi.passkeyStepUpFinish(begin.challenge_token, credential);
}

async function submit() {
  loading.value = true;
  error.value = "";
  try {
    const result = await authApi.stepUp(password.value || undefined);
    if (result.passkey_required) {
      await verifyWithPasskey();
    }
    stepUp.complete();
  } catch (e) {
    error.value = messageFor(e);
  } finally {
    loading.value = false;
  }
}

async function usePasskey() {
  loading.value = true;
  error.value = "";
  try {
    await verifyWithPasskey();
    stepUp.complete();
  } catch (e) {
    error.value = messageFor(e);
  } finally {
    loading.value = false;
  }
}
</script>

<template>
  <Dialog :open="stepUp.required" @update:open="(open) => !open && stepUp.cancel()">
    <DialogContent class="z-[100]" overlay-class="z-[100]">
      <DialogHeader>
        <DialogTitle>{{ t("auth.stepUpTitle") }}</DialogTitle>
        <DialogDescription>{{ t("auth.stepUpDescription") }}</DialogDescription>
      </DialogHeader>
      <form class="flex flex-col gap-3" @submit.prevent="submit">
        <div class="flex flex-col gap-1.5">
          <Label for="global-step-up-password">{{ t("auth.password") }}</Label>
          <Input
            id="global-step-up-password"
            v-model="password"
            type="password"
            autocomplete="current-password"
            :disabled="loading"
          />
        </div>
        <p v-if="error" class="text-xs text-destructive">{{ error }}</p>
        <DialogFooter class="gap-2">
          <Button
            type="button"
            variant="outline"
            class="gap-2"
            :disabled="loading"
            @click="usePasskey"
          >
            <Fingerprint class="h-4 w-4" />
            {{ t("auth.passkeyLogin") }}
          </Button>
          <Button type="submit" :disabled="loading">
            <Loader2 v-if="loading" class="h-4 w-4 animate-spin" />
            {{ t("auth.verifyButton") }}
          </Button>
        </DialogFooter>
      </form>
    </DialogContent>
  </Dialog>
</template>
