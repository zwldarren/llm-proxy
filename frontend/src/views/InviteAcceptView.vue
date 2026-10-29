<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useI18n } from "vue-i18n";
import { useRoute, useRouter } from "vue-router";
import { Loader2 } from "@lucide/vue";
import { toast } from "vue-sonner";
import AuthShell from "@/components/common/AuthShell.vue";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { invitationsApi } from "@/services/api/invitations";
import { authApi } from "@/services/api/auth";
import { HttpError } from "@/services/http";
import { useAuthStore } from "@/stores/auth";
import { getErrorMessage } from "@/utils/error";
import { passwordRequirementsText, validatePasswordPair } from "@/utils/password";
import type { OAuthProvider, PublicInvitation } from "@/types/schemas";

const route = useRoute();
const router = useRouter();
const { t } = useI18n();
const authStore = useAuthStore();
const token = String(route.params.token || "");
const invitation = ref<PublicInvitation | null>(null);
const loading = ref(true);
const submitting = ref(false);
const error = ref("");
const username = ref("");
const password = ref("");
const confirmPassword = ref("");
const providers = ref<OAuthProvider[]>([]);

onMounted(async () => {
  try {
    invitation.value = await invitationsApi.getPublic(token);
  } catch (e: unknown) {
    // 404 = unknown/used token; 400/410 = expired or revoked. All three mean
    // the same thing to the invitee, so never surface the raw HTTP error.
    error.value =
      e instanceof HttpError && [400, 404, 410].includes(e.status)
        ? t("auth.inviteExpired")
        : e instanceof Error
          ? e.message
          : t("common.error");
  } finally {
    loading.value = false;
  }
  try {
    providers.value = (await authApi.listOAuthProviders()).providers;
  } catch {
    providers.value = [];
  }
});

function startOAuth(providerId: string) {
  window.location.href = `/api/auth/oauth/${providerId}/begin?invite=${encodeURIComponent(token)}`;
}

async function accept() {
  error.value = "";
  const passwordError = validatePasswordPair(password.value, confirmPassword.value, {
    required: true,
  });
  if (passwordError) {
    error.value = passwordError;
    return;
  }
  submitting.value = true;
  try {
    await authStore.acceptInvitation(token, username.value, password.value);
    // The invitation is now consumed and the new account is signed in. A
    // passkey is the recommended daily sign-in factor, so land on the
    // security page where it can be added immediately.
    toast.success(t("auth.inviteAccepted"));
    await router.push("/account/security?from=invite");
  } catch (e: unknown) {
    error.value = getErrorMessage(e) || t("common.error");
  } finally {
    submitting.value = false;
  }
}
</script>

<template>
  <AuthShell
    :title="t('auth.inviteTitle')"
    :description="invitation ? t('auth.inviteDescription', { email: invitation.email }) : ''"
  >
    <div v-if="loading" class="flex justify-center py-10">
      <Loader2 class="h-5 w-5 animate-spin text-muted-foreground" />
    </div>

    <Alert v-else-if="error && !invitation" variant="destructive">
      <AlertDescription>{{ error }}</AlertDescription>
    </Alert>

    <form v-else-if="invitation" class="flex flex-col gap-4" @submit.prevent="accept">
      <Alert v-if="error" variant="destructive">
        <AlertDescription>{{ error }}</AlertDescription>
      </Alert>
      <div class="flex flex-col gap-1.5">
        <Label for="invite-username">{{ t("auth.username") }}</Label>
        <Input
          id="invite-username"
          v-model="username"
          autocomplete="username"
          autocapitalize="none"
          spellcheck="false"
          required
          class="auth-input h-11"
        />
      </div>
      <div class="flex flex-col gap-1.5">
        <Label for="invite-password">{{ t("auth.password") }}</Label>
        <Input
          id="invite-password"
          v-model="password"
          type="password"
          autocomplete="new-password"
          required
          class="auth-input h-11"
        />
        <p class="text-xs text-muted-foreground">
          {{ t("auth.invitePasswordHint") }}
        </p>
      </div>
      <div class="flex flex-col gap-1.5">
        <Label for="invite-confirm">{{ t("auth.confirmPassword") }}</Label>
        <Input
          id="invite-confirm"
          v-model="confirmPassword"
          type="password"
          autocomplete="new-password"
          required
          class="auth-input h-11"
        />
        <p class="text-xs text-muted-foreground">{{ passwordRequirementsText() }}</p>
      </div>
      <Button type="submit" :disabled="submitting" class="h-11 w-full">
        <Loader2 v-if="submitting" class="h-4 w-4 animate-spin" />
        {{ submitting ? t("common.loading") : t("auth.inviteAccept") }}
      </Button>
      <div v-if="providers.length" class="mt-2 flex flex-col gap-2">
        <div class="flex items-center gap-3 text-xs text-muted-foreground">
          <span class="h-px flex-1 bg-border" />
          {{ t("auth.orContinueWith") }}
          <span class="h-px flex-1 bg-border" />
        </div>
        <Button
          v-for="provider in providers"
          :key="provider.id"
          type="button"
          variant="outline"
          class="h-11 w-full"
          @click="startOAuth(provider.id)"
        >
          {{ provider.name }}
        </Button>
      </div>
    </form>
  </AuthShell>
</template>
