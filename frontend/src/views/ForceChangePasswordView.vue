<script setup lang="ts">
import { useI18n } from "vue-i18n";
import { useRouter } from "vue-router";
import { Loader2 } from "@lucide/vue";
import { toast } from "vue-sonner";
import LogoIcon from "@/components/common/LogoIcon.vue";
import { useChangePassword } from "@/composables/useChangePassword";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAuthStore } from "@/stores/auth";

const router = useRouter();
const { t } = useI18n();
const authStore = useAuthStore();

// The forced-change page shares the sidebar dialog's form logic (validation,
// error mapping, API call); only the success handling differs: the flag and
// every token are revoked server-side, so the session ends and the user is
// sent back to the login screen.
const { currentPassword, newPassword, confirmPassword, error, loading, passwordHint, submit } =
  useChangePassword();

const handleSubmit = async () => {
  if (await submit()) {
    authStore.clearLocalSession();
    toast.info(t("auth.passwordChangedRelogin"));
    await router.replace("/login");
  }
};

const handleSignOut = async () => {
  // Logout stays reachable while the forced-change flag is set.
  await authStore.logout();
  await router.push("/login");
};
</script>

<template>
  <div
    class="login-bg relative flex min-h-screen flex-col items-center justify-center overflow-hidden bg-background px-4"
  >
    <main
      class="relative z-10 flex w-full max-w-sm flex-col animate-in fade-in slide-in-from-bottom-4 duration-500"
    >
      <!-- Brand mark -->
      <div class="mb-8 flex flex-col items-center text-center">
        <div
          class="auth-mark ring-1 ring-primary/25 flex h-14 w-14 items-center justify-center rounded-2xl text-foreground"
        >
          <LogoIcon class="h-8 w-8" />
        </div>
        <span class="mt-6 text-xs text-muted-foreground"> LLM Proxy </span>
        <h1
          class="brand-heading mt-2 text-xl sm:text-2xl leading-tight tracking-tight text-foreground"
        >
          {{ t("auth.forcedPasswordTitle") }}
        </h1>
        <p class="mt-2 max-w-sm text-sm text-muted-foreground">
          {{ t("auth.forcedPasswordDescription") }}
        </p>
      </div>

      <form
        @submit.prevent="handleSubmit"
        class="flex flex-col gap-4"
        :aria-describedby="error ? 'forced-password-error' : undefined"
        :aria-busy="loading"
      >
        <Alert
          v-if="error"
          variant="destructive"
          class="animate-in slide-in-from-top-2 fade-in duration-300"
          id="forced-password-error"
        >
          <AlertDescription>
            {{ error }}
          </AlertDescription>
        </Alert>

        <div class="flex flex-col gap-1.5">
          <Label for="current-password" class="text-sm font-medium text-foreground/90">
            {{ t("profile.currentPassword") }}
          </Label>
          <Input
            id="current-password"
            v-model="currentPassword"
            type="password"
            autocomplete="current-password"
            required
            :disabled="loading"
            class="auth-input h-11"
          />
        </div>

        <div class="flex flex-col gap-1.5">
          <Label for="new-password" class="text-sm font-medium text-foreground/90">
            {{ t("profile.newPassword") }}
          </Label>
          <Input
            id="new-password"
            v-model="newPassword"
            type="password"
            autocomplete="new-password"
            required
            :disabled="loading"
            class="auth-input h-11"
          />
          <p v-if="newPassword" id="new-password-hint" class="mt-1 text-xs text-muted-foreground">
            {{ passwordHint }}
          </p>
        </div>

        <div class="flex flex-col gap-1.5">
          <Label for="confirm-new-password" class="text-sm font-medium text-foreground/90">
            {{ t("profile.confirmPassword") }}
          </Label>
          <Input
            id="confirm-new-password"
            v-model="confirmPassword"
            type="password"
            autocomplete="new-password"
            required
            :disabled="loading"
            class="auth-input h-11"
            @keydown.enter="handleSubmit"
          />
        </div>

        <Button
          type="submit"
          :disabled="loading"
          class="btn-press mt-1 flex h-11 w-full items-center justify-center gap-2"
          size="lg"
        >
          <Loader2 v-if="loading" class="h-4 w-4 animate-spin text-primary-foreground" />
          {{ loading ? t("common.loading") : t("profile.changePassword") }}
        </Button>

        <Button
          type="button"
          variant="ghost"
          :disabled="loading"
          class="w-full text-muted-foreground"
          @click="handleSignOut"
        >
          {{ t("auth.logout") }}
        </Button>
      </form>
    </main>

    <footer class="absolute bottom-5 left-0 right-0 z-10 text-center">
      <span class="text-xs text-muted-foreground">{{ t("app.name") }}</span>
    </footer>
  </div>
</template>
