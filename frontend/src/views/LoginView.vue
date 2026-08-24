<script setup lang="ts">
import { ref } from "vue";
import { useI18n } from "vue-i18n";
import { useRoute, useRouter } from "vue-router";
import { Loader2 } from "@lucide/vue";
import LogoIcon from "@/components/common/LogoIcon.vue";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { HttpError } from "@/services/http";
import { useAuthStore } from "@/stores/auth";

const router = useRouter();
const route = useRoute();
const { t } = useI18n();
const authStore = useAuthStore();

const username = ref("");
const password = ref("");
const error = ref("");
const loading = ref(false);

const handleLogin = async () => {
  if (loading.value) return;

  error.value = "";
  loading.value = true;

  try {
    await authStore.login({
      username: username.value,
      password: password.value,
    });
    const redirect = (route.query.redirect as string) || "/";
    await router.push(redirect);
  } catch (e: unknown) {
    if (e instanceof HttpError && e.status === 401) {
      error.value = t("auth.loginError");
    } else {
      error.value = e instanceof Error ? e.message : t("common.error");
    }
    loading.value = false;
  }
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
          {{ t("auth.loginTitle") }}
        </h1>
        <p class="mt-2 max-w-sm text-sm text-muted-foreground">
          {{ t("auth.loginDescription") }}
        </p>
      </div>

      <form
        @submit.prevent="handleLogin"
        class="flex flex-col gap-4"
        :aria-describedby="error ? 'login-error' : undefined"
        :aria-busy="loading"
      >
        <Alert
          v-if="error"
          variant="destructive"
          class="animate-in slide-in-from-top-2 fade-in duration-300"
          id="login-error"
        >
          <AlertDescription>
            {{ error }}
          </AlertDescription>
        </Alert>

        <div class="flex flex-col gap-1.5">
          <Label for="username" class="text-sm font-medium text-foreground/90">
            {{ t("auth.username") }}
          </Label>
          <Input
            id="username"
            v-model="username"
            type="text"
            autocomplete="username"
            required
            :disabled="loading"
            class="auth-input h-11"
          />
        </div>

        <div class="flex flex-col gap-1.5">
          <Label for="password" class="text-sm font-medium text-foreground/90">
            {{ t("auth.password") }}
          </Label>
          <Input
            id="password"
            v-model="password"
            type="password"
            autocomplete="current-password"
            required
            :disabled="loading"
            class="auth-input h-11"
          />
        </div>

        <Button
          type="submit"
          :disabled="loading"
          class="btn-press mt-1 flex h-11 w-full items-center justify-center gap-2"
          size="lg"
        >
          <Loader2 v-if="loading" class="h-4 w-4 animate-spin text-primary-foreground" />
          {{ loading ? t("common.loading") : t("auth.loginButton") }}
        </Button>
      </form>
    </main>

    <footer class="absolute bottom-5 left-0 right-0 z-10 text-center">
      <span class="text-xs text-muted-foreground">{{ t("app.name") }}</span>
    </footer>
  </div>
</template>
