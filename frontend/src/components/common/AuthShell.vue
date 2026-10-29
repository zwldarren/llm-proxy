<script setup lang="ts">
import { useI18n } from "vue-i18n";
import LogoIcon from "@/components/common/LogoIcon.vue";
import LanguageSelector from "@/components/settings/LanguageSelector.vue";
import ThemeSelector from "@/components/settings/ThemeSelector.vue";

interface Props {
  title: string;
  description?: string;
}

withDefaults(defineProps<Props>(), {
  description: "",
});

const { t } = useI18n();
</script>

<template>
  <div class="login-bg relative min-h-screen min-h-dvh overflow-x-hidden bg-background">
    <main class="relative z-10 flex min-h-screen min-h-dvh flex-col px-4 py-8 sm:py-10">
      <div
        class="m-auto flex w-full max-w-sm flex-col animate-in fade-in slide-in-from-bottom-4 duration-500"
      >
        <header class="mb-8 flex flex-col items-center text-center">
          <div
            class="auth-mark flex size-14 items-center justify-center rounded-xl text-foreground ring-1 ring-primary/25"
          >
            <LogoIcon class="size-8" />
          </div>
          <span class="mt-6 text-xs text-muted-foreground">{{ t("app.name") }}</span>
          <h1 class="brand-heading mt-2 text-xl leading-tight tracking-tight sm:text-2xl">
            {{ title }}
          </h1>
          <p v-if="description" class="mt-2 max-w-sm text-sm text-muted-foreground">
            {{ description }}
          </p>
        </header>

        <slot />

        <!-- Sign-in/out pages are the only surface a locked-out operator can
             reach, so language and theme choices must be available here too. -->
        <footer
          class="mt-8 flex flex-wrap items-center justify-center gap-2 border-t border-border/50 pt-5"
        >
          <ThemeSelector />
          <LanguageSelector />
        </footer>
      </div>
    </main>
  </div>
</template>
