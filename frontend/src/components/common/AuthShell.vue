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
      <div class="auth-enter m-auto flex w-full max-w-sm flex-col">
        <!-- Identity is a lockup, not a kicker: the mark and the product name
             sit on one line so the heading below is the only thing above the
             description. A stacked 11px label there would read as an eyebrow. -->
        <header class="mb-8 flex flex-col items-center text-center">
          <div class="flex items-center gap-3">
            <div
              class="auth-mark flex size-12 items-center justify-center rounded-xl text-foreground ring-1 ring-primary/25"
            >
              <LogoIcon class="size-7" />
            </div>
            <span class="brand-heading text-sm text-foreground">{{ t("app.name") }}</span>
          </div>
          <h1 class="brand-heading mt-6 text-xl leading-tight tracking-tight sm:text-2xl">
            {{ title }}
          </h1>
          <p v-if="description" class="mt-2 max-w-sm text-sm text-muted-foreground">
            {{ description }}
          </p>
        </header>

        <slot />

        <!-- Sign-in/out pages are the only surface a locked-out operator can
             reach, so language and theme choices must be available here too.
             They are chrome, not part of the task: compact, unboxed, and set
             apart by space rather than a rule. -->
        <footer class="mt-10 flex flex-wrap items-center justify-center gap-x-3 gap-y-2">
          <ThemeSelector compact />
          <span class="h-4 w-px bg-border/70" aria-hidden="true" />
          <LanguageSelector compact />
        </footer>
      </div>
    </main>
  </div>
</template>
