<script setup lang="ts">
import { CircleAlert, Copy, RotateCw } from "@lucide/vue";
import { computed, ref } from "vue";
import { useI18n } from "vue-i18n";
import { toast } from "vue-sonner";
import { Button } from "@/components/ui/button";

/**
 * The console's recovery surface for a render-time crash.
 *
 * Mounted by `App.vue`'s error boundary when any descendant throws during
 * render, a lifecycle hook or an event handler. It exists so one broken screen
 * cannot take the whole tab with it: the operator gets a way back into the
 * console, plus the technical detail needed to report what broke.
 */
const props = defineProps<{
  /** The captured error; its message names the failure. */
  error: Error;
  /** Vue's description of where it was captured ("render function", ...). */
  info?: string | null;
  /** Name of the component that threw, when Vue reported one. */
  componentName?: string | null;
}>();

const emit = defineEmits<{
  (e: "home"): void;
  (e: "reload"): void;
}>();

const { t } = useI18n();

/** Where the error came from, as a single mono line above the stack. */
const origin = computed(() => {
  const parts = [props.componentName, props.info].filter(Boolean);
  return parts.length > 0 ? `<${parts.join(" · ")}>` : "";
});

/** The stack, trimmed to the frames a bug report actually needs. */
const stack = computed(() => {
  const raw = props.error.stack?.trim();
  if (!raw) return origin.value;
  const frames = raw.split("\n").slice(0, 12).join("\n");
  return origin.value ? `${origin.value}\n${frames}` : frames;
});

const copied = ref(false);

async function copyDetails(): Promise<void> {
  try {
    await navigator.clipboard.writeText(`${props.error.message}\n\n${stack.value}`);
    copied.value = true;
    setTimeout(() => (copied.value = false), 2000);
  } catch {
    toast.error(t("common.copyFailed"));
  }
}
</script>

<template>
  <main
    class="fixed inset-0 z-[100] flex items-center justify-center overflow-y-auto bg-background px-4 py-10"
    role="alert"
    aria-live="assertive"
  >
    <div class="w-full max-w-md">
      <div
        class="flex size-12 items-center justify-center rounded-xl bg-destructive/10 ring-1 ring-destructive/25"
      >
        <CircleAlert class="size-6 text-destructive" aria-hidden="true" />
      </div>

      <h1 class="brand-heading mt-6 text-xl leading-tight tracking-tight sm:text-2xl">
        {{ t("errors.somethingWrong") }}
      </h1>
      <p class="mt-2 text-sm text-muted-foreground">
        {{ t("errors.renderFailed") }}
      </p>

      <!-- The raw message stays visible: it is the one clue an operator can
           quote in a bug report. The stack stays behind a disclosure so the
           screen still reads as an explanation, not a crash dump. -->
      <p class="mt-4 font-mono text-xs leading-relaxed break-words text-foreground/70">
        {{ error.message }}
      </p>

      <details class="group mt-4 rounded-md border border-border/60 bg-muted/30">
        <summary
          class="flex cursor-pointer list-none items-center gap-2 px-3 py-2 text-xs text-muted-foreground transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50 focus-visible:outline-none"
        >
          <span class="flex-1">{{ t("errors.errorDetails") }}</span>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            class="h-6 gap-1.5 px-2 text-xs text-muted-foreground hover:text-foreground"
            @click.prevent.stop="copyDetails"
          >
            <Copy class="size-3" aria-hidden="true" />
            {{ copied ? t("common.copied") : t("common.copy") }}
          </Button>
        </summary>
        <pre
          class="max-h-48 overflow-auto border-t border-border/60 px-3 py-2 font-mono text-[11px] leading-relaxed whitespace-pre-wrap text-muted-foreground"
          >{{ stack }}</pre>
      </details>

      <div class="mt-6 flex flex-wrap items-center gap-2">
        <Button type="button" @click="emit('home')">
          {{ t("errors.backToDashboard") }}
        </Button>
        <Button type="button" variant="outline" class="gap-2" @click="emit('reload')">
          <RotateCw class="size-4" aria-hidden="true" />
          {{ t("common.reload") }}
        </Button>
      </div>
    </div>
  </main>
</template>
