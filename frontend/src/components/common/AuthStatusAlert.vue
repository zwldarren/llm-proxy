<script setup lang="ts">
import type { Component } from "vue";
import { computed } from "vue";
import { CircleAlert, CircleCheck, Info } from "@lucide/vue";
import { Alert, AlertDescription } from "@/components/ui/alert";

/**
 * Status message for the authentication screens.
 *
 * The alert carries an icon so the state is not communicated by red text alone,
 * and the icon pairs with the message instead of standing in for it.
 */
interface Props {
  /** Error, success or neutral information. */
  tone?: "error" | "success" | "info";
  id?: string;
}

const props = withDefaults(defineProps<Props>(), {
  tone: "info",
  id: undefined,
});

const TONES = {
  error: {
    variant: "destructive",
    role: "alert",
    className: "",
    icon: CircleAlert,
    iconClass: "",
  },
  success: {
    variant: "default",
    role: "status",
    className: "border-status-success/30 bg-status-success/10 text-status-success",
    icon: CircleCheck,
    iconClass: "text-status-success",
  },
  info: {
    variant: "default",
    role: "alert",
    className: "",
    icon: Info,
    iconClass: "",
  },
} satisfies Record<
  NonNullable<Props["tone"]>,
  {
    variant: "default" | "destructive";
    role: "alert" | "status";
    className: string;
    icon: Component;
    iconClass: string;
  }
>;

const tone = computed(() => TONES[props.tone]);
</script>

<template>
  <!-- A success confirmation is non-interruptive; failures stay assertive. -->
  <Alert :variant="tone.variant" :class="tone.className" :role="tone.role" :id="id">
    <AlertDescription>
      <span class="flex items-start gap-2">
        <component
          :is="tone.icon"
          class="mt-0.5 size-4 shrink-0"
          :class="tone.iconClass"
          aria-hidden="true"
        />
        <span class="min-w-0 break-words"><slot /></span>
      </span>
    </AlertDescription>
  </Alert>
</template>
