<script setup lang="ts" generic="T extends string | number">
import { computed } from "vue";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

/**
 * One labelled form field: label, input, and the line under it that carries
 * either the validation error or a help text, with the aria wiring between
 * them. The console-auth form repeats this shape for every credential it
 * stores, so it lives here once instead of inline per field.
 */
const props = defineProps<{
  id: string;
  label: string;
  modelValue: T;
  type?: "text" | "email" | "password" | "number";
  error?: string;
  help?: string;
  autocomplete?: string;
  maxlength?: number;
  inputmode?: "numeric" | "email" | "text";
  min?: number;
  max?: number;
  /** Commit the value as a number; an unparseable entry stays as typed. */
  numeric?: boolean;
  /** Grid placement of the wrapping cell, e.g. `sm:col-span-2`. */
  cellClass?: string;
}>();

const emit = defineEmits<{
  "update:modelValue": [value: T];
  input: [];
}>();

/** Vue's `.number` modifier semantics: keep the raw text when it is not a number. */
function coerce(text: string): string | number {
  if (!props.numeric) return text;
  const parsed = Number.parseFloat(text);
  return Number.isNaN(parsed) ? text : parsed;
}

const value = computed({
  get: () => String(props.modelValue ?? ""),
  set: (next: string | number | null) => {
    // The cast holds because a numeric field is bound to a numeric model and
    // an unparseable entry keeps the text the input already showed.
    emit("update:modelValue", coerce(String(next ?? "")) as T);
  },
});

const describedBy = computed(() => {
  if (props.error) return `${props.id}-error`;
  return props.help ? `${props.id}-help` : undefined;
});
</script>

<template>
  <div class="space-y-2" :class="cellClass">
    <Label :for="id">{{ label }}</Label>
    <Input
      :id="id"
      v-model="value"
      :type="type ?? 'text'"
      :autocomplete="autocomplete ?? 'off'"
      :maxlength="maxlength"
      :inputmode="inputmode"
      :min="min"
      :max="max"
      :class="type === 'number' ? 'text-data' : undefined"
      :aria-invalid="!!error"
      :aria-describedby="describedBy"
      @input="emit('input')"
    />
    <p v-if="error" :id="`${id}-error`" class="text-xs text-destructive">{{ error }}</p>
    <p v-else-if="help" :id="`${id}-help`" class="text-xs text-muted-foreground">{{ help }}</p>
  </div>
</template>
