<script setup lang="ts">
import { ChevronDown, ChevronUp } from "@lucide/vue";
import type { HTMLAttributes } from "vue";
import { useI18n } from "vue-i18n";
import { InputGroup, InputGroupInput, InputGroupAddon } from "@/components/ui/input-group";
import { cn } from "@/lib/utils";

/**
 * Compact number input with inline chevron steppers and an optional unit
 * suffix. Used across settings sections for numeric fields (retention days,
 * retry counts, cooldowns…). Emits `null` when the field is cleared so the
 * parent can decide on a fallback.
 */
const props = withDefaults(
  defineProps<{
    modelValue: number | null;
    min?: number;
    max?: number;
    step?: number;
    suffix?: string;
    id?: string;
    placeholder?: string;
    class?: HTMLAttributes["class"];
  }>(),
  {
    min: undefined,
    max: undefined,
    step: 1,
    suffix: "",
    id: undefined,
    placeholder: undefined,
    class: "",
  }
);

const emit = defineEmits<{
  (e: "update:modelValue", value: number | null): void;
}>();

const { t } = useI18n();

function clamp(value: number): number {
  let v = value;
  if (props.max !== undefined) v = Math.min(props.max, v);
  if (props.min !== undefined) v = Math.max(props.min, v);
  return v;
}

function base(): number {
  return props.modelValue ?? props.min ?? 0;
}

function increment() {
  emit("update:modelValue", clamp(base() + props.step));
}

function decrement() {
  emit("update:modelValue", clamp(base() - props.step));
}

function onInput(value: string | number | null) {
  if (value === "" || value === null) {
    emit("update:modelValue", null);
    return;
  }
  const n = Number(value);
  emit("update:modelValue", Number.isNaN(n) ? null : n);
}
</script>

<template>
  <InputGroup :class="cn('w-32', props.class)">
    <InputGroupInput
      :id="props.id"
      type="number"
      :min="props.min"
      :max="props.max"
      :placeholder="props.placeholder"
      :model-value="props.modelValue"
      @update:model-value="onInput"
    />
    <div
      class="flex flex-col items-center justify-center border-l border-border/40 h-full px-0.5 gap-0.5"
    >
      <button
        type="button"
        class="inline-flex items-center justify-center h-3.5 w-3.5 rounded text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
        :aria-label="t('common.increment')"
        @click="increment"
      >
        <ChevronUp class="size-3" />
      </button>
      <button
        type="button"
        class="inline-flex items-center justify-center h-3.5 w-3.5 rounded text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
        :aria-label="t('common.decrement')"
        @click="decrement"
      >
        <ChevronDown class="size-3" />
      </button>
    </div>
    <InputGroupAddon v-if="props.suffix" align="inline-end" class="pl-1.5">
      {{ props.suffix }}
    </InputGroupAddon>
  </InputGroup>
</template>
