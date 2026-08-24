<script setup lang="ts">
import { ChevronDown, ChevronUp } from "@lucide/vue";
import { computed, useAttrs, type HTMLAttributes } from "vue";
import {
  NumberFieldDecrement,
  NumberFieldIncrement,
  NumberFieldInput,
  NumberFieldRoot,
} from "reka-ui";
import { cn } from "@/lib/utils";

defineOptions({
  inheritAttrs: false,
});

const props = defineProps<{
  defaultValue?: number;
  modelValue?: number | null;
  class?: HTMLAttributes["class"];
}>();

const emits = defineEmits<{
  (e: "update:modelValue", payload: number | null): void;
}>();

const attrs = useAttrs();

function toNumber(value: unknown): number | undefined {
  if (value === undefined || value === null || value === "") return undefined;
  const n = typeof value === "number" ? value : Number(value);
  return Number.isNaN(n) ? undefined : n;
}

const formatOptions = {
  useGrouping: false,
  maximumFractionDigits: 20,
} satisfies Intl.NumberFormatOptions;

// Warn when a deprecated `suffix` prop is passed via attrs
if (import.meta.env.DEV && "suffix" in attrs) {
  console.warn(
    "[NumberInput] The `suffix` prop is no longer supported. Use a wrapper element or the `class` prop to add suffix styling."
  );
}

const rootProps = computed(() => ({
  modelValue: props.modelValue ?? null,
  defaultValue: props.defaultValue,
  min: toNumber(attrs.min),
  max: toNumber(attrs.max),
  step: toNumber(attrs.step),
  stepSnapping: false,
  formatOptions,
  disabled: attrs.disabled as boolean | undefined,
  id: attrs.id as string | undefined,
  name: attrs.name as string | undefined,
  readonly: attrs.readonly as boolean | undefined,
  required: attrs.required as boolean | undefined,
}));

const inputAttrs = computed(() => {
  const rest: Record<string, unknown> = {};
  const rootKeys = new Set(Object.keys(rootProps.value));
  const skip = new Set<string>([...rootKeys, "class"]);
  for (const [key, value] of Object.entries(attrs)) {
    if (!skip.has(key)) rest[key] = value;
  }
  return rest;
});

function onUpdate(value: number | undefined) {
  emits("update:modelValue", value ?? null);
}
</script>

<template>
  <NumberFieldRoot v-bind="rootProps" class="relative w-full" @update:model-value="onUpdate">
    <NumberFieldInput
      v-bind="inputAttrs"
      data-slot="input"
      :class="
        cn(
          'file:text-foreground placeholder:text-muted-foreground dark:bg-input/40 border-input/80 h-10 w-full min-w-0 rounded-md border bg-background/75 px-3.5 py-2 text-base shadow-[inset_0_1px_0_hsl(var(--background)/0.9)] backdrop-blur-sm transition-[color,box-shadow,border-color,background-color] outline-none file:inline-flex file:h-7 file:border-0 file:bg-transparent file:text-sm file:font-medium disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50 md:text-sm',
          'focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]',
          'aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive',
          'pr-8',
          props.class
        )
      "
    />
    <div
      class="absolute inset-y-[2px] right-[2px] flex w-7 flex-col divide-y divide-input/60 overflow-hidden rounded-r-[calc(var(--radius)-4px)] border-l border-input/70 isolate"
    >
      <NumberFieldIncrement
        class="flex flex-1 items-center justify-center rounded-tr-[calc(var(--radius)-4px)] text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:pointer-events-none disabled:opacity-30 data-pressed:bg-muted"
      >
        <ChevronUp class="size-3" />
      </NumberFieldIncrement>
      <NumberFieldDecrement
        class="flex flex-1 items-center justify-center rounded-br-[calc(var(--radius)-4px)] text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:pointer-events-none disabled:opacity-30 data-pressed:bg-muted"
      >
        <ChevronDown class="size-3" />
      </NumberFieldDecrement>
    </div>
  </NumberFieldRoot>
</template>
