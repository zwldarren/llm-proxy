<script setup lang="ts">
import type { HTMLAttributes } from "vue";
import { useAttrs, ref, watch } from "vue";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { parseClockInput } from "@/utils/clock";

/**
 * A 24-hour clock field: `HH:MM`, always, whatever the browser's locale.
 *
 * The native `<input type="time">` follows the browser locale, so an en-US
 * browser renders a 12-hour clock with AM/PM and cannot be asked to do
 * otherwise. This field keeps the clock the wire format uses: type `9`, `930`
 * or `09:30` and it commits `09:30`; unreadable input reverts on blur instead
 * of reaching the form.
 */
defineOptions({ name: "TimeInput", inheritAttrs: false });

const props = defineProps<{
  /** A normalized `HH:MM` clock, or null when unset. */
  modelValue?: string | null;
  class?: HTMLAttributes["class"];
}>();

const emits = defineEmits<{
  (e: "update:modelValue", payload: string | null): void;
}>();

const attrs = useAttrs();
const buffer = ref(props.modelValue ?? "");

/**
 * While the field is focused the buffer holds the operator's own text, not the
 * normalized value — rewriting `1` to `01:00` mid-word would fight the keyboard.
 */
let editing = false;

watch(
  () => props.modelValue,
  (value) => {
    if (!editing) buffer.value = value ?? "";
  }
);

function onInput(event: Event) {
  editing = true;
  buffer.value = (event.target as HTMLInputElement).value;
}

/** Normalize on commit; an unreadable clock falls back to the stored value. */
function commit() {
  editing = false;
  const parsed = parseClockInput(buffer.value);
  if (!parsed) {
    buffer.value = props.modelValue ?? "";
    return;
  }
  buffer.value = parsed;
  if (parsed !== props.modelValue) emits("update:modelValue", parsed);
}

function onFocus(event: FocusEvent) {
  editing = true;
  (event.target as HTMLInputElement).select();
}

function onKeydown(event: KeyboardEvent) {
  const input = event.target as HTMLInputElement;
  if (event.key === "Enter") {
    event.preventDefault();
    input.blur();
  } else if (event.key === "Escape") {
    buffer.value = props.modelValue ?? "";
    editing = false;
    input.blur();
  }
}
</script>

<template>
  <Input
    v-bind="attrs"
    :model-value="buffer"
    type="text"
    inputmode="numeric"
    maxlength="5"
    autocomplete="off"
    spellcheck="false"
    :class="cn('text-data', props.class)"
    @input="onInput"
    @focus="onFocus"
    @blur="commit"
    @keydown="onKeydown"
  />
</template>
