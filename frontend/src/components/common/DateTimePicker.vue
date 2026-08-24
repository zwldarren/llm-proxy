<script setup lang="ts">
import { CalendarDate, getLocalTimeZone, today } from "@internationalized/date";
import { Calendar as CalendarIcon, Clock, X } from "@lucide/vue";
import { computed, ref } from "vue";
import { Button } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import { Input } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";

interface Props {
  /** ISO 8601 timestamp (UTC), or null when unset. */
  modelValue: string | null;
  placeholder?: string;
  timeLabel?: string;
  clearLabel?: string;
  disabled?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  placeholder: "Pick a date",
  timeLabel: "Time",
  clearLabel: "Clear",
  disabled: false,
});

const emit = defineEmits<{
  "update:modelValue": [value: string | null];
}>();

const open = ref(false);

const pad = (n: number) => String(n).padStart(2, "0");

/** The model value as a local Date, or null when unset/invalid. */
const localDate = computed(() => {
  if (!props.modelValue) return null;
  const d = new Date(props.modelValue);
  return Number.isNaN(d.getTime()) ? null : d;
});

const displayText = computed(() => {
  const d = localDate.value;
  if (!d) return null;
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
});

/**
 * Clamp a candidate so the emitted timestamp never lands in the past: an
 * expiry in the past would be dead on arrival. This covers picking today's
 * date with a time that has already passed (the calendar's min-date only
 * restricts the day, not the time-of-day).
 */
const notInPast = (d: Date): Date => {
  const now = new Date();
  return d.getTime() < now.getTime() ? now : d;
};

const calendarValue = computed<CalendarDate | undefined>({
  get: () => {
    const d = localDate.value;
    return d ? new CalendarDate(d.getFullYear(), d.getMonth() + 1, d.getDate()) : undefined;
  },
  set: (value) => {
    if (!value) return;
    // Keep the existing time; a fresh date defaults to end-of-day so an
    // expiry picked as "that day" stays valid through the day.
    const base = localDate.value;
    const hours = base ? base.getHours() : 23;
    const minutes = base ? base.getMinutes() : 59;
    emit(
      "update:modelValue",
      notInPast(new Date(value.year, value.month - 1, value.day, hours, minutes)).toISOString()
    );
  },
});

const timeValue = computed<string>({
  get: () => {
    const d = localDate.value;
    return d ? `${pad(d.getHours())}:${pad(d.getMinutes())}` : "";
  },
  set: (value) => {
    if (!value) {
      // Clearing the time input clears the timestamp too; silently keeping
      // the old value would desync the model from the empty field.
      emit("update:modelValue", null);
      return;
    }
    const [hours, minutes] = value.split(":").map(Number);
    if (!Number.isFinite(hours) || !Number.isFinite(minutes)) return;
    // type="time" normally constrains input, but guard the range explicitly
    // so out-of-range values cannot make new Date() silently roll over into
    // the next hour/day.
    if (hours < 0 || hours > 23 || minutes < 0 || minutes > 59) return;
    const base = localDate.value ?? new Date();
    emit(
      "update:modelValue",
      notInPast(
        new Date(base.getFullYear(), base.getMonth(), base.getDate(), hours, minutes, 0, 0)
      ).toISOString()
    );
  },
});

// Past dates cannot be picked: an expiry in the past would be dead on arrival.
const minDate = today(getLocalTimeZone());

const clear = () => {
  emit("update:modelValue", null);
  open.value = false;
};
</script>

<template>
  <Popover v-model:open="open">
    <PopoverTrigger as-child>
      <Button
        variant="outline"
        class="w-full justify-start text-left font-normal px-3"
        :class="{ 'text-muted-foreground': !displayText }"
        :disabled="disabled"
      >
        <CalendarIcon class="mr-2 h-4 w-4 shrink-0 text-muted-foreground/80" aria-hidden="true" />
        <span class="flex-1 truncate font-mono text-xs">{{ displayText ?? placeholder }}</span>
        <X
          v-if="displayText && !disabled"
          class="ml-2 h-3.5 w-3.5 shrink-0 opacity-60 hover:opacity-100 cursor-pointer"
          :aria-label="clearLabel"
          @click.stop="clear"
        />
      </Button>
    </PopoverTrigger>
    <PopoverContent class="w-auto p-0" align="start">
      <Calendar v-model="calendarValue" :min-value="minDate" />
      <div class="flex items-center gap-2 border-t border-border/60 px-3 py-2">
        <Clock class="h-4 w-4 shrink-0 text-muted-foreground/80" aria-hidden="true" />
        <Input
          v-model="timeValue"
          type="time"
          class="h-8 font-mono text-xs"
          :aria-label="timeLabel"
          :disabled="!localDate"
        />
      </div>
    </PopoverContent>
  </Popover>
</template>
