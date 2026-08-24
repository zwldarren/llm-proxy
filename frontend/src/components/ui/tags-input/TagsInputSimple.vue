<script setup lang="ts">
import { computed, type HTMLAttributes } from "vue";
import {
  TagsInput,
  TagsInputInput,
  TagsInputItem,
  TagsInputItemDelete,
  TagsInputItemText,
} from "./";
import { cn } from "@/lib/utils";

const props = withDefaults(
  defineProps<{
    modelValue?: string[];
    placeholder?: string;
    disabled?: boolean;
    validate?: (value: string) => boolean | string;
    class?: HTMLAttributes["class"];
  }>(),
  {
    modelValue: () => [],
    placeholder: "",
    disabled: false,
  }
);

const emit = defineEmits<{
  (e: "update:modelValue", value: string[]): void;
}>();

const tags = computed({
  get: () => props.modelValue,
  set: (val) => {
    let sanitized = [...new Set(val.map((s) => s.trim()).filter(Boolean))];
    const validate = props.validate;
    if (validate) {
      sanitized = sanitized.filter((s) => validate(s) === true);
    }
    emit("update:modelValue", sanitized);
  },
});

// Split on comma, semicolon, or newline
const delimiterRegExp = /[,\n;]+/;
</script>

<template>
  <TagsInput
    v-model="tags"
    :disabled="disabled"
    :add-on-paste="true"
    :add-on-blur="true"
    :delimiter="delimiterRegExp"
    :class="
      cn(
        'w-full bg-background/50 border-input shadow-[inset_0_1px_0_rgba(255,255,255,0.05)] backdrop-blur-sm transition-[color,box-shadow,border-color,background-color] p-1.5 min-h-[40px]',
        props.class
      )
    "
  >
    <TagsInputItem
      v-for="(item, index) in tags"
      :key="`${item}-${index}`"
      :value="item"
      class="font-mono text-xs border border-border/50 bg-secondary/80 hover:bg-secondary py-0.5 pl-2 pr-1 h-6 rounded-md"
    >
      <TagsInputItemText class="px-0.5 text-xs text-foreground" />
      <TagsInputItemDelete
        class="ml-1 text-muted-foreground hover:text-foreground hover:bg-muted-foreground/10 rounded-sm p-0.5"
      />
    </TagsInputItem>

    <TagsInputInput
      :placeholder="tags.length === 0 ? placeholder : ''"
      class="font-mono text-xs placeholder:text-muted-foreground flex-1 min-w-[120px]"
    />
  </TagsInput>
</template>
