<script setup lang="ts">
import { ref } from "vue";
import { useI18n } from "vue-i18n";
import { Eye, EyeOff } from "@lucide/vue";
import { Input } from "@/components/ui/input";

/**
 * Password field with the same reveal control everywhere in the console.
 *
 * The reveal is local to the input and never changes autocomplete or what the
 * browser remembers. Consumers pass id, autocomplete and ARIA attributes
 * through; the button owns the pressed state and an accessible label.
 */
defineOptions({ inheritAttrs: false });

interface Props {
  modelValue: string;
  id?: string;
  autocomplete?: string;
  disabled?: boolean;
  required?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  id: undefined,
  autocomplete: "current-password",
  disabled: false,
  required: false,
});

const emit = defineEmits<{
  (e: "update:modelValue", value: string): void;
}>();

const { t } = useI18n();
const reveal = ref(false);
</script>

<template>
  <div class="relative">
    <Input
      v-bind="$attrs"
      :id="props.id"
      :model-value="props.modelValue"
      :type="reveal ? 'text' : 'password'"
      :autocomplete="props.autocomplete"
      :disabled="props.disabled"
      :required="props.required"
      class="pr-12"
      @update:model-value="emit('update:modelValue', String($event ?? ''))"
    />
    <button
      type="button"
      class="absolute inset-y-0 right-0 flex w-11 items-center justify-center rounded-r-md text-muted-foreground transition-colors outline-none hover:text-foreground focus-visible:bg-muted focus-visible:text-foreground focus-visible:ring-1 focus-visible:ring-ring/40 focus-visible:ring-inset disabled:pointer-events-none disabled:opacity-50"
      :disabled="props.disabled"
      :aria-label="reveal ? t('auth.hidePassword') : t('auth.showPassword')"
      :aria-pressed="reveal"
      @click="reveal = !reveal"
    >
      <component :is="reveal ? EyeOff : Eye" class="size-4" aria-hidden="true" />
    </button>
  </div>
</template>
