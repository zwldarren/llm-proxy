<script setup lang="ts">
import { Edit, Trash2 } from "@lucide/vue";
import { useI18n } from "vue-i18n";
import { Button } from "@/components/ui/button";

defineProps<{
  isLoading?: boolean;
}>();

const emit = defineEmits<{
  edit: [];
  delete: [];
}>();

const { t } = useI18n();
</script>

<template>
  <Button
    variant="ghost"
    size="icon"
    class="h-9 w-9"
    :disabled="isLoading"
    :aria-label="t('common.edit')"
    @click.stop="emit('edit')"
  >
    <Edit class="w-4 h-4" />
  </Button>
  <Button
    variant="ghost"
    size="icon"
    class="h-9 w-9 text-destructive hover:text-destructive hover:bg-destructive/10"
    :disabled="isLoading"
    :aria-label="t('common.delete')"
    @click.stop="emit('delete')"
  >
    <Trash2 class="w-4 h-4" />
  </Button>
</template>
