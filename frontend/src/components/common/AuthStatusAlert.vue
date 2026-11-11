<script setup lang="ts">
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

withDefaults(defineProps<Props>(), {
  tone: "info",
  id: undefined,
});
</script>

<template>
  <Alert :variant="tone === 'error' ? 'destructive' : 'default'" :id="id">
    <AlertDescription>
      <span class="flex items-start gap-2">
        <CircleAlert v-if="tone === 'error'" class="mt-0.5 size-4 shrink-0" aria-hidden="true" />
        <CircleCheck
          v-else-if="tone === 'success'"
          class="mt-0.5 size-4 shrink-0 text-status-success"
          aria-hidden="true"
        />
        <Info v-else class="mt-0.5 size-4 shrink-0" aria-hidden="true" />
        <span class="min-w-0 break-words"><slot /></span>
      </span>
    </AlertDescription>
  </Alert>
</template>
