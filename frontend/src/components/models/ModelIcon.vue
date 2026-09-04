<script setup lang="ts">
import { ImageOff } from "@lucide/vue";
import { computed, ref, watch } from "vue";
import { getIconUrl, isMonoIcon } from "@/utils/icons";

/**
 * Model icon tile: custom icon_url, then a well-known lobe icon derived from
 * the model name, then an ImageOff placeholder. The error state resets when
 * the name/icon changes so a fixed URL shows without a remount.
 */
const props = withDefaults(
  defineProps<{
    name: string;
    iconUrl?: string | null;
    /** sm = 28px table-row tile, md = 32px list-row tile. */
    size?: "sm" | "md";
    /** Decorative icons sit next to a visible model name and get an empty alt. */
    decorative?: boolean;
  }>(),
  { iconUrl: null, size: "md", decorative: true }
);

const failed = ref(false);
watch([() => props.iconUrl, () => props.name], () => {
  failed.value = false;
});

const url = computed(() => getIconUrl(props.iconUrl, props.name));
const mono = computed(() => isMonoIcon(props.name));
</script>

<template>
  <span
    :class="[
      'flex items-center justify-center shrink-0 overflow-hidden',
      size === 'sm' ? 'w-7 h-7 rounded-md' : 'w-8 h-8 rounded-lg',
      url ? 'bg-card border border-border' : 'bg-primary/10',
    ]"
  >
    <img
      v-if="url && !failed"
      :src="url"
      :alt="decorative ? '' : name"
      :class="[
        mono ? 'icon-mono' : null,
        size === 'sm' ? 'w-4.5 h-4.5' : 'w-5 h-5',
        'object-contain',
      ]"
      loading="lazy"
      @error="failed = true"
    />
    <ImageOff v-else class="w-4 h-4 text-muted-foreground" />
  </span>
</template>
