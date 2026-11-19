<script setup lang="ts">
import { computed } from "vue";
import BrandMark from "@/components/common/BrandMark.vue";
import { modelIcon } from "@/utils/icons";

/**
 * Model icon tile: a configured `icon_url`, else the model's brand from the icon
 * catalog, else the provider's brand, else the model name's initial (see
 * `BrandMark`).
 */
const props = withDefaults(
  defineProps<{
    name: string;
    iconUrl?: string | null;
    /** Provider hint for a model name that carries no brand of its own. */
    provider?: string | null;
    /** sm = 28px table-row tile, md = 32px list-row tile. */
    size?: "sm" | "md";
    /** Decorative icons sit next to a visible model name and get an empty alt. */
    decorative?: boolean;
  }>(),
  { iconUrl: null, provider: null, size: "md", decorative: true }
);

const icon = computed(() =>
  modelIcon({ name: props.name, provider: props.provider, iconUrl: props.iconUrl })
);
</script>

<template>
  <BrandMark :icon="icon" :label="name" :size="size" :decorative="decorative" />
</template>
