<script setup lang="ts">
import type { Component, HTMLAttributes } from "vue";
import DialogHeader from "../dialog/DialogHeader.vue";

const props = withDefaults(
  defineProps<{
    class?: HTMLAttributes["class"];
    variant?: "band" | "sr-only";
    icon?: Component;
    title?: string;
    description?: string;
  }>(),
  {
    variant: "band",
  }
);
</script>

<template>
  <DialogHeader
    data-slot="sheet-header"
    :variant="props.variant"
    :icon="props.icon"
    :title="props.title"
    :description="props.description"
    :class="props.class"
  >
    <template v-if="$slots.title" #title><slot name="title" /></template>
    <template v-if="$slots.description" #description><slot name="description" /></template>
    <template v-if="$slots.actions" #actions><slot name="actions" /></template>
    <template v-if="$slots.default" #default><slot /></template>
  </DialogHeader>
</template>
