<script setup lang="ts">
import { defineAsyncComponent } from "vue";
import TableSkeleton from "@/components/common/TableSkeleton.vue";
import { useAuthStore } from "@/stores/auth";

/**
 * Single entry point for everything model-related at /models.
 *
 * The old split between "Model Catalog" (/models) and "Models" config
 * (/config/models) showed the same underlying model table twice — the
 * catalog was a strict subset minus pricing and actions. The views are now
 * merged behind one route, switched by role:
 *
 * - admins get the full management UI (pricing, routing, providers, CRUD)
 * - other authenticated users get the read-only browse view (no pricing,
 *   per-user allowlist filtering still applied server-side)
 *
 * Keep the component name "ModelsView": App.vue's KeepAlive include list
 * matches on it to cache this route across navigations.
 */

defineOptions({ name: "ModelsView" });

// Delayed skeleton fallback: invisible when chunks are warm (hover/idle
// prefetch), geometry-mirroring feedback instead of a blank page on slow
// cold loads.
const asyncViewOptions = { loadingComponent: TableSkeleton, delay: 200 };

const ModelsAdminView = defineAsyncComponent({
  loader: () => import("@/views/config/ModelsView.vue"),
  ...asyncViewOptions,
});
const ModelBrowseView = defineAsyncComponent({
  loader: () => import("@/views/ModelPlazaView.vue"),
  ...asyncViewOptions,
});

const authStore = useAuthStore();
</script>

<template>
  <component :is="authStore.isAdmin ? ModelsAdminView : ModelBrowseView" />
</template>
