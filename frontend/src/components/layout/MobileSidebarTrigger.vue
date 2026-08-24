<script setup lang="ts">
import { Menu } from "@lucide/vue";
import { useI18n } from "vue-i18n";
import { Button } from "@/components/ui/button";
import { useSidebar } from "@/components/ui/sidebar";

const { t } = useI18n();
const { toggleSidebar, openMobile } = useSidebar();
</script>

<template>
  <Button
    variant="ghost"
    size="icon"
    :aria-expanded="openMobile"
    aria-haspopup="dialog"
    :aria-label="t('nav.toggleMenu')"
    :title="t('nav.toggleMenu')"
    class="min-h-11 min-w-11"
    @click="toggleSidebar"
  >
    <Menu data-icon="inline-start" />
    <span class="sr-only">{{ t("nav.toggleMenu") }}</span>
  </Button>
</template>
