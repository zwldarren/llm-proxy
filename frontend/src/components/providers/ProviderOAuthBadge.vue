<script setup lang="ts">
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { Badge } from "@/components/ui/badge";
import { OPENAI_CHATGPT_PROVIDER_TYPE } from "@/constants/providers";
import type { ProviderRead } from "@/types/schemas";

/**
 * Connection state of an OAuth-backed provider.
 *
 * Renders nothing for every other provider type, so list and table cells can
 * include it unconditionally and keep one definition of the label (and of the
 * type that has OAuth at all).
 */
const props = defineProps<{ provider: ProviderRead }>();

const { t } = useI18n();

const isOAuthProvider = computed(() => props.provider.type === OPENAI_CHATGPT_PROVIDER_TYPE);

const label = computed(() =>
  props.provider.oauth_status?.connected
    ? t("providers.oauthConnected")
    : t("providers.oauthSignInRequired")
);
</script>

<template>
  <Badge
    v-if="isOAuthProvider"
    variant="outline"
    class="text-[11px] font-medium px-1.5 py-0 text-muted-foreground shrink-0"
  >
    {{ label }}
  </Badge>
</template>
