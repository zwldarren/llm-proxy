<script setup lang="ts">
import { AlertCircle, RotateCcw, Shield } from "@lucide/vue";
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { SettingsItem, SettingsSection } from "@/components/settings";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { TagsInputSimple } from "@/components/ui/tags-input";
import type { AutoSaveState } from "@/composables/useSettingAutoSave";
import { useAutoSaveRefs } from "@/composables/useAutoSaveRefs";
import type { McpSecurityPolicyConfig } from "@/types/schemas";

const props = defineProps<{
  autoSave: AutoSaveState<McpSecurityPolicyConfig>;
}>();

const emit = defineEmits<{
  (e: "resetDefaults"): void;
}>();

const { t } = useI18n();

const { state, pending, error } = useAutoSaveRefs(props.autoSave);

function isValidCidr(value: string): boolean {
  // IPv4 CIDR: a.b.c.d/n where n is 0-32
  const ipv4Cidr =
    /^(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\/(?:3[0-2]|[1-2]?\d)$/;
  if (ipv4Cidr.test(value)) {
    return true;
  }
  // IPv6 CIDR: accept common compressed forms with /n (0-128)
  const ipv6Cidr = /^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\/(?:12[0-8]|1[01]\d|[1-9]?\d)$/;
  if (ipv6Cidr.test(value)) {
    return true;
  }
  // IPv6 compressed forms (::) - simplified check
  if (
    value.includes(":") &&
    value.includes("/") &&
    /^[0-9a-fA-F:]*::?[0-9a-fA-F:]*\/(?:12[0-8]|1[01]\d|[1-9]?\d)$/.test(value)
  ) {
    return true;
  }
  return false;
}

const commandOverlap = computed(() => {
  const allowed = new Set(state.value.allowed_commands.map((c) => c.toLowerCase()));
  return state.value.blocked_commands.map((c) => c.toLowerCase()).filter((c) => allowed.has(c));
});

const envKeyOverlap = computed(() => {
  const allowed = new Set(state.value.allowed_env_keys.map((k) => k.toUpperCase()));
  return state.value.blocked_env_keys.map((k) => k.toUpperCase()).filter((k) => allowed.has(k));
});
</script>

<template>
  <SettingsSection
    :title="t('mcpSecurity.title')"
    :icon="Shield"
    :description="t('mcpSecurity.description')"
  >
    <template #actions>
      <Button
        variant="ghost"
        size="sm"
        class="h-7 gap-1.5 text-xs text-muted-foreground hover:text-foreground"
        @click="emit('resetDefaults')"
      >
        <RotateCcw class="size-3.5" />
        {{ t("mcpSecurity.resetDefaults") }}
      </Button>
    </template>

    <!-- Require Key MCP Permissions -->
    <SettingsItem
      :title="t('mcpSecurity.requireKeyMcpPermissions')"
      :description="t('mcpSecurity.requireKeyMcpPermissionsDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <Switch v-model="state.require_key_mcp_permissions" />
      </template>
    </SettingsItem>

    <!-- Allowed Commands -->
    <SettingsItem
      :title="t('mcpSecurity.allowedCommands')"
      :description="t('mcpSecurity.allowedCommandsDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <TagsInputSimple
          v-model="state.allowed_commands"
          :placeholder="t('mcpSecurity.itemPlaceholder')"
          class="max-w-md"
        />
      </template>
    </SettingsItem>

    <!-- Blocked Commands -->
    <SettingsItem
      :title="t('mcpSecurity.blockedCommands')"
      :description="t('mcpSecurity.blockedCommandsDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <TagsInputSimple
          v-model="state.blocked_commands"
          :placeholder="t('mcpSecurity.itemPlaceholder')"
          class="max-w-md"
        />
        <p
          v-if="commandOverlap.length > 0"
          class="text-xs text-action-amber mt-1.5 flex items-center gap-1"
        >
          <AlertCircle class="w-3.5 h-3.5" />
          {{ t("mcpSecurity.overlapWarning", { items: commandOverlap.join(", ") }) }}
        </p>
      </template>
    </SettingsItem>

    <!-- Allowed Env Keys -->
    <SettingsItem
      :title="t('mcpSecurity.allowedEnvKeys')"
      :description="t('mcpSecurity.allowedEnvKeysDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <TagsInputSimple
          v-model="state.allowed_env_keys"
          :placeholder="t('mcpSecurity.itemPlaceholder')"
          class="max-w-md"
        />
      </template>
    </SettingsItem>

    <!-- Blocked Env Keys -->
    <SettingsItem
      :title="t('mcpSecurity.blockedEnvKeys')"
      :description="t('mcpSecurity.blockedEnvKeysDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <TagsInputSimple
          v-model="state.blocked_env_keys"
          :placeholder="t('mcpSecurity.itemPlaceholder')"
          class="max-w-md"
        />
        <p
          v-if="envKeyOverlap.length > 0"
          class="text-xs text-action-amber mt-1.5 flex items-center gap-1"
        >
          <AlertCircle class="w-3.5 h-3.5" />
          {{ t("mcpSecurity.overlapWarning", { items: envKeyOverlap.join(", ") }) }}
        </p>
      </template>
    </SettingsItem>

    <!-- Blocked URL Hosts -->
    <SettingsItem
      :title="t('mcpSecurity.blockedUrlHosts')"
      :description="t('mcpSecurity.blockedUrlHostsDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <TagsInputSimple
          v-model="state.blocked_url_hosts"
          :placeholder="t('mcpSecurity.itemPlaceholder')"
          class="max-w-md"
        />
      </template>
    </SettingsItem>

    <!-- Blocked URL IPs -->
    <SettingsItem
      :title="t('mcpSecurity.blockedUrlIps')"
      :description="t('mcpSecurity.blockedUrlIpsDescription')"
      :loading="pending"
      :error="error"
    >
      <template #action>
        <TagsInputSimple
          v-model="state.blocked_url_ips"
          :placeholder="t('mcpSecurity.itemPlaceholder')"
          :validate="isValidCidr"
          class="max-w-md"
        />
      </template>
    </SettingsItem>
  </SettingsSection>
</template>
