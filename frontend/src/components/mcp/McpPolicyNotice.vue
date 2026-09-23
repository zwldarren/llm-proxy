<script setup lang="ts">
import { AlertTriangle, CheckCircle2, Info } from "@lucide/vue";
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { RouterLink } from "vue-router";
import { Button } from "@/components/ui/button";
import type { McpSecurityPolicyConfig } from "@/types/schemas";
import { disallowedEnvKeys, isBlockedCommand, isCommandAllowed } from "@/utils/mcpPolicy";

/**
 * Inline, proactive feedback about the MCP security policy for the stdio
 * server form. The backend enforces the policy on save; this notice explains
 * *why* a server may be rejected before the operator hits Save, and links
 * straight to the policy settings.
 *
 * Render one instance next to the command field (`show-command`) and another
 * next to the env field (`show-env`) so each warning sits beside the input it
 * refers to.
 */
const props = withDefaults(
  defineProps<{
    policy: McpSecurityPolicyConfig;
    command?: string;
    args?: string[];
    envKeys?: string[];
    showCommand?: boolean;
    showEnv?: boolean;
  }>(),
  {
    command: "",
    args: () => [],
    envKeys: () => [],
    showCommand: true,
    showEnv: true,
  }
);

const { t } = useI18n();

const mcpSecurityLink = {
  name: "settings",
  query: { tab: "advanced" },
  hash: "#mcpSecurity",
};

const allowedCommands = computed(() => props.policy.allowed_commands.filter((c) => c.trim()));
const hasAllowlist = computed(() => allowedCommands.value.length > 0);

const commandState = computed<"empty" | "allowed" | "blocked" | "denied">(() => {
  const command = props.command.trim();
  if (!command) return "empty";
  if (isBlockedCommand(command, props.policy)) return "blocked";
  return isCommandAllowed(command, props.args, props.policy) ? "allowed" : "denied";
});

const droppedEnvKeys = computed(() => disallowedEnvKeys(props.envKeys, props.policy));
</script>

<template>
  <div v-if="showCommand || (showEnv && droppedEnvKeys.length > 0)" class="space-y-2">
    <template v-if="showCommand">
      <!-- No commands allowlisted at all -->
      <div
        v-if="!hasAllowlist"
        class="rounded-lg border border-action-amber/30 bg-action-amber/10 px-3 py-2.5 space-y-2"
      >
        <div class="flex items-center gap-1.5 text-action-amber text-xs font-medium">
          <AlertTriangle class="size-3.5 shrink-0" />
          {{ t("mcpServers.policy.noCommandsAllowedTitle") }}
        </div>
        <p class="text-[11px] text-action-amber/90 leading-relaxed">
          {{ t("mcpServers.policy.noCommandsAllowedDescription") }}
        </p>
        <Button
          as-child
          variant="outline"
          size="sm"
          class="h-6 gap-1 text-[11px] border-action-amber/30 text-action-amber hover:bg-action-amber/10 hover:text-action-amber"
        >
          <RouterLink :to="mcpSecurityLink">
            {{ t("mcpServers.policy.openSettings") }}
          </RouterLink>
        </Button>
      </div>

      <template v-else>
        <!-- Current allowlist, so operators can see what will work -->
        <div class="flex items-start gap-1.5 text-[11px] text-muted-foreground">
          <Info class="size-3.5 shrink-0 translate-y-px" />
          <span>
            {{ t("mcpServers.policy.allowedCommandsLabel") }}
            <span class="font-mono text-foreground/80">{{ allowedCommands.join(", ") }}</span>
          </span>
        </div>

        <!-- Entered command is explicitly blocked -->
        <div
          v-if="commandState === 'blocked'"
          class="rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2.5 space-y-1"
        >
          <div class="flex items-center gap-1.5 text-destructive text-xs font-medium">
            <AlertTriangle class="size-3.5 shrink-0" />
            {{ t("mcpServers.policy.commandBlockedTitle", { command: command.trim() }) }}
          </div>
          <p class="text-[11px] text-destructive/90 leading-relaxed">
            {{ t("mcpServers.policy.commandBlockedDescription") }}
          </p>
        </div>

        <!-- Entered command is not in the allowlist -->
        <div
          v-else-if="commandState === 'denied'"
          class="rounded-lg border border-action-amber/30 bg-action-amber/10 px-3 py-2.5 space-y-2"
        >
          <div class="flex items-center gap-1.5 text-action-amber text-xs font-medium">
            <AlertTriangle class="size-3.5 shrink-0" />
            {{ t("mcpServers.policy.commandNotAllowedTitle", { command: command.trim() }) }}
          </div>
          <p class="text-[11px] text-action-amber/90 leading-relaxed">
            {{ t("mcpServers.policy.commandNotAllowedDescription", { command: command.trim() }) }}
          </p>
          <Button
            as-child
            variant="outline"
            size="sm"
            class="h-6 gap-1 text-[11px] border-action-amber/30 text-action-amber hover:bg-action-amber/10 hover:text-action-amber"
          >
            <RouterLink :to="mcpSecurityLink">
              {{ t("mcpServers.policy.openSettings") }}
            </RouterLink>
          </Button>
        </div>

        <!-- Entered command is allowlisted -->
        <div
          v-else-if="commandState === 'allowed'"
          class="flex items-center gap-1.5 text-[11px] text-status-success"
        >
          <CheckCircle2 class="size-3.5 shrink-0" />
          {{ t("mcpServers.policy.commandAllowed") }}
        </div>
      </template>
    </template>

    <!-- Env keys that would be silently dropped by the policy -->
    <div
      v-if="showEnv && droppedEnvKeys.length > 0"
      class="rounded-lg border border-action-amber/30 bg-action-amber/10 px-3 py-2.5 space-y-2"
    >
      <div class="flex items-center gap-1.5 text-action-amber text-xs font-medium">
        <AlertTriangle class="size-3.5 shrink-0" />
        {{ t("mcpServers.policy.envKeysDroppedTitle") }}
      </div>
      <p class="text-[11px] text-action-amber/90 leading-relaxed">
        <span class="font-mono text-action-amber">{{ droppedEnvKeys.join(", ") }}</span>
        — {{ t("mcpServers.policy.envKeysDroppedDescription") }}
      </p>
      <Button
        as-child
        variant="outline"
        size="sm"
        class="h-6 gap-1 text-[11px] border-action-amber/30 text-action-amber hover:bg-action-amber/10 hover:text-action-amber"
      >
        <RouterLink :to="mcpSecurityLink">
          {{ t("mcpServers.policy.openSettings") }}
        </RouterLink>
      </Button>
    </div>
  </div>
</template>
