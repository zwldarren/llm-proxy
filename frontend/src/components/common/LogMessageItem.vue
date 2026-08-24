<script setup lang="ts">
import { formatMessageContent } from "@/utils/logFormat";
import { Brain, Settings, User, Wrench } from "@lucide/vue";

defineProps<{
  msg: {
    role: string;
    content: unknown;
    tool_calls?: Array<{
      id?: string;
      function?: { name?: string; arguments?: string };
    }>;
  };
}>();
</script>

<template>
  <div
    :class="[
      'p-3.5 rounded-xl border flex flex-col gap-2 relative overflow-hidden transition-all duration-300 hover:border-border/80 shadow-xs',
      msg.role === 'user'
        ? 'border-action-blue/20 bg-action-blue/5'
        : msg.role === 'assistant'
          ? 'border-action-violet/20 bg-action-violet/5'
          : 'border-border/60 bg-muted/10',
    ]"
  >
    <!-- Role title & indicator -->
    <div class="flex items-center justify-between">
      <span
        :class="[
          'text-[11px] font-bold uppercase tracking-wider px-2 py-0.5 rounded border shadow-xs',
          msg.role === 'user'
            ? 'bg-action-blue/10 border-action-blue/20 text-action-blue'
            : msg.role === 'assistant'
              ? 'bg-action-violet/10 border-action-violet/20 text-action-violet'
              : 'bg-muted border-border/80 text-muted-foreground',
        ]"
      >
        {{ msg.role }}
      </span>

      <User v-if="msg.role === 'user'" class="size-3.5 text-action-blue/60" />
      <Brain v-else-if="msg.role === 'assistant'" class="size-3.5 text-action-violet/60" />
      <Settings v-else class="size-3.5 text-muted-foreground/60" />
    </div>

    <!-- Content text -->
    <div class="text-xs font-mono text-foreground/90 whitespace-pre-wrap leading-relaxed">
      {{ formatMessageContent(msg.content) }}
    </div>

    <!-- Tool Calls inside message if any -->
    <div
      v-if="msg.tool_calls && msg.tool_calls.length > 0"
      class="mt-2 space-y-2 border-t border-border/20 pt-2"
    >
      <span class="text-[11px] font-bold text-muted-foreground uppercase tracking-wider block"
        >Tool Call from Turn</span
      >
      <div
        v-for="(tc, tcIdx) in msg.tool_calls"
        :key="tcIdx"
        class="bg-card/60 border border-border/40 p-2.5 rounded-lg text-xs"
      >
        <div class="flex items-center gap-1.5 mb-1.5">
          <Wrench class="size-3 text-action-amber" />
          <span class="font-bold font-mono text-action-amber">{{ tc.function?.name }}</span>
        </div>
        <pre
          class="font-mono text-[11px] text-muted-foreground overflow-x-auto bg-muted/40 p-1.5 rounded"
          >{{ tc.function?.arguments }}</pre>
      </div>
    </div>
  </div>
</template>
