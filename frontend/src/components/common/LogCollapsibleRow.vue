/** * Collapsible row shared by the Logs I/O views. * * Encodes the mechanics that used to repeat
across every accordion * (toggle row + rotating chevron + panel) on top of the shadcn-vue/reka *
Collapsible primitives, which provide the aria wiring. * * The default `cardClass` is THE shared row
shell — one visual language for every * standalone collapsible (system prompt, audit raw-header
accordions): a hairline row * separated by top/bottom borders — no rounded box, no shadow
(One-Console * pattern: one seamless surface separated only by hairlines). Rows INSIDE a *
`divide-y` list container (messages, output items) pass `card-class="bg-transparent"` * instead: the
container owns the hairlines, the row draws nothing. Inner content * never gets its own bordered box
either (no card-in-card). * * - `chevronPosition` "start" for item-style rows (label follows
chevron), * "end" for the raw accordions with a justify-between header. */
<script setup lang="ts">
import { ChevronDown } from "@lucide/vue";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";

withDefaults(
  defineProps<{
    open: boolean;
    /** Card container classes (border, background, rounding). */
    cardClass?: string;
    /** Toggle-row classes (padding, typography, hover state). */
    buttonClass: string;
    /** Chevron size/color tweaks on top of the shared base classes. */
    chevronClass?: string;
    /** Panel wrapper classes (padding, divider, etc.). */
    contentClass?: string;
    chevronPosition?: "start" | "end";
  }>(),
  {
    cardClass: "border-y border-border/40 transition-colors",
    chevronClass: "size-3.5 shrink-0 text-muted-foreground/60",
    contentClass: "",
    chevronPosition: "start",
  }
);

const emit = defineEmits<{ (e: "toggle"): void }>();
</script>

<template>
  <Collapsible :open="open" @update:open="emit('toggle')" :class="cardClass">
    <CollapsibleTrigger
      class="w-full flex items-center text-left cursor-pointer transition-colors"
      :class="chevronPosition === 'start' ? 'gap-2' : 'gap-2 justify-between'"
    >
      <template v-if="chevronPosition === 'start'">
        <ChevronDown
          class="size-3.5 shrink-0 transition-transform duration-200"
          :class="[chevronClass, { 'rotate-180': open }]"
        />
        <slot name="header" />
      </template>
      <template v-else>
        <span class="flex items-center gap-2 min-w-0">
          <slot name="header" />
        </span>
        <ChevronDown
          class="size-3.5 shrink-0 text-muted-foreground/60 transition-transform duration-200"
          :class="[chevronClass, { 'rotate-180': open }]"
        />
      </template>
    </CollapsibleTrigger>
    <CollapsibleContent :class="contentClass">
      <slot />
    </CollapsibleContent>
  </Collapsible>
</template>
