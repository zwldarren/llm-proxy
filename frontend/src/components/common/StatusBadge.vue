<script setup lang="ts">
import { computed } from "vue";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface Props {
  variant?: "status" | "http" | "default";
  status?: "success" | "warning" | "error" | "unknown";
  httpMethod?: string;
  /** Set the slot in the mono voice (IBM Plex Mono, tabular-nums) for numeric/technical tokens. */
  mono?: boolean;
  class?: string;
}

const props = withDefaults(defineProps<Props>(), {
  variant: "default",
  status: "unknown",
  mono: false,
  class: "",
});

const statusClasses = computed(() => {
  if (props.variant === "status") {
    switch (props.status) {
      case "success":
        return "bg-status-success/15 text-status-success border-status-success/30";
      case "warning":
        return "bg-status-warning/15 text-status-warning border-status-warning/30";
      case "error":
        return "bg-status-error/15 text-status-error border-status-error/30";
      default:
        return "bg-status-unknown/15 text-status-unknown border-status-unknown/30";
    }
  }

  if (props.variant === "http") {
    switch (props.httpMethod) {
      case "GET":
        return "bg-http-get/15 text-http-get border-http-get/25";
      case "POST":
        return "bg-http-post/15 text-http-post border-http-post/25";
      case "PUT":
      case "PATCH":
        return "bg-http-put/15 text-http-put border-http-put/25";
      case "DELETE":
        return "bg-http-delete/15 text-http-delete border-http-delete/25";
      default:
        return "bg-muted text-muted-foreground border-border";
    }
  }

  return "";
});

// DESIGN.md Mono-Data Rule: numeric/technical tokens (status codes, HTTP
// methods) render in the mono voice; word labels keep the sans voice.
// `variant="http"` is mono by definition since its content is an HTTP method.
const monoClasses = computed(() =>
  props.mono || props.variant === "http" ? "font-mono tabular-nums" : "font-medium"
);
</script>

<template>
  <Badge
    variant="outline"
    :class="cn(statusClasses, 'px-2 py-0.5 text-xs max-w-full', monoClasses, props.class)"
    role="status"
  >
    <span class="truncate"><slot /></span>
  </Badge>
</template>
