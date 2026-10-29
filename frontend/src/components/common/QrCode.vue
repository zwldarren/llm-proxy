<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import { toCanvas } from "qrcode";

/**
 * Renders a QR symbol for the given text (an `otpauth://` URI for TOTP
 * enrollment).
 *
 * The symbol is drawn into a canvas rather than an <img> so it never leaves the
 * page: the TOTP provisioning URI embeds the shared secret. Callers own the
 * quiet zone — the symbol must stay dark-on-light in both themes.
 */
const props = defineProps<{
  value: string;
  /** Rendered width in CSS pixels. */
  size?: number;
  /** Accessible name; the canvas is exposed as an image. */
  label?: string;
}>();

const canvas = ref<HTMLCanvasElement | null>(null);
const failed = ref(false);

async function render() {
  if (!canvas.value || !props.value) return;
  try {
    await toCanvas(canvas.value, props.value, {
      width: props.size ?? 152,
      margin: 1,
      errorCorrectionLevel: "M",
    });
    failed.value = false;
  } catch {
    // An unrenderable URI must not break the surrounding enrollment form: the
    // manual secret is still shown next to it.
    failed.value = true;
  }
}

watch([() => props.value, () => props.size, canvas], render);
onMounted(render);
</script>

<template>
  <canvas v-show="!failed" ref="canvas" role="img" :aria-label="label" class="block" />
</template>
