<script setup lang="ts">
/**
 * Dedicated image preview for image_generation / image_edit log responses.
 *
 * Renders generated images as real <img> elements (data URLs for base64, or
 * direct URLs) — never feeds megabyte-sized base64 strings into a JSON tree
 * viewer, which is what froze the Logs details page on image requests.
 *
 * Supports lightbox zoom, download, and revised-prompt display. The lightbox
 * is a nested modal dialog: the surrounding log Sheet's focus trap is paused
 * while it is open (reka-ui focus-scope stack), Escape closes only the
 * lightbox, and focus returns to the triggering thumbnail on close.
 */
import { Download, ExternalLink, Maximize2 } from "@lucide/vue";
import { computed, nextTick, ref } from "vue";
import { useI18n } from "vue-i18n";
import { DialogContent, DialogDescription, DialogPortal, DialogRoot, DialogTitle } from "reka-ui";
import { Button } from "@/components/ui/button";
import type { ImageInfo } from "@/utils/logResponseParser";
import { buildImageDataUrl } from "@/utils/logResponseParser";

const props = defineProps<{
  images: ImageInfo[];
  /** Optional output_format hint echoed by the response */
  outputFormat?: string;
}>();

const { t } = useI18n();

interface ResolvedImage extends ImageInfo {
  /** Computed display URL (url or data URL) */
  displayUrl: string | undefined;
  /** Byte size estimate for base64 payloads (rough) */
  approxBytes: number;
  isBase64: boolean;
}

const resolved = computed<ResolvedImage[]>(() =>
  props.images.map((img) => {
    const url = buildImageDataUrl(img);
    const isBase64 = Boolean(img.b64Json);
    // base64 length * 0.75 ≈ decoded byte size
    const approxBytes = img.b64Json ? Math.round(img.b64Json.length * 0.75) : 0;
    return { ...img, displayUrl: url, approxBytes, isBase64 };
  })
);

interface LightboxState {
  /** URL currently displayed in the lightbox */
  url: string;
  /** Zero-based index of the displayed image within `resolved` */
  index: number;
  /** Image metadata of the displayed image (for alt text) */
  image: ResolvedImage;
}

// Lightbox state plus the trigger element that receives focus back on close.
const lightbox = ref<LightboxState | null>(null);
const lightboxTrigger = ref<HTMLElement | null>(null);

function openLightbox(img: ResolvedImage, index: number, event: MouseEvent) {
  if (!img.displayUrl) return;
  lightboxTrigger.value = event.currentTarget as HTMLElement | null;
  lightbox.value = { url: img.displayUrl, index, image: img };
}

function closeLightbox() {
  const trigger = lightboxTrigger.value;
  lightboxTrigger.value = null;
  lightbox.value = null;
  // The reka-ui dialog restores focus to the trigger on close; do it manually
  // as well so it also works when the trigger never held focus (e.g. Safari).
  nextTick(() => trigger?.focus({ preventScroll: true }));
}

function formatBytes(bytes: number): string {
  if (bytes <= 0) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function downloadName(index: number, img: ResolvedImage): string {
  const ext = img.outputFormat || props.outputFormat || "png";
  return `image-${index + 1}.${ext}`;
}

function downloadImage(img: ResolvedImage, index: number) {
  if (!img.displayUrl) return;
  const a = document.createElement("a");
  a.href = img.displayUrl;
  a.download = downloadName(index, img);
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function openInNewTab(url: string) {
  window.open(url, "_blank", "noopener,noreferrer");
}
</script>

<template>
  <div v-if="resolved.length > 0" class="space-y-3">
    <div class="flex items-center justify-between pb-1">
      <h4 class="text-xs uppercase font-bold text-muted-foreground flex items-center gap-2 pl-1">
        <Maximize2 class="size-3.5" />
        {{ t("logs.generatedImages") }}
        <span class="text-muted-foreground/60 font-mono normal-case tracking-normal">
          ({{ resolved.length }})
        </span>
      </h4>
    </div>

    <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
      <div
        v-for="(img, index) in resolved"
        :key="index"
        class="group relative rounded-lg border border-border/50 bg-muted/10 overflow-hidden flex flex-col"
      >
        <!-- Image canvas -->
        <button
          type="button"
          class="focus-visible-ring relative block w-full aspect-square bg-muted/20 overflow-hidden cursor-zoom-in"
          :aria-label="t('logs.openImageN', { n: index + 1 })"
          @click="openLightbox(img, index, $event)"
        >
          <img
            v-if="img.displayUrl"
            :src="img.displayUrl"
            :alt="img.revisedPrompt || t('logs.imageAltN', { n: index + 1 })"
            class="w-full h-full object-contain transition-opacity hover:opacity-90"
            loading="lazy"
          />
          <div
            v-else
            class="absolute inset-0 flex items-center justify-center text-muted-foreground text-xs italic"
          >
            {{ t("logs.noImageData") }}
          </div>
          <!-- Hover zoom hint -->
          <div
            class="absolute top-2 right-2 p-1.5 rounded-md bg-background/70 backdrop-blur-sm opacity-0 group-hover:opacity-100 transition-opacity"
          >
            <Maximize2 class="size-3.5 text-foreground/80" />
          </div>
        </button>

        <!-- Metadata + actions bar -->
        <div class="p-2.5 space-y-2 border-t border-border/40">
          <div class="flex items-center justify-between gap-2">
            <span class="text-[11px] font-mono text-muted-foreground uppercase tracking-wider">
              #{{ index + 1 }}
            </span>
            <div class="flex items-center gap-1">
              <Button
                v-if="img.displayUrl"
                variant="ghost"
                size="icon"
                class="h-7 w-7"
                :aria-label="t('logs.openInNewTabN', { n: index + 1 })"
                :title="t('logs.openInNewTab')"
                @click="openInNewTab(img.displayUrl)"
              >
                <ExternalLink class="size-3.5" />
              </Button>
              <Button
                v-if="img.displayUrl"
                variant="ghost"
                size="icon"
                class="h-7 w-7"
                :aria-label="t('logs.downloadImageN', { n: index + 1 })"
                :title="t('logs.downloadImage')"
                @click="downloadImage(img, index)"
              >
                <Download class="size-3.5" />
              </Button>
            </div>
          </div>

          <!-- Revised prompt -->
          <div
            v-if="img.revisedPrompt"
            class="text-[11px] text-muted-foreground/90 leading-relaxed"
          >
            <span class="text-muted-foreground/60 font-medium">{{ t("logs.revisedPrompt") }}:</span>
            <span class="font-mono">{{ img.revisedPrompt }}</span>
          </div>

          <!-- Source badge + size -->
          <div class="flex items-center gap-2 text-[11px] font-mono text-muted-foreground/70">
            <span
              class="px-1.5 py-0.5 rounded border border-border/30"
              :class="
                img.isBase64
                  ? 'bg-action-blue/5 text-action-blue'
                  : 'bg-status-success/5 text-status-success'
              "
            >
              {{ img.isBase64 ? "b64" : "url" }}
            </span>
            <span v-if="img.approxBytes > 0">{{ formatBytes(img.approxBytes) }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Lightbox — nested modal dialog teleported to <body>. Being a reka-ui
       Dialog layered over the log Sheet pauses the Sheet's focus trap while
       open, gives the dialog an accessible name and description, traps Tab
       inside, handles Escape (top layer only), and restores focus to the
       triggering thumbnail on close. -->
  <DialogRoot :open="lightbox !== null">
    <DialogPortal>
      <DialogContent
        class="fixed inset-0 z-50 flex items-center justify-center bg-background/95 backdrop-blur-sm p-4 animate-in fade-in"
        @click="closeLightbox"
      >
        <DialogTitle class="sr-only">{{ t("logs.imagePreview") }}</DialogTitle>
        <DialogDescription class="sr-only">
          {{ t("logs.imageOfN", { n: (lightbox?.index ?? 0) + 1, total: resolved.length }) }}
        </DialogDescription>
        <button
          type="button"
          class="focus-visible-ring absolute top-6 right-6 p-2 rounded-md text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
          :aria-label="t('common.close')"
          @click.stop="closeLightbox"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <path d="M18 6 6 18" />
            <path d="m6 6 12 12" />
          </svg>
        </button>
        <img
          v-if="lightbox"
          :src="lightbox.url"
          :alt="lightbox.image.revisedPrompt || t('logs.imageAltN', { n: lightbox.index + 1 })"
          class="max-w-full max-h-[90vh] object-contain shadow-2xl border border-border/20 animate-in zoom-in-95 duration-200"
          @click.stop
        />
      </DialogContent>
    </DialogPortal>
  </DialogRoot>
</template>
