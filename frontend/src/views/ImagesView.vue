<script setup lang="ts">
import {
  ImageIcon,
  Loader2,
  Plus,
  Settings2,
  Trash2,
  Upload,
  X,
  Cpu,
  AlertTriangle,
  FileJson2,
} from "@lucide/vue";

defineOptions({ name: "ImagesView" });
import { computed, onActivated, onMounted, ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import { toast } from "vue-sonner";
import EmptyState from "@/components/common/EmptyState.vue";
import ImagesSettings from "@/components/images/ImagesSettings.vue";
import RunInspector from "@/components/playground/RunInspector.vue";
import RunSpecimen from "@/components/playground/RunSpecimen.vue";
import SpecimenTray from "@/components/playground/SpecimenTray.vue";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { chatApi } from "@/services/api/chat";
import { getErrorMessage } from "@/utils/error";
import { imagesApi } from "@/services/api/images";
import { STORAGE_KEYS } from "@/constants/storageKeys";
import { formatLatency, endpointName, makeRunId, appendRun } from "@/utils/runs";
import { useAuthStore } from "@/stores/auth";
import type { ImageData, UploadedImage } from "@/types/schemas";
import type { ImageRun } from "@/types/runs";

const { t } = useI18n();

// Models
interface ModelOption {
  id: string;
  provider: string;
}

const models = ref<ModelOption[]>([]);
const selectedModel = ref<string | null>(localStorage.getItem(STORAGE_KEYS.IMAGES_MODEL));
const isLoadingModels = ref(false);

import { getModelIconUrl, isMonoIcon } from "@/utils/icons";
import { useModelStore } from "@/stores/models";

const modelStore = useModelStore();

const getModelIcon = (modelId: string | null | undefined) => {
  if (!modelId) return null;
  const model = models.value.find((m) => m.id === modelId);
  return getModelIconUrl(modelId, model?.provider);
};

watch(selectedModel, (val) => {
  if (val) {
    localStorage.setItem(STORAGE_KEYS.IMAGES_MODEL, val);
  } else {
    localStorage.removeItem(STORAGE_KEYS.IMAGES_MODEL);
  }
});

// Prompt
const prompt = ref("");

// Uploaded images
const uploadedImages = ref<UploadedImage[]>([]);
const maskImage = ref<UploadedImage | null>(null);
const imageInputRef = ref<HTMLInputElement | null>(null);
const maskInputRef = ref<HTMLInputElement | null>(null);
const isDragOverMain = ref(false);

// Settings
const showSettings = ref(false);
const numberOfImages = ref(1);
const size = ref("auto");
const quality = ref("auto");
const background = ref<string | null>(null);
const moderation = ref<string | null>(null);
const outputCompression = ref<number | null>(null);
const outputFormat = ref<string | null>(null);
const partialImages = ref<number | null>(null);

// Results — run history: every generation/edit deposits a specimen in the tray.
/** Keep the tray bounded — base64 payloads are heavy. */
const MAX_RUNS = 12;

const runs = ref<ImageRun[]>([]);
const activeRunId = ref<string | null>(null);
const isGenerating = ref(false);

// Computed
const activeRun = computed(() => runs.value.find((r) => r.id === activeRunId.value) ?? null);

const activeRunNumber = computed(() => runs.value.findIndex((r) => r.id === activeRunId.value) + 1);

const isEditMode = computed(() => uploadedImages.value.length > 0);

const selectRun = (id: string) => {
  activeRunId.value = id;
};

// The run inspector opens from the canvas readout; specimens stay selectors.
const showInspector = ref(false);

const runEndpoint = (run: ImageRun): string =>
  run.mode === "edits" ? "/v1/images/edits" : "/v1/images/generations";

const runThumb = (run: ImageRun): string | null => {
  const first = run.images[0];
  return first ? getImageSource(first) : null;
};

const clearRuns = () => {
  runs.value = [];
  activeRunId.value = null;
};

const authStore = useAuthStore();

const apiKey = computed(() => authStore.sessionApiKey ?? "");

const isSubmitDisabled = computed(
  () => !prompt.value.trim() || isGenerating.value || !selectedModel.value || !apiKey.value.trim()
);

// File helpers
function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.readAsDataURL(file);
  });
}

import { uuid } from "@/utils/uuid";

async function handleImageFiles(files: FileList | File[]) {
  const fileArr = Array.from(files);
  const newImages: UploadedImage[] = [];
  for (const file of fileArr) {
    if (!file.type.startsWith("image/")) continue;
    const base64 = await fileToBase64(file);
    newImages.push({ id: uuid(), base64, file });
  }
  uploadedImages.value = [...uploadedImages.value, ...newImages].slice(0, 16);
}

async function handleMaskFile(file: File) {
  if (!file.type.startsWith("image/")) return;
  const base64 = await fileToBase64(file);
  maskImage.value = { id: uuid(), base64, file };
}

function removeUploadedImage(id: string) {
  uploadedImages.value = uploadedImages.value.filter((img) => img.id !== id);
}

function removeMaskImage() {
  maskImage.value = null;
  if (maskInputRef.value) maskInputRef.value.value = "";
}

function onImagesInputChange(e: Event) {
  const input = e.target as HTMLInputElement;
  if (input.files?.length) handleImageFiles(input.files);
  input.value = "";
}

function onMaskInputChange(e: Event) {
  const input = e.target as HTMLInputElement;
  if (input.files?.[0]) handleMaskFile(input.files[0]);
  input.value = "";
}

function onDragOverMain(e: DragEvent) {
  e.preventDefault();
  isDragOverMain.value = true;
}

function onDragLeaveMain() {
  isDragOverMain.value = false;
}

function onDropMain(e: DragEvent) {
  e.preventDefault();
  isDragOverMain.value = false;
  if (e.dataTransfer?.files?.length) handleImageFiles(e.dataTransfer.files);
}

// API Key
async function loadModels() {
  try {
    isLoadingModels.value = true;
    const res = await chatApi.getModels(apiKey.value);

    // Map each model and resolve provider name via modelStore mapping
    models.value = res.data.map((m) => {
      let provider = m.provider;
      if (!provider) {
        const configModel = modelStore.models.find((cm) => cm.name === m.id);
        provider = configModel?.providers?.[0]?.provider_name || "";
      }
      return {
        id: m.id,
        provider: provider,
      };
    });

    const savedModel = localStorage.getItem(STORAGE_KEYS.IMAGES_MODEL);
    if (savedModel && models.value.some((m) => m.id === savedModel)) {
      selectedModel.value = savedModel;
    } else if (models.value.length > 0) {
      selectedModel.value = models.value[0]?.id ?? null;
    }
  } catch (error) {
    console.error("Failed to load models:", error);
    models.value = [];
    selectedModel.value = null;
  } finally {
    isLoadingModels.value = false;
  }
}

function resetSettings() {
  numberOfImages.value = 1;
  size.value = "auto";
  quality.value = "auto";
  background.value = null;
  moderation.value = null;
  outputCompression.value = null;
  outputFormat.value = null;
  partialImages.value = null;
}

async function handleSubmit() {
  if (!prompt.value.trim()) {
    toast.warning(t("images.noPrompt"));
    return;
  }
  if (!selectedModel.value) {
    toast.warning(t("images.noModel"));
    return;
  }
  if (!apiKey.value.trim()) {
    toast.warning(t("images.apiKeyRequired"));
    return;
  }

  isGenerating.value = true;

  // Deposit the run specimen immediately — it settles to ok or error.
  const run: ImageRun = {
    id: makeRunId(),
    mode: isEditMode.value ? "edits" : "generations",
    model: selectedModel.value,
    prompt: prompt.value.trim(),
    n: numberOfImages.value,
    size: size.value,
    quality: quality.value,
    startedAt: Date.now(),
    status: "streaming",
    payload: {},
    images: [],
  };
  runs.value = appendRun(runs.value, run, MAX_RUNS);
  activeRunId.value = run.id;
  const runStart = performance.now();

  try {
    const model = selectedModel.value;
    const trimmedPrompt = prompt.value.trim();
    const key = apiKey.value.trim();

    let images: ImageData[];
    if (run.mode === "edits") {
      const body = {
        model,
        prompt: trimmedPrompt,
        images: uploadedImages.value.map((img) => ({
          image_url: img.base64,
        })),
        mask: maskImage.value ? { image_url: maskImage.value.base64 } : undefined,
        background: background.value || undefined,
        moderation: moderation.value || undefined,
        n: numberOfImages.value,
        output_compression: outputCompression.value ?? undefined,
        output_format: outputFormat.value || undefined,
        partial_images: partialImages.value ?? undefined,
        quality: quality.value,
        size: size.value !== "auto" ? size.value : undefined,
      };
      run.payload = JSON.parse(JSON.stringify(body));
      const response = await imagesApi.editImage(body, key);
      images = response.data || [];
    } else {
      const body = {
        prompt: trimmedPrompt,
        model,
        n: numberOfImages.value,
        size: size.value !== "auto" ? size.value : undefined,
        quality: quality.value,
        background: background.value || undefined,
        moderation: moderation.value || undefined,
        output_compression: outputCompression.value ?? undefined,
        output_format: outputFormat.value || undefined,
        partial_images: partialImages.value ?? undefined,
        response_format: "url" as const,
      };
      run.payload = JSON.parse(JSON.stringify(body));
      const response = await imagesApi.generateImage(body, key);
      images = response.data || [];
    }

    run.images = images;
    run.status = "ok";
    run.latencyMs = Math.round(performance.now() - runStart);

    if (images.length === 0) {
      toast.warning(t("images.noResults"));
    }
  } catch (error) {
    run.status = "error";
    run.latencyMs = Math.round(performance.now() - runStart);
    run.errorMessage = getErrorMessage(error);
    toast.error(t("images.generationFailed"), { description: run.errorMessage });
  } finally {
    isGenerating.value = false;
  }
}

function getImageSource(image: ImageData): string | null {
  if (image.url) return image.url;
  if (image.b64_json) return `data:image/png;base64,${image.b64_json}`;
  return null;
}

async function downloadImage(image: ImageData, index: number) {
  try {
    let blob: Blob;
    if (image.b64_json) {
      const binaryStr = atob(image.b64_json);
      const bytes = Uint8Array.from(binaryStr, (c) => c.charCodeAt(0));
      blob = new Blob([bytes], { type: "image/png" });
    } else if (image.url) {
      const response = await fetch(image.url);
      blob = await response.blob();
    } else {
      return;
    }
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `image-${index + 1}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch {
    toast.error(t("common.error"));
  }
}

function openInNewTab(image: ImageData) {
  const src = getImageSource(image);
  if (src) window.open(src, "_blank");
}

const isFirstLoad = ref(true);

onMounted(async () => {
  if (isFirstLoad.value) {
    if (authStore.isAdmin) {
      try {
        await modelStore.fetchModels();
      } catch (err) {
        console.error("Failed to fetch models store in images view:", err);
      }
    }
    await loadModels();
    isFirstLoad.value = false;
  }
});

onActivated(async () => {
  if (!isFirstLoad.value) {
    if (authStore.isAdmin) {
      try {
        await modelStore.fetchModels(true);
      } catch (err) {
        console.error("Failed to fetch models store in images view on activation:", err);
      }
    }
  }
});

watch(
  () => modelStore.models,
  async () => {
    await loadModels();
  }
);
</script>

<template>
  <AppLayout layoutMode="full">
    <template #header>
      <header
        class="config-header-bar flex items-center justify-between px-4 sm:px-6 h-14 z-20 shrink-0"
      >
        <div class="flex items-center gap-2">
          <!-- Model Selector -->
          <Select v-if="models.length > 0" v-model="selectedModel">
            <SelectTrigger
              class="border border-border/60 bg-transparent hover:bg-muted/10 shadow-none focus-visible:ring-1 focus-visible:ring-foreground focus-visible:ring-offset-0 rounded-md h-8 px-2.5 gap-3 transition-colors text-foreground flex items-center min-w-0 font-mono text-[11px]"
            >
              <div class="flex items-center gap-2 min-w-0">
                <div
                  v-if="selectedModel"
                  class="w-4 h-4 rounded flex items-center justify-center shrink-0 overflow-hidden bg-background/50 border border-border/40"
                >
                  <img
                    v-if="getModelIcon(selectedModel)"
                    :src="getModelIcon(selectedModel)!"
                    :alt="selectedModel"
                    :class="[
                      isMonoIcon(selectedModel) ? 'icon-mono' : null,
                      'w-3.5 h-3.5 object-contain',
                    ]"
                    loading="lazy"
                  />
                  <Cpu v-else class="w-2.5 h-2.5 text-muted-foreground" />
                </div>
                <span class="font-medium truncate min-w-0 max-w-[200px]" v-if="selectedModel">
                  {{ selectedModel }}
                </span>
                <span v-else class="text-muted-foreground">{{ t("images.selectModel") }}</span>
              </div>
            </SelectTrigger>
            <SelectContent class="min-w-64 rounded-md border border-border/80 shadow-md">
              <SelectItem
                v-for="modelOption in models"
                :key="modelOption.id"
                :value="modelOption.id"
                class="rounded-sm"
              >
                <div class="flex items-center gap-2.5 min-w-0 w-full py-0.5">
                  <div
                    class="w-4 h-4 rounded flex items-center justify-center shrink-0 overflow-hidden bg-background/50 border border-border/40"
                  >
                    <img
                      v-if="getModelIcon(modelOption.id)"
                      :src="getModelIcon(modelOption.id)!"
                      :alt="modelOption.id"
                      :class="[
                        isMonoIcon(modelOption.id) ? 'icon-mono' : null,
                        'w-3.5 h-3.5 object-contain',
                      ]"
                      loading="lazy"
                    />
                    <Cpu v-else class="w-2.5 h-2.5 text-muted-foreground" />
                  </div>
                  <span
                    class="min-w-0 truncate max-w-[40vw] sm:max-w-[250px] text-[11px] font-mono text-foreground leading-none"
                    :title="modelOption.id"
                    >{{ modelOption.id }}</span
                  >
                </div>
              </SelectItem>
            </SelectContent>
          </Select>

          <!-- Model Selector Placeholder/Loading -->
          <div
            v-else
            class="border border-border/60 bg-transparent opacity-85 rounded-md h-8 px-2.5 gap-3 text-muted-foreground flex items-center min-w-0 font-mono text-[11px] select-none cursor-not-allowed"
          >
            <div class="flex items-center gap-2 min-w-0">
              <Loader2
                v-if="isLoadingModels"
                class="w-3 h-3 animate-spin shrink-0 text-muted-foreground/70"
              />
              <span class="font-medium text-muted-foreground/70 truncate min-w-0 max-w-[200px]">
                {{ isLoadingModels ? `${t("common.loading")}…` : t("images.selectModel") }}
              </span>
            </div>
          </div>

          <!-- Hairline divider between the primary model picker and the secondary endpoint -->
          <div class="hidden sm:block h-4 w-px bg-border/60 shrink-0" aria-hidden="true" />

          <!-- Endpoint Selector (secondary, mono) -->
          <Select
            :model-value="isEditMode ? '/v1/images/edits' : '/v1/images/generations'"
            disabled
          >
            <SelectTrigger
              class="border border-border/60 bg-transparent opacity-80 cursor-default shadow-none focus-visible:ring-1 focus-visible:ring-foreground focus-visible:ring-offset-0 rounded-md h-8 px-2.5 gap-2 transition-colors text-foreground hidden sm:flex items-center font-mono text-[11px]"
            >
              <div class="flex items-center gap-2">
                <span class="text-muted-foreground font-mono text-[11px] select-none">API</span>
                <span class="font-medium text-muted-foreground">
                  {{ isEditMode ? "/v1/images/edits" : "/v1/images/generations" }}
                </span>
              </div>
            </SelectTrigger>
          </Select>
        </div>

        <div class="flex items-center gap-1">
          <!-- Settings Toggle -->
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  @click="showSettings = !showSettings"
                  class="relative h-9 w-9 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted/50"
                  :class="{ 'bg-muted text-foreground': showSettings }"
                >
                  <Settings2 class="w-4 h-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>{{ t("images.settings") }}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>

          <!-- Clear Results -->
          <TooltipProvider v-if="runs.length > 0">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  @click="clearRuns"
                  class="h-9 w-9 rounded-lg text-muted-foreground hover:text-destructive hover:bg-destructive/10"
                >
                  <Trash2 class="w-4 h-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>{{ t("images.clearResults") }}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </header>
    </template>

    <!-- Main Content -->
    <div class="brand-main-shell flex-1 flex relative overflow-hidden">
      <div class="flex-1 flex flex-col overflow-hidden">
        <!-- Creative Studio -->
        <div class="flex-none p-4 sm:p-6">
          <div class="max-w-3xl mx-auto">
            <!-- Main Canvas Card -->
            <div
              class="relative flex flex-col brand-panel transition-all duration-200 focus-within:border-primary focus-within:ring-1 focus-within:ring-primary"
              :class="{
                'border-primary bg-muted/10 ring-1 ring-primary': isDragOverMain,
              }"
              @dragover="onDragOverMain"
              @dragleave="onDragLeaveMain"
              @drop="onDropMain"
            >
              <!-- Prompt Area -->
              <div class="p-4 sm:p-5">
                <Textarea
                  v-model="prompt"
                  :placeholder="t('images.enterPrompt')"
                  class="min-h-32 resize-none border-0 bg-transparent shadow-none text-base leading-relaxed placeholder:text-muted-foreground/40 focus-visible:ring-0 focus-visible:ring-offset-0"
                  :disabled="isGenerating"
                />
              </div>

              <!-- Thumbnails strip -->
              <div v-if="uploadedImages.length > 0" class="px-4 sm:px-5 pb-3">
                <div class="flex items-center gap-2 overflow-x-auto pb-1">
                  <div
                    v-for="img in uploadedImages"
                    :key="img.id"
                    class="relative group shrink-0 w-16 h-16 rounded-md overflow-hidden border border-border/80 bg-background"
                  >
                    <img
                      :src="img.base64"
                      class="w-full h-full object-cover"
                      :alt="img.file.name"
                    />
                    <button
                      type="button"
                      @click.stop="removeUploadedImage(img.id)"
                      :aria-label="t('images.removeImage', { name: img.file.name })"
                      class="absolute top-1 right-1 min-w-6 min-h-6 w-6 h-6 rounded-full bg-black/60 flex items-center justify-center opacity-0 group-hover:opacity-100 focus-visible:opacity-100 active:opacity-100 [@media(hover:none)]:opacity-100 transition-opacity"
                    >
                      <X class="w-3.5 h-3.5 text-white" />
                    </button>
                  </div>

                  <button
                    @click="imageInputRef?.click()"
                    class="shrink-0 w-16 h-16 rounded-md border border-dashed border-border/70 flex items-center justify-center text-muted-foreground hover:text-foreground hover:border-muted-foreground/50 transition-colors"
                  >
                    <Plus class="w-5 h-5" />
                  </button>
                </div>
              </div>

              <!-- Mask thumbnail -->
              <div v-if="maskImage" class="px-4 sm:px-5 pb-3 flex items-center gap-2">
                <span class="text-xs text-muted-foreground font-medium">{{
                  t("images.maskLabel")
                }}</span>
                <div
                  class="relative group w-16 h-16 rounded-md overflow-hidden border border-dashed border-foreground/30 bg-background"
                >
                  <img
                    :src="maskImage.base64"
                    class="w-full h-full object-cover opacity-60"
                    :alt="maskImage.file.name"
                  />
                  <div
                    class="absolute inset-0 flex items-center justify-center pointer-events-none"
                  >
                    <span
                      class="text-code-xs text-muted-foreground bg-background/80 px-1.5 py-0.5 rounded"
                    >
                      MASK
                    </span>
                  </div>
                  <button
                    type="button"
                    @click.stop="removeMaskImage()"
                    :aria-label="t('images.removeMask')"
                    class="absolute top-1 right-1 min-w-6 min-h-6 w-6 h-6 rounded-full bg-black/60 flex items-center justify-center opacity-0 group-hover:opacity-100 focus-visible:opacity-100 active:opacity-100 [@media(hover:none)]:opacity-100 transition-opacity"
                  >
                    <X class="w-3.5 h-3.5 text-white" />
                  </button>
                </div>
              </div>

              <!-- Action Toolbar -->
              <div
                class="flex items-center flex-wrap gap-1.5 px-3 sm:px-4 py-3 border-t border-border/60 bg-transparent"
              >
                <Button
                  variant="ghost"
                  size="sm"
                  class="h-8 gap-1.5 text-xs text-muted-foreground hover:text-foreground"
                  @click="imageInputRef?.click()"
                >
                  <ImageIcon class="w-3.5 h-3.5" />
                  <span v-if="uploadedImages.length > 0">{{ uploadedImages.length }}</span>
                  <span v-else>{{ t("images.addImages") }}</span>
                </Button>

                <Button
                  variant="ghost"
                  size="sm"
                  class="h-8 gap-1.5 text-xs text-muted-foreground hover:text-foreground"
                  :class="{
                    'text-background bg-foreground hover:bg-foreground/90 hover:text-background':
                      maskImage,
                  }"
                  @click="maskInputRef?.click()"
                >
                  <Upload class="w-3.5 h-3.5" />
                  {{ maskImage ? t("images.maskAdded") : t("images.addMask") }}
                </Button>

                <div class="flex-1"></div>

                <Button
                  size="sm"
                  :disabled="isSubmitDisabled"
                  class="h-9 px-5 gap-1.5 btn-action"
                  @click="handleSubmit"
                >
                  <Loader2 v-if="isGenerating" class="w-4 h-4 animate-spin" />
                  <ImageIcon v-else class="w-4 h-4" />
                  <span>
                    {{
                      isGenerating
                        ? isEditMode
                          ? t("images.editing")
                          : t("images.generating")
                        : t("images.create")
                    }}
                  </span>
                </Button>
              </div>

              <!-- Hidden file inputs -->
              <input
                ref="imageInputRef"
                type="file"
                accept="image/*"
                multiple
                class="hidden"
                @change="onImagesInputChange"
              />
              <input
                ref="maskInputRef"
                type="file"
                accept="image/*"
                class="hidden"
                @change="onMaskInputChange"
              />
            </div>

            <!-- Drag hint -->
            <p
              v-if="!uploadedImages.length && !isDragOverMain"
              class="text-center text-xs text-muted-foreground/50 mt-2"
            >
              {{ t("images.dragHint") }}
            </p>
          </div>
        </div>

        <!-- Results canvas: renders the active run -->
        <div class="flex-1 overflow-y-auto px-4 sm:px-6 pb-6">
          <div class="max-w-3xl mx-auto">
            <EmptyState
              v-if="runs.length === 0"
              :text="t('images.noResults')"
              :show-cta="false"
              class="py-16"
            />

            <template v-else-if="activeRun">
              <!-- Run readout: mono telemetry line above the canvas -->
              <div class="flex items-center justify-between gap-3 mb-4">
                <div class="flex items-center gap-2 min-w-0 text-data-xs text-muted-foreground">
                  <span class="text-foreground/80">
                    #{{ String(activeRunNumber).padStart(2, "0") }}
                  </span>
                  <span class="shrink-0">
                    {{ activeRun.mode === "edits" ? "/v1/images/edits" : "/v1/images/generations" }}
                  </span>
                  <span class="hidden sm:inline truncate">{{ activeRun.model }}</span>
                </div>
                <div class="flex items-center gap-2 shrink-0 text-data-xs text-muted-foreground">
                  <span>{{ activeRun.size }}</span>
                  <span aria-hidden="true" class="text-border">·</span>
                  <span>n{{ activeRun.n }}</span>
                  <span aria-hidden="true" class="text-border">·</span>
                  <span>{{ formatLatency(activeRun.latencyMs) }}</span>
                  <button
                    type="button"
                    class="ml-1 flex items-center justify-center size-7 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
                    :title="t('playground.inspectRun')"
                    :aria-label="t('playground.inspectRun')"
                    @click="showInspector = true"
                  >
                    <FileJson2 class="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              <!-- In-flight: skeleton frames sized to the expected grid -->
              <div v-if="activeRun.status === 'streaming'" class="space-y-4">
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div
                    v-for="i in activeRun.n"
                    :key="i"
                    class="aspect-square rounded-md border border-border/60 bg-muted/30 animate-pulse"
                  />
                </div>
                <p class="text-center text-code-xs text-muted-foreground">
                  {{ activeRun.mode === "edits" ? t("images.editing") : t("images.generating") }}…
                </p>
              </div>

              <!-- Held failure: the run freezes on the canvas with its error -->
              <div
                v-else-if="activeRun.status === 'error'"
                class="rounded-xl border border-status-error/30 bg-status-error/5 p-4 flex items-start gap-3"
                role="alert"
              >
                <AlertTriangle class="w-4 h-4 text-status-error shrink-0 mt-0.5" />
                <div class="space-y-1 min-w-0">
                  <p class="text-sm font-medium text-status-error">
                    {{ t("playground.runFailed") }}
                  </p>
                  <p class="text-code-xs text-status-error/90 break-words leading-relaxed">
                    {{ activeRun.errorMessage }}
                  </p>
                </div>
              </div>

              <!-- Gallery -->
              <div v-else class="grid grid-cols-1 sm:grid-cols-2 gap-4 stagger-fast">
                <div
                  v-for="(image, index) in activeRun.images"
                  :key="image.url || image.b64_json || `img-${index}`"
                  class="group relative rounded-md border border-border/60 overflow-hidden bg-card/75 shadow-xs"
                >
                  <img
                    v-if="image.url"
                    :src="image.url"
                    :alt="`Generated image: ${activeRun.prompt}`"
                    loading="lazy"
                    class="w-full aspect-square object-cover"
                  />
                  <img
                    v-else-if="image.b64_json"
                    :src="`data:image/png;base64,${image.b64_json}`"
                    :alt="`Generated image: ${activeRun.prompt}`"
                    loading="lazy"
                    class="w-full aspect-square object-cover"
                  />
                  <div
                    class="hidden sm:flex absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity items-center justify-center gap-2"
                  >
                    <Button
                      v-if="image.url || image.b64_json"
                      variant="secondary"
                      size="sm"
                      class="h-10 px-4"
                      @click="openInNewTab(image)"
                    >
                      {{ t("images.openInNewTab") }}
                    </Button>
                    <Button
                      v-if="image.url || image.b64_json"
                      variant="secondary"
                      size="sm"
                      class="h-10 px-4"
                      @click="downloadImage(image, index)"
                    >
                      {{ t("images.download") }}
                    </Button>
                  </div>
                  <div
                    v-if="image.url || image.b64_json"
                    class="flex sm:hidden items-center justify-center gap-2 p-2 border-t border-border bg-muted/50"
                  >
                    <Button
                      variant="secondary"
                      size="sm"
                      class="h-10 px-4 flex-1"
                      @click="openInNewTab(image)"
                    >
                      {{ t("images.openInNewTab") }}
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      class="h-10 px-4 flex-1"
                      @click="downloadImage(image, index)"
                    >
                      {{ t("images.download") }}
                    </Button>
                  </div>
                  <div v-if="image.revised_prompt" class="p-2 border-t border-border bg-muted/50">
                    <p class="text-xs text-muted-foreground line-clamp-2">
                      <span class="font-medium">{{ t("images.revisedPrompt") }}:</span>
                      {{ image.revised_prompt }}
                    </p>
                  </div>
                </div>
              </div>
            </template>
          </div>
        </div>

        <!-- Run tray: every generation deposits a specimen -->
        <SpecimenTray :count="runs.length">
          <RunSpecimen
            v-for="(run, i) in runs"
            :key="run.id"
            :status="run.status"
            :selected="activeRunId === run.id"
            @click="selectRun(run.id)"
          >
            <img
              v-if="runThumb(run)"
              :src="runThumb(run)!"
              class="size-5 rounded-sm object-cover border border-border/50"
              alt=""
            />
            <span class="text-foreground/80">#{{ String(i + 1).padStart(2, "0") }}</span>
            <span>{{ endpointName(runEndpoint(run)) }}</span>
            <span class="text-muted-foreground/70">{{ formatLatency(run.latencyMs) }}</span>
          </RunSpecimen>
        </SpecimenTray>
      </div>

      <RunInspector
        v-if="activeRun"
        :open="showInspector"
        :run-number="activeRunNumber"
        :status="activeRun.status"
        :endpoint="runEndpoint(activeRun)"
        :model="activeRun.model"
        :started-at="activeRun.startedAt"
        :latency-ms="activeRun.latencyMs"
        :error-message="activeRun.errorMessage"
        :payload="activeRun.payload"
        :extra-rows="[
          { label: t('images.size'), value: activeRun.size },
          { label: t('images.quality'), value: activeRun.quality },
          { label: t('images.numberOfImages'), value: String(activeRun.n) },
        ]"
        @close="showInspector = false"
      />

      <ImagesSettings
        v-model:open="showSettings"
        v-model:numberOfImages="numberOfImages"
        v-model:size="size"
        v-model:quality="quality"
        v-model:background="background"
        v-model:moderation="moderation"
        v-model:outputCompression="outputCompression"
        v-model:outputFormat="outputFormat"
        v-model:partialImages="partialImages"
        @reset="resetSettings"
      />
    </div>
  </AppLayout>
</template>
