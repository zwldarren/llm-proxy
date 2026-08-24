<script setup lang="ts">
/**
 * Images generation settings — right-side drawer sharing the ChatSettings
 * grammar (fixed overlay on mobile, relative lane on desktop).
 */
import { Sliders, Paintbrush, X } from "@lucide/vue";
import { useI18n } from "vue-i18n";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { NumberInput } from "@/components/ui/number-input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const open = defineModel<boolean>("open", { required: true });
const numberOfImages = defineModel<number>("numberOfImages", { required: true });
const size = defineModel<string>("size", { required: true });
const quality = defineModel<string>("quality", { required: true });
const background = defineModel<string | null>("background", { required: true });
const moderation = defineModel<string | null>("moderation", { required: true });
const outputCompression = defineModel<number | null>("outputCompression", { required: true });
const outputFormat = defineModel<string | null>("outputFormat", { required: true });
const partialImages = defineModel<number | null>("partialImages", { required: true });

const emit = defineEmits<{
  reset: [];
}>();

const { t } = useI18n();

const close = () => {
  open.value = false;
};

const sizeOptions = [
  { value: "auto", label: t("images.sizeAuto") },
  { value: "512x512", label: t("images.size512") },
  { value: "1024x1024", label: t("images.size1024x1024") },
  { value: "1536x1024", label: t("images.size1536x1024") },
  { value: "1024x1536", label: t("images.size1024x1536") },
];
</script>

<template>
  <!-- Settings overlay (mobile) -->
  <transition
    enter-active-class="settings-overlay-enter-active"
    leave-active-class="settings-overlay-leave-active"
  >
    <div
      v-if="open"
      class="fixed inset-0 overlay-light backdrop-blur-[2px] z-40 lg:hidden"
      @click="close"
    />
  </transition>

  <!-- Settings panel -->
  <transition
    enter-active-class="settings-panel-enter-active"
    leave-active-class="settings-panel-leave-active"
  >
    <aside
      v-if="open"
      class="fixed right-0 top-0 bottom-0 w-80 z-40 lg:relative lg:z-auto shrink-0 border-l border-border/50 bg-card overflow-hidden will-change-transform"
      :aria-label="t('images.settings')"
    >
      <div class="h-full flex flex-col">
        <!-- Panel header -->
        <div class="p-4 border-b border-border/50 lg:pt-4 pt-[calc(env(safe-area-inset-top)+1rem)]">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-2">
              <div class="icon-container p-1.5">
                <Sliders class="w-4 h-4 text-primary" />
              </div>
              <h3 class="font-semibold text-sm">{{ t("images.settings") }}</h3>
            </div>
            <div class="flex items-center gap-1">
              <Button
                variant="ghost"
                size="sm"
                @click="emit('reset')"
                class="h-7 px-2 text-xs text-muted-foreground hover:text-foreground"
              >
                <Paintbrush class="w-3 h-3 mr-1" />
                {{ t("chat.resetSettings") }}
              </Button>
              <Button variant="ghost" size="icon" class="h-10 w-10" @click="close">
                <X class="w-4 h-4" />
              </Button>
            </div>
          </div>
        </div>

        <!-- Settings content -->
        <div class="flex-1 overflow-y-auto p-4 space-y-5">
          <div class="space-y-2">
            <Label class="text-xs font-medium text-muted-foreground">{{
              t("images.numberOfImages")
            }}</Label>
            <Select v-model="numberOfImages">
              <SelectTrigger class="h-9">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem v-for="n in 4" :key="n" :value="n">{{ n }}</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div class="space-y-2">
            <Label class="text-xs font-medium text-muted-foreground">{{ t("images.size") }}</Label>
            <Select v-model="size">
              <SelectTrigger class="h-9">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem v-for="opt in sizeOptions" :key="opt.value" :value="opt.value">
                  {{ opt.label }}
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div class="space-y-2">
            <Label class="text-xs font-medium text-muted-foreground">{{
              t("images.quality")
            }}</Label>
            <Select v-model="quality">
              <SelectTrigger class="h-9">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">{{ t("images.qualityAuto") }}</SelectItem>
                <SelectItem value="low">{{ t("images.qualityLow") }}</SelectItem>
                <SelectItem value="medium">{{ t("images.qualityMedium") }}</SelectItem>
                <SelectItem value="high">{{ t("images.qualityHigh") }}</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div class="space-y-2">
            <Label class="text-xs font-medium text-muted-foreground">{{
              t("images.background")
            }}</Label>
            <Select v-model="background">
              <SelectTrigger class="h-9">
                <SelectValue :placeholder="t('images.backgroundAuto')" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">{{ t("images.backgroundAuto") }}</SelectItem>
                <SelectItem value="transparent">{{ t("images.backgroundTransparent") }}</SelectItem>
                <SelectItem value="opaque">{{ t("images.backgroundOpaque") }}</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div class="space-y-2">
            <Label class="text-xs font-medium text-muted-foreground">{{
              t("images.moderation")
            }}</Label>
            <Select v-model="moderation">
              <SelectTrigger class="h-9">
                <SelectValue :placeholder="t('images.moderationAuto')" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">{{ t("images.moderationAuto") }}</SelectItem>
                <SelectItem value="low">{{ t("images.moderationLow") }}</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div class="space-y-2">
            <Label class="text-xs font-medium text-muted-foreground">{{
              t("images.outputFormat")
            }}</Label>
            <Select v-model="outputFormat">
              <SelectTrigger class="h-9">
                <SelectValue :placeholder="t('common.select')" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="png">{{ t("images.outputFormatPNG") }}</SelectItem>
                <SelectItem value="jpeg">{{ t("images.outputFormatJPEG") }}</SelectItem>
                <SelectItem value="webp">{{ t("images.outputFormatWebP") }}</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div class="space-y-2">
            <Label class="text-xs font-medium text-muted-foreground">{{
              t("images.outputCompression")
            }}</Label>
            <NumberInput
              v-model.number="outputCompression"
              class="h-9"
              :placeholder="t('common.optional')"
            />
          </div>

          <div class="space-y-2">
            <Label class="text-xs font-medium text-muted-foreground">{{
              t("images.partialImages")
            }}</Label>
            <NumberInput
              v-model.number="partialImages"
              class="h-9"
              :placeholder="t('common.optional')"
              min="0"
              max="3"
            />
          </div>
        </div>
      </div>
    </aside>
  </transition>
</template>
