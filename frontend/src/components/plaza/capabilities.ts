import { AudioLines, Binary, Eye, ImagePlus, Mic, Radio, type LucideIcon } from "@lucide/vue";
import type { ModelCapability, ModelRead } from "@/types/schemas";

interface CapabilityMeta {
  /** i18n key under the `plaza.capability` namespace */
  labelKey: string;
  icon: LucideIcon;
  /** Subtle semantic tint classes (badge/chip), per DESIGN.md action palette */
  badgeClass: string;
  /** Text tint used when the icon appears outside a badge */
  iconClass: string;
}

/**
 * Display metadata for model capability tags shown in the model plaza.
 * Order here defines the display order of badges and filter chips.
 */
export const CAPABILITY_META: Record<ModelCapability, CapabilityMeta> = {
  vision: {
    labelKey: "plaza.capability.vision",
    icon: Eye,
    badgeClass: "border-action-blue/30 bg-action-blue/5 text-action-blue",
    iconClass: "text-action-blue",
  },
  image_generation: {
    labelKey: "plaza.capability.imageGeneration",
    icon: ImagePlus,
    badgeClass: "border-action-violet/30 bg-action-violet/5 text-action-violet",
    iconClass: "text-action-violet",
  },
  tts: {
    labelKey: "plaza.capability.tts",
    icon: AudioLines,
    badgeClass: "border-action-amber/30 bg-action-amber/5 text-action-amber",
    iconClass: "text-action-amber",
  },
  stt: {
    labelKey: "plaza.capability.stt",
    icon: Mic,
    badgeClass: "border-action-rose/30 bg-action-rose/5 text-action-rose",
    iconClass: "text-action-rose",
  },
  embedding: {
    labelKey: "plaza.capability.embedding",
    icon: Binary,
    badgeClass: "border-action-teal/30 bg-action-teal/5 text-action-teal",
    iconClass: "text-action-teal",
  },
  realtime: {
    labelKey: "plaza.capability.realtime",
    icon: Radio,
    badgeClass: "border-action-blue/30 bg-action-blue/5 text-action-blue",
    iconClass: "text-action-blue",
  },
};

export const CAPABILITY_ORDER = Object.keys(CAPABILITY_META) as ModelCapability[];

/**
 * Derive display capability tags from an admin-facing model's supports_* flags.
 * Mirrors the catalog endpoint's derivation so admin views show the same badges.
 */
export function deriveModelCapabilities(model: ModelRead): ModelCapability[] {
  const caps: ModelCapability[] = [];
  if (model.supports_images) caps.push("vision");
  if (model.supports_image_generation) caps.push("image_generation");
  if (model.supports_tts) caps.push("tts");
  if (model.supports_stt) caps.push("stt");
  if (model.supports_embedding) caps.push("embedding");
  if (model.supports_realtime) caps.push("realtime");
  return caps;
}
