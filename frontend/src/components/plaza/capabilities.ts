import {
  AudioLines,
  Binary,
  Brain,
  Braces,
  Eye,
  FlaskConical,
  ImagePlus,
  LockOpen,
  Mic,
  Paperclip,
  Radio,
  Thermometer,
  Wrench,
  type LucideIcon,
} from "@lucide/vue";
import type { ModelCapability } from "@/types/schemas";

interface CapabilityMeta {
  /** i18n key under the `plaza.capability` namespace */
  labelKey: string;
  icon: LucideIcon;
  /** Subtle semantic tint classes (badge/chip), per DESIGN.md action palette */
  badgeClass: string;
  /** Text tint used when the icon appears outside a badge */
  iconClass: string;
  /**
   * Proxy-bound capabilities gate actual routing/endpoint behavior;
   * informational ones mirror models.dev attributes for display only.
   */
  bound: boolean;
}

/**
 * Display metadata for model capability tags shown in the model plaza.
 * Order here defines the display order of badges and filter chips: proxy-bound
 * capabilities first, then informational models.dev attributes.
 */
export const CAPABILITY_META: Record<ModelCapability, CapabilityMeta> = {
  vision: {
    labelKey: "plaza.capability.vision",
    icon: Eye,
    badgeClass: "border-action-blue/30 bg-action-blue/5 text-action-blue",
    iconClass: "text-action-blue",
    bound: true,
  },
  image_generation: {
    labelKey: "plaza.capability.imageGeneration",
    icon: ImagePlus,
    badgeClass: "border-action-violet/30 bg-action-violet/5 text-action-violet",
    iconClass: "text-action-violet",
    bound: true,
  },
  tts: {
    labelKey: "plaza.capability.tts",
    icon: AudioLines,
    badgeClass: "border-action-amber/30 bg-action-amber/5 text-action-amber",
    iconClass: "text-action-amber",
    bound: true,
  },
  stt: {
    labelKey: "plaza.capability.stt",
    icon: Mic,
    badgeClass: "border-action-rose/30 bg-action-rose/5 text-action-rose",
    iconClass: "text-action-rose",
    bound: true,
  },
  embedding: {
    labelKey: "plaza.capability.embedding",
    icon: Binary,
    badgeClass: "border-action-teal/30 bg-action-teal/5 text-action-teal",
    iconClass: "text-action-teal",
    bound: true,
  },
  realtime: {
    labelKey: "plaza.capability.realtime",
    icon: Radio,
    badgeClass: "border-action-blue/30 bg-action-blue/5 text-action-blue",
    iconClass: "text-action-blue",
    bound: true,
  },
  reasoning: {
    labelKey: "plaza.capability.reasoning",
    icon: Brain,
    badgeClass: "border-action-violet/30 bg-action-violet/5 text-action-violet",
    iconClass: "text-action-violet",
    bound: false,
  },
  tool_call: {
    labelKey: "plaza.capability.toolCall",
    icon: Wrench,
    badgeClass: "border-action-teal/30 bg-action-teal/5 text-action-teal",
    iconClass: "text-action-teal",
    bound: false,
  },
  structured_output: {
    labelKey: "plaza.capability.structuredOutput",
    icon: Braces,
    badgeClass: "border-action-teal/30 bg-action-teal/5 text-action-teal",
    iconClass: "text-action-teal",
    bound: false,
  },
  attachment: {
    labelKey: "plaza.capability.attachment",
    icon: Paperclip,
    badgeClass: "border-action-blue/30 bg-action-blue/5 text-action-blue",
    iconClass: "text-action-blue",
    bound: false,
  },
  temperature: {
    labelKey: "plaza.capability.temperature",
    icon: Thermometer,
    badgeClass: "border-action-amber/30 bg-action-amber/5 text-action-amber",
    iconClass: "text-action-amber",
    bound: false,
  },
  open_weights: {
    labelKey: "plaza.capability.openWeights",
    icon: LockOpen,
    badgeClass: "border-action-teal/30 bg-action-teal/5 text-action-teal",
    iconClass: "text-action-teal",
    bound: false,
  },
  experimental: {
    labelKey: "plaza.capability.experimental",
    icon: FlaskConical,
    badgeClass: "border-action-amber/30 bg-action-amber/5 text-action-amber",
    iconClass: "text-action-amber",
    bound: false,
  },
};

export const CAPABILITY_ORDER = Object.keys(CAPABILITY_META) as ModelCapability[];

/** Capabilities that gate proxy behavior (routing/endpoint binding). */
export const BOUND_CAPABILITIES = (Object.keys(CAPABILITY_META) as ModelCapability[]).filter(
  (cap) => CAPABILITY_META[cap].bound
);

/** Informational capabilities mirroring models.dev attributes (display-only). */
export const INFO_CAPABILITIES = (Object.keys(CAPABILITY_META) as ModelCapability[]).filter(
  (cap) => !CAPABILITY_META[cap].bound
);
