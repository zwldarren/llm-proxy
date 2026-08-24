import type { ProviderTypeInfo } from "@/types/schemas";

/**
 * Build a CDN URL for a Lobe icon.
 *
 * Local re-implementation of @lobehub/icons' getLobeIconCDN. That package is a
 * React component library (peer deps on react/react-dom/antd), so importing it
 * pulled all of React into this Vue app's bundle. We only need this small,
 * dependency-free URL builder.
 */
type LobeIconCdn = "github" | "aliyun" | "unpkg";
type LobeIconFormat = "svg" | "png" | "webp" | "avatar";
type LobeIconType = "mono" | "color" | "text" | "text-cn" | "text-color" | "brand" | "brand-color";

interface LobeIconCdnConfig {
  cdn?: LobeIconCdn;
  format?: LobeIconFormat;
  isDarkMode?: boolean;
  type?: LobeIconType;
}

const GITHUB_ICON_CDN = (format: LobeIconFormat) =>
  `https://raw.githubusercontent.com/lobehub/lobe-icons/refs/heads/master/packages/static-${format}`;
const ALIYUN_ICON_CDN = (format: LobeIconFormat) =>
  `https://registry.npmmirror.com/@lobehub/icons-static-${format}/latest/files`;
const UNPKG_ICON_CDN = (format: LobeIconFormat) =>
  `https://unpkg.com/@lobehub/icons-static-${format}@latest`;

function getLobeIconCDN(id: string, config: LobeIconCdnConfig = {}): string {
  const { format = "png", isDarkMode = false, type = "color", cdn = "github" } = config;

  let baseUrl: string;
  switch (cdn) {
    case "unpkg":
      baseUrl = UNPKG_ICON_CDN(format);
      break;
    case "aliyun":
      baseUrl = ALIYUN_ICON_CDN(format);
      break;
    default:
      baseUrl = GITHUB_ICON_CDN(format);
  }

  if (format === "avatar") {
    return `${baseUrl}/avatars/${id.toLowerCase()}.webp`;
  }

  const slug = `${id.toLowerCase()}${type === "mono" ? "" : `-${type}`}`;
  if (format === "svg") {
    return `${baseUrl}/icons/${slug}.svg`;
  }
  return `${baseUrl}/${isDarkMode ? "dark" : "light"}/${slug}.${format === "webp" ? "webp" : "png"}`;
}

const CDN_COLOR = { format: "svg" as const, type: "color" as const, cdn: "unpkg" as const };
const CDN_MONO = { format: "svg" as const, type: "mono" as const, cdn: "unpkg" as const };

interface IconEntry {
  url: string;
  type: "mono" | "color";
}

const DEFAULT_ICON_ENTRIES: Record<string, IconEntry> = {
  gpt: { url: getLobeIconCDN("openai", CDN_MONO), type: "mono" },
  claude: { url: getLobeIconCDN("claude", CDN_COLOR), type: "color" },
  gemini: { url: getLobeIconCDN("gemini", CDN_COLOR), type: "color" },
  grok: { url: getLobeIconCDN("xai", CDN_MONO), type: "mono" },
  qwen: { url: getLobeIconCDN("qwen", CDN_COLOR), type: "color" },
  deepseek: { url: getLobeIconCDN("deepseek", CDN_COLOR), type: "color" },
  kimi: { url: getLobeIconCDN("moonshot", CDN_MONO), type: "mono" },
  minimax: { url: getLobeIconCDN("minimax", CDN_COLOR), type: "color" },
  glm: { url: getLobeIconCDN("zai", CDN_MONO), type: "mono" },
  stepfun: { url: getLobeIconCDN("stepfun", CDN_COLOR), type: "color" },
  nemotron: { url: getLobeIconCDN("nvidia", CDN_COLOR), type: "color" },
  mimo: { url: getLobeIconCDN("XiaomiMiMo", CDN_MONO), type: "mono" },
  hy: { url: getLobeIconCDN("hunyuan", CDN_COLOR), type: "color" },
  mistral: { url: getLobeIconCDN("Mistral", CDN_COLOR), type: "color" },
  longcat: { url: getLobeIconCDN("LongCat", CDN_COLOR), type: "color" },
};

/** Check if a model name maps to a mono icon. */
export function isMonoIcon(name: string | null | undefined): boolean {
  if (!name) return false;
  const lower = name.toLowerCase();
  for (const [prefix, entry] of Object.entries(DEFAULT_ICON_ENTRIES)) {
    if (lower.includes(prefix) && entry.type === "mono") return true;
  }
  return false;
}

/** Check if a provider type uses a mono icon. */
export function isMonoProvider(providerType: string | null | undefined): boolean {
  if (!providerType) return false;
  const mapping = PROVIDER_ICON_MAP[providerType.toLowerCase()];
  return mapping?.type === "mono";
}

function getDefaultIconUrl(name: string | null | undefined): string | null {
  if (!name) return null;

  const lowerName = name.toLowerCase();

  for (const [prefix, entry] of Object.entries(DEFAULT_ICON_ENTRIES)) {
    if (lowerName.startsWith(prefix)) {
      return entry.url;
    }
  }

  for (const [prefix, entry] of Object.entries(DEFAULT_ICON_ENTRIES)) {
    if (lowerName.includes(prefix)) {
      return entry.url;
    }
  }

  return null;
}

export function getIconUrl(
  iconUrl: string | null | undefined,
  name: string | null | undefined
): string | null {
  if (iconUrl) {
    return iconUrl;
  }
  return getDefaultIconUrl(name);
}

const PROVIDER_ICON_MAP: Record<string, { id: string; type: "mono" | "color" }> = {
  openai: { id: "openai", type: "mono" },
  "openai-compatible": { id: "openai", type: "mono" },
  anthropic: { id: "anthropic", type: "color" },
  gemini: { id: "gemini", type: "color" },
  google: { id: "gemini", type: "color" },
  deepseek: { id: "deepseek", type: "color" },
  openrouter: { id: "openrouter", type: "mono" },
  ollama: { id: "ollama", type: "mono" },
  zai: { id: "zai", type: "mono" },
  "zai-coding": { id: "zai", type: "mono" },
  zhipu: { id: "zhipu", type: "color" },
  "zhipu-coding": { id: "zhipu", type: "color" },
  moonshot: { id: "moonshot", type: "mono" },
  "kimi-code": { id: "kimi", type: "color" },
  minimax: { id: "minimax", type: "color" },
  qwen: { id: "qwen", type: "color" },
  "qwen-intl": { id: "qwen", type: "color" },
  mistral: { id: "mistral", type: "color" },
  xai: { id: "xai", type: "mono" },
};

export function getProviderIconUrl(
  providerType: string,
  options?: { type?: "mono" | "color"; format?: "svg" | "png" }
): string | null {
  const mapping = PROVIDER_ICON_MAP[providerType.toLowerCase()];
  if (!mapping) {
    return null;
  }
  return getLobeIconCDN(mapping.id, {
    format: options?.format ?? "svg",
    type: mapping.type,
    cdn: "unpkg",
  });
}

/**
 * Build the Lobe icon URL from backend provider-type metadata, or null when
 * the type declares no icon. Callers fall back to the static map
 * (``getProviderIconUrl``) for types without backend metadata.
 */
export function iconUrlFromMetadata(info: ProviderTypeInfo | undefined | null): string | null {
  if (!info?.icon_id) {
    return null;
  }
  return getLobeIconCDN(info.icon_id, {
    format: "svg",
    type: info.icon_variant ?? "color",
    cdn: "unpkg",
  });
}

export function getModelIconUrl(
  modelId: string | null | undefined,
  provider?: string | null | undefined,
  customIconUrl?: string | null | undefined
): string | null {
  if (customIconUrl) {
    return customIconUrl;
  }
  const defaultIcon = getDefaultIconUrl(modelId);
  if (defaultIcon) {
    return defaultIcon;
  }
  if (provider) {
    const providerIcon = getProviderIconUrl(provider);
    if (providerIcon) return providerIcon;
  }
  return null;
}
