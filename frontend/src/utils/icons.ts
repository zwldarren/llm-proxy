import {
  LOBE_COLOR_BRANDS,
  LOBE_ICONS_VERSION,
  LOBE_MONO_BRANDS,
} from "@/utils/lobeBrands.generated";

/**
 * Brand marks for models and providers.
 *
 * Identifiers are matched against the whole `@lobehub/icons` catalog
 * (`lobeBrands.generated.ts`, refreshed with `bun run icons:sync`) instead of a
 * hand-written table, so a new model from a known brand and a niche provider
 * both get a logo without anyone editing a map. Anything left over falls back
 * to a monogram tile in `BrandMark.vue` rather than a generic chip.
 *
 * Resolution returns the URL *and* the mono/colour decision together
 * ({@link BrandIcon}). Callers must not derive those separately: the old
 * separate `getIconUrl`/`isMonoIcon` pair re-matched the brand twice and could
 * disagree, inverting a colour logo or missing the one it had chosen.
 */

export interface BrandIcon {
  url: string;
  /**
   * True when the mark is a `currentColor` SVG that inherits the text colour.
   * Fixed-palette marks stay as shipped.
   */
  mono: boolean;
}

/**
 * Icon CDN base, pinned to the catalog version.
 *
 * The catalog is generated from one specific release, so the CDN path is pinned
 * to it as well: a brand can only be matched if the file is guaranteed to be
 * there. Bumping the package means re-running `bun run icons:sync`.
 */
const ICON_CDN = `https://unpkg.com/@lobehub/icons-static-svg@${LOBE_ICONS_VERSION}/icons`;

/** Brands with a `currentColor` mark (`<brand>.svg`). */
const MONO_BRANDS = new Set<string>(LOBE_MONO_BRANDS);

/** Brands with a fixed-palette mark (`<brand>-color.svg`). */
const COLOR_BRANDS = new Set<string>(LOBE_COLOR_BRANDS);

const BRANDS = new Set<string>([...MONO_BRANDS, ...COLOR_BRANDS]);

/**
 * Words that appear inside model ids but never name a brand. Matching them
 * would paint, say, `chat-1` with whatever logo happens to be called "chat".
 *
 * Brands win over this list even if someone adds one to both, so a stopword can
 * never shadow a real logo.
 */
const STOPWORDS = new Set(
  [
    "agent",
    "air",
    "audio",
    "auto",
    "base",
    "beta",
    "chat",
    "coder",
    "coding",
    "custom",
    "deep",
    "default",
    "distill",
    "edge",
    "embed",
    "embedding",
    "experimental",
    "fast",
    "flash",
    "free",
    "gen",
    "guard",
    "haiku",
    "high",
    "image",
    "instruct",
    "it",
    "judge",
    "large",
    "latest",
    "lite",
    "low",
    "max",
    "medium",
    "mini",
    "model",
    "nano",
    "none",
    "online",
    "opus",
    "plus",
    "preview",
    "pro",
    "prompt",
    "quality",
    "realtime",
    "reasoning",
    "rerank",
    "research",
    "small",
    "sonnet",
    "stable",
    "standard",
    "super",
    "text",
    "thinking",
    "tool",
    "tooling",
    "tools",
    "turbo",
    "ultra",
    "unknown",
    "value",
    "video",
    "vision",
  ].filter((word) => !BRANDS.has(word))
);

/**
 * Family, product and company names that map onto a brand the catalog keys
 * differently. Everything derivable from the id itself (org prefixes, `-ai`
 * suffixes, `qwen3`-style version suffixes) is handled by the matcher; this
 * table is only for meaning the matcher cannot infer.
 */
const BRAND_ALIASES = new Map<string, string>(
  Object.entries({
    allenai: "ai2",
    amazon: "aws",
    cf: "cloudflare",
    chatgpt: "openai",
    codestral: "mistral",
    command: "cohere",
    devstral: "mistral",
    dots: "dotsstudio",
    ernie: "baidu",
    glm: "zai",
    gpt: "openai",
    hermes: "nousresearch",
    hf: "huggingface",
    hy: "hunyuan",
    kat: "kwaipilot",
    llama: "meta",
    magistral: "mistral",
    meituan: "longcat",
    mimo: "xiaomimimo",
    ministral: "mistral",
    nemotron: "nvidia",
    o1: "openai",
    o3: "openai",
    o4: "openai",
    phi: "microsoft",
    sonar: "perplexity",
    step: "stepfun",
    titan: "aws",
    vertex: "gemini",
    xiaomi: "xiaomimimo",
  }).filter(([name, brand]) => BRANDS.has(brand) && !BRANDS.has(name))
);

/**
 * Token -> brand for the "brand with a longer name" case (`agnes` for
 * `agnesai`, `topaz` for `topazlabs`). Built from the catalog so it stays in
 * step with it; a prefix shared by two brands resolves to nothing.
 */
const PREFIX_BRANDS = new Map<string, string | null>();
for (const brand of BRANDS) {
  for (let length = 4; length < brand.length; length += 1) {
    const prefix = brand.slice(0, length);
    PREFIX_BRANDS.set(prefix, PREFIX_BRANDS.has(prefix) ? null : brand);
  }
}

/**
 * Tokens shorter than this are only matched whole, never as a prefix: `fal` or
 * `exa` are brands, `fa`/`ex` are noise.
 */
const MIN_PREFIX_LENGTH = 4;

/** Version-ish prefixes (`v0`, `v1.5`) are never brands. */
const VERSION_TOKEN = /^v\d/;

/** `qwen3`, `gpt4o`, `glm4` (from `glm-4.5`) — a brand followed by its version. */
const VERSIONED_TOKEN = /^([a-z]+)\d[\da-z]*$/;

/** `mistralai`, `moonshotai` — a brand followed by the company suffix. */
const COMPANY_TOKEN = /^([a-z]+)ai$/;

function tokens(value: string): string[] {
  return value
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(Boolean);
}

function isUsableToken(token: string): boolean {
  return token.length >= 2 && !STOPWORDS.has(token) && !VERSION_TOKEN.test(token);
}

function brandForStem(stem: string): string | null {
  if (!isUsableToken(stem)) {
    return null;
  }
  if (BRANDS.has(stem)) {
    return stem;
  }
  return BRAND_ALIASES.get(stem) ?? null;
}

function brandForToken(token: string, allowPrefix: boolean): string | null {
  if (!isUsableToken(token)) {
    return null;
  }
  if (BRANDS.has(token)) {
    return token;
  }
  const aliased = BRAND_ALIASES.get(token);
  if (aliased) {
    return aliased;
  }
  const versioned = VERSIONED_TOKEN.exec(token);
  if (versioned) {
    const stem = brandForStem(versioned[1]);
    if (stem) {
      return stem;
    }
  }
  const company = COMPANY_TOKEN.exec(token);
  if (company) {
    const stem = brandForStem(company[1]);
    if (stem) {
      return stem;
    }
  }
  return allowPrefix ? (PREFIX_BRANDS.get(token) ?? null) : null;
}

/** One `/`- or `:`-separated segment of an identifier. */
function brandForSegment(segment: string): string | null {
  const parts = tokens(segment);
  if (parts.length === 0) {
    return null;
  }
  // "openai", "x-ai" -> "xai", "together_ai" -> "togetherai"
  const joined = parts.join("");
  const whole = brandForStem(joined);
  if (whole) {
    return whole;
  }
  // "mistralai" -> "mistral"; the whole-segment form is checked first so a real
  // brand ending in "ai" (`ai21`, `baai`) is not truncated.
  if (joined.endsWith("ai")) {
    const truncated = brandForStem(joined.slice(0, -2));
    if (truncated) {
      return truncated;
    }
  }
  for (const [index, part] of parts.entries()) {
    // Longer-name matching is only sound for the leading word: it is what a
    // truncated brand looks like (`agnes-2.0-flash`), while a bare word in the
    // middle of an id (`lyria-3-clip-preview`) is usually a qualifier.
    const brand = brandForToken(part, index === 0 && part.length >= MIN_PREFIX_LENGTH);
    if (brand) {
      return brand;
    }
  }
  return null;
}

/** Memo for `resolveBrand`, keyed by the raw identifier. */
const RESOLVE_CACHE = new Map<string, string | null>();
const RESOLVE_CACHE_LIMIT = 2000;

/**
 * The icon-catalog brand an identifier refers to, or null.
 *
 * The identifier may be a model id (`claude-sonnet-4-5`, `qwen3-235b`), an
 * `org/model` id (`x-ai/grok-4`) or a provider type (`openai-compatible`).
 * Segments are searched right to left, so the model's own brand wins over the
 * hosting platform in a `deepinfra/bytedance/seed` style id.
 */
export function resolveBrand(value: string | null | undefined): string | null {
  const identifier = value?.trim();
  if (!identifier) {
    return null;
  }
  const cached = RESOLVE_CACHE.get(identifier);
  if (cached !== undefined) {
    return cached;
  }
  const segments = identifier.split(/[/:@]/).filter(Boolean);
  let brand: string | null = null;
  for (let index = segments.length - 1; index >= 0 && !brand; index -= 1) {
    brand = brandForSegment(segments[index]);
  }
  if (RESOLVE_CACHE.size >= RESOLVE_CACHE_LIMIT) {
    RESOLVE_CACHE.clear();
  }
  RESOLVE_CACHE.set(identifier, brand);
  return brand;
}

function iconUrl(slug: string): string {
  return `${ICON_CDN}/${slug}.svg`;
}

/**
 * A brand's mark.
 *
 * `variant` defaults to "auto": the fixed-palette mark when the brand ships one
 * (recognition beats monochrome), otherwise the mono mark. "mono" and "color"
 * force one, falling back to the other when the brand only has that one.
 */
export function brandIcon(
  brand: string | null | undefined,
  variant?: string | null
): BrandIcon | null {
  const slug = brand?.trim().toLowerCase();
  if (!slug || !BRANDS.has(slug)) {
    return null;
  }
  const wantsColor = variant !== "mono" && COLOR_BRANDS.has(slug);
  return { url: iconUrl(wantsColor ? `${slug}-color` : slug), mono: !wantsColor };
}

export interface ModelIconInput {
  /** Model id or display name. */
  name?: string | null;
  /**
   * Provider hint (type or configured name), used only when the model name
   * itself carries no brand — a custom `fast-model-1` on OpenRouter still gets
   * the OpenRouter mark.
   */
  provider?: string | null;
  /** Explicit `icon_url` from the model config; wins over any derived mark. */
  iconUrl?: string | null;
}

/** Brand mark for a model: configured icon, then model brand, then provider. */
export function modelIcon(input: ModelIconInput): BrandIcon | null {
  const custom = input.iconUrl?.trim();
  if (custom) {
    // A configured URL is used exactly as given, never tinted.
    return { url: custom, mono: false };
  }
  const brand = resolveBrand(input.name) ?? resolveBrand(input.provider);
  return brandIcon(brand);
}

/** Backend branding a provider *type* declares (adapter `LOBE_ICON_ID`). */
export interface ProviderBrandDeclaration {
  icon_id?: string | null;
  icon_variant?: string | null;
}

export interface ProviderIconInput {
  /** Provider type (`openai`, `zai-coding`, ...). */
  type?: string | null;
  /** Configured provider name, matched when the type itself is unknown. */
  name?: string | null;
  /** Explicit `icon_url` from the provider config; wins over derived marks. */
  iconUrl?: string | null;
  /** Branding declared by the provider type's adapter, if any. */
  brand?: ProviderBrandDeclaration | null;
}

/**
 * Brand mark for a provider: configured icon, then the adapter's declared
 * branding, then a catalog match on the type and name.
 */
export function providerIcon(input: ProviderIconInput): BrandIcon | null {
  const custom = input.iconUrl?.trim();
  if (custom) {
    return { url: custom, mono: false };
  }
  const declared = brandIcon(input.brand?.icon_id, input.brand?.icon_variant);
  if (declared) {
    return declared;
  }
  const brand = resolveBrand(input.type) ?? resolveBrand(input.name);
  return brandIcon(brand);
}

/**
 * Letter for the fallback tile: the first letter or digit of a name, with
 * leading punctuation and emoji skipped. CJK names yield their first character,
 * which is their readable initial.
 */
export function monogramInitial(label: string | null | undefined): string {
  if (!label) {
    return "?";
  }
  for (const character of label.trim()) {
    if (/\p{L}|\p{N}/u.test(character)) {
      return character.toUpperCase();
    }
  }
  return "?";
}
