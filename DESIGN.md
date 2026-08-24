---
name: LLM Proxy
description: Monochrome editorial admin console for a self-hostable multi-provider LLM gateway.
colors:
  # Dark-first system. Values below are the dark-mode (primary) hex;
  # canonical HSL tokens + light-mode equivalents are documented in §Colors.
  surface: "#090A0B"          # --background    hsl(220 12% 4%)
  ink: "#F5F5F5"              # --foreground    hsl(0 0% 96%)
  card: "#151619"             # --card / --popover  hsl(220 10% 9%)
  sidebar: "#0F1114"          # --sidebar       hsl(220 14% 7%)
  secondary: "#1E2024"        # --secondary     hsl(220 10% 13%)
  muted: "#23252A"            # --muted         hsl(220 9% 15%)
  muted-ink: "#999CA3"        # --muted-foreground  hsl(220 5% 62%)
  accent: "#292C32"           # --accent        hsl(220 10% 18%)
  border: "#32363E"           # --border / --input  hsl(220 10% 22%)
  ring: "#FAFAFA"             # --ring          hsl(0 0% 98%)
  primary: "#FAFAFA"          # --primary (inverted ink for CTAs)  hsl(0 0% 98%)
  primary-ink: "#111317"      # --primary-foreground  hsl(220 15% 8%)
  destructive: "#C44D45"      # --destructive   hsl(4 52% 52%)
  action-blue: "#8BA3BB"      # GET             hsl(210 26% 64%)
  action-violet: "#C8C0D8"    # POST            hsl(260 24% 80%)
  action-amber: "#B69E7C"     # PUT / PATCH     hsl(35 28% 60%)
  action-rose: "#BD8992"      # DELETE          hsl(350 28% 64%)
  action-teal: "#85B7B7"      # embedding       hsl(180 26% 62%)
  status-success: "#67C194"   # --status-success hsl(150 42% 58%)
  status-warning: "#C19767"   # --status-warning hsl(32 42% 58%)
  status-error: "#DD9792"     # --status-error   hsl(4 52% 72%)
  status-unknown: "#8F9299"   # --status-unknown hsl(220 5% 58%)
  json-string: "#79D8A0"      # --json-string   hsl(145 55% 66%)
  json-number: "#78ADE2"      # --json-number   hsl(210 65% 68%)
  json-boolean: "#EA8696"     # --json-boolean  hsl(350 70% 72%)
  json-null: "#848FA4"        # --json-null     hsl(220 15% 58%)
  json-key: "#BCC3D2"         # --json-key      hsl(220 20% 78%)
  json-keyword: "#B497ED"     # --json-keyword  hsl(260 70% 76%)
  json-function: "#70C2EB"    # --json-function hsl(200 75% 68%)
  code-bg: "#121416"          # --code-bg       hsl(220 10% 8%)
  code-header-bg: "#1A1C20"   # --code-header-bg hsl(220 10% 11.5%)
  code-border: "#292C32"      # --code-border   hsl(220 10% 18%)
  error-light: "#CD7670"      # --error-light   hsl(4 48% 62%)
  error-dark: "#DD9792"       # --error-dark    hsl(4 52% 72%)
  success-light: "#4CA97A"    # --success-light hsl(150 38% 48%)
  success-dark: "#67C194"     # --success-dark  hsl(150 42% 58%)
typography:
  display:
    fontFamily: "Space Grotesk, Space Grotesk Fallback, Noto Sans SC, Manrope, sans-serif"
    fontSize: "1.25rem–1.5rem (responsive page titles)"
    fontWeight: 600
    lineHeight: 1.25
    letterSpacing: "-0.02em"
  headline:
    fontFamily: "Space Grotesk, Space Grotesk Fallback, Noto Sans SC, sans-serif"
    fontSize: "2rem"
    fontWeight: 400
    lineHeight: 1.1
    letterSpacing: "-0.02em"
  title:
    fontFamily: "Manrope, Noto Sans SC, sans-serif"
    fontSize: "1.125rem"
    fontWeight: 600
    lineHeight: 1.375
    letterSpacing: "0"
  body:
    fontFamily: "Manrope, Manrope Fallback, Noto Sans SC, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: "0"
  secondary:
    fontFamily: "Manrope, Noto Sans SC, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: "0"
  label:
    fontFamily: "Manrope, Noto Sans SC, sans-serif"
    fontSize: "0.75rem"
    fontWeight: 500
    lineHeight: 1.25
    letterSpacing: "0.02em"
  mono:
    fontFamily: "IBM Plex Mono, IBM Plex Mono Fallback, JetBrains Mono, ui-monospace, monospace"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: "0"
    fontVariantNumeric: "tabular-nums"
  metric:
    fontFamily: "IBM Plex Mono, JetBrains Mono, ui-monospace, monospace"
    fontSize: "1.5rem"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "-0.01em"
    fontVariantNumeric: "tabular-nums slashed-zero"
rounded:
  sm: "8px"     # calc(--radius - 4px)
  md: "10px"    # calc(--radius - 2px) — buttons, inputs, badges
  lg: "12px"    # --radius (0.75rem) — nav items, tooltips, code containers
  xl: "16px"    # calc(--radius + 4px) — cards, panels, section containers, icon containers
  pill: "9999px" # tags, status chips, the range thumb
spacing:
  "2": "8px"
  "3": "12px"
  "4": "16px"
  "5": "20px"
  "6": "24px"
  "8": "32px"
  "10": "40px"
  "12": "48px"
  "16": "64px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.primary-ink}"
    typography: "{typography.label}"
    rounded: "{rounded.md}"
    padding: "0 16px"
    height: "40px"
  button-destructive:
    backgroundColor: "{colors.destructive}"
    textColor: "#FFFFFF"
    typography: "{typography.label}"
    rounded: "{rounded.md}"
    padding: "0 16px"
    height: "40px"
  button-outline:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    rounded: "{rounded.md}"
    padding: "0 16px"
    height: "40px"
  button-ghost:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    rounded: "{rounded.md}"
    padding: "0 16px"
    height: "40px"
  input:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.md}"
    padding: "8px 14px"
    height: "40px"
  card:
    backgroundColor: "{colors.card}"
    textColor: "{colors.ink}"
    rounded: "{rounded.xl}"
    padding: "16px"
  badge-status:
    backgroundColor: "{colors.muted}"
    textColor: "{colors.muted-ink}"
    typography: "{typography.label}"
    rounded: "{rounded.pill}"
    padding: "2px 8px"
  nav-item:
    backgroundColor: "transparent"
    textColor: "{colors.muted-ink}"
    typography: "{typography.label}"
    rounded: "{rounded.lg}"
    padding: "10px 12px"
    height: "44px"
  code-block:
    backgroundColor: "{colors.code-bg}"
    textColor: "{colors.ink}"
    typography: "{typography.mono}"
    rounded: "{rounded.xl}"
    padding: "16px"
---

# Design System: LLM Proxy

## Overview

**Creative North Star: "The Operator's Console"**

LLM Proxy's admin UI is an infrastructure instrument panel — a dark, monochrome console an operator reads mid-incident and trusts on the first glance. It pairs the discipline of editorial typography (Space Grotesk display, Manrope body, IBM Plex Mono for every number and code value) with the restraint of a telemetry surface: tonal layering, hairline borders, and ambient glow in place of ornament. The operator is technical, time-pressed, and often a stranger deploying the proxy for the first time, so the system earns trust through precision, not decoration.

The palette is monochrome editorial — deep cool-neutral voids in dark mode, crisp near-white in light mode — with a barely-perceptible hue tinting system reserved for semantic differentiation only: HTTP methods, status, and action types. Saturation is deliberately low; color never decorates, it disambiguates. The atmosphere is built from two fixed layers behind every screen: a trio of soft radial foreground-glints and a faint 42px grid masked toward the viewport center — present enough to read as a designed surface, quiet enough never to compete with data.

This system explicitly rejects the generic 2026 AI-tool aesthetic (cream/sand warm-neutral bodies, gradient-text headlines, identical icon-card grids, big-number hero-metric templates), loud SaaS gradients, playful/consumer affordances, and circa-2008 enterprise chrome. It is **dark-first**: dark mode is the primary experience and the canonical reference for every token below; light mode is a complete, first-class translation, not an afterthought.

**Key Characteristics:**
- Dark-first monochrome with cool-neutral (220°) undertones; light mode is a crisp mirror.
- Editorial type pairing: geometric Space Grotesk display + humanist Manrope body + IBM Plex Mono for data.
- Flat-by-default surfaces; depth via tonal background steps + 1px hairline borders, never structural drop shadows.
- Low-saturation semantic tints (violet/blue/amber/rose/teal + status + syntax) — always paired with text or icon, never color-only.
- Motion is exponential ease-out (`cubic-bezier(0.16, 1, 0.3, 1)`), 150–500ms, GPU-only, and globally neutralized under `prefers-reduced-motion`.
- Tabular-numeric monospace for every metric, token count, cost, and timestamp.

## Colors

A two-theme monochrome system with a cool-neutral (hue 220°) undertone. Dark mode is canonical (first value); light mode is the crisp mirror (second value). All tokens are HSL in source (`hsl(H S% L%)`); hex is the sRGB rendering. **The HSL token is the source of truth; edit the token, not the hex.**

### Primary
- **Console White / Near-Black Ink** (`--primary`): the inverted ink used for primary CTAs and active state. Dark `hsl(0 0% 98%)` `#FAFAFA` / Light `hsl(220 8% 5%)` `#0C0C0E`. In dark mode the CTA is a white-on-black button; in light mode it flips to black-on-white. The inversion is the point — the primary action is the highest-contrast object on the screen.
- **Primary Foreground** (`--primary-foreground`): text on the primary surface. Dark `hsl(220 15% 8%)` `#111317` / Light `hsl(0 0% 98.5%)` `#FBFBFB`.

### Neutral
- **Deep Cool-Neutral Void** (`--background`): the page surface. Dark `hsl(220 12% 4%)` `#090A0B` / Light `hsl(220 3% 98.5%)` `#FBFBFB`. The main content sits on a faint `--muted/5` tint above this to separate the page from the chrome.
- **Crisp White / Near-Black** (`--foreground`, `--ink`): body text and icon ink. Dark `hsl(0 0% 96%)` `#F5F5F5` / Light `hsl(220 8% 5%)` `#0C0C0E`.
- **Raised Slate** (`--card`, `--popover`): cards, popovers, dialogs. Dark `hsl(220 10% 9%)` `#151619` / Light `hsl(0 0% 100%)` `#FFFFFF`. One step lighter than the void in dark mode; pure white in light mode.
- **Deeper Slate** (`--sidebar`): the navigation rail. Dark `hsl(220 14% 7%)` `#0F1114` / Light `hsl(220 5% 97.5%)` `#F8F9F9`. Slightly cooler and darker than the card to read as a recessed column.
- **Muted Slate** (`--muted`): secondary fills, table hover, filter bars. Dark `hsl(220 9% 15%)` `#23252A` / Light `hsl(220 4% 94%)` `#EFEFF0`.
- **Soft Slate** (`--muted-foreground`): secondary text, captions, metadata. Dark `hsl(220 5% 62%)` `#999CA3` / Light `hsl(220 4% 38%)` `#5D6065`. Tuned for WCAG AA against both surfaces — never lighten this "for elegance."
- **Hairline Slate** (`--border`, `--input`): every 1px divider and input stroke. Dark `hsl(220 10% 22%)` `#32363E` / Light `hsl(220 4% 88%)` `#DFE0E2`.

### Action & HTTP Method Tints (Tertiary — semantic only)
Muted but perceptible hues used to disambiguate HTTP methods, action types, and categorical data (usage bars, chart series). Chroma is tuned so each hue reads as its color while staying clearly muted — never decoration, never large-surface fills.
- **Muted Azure** (`--action-blue` / `--http-get`, GET): Dark `hsl(210 26% 64%)` `#8BA3BB` / Light `hsl(210 22% 35%)` `#46596D`.
- **Pale Iris** (`--action-violet` / `--http-post`, POST): Dark `hsl(260 24% 80%)` `#C8C0D8` / Light `hsl(260 22% 9%)` `#15121C`.
- **Muted Amber** (`--action-amber` / `--http-put`, PUT/PATCH): Dark `hsl(35 28% 60%)` `#B69E7C` / Light `hsl(35 24% 38%)` `#78654A` (`--http-put` mirrors one step lighter at `hsl(35 24% 48%)`).
- **Muted Coral** (`--action-rose` / `--http-delete`, DELETE): Dark `hsl(350 28% 64%)` `#BD8992` / Light `hsl(350 26% 45%)` `#91555F`.
- **Muted Teal** (`--action-teal`, embedding): the fifth capability tint in the model plaza — vision→azure, image generation→iris, TTS→amber, STT→coral, embedding→teal. Dark `hsl(180 26% 62%)` `#85B7B7` / Light `hsl(180 24% 30%)` `#3A5F5F`.

### Status Tints (Tertiary — semantic only)
Desaturated but recognizable; always rendered as a tinted badge (`bg/15 text border/30`) with a text label and `role="status"`. Success and error pair across themes via light/dark aliases — `--status-success` resolves to `--success-light` in light mode and `--success-dark` in dark mode; `--status-error` resolves to `--error-light` / `--error-dark`. The pairs are first-class tokens, used directly by error panels.
- **Sage Mint** (`--status-success`): Dark `hsl(150 42% 58%)` `#67C194` / Light `hsl(150 42% 34%)` `#327B57`.
- **Muted Gold** (`--status-warning`): Dark `hsl(32 42% 58%)` `#C19767` / Light `hsl(32 42% 42%)` `#986E3E`.
- **Salmon Rose** (`--status-error`): Dark `hsl(4 52% 72%)` `#DD9792` / Light `hsl(4 45% 45%)` `#A6463F`.
- **Neutral Slate** (`--status-unknown`): Dark `hsl(220 5% 58%)` `#8F9299` / Light `hsl(220 3% 45%)` `#6F7276`.
- **Desaturated Rose** (`--destructive`): destructive actions and the logout hover. Dark `hsl(4 52% 52%)` `#C44D45` / Light `hsl(4 48% 46%)` `#AE453D`.

### Syntax & Code Palette (code surfaces only)

A separate low-chroma family for code and payload surfaces — JSON viewers, highlighted snippets, and chat code blocks. It differentiates structure for scanning, never decoration, and never leaks into badges or chart series.
- **JSON syntax tokens** (`--json-string`, `--json-number`, `--json-boolean`, `--json-null`, `--json-key`, `--json-keyword`, `--json-function`): the highlighting palette behind `JsonViewer` and the client-side highlighter (`.text-json-*`). Strings and numbers split green/azure (`--json-string` Dark `hsl(145 55% 66%)` `#79D8A0` / Light `hsl(145 60% 34%)` `#238B4E`), booleans and keywords carry violet-rose warmth (`--json-keyword` Dark `hsl(260 70% 76%)` `#B497ED` / Light `hsl(260 60% 45%)` `#5C2EB8`), keys stay near-neutral (`--json-key` Dark `hsl(220 20% 78%)` `#BCC3D2` / Light `hsl(220 20% 25%)` `#333C4D`), and comments fall back to `--muted-foreground` italic.
- **Code block tokens** (`--code-bg`, `--code-header-bg`, `--code-border`): the progressive-gray surfaces behind code blocks (chat messages, request/response payloads) — a `rounded-xl` wrapper with a header bar one step lighter than the body. Dark `hsl(220 10% 8%)` `#121416` / header `hsl(220 10% 11.5%)` `#1A1C20` / border `hsl(220 10% 18%)` `#292C32`; Light `hsl(220 6% 96%)` `#F4F5F5` / header `hsl(220 6% 92.5%)` `#EBEBED` / border `hsl(220 8% 86%)` `#D8DADE`.

### Named Rules
**The Monochrome-First Rule.** Black, white, and the cool-neutral slate ramp carry 90%+ of any screen. The action/status tints are permitted only on badges, icon containers, metric values, and hover glows — never as section backgrounds, never as button colors (except `destructive`), never as large fills. If a screen reads as "colorful," it has failed this rule.

**The Tint, Not the Hue Rule.** Status and HTTP colors render at low alpha (`/15` background, `/25–30` border) with the full token as text. The badge is a labeled chip, not a swatch. Color is redundant information; the text label and `role="status"` are the primary signal.

**The Cool-Neutral Undertone Rule.** Every neutral token carries hue 220° at very low chroma (3–14%). Never introduce a true-gray (`0 0%`) neutral or a warm-neutral (`hue 30–100`) — both break the cool-console atmosphere and edge toward the cream/SaaS default this system rejects.

**The Quiet Syntax Rule.** The JSON and code palettes differentiate structure at low chroma for scanning — and only on code surfaces (JSON viewers, highlighted snippets, code blocks). Never reuse syntax tokens for badges, chart series, or fills; a rainbow syntax theme is a defect, not a feature.

## Typography

**Display Font:** Space Grotesk (with Noto Sans SC + Manrope fallback, metric-matched)
**Body Font:** Manrope (with Noto Sans SC fallback, metric-matched)
**Mono Font:** IBM Plex Mono (with JetBrains Mono / ui-monospace fallback, metric-matched)

**Character:** A geometric-grotesk display paired with a humanist body and a purpose-built data mono. The display carries page identity at small sizes (in-app titles, not marketing heroics); the body is workhorse UI text; the mono is the voice of the product's data — every number, token, code value, HTTP method, and timestamp is set in IBM Plex Mono with `tabular-nums` so columns align and values don't dance.

### Font Fallbacks (Metric-Matched)
To minimize layout shift during font loading, metric-adjusted fallbacks are defined as `@font-face` families:
- **Manrope Fallback**: `Segoe UI` / `Arial` with `size-adjust: 103.5%`, `ascent-override: 92%`, `descent-override: 24%`.
- **Space Grotesk Fallback**: `Segoe UI` / `Arial` with `size-adjust: 106%`, `ascent-override: 88%`, `descent-override: 22%`.
- **IBM Plex Mono Fallback**: `Consolas` / `Monaco` with `size-adjust: 98%`, `ascent-override: 94%`, `descent-override: 22%`.

### Hierarchy
The scale is a 1.25 (Major Third) modular scale on a 16px base, expressed in fixed `rem` so app UI stays spatially predictable.

- **Display** (Space Grotesk, 600, `text-xl`/`text-2xl` 1.25–1.5rem, line-height 1.25, tracking −0.02em): page titles via `.brand-heading`. In-app weight is 600 (not 400); display weight 400 is reserved for rare marketing headlines only. Floor tracking is −0.04em.
- **Headline** (Space Grotesk, 400, 2rem, line-height 1.1, tracking −0.02em): large display headlines, rarely used in-app (`.text-display-lg`/`.text-display-xl`).
- **Title** (Manrope, 600, 1.125rem, line-height 1.375): card titles and form-section headings (`.text-title`).
- **Body** (Manrope, 400, 1rem, line-height 1.5): primary reading text (`.text-body`). Max line length ~65–75ch (`.text-measure`).
- **Secondary** (Manrope, 400, 0.875rem, line-height 1.5): descriptions, secondary text (`.text-secondary`).
- **Label** (Manrope, 500, 0.75rem, tracking 0.02em, uppercase when eyebrow): captions, metadata, badges, form labels (`.text-caption`, `.text-label`).
- **Mono** (IBM Plex Mono, 400, 0.875rem, `tabular-nums`): code, JSON, technical values (`.text-code`, `.text-data`).
- **Metric** (IBM Plex Mono, 700, 1.5rem, `tabular-nums slashed-zero`, line-height 1): large stats (`.text-metric`).

### Dark-Mode Readability Adjustments
Dark mode adds 0.05–0.1 to body line-height for light-on-dark readability: `.dark body` → 1.55, `.text-body` → 1.6, `.text-secondary` → 1.55, `.text-caption` → 1.35.

### Named Rules
**The Three Voices Rule.** Space Grotesk + Manrope + IBM Plex Mono is the whole system. Never introduce a fourth family. A number or code value set in Manrope is a defect, not a stylistic choice.

**The Mono-Data Rule.** Every numeric value — tokens, cost, latency, counts, timestamps, status codes, HTTP methods — is set in IBM Plex Mono with `tabular-nums`. Use `.text-data` inline and `.text-metric` for large stats.

**The Display Restraint Rule.** Display weight is 600 in-app, sizes cap at `text-2xl` (1.5rem) for page titles, and emphasis comes from weight + size in a single solid color — never from `background-clip: text` or a gradient.

## Layout

A fixed left-rail / fluid-`main` app shell built for scan-first density, not marketing space.

- **App shell:** `flex h-screen overflow-hidden`. Fixed `NavSidebar` (desktop `hidden md:flex`, mobile `Sheet` drawer) + `main` on a faint `bg-muted/5` page tint with responsive padding `px-4 sm:px-6 lg:px-10 xl:px-16 py-5 sm:py-6`. A `full` layout mode (no padding, child-managed) drives the console pages.
- **Sidebar:** `w-64` expanded / `w-16` collapsed, `border-r border-sidebar-border`, a `brand-sidebar-shell` (180° sidebar→sidebar/96 gradient), and a 2px `.sidebar-accent` gradient rail line. Collapsible with 100ms-delay tooltip labels in collapsed state.
- **Page header:** one per page, the consistent anchor — `flex` row with an 8×8 `.icon-container` (tinted tile, top sheen), a `.brand-heading` title (`text-xl sm:text-2xl`, `#page-title`), a `text-muted-foreground text-xs max-w-2xl` description, and an `actions` slot (`animate-in fade-in duration-300`, optionally in a `rounded-xl border-border/55 bg-card/76` toolbar).
- **Grids:** `.card-grid-2` (`grid-cols-1 lg:grid-cols-2`, gap 4/6), `.card-grid-3` (`md:grid-cols-3`, gap 4), `.card-grid-4` (`grid-cols-2 lg:grid-cols-4`, gap 4).
- **Spacing rhythm:** Tailwind scale — 8 / 12 / 16 / 20 / 24 / 32 / 40 / 48 / 64px. Page sections use `space-y-4 md:space-y-6`; `.page-section` is the canonical wrapper.
- **Console layout (signature — management pages):** the "card wrapping card" anti-pattern is explicitly rejected. Providers, API Keys, Models, MCP Servers use a flush console: `config-header-bar` (sticky `bg-background` + `border-b/60`) + `config-toolbar` (flush `bg-background` + `border-b/60`) + `config-content` (scrollable) with a pinned `config-thead` (`bg-background`, header cells `bg-muted/50 border-b-2 border-border/70`). One seamless surface separated only by hairlines; table rows sit on the faint page tint.
- **Tables:** dense, hairline-separated, `whitespace-nowrap` by default; `.hide-on-mobile` / `.hide-on-tablet` drop less-important columns at `lg`/`md`; touch cells are `min-h-11 min-w-11` (44px).
- **Measure:** long-form body capped at 65ch (`.text-measure`), prose at 70ch (`.text-prose`).
- **Responsive:** mobile-first; `sm:640 md:768 lg:1024 xl:1280`. Container queries (`@sm/@md/@lg`) exist for component-level responsiveness.
- **Atmosphere layers:** two fixed `body::before`/`::after` layers behind every screen — a trio of soft radial foreground-glints (opacity 0.05–0.12) and a 42px grid masked toward viewport center (opacity 0.45–0.6), dimmer in dark mode. They never sit above content (`z-index: -2/-1`, `pointer-events: none`).

### Named Rules
**The One-Console Rule.** Management pages are one seamless surface — sticky header, flush toolbar, scrollable body — separated only by `border-b/60` hairlines. Never wrap a console table in an outer Card; never compete with the table's own thead.

**The Page-Header Anchor Rule.** Every route opens with exactly one `.page-header` (tinted icon container + `.brand-heading` title + muted description + actions). It is the consistent entry point a stranger trusts cold.

## Elevation & Depth

Flat-by-default. Depth is conveyed by **tonal background steps** (`--card` over `--background`, the `--muted/5` page tint, `--muted/50` table hover) and **1px hairline borders**, not by structural drop shadows. Shadows, when they appear, are ambient and state-driven, never decorative.

### Shadow Vocabulary
- **Rest card shadow** (`shadow-xs`): the default on `.card-container` / `.card-base` / primary buttons — a barely-there lift that confirms the card sits above the page. `.card-flat` omits it entirely.
- **Section-card wash** (`0 8px 24px -16px hsl(<color> / 0.2–0.25)`): a soft, colored, downward-diffused glow under the tinted section cards (`.section-card-primary/-blue/-amber/-success`). Reserved for semantically-grouped regions; never a default card style.
- **Sidebar edge shadow** (`16px 0 36px -32px sidebar/0.95`): the soft right-edge shadow on the nav rail.
- **Hover glow** (`0 0 20px -6px hsl(<color> / 0.35–0.4)`): `.hover-glow-primary/-blue/-amber/-success` — a constrained ambient halo on interactive elements in the matching semantic color.
- **Inset top highlight** (`inset 0 1px 0 hsl(var(--foreground) / 0.06)` on cards; `inset 0 1px 0 hsl(var(--background) / 0.9)` on inputs): the single-pixel top sheen that reads as a lit edge on raised surfaces.
- **Auth-mark shadow** (`0 18px 36px -20px hsl(var(--primary) / 0.55)` + inset): the floating login mark.

### Named Rules
**The Flat-By-Default Rule.** Surfaces are flat at rest. Shadows appear only as a response to state (hover, elevation, focus) or as the constrained section-card / auth-mark exceptions. Never pair a 1px border with a 16px+ blur drop shadow on the same element (the ghost-card pattern) — pick a tonal step or `shadow-xs`, never a wide decorative shadow.

**The Tonal-Step Rule.** When you need to separate two surfaces, step the background token (`--background` → `--card` → `--muted`) before reaching for a shadow. A 1px `--border` hairline plus a tonal step is the default depth recipe.

## Shapes

Rounded, restrained, and tiered by component class. Corners never get loud.

- **Base radius** (`--radius`): 0.75rem (12px). All other radii derive from it.
- **Cards, panels, section containers, icon containers** (`rounded-xl`, 16px): the largest radius in the system. Cap for cards and panels is 16px — no 20/24/28/32/40px card radii.
- **Buttons, inputs, badges' outer containers, code containers** (`rounded-md`, 10px): the control radius.
- **Nav items, tooltips, code containers** (`rounded-lg`, 12px): nav items get a slightly softer corner than controls.
- **Tags, status chips, the range thumb** (`rounded-full` / pill, 9999px): full-pill is reserved for tags and status chips, never for cards or buttons.
- **Borders:** 1px `--border` everywhere by default; the one 2px exception is the active-nav gradient indicator and the `config-thead` bottom rule. `border-left/right` colored accent stripes on cards/list items/alerts are prohibited (the 2px active nav indicator is the dedicated active-state affordance, not decoration).

### Named Rules
**The Radius Cap Rule.** Cards and panels cap at `rounded-xl` (16px). Save full-pill for tags and status chips; `rounded-lg` (12px) for nav items. No 24/28/32/40px card radii, no pill buttons.

**The One 2px Stripe Rule.** The only colored >1px stripe in the system is the 2px active-nav gradient indicator. Never use a colored `border-left/right` accent stripe on cards, list items, or alerts.

## Components

For each component, the character line first, then shape, color, states, and distinctive behavior.

### Buttons
Tactile and confident, but quiet — the primary CTA is the highest-contrast object on screen, everything else recedes.
- **Shape:** `rounded-md` (10px), `h-10 px-4 py-2` default; `text-sm font-semibold tracking-[0.01em]`; `transition-[color,background-color,border-color,box-shadow] 200ms`.
- **Primary:** `bg-primary text-primary-foreground shadow-xs hover:bg-primary/90` — white-on-black (dark) / black-on-white (light).
- **Destructive:** `bg-destructive text-white shadow-xs hover:bg-destructive/90`; focus ring is `ring-destructive/20` (`/40` dark).
- **Outline:** `border border-border/80 bg-background/80 shadow-xs backdrop-blur-sm hover:border-primary/35 hover:bg-accent/70` — the one place `backdrop-blur` appears on a button, a deliberate constrained set. Dark mode uses `bg-input/30 border-input`.
- **Secondary:** `bg-secondary/85 text-secondary-foreground hover:bg-secondary`.
- **Ghost:** `hover:bg-accent/70 hover:text-accent-foreground` — no border, no fill at rest.
- **Link:** `text-primary underline-offset-4 hover:underline`.
- **Focus (all):** `focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]` — a 3px ring, the strongest focus signal in the system. `aria-invalid` switches the ring to `ring-destructive/20` (`/40` dark) and the border to `destructive`.
- **Sizes:** default `h-10`; `sm` `h-9`; `lg` `h-11`; `icon` `size-10`; `icon-sm` `size-9`; `icon-lg` `size-11`. Icons scale to `size-4` by default.
- **Press:** `.btn-press:active { transform: scale(0.98) }` (100ms), neutralized under reduced motion.

### Status Badge (signature)
A labeled chip, not a swatch. Color is redundant; the text label and `role="status"` are the signal.
- **Shape:** `rounded-full border px-2 py-0.5 text-xs font-medium`, `Badge variant="outline"`, `role="status"`, slot `truncate`.
- **State:** rendered as `bg-<token>/15 text-<token> border-<token>/25–30`. Two families: `status` (success/warning/error/unknown) and `http` (GET→blue, POST→violet, PUT/PATCH→amber, DELETE→rose).
- **Base badge variants** (non-status): `default` (primary), `secondary`, `destructive`, `warning` (`amber-500/15 text-amber-600 dark:text-amber-400`), `outline` (text-only, hover bg-accent).

### Cards / Containers
Flat raised slates, hairline-edged, tonally stepped above the void.
- **Corner:** `rounded-xl` (16px).
- **Background:** `bg-card` (`--card`), one step lighter than `--background`.
- **Shadow:** `shadow-xs` at rest (`.card-container`, `.card-base`); `.card-flat` omits the shadow entirely.
- **Hover:** border tightens — `.card-container:hover` → `border-border/80`; `.hover-card:hover` → `border-primary/40`. Never a shadow lift.
- **Tinted section cards** (`.section-card-primary/-blue/-amber/-success`): `rounded-xl` + `border-<color>/20` + `bg-linear-to-br from-<color>/8 via-card/95 to-card` + the section-card wash shadow. Reserved for semantically-grouped regions; not a default card style.

### Inputs / Fields
A lit-edge control with the strongest focus signal in the system.
- **Style:** `h-10 rounded-md border border-input/80 bg-background/75 px-3.5 py-2 text-base md:text-sm`, `shadow-[inset_0_1px_0_hsl(var(--background)/0.9)]` (the inset top highlight), `backdrop-blur-sm`, `transition-[color,box-shadow,border-color,background-color]`.
- **Focus:** `focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]` — a 3px primary ring, the strongest focus signal in the system.
- **Invalid:** `aria-invalid:border-destructive` + `ring-destructive/20`.
- **Placeholder:** `text-muted-foreground` (tuned for AA contrast, not lightened).
- **Auth input (login/setup):** `.auth-input` — `bg-card/62 backdrop-blur(10px)`, `border-border/70`; focus shifts to `bg-background/92`, `border-primary/55`, `box-shadow: 0 0 0 3px primary/18`. The other place `backdrop-blur` is allowed.

### Navigation (`NavSidebar`)
A recessed column with a gradient rail; items are quiet until active, then marked by a 2px gradient indicator.
- **Shell:** `w-64`/`w-16`, `border-r border-sidebar-border`, `brand-sidebar-shell` gradient, 2px `.sidebar-accent` rail line, soft right shadow.
- **Items:** `rounded-lg px-3 py-2.5 text-sm font-medium transition-all 200ms`, `min-h-11 min-w-11`. Active: `bg-sidebar-accent/95` + the 2px gradient indicator line (`from-sidebar-primary to-sidebar-primary/60`, `rounded-r-full`, active glow) + icon `scale-110`. Hover: `bg-sidebar-accent/70` + a horizontal reflection sweep (`via-sidebar-primary/12`, opacity 0→100). Logout hover tints `destructive/16`.
- **Section labels:** `text-[11px] font-semibold tracking-[0.14em] uppercase text-sidebar-foreground/50` — a tracked eyebrow used **only** for the three nav section groups (Overview / Tools / Config), never as a page-section eyebrow.

### Color-Tinted Icon Container (signature)
`.icon-container` and its `-primary/-blue/-amber/-success` variants: `p-2.5 rounded-xl min-h-11 min-w-11`, `bg-linear-to-br from-<color>/15–20 via-<color>/10 to-<color>/5–12`, `ring-1 ring-<color>/25–30`, `inset 0 1px 0 <color>/0.2–0.26`. The signature treatment for the icon beside a page title or section heading — a flat, tinted, lightly-ringed tile with a top sheen.

### Range Slider (signature control)
Used by Smart Routing mode-weight sliders. `appearance-none` + `.range-thumb`; track `0.5rem rounded-full bg-muted`, thumb `1rem rounded-full border-2 bg-background bg-primary` with a `primary/40` glow; hover `scale(1.12)`, active `scale(0.96)`, focus `0 0 0 4px ring`. Reduced-motion removes the thumb transform.

### Toasts (vue-sonner)
Themed via the same tokens; sit on `--popover`, hairline `--border`, `rounded-lg`, with the semantic status tints for success/warning/error variants. No celebratory motion.

### Code Blocks & JSON Viewer

Code is a quiet, progressive-gray surface — never a dark editor pane, never a rainbow theme.
- **Code block** (chat messages, log payloads): `rounded-xl` wrapper on `--code-bg` with a `border-b border-code-border` header bar on `--code-header-bg` (language label + copy button, `text-xs text-muted-foreground`), then a `p-4` `<pre>` in IBM Plex Mono `text-[13px] leading-relaxed text-foreground/90`.
- **JSON viewer** (`JsonViewer`): `vue-json-pretty` themed through the syntax tokens at 12px mono (`--vue-json-pretty-theme-*`): keys in `--json-key`, strings/numbers/booleans/null in their syntax tokens, property names in `--foreground`.

## Do's and Don'ts

Concrete visual guardrails grounded in the implemented system.

### Do:
- **Do** set every numeric value — tokens, cost, latency, counts, timestamps, status codes — in IBM Plex Mono with `tabular-nums` (`.text-data` inline, `.text-metric` for large stats).
- **Do** convey depth with tonal background steps (`--card` over `--background`, `--muted/5` page tint) and 1px `--border` hairlines; reserve shadow for `shadow-xs` at rest and ambient glow on hover.
- **Do** render status and HTTP semantics as labeled `StatusBadge` chips at `/15` tint + `/25–30` border, with `role="status"` and a text label as the primary signal.
- **Do** use the `config-page-reveal` entrance on every page and the `stagger-children` / `row-stagger` cascades on first render — and let the global `prefers-reduced-motion` rule neutralize them.
- **Do** keep display tracking at −0.02em (floor −0.04em), display weight 600 in-app, and page titles at `text-xl sm:text-2xl`.
- **Do** hit WCAG AA on both themes: tune `--muted-foreground` for readability, never lighten muted text "for elegance," and bump dark-mode body line-height to 1.55–1.6.
- **Do** keep touch targets ≥44px (`min-h-11 min-w-11`) on nav and mobile controls.
- **Do** render JSON and code with the syntax palette on code surfaces only — subtle chroma that reads as structure, never a rainbow theme.

### Don't:
- **Don't** make it playful/consumer — no cute illustrations, mascots, gamification, or celebratory confetti. This is infrastructure. *(PRODUCT.md anti-reference.)*
- **Don't** ship boring/outdated admin panels — no dense gray-on-gray tables, no circa-2008 enterprise chrome, no cluttered "everything on one screen" dashboards. *(PRODUCT.md anti-reference.)*
- **Don't** use loud color accents — no saturated SaaS gradients, no neon, no rainbow status badges. Color is a subtle low-saturation tinting system for semantic differentiation, never decoration. *(PRODUCT.md anti-reference.)*
- **Don't** ship the generic 2026 AI-tool aesthetic — no cream/sand warm-neutral body backgrounds, no gradient-text headlines, no identical icon+heading+text card grids, no big-number hero-metric template. This is monochrome editorial, not warm SaaS. *(PRODUCT.md anti-reference.)*
- **Don't** pair a 1px border with a 16px+ blur drop shadow on the same element (the ghost-card pattern). Pick a tonal step or `shadow-xs`, never a wide decorative shadow.
- **Don't** over-round — cards and panels cap at `rounded-xl` (16px); save full-pill for tags and status chips, `rounded-lg` (12px) for nav items. No 24/28/32/40px card radii.
- **Don't** use `border-left`/`border-right` greater than 1px as a colored accent stripe on cards, list items, or alerts. The 2px active nav indicator is the one exception, and it is a dedicated active-state affordance, not decoration.
- **Don't** use glassmorphism decoratively. `backdrop-blur-sm` appears only on `outline` buttons, inputs, and the login-screen `auth-input`/`auth-mark` — a deliberate, constrained set, never a default card treatment.
- **Don't** use `background-clip: text` with a gradient. Display emphasis comes from weight and size, in a single solid color.
- **Don't** put a tiny uppercase tracked eyebrow above every page section. The tracked eyebrow is reserved for the three nav section groups only.
- **Don't** set a metric or number in Manrope, or introduce a fourth type family. Space Grotesk + Manrope + IBM Plex Mono is the whole system.
- **Don't** introduce true-gray (`0 0%`) or warm-neutral (`hue 30–100`) tokens. Every neutral carries the 220° cool undertone.
- **Don't** reuse the JSON/code syntax tokens outside code surfaces — no rainbow syntax themes, no syntax-colored badges or chart series.
- **Don't** animate layout properties. Motion is `transform`/`opacity` only, exponential ease-out (`cubic-bezier(0.16, 1, 0.3, 1)`), 150–500ms, with a reduced-motion alternative on every animation.
