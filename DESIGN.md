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
  sidebar-accent: "#22252B"  # --sidebar-accent  hsl(220 12% 15%) — active nav step
  sidebar-ink: "#EBEBEB"      # --sidebar-foreground  hsl(0 0% 92%)
  sidebar-accent-ink: "#F0F0F0"  # --sidebar-accent-foreground  hsl(0 0% 94%)
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
  overlay: "#000000"          # --overlay      hsl(0 0% 0%) — modal/sheet scrim
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
  eyebrow:
    fontFamily: "Manrope, Noto Sans SC, sans-serif"
    fontSize: "11px"
    fontWeight: 600
    lineHeight: 1.25
    letterSpacing: "0.05em"
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
  auth-input:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.md}"
    padding: "8px 14px"
    height: "44px"
  card:
    backgroundColor: "{colors.card}"
    textColor: "{colors.ink}"
    rounded: "{rounded.xl}"
    padding: "20px"
  badge-status:
    backgroundColor: "{colors.muted}"
    textColor: "{colors.muted-ink}"
    typography: "{typography.label}"
    rounded: "{rounded.pill}"
    padding: "2px 8px"
  nav-item:
    backgroundColor: "transparent"
    textColor: "{colors.sidebar-ink}"
    typography: "{typography.label}"
    rounded: "{rounded.lg}"
    padding: "8px"
    height: "32px"
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
- Three structural grammars, chosen by task: the flush console (management tables), stacked hairline panels (settings and security forms), and the containerless vestibule (auth).
- The 11px tracked uppercase label is a **label rail** — an in-panel micro-heading — never a page-section kicker.

## Colors

A two-theme monochrome system with a cool-neutral (hue 220°) undertone. Dark mode is canonical (first value); light mode is the crisp mirror (second value). All tokens are HSL in source (`hsl(H S% L%)`); hex is the sRGB rendering. **The HSL token is the source of truth; edit the token, not the hex.**

### Primary
- **Console White / Near-Black Ink** (`--primary`): the inverted ink used for primary CTAs and active state. Dark `hsl(0 0% 98%)` `#FAFAFA` / Light `hsl(220 8% 5%)` `#0C0C0E`. In dark mode the CTA is a white-on-black button; in light mode it flips to black-on-white. The inversion is the point — the primary action is the highest-contrast object on the screen.
- **Primary Foreground** (`--primary-foreground`): text on the primary surface. Dark `hsl(220 15% 8%)` `#111317` / Light `hsl(0 0% 98.5%)` `#FBFBFB`.
- **Vestibule Glow**: on the auth screens `--primary` becomes a light source. The 48px auth mark's `135deg` gradient runs `primary/0.16 → primary/0.06 → action-blue/0.12`, and the 620px radial behind it is `primary/0.12 → action-blue/0.06 → transparent`. Primary is the only token permitted to glow that large, and only on this one surface.

### Neutral
- **Deep Cool-Neutral Void** (`--background`): the page surface. Dark `hsl(220 12% 4%)` `#090A0B` / Light `hsl(220 3% 98.5%)` `#FBFBFB`. The main content sits on a faint `--muted/5` tint above this to separate the page from the chrome.
- **Crisp White / Near-Black** (`--foreground`): body text and icon ink. Dark `hsl(0 0% 96%)` `#F5F5F5` / Light `hsl(220 8% 5%)` `#0C0C0E`.
- **Raised Slate** (`--card`, `--popover`): cards, popovers, dialogs. Dark `hsl(220 10% 9%)` `#151619` / Light `hsl(0 0% 100%)` `#FFFFFF`. One step lighter than the void in dark mode; pure white in light mode.
- **Deeper Slate** (`--sidebar`): the navigation rail. Dark `hsl(220 14% 7%)` `#0F1114` / Light `hsl(220 5% 97.5%)` `#F8F9F9`. Slightly cooler and darker than the card to read as a recessed column. Its own ink pair lives with it — `--sidebar-foreground` Dark `hsl(0 0% 92%)` `#EBEBEB` / Light `hsl(220 6% 12%)` `#1D1E20`, and `--sidebar-accent-foreground` Dark `hsl(0 0% 94%)` / Light `hsl(220 6% 18%)` — so the rail never borrows the page's text color.
- **Muted Slate** (`--muted`): secondary fills, table hover, filter bars. Dark `hsl(220 9% 15%)` `#23252A` / Light `hsl(220 4% 94%)` `#EFEFF0`.
- **Soft Slate** (`--muted-foreground`): secondary text, captions, metadata. Dark `hsl(220 5% 62%)` `#999CA3` / Light `hsl(220 4% 38%)` `#5D6065`. Tuned for WCAG AA against both surfaces — never lighten this "for elegance."
- **Hairline Slate** (`--border`, `--input`): every 1px divider and input stroke. Dark `hsl(220 10% 22%)` `#32363E` / Light `hsl(220 4% 88%)` `#DFE0E2`.
- **Active Step** (`--sidebar-accent`): the filled tonal step that marks the current nav item. Dark `hsl(220 12% 15%)` `#22252B` / Light `hsl(220 4% 93%)`. State is a step on this ramp, never a stripe or a color.
- **Scrim Black** (`--overlay`): the modal, sheet, and image-lightbox scrim, `hsl(0 0% 0%)` at dialog-managed opacity. It is the only true black in the system and it never appears as a surface.

### Action & HTTP Method Tints (Tertiary — semantic only)
Muted but perceptible hues used to disambiguate HTTP methods, action types, and categorical data (usage bars, chart series). Chroma is tuned so each hue reads as its color while staying clearly muted — never decoration, never large-surface fills.
- **Muted Azure** (`--action-blue` / `--http-get`, GET): Dark `hsl(210 26% 64%)` `#8BA3BB` / Light `hsl(210 22% 35%)` `#46596D`.
- **Pale Iris** (`--action-violet` / `--http-post`, POST): Dark `hsl(260 24% 80%)` `#C8C0D8` / Light `hsl(260 22% 9%)` `#15121C`.
- **Muted Amber** (`--action-amber` / `--http-put`, PUT/PATCH): Dark `hsl(35 28% 60%)` `#B69E7C` / Light `hsl(35 24% 38%)` `#78654A` (`--http-put` mirrors one step lighter at `hsl(35 24% 48%)`).
- **Muted Coral** (`--action-rose` / `--http-delete`, DELETE): Dark `hsl(350 28% 64%)` `#BD8992` / Light `hsl(350 26% 45%)` `#91555F`.
- **Muted Teal** (`--action-teal`, embedding): the fifth capability tint in the model plaza — vision→azure, image generation→iris, TTS→amber, STT→coral, embedding→teal. Dark `hsl(180 26% 62%)` `#85B7B7` / Light `hsl(180 24% 30%)` `#3A5F5F`.

### Status Tints (Tertiary — semantic only)
Desaturated but recognizable; always rendered as a tinted badge (`bg/15 text border/30`) with a text label and `role="status"`. Success and error pair across themes via light/dark aliases — `--status-success` resolves to `--success-light` in light mode and `--success-dark` in dark mode; `--status-error` resolves to `--error-light` / `--error-dark`. Surface code consumes the resolved `--status-*` token only: the light/dark halves exist to define it, never as a second encoding alongside it.
- **Sage Mint** (`--status-success`): Dark `hsl(150 42% 58%)` `#67C194` / Light `hsl(150 42% 30%)` `#2C6D4D`.
- **Muted Gold** (`--status-warning`): Dark `hsl(32 42% 58%)` `#C19767` / Light `hsl(32 42% 34%)` `#7B5932`.
  (Light-mode status lightness is set by the AA requirement against the badge's own `/15` tint, not against the page alone — see §Colors.)
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
- **Headline** (Space Grotesk, 400, 2rem, line-height 1.1, tracking −0.02em): large display headlines, rarely used in-app.
- **Title** (Manrope, 600, 1.125rem, line-height 1.375): card titles and form-section headings (`text-lg font-semibold`).
- **Body** (Manrope, 400, 1rem, line-height 1.5): primary reading text (`text-base`). Max line length ~65–75ch (`max-w-[65ch]`).
- **Secondary** (Manrope, 400, 0.875rem, line-height 1.5): descriptions, secondary text (`text-sm`, usually `text-muted-foreground`).
- **Label** (Manrope, 500, 0.75rem, tracking 0.02em): captions, metadata, badges, form labels (`text-xs font-medium`).
- **Label Rail** (Manrope, 600, 11px, tracking 0.05em, uppercase): the in-panel micro-heading — the label row above a spec-sheet value, a log-detail panel heading, a segmented view toggle. The nav groups raise it to `tracking-[0.14em]`.
- **Mono** (IBM Plex Mono, 400, 0.875rem, `tabular-nums`): code, JSON, technical values (`.text-code`, `.text-data`).
- **Metric** (IBM Plex Mono, 700, 1.5rem, `tabular-nums slashed-zero`, line-height 1): large stats (`.text-metric`).

### Dark-Mode Readability Adjustments
Dark mode adds 0.05 to body line-height for light-on-dark readability: `.dark body` → 1.55.

### Named Rules
**The Three Voices Rule.** Space Grotesk + Manrope + IBM Plex Mono is the whole system. Never introduce a fourth family. A number or code value set in Manrope is a defect, not a stylistic choice.

**The Mono-Data Rule.** Every numeric value — tokens, cost, latency, counts, timestamps, status codes, HTTP methods — is set in IBM Plex Mono with `tabular-nums`. Use `.text-data` inline and `.text-metric` for large stats.

**The Display Restraint Rule.** Display weight is 600 in-app, sizes cap at `text-2xl` (1.5rem) for page titles, and emphasis comes from weight + size in a single solid color — never from `background-clip: text` or a gradient.

**The Label-Rail Rule.** The 11px tracked uppercase label is a *rail*, not a *kicker*: it may sit **inside** a panel to name the value beneath it (spec-sheet cells, log-detail headings, segmented toggles, nav groups), and it never sits **above** a page or a page section to introduce content. If deleting it would leave the heading unable to say what follows, it was a kicker and does not belong.

## Layout

A fixed left-rail / fluid-`main` app shell built for scan-first density, not marketing space.

- **App shell:** `flex h-screen overflow-hidden`. Fixed `NavSidebar` (desktop `hidden md:flex`, mobile `Sheet` drawer) + `main` on a faint `bg-muted/5` page tint with responsive padding `px-4 sm:px-6 py-5 sm:py-6`. A `full` layout mode (no padding, child-managed) drives the console pages.
- **Sidebar:** `16rem` expanded / `3rem` icon-collapsed (`--sidebar-width` → `--sidebar-width-icon`), `bg-sidebar` on a `border-r border-sidebar-border` rail. Collapsible with 200ms-delay tooltip labels in collapsed state.
- **Page header:** one per page, the consistent anchor — `flex` row with an `.icon-container` at its `.icon-container-sm` page-header size (tinted tile, top sheen), a `.brand-heading` title (`text-xl sm:text-2xl`, `#page-title`), a `text-muted-foreground text-xs max-w-2xl` description, and an `actions` slot (`animate-in fade-in duration-300`, optionally in a `rounded-xl border-border/55 bg-card/76` toolbar).
- **Grids:** Tailwind grid utilities, composed per surface — `grid-cols-1 lg:grid-cols-2` (gap 4/6), `md:grid-cols-3` (gap 4), `grid-cols-2 lg:grid-cols-4` (gap 4).
- **Spacing rhythm:** Tailwind scale — 8 / 12 / 16 / 20 / 24 / 32 / 40 / 48 / 64px. Page sections are `space-y-4 md:space-y-6`.
- **Console layout (signature — management pages):** the "card wrapping card" anti-pattern is explicitly rejected. Providers, Models, API Keys, MCP Servers, Team, and Settings use a flush console: `config-header-bar` (sticky `bg-background` + `border-b/60`) + `config-toolbar` (flush `bg-background` + `border-b/60`) + `config-content` (scrollable) with a pinned `config-thead` (`bg-background`, header cells `bg-muted/50 border-b-2 border-border/70`). One seamless surface separated only by hairlines; table rows sit on the faint page tint. Settings replaces the toolbar with a two-segment tab strip inside the page header. Section jumps live beside that strip in a compact dropdown whose trigger names the section currently in view — a bare "Sections" label told the operator nothing about their position on a ten-section page, so the scrollspy drives the label (`Request Policy`, `Circuit Breaker`) and `Sections` survives only as the button's `title` and as the stem of its accessible name (`Sections: Request Policy`). General is a flat option list, Advanced groups its sections under Routing & models / Resilience & delivery / Access & limits. The page is instance-wide and admin-only: theme, language and per-user tracing are personal settings and live in the account workspace instead. The page header stays on the shared `px-4 sm:px-6` console gutter while the cards use one centered `max-w-4xl` column. There is no second left rail and no full-width navigation band above the content; the section menu is available from the persistent header without competing with the first section.
- **Log Management grouping:** the settings panel divides its decisions into **Capture** (body/raw capture and masking), **Retention** (retention and manual cleanup), and **Audit** (body and audit sampling, size and retention). Each group exposes at most four primary controls; grouping changes reading order only, never the auto-save contract.
- **Model list-row column grammar (signature — Models page, both roles):** list and table modes are one column grammar. List rows carry fixed-width data columns (`.model-col-ctx` 80px, `.model-col-price` 96px, `.model-col-tier` 80px, `.model-col-actions` 76px) under a pinned `ModelListHead` (sticky, `bg-muted/50` + the 2px bottom rule, same gutter/gap/widths as the rows) whose cells repeat the table's sort controls — so list mode sorts and scans exactly like the table. The name cell is indented by the icon footprint (`pl-11`); model names are mono `text-[13px]` in every mode (they are API identifiers, the Mono-Data Rule). Collapse is by tier: the full grammar shows at `lg`; `sm–lg` keeps IN/OUT only (CACHED lives in either's pricing popover); below `sm` columns hide and each row's meta line carries context inline.
- **Tables:** dense, hairline-separated, `whitespace-nowrap` by default; touch cells are `min-h-11 min-w-11` (44px).
- **Auth layout (signature — "The Vestibule"):** every pre-console screen — login, first-run setup, force-change-password, invite acceptance, OAuth completion, email verification — renders through `AuthShell`; see §Components → Auth Screen for the full spec.
- **Personal-settings layout (Account):** one shared Account shell — a centered `max-w-4xl` column whose page header carries a three-segment **Profile / Preferences / Sign-in & security** tab strip (the same segmented control grammar as Settings). `/account`, `/account/preferences` and `/account/security` remain deep-linkable routes; the shell, not a per-page action button, is the navigation between them. Profile is identity only — a summary with editable username/email rows; Preferences holds what is personal but not identity (appearance: theme and language, stored in the browser; per-user tracing backends); Sign-in & security is every credential in strength order (passkeys → password → two-factor → linked providers → current session). See §Components → Personal-Settings Panels for the panel spec; personal settings are read top-to-bottom, not scanned as a grid.
- **Instance sign-in (admin form):** `/config/console-auth` is instance-wide login policy, not personal security. It uses the single-form variant: the standard full-width `config-header-bar` stays on the shared console gutter, with one centered `max-w-3xl` form column, stacked panels and a sticky save bar. No section index; the security tab links admins here instead of merging the two scopes.
- **Measure:** long-form body capped at 65ch (`max-w-[65ch]`), prose at 70ch (`max-w-[70ch]`).
- **Responsive:** mobile-first; `sm:640 md:768 lg:1024 xl:1280`. Container queries (`@sm/@md/@lg`) exist for component-level responsiveness.
- **Atmosphere layers:** two fixed `body::before`/`::after` layers behind every screen — a trio of soft radial foreground-glints (opacity 0.05–0.12) and a 42px grid masked toward viewport center (opacity 0.45–0.6), dimmer in dark mode. They never sit above content (`z-index: -2/-1`, `pointer-events: none`).

### Named Rules
**The One-Console Rule.** Management pages are one seamless surface — sticky header, flush toolbar, scrollable body — separated only by `border-b/60` hairlines. Never wrap a console table in an outer Card; never compete with the table's own thead.

**The Page-Header Anchor Rule.** Every route opens with exactly one `.page-header` (tinted icon container + `.brand-heading` title + muted description + actions). It is the consistent entry point a stranger trusts cold.

**The Task-Shape Rule.** Structure follows the task, not the page: management tables are one flush console; settings and security are stacked hairline panels; auth is containerless. Three grammars, one per task shape — never two on one screen.

## Elevation & Depth

Flat-by-default. Depth is conveyed by **tonal background steps** (`--card` over `--background`, the `--muted/5` page tint, `--muted/50` table hover) and **1px hairline borders**, not by structural drop shadows. Shadows, when they appear, are ambient and state-driven, never decorative.

### Shadow Vocabulary
- **Rest card shadow** (`shadow-xs`): the lift on primary buttons, the active settings tab, and the sidebar collapse handle. Panels do **not** carry it — their 1px `border-border/60` does the elevation work. (The `.brand-panel` utility carries `shadow-sm` at 7 call sites; it is not part of this recipe.)
- **Hover glow** (`0 0 20px -6px hsl(<color> / 0.35–0.4)`): `.hover-glow-primary/-blue/-amber/-success` — a constrained ambient halo on interactive elements in the matching semantic color.
- **Inset top highlight** (`inset 0 1px 0 hsl(var(--foreground) / 0.06)` on cards; `inset 0 1px 0 hsl(var(--background) / 0.9)` on inputs): the single-pixel top sheen that reads as a lit edge on raised surfaces.
- **Auth-mark shadow** (`0 18px 36px -20px hsl(var(--primary) / 0.55)` + inset): the floating vestibule mark — the single place where a shadow is ambient rather than state-driven, because the mark floats on the atmosphere instead of on a surface.
- **Auth-field focus** (`border-color` → `hsl(var(--primary) / 0.65)`, then `0 0 0 1px primary/0.55` and `0 0 0 2px primary/0.12`): the vestibule field's focus signal — the border itself sharpens into a crisp line with a faint glow, in place of the console's 3px `ring-ring/50`. A sign-in screen is allowed exactly one loud object (the white CTA); the field is not it.
- **Anchor flash** (`0 0 0 2px hsl(var(--primary) / 0.32)`): the 900ms `anchor-target-flash` ring that marks a section reached by a deep link, holding as a static 2px ring under reduced motion.

### Named Rules
**The Flat-By-Default Rule.** Surfaces are flat at rest. Shadows appear only as a response to state (hover, elevation, focus) or as the constrained auth-mark exception. Never pair a 1px border with a 16px+ blur drop shadow on the same element (the ghost-card pattern) — pick a tonal step or `shadow-xs`, never a wide decorative shadow.

**The Tonal-Step Rule.** When you need to separate two surfaces, step the background token (`--background` → `--card` → `--muted`) before reaching for a shadow. A 1px `--border` hairline plus a tonal step is the default depth recipe.

## Shapes

Rounded, restrained, and tiered by component class. Corners never get loud.

- **Base radius** (`--radius`): 0.75rem (12px). All other radii derive from it.
- **Cards, panels, section containers, icon containers** (`rounded-xl`, 16px): the largest radius in the system. Cap for cards and panels is 16px — no 20/24/28/32/40px card radii.
- **Buttons, inputs, badges' outer containers, code containers** (`rounded-md`, 10px): the control radius.
- **Nav items, tooltips, code containers** (`rounded-lg`, 12px): nav items get a slightly softer corner than controls.
- **Tags, status chips, the range thumb** (`rounded-full` / pill, 9999px): full-pill is reserved for tags and status chips, never for cards or buttons.
- **Identity tiles** (`rounded-xl`, 16px): the 48px `auth-mark` and the 56px account initials tile take the card corner, never the pill — an identity mark is a panel, not a chip.
- **Control heights:** 36 / 40 / 44 / 48px — `h-9` compact, `h-10` default, `h-11` for auth fields, the auth CTA, and every touch target on small screens (`max-sm:h-11`), `h-12` for the MFA code field.
- **Borders:** 1px `--border` everywhere by default. The only >1px rules are neutral 2px bottom rules on the pinned console header cells (`config-thead` / `ModelListHead`, `border-b-2 border-border/70`). A colored `border-left/right` accent stripe on cards, list items, or alerts is prohibited: active state is a tonal step, never a stripe.

### Named Rules
**The Radius Cap Rule.** Cards and panels cap at `rounded-xl` (16px). Save full-pill for tags and status chips; `rounded-lg` (12px) for nav items. No 24/28/32/40px card radii, no pill buttons.

**The No-Stripe Rule.** A colored >1px stripe never marks state anywhere in the system — active nav items are a filled `--sidebar-accent` step plus `font-medium`, and the only >1px rules are the neutral 2px console header bottom rules. Never use a colored `border-left/right` accent stripe on cards, list items, or alerts.

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
- **Sizes:** default `h-10`; `sm` `h-9`; `lg` `h-11`; `icon` `size-10`; `icon-sm` `size-9`; `icon-lg` `size-11`. Default, `sm`, `icon` and `icon-sm` lift to a 44px floor on small screens (`max-sm:h-11` / `max-sm:size-11`), so no button sits under the touch minimum on a phone. Icons scale to `size-4` by default.
- **Auth CTA:** primary, `h-11 w-full gap-2` (with `.btn-press`) — 44px tall, full column width. The vestibule shows exactly one white button: the passkey button when the account has one, otherwise the password submit; whichever loses becomes `outline`.
- **Press:** `.btn-press:active { transform: scale(0.98) }` (100ms), neutralized under reduced motion.

### Status Badge (signature)
A labeled chip, not a swatch. Color is redundant; the text label and `role="status"` are the signal.
- **Shape:** `rounded-full border px-2 py-0.5 text-xs font-medium`, `Badge variant="outline"`, `role="status"`, slot `truncate`.
- **State:** rendered as `bg-<token>/15 text-<token> border-<token>/25–30`. Two families: `status` (success/warning/error/unknown) and `http` (GET→blue, POST→violet, PUT/PATCH→amber, DELETE→rose).
- **Base badge variants** (non-status): `default` (primary), `secondary`, `destructive`, `warning` (`amber-500/15 text-amber-600 dark:text-amber-400`), `outline` (text-only, hover bg-accent).
- **Voice:** chips carrying a **technical token** — HTTP method, status code, duration — set their slot in mono (`font-mono tabular-nums`); chips carrying a **word** — "Admin", "Viewer", "Success" — keep the sans voice. A word label in mono reads as a code value and is a defect.

### Cards / Containers
Flat raised slates, hairline-edged, tonally stepped above the void. There is no `Card` component in this codebase — the recipe is a class string each surface repeats, so it is stated once, here.
- **Recipe:** `rounded-xl border border-border/60 bg-card p-5` — `p-5 sm:p-6` when the panel is a page body, `px-4 py-3` for a compact row group.
- **Corner:** `rounded-xl` (16px).
- **Background:** `bg-card` (`--card`), one step lighter than `--background`.
- **Shadow:** none at rest. The 1px `border-border/60` does the elevation work; the `.brand-panel` utility's `shadow-sm` is a separate utility, not this recipe. Hover never adds a shadow.
- **Hover:** the border tightens (`hover:border-border/80`). Never a shadow lift.

### Inputs / Fields
A lit-edge control with the strongest focus signal in the system.
- **Style:** `h-10 rounded-md border border-input/80 bg-background/75 px-3.5 py-2 text-base md:text-sm`, `shadow-[inset_0_1px_0_hsl(var(--background)/0.9)]` (the inset top highlight), `backdrop-blur-sm`, `transition-[color,box-shadow,border-color,background-color]`. Dark mode lifts the fill to `dark:bg-input/40` so the field reads as a recess in the slate rather than a hole in it; small screens grow it to `h-11`.
- **Focus:** `focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]` — a 3px primary ring, the strongest focus signal in the system.
- **Invalid:** `aria-invalid:border-destructive` + `ring-destructive/20`.
- **Placeholder:** `text-muted-foreground` (tuned for AA contrast, not lightened).
- **Auth input (login/setup):** the standard `Input` at `h-11` — same fill, same inset sheen — or `h-12 text-center font-mono text-lg tracking-[0.35em]` for the MFA code field. `.auth-input` in `main.css` changes only the **state** signals, and it is declared **outside `@layer components` on purpose**: `Input.vue` sets the same properties as utilities, and a utility layer outranks every earlier layer no matter how specific the selector is, so an in-layer version never reached the screen. What it owns: focus (`border-color` → `primary/0.65`, plus a 1px `primary/0.55` line and a 2px `primary/0.12` glow, with the `inset 0 1px 0 background/0.9` sheen restated so it survives), hover (`primary/0.28`, guarded with `:not(:disabled):not(:focus-visible):not([aria-invalid="true"])`), and invalid-while-focused (`destructive` border with the same glow shape). Every property the block overrides is restated inside it — hand one back to the utilities and the red error border silently disappears on all six auth screens. Never patch it with `!important`.

### Auth Screen — "The Vestibule" (signature surface)
One centered column, no card, one white action. Every screen before the console opens renders through `AuthShell`: login, first-run setup, force-change-password, invite acceptance, OAuth completion, email verification.
- **Shell:** `min-h-dvh` column, `px-4 py-8 sm:py-10`; block `max-w-sm` centered; the `.auth-enter` entrance (500ms `cubic-bezier(0.16, 1, 0.3, 1)`, a 14px rise plus fade, from an already-visible default).
- **Atmosphere:** `.login-bg` — a 34px `primary/0.09` grid masked from `50% 20%` (35%) out to 90%, held at `0.45` opacity in dark mode and `0.32` in light (the identical stroke reads heavier once it is drawn in near-black), plus a 620px radial `primary/0.12 → action-blue/0.06 → transparent` at `top: 18%` with `blur(8px)`. Both sit behind the form and take no pointer events. Note this is a **different, tighter** grid than the 42px page atmosphere.
- **Header — identity lockup:** the 48px `.auth-mark` (`rounded-xl`, `ring-1 ring-primary/25`, a 28px logo) sits **beside** the product name (`brand-heading text-sm text-foreground`, `gap-3`), so mark and name read as one lockup. The screen title follows as `.brand-heading mt-6 text-xl sm:text-2xl`, then an optional `mt-2 max-w-sm text-sm text-muted-foreground` description. The product name is never a stacked line above the title: a small label in that position is an eyebrow, and the system does not use one.
- **Fields:** `.auth-input h-11` inside `flex flex-col gap-4` groups (`gap-1.5` within a group); labels `text-sm font-medium text-foreground/90`. The field keeps the console's fill and sheen; only its state signals are the vestibule's own — hover lifts the border to `primary/0.28`, focus sharpens it to `primary/0.65` with a faint glow instead of taking the 3px ring, so the white CTA stays the loudest object on the screen. A password field carries its recovery entry inline, right-aligned in the label row (`text-xs text-muted-foreground`, `hover:text-foreground hover:underline`), a trailing `w-11` reveal button (`Eye`/`EyeOff`, `aria-pressed`, `aria-label` naming the action), and a `text-xs text-muted-foreground` Caps Lock note while the hardware key is down.
- **Credential order:** the passkey action leads — **primary** `h-11 w-full gap-2` with a `Fingerprint` `size-4` glyph on a browser that has proven a passkey, **outline** otherwise — then the hairline divider, then the **whole** password path (identifier, password, submit). The divider opens one branch and never splits the identifier off above it. The password submit is the **outline** twin when the passkey button is primary and **primary** otherwise; OIDC providers follow as `variant="outline" h-11 w-full gap-2` rows. The credential the operator is most likely to hold sits highest, and the white button follows it — never two whites on one screen.
- **Dividers:** `h-px flex-1 bg-border` on either side of a `text-xs text-muted-foreground` label — a hairline with a word in it, never a bordered band.
- **Utility row:** theme and language sit in a footer under the form (`mt-10`), held apart by space and a single `h-4 w-px` hairline — no rule across the column, no card. Both use the compact variant of their selector: `ThemeSelector compact` is icon-only (`Sun`/`Moon`/`Monitor`, `size-9`, `title` plus an `sr-only` label, active item on a `bg-muted` step) and `LanguageSelector compact` drops its border and height (`h-9`, borderless trigger). Both grow to 44px under `md`. The labelled forms stay in Account → Preferences.
- **MFA challenge:** one `h-12` code field (`required` is semantic only — the form is `novalidate`), plus text-link actions at `h-8 px-0 text-xs text-muted-foreground` — the only `link` buttons in the vestibule. "Use a different account" and "go back" both return to the same column, never to a new screen.
- **Errors and empty fields:** the server's verdict is a destructive `Alert` above the form (`mb-4`, `CircleAlert` glyph, `animate-in fade-in slide-in-from-top-2 duration-300`), and the session-ended notice takes the same shape with `Info`. A field the operator left empty is not an alert: it gets `aria-invalid`, focus, and a `text-xs text-destructive` hint beneath it, id-referenced from `aria-describedby`. Every auth form is `novalidate`, so the browser's own bubbles never mix with these. Editing a field retires both — a stale verdict never sits under corrected text.
- **Loading copy:** a submit in flight names its action — "Signing in…", "Verifying…", "Waiting for passkey…" — never a generic "Loading…".
- **Motion:** the block entrance is the screen's only authored moment; the error alert is the one nested arrival. Everything else is a spinner, and the global `prefers-reduced-motion` rule neutralizes both.

### Navigation (`NavSidebar`)
A recessed rail; items are quiet at rest and marked by a filled tonal step when active.
- **Shell:** `--sidebar-width` 16rem / icon-collapsed 3rem / mobile drawer 18rem, `bg-sidebar`, `border-r border-sidebar-border` (left side), fixed `inset-y-0`, width transition 200ms ease-linear. The inner shell is a `role="complementary"` landmark so the brand header and account footer (which sit outside the inner `<nav>`) stay reachable by landmark navigation.
- **Items:** shadcn `SidebarMenuButton` — `rounded-md p-2 text-sm`, `h-8` (`max-md:h-11` for touch), `SidebarMenu` `gap-0.5`, `transition-colors`. Active: `data-[active=true]:bg-sidebar-accent` (`#22252B` in dark) + `text-sidebar-accent-foreground` + `font-medium`, and the link itself is `aria-current="page"` — the tonal step plus the label carries the state, never a stripe. Hover uses the same step. Collapsed state shows the label as a tooltip.
- **Brand header / account footer:** the version chip is a mono `text-[10px]` badge on `border-sidebar-border/60`, with a `bg-status-warning` dot plus `sr-only` text when an update is available. The footer is the account menu: a 32px `rounded-md bg-muted` initials tile + `text-xs font-semibold` name + `text-[11px]` role, opening an upward `DropdownMenu` (`w-56`, `slide-in-from-bottom-2 150ms`) that carries **Account**, **Preferences** and **Sign-in & security** as its first group, then logout as `text-destructive focus:bg-destructive/10`. The account workspace lives here, not in the nav sections — it is identity, not configuration.
- **Section labels:** `text-[11px] font-semibold tracking-[0.14em] uppercase text-sidebar-foreground/70` — the label rail at its widest tracking, used **only** for the nav section groups (Overview / Catalog for viewers / Playground / Access / Configuration). Access covers who may call the proxy (own API keys for everyone; Team and Instance sign-in for admins); Configuration is the admin-only instance surface (Providers, Models, MCP Servers, Settings). No group carries a single item whose name repeats the label.

### Personal-Settings Panels (Account)
Read top-to-bottom, not scanned: one centered `max-w-4xl` column under the Account shell's Profile / Preferences / Sign-in & security tabs, made of stacked hairline panels, one topic each, ordered by task frequency and factor strength.
- **Panel:** the standard panel recipe (§Cards / Containers), one topic per panel, ordered by task frequency and factor strength; anchor targets carry `scroll-mt-6`. Heading `font-medium` (the identity name is `text-lg font-semibold`), optional status/count badge beside it, description `mt-1 text-xs text-muted-foreground`, muted topic icon at the trailing edge, body `mt-4`. The header tab strip is the only navigation between the identity, preference and credential halves; do not add a second in-page menu or a pair of cross-links.
- **Instance sign-in:** the admin-only `/config/console-auth` page keeps the same panel recipe in a centered `max-w-3xl` form column with a sticky save bar (§Layout → Instance sign-in) — instance policy stays visually and structurally separate from personal credentials. SMTP details appear only when delivery is enabled, saved OIDC providers start collapsed, and the form provides field-level errors, discard and dirty-leave protection.
- **Rows:** definition rows `flex flex-wrap items-center justify-between gap-3 py-3` inside a `divide-y divide-border/50` `<dl>`; list rows `rounded-lg border border-border/50 px-3 py-2.5`; empty states reuse that hairline row with a `bg-muted/15` step and centered muted copy. Every row control clears `min-h-11` on touch.
- **Identity tile:** a 56px `rounded-xl border border-border/50 bg-muted` initials tile, `text-lg font-semibold text-muted-foreground`, `aria-hidden`, beside a `truncate text-lg font-semibold` username and a `truncate text-xs text-muted-foreground` email.
- **Values:** identifiers, dates, and counts in mono (`.text-data`, `text-xs font-mono`); prose in Manrope. A username or ID in the display face is a defect.

### Spec-Sheet Cell (`MetricCell`, signature pattern)
The log-details metric grid is the densest thing in the product, and its cell is reused verbatim instead of re-declared per view.
- **Cell:** `bg-card p-3 sm:p-4 flex flex-col gap-1.5 sm:gap-2 min-w-0`. The parent grid owns the hairline dividers; cells never draw their own.
- **Label rail:** `text-[11px] font-semibold uppercase tracking-wider text-muted-foreground`, with a trailing icon slot.
- **Value:** `.text-data` inline; `.text-metric` when the cell is the hero of its row. Semantic color (`.metric-blue`, `.metric-amber`, `.metric-success`, `.metric-rose`) is earned only when the value *is* a category, never to decorate a plain count.

### Segmented Tab Strip (Settings)
Two mutually-exclusive views of one page, chosen in place rather than by route.
- **Shell:** `flex items-center gap-0 bg-muted/60 p-[3px] rounded-lg border border-border/55` — 36px tall, so it matches the `sm` ghost buttons sitting beside it in the header, `role="group"` with the page name as its accessible name.
- **Segment:** `px-3 py-1.5 text-xs rounded-md border transition-all duration-200` (`max-md:min-h-11 max-md:px-4` for touch, which lifts the strip to 44px). Both states carry a border — active `bg-accent text-foreground border-border/70 shadow-xs`, inactive `border-transparent text-muted-foreground hover:bg-accent/40 hover:text-foreground` — so the label never shifts by a pixel when the selection moves.
- **State:** `aria-pressed` on each segment. The active segment is a raised slate: deeper track, tonal step, hairline edge, drop shadow. It must read as lifted, not as a hole; `bg-background` on the pill is a defect because on this page `--background` *is* the page, which renders the selected segment identical to the surface behind the track.
- **No 3:1 fill, on purpose:** on the `#090A0B` page no fill in this palette reaches 3:1 against a track (`--accent` tops out around 1.3:1, and 3:1 would need a `rgb(~98)` gray). The state therefore leans on three small cues together — the tonal step, the 1px edge, the drop shadow — plus the label moving from `text-muted-foreground` (6.7:1) to `text-foreground` (~17:1), which is what actually carries it. Do not "fix" the missing 3:1 by lightening the pill to a mid-gray: that promotes a header tab to primary-CTA weight.

### Role-Switched Routes
One route may render different views by role when both roles work on the same data: `/models` renders the management UI for admins and the read-only catalog for viewers. Rules for this pattern:
- The role switch must be **synchronous** (derived from the auth store's JWT, never async) so there is no flash of the wrong view.
- Nav placement and command-palette grouping follow **each role's mental model** (Catalog for viewers, Configuration for admins) while pointing at the same route; legacy URLs redirect by route name.
- Keep this pattern rare — admin tools otherwise live under `/config/*` with an `adminOnly` route guard.

### Color-Tinted Icon Container (signature)
`.icon-container` (and its `.icon-container-sm` page-header size, `h-8 w-8 rounded-md`): `p-2.5 rounded-xl min-h-11 min-w-11`, `bg-linear-to-br from-primary/20 via-primary/10 to-action-blue/12`, `ring-1 ring-primary/30`, `inset 0 1px 0 primary/0.26`. The signature treatment for the icon beside a page title or section heading — a flat, tinted, lightly-ringed tile with a top sheen.

### Range Slider (signature control)
Used by Smart Routing mode-weight sliders. `appearance-none` + `.range-thumb`; track `0.5rem rounded-full bg-muted`, thumb `1rem rounded-full border-2 bg-background bg-primary` with a `primary/40` glow; hover `scale(1.12)`, active `scale(0.96)`, focus `0 0 0 4px ring`. Reduced-motion removes the thumb transform.

### Toasts (vue-sonner)
Themed via the same tokens; sit on `--popover`, hairline `--border`, `rounded-lg`, with the semantic status tints for success/warning/error variants. No celebratory motion.

### Step-Up Dialog
A reauthentication gate that interrupts without leaving the current page — the one dialog the console opens on its own.
- **Shell:** the standard `Dialog` (`bg-card`, hairline border, `rounded-xl`, scrim `--overlay`), so it reads as a panel lifted off the console rather than a new screen.
- **Body:** `flex flex-col gap-3`, one `Label` + `Input` group at `gap-1.5`, error text `text-xs text-destructive` directly under the field, `DialogFooter` with `gap-2`.
- **Actions:** the passkey path is the outline button (`gap-2`, `Fingerprint` `h-4 w-4`); the password path is the primary submit. Both disable together while either is in flight, with a `Loader2` spinner in the primary.
- **Rule:** a step-up never navigates. Closing it cancels the pending action and leaves the operator exactly where they were.

### Recovery Surface (render crash)
`AppErrorState`, mounted by `App.vue`'s error boundary when a descendant throws during render, in a lifecycle hook or in an event handler. The console chrome goes down with the failing tree, so this is a full-height centered `max-w-md` column with one white action — the same "one screen, one action" rule as the vestibule.
- **Header:** 48px `rounded-xl` tile on `bg-destructive/10` with `ring-1 ring-destructive/25` and a `CircleAlert size-6 text-destructive`; then `.brand-heading` title (`errors.somethingWrong`), one `text-sm text-muted-foreground` sentence saying the screen failed to render, and the raw `error.message` in `font-mono text-xs text-foreground/70` — the single detail an operator can quote in a bug report.
- **Details:** a `<details>` on `bg-muted/30` with `border-border/60`; the summary row (`text-xs`) carries a ghost `Copy` button, and the `pre` (`max-h-48`, `text-[11px]` mono) opens with `<ComponentName · hook>` before 12 stack frames.
- **Actions:** primary `Back to dashboard` (`router.replace("/")`, clearing the boundary) and outline `Reload`. The raw message stays visible; the stack stays behind the disclosure so the screen still reads as an explanation rather than a crash dump.
- **Rule:** the boundary clears itself whenever the route changes. A redirect (401 → `/login`, forced password change) must never be hidden behind a crash screen belonging to the route the operator already left.

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
- **Do** use the `config-page-reveal` entrance on every page and the `stagger-fast` / `row-stagger` cascades on first render — and let the global `prefers-reduced-motion` rule neutralize them.
- **Do** keep display tracking at −0.02em (floor −0.04em), display weight 600 in-app, and page titles at `text-xl sm:text-2xl`.
- **Do** hit WCAG AA on both themes: tune `--muted-foreground` for readability, never lighten muted text "for elegance," and bump dark-mode body line-height to 1.55–1.6.
- **Do** keep touch targets ≥44px (`min-h-11 min-w-11`) on nav, mobile controls, and the `max-sm` floor the button and input components apply to themselves.
- **Do** render JSON and code with the syntax palette on code surfaces only — subtle chroma that reads as structure, never a rainbow theme.
- **Do** build every pre-console screen through `AuthShell` and leave it containerless: one centered `max-w-sm` column, a 48px `auth-mark` locked up with the product name, 44px glass fields, and exactly one white CTA — the passkey button when the account has one, the password submit otherwise.
- **Do** pick the structural grammar by task — flush console for management tables, full-width configuration headers with centered form columns for Settings and instance policy, stacked hairline panels in a centered column for personal settings, containerless for auth — and never mix two on one screen.
- **Do** use the label rail *inside* a panel to name the value beneath it, and let a page heading stand without one.

### Don't:
- **Don't** make it playful/consumer — no cute illustrations, mascots, gamification, or celebratory confetti. This is infrastructure. *(PRODUCT.md anti-reference.)*
- **Don't** ship boring/outdated admin panels — no dense gray-on-gray tables, no circa-2008 enterprise chrome, no cluttered "everything on one screen" dashboards. *(PRODUCT.md anti-reference.)*
- **Don't** use loud color accents — no saturated SaaS gradients, no neon, no rainbow status badges. Color is a subtle low-saturation tinting system for semantic differentiation, never decoration. *(PRODUCT.md anti-reference.)*
- **Don't** ship the generic 2026 AI-tool aesthetic — no cream/sand warm-neutral body backgrounds, no gradient-text headlines, no identical icon+heading+text card grids, no big-number hero-metric template. This is monochrome editorial, not warm SaaS. *(PRODUCT.md anti-reference.)*
- **Don't** pair a 1px border with a 16px+ blur drop shadow on the same element (the ghost-card pattern). Pick a tonal step or `shadow-xs`, never a wide decorative shadow.
- **Don't** over-round — cards and panels cap at `rounded-xl` (16px); save full-pill for tags and status chips, `rounded-lg` (12px) for nav items. No 24/28/32/40px card radii.
- **Don't** use `border-left`/`border-right` greater than 1px as a colored accent stripe on cards, list items, or alerts. State is a tonal step, never a stripe.
- **Don't** use glassmorphism decoratively. `backdrop-blur` appears only on `outline` buttons and inputs — two constrained sets, never a default card treatment. (`auth-mark` carries a gradient and a shadow, not a blur.)
- **Don't** use `background-clip: text` with a gradient. Display emphasis comes from weight and size, in a single solid color.
- **Don't** put a tiny uppercase tracked eyebrow above a page or a page section. The label rail belongs inside a panel, above the value it names; a kicker above a heading is the pattern this system does not use.
- **Don't** set a metric or number in Manrope, or introduce a fourth type family. Space Grotesk + Manrope + IBM Plex Mono is the whole system.
- **Don't** introduce true-gray (`0 0%`) or warm-neutral (`hue 30–100`) tokens. Every neutral carries the 220° cool undertone.
- **Don't** reuse the JSON/code syntax tokens outside code surfaces — no rainbow syntax themes, no syntax-colored badges or chart series.
- **Don't** animate layout properties. Motion is `transform`/`opacity` only, exponential ease-out (`cubic-bezier(0.16, 1, 0.3, 1)`), 150–500ms, with a reduced-motion alternative on every animation.
- **Don't** wrap an auth form in a card, panel, or bordered container. The vestibule has no chrome — a card there turns a threshold into an object on a page.
- **Don't** wrap a flush console table in an outer card, and don't give a stacked settings panel the console's sticky header bar. Each grammar is complete on its own.
