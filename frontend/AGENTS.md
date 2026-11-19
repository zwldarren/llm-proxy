# Frontend - Agent Instructions

## Commands

- **Install**: `bun install`
- **Dev server**: `bun dev` (Vite hot-reload)
- **Build**: `bun run build` (type-check + minify)
- **Type check only**: `bun run type-check`
- **Lint**: `bun run lint` (ESLint + auto-fix)
- **Lint check**: `bun run lint:check` (ESLint without auto-fix)
- **Format**: `bun run format` (Prettier formatting)
- **Format check**: `bun run format:check` (Prettier check without writing)
- **Install shadcn-vue component**: `bunx shadcn-vue@latest add ComponentName`

## Tech Stack

- **Framework**: Vue 3 + TypeScript
- **UI**: shadcn-vue (Radix Vue + Reka UI) + tailwindcss v4
- **Build tool**: Vite (with Rolldown)
- **Package manager**: Bun
- **Linter**: ESLint
- **Formatter**: Prettier
- **State management**: Pinia
- **Routing**: Vue Router

## Code Style Guidelines

- **Formatting**: Prettier with tabs (configured in .prettierrc.json)
- **Linting**: ESLint with Vue/TypeScript support (configured in eslint.config.js)
- **Imports**: Use `@/` alias for src/\* imports
- **TypeScript**: Enable strict mode, use Vue 3 composition API
- **Vue components**: Use `<script setup lang="ts">` syntax
- **Styling**: Tailwind CSS v4 with Utility-first approach
- **Error handling**: Use Vue's error boundaries and try/catch for async operations
- **Naming**: PascalCase for components, camelCase for variables/functions

## Tests

- **Run**: `bun run test` (Vitest + happy-dom)
- `src/test/setup.ts` installs happy-dom `Storage` instances over `globalThis.localStorage`
  and `sessionStorage` for every test file. Node's own Web Storage globals shadow the
  environment's (`localStorage` resolves to `undefined` without `--localstorage-file`),
  and Vitest skips window keys that already exist on `globalThis`, so without the setup
  file every suite that imports `src/i18n/index.ts` fails at import with
  `Cannot read properties of undefined (reading 'getItem')`.
- Views mount their children through `defineAsyncComponent`, so mounting one in a test
  starts module transforms that can still be running when the test ends. CI fails the
  whole run for that, even when every test passed:
  `EnvironmentTeardownError: Cannot load '…' … after the environment was torn down`.
  Mock the lazy children (`vi.mock` with `__esModule: true`, otherwise Vue keeps the
  mocked namespace as the component) so a view test only loads what the view imports.

## Icons

Model and provider logos are not hand-mapped. `utils/icons.ts` matches the identifier
against the whole `@lobehub/icons` catalog, so a new model from a known brand and a niche
provider type both resolve without editing a table:

```ts
import { modelIcon, providerIcon } from "@/utils/icons";

modelIcon({ name: model.name, provider: names?.[0], iconUrl: model.icon_url });
providerIcon({ type: p.type, name: p.name, iconUrl: p.icon_url, brand: typeInfo });
```

- Both return `{ url, mono }` or `null`. Resolve **once** and pass the result to
  `BrandMark`; never derive the URL and the mono flag separately — that is how a colour
  mark used to end up inverted.
- Rendering goes through `components/common/BrandMark.vue`, which owns the frame, the
  `.icon-mono` theming, the load-failure latch and the fallback. Without a slot it falls
  back to the name's initial; pass a lucide glyph in the default slot to keep a caller's
  own chip (dense log rows).
- Resolution order: configured `icon_url` → model brand → provider brand → initial.
  For providers the adapter's declared `LOBE_ICON_ID`/`LOBE_ICON_VARIANT` wins over the
  catalog match.
- `utils/lobeBrands.generated.ts` is generated: run `bun run icons:sync` to pick up new
  brands (it bumps the pinned CDN version too). Never edit it by hand.

## Development Guidelines

- Use shadcn-vue for component library (Radix Vue + Reka UI)
- Add components with: `npx shadcn-vue@latest add ComponentName`
- Components are configured via components.json, themes via Tailwind CSS
- Always run `bun run lint && bun run format:check` before committing changes
- All frontend user-facing text must support internationalization
- All code comments must be written in English, regardless of target locale

## DESIGN CONTEXT

### Users

Individual hobbyists and developers building AI applications. Technical users who integrate multiple LLM providers, monitor usage/costs, configure providers and models. They value efficiency and clarity over hand-holding.

### Brand Personality

**Technical Sophistication, Reliable, Modern**

- **Voice**: Direct, precise, professional without being corporate
- **Tone**: Confident but not arrogant; helpful without being verbose
- **Emotional Goals**: Users should feel in control, that the system is trustworthy, and that they can accomplish tasks quickly without confusion

### Aesthetic Direction

**Visual Tone**: Monochrome Editorial Theme - minimalist, black-first (light) / deep cool-neutral dark (dark), with subtle atmospheric gradients and fine grid patterns. Clean typography using Space Grotesk (display), Manrope (body), IBM Plex Mono (code), Noto Sans SC (CJK fallback).

**Anti-References**:

- NOT playful/consumer apps (no cute illustrations, gamification)
- NOT boring/outdated admin panels (no dense tables, gray-on-gray)
- NO loud color accents - use subtle tinting with low saturation

### Design Principles

1. **Clarity Over Cleverness** - Every element has a clear purpose. Labels are explicit, icons are standard, hierarchy is obvious.
2. **Confidence Through Feedback** - Actions have visible, immediate feedback. Users never wonder "did that work?"
3. **Technical Sophistication** - Embrace the technical nature. Show API details, technical metrics clearly.
4. **Dark-First Design** - Primary experience is dark mode. Light mode is well-supported but secondary.
5. **Respectful of Time** - Power users navigate quickly. Keyboard shortcuts, collapsible sections, efficient tables.

### Typography

- **Display/Headlines**: Space Grotesk (500/600/700) - geometric, technical character for headings
- **Body**: Manrope + Noto Sans SC - clean, readable sans-serif for UI text
- **Code/Monospace**: IBM Plex Mono - excellent legibility for code and data
- **Type Scale**: 1.25 ratio modular scale (xs/sm/base/lg/xl/2xl/3xl)
- **Numeric Data**: Use `.text-data` class for tabular-nums alignment

### Color Palette

**Brand Direction**: Monochrome Editorial - black/white with cool-neutral grays

**Light Mode**:

- **Primary**: `hsl(220 8% 5%)` - Near black, main text and CTAs
- **Background**: `hsl(220 3% 98.5%)` - Crisp off-white
- **Muted**: `hsl(220 4% 94%)` - Subtle backgrounds, borders
- **Action colors** (subtle tinting): violet, blue, amber, rose - all low saturation

**Dark Mode**:

- **Primary**: `hsl(0 0% 98%)` - Crisp white
- **Background**: `hsl(220 12% 4%)` - Deep cool-neutral dark
- **Muted**: `hsl(220 9% 15%)` - Subtle backgrounds, borders
- **Action colors** (subtle tinting): violet, blue, amber, rose - low saturation

**HTTP Methods**: GET (blue), POST (violet), PUT/PATCH (amber), DELETE (rose) - all subtle

### Accessibility

- **Motion**: Respect `prefers-reduced-motion`
- **Contrast**: WCAG AA minimum
- **Focus**: Visible focus rings on all interactive elements
- **i18n**: All UI text internationalized (English primary, Chinese secondary)
