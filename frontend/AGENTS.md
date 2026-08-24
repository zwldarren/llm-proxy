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
