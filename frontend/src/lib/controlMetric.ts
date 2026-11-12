/**
 * The two metrics a page uses for controls outside its content: the page-header
 * action row and a toolbar band.
 *
 * Both exist because a row of related controls drifts into three heights as
 * soon as each one is styled where it is used — the Usage header had a 36px
 * picker next to a 40px date button, the Logs header a 40px refresh button
 * beside a 36px filter toggle, and the management toolbars a 44px search field
 * beside a 38px view switch. Styling the row as one metric fixes each case and
 * keeps the next one from reappearing.
 *
 * These are strings, not components: a select trigger, a popover trigger and a
 * plain button all draw from the same surface here, and `cn` resolves a metric
 * against each component's own base classes (its utilities are the last word,
 * so a conflicting base height or hover colour is replaced, not appended).
 *
 * Heights follow the pointer, not the viewport: 36/40px on a fine pointer,
 * 44px on a coarse one, which the `max-sm` rules on the primitives and the
 * `@media (pointer: coarse)` floor in `main.css` already enforce.
 */

/**
 * A chip — the surface shared by every picker and popover trigger in a header
 * (the model picker, the date range, the protocol endpoint). Same border, same
 * hover, same focus ring, so a control that opens a menu and one that opens a
 * calendar read as the same kind of object.
 *
 * Call sites add what is specific to them: mono type for an API identifier,
 * a max width for a long name, `hover:bg-transparent` for a read-only chip.
 *
 * `text-sm`, `whitespace-nowrap` and `cursor-pointer` are here rather than
 * inherited because a chip may be a bare `<button>`, which starts at the
 * document's 16px body text and the browser's arrow cursor.
 *
 * The dark hover restates the light one because the primitives below (Select
 * trigger, Ghost button) each ship their own `dark:hover:bg-*` at higher
 * specificity.
 */
export const HEADER_CHIP_CLASS =
  "h-9 cursor-pointer gap-2.5 whitespace-nowrap rounded-md border border-border/60 bg-transparent px-2.5 text-sm font-normal text-foreground shadow-none backdrop-blur-none transition-colors hover:border-border/80 hover:bg-muted/10 dark:hover:bg-muted/10 focus-visible:ring-1 focus-visible:ring-foreground focus-visible:ring-offset-0";

/**
 * The chip's square ghost sibling: an icon-only header action (refresh,
 * settings, pause). Quiet at rest, tonal when on.
 *
 * The explicit `dark:hover:` pairs restate the light hover on purpose: the
 * primitives underneath (Ghost) carry their own `dark:hover:bg-accent/50`,
 * which is both stronger and more specific than a plain `hover:` utility, so
 * without them the same button would hover in two different tints.
 */
export const HEADER_ICON_BUTTON_CLASS =
  "h-9 w-9 rounded-lg text-muted-foreground hover:bg-muted/50 hover:text-foreground dark:hover:bg-muted/50";

/**
 * The on-state for a header toggle, whichever control carries it. One tonal
 * step instead of the inverted-ink fill that used to make a switched-on filter
 * button look like a primary action.
 */
export const HEADER_ACTIVE_CLASS = "bg-muted text-foreground";
