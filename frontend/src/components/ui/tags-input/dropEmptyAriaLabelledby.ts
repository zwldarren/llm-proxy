import type { Directive } from "vue";

/**
 * Reka renders the token before its text span has minted an id, so the item and
 * its remove button ship `aria-labelledby=""`. Remove the empty reference; the
 * remove button gets its name from an explicit aria-label instead.
 */
function dropEmptyReference(element: HTMLElement): void {
  if (element.getAttribute("aria-labelledby") === "") {
    element.removeAttribute("aria-labelledby");
  }
}

export const vDropEmptyAriaLabelledby: Directive<HTMLElement> = {
  mounted: dropEmptyReference,
  updated: dropEmptyReference,
};
