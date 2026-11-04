import type { Directive } from "vue";

const TARGET_SELECTOR = '[aria-live], [data-slot$="overlay"]';
const SKIP_TAGS = new Set(["SCRIPT", "STYLE", "LINK", "TEMPLATE", "NOSCRIPT", "META", "TITLE"]);

/**
 * Locks are ref-counted per element, per document. A modal that opens while
 * another one already holds the page inert must not snapshot "already inert"
 * and later re-apply it: closing the outer modal first would drop the lock the
 * inner one still needs, and closing the inner one afterwards would leave the
 * page permanently inert.
 */
interface DocumentLocks {
  /** Previous `inert` state, captured by the first lock that touched an element. */
  origin: Map<HTMLElement, boolean>;
  /** How many active locks need each element inert. */
  refs: Map<HTMLElement, number>;
}

const documents = new WeakMap<Document, DocumentLocks>();

function locksFor(document: Document): DocumentLocks {
  let state = documents.get(document);
  if (!state) {
    state = { origin: new Map(), refs: new Map() };
    documents.set(document, state);
  }
  return state;
}

/**
 * Mark every background sibling inert while a modal owns the page.
 *
 * The dialog's own subtree, live regions and scrims stay interactive so a
 * toast can still announce and an outside click can still dismiss. Nodes are
 * changed only once, on mount: portals opened later by controls inside the
 * dialog (selects, popovers) must not inherit an inert ancestor.
 */
export function inertBackground(content: HTMLElement): () => void {
  const document = content.ownerDocument;
  const body = document.body;
  if (!body) return () => {};

  const state = locksFor(document);
  const owned = new Set<HTMLElement>();

  const targets = new Set<Element>([content, ...body.querySelectorAll(TARGET_SELECTOR)]);
  const kept = new Set<Element>();
  for (const target of targets) {
    let node: Element | null = target;
    while (node) {
      kept.add(node);
      node = node.parentElement;
    }
  }

  const lock = (element: HTMLElement) => {
    if (owned.has(element)) return;
    if (!state.refs.has(element)) state.origin.set(element, element.hasAttribute("inert"));
    state.refs.set(element, (state.refs.get(element) ?? 0) + 1);
    element.setAttribute("inert", "");
    owned.add(element);
  };

  const visit = (parent: Element): void => {
    if (targets.has(parent)) return;
    for (const child of Array.from(parent.children)) {
      if (kept.has(child)) {
        visit(child);
        continue;
      }
      if (!(child instanceof HTMLElement) || SKIP_TAGS.has(child.tagName)) continue;
      lock(child);
    }
  };

  visit(body);

  let released = false;
  return () => {
    if (released) return;
    released = true;
    for (const element of owned) {
      const remaining = (state.refs.get(element) ?? 1) - 1;
      if (remaining > 0) {
        state.refs.set(element, remaining);
        continue;
      }
      state.refs.delete(element);
      if (state.origin.get(element)) element.setAttribute("inert", "");
      else element.removeAttribute("inert");
      state.origin.delete(element);
    }
    owned.clear();
  };
}

const restorers = new WeakMap<HTMLElement, () => void>();

export const vInertBackground: Directive<HTMLElement> = {
  mounted(element) {
    restorers.set(element, inertBackground(element));
  },
  unmounted(element) {
    restorers.get(element)?.();
    restorers.delete(element);
  },
};
