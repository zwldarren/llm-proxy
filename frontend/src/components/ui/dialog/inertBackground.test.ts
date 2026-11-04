import { afterEach, describe, expect, it } from "vitest";
import { inertBackground } from "./inertBackground";

afterEach(() => {
  document.body.innerHTML = "";
});

describe("inertBackground", () => {
  it("inerts the page while keeping the dialog, scrim and live regions usable", () => {
    document.body.innerHTML = `
      <div id="app"><button id="background-action">Background</button></div>
      <div id="overlay" data-slot="dialog-overlay"></div>
      <div id="dialog" data-slot="dialog-content"><button>Inside</button></div>
      <div id="toast" aria-live="polite">Saved</div>
    `;
    const dialog = document.getElementById("dialog")!;

    const restore = inertBackground(dialog);

    expect(document.getElementById("app")!.hasAttribute("inert")).toBe(true);
    expect(document.getElementById("overlay")!.hasAttribute("inert")).toBe(false);
    expect(dialog.hasAttribute("inert")).toBe(false);
    expect(document.getElementById("toast")!.hasAttribute("inert")).toBe(false);

    restore();
    expect(document.getElementById("app")!.hasAttribute("inert")).toBe(false);
  });

  it("preserves the outer lock when a nested dialog closes", () => {
    document.body.innerHTML = `
      <div id="app">Background</div>
      <div id="base" data-slot="dialog-content">Base</div>
      <div id="top" data-slot="dialog-content">Top</div>
    `;
    const app = document.getElementById("app")!;
    const base = document.getElementById("base")!;
    const restoreBase = inertBackground(base);
    const restoreTop = inertBackground(document.getElementById("top")!);

    expect(app.hasAttribute("inert")).toBe(true);
    expect(base.hasAttribute("inert")).toBe(true);

    restoreTop();
    expect(base.hasAttribute("inert")).toBe(false);
    expect(app.hasAttribute("inert")).toBe(true);

    restoreBase();
    expect(app.hasAttribute("inert")).toBe(false);
  });

  it("keeps the page locked when the outer modal closes before a nested one", () => {
    document.body.innerHTML = `
      <div id="app">Background</div>
      <div id="base" data-slot="dialog-content">Base</div>
      <div id="top" data-slot="dialog-content">Top</div>
    `;
    const app = document.getElementById("app")!;
    const base = document.getElementById("base")!;
    const restoreBase = inertBackground(base);
    const restoreTop = inertBackground(document.getElementById("top")!);

    // The save inside the base dialog completes while the step-up modal is
    // still open: the page must stay inert for the modal that is still up.
    restoreBase();
    expect(app.hasAttribute("inert")).toBe(true);
    expect(base.hasAttribute("inert")).toBe(true);

    restoreTop();
    expect(app.hasAttribute("inert")).toBe(false);
    expect(base.hasAttribute("inert")).toBe(false);
  });
});
