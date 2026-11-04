import { Storage } from "happy-dom";

/**
 * Test-environment repair, loaded before every test file.
 *
 * Node ships its own Web Storage globals: `localStorage` is a getter that
 * resolves to `undefined` unless the process was started with
 * `--localstorage-file`, and `sessionStorage` is a process-wide store that would
 * leak state between test files. Vitest copies the DOM environment's window keys
 * onto `globalThis` but skips names that already exist there, so Node's versions
 * win and `localStorage.getItem(...)` throws in every test that touches storage.
 * That includes `src/i18n/index.ts`, which reads the saved locale at import time,
 * so the failure takes down every suite that reaches it.
 *
 * Install the DOM environment's own Storage implementation instead, fresh for
 * each test file.
 */
for (const name of ["localStorage", "sessionStorage"] as const) {
  Object.defineProperty(globalThis, name, {
    value: new Storage(),
    configurable: true,
    writable: true,
  });
}
