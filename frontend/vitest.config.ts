import path from "node:path";
import vue from "@vitejs/plugin-vue";
import { defineConfig } from "vitest/config";

// Force the test mode before Vitest forks its workers. @vue/test-utils is loaded
// through its `node` (CJS) export and requires `vue`, whose export map picks the
// production build from NODE_ENV. That build drops the devtools hook test-utils
// records emitted events with, so `wrapper.emitted()` silently returns undefined
// and every emit assertion fails. Setting it in `test.env` is too late (the
// workers already resolved vue) and a Vite alias cannot reach the require.
process.env.NODE_ENV = "test";

export default defineConfig({
	plugins: [vue()],
	resolve: {
		alias: [{ find: "@", replacement: path.resolve(__dirname, "./src") }],
	},
	test: {
		environment: "happy-dom",
	},
});
