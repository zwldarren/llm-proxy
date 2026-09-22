import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import vue from "@vitejs/plugin-vue";
import Components from "unplugin-vue-components/vite";
import { defineConfig } from "vite";
import type { Plugin } from "vite";
import vueDevTools from "vite-plugin-vue-devtools";
import tsconfigPaths from "vite-tsconfig-paths";

/**
 * Drop the legacy `.woff` source from bundled @fontsource CSS. Every browser
 * that can run this app supports woff2, so the fallback only doubles the
 * emitted font payload with files nothing requests.
 */
function dropLegacyWoff(): Plugin {
	return {
		name: "drop-legacy-woff",
		enforce: "pre",
		transform(code, id) {
			if (!id.includes("@fontsource") || !id.endsWith(".css")) return null;
			const stripped = code.replace(/, url\([^)]*\.woff\) format\('woff'\)/g, "");
			return stripped === code ? null : { code: stripped, map: null };
		},
	};
}

// Backend API URL - can be set via VITE_API_BASE_URL environment variable
const apiBaseUrl = process.env.VITE_API_BASE_URL || "http://localhost:8000";

const isDev = process.env.NODE_ENV !== "production";

// https://vite.dev/config/
export default defineConfig({
	plugins: [
		vue(),
		dropLegacyWoff(),
		// Only enable Vue DevTools in development to keep production builds lean
		...(isDev ? [vueDevTools()] : []),
		Components({
			dirs: ["src/components"],
			deep: true,
			dts: "components.d.ts",
		}),
		tailwindcss(),
		tsconfigPaths(),
	],
	resolve: {
		alias: [{ find: "@", replacement: path.resolve(__dirname, "./src") }],
	},
	build: {
		// Raise warning threshold for chunk sizes (KB)
		chunkSizeWarningLimit: 500,
		rollupOptions: {
			output: {
				// Split heavy third-party libraries into dedicated chunks to improve
				// caching and reduce the initial JS payload.
				//
				// Rolldown's chunk groups also capture their modules' dependencies, so a
				// dependency that has its own group needs a higher priority than the group
				// depending on it. Without that, the charts group hoisted the Vue runtime
				// and pulled chart.js onto the critical path of every route.
				advancedChunks: {
					groups: [
						{
							name: "chunk-vue",
							test: /node_modules[\\/](@vue|vue|vue-router|pinia)[\\/]/,
							priority: 100,
						},
						{
							name: "chunk-charts",
							test: /node_modules[\\/](chart\.js|vue-chartjs)[\\/]/,
							priority: 90,
						},
						{ name: "chunk-json", test: /node_modules[\\/]vue-json-pretty[\\/]/, priority: 90 },
						{ name: "chunk-markdown", test: /node_modules[\\/]markdown-it[\\/]/, priority: 90 },
						{ name: "chunk-i18n", test: /node_modules[\\/]vue-i18n[\\/]/, priority: 90 },
						{ name: "chunk-sanitizer", test: /node_modules[\\/]dompurify[\\/]/, priority: 90 },
						{
							// @fontsource CSS must follow its importer instead of the vendor chunk:
							// the CJK faces are imported dynamically and belong on a lazy stylesheet
							// rather than the render-blocking one.
							name: (id: string) => (id.includes("@fontsource") ? null : "chunk-vendor"),
							test: /node_modules[\\/]/,
							priority: 1,
						},
					]
				},
			},
			onwarn(warning, warn) {
				// Suppress vue-i18n currentInstance warning
				if (
					warning.code === "IMPORT_IS_UNDEFINED" &&
					warning.message.includes("currentInstance")
				) {
					return;
				}
				warn(warning);
			},
		},
	},
	server: {
		host: "0.0.0.0",
		proxy: {
			"/api": {
				target: apiBaseUrl,
				changeOrigin: true,
			},
			"/v1": {
				target: apiBaseUrl,
				changeOrigin: true,
			},
		},
	},
});
