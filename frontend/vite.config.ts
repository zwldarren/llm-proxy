import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import vue from "@vitejs/plugin-vue";
import Components from "unplugin-vue-components/vite";
import { defineConfig } from "vite";
import vueDevTools from "vite-plugin-vue-devtools";
import tsconfigPaths from "vite-tsconfig-paths";

// Backend API URL - can be set via VITE_API_BASE_URL environment variable
const apiBaseUrl = process.env.VITE_API_BASE_URL || "http://localhost:8000";

const isDev = process.env.NODE_ENV !== "production";

// https://vite.dev/config/
export default defineConfig({
	plugins: [
		vue(),
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
				// Split heavy third-party libraries into dedicated chunks
				// to improve caching and reduce initial JS payload.
				manualChunks(id) {
					if (id.includes("node_modules")) {
						if (id.includes("chart.js") || id.includes("vue-chartjs")) {
							return "chunk-charts";
						}
						if (id.includes("vue-json-pretty")) {
							return "chunk-json";
						}
						if (id.includes("markdown-it")) {
							return "chunk-markdown";
						}
						if (id.includes("@tanstack/vue-table")) {
							return "chunk-table";
						}
						if (id.includes("vue-i18n")) {
							return "chunk-i18n";
						}
						if (id.includes("dompurify")) {
							return "chunk-sanitizer";
						}
						return "chunk-vendor";
					}
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