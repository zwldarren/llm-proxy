import js from "@eslint/js";
import pluginVue from "eslint-plugin-vue";
import vueTsEslintConfig from "@vue/eslint-config-typescript";
import prettierConfig from "@vue/eslint-config-prettier";

export default [
	{
		name: "app/files-to-lint",
		files: ["**/*.{ts,mts,tsx,vue}"],
	},

	{
		name: "app/files-to-ignore",
		ignores: [
			"**/dist/**",
			"**/dist-ssr/**",
			"**/coverage/**",
			"**/node_modules/**",
			"**/components.d.ts",
			"src/components/ui/**",
		],
	},

	js.configs.recommended,
	...pluginVue.configs["flat/essential"],
	...vueTsEslintConfig(),
	prettierConfig,

	{
		name: "app/vue-rules",
		files: ["**/*.vue"],
		rules: {
			"vue/multi-word-component-names": "off",
			"vue/no-unused-vars": "warn",
			"vue/no-unused-components": "warn",
			// Disable parsing error rule since valid Vue code like :class="class" triggers false positives
			"vue/no-parsing-error": "off",
		},
	},

	{
		name: "app/typescript-rules",
		files: ["**/*.{ts,tsx,vue}"],
		rules: {
			"@typescript-eslint/no-unused-vars": [
				"warn",
				{
					argsIgnorePattern: "^_",
					varsIgnorePattern: "^_",
				},
			],
			"@typescript-eslint/no-explicit-any": "warn",
			"@typescript-eslint/no-empty-object-type": "off",
		},
	},

	{
		name: "app/test-rules",
		files: ["**/*.test.{ts,tsx}", "**/*.spec.{ts,tsx}"],
		rules: {
			"vue/multi-word-component-names": "off",
			"vue/no-reserved-component-names": "off",
		},
	},
];
