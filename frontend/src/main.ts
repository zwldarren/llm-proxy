import "./assets/main.css";

import "@fontsource/manrope/400.css";
import "@fontsource/manrope/500.css";
import "@fontsource/manrope/600.css";
import "@fontsource/manrope/700.css";
import "@fontsource/space-grotesk/500.css";
import "@fontsource/space-grotesk/600.css";
import "@fontsource/space-grotesk/700.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "@fontsource/ibm-plex-mono/600.css";

import { createPinia } from "pinia";
import { createApp } from "vue";

import App from "./App.vue";
import { initializeTheme } from "./composables/useTheme";
import i18n, { initializeLocale } from "./i18n";
import router from "./router";
import { useAuthStore } from "./stores/auth";

initializeTheme();

// Chinese ships as a separate chunk (see i18n/index.ts); resolve the persisted
// locale before mounting so a Chinese session never flashes English. Resolves
// immediately for the default locale.
await initializeLocale();

const app = createApp(App);

app.use(createPinia());
app.use(i18n);

// Mount without waiting on the network. The router's beforeEach guard awaits
// the memoized setup-status check itself, so the first navigation still sees
// the resolved state — but the shell, the theme and the app CSS are no longer
// gated behind an API round-trip.
const authStore = useAuthStore();
void authStore.ensureSetupStatus();

app.use(router);
app.mount("#app");
