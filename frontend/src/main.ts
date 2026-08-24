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
import "@fontsource/noto-sans-sc/400.css";
import "@fontsource/noto-sans-sc/500.css";
import "@fontsource/noto-sans-sc/600.css";
import "@fontsource/noto-sans-sc/700.css";

import { createPinia } from "pinia";
import { createApp } from "vue";

import App from "./App.vue";
import { initializeTheme } from "./composables/useTheme";
import i18n from "./i18n";
import router from "./router";
import { useAuthStore } from "./stores/auth";

initializeTheme();

const app = createApp(App);

app.use(createPinia());
app.use(i18n);

// Check first-run setup status BEFORE installing the router. Vue Router starts
// its initial navigation synchronously inside install() (app.use(router)), so
// the beforeEach guard would run with needsSetup still null (tri-state default)
// and wrongly allow navigating to /setup when an admin already exists. Resolving
// needsSetup first ensures the guard sees the correct state on first navigation.
const authStore = useAuthStore();
try {
  await authStore.checkSetupStatus();
} catch {
  // If the backend is unreachable, leave needsSetup as false and let the
  // normal login flow surface any connection issues.
}

app.use(router);
app.mount("#app");
