import { createApp } from "vue";
import { createRouter, createWebHistory } from "vue-router/auto";
import { createPinia } from "pinia";
import { Icon } from "@iconify/vue";
import { createAuth0 } from "@auth0/auth0-vue";
import piniaPluginPersistedstate from "pinia-plugin-persistedstate";
import App from "./App.vue";

import "@unocss/reset/tailwind.css";
import "./styles/main.scss";
import "uno.css";
const pinia = createPinia();
pinia.use(piniaPluginPersistedstate);
const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
});
createApp(App).use(
  createAuth0({
    domain: "dev-tvhqmk7a.us.auth0.com",
    clientId: "53p0EBRRWxSYA3mSywbxhEeIlIexYWbs",
    authorizationParams: {
      redirect_uri: window.location.origin
    }
  })
).use(router).use(pinia).component("Icon", Icon).mount("#app");
