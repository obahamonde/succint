import path from "node:path";
import { defineConfig } from "vite";
import Vue from "@vitejs/plugin-vue";
import Components from "unplugin-vue-components/vite";
import AutoImport from "unplugin-auto-import/vite";
import UnoCSS from "unocss/vite";
import VueMacros from "unplugin-vue-macros/vite";
import VueRouter from "unplugin-vue-router/vite";
export default defineConfig({
  base: "/",
  resolve: {
    alias: {
      "~/": `${path.resolve(__dirname, "src")}/`,
    },
  },
  server: {
    proxy: {
      "/api": {
        target: "https://api.oscarbahamonde.com/api",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
  build: {
    outDir: "../static",
    emptyOutDir: true,
  },
  plugins: [
    VueMacros({
      defineOptions: false,
      defineModels: false,
      plugins: {
        vue: Vue({
          script: {
            propsDestructure: true,
            defineModel: true,
          },
        }),
      },
    }),
    VueRouter(),
    AutoImport({
      imports: ["vue", "@vueuse/core", "vue-router", "pinia"],
      dts: "src/auto-imports.d.ts",
      dirs: ["src/composables", "src/types", "src/store"],
      vueTemplate: true,
    }),
    Components({
      dts: "src/components.d.ts",
    }),
    UnoCSS(),
  ],
});
