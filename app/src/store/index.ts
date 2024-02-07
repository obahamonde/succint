import { acceptHMRUpdate, defineStore } from "pinia";
import type { User, Notification, Message } from "~/types";

export const useStore = defineStore("state", () => {
  const state = reactive({
    notifications: [] as Notification[],
    user: null as User | null,
    messages: [] as Message[]
  });

  return {
    state,
  };
});
if (import.meta.hot)
  import.meta.hot.accept(acceptHMRUpdate(useStore as any, import.meta.hot));