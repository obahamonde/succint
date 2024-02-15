<script setup lang="ts">
import type { User } from "~/types";
const props = defineProps<{
  user: User;
}>();
const showInfo = ref(false);
</script>
<template>
  <section class="bottom-4 absolute m-4">
    <div
      class="col center font-mono text-xs text-accent my-4 animate-fade-in-up"
      v-if="showInfo"
    >
      <span>{{ props.user.name }}</span>
      <strong class="mx-4">{{ props.user.locale!.toUpperCase() }}</strong>
    </div>
    <img
      :src="props.user.picture"
      alt="avatar"
      class="x4 cp scale rf"
      @click="showInfo = !showInfo"
    />
  </section>
</template>
