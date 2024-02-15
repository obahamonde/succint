<script setup lang="ts">
const showLanding = ref(true)
</script>
<template>
<main class="col center p-12 h-screen" v-if="showLanding">
<UiLanding />
<button class="btn-get"
@click="showLanding = false"
>Get Started</button>
</main>
<main v-else>
<UiChatContainer/>
</main>
</template>