<template>
  <div class="col center p-4">
    <UiBtnDark class="tr m-4 fixed" />
    <div class="typewriter">
      <h1 class="p-4 text-2xl max-w-144 min-w-72 font-extrabold font-sans drop-shadow gradient-text">
        {{ typewriterText }}
      </h1>
    </div>
  </div>
</template>

<script setup lang="ts">
const fullText = 'Hi I\'m Oscar Bahamonde a Multidisciplinary Engineer, thanks for visiting my project!';
const typewriterText = ref('');

let index = 0;
const typeWriterEffect = () => {
  if (index < fullText.length) {
    typewriterText.value += fullText.charAt(index);
    index++;
    setTimeout(typeWriterEffect, 50); // Adjust speed as necessary
  }
};

onMounted(() => {
  typeWriterEffect();
});
</script>
<style scoped>
.gradient-text {
  background: linear-gradient(to bottom right, #0d9488, #00c2b8);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  display: inline-block;
}

@keyframes typewriter {
  from { width: 0; }
  to { width: 100%; }
}

.typewriter h1 {
  overflow: hidden;
  white-space: wrap;
  margin: 0 auto;
  animation: typewriter 2s steps(20, end), blink-caret .75s step-end infinite;
}


</style>