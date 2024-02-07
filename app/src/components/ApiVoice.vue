<script setup lang="ts">
const lang = ref("en-US")
const { start, stop, isListening, result } = useSpeechRecognition({lang:lang.value})
const emit = defineEmits(["send"])
const toggle = () => {
	if (isListening.value) {
		stop()
		emit("send",result.value)
	} else {
		start()
	}
	result.value = ""
}
</script>
<template>
<button class="btn-icon">

	<Icon :icon="isListening ? 'mdi-microphone-off' : 'mdi-microphone'"
		@click="toggle"
		class="x3"
	/>

</button>

</template>
<style scoped>

</style>