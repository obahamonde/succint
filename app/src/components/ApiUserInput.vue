<script setup lang="ts">
import type { User, Message } from "~/types"
const props = defineProps({
	user: {
		type: Object as PropType<User>,
		default: () => ({}),
	},	
})
const stream = ref(true)
const userInput = ref("")

const isLoading = ref(false)

const messages = ref<Message[]>([])

const appendMessage = (message: Message) => {
	messages.value.unshift(message)
}
 
const handler = async(prompt:string)=>{
	if (stream.value) {
	appendMessage({
		role: "user",
		content: userInput.value,
	})
	 appendMessage({
		role: "assistant",
		content: "",
	})
	const cleanup = useEvent<Message>(`/api/chat?inputs=${prompt}`, 
	(data) => {
		messages.value[0].content+=data.content
})
onUnmounted(() => {
	cleanup()
})
userInput.value = ""
}

else {
	appendMessage({
		role: "user",
		content: userInput.value,
	})
	const response = await fetch("/api/run?inputs=" + prompt)
	const content = await response.text()
	content ? appendMessage({
		role: "assistant",
		content: content
	}) : null
userInput.value = ""
}
}

const handleVoice = (prompt:string)=>{
	userInput.value = prompt
	handler(prompt)
}
</script>
<template>
<div class="col center p-4 ">
	<Icon :icon="stream?'mdi-chat':'mdi-lightning-bolt-outline'" class="x3 sh rf cp scale" @click="stream=!stream" />
	    <footer class="row start gap-4 p-4  min-w-168">	
			
			<ApiVoice @send="handleVoice($event)" />
			<input v-model="userInput"
			class="w-full b-primary dark:b-secondary b-2 p-4 rounded-lg bg-info dark:bg-accent	dark:text-white"
			@keyup.enter="handler(userInput)" />
				<UiFile  :user="props.user"/>
		</footer>
		<ApiMessage :messages="messages" :user="props.user" />	
	<div v-if="isLoading" class="loading">
		<Icon icon="mdi-loading mdi-spin" class="x4"/>
	</div>
</div>
</template>