<script setup lang="ts">
import type { User, Message } from "~/types"
const props = defineProps({
	user: {
		type: Object as PropType<User>,
		default: () => ({}),
	},	
})
const userInput = ref("")

const isLoading = ref(false)

const messages = ref<Message[]>([])

const appendMessage = (message: Message) => {
	messages.value.unshift(message)
}

const appendUserAndAssistantMessages = (userContent: string, assistantContent: string = "") => {
	appendMessage({ role: "user", content: userContent });
	appendMessage({ role: "assistant", content: assistantContent });
}

const handler = async(prompt:string)=>{
	
		appendUserAndAssistantMessages(userInput.value);
		const params = new URLSearchParams({ inputs: prompt });
		const cleanup = useEvent<Message>(`/api/chat/${props.user.sub}?${params}`, 
		(data) => {
			messages.value[0].content += data.content
		})
		onUnmounted(() => {
			cleanup()
		})
		userInput.value = ""
}

const handleVoice = (prompt:string)=>{
	userInput.value = prompt
	handler(prompt)
}
</script>
<template>
<div class="col center p-4 ">

	<ApiMessage :messages="messages" :user="props.user" />	
	    <footer class="row start gap-2 p-4 bottom-4 absolute  min-w-192">	
			
			<ApiVoice @send="handleVoice($event)" />
			<textarea v-model="userInput" placeholder="Message ChatGPT..."
			class="w-full b-primary dark:b-secondary b-2 p-2 rounded-lg bg-info dark:bg-accent	dark:text-white"
			@keyup.enter="handler(userInput)" />
				<UiFile  :user="props.user"/>
		</footer>
	
	<div v-if="isLoading" class="loading">
		<Icon icon="mdi-loading mdi-spin" class="x4"/>
	</div>
</div>
</template>