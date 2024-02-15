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

const messagesContainer = ref<HTMLElement | null>(null)
watch(messages, async () => {
  await nextTick();
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight + 1000;
  }
}, { deep: true }); 

const appendMessage = (message: Message) => {
	messages.value.push(message)
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
			messages.value[messages.value.length-1].content += data.content
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
<div class="col center p-4 overflow-auto " ref="messagesContainer">
	<div class=" mb-24">
	<ApiMessage :messages="messages" :user="props.user" />	
</div>
	   <footer class="flex flex-wrap justify-start items-center gap-2 p-4 fixed bottom-0 left-0 right-0 mx-auto w-full max-w-192">
  <ApiVoice @send="handleVoice($event)" class="flex-none" />
  <textarea v-model="userInput" placeholder="Message NotChatGPT..."
    class="flex-grow h-12 b-primary dark:b-secondary b-2 p-2 rounded-lg bg-info dark:bg-accent dark:text-white resize-none"
    @keyup.enter.prevent="handler(userInput)" />
  <UiFile :user="props.user" class="flex-none"/>
</footer>
	
	<div v-if="isLoading" class="loading">
		<Icon icon="mdi-loading mdi-spin" class="x4"/>
	</div>
</div>
</template>