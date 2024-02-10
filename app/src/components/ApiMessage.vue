<script setup lang="ts">
import { Message, User } from '~/types';
const props = defineProps({
	messages: {
		type: Array as PropType<Message[]>,
		default: () => ([]),
	},
	user: {
		type: Object as PropType<User>,
		default: () => ({}),
	},
})
</script>
<template>
	<div class="w-full mx-auto p-4">
		<div class="rounded-lg">
			<div v-for="message in props.messages" :key="message.content" class="col start my-4">
				<p class="row">
				<img :src="message.role === 'user' ? props.user.picture : isDark ? './chatgpt-dark.png' : 'chatgpt-light.png'" class="x2 rf m-4"/>
				<UIRichText :content="message.content" />
					</p>
			</div>
		</div>
	</div>
</template>
