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
	<div class="w-1/2 mx-auto p-4">
		<div class="sh rounded-lg backdrop-blur-md ">
			<div v-for="message in props.messages" :key="message.content" class="col center">
				<p class="row center">
				<img :src="message.role === 'user' ? props.user.picture : '/mistress.png'" class="x4 rf m-4"/>
				<UIRichText :content="message.content" />
					</p>
			</div>
		</div>
	</div>
</template>
