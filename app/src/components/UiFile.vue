<script setup lang="ts">
import type { User } from "~/types"
const urls = ref<{url: string}[]>([])
const props = defineProps({
	user: {
		type: Object as PropType<User>,
		default: () => ({}),
	},	
})
const showModal = ref(false)
const handleUpload = (url: string) => {
	urls.value.push(JSON.parse(url))
}

</script>
<template>
<button class="btn-icon m-4">
	<Icon icon="mdi-attachment" class="rotate--90 x2" @click="showModal=!showModal"/>
</button>
<UiModal @close="showModal=!showModal" v-if="showModal">
<template #body>
<ApiFile :user="props.user" endpoint="upload" @upload="handleUpload($event)" />

<div
class="grid grid-cols-3 gap-4"
>
<a v-for="u in urls" :href="u.url" target="_blank">
<Icon icon="mdi-file" class="x2" />
<span>{{u.url}}</span>
</a>
</div>
</template>
</UiModal>
</template>
<style scoped>

</style>