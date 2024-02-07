<script setup lang="ts">
import type { User } from "~/types"
const props = defineProps({
	user: {
		type: Object as PropType<User>,
		default: () => ({}),
	},	
})
const showModal = ref(false)

</script>
<template>
<button class="btn-icon">
	<Icon icon="mdi-attachment" class="rotate--90 x3" @click="showModal=!showModal"/>
</button>
<UiModal @close="showModal=!showModal" v-if="showModal">
<template #body>
<ApiFile :user="props.user" />
</template>
</UiModal>
</template>
<style scoped>

</style>