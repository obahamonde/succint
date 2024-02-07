<script setup lang="ts">
defineProps({
	user: {
		type: String,
		required: true
	}
})
</script>
<template>
<div>{{ user }}</div>
</template>
<style scoped>

</style>