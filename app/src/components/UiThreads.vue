<script setup lang="ts">
type Thread = {
	agent_id: string;
	user_id: string;
	title: string;
}

const threads = ref<Thread[]>([{ agent_id: '1', user_id: '1', title: 'Thread 1'}])
const visible = ref(false);

</script>
<template>
<div class="animate-slide-in-left dark:bg-gray-800 bg-gray-300 h-screen w-72 " v-if="visible">
<ul class="flex flex-col items-center justify-center h-full">
<li class="p-4" v-for="thread in threads" :key="thread.agent_id">
{{ thread.title }}
</li>
</ul>
</div>
 <button @click="visible = !visible"

:class="visible ? 'left-0 animate-slide-in-left' : 'right-0 '"
 class="btn-icon"><Icon :icon="!visible ? 'mdi-chevron-right' : 'mdi-chevron-left'" class="x2" 

/></button>
</template>
<style scoped>

</style>