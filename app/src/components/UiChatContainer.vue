<script setup lang="ts">
const visible = ref(false);
</script>
<template>
<div>
<ApiAuthorizer>
 <template v-slot="{ user }">
  <main class="row center overflow-auto">
     <UiThreads :user="user"  :hidden="!visible"/>
   

     <UiBtnDark class="tr m-4 fixed" />
    <section class="overflow-auto h-screen w-full col center">
  <img :class="isDark ? '' : 'invert'" class="x6 rf m-4" src="/logo.png"/>
  <div class="col start">
   <ApiChatContainer :user="user"/></div>
</section>
  </main>
  </template>
  </ApiAuthorizer>


</div>
</template>
<style scoped>

</style>