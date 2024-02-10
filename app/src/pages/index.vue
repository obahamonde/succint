<template>
<ApiAuthorizer>
 <template v-slot="{ user }">
  <main class="row center">
     <ApiThreads :user="user"   />
    <section class="overflow-auto w-full col center">
    <img :src="isDark ? './chatgpt-dark.png' : 'chatgpt-light.png'" class="x6 rf m-4"/>
   <ApiChatContainer :user="user"/>
</section>
  </main>
  </template>
  </ApiAuthorizer>
</template>
