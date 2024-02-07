
import ApiAuthorizer from '~/components/ApiAuthorizer.vue';

<template>
<ApiAuthorizer>
 <template v-slot="{ user }">
  <main class="col center p-8">
   <img src="/mistress.png" class="x6 rf m-4"/>
   <ApiUserInput :user="user"/>
  </main>
  </template>
  </ApiAuthorizer>
</template>
