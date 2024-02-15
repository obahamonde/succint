<template>
<button class="btn-icon">
<Icon :icon="isDark ? 'mdi-moon-waning-crescent' : 'mdi-white-balance-sunny'" class="x2"
@click="toggleDark()"
 />
</button>
</template>
