<template>
  <span v-html="html"></span>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import katex from 'katex'
import 'katex/dist/katex.min.css'

const props = defineProps({
  equation: {
    type: String,
    required: true,
  },
  block: {
    type: Boolean,
    default: false,
  },
})

const html = computed(() => {
  return katex.renderToString(props.equation, {
    throwOnError: false,
    displayMode: props.block,
  })
})
</script>
