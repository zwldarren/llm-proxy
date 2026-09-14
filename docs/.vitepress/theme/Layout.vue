<script setup lang="ts">
import DefaultTheme from 'vitepress/theme'
import { onMounted } from 'vue'

// Copy-path buttons on the API endpoint header cards. The buttons are emitted
// by a markdown-it renderer (plain HTML, no component), so clicks are handled
// via delegation at the document level — the Layout persists across routes.
const onDocumentClick = (event: MouseEvent) => {
  const target = (event.target as HTMLElement | null)?.closest<HTMLElement>(
    '.api-endpoint__copy',
  )
  if (!target) return
  const path = target
    .closest('.api-endpoint')
    ?.querySelector('.api-endpoint__path')
    ?.textContent
  if (!path) return

  void copyText(path).finally(() => {
    target.classList.add('is-copied')
    target.textContent = 'Copied'
    window.setTimeout(() => {
      target.classList.remove('is-copied')
      target.textContent = 'Copy'
    }, 1500)
  })
}

// Async Clipboard API requires a secure context and permissions; fall back to a
// temporary textarea + execCommand for browsers that deny the promise API.
const copyText = (text: string): Promise<void> => {
  const fallback = () => {
    const textarea = document.createElement('textarea')
    textarea.value = text
    textarea.style.position = 'fixed'
    textarea.style.opacity = '0'
    document.body.append(textarea)
    textarea.select()
    document.execCommand('copy')
    textarea.remove()
  }
  return navigator.clipboard?.writeText
    ? navigator.clipboard.writeText(text).catch(fallback)
    : Promise.resolve(fallback())
}

onMounted(() => document.addEventListener('click', onDocumentClick))
</script>

<template>
  <DefaultTheme.Layout />
</template>