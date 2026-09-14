<script setup lang="ts">
import DefaultTheme from 'vitepress/theme'
import { onMounted, onUnmounted } from 'vue'

// One pending label reset per button, so a repeat click replaces its own reset
// timer instead of letting the older one flip the label back mid-feedback.
const copyResetTimers = new WeakMap<HTMLElement, number>()

const flashLabel = (target: HTMLElement, label: string) => {
  window.clearTimeout(copyResetTimers.get(target))
  target.classList.toggle('is-copied', label === 'Copied')
  target.textContent = label
  copyResetTimers.set(
    target,
    window.setTimeout(() => {
      target.classList.remove('is-copied')
      target.textContent = 'Copy'
      copyResetTimers.delete(target)
    }, 1500),
  )
}

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

  // Never claim a copy that did not happen: a failed clipboard write reports
  // the failure instead of the success label.
  void copyText(path).then(
    () => flashLabel(target, 'Copied'),
    () => flashLabel(target, 'Copy failed'),
  )
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
    const copied = document.execCommand('copy')
    textarea.remove()
    if (!copied) throw new Error('execCommand("copy") failed')
  }
  return navigator.clipboard?.writeText
    ? navigator.clipboard.writeText(text).catch(fallback)
    : Promise.resolve().then(fallback)
}

onMounted(() => document.addEventListener('click', onDocumentClick))
onUnmounted(() => document.removeEventListener('click', onDocumentClick))
</script>

<template>
  <DefaultTheme.Layout />
</template>