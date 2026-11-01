<script setup lang="ts">
import DefaultTheme from 'vitepress/theme'
import { nextTick, onMounted, onUnmounted, watch } from 'vue'
import { useRoute } from 'vitepress'

// One pending label reset per button, so a repeat click replaces its own reset
// timer instead of letting the older one flip the label back mid-feedback.
const copyResetTimers = new WeakMap<HTMLElement, number>()

// The API endpoint cards are plain HTML emitted by a markdown-it renderer, so the
// button's initial label is English. Relabel it on Chinese pages, and keep the
// click feedback in the page's language too.
const copyLabels = () => {
  // Read the path as well as <html lang>: on a cold dev-server load the language
  // attribute is set after hydration, while the URL is already final.
  const chinese =
    document.documentElement.lang.toLowerCase().startsWith('zh') ||
    window.location.pathname.includes('/zh/')
  return chinese
    ? { copy: '复制', copied: '已复制', failed: '复制失败' }
    : { copy: 'Copy', copied: 'Copied', failed: 'Copy failed' }
}

const route = useRoute()

const relabelCopyButtons = async () => {
  await nextTick()
  const labels = copyLabels()
  for (const button of document.querySelectorAll<HTMLElement>('.api-endpoint__copy')) {
    button.textContent = labels.copy
    button.setAttribute('aria-label', labels.copy)
  }
}

const flashLabel = (target: HTMLElement, state: 'copied' | 'failed') => {
  const labels = copyLabels()
  window.clearTimeout(copyResetTimers.get(target))
  target.classList.toggle('is-copied', state === 'copied')
  target.textContent = state === 'copied' ? labels.copied : labels.failed
  copyResetTimers.set(
    target,
    window.setTimeout(() => {
      target.classList.remove('is-copied')
      target.textContent = labels.copy
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
    () => flashLabel(target, 'copied'),
    () => flashLabel(target, 'failed'),
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

onMounted(() => {
  document.addEventListener('click', onDocumentClick)
  void relabelCopyButtons()
})

// The layout survives client-side navigation, so a page rendered after this
// component mounted needs the same relabel pass.
watch(() => route.path, () => void relabelCopyButtons())
onUnmounted(() => document.removeEventListener('click', onDocumentClick))
</script>

<template>
  <DefaultTheme.Layout />
</template>