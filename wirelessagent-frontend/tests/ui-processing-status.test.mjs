import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import assert from 'node:assert/strict'

const appVue = readFileSync(new URL('../src/App.vue', import.meta.url), 'utf8')
const processingPipelineVue = readFileSync(
  new URL('../src/components/ProcessingPipeline.vue', import.meta.url),
  'utf8'
)

test('processing runs inline without a blocking overlay', () => {
  assert.equal(appVue.includes('processing-overlay'), false)
  assert.equal(appVue.includes('role="dialog"'), false)
  assert.equal(processingPipelineVue.includes('processing-status'), true)
  assert.equal(appVue.includes(':processing="processing"'), true)
})
