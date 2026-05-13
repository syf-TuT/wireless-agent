import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import assert from 'node:assert/strict'

const apiSource = readFileSync(new URL('../src/services/api.ts', import.meta.url), 'utf8')
const appSource = readFileSync(new URL('../src/App.vue', import.meta.url), 'utf8')

test('api service exposes stream processing against the stream endpoint', () => {
  assert.match(apiSource, /processCSVStream/)
  assert.match(apiSource, /\/process-csv-stream/)
  assert.match(apiSource, /getReader\(/)
  assert.match(apiSource, /TextDecoder/)
})

test('app consumes stream events instead of waiting for batch results', () => {
  assert.match(appSource, /processCSVStream/)
  assert.match(appSource, /handleStreamEvent/)
  assert.doesNotMatch(appSource, /const response = await apiService\.processCSV/)
})
