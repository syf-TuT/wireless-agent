<template>
  <div class="app-shell">
    <header class="app-header">
      <div class="brand">
        <div class="brand-icon">
          <el-icon><connection /></el-icon>
        </div>
        <div>
          <h1>无线智能体框架实验仿真</h1>
        </div>
      </div>
      <VersionTag :backend-online="backendOnline" @update:useKnowledgeBase="handleVersionUpdate" />
    </header>

    <main class="app-main">
      <SystemOverview
        :results="results"
        :processing="processing"
        :use-knowledge-base="useKnowledgeBase"
        :backend-online="backendOnline"
      />

      <SliceResourcePool :results="results" />

      <section class="workflow-grid">
        <div class="panel upload-panel">
          <div class="panel-heading">
            <div>
              <span class="eyebrow">数据接入</span>
              <h2>CSV 批量任务</h2>
            </div>
          </div>
          <FileUpload ref="fileUploadRef" @process="handleFileProcess" @clear="handleFileClear" />
        </div>

        <div class="panel pipeline-panel">
          <div class="panel-heading">
            <div>
              <span class="eyebrow">工程流程</span>
              <h2>处理流水线</h2>
            </div>
          </div>
          <ProcessingPipeline
            :current-stage="pipelineStage"
            :failed="pipelineFailed"
            :processing="processing"
            :message="processingMessage"
            :upload-progress="uploadProgress"
          />
        </div>

        <div class="panel log-panel">
          <ProcessLog :logs="logs" @clear="clearLogs" />
        </div>
      </section>

      <section class="panel results-panel">
        <ResultsDisplay :results="results" @export="exportResults" @clear="clearResults" />
      </section>
    </main>

  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { Connection } from '@element-plus/icons-vue'
import FileUpload from './components/FileUpload.vue'
import ProcessLog from './components/ProcessLog.vue'
import ProcessingPipeline from './components/ProcessingPipeline.vue'
import ResultsDisplay from './components/ResultsDisplay.vue'
import SliceResourcePool from './components/SliceResourcePool.vue'
import SystemOverview from './components/SystemOverview.vue'
import VersionTag from './components/VersionTag.vue'
import apiService from './services/api'
import type { StreamEvent } from './services/api'

interface LogEntry {
  type: 'info' | 'success' | 'warning' | 'error'
  message: string
  time: string
}

interface AllocationResult {
  user_id: string
  request: string
  cqi: number
  slice_type: string
  bandwidth: number
  rate: number
  latency: number
  allocation_failed: boolean
  adjustments_made: boolean
}

const fileUploadRef = ref()
const logs = ref<LogEntry[]>([])
const results = ref<AllocationResult[]>([])
const processing = ref(false)
const uploadProgress = ref(0)
const useKnowledgeBase = ref(true)
const processingMessage = ref('正在处理文件...')
const pipelineStage = ref(0)
const pipelineFailed = ref(false)
const backendOnline = ref(false)

const addLog = (type: LogEntry['type'], message: string) => {
  const now = new Date()
  const time = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`
  logs.value.push({ type, message, time })
}

const checkBackend = async () => {
  try {
    await apiService.getHealth()
    backendOnline.value = true
  } catch {
    backendOnline.value = false
  }
}

onMounted(() => {
  checkBackend()
})

const handleVersionUpdate = (value: boolean) => {
  useKnowledgeBase.value = value
  addLog('info', `已切换到${value ? '知识库增强' : '基础'}处理模式`)
}

const handleFileProcess = async (file: File) => {
  try {
    processing.value = true
    pipelineFailed.value = false
    pipelineStage.value = 1
    uploadProgress.value = 0
    results.value = []
    processingMessage.value = '正在上传并校验 CSV 文件...'
    addLog('info', `开始处理文件：${file.name}`)
    addLog('info', `文件大小：${(file.size / 1024).toFixed(2)} KB`)
    addLog('info', `当前模式：${useKnowledgeBase.value ? '知识库增强' : '基础模式'}`)

    await apiService.processCSVStream(file, useKnowledgeBase.value, handleStreamEvent)
  } catch (error: any) {
    console.error('Process error:', error)
    pipelineFailed.value = true
    processingMessage.value = '处理失败'
    addLog('error', `处理失败：${error?.message || '未知错误'}`)
    ElMessage.error('处理失败，请检查后端服务与 CSV 文件格式')
  } finally {
    processing.value = false
    uploadProgress.value = 0
    processingMessage.value = '正在处理文件...'
    if (fileUploadRef.value) {
      fileUploadRef.value.setProcessing(false)
    }
    checkBackend()
  }
}

const handleStreamEvent = (event: StreamEvent) => {
  if (event.type === 'log') {
    logs.value.push(event.log)
    return
  }

  if (event.type === 'result') {
    results.value = [...results.value, normalizeAllocationResult(event.result)]
    return
  }

  if (event.type === 'progress') {
    pipelineStage.value = event.stage || pipelineStage.value
    uploadProgress.value = event.total > 0 ? Math.round((event.processed / event.total) * 100) : 0
    processingMessage.value = event.message || `已处理 ${event.processed}/${event.total} 个用户`
    return
  }

  if (event.type === 'complete') {
    pipelineStage.value = 7
    uploadProgress.value = 100
    processingMessage.value = '处理完成'
    addLog('success', `处理完成，共 ${event.total} 条用户记录`)
    addLog('info', `成功分配：${event.success} 条`)
    addLog('info', `分配失败：${event.failed} 条`)
    ElMessage.success('处理完成')
    return
  }

  if (event.type === 'error') {
    pipelineFailed.value = true
    processingMessage.value = '处理失败'
    addLog('error', event.message)
  }
}

const normalizeAllocationResult = (result: Partial<AllocationResult>): AllocationResult => ({
  user_id: String(result.user_id ?? ''),
  request: String(result.request ?? ''),
  cqi: Number(result.cqi ?? 0),
  slice_type: String(result.slice_type ?? 'Failed'),
  bandwidth: Number(result.bandwidth ?? 0),
  rate: Number(result.rate ?? 0),
  latency: Number(result.latency ?? 0),
  allocation_failed: Boolean(result.allocation_failed),
  adjustments_made: Boolean(result.adjustments_made)
})

const handleFileClear = () => {
  pipelineStage.value = 0
  pipelineFailed.value = false
  addLog('info', '已清除待上传文件')
}

const clearLogs = () => {
  logs.value = []
  addLog('info', '日志已清空')
}

const clearResults = () => {
  results.value = []
  pipelineStage.value = 0
  pipelineFailed.value = false
  addLog('info', '结果已清空')
}

const exportResults = () => {
  addLog('success', '结果已导出')
}
</script>

<style>
* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  background: #f4f7fb;
  color: #172033;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}

#app {
  min-height: 100vh;
}

.app-shell {
  min-height: 100vh;
}

.app-header {
  position: relative;
  z-index: 20;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  padding: 18px 32px;
  background: rgba(255, 255, 255, 0.94);
  border-bottom: 1px solid #dbe4ef;
  backdrop-filter: blur(14px);
}

.brand {
  display: flex;
  align-items: center;
  gap: 14px;
}

.brand-icon {
  width: 46px;
  height: 46px;
  display: grid;
  place-items: center;
  border-radius: 10px;
  background: #0f766e;
  color: #ffffff;
  font-size: 25px;
}

.brand h1 {
  margin: 0;
  font-size: 22px;
  line-height: 1.25;
  font-weight: 700;
}

.brand p {
  margin: 4px 0 0;
  color: #64748b;
  font-size: 13px;
}

.app-main {
  width: min(1680px, 100%);
  margin: 0 auto;
  padding: 24px 32px 40px;
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.workflow-grid {
  display: grid;
  grid-template-columns: minmax(280px, 0.85fr) minmax(320px, 1fr) minmax(320px, 1fr);
  gap: 18px;
  align-items: stretch;
}

.panel {
  background: #ffffff;
  border: 1px solid #dbe4ef;
  border-radius: 8px;
  box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
}

.panel-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 18px 18px 0;
}

.panel-heading h2 {
  margin: 4px 0 0;
  font-size: 17px;
}

.eyebrow {
  color: #0f766e;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0;
}

.upload-panel,
.pipeline-panel,
.log-panel {
  min-height: 360px;
}

.results-panel {
  overflow: hidden;
}

@media (max-width: 1180px) {
  .workflow-grid {
    grid-template-columns: 1fr 1fr;
  }

  .log-panel {
    grid-column: 1 / -1;
  }
}

@media (max-width: 760px) {
  .app-header {
    align-items: flex-start;
    flex-direction: column;
    padding: 16px;
  }

  .app-main {
    padding: 16px;
  }

  .workflow-grid {
    grid-template-columns: 1fr;
  }

  .brand h1 {
    font-size: 18px;
  }
}
</style>
