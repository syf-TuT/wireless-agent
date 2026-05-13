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
    processingMessage.value = '正在上传并校验 CSV 文件...'
    addLog('info', `开始处理文件：${file.name}`)
    addLog('info', `文件大小：${(file.size / 1024).toFixed(2)} KB`)
    addLog('info', `当前模式：${useKnowledgeBase.value ? '知识库增强' : '基础模式'}`)

    const response = await apiService.processCSV(file, useKnowledgeBase.value, (progress) => {
      uploadProgress.value = progress
      if (progress >= 100) {
        pipelineStage.value = 2
      }
    })

    addLog('success', 'CSV 文件上传完成')
    pipelineStage.value = 3
    processingMessage.value = '正在解析用户请求...'
    addLog('info', '后端开始解析用户请求与信道数据')

    await wait(300)
    pipelineStage.value = 4
    processingMessage.value = '正在进行意图识别...'
    addLog('info', '调用 LLM 完成业务意图识别')

    await wait(300)
    pipelineStage.value = 5
    processingMessage.value = '正在读取 CQI 并计算资源需求...'
    addLog('info', '读取 CQI 指标并计算切片带宽')

    await wait(300)
    pipelineStage.value = 6
    processingMessage.value = '正在分配网络切片资源...'
    addLog('info', '开始执行切片资源分配')

    if (response?.results) {
      results.value = response.results
      for (const result of response.results) {
        if (result.allocation_failed) {
          addLog('error', `用户 ${result.user_id} 分配失败：${result.slice_type}`)
        } else {
          addLog('success', `用户 ${result.user_id} 分配到 ${result.slice_type}，带宽 ${result.bandwidth} MHz`)
        }
      }

      pipelineStage.value = 7
      processingMessage.value = '正在生成结果...'
      await wait(300)
      addLog('success', `处理完成，共 ${response.results.length} 条用户记录`)
      addLog('info', `成功分配：${results.value.filter(r => !r.allocation_failed).length} 条`)
      addLog('info', `分配失败：${results.value.filter(r => r.allocation_failed).length} 条`)
      ElMessage.success('处理完成')
    } else {
      addLog('warning', '后端未返回有效结果数据')
      ElMessage.warning('未收到有效结果数据')
    }
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

const wait = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

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
