<template>
  <div class="log-container">
    <div class="log-header">
      <div>
        <span class="eyebrow">审计记录</span>
        <h2>处理日志</h2>
      </div>
      <el-badge :value="logs.length" class="log-count" />
    </div>

    <div class="filter-row">
      <button
        v-for="filter in filterOptions"
        :key="filter.value"
        type="button"
        :class="{ active: activeFilter === filter.value }"
        @click="activeFilter = filter.value"
      >
        {{ filter.label }}
      </button>
    </div>

    <div ref="logContentRef" class="log-list">
      <div v-if="filteredLogs.length === 0" class="empty-log">暂无日志</div>
      <div v-for="(log, index) in filteredLogs" :key="`${log.time}-${index}`" :class="['log-item', log.type]">
        <span class="log-time">{{ log.time }}</span>
        <span class="log-type">{{ logTypeLabel(log.type) }}</span>
        <span class="log-message">{{ log.message }}</span>
      </div>
    </div>

    <div class="log-footer">
      <el-button size="small" @click="clearLogs">
        <el-icon><delete /></el-icon>
        清空
      </el-button>
      <el-button size="small" @click="exportLogs">
        <el-icon><download /></el-icon>
        导出
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import { Delete, Download } from '@element-plus/icons-vue'

interface LogEntry {
  type: 'info' | 'success' | 'warning' | 'error'
  message: string
  time: string
}

const props = defineProps<{
  logs: LogEntry[]
}>()

const emit = defineEmits<{
  clear: []
}>()

type FilterType = 'all' | LogEntry['type']

const activeFilter = ref<FilterType>('all')
const logContentRef = ref<HTMLElement>()
const filterOptions: Array<{ value: FilterType; label: string }> = [
  { value: 'all', label: '全部' },
  { value: 'info', label: '信息' },
  { value: 'success', label: '成功' },
  { value: 'warning', label: '警告' },
  { value: 'error', label: '错误' }
]

const filteredLogs = computed(() => {
  if (activeFilter.value === 'all') return props.logs
  return props.logs.filter(log => log.type === activeFilter.value)
})

watch(() => filteredLogs.value.length, async () => {
  await nextTick()
  if (logContentRef.value) {
    logContentRef.value.scrollTop = logContentRef.value.scrollHeight
  }
})

const logTypeLabel = (type: LogEntry['type']) => {
  const map = {
    info: '信息',
    success: '成功',
    warning: '警告',
    error: '错误'
  }
  return map[type]
}

const clearLogs = () => {
  emit('clear')
}

const exportLogs = () => {
  const logText = props.logs
    .map(log => `[${log.time}] [${logTypeLabel(log.type)}] ${log.message}`)
    .join('\n')

  const blob = new Blob([logText], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `process_log_${Date.now()}.txt`
  a.click()
  URL.revokeObjectURL(url)
}
</script>

<style scoped>
.log-container {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.log-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 18px 18px 0;
}

.eyebrow {
  color: #0f766e;
  font-size: 12px;
  font-weight: 700;
}

h2 {
  margin: 4px 0 0;
  font-size: 17px;
}

.filter-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 14px 18px 10px;
}

.filter-row button {
  height: 28px;
  padding: 0 10px;
  border: 1px solid #cbd5e1;
  border-radius: 999px;
  color: #475569;
  background: #ffffff;
  cursor: pointer;
}

.filter-row button.active {
  color: #ffffff;
  border-color: #0f766e;
  background: #0f766e;
}

.log-list {
  flex: 1;
  min-height: 0;
  margin: 0 18px;
  padding: 10px;
  overflow: auto;
  background: #0f172a;
  border-radius: 8px;
}

.empty-log {
  padding: 28px 0;
  text-align: center;
  color: #94a3b8;
  font-size: 13px;
}

.log-item {
  display: grid;
  grid-template-columns: 64px 44px 1fr;
  gap: 8px;
  align-items: baseline;
  padding: 7px 0;
  border-bottom: 1px solid rgba(148, 163, 184, 0.14);
  color: #cbd5e1;
  font-family: "SF Mono", Consolas, monospace;
  font-size: 12px;
}

.log-time {
  color: #94a3b8;
}

.log-type {
  font-weight: 700;
}

.log-item.success .log-type {
  color: #86efac;
}

.log-item.warning .log-type {
  color: #fbbf24;
}

.log-item.error .log-type {
  color: #fb7185;
}

.log-item.info .log-type {
  color: #93c5fd;
}

.log-footer {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  padding: 12px 18px 18px;
}
</style>
