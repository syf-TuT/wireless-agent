<template>
  <section class="overview-grid">
    <article v-for="metric in metrics" :key="metric.label" class="metric-card">
      <div class="metric-top">
        <span :class="['metric-icon', metric.tone]">
          <el-icon><component :is="metric.icon" /></el-icon>
        </span>
        <span class="metric-label">{{ metric.label }}</span>
      </div>
      <strong>{{ metric.value }}</strong>
      <p>{{ metric.caption }}</p>
    </article>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { CircleCheck, CircleClose, Cpu, DataLine, Link, Operation } from '@element-plus/icons-vue'

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

const props = defineProps<{
  results: AllocationResult[]
  processing: boolean
  useKnowledgeBase: boolean
  backendOnline: boolean
}>()

const successful = computed(() => props.results.filter(item => !item.allocation_failed))
const failedCount = computed(() => props.results.length - successful.value.length)
const successRate = computed(() => {
  if (!props.results.length) return '0%'
  return `${Math.round((successful.value.length / props.results.length) * 100)}%`
})
const avgCqi = computed(() => {
  if (!props.results.length) return '-'
  const total = props.results.reduce((sum, item) => sum + Number(item.cqi || 0), 0)
  return (total / props.results.length).toFixed(1)
})
const avgBandwidth = computed(() => {
  if (!successful.value.length) return '-'
  const total = successful.value.reduce((sum, item) => sum + Number(item.bandwidth || 0), 0)
  return `${(total / successful.value.length).toFixed(1)} MHz`
})

const metrics = computed(() => [
  {
    label: '后端连接',
    value: props.backendOnline ? '在线' : '未连接',
    caption: props.backendOnline ? 'FastAPI 服务可访问' : '请先启动 backend_server.py',
    icon: Link,
    tone: props.backendOnline ? 'green' : 'red'
  },
  {
    label: '处理模式',
    value: props.useKnowledgeBase ? '知识库增强' : '基础模式',
    caption: props.processing ? '任务执行中' : '等待任务输入',
    icon: Operation,
    tone: 'blue'
  },
  {
    label: '处理用户',
    value: props.results.length,
    caption: `成功 ${successful.value.length} / 失败 ${failedCount.value}`,
    icon: Cpu,
    tone: 'teal'
  },
  {
    label: '分配成功率',
    value: successRate.value,
    caption: '按当前批次统计',
    icon: failedCount.value > 0 ? CircleClose : CircleCheck,
    tone: failedCount.value > 0 ? 'amber' : 'green'
  },
  {
    label: '平均 CQI',
    value: avgCqi.value,
    caption: '来自射线追踪信道指标',
    icon: DataLine,
    tone: 'purple'
  },
  {
    label: '平均带宽',
    value: avgBandwidth.value,
    caption: '仅统计分配成功用户',
    icon: DataLine,
    tone: 'blue'
  }
])
</script>

<style scoped>
.overview-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 14px;
}

.metric-card {
  min-width: 0;
  padding: 16px;
  background: #ffffff;
  border: 1px solid #dbe4ef;
  border-radius: 8px;
  box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
}

.metric-top {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 28px;
}

.metric-icon {
  width: 28px;
  height: 28px;
  display: inline-grid;
  place-items: center;
  border-radius: 8px;
  color: #ffffff;
}

.metric-icon.green {
  background: #16a34a;
}

.metric-icon.red {
  background: #e11d48;
}

.metric-icon.blue {
  background: #2563eb;
}

.metric-icon.teal {
  background: #0f766e;
}

.metric-icon.amber {
  background: #d97706;
}

.metric-icon.purple {
  background: #7c3aed;
}

.metric-label {
  color: #64748b;
  font-size: 12px;
  font-weight: 700;
}

strong {
  display: block;
  margin-top: 12px;
  color: #172033;
  font-size: 25px;
  line-height: 1.1;
}

p {
  margin: 8px 0 0;
  color: #64748b;
  font-size: 12px;
}

@media (max-width: 1320px) {
  .overview-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}

@media (max-width: 720px) {
  .overview-grid {
    grid-template-columns: 1fr;
  }
}
</style>
