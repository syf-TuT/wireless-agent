<template>
  <div class="results-container">
    <div class="results-header">
      <div>
        <span class="eyebrow">分配追踪</span>
        <h2>用户级切片分配结果</h2>
      </div>
      <div class="header-actions">
        <el-tag v-if="results.length" type="success">共 {{ results.length }} 条记录</el-tag>
        <el-button size="small" :disabled="!results.length" @click="clearResults">
          <el-icon><delete /></el-icon>
          清空
        </el-button>
        <el-button type="primary" size="small" :disabled="!results.length" @click="exportResults">
          <el-icon><download /></el-icon>
          导出
        </el-button>
      </div>
    </div>

    <div v-if="results.length === 0" class="empty-results">
      <el-empty description="暂无分配结果">
        <template #description>
          <div>
            <strong>暂无分配结果</strong>
            <p>上传 CSV 并启动处理后，这里会展示每个用户的切片分配状态。</p>
          </div>
        </template>
      </el-empty>
    </div>

    <el-tabs v-else v-model="activeTab" class="results-tabs">
      <el-tab-pane label="用户分配明细" name="detail">
        <el-table
          :data="results"
          stripe
          border
          height="520"
          :header-cell-style="{ background: '#f8fafc', color: '#172033', fontWeight: '700' }"
        >
          <el-table-column prop="user_id" label="用户 ID" width="110" fixed align="center" />
          <el-table-column prop="request" label="用户请求" min-width="300" show-overflow-tooltip />
          <el-table-column prop="slice_type" label="识别切片" width="120" align="center">
            <template #default="{ row }">
              <span :class="['slice-badge', sliceClass(row.slice_type)]">{{ row.slice_type }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="cqi" label="CQI" width="90" align="center">
            <template #default="{ row }">
              <span :class="['cqi-badge', cqiClass(row.cqi)]">{{ row.cqi }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="bandwidth" label="分配带宽" width="120" align="center">
            <template #default="{ row }">{{ formatNumber(row.bandwidth) }} MHz</template>
          </el-table-column>
          <el-table-column prop="rate" label="速率" width="120" align="center">
            <template #default="{ row }">{{ formatNumber(row.rate) }} Mbps</template>
          </el-table-column>
          <el-table-column prop="latency" label="时延" width="100" align="center">
            <template #default="{ row }">{{ formatNumber(row.latency) }} ms</template>
          </el-table-column>
          <el-table-column label="状态" width="110" align="center">
            <template #default="{ row }">
              <el-tag :type="row.allocation_failed ? 'danger' : 'success'" effect="plain">
                {{ row.allocation_failed ? '失败' : '成功' }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column label="动态调整" width="110" align="center">
            <template #default="{ row }">
              <el-tag :type="row.adjustments_made ? 'warning' : 'info'" effect="plain">
                {{ row.adjustments_made ? '已触发' : '未触发' }}
              </el-tag>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>

      <el-tab-pane label="运行统计" name="statistics">
        <div class="statistics-grid">
          <article class="stat-box">
            <span>成功分配</span>
            <strong>{{ successCount }}</strong>
          </article>
          <article class="stat-box">
            <span>分配失败</span>
            <strong>{{ failedCount }}</strong>
          </article>
          <article class="stat-box">
            <span>动态调整</span>
            <strong>{{ adjustedCount }}</strong>
          </article>
          <article class="stat-box">
            <span>平均速率</span>
            <strong>{{ averageRate }} Mbps</strong>
          </article>
        </div>

        <div class="chart-grid">
          <StatisticsChart title="切片分布" type="pie" :data="pieChartData" :colors="['#2563eb', '#16a34a', '#d97706']" />
          <StatisticsChart title="资源利用率" type="gauge" :data="gaugeData" />
        </div>
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { Delete, Download } from '@element-plus/icons-vue'
import StatisticsChart from './StatisticsChart.vue'

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
}>()

const emit = defineEmits<{
  export: []
  clear: []
}>()

const activeTab = ref('detail')
const successfulResults = computed(() => props.results.filter(result => !result.allocation_failed))
const successCount = computed(() => successfulResults.value.length)
const failedCount = computed(() => props.results.length - successCount.value)
const adjustedCount = computed(() => props.results.filter(result => result.adjustments_made).length)
const averageRate = computed(() => {
  if (!successfulResults.value.length) return '0.0'
  const total = successfulResults.value.reduce((sum, result) => sum + Number(result.rate || 0), 0)
  return (total / successfulResults.value.length).toFixed(1)
})

const pieChartData = computed(() => [
  { name: 'eMBB', value: countBySlice('eMBB') },
  { name: 'URLLC', value: countBySlice('URLLC') },
  { name: 'mMTC', value: countBySlice('mMTC') }
])

const gaugeData = computed(() => {
  const usedBandwidth = successfulResults.value.reduce((sum, result) => sum + Number(result.bandwidth || 0), 0)
  const utilization = Math.min(100, Math.round((usedBandwidth / 130) * 100))
  return [{ name: '资源利用率', value: utilization }]
})

const countBySlice = (sliceType: string) =>
  props.results.filter(result => result.slice_type?.toLowerCase() === sliceType.toLowerCase()).length

const formatNumber = (value: number) => Number(value || 0).toFixed(1)

const cqiClass = (cqi: number) => {
  if (cqi >= 13) return 'excellent'
  if (cqi >= 9) return 'good'
  if (cqi >= 4) return 'medium'
  return 'low'
}

const sliceClass = (sliceType: string) => {
  const normalized = sliceType?.toLowerCase()
  if (normalized === 'embb') return 'embb'
  if (normalized === 'urllc') return 'urllc'
  if (normalized === 'mmtc') return 'mmtc'
  return 'default'
}

const exportResults = () => {
  const headers = ['用户ID', '用户请求', 'CQI', '切片类型', '带宽(MHz)', '速率(Mbps)', '时延(ms)', '状态', '动态调整']
  const rows = props.results.map(result => [
    result.user_id,
    result.request,
    result.cqi,
    result.slice_type,
    result.bandwidth,
    result.rate,
    result.latency,
    result.allocation_failed ? '失败' : '成功',
    result.adjustments_made ? '已触发' : '未触发'
  ])
  const csv = [headers, ...rows]
    .map(row => row.map(cell => `"${String(cell ?? '').replace(/"/g, '""')}"`).join(','))
    .join('\n')

  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `allocation_results_${Date.now()}.csv`
  a.click()
  URL.revokeObjectURL(url)
  ElMessage.success('结果已导出')
  emit('export')
}

const clearResults = () => {
  emit('clear')
}
</script>

<style scoped>
.results-container {
  min-height: 640px;
  display: flex;
  flex-direction: column;
}

.results-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 18px;
  border-bottom: 1px solid #e2e8f0;
}

.eyebrow {
  color: #0f766e;
  font-size: 12px;
  font-weight: 700;
}

h2 {
  margin: 4px 0 0;
  font-size: 18px;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.empty-results {
  flex: 1;
  display: grid;
  place-items: center;
  min-height: 420px;
}

.empty-results p {
  margin: 8px 0 0;
  color: #64748b;
}

.results-tabs {
  flex: 1;
  padding: 0 18px 18px;
}

:deep(.el-tabs__header) {
  margin-bottom: 14px;
}

.slice-badge,
.cqi-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 44px;
  height: 26px;
  padding: 0 9px;
  border-radius: 999px;
  color: #ffffff;
  font-size: 12px;
  font-weight: 800;
}

.slice-badge.embb {
  background: #2563eb;
}

.slice-badge.urllc {
  background: #16a34a;
}

.slice-badge.mmtc {
  background: #d97706;
}

.slice-badge.default {
  background: #64748b;
}

.cqi-badge.excellent {
  background: #16a34a;
}

.cqi-badge.good {
  background: #0f766e;
}

.cqi-badge.medium {
  background: #d97706;
}

.cqi-badge.low {
  background: #e11d48;
}

.statistics-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}

.stat-box {
  padding: 16px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
}

.stat-box span {
  color: #64748b;
  font-size: 12px;
}

.stat-box strong {
  display: block;
  margin-top: 8px;
  color: #172033;
  font-size: 24px;
}

.chart-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

@media (max-width: 900px) {
  .statistics-grid,
  .chart-grid {
    grid-template-columns: 1fr;
  }
}
</style>
