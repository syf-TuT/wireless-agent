<template>
  <section class="resource-grid">
    <article v-for="slice in sliceSummaries" :key="slice.name" class="resource-card">
      <div class="resource-header">
        <div>
          <span class="slice-name">{{ slice.name }}</span>
          <h3>{{ slice.title }}</h3>
        </div>
        <span class="user-count">{{ slice.count }} 用户</span>
      </div>
      <div class="resource-values">
        <div>
          <span>总带宽</span>
          <strong>{{ slice.capacity }} MHz</strong>
        </div>
        <div>
          <span>已分配</span>
          <strong>{{ slice.used.toFixed(1) }} MHz</strong>
        </div>
        <div>
          <span>剩余</span>
          <strong>{{ slice.remaining.toFixed(1) }} MHz</strong>
        </div>
      </div>
      <el-progress
        :percentage="slice.utilization"
        :stroke-width="10"
        :color="slice.color"
        :show-text="false"
      />
      <div class="resource-foot">
        <span>资源利用率</span>
        <strong>{{ slice.utilization }}%</strong>
      </div>
    </article>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'

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

const definitions = [
  { name: 'eMBB', title: '增强移动宽带', capacity: 90, color: '#2563eb' },
  { name: 'URLLC', title: '低时延高可靠通信', capacity: 30, color: '#16a34a' },
  { name: 'mMTC', title: '海量机器类通信', capacity: 10, color: '#d97706' }
]

const normalizeSlice = (value: string) => value?.toLowerCase()

const sliceSummaries = computed(() => definitions.map(definition => {
  const matched = props.results.filter(result =>
    !result.allocation_failed && normalizeSlice(result.slice_type) === definition.name.toLowerCase()
  )
  const used = matched.reduce((sum, result) => sum + Number(result.bandwidth || 0), 0)
  const cappedUsed = Math.min(used, definition.capacity)
  return {
    ...definition,
    count: matched.length,
    used: cappedUsed,
    remaining: Math.max(0, definition.capacity - cappedUsed),
    utilization: Math.round((cappedUsed / definition.capacity) * 100)
  }
}))
</script>

<style scoped>
.resource-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
}

.resource-card {
  padding: 18px;
  background: #ffffff;
  border: 1px solid #dbe4ef;
  border-radius: 8px;
  box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
}

.resource-header,
.resource-foot {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.slice-name {
  color: #0f766e;
  font-size: 12px;
  font-weight: 800;
}

h3 {
  margin: 4px 0 0;
  font-size: 16px;
}

.user-count {
  padding: 5px 9px;
  color: #334155;
  background: #f1f5f9;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
}

.resource-values {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin: 18px 0 14px;
}

.resource-values div {
  min-width: 0;
  padding: 10px;
  background: #f8fafc;
  border-radius: 8px;
}

.resource-values span,
.resource-foot span {
  color: #64748b;
  font-size: 12px;
}

.resource-values strong {
  display: block;
  margin-top: 4px;
  color: #172033;
  font-size: 14px;
}

.resource-foot {
  margin-top: 10px;
}

.resource-foot strong {
  color: #172033;
  font-size: 13px;
}

@media (max-width: 980px) {
  .resource-grid {
    grid-template-columns: 1fr;
  }
}
</style>
