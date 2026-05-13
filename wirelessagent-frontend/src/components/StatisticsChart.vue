<template>
  <div class="chart-wrapper">
    <div class="chart-title">{{ title }}</div>
    <v-chart class="chart" :option="chartOption" autoresize />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart, GaugeChart, PieChart } from 'echarts/charts'
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components'

use([CanvasRenderer, PieChart, BarChart, GaugeChart, GridComponent, TooltipComponent, LegendComponent])

const props = defineProps<{
  title: string
  type: 'pie' | 'bar' | 'gauge'
  data: any[]
  colors?: string[]
}>()

const defaultColors = ['#2563eb', '#16a34a', '#d97706', '#0f766e', '#7c3aed']

const chartOption = computed(() => {
  const colors = props.colors || defaultColors

  if (props.type === 'pie') {
    return {
      color: colors,
      tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
      legend: { bottom: 0, textStyle: { color: '#64748b' } },
      series: [
        {
          type: 'pie',
          radius: ['46%', '70%'],
          center: ['50%', '44%'],
          label: { show: false },
          itemStyle: { borderColor: '#fff', borderWidth: 2 },
          data: props.data
        }
      ]
    }
  }

  if (props.type === 'bar') {
    return {
      color: colors,
      tooltip: { trigger: 'axis' },
      grid: { left: 36, right: 20, top: 28, bottom: 34 },
      xAxis: {
        type: 'category',
        data: props.data.map(item => item.name),
        axisLabel: { color: '#64748b' }
      },
      yAxis: {
        type: 'value',
        axisLabel: { color: '#64748b' },
        splitLine: { lineStyle: { color: '#e2e8f0' } }
      },
      series: [
        {
          type: 'bar',
          barWidth: '46%',
          data: props.data.map(item => item.value),
          itemStyle: { borderRadius: [6, 6, 0, 0] }
        }
      ]
    }
  }

  const value = props.data[0]?.value || 0
  return {
    series: [
      {
        type: 'gauge',
        min: 0,
        max: 100,
        radius: '86%',
        progress: { show: true, width: 12 },
        axisLine: { lineStyle: { width: 12, color: [[1, '#e2e8f0']] } },
        pointer: { width: 5 },
        axisTick: { show: false },
        splitLine: { show: false },
        axisLabel: { show: false },
        detail: {
          valueAnimation: true,
          formatter: '{value}%',
          color: '#172033',
          fontSize: 24,
          fontWeight: 700
        },
        data: [{ value }]
      }
    ]
  }
})
</script>

<style scoped>
.chart-wrapper {
  height: 320px;
  padding: 16px;
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
}

.chart-title {
  margin-bottom: 8px;
  color: #172033;
  font-size: 15px;
  font-weight: 700;
}

.chart {
  width: 100%;
  height: calc(100% - 28px);
}
</style>
