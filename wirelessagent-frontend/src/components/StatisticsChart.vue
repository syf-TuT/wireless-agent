<template>
  <div class="chart-wrapper">
    <div class="chart-title">
      <span class="title-line"></span>
      {{ title }}
    </div>
    <v-chart class="chart" :option="chartOption" autoresize />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { PieChart, BarChart, GaugeChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent
} from 'echarts/components'

use([
  CanvasRenderer,
  PieChart,
  BarChart,
  GaugeChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent
])

const props = defineProps<{
  title: string
  type: 'pie' | 'bar' | 'gauge'
  data: any[]
  colors?: string[]
}>()

const defaultColors = [
  '#c084fc',
  '#a78bfa',
  '#c4b5fd',
  '#06b6d4',
  '#22c55e',
  '#f59e0b'
]

const chartOption = computed(() => {
  const colorList = props.colors || defaultColors

  if (props.type === 'pie') {
    return {
      tooltip: {
        trigger: 'item',
        formatter: '{b}: {c} ({d}%)',
        backgroundColor: 'rgba(255, 255, 255, 0.95)',
        borderColor: 'rgba(196, 181, 253, 0.3)',
        borderWidth: 1,
        textStyle: {
          color: '#1e293b'
        }
      },
      legend: {
        orient: 'horizontal',
        bottom: 0,
        textStyle: {
          color: '#64748b'
        }
      },
      series: [
        {
          name: props.title,
          type: 'pie',
          radius: ['45%', '70%'],
          center: ['50%', '45%'],
          avoidLabelOverlap: false,
          itemStyle: {
            borderRadius: 8,
            borderColor: '#fff',
            borderWidth: 2
          },
          label: {
            show: false
          },
          emphasis: {
            label: {
              show: true,
              fontSize: 14,
              fontWeight: 'bold',
              color: '#1e293b'
            },
            itemStyle: {
              shadowBlur: 10,
              shadowOffsetX: 0,
              shadowColor: 'rgba(0, 0, 0, 0.2)'
            }
          },
          labelLine: {
            show: false
          },
          data: props.data,
          color: colorList,
          animationType: 'scale',
          animationEasing: 'elasticOut' as const,
          animationDelay: (idx: number) => idx * 100
        }
      ]
    }
  }

  if (props.type === 'bar') {
    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow'
        },
        backgroundColor: 'rgba(255, 255, 255, 0.95)',
        borderColor: 'rgba(196, 181, 253, 0.3)',
        borderWidth: 1,
        textStyle: {
          color: '#1e293b'
        }
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        top: '10%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: props.data.map((item: any) => item.name),
        axisLine: {
          lineStyle: {
            color: 'rgba(196, 181, 253, 0.2)'
          }
        },
        axisLabel: {
          color: '#64748b'
        }
      },
      yAxis: {
        type: 'value',
        name: 'Mbps',
        nameTextStyle: {
          color: '#64748b'
        },
        axisLine: {
          show: false
        },
        axisLabel: {
          color: '#64748b'
        },
        splitLine: {
          lineStyle: {
            color: 'rgba(196, 181, 253, 0.1)'
          }
        }
      },
      series: [
        {
          name: '速率',
          type: 'bar',
          barWidth: '50%',
          data: props.data.map((item: any) => item.value),
          itemStyle: {
            borderRadius: [6, 6, 0, 0],
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: '#c084fc' },
                { offset: 1, color: '#a78bfa' }
              ]
            }
          },
          emphasis: {
            itemStyle: {
              color: {
                type: 'linear',
                x: 0,
                y: 0,
                x2: 0,
                y2: 1,
                colorStops: [
                  { offset: 0, color: '#c084fc' },
                  { offset: 1, color: '#818cf8' }
                ]
              }
            }
          },
          animationDelay: (idx: number) => idx * 50
        }
      ],
      animationEasing: 'elasticOut' as const,
      animationDelayUpdate: (idx: number) => idx * 15
    }
  }

  if (props.type === 'gauge') {
    const value = props.data[0]?.value || 0
    return {
      series: [
        {
          type: 'gauge',
          startAngle: 200,
          endAngle: -20,
          min: 0,
          max: 100,
          splitNumber: 10,
          radius: '85%',
          center: ['50%', '50%'],
          itemStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 1,
              y2: 0,
              colorStops: [
                { offset: 0, color: '#a78bfa' },
                { offset: 0.5, color: '#c084fc' },
                { offset: 1, color: '#8b5cf6' }
              ]
            }
          },
          progress: {
            show: true,
            width: 14,
            roundCap: true
          },
          pointer: {
            show: true,
            length: '60%',
            width: 6,
            itemStyle: {
              color: '#a78bfa'
            }
          },
          axisLine: {
            lineStyle: {
              width: 14,
              color: [[1, 'rgba(196, 181, 253, 0.15)']]
            },
            roundCap: true
          },
          axisTick: {
            show: false
          },
          splitLine: {
            show: false
          },
          axisLabel: {
            show: false
          },
          title: {
            show: false
          },
          detail: {
            width: '50%',
            lineHeight: 28,
            borderRadius: 6,
            offsetCenter: [0, '45%'],
            fontSize: 22,
            fontWeight: 'bold',
            formatter: '{value}%',
            color: '#1e293b'
          },
          data: [{ value, name: '网络利用率' }]
        }
      ],
      animationDuration: 1500,
      animationEasing: 'cubicOut' as const
    }
  }

  return {}
})
</script>

<style scoped>
.chart-wrapper {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
}

.chart-title {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 16px;
  font-weight: 600;
  color: #1e293b;
  margin-bottom: 16px;
  padding-left: 12px;
}

.title-line {
  width: 4px;
  height: 18px;
  background: linear-gradient(180deg, #c084fc 0%, #a78bfa 100%);
  border-radius: 2px;
}

.chart {
  flex: 1;
  min-height: 200px;
}
</style>
