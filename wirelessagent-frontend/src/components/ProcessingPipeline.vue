<template>
  <div class="pipeline-shell">
    <div v-if="processing || failed" class="processing-status" :class="{ failed }">
      <div class="status-copy">
        <span class="status-dot" aria-hidden="true"></span>
        <div>
          <strong>{{ statusTitle }}</strong>
          <p>{{ message }}</p>
        </div>
      </div>
      <el-progress
        v-if="processing"
        :percentage="visibleProgress"
        :show-text="false"
        :stroke-width="8"
      />
    </div>

    <ol class="pipeline">
      <li v-for="(stage, index) in stages" :key="stage" :class="stageClass(index + 1)">
        <span class="stage-index">
          <el-icon v-if="failed && currentStage === index + 1"><circle-close /></el-icon>
          <el-icon v-else-if="currentStage > index + 1"><circle-check /></el-icon>
          <span v-else>{{ index + 1 }}</span>
        </span>
        <div>
          <strong>{{ stage }}</strong>
          <p>{{ stageDescription(index + 1) }}</p>
        </div>
      </li>
    </ol>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { CircleCheck, CircleClose } from '@element-plus/icons-vue'

const props = defineProps<{
  currentStage: number
  failed: boolean
  processing: boolean
  message: string
  uploadProgress: number
}>()

const stages = ['CSV 上传', '数据校验', '请求解析', '意图识别', 'CQI 读取', '切片分配', '结果生成']

const descriptions = [
  '接收仿真数据和用户请求',
  '检查 CSV 字段和文件格式',
  '提取用户业务需求',
  '调用 LLM 或知识库增强链路',
  '读取信道质量并映射资源需求',
  '按切片类型写入资源池',
  '返回表格、统计和日志'
]

const stageClass = (stageNumber: number) => ({
  stage: true,
  completed: props.currentStage > stageNumber,
  active: props.currentStage === stageNumber && !props.failed,
  failed: props.currentStage === stageNumber && props.failed,
  waiting: props.currentStage < stageNumber
})

const stageDescription = (stageNumber: number) => descriptions[stageNumber - 1]

const statusTitle = computed(() => (props.failed ? '处理失败' : '正在处理'))

const visibleProgress = computed(() => {
  if (props.uploadProgress > 0 && props.uploadProgress < 100) {
    return props.uploadProgress
  }

  if (props.currentStage <= 0) return 0
  return Math.min(100, Math.round((props.currentStage / stages.length) * 100))
})
</script>

<style scoped>
.pipeline-shell {
  height: calc(100% - 56px);
  padding: 16px 18px 18px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.processing-status {
  padding: 12px;
  border: 1px solid #99f6e4;
  border-radius: 8px;
  background: #f0fdfa;
}

.processing-status.failed {
  border-color: #fecdd3;
  background: #fff1f2;
}

.status-copy {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  margin-bottom: 10px;
}

.status-dot {
  width: 10px;
  height: 10px;
  flex: 0 0 auto;
  margin-top: 5px;
  border-radius: 50%;
  background: #0f766e;
  box-shadow: 0 0 0 5px rgba(15, 118, 110, 0.12);
}

.processing-status.failed .status-dot {
  background: #e11d48;
  box-shadow: 0 0 0 5px rgba(225, 29, 72, 0.12);
}

.status-copy strong {
  display: block;
  color: #172033;
  font-size: 14px;
}

.status-copy p {
  margin: 3px 0 0;
  color: #64748b;
  font-size: 12px;
  line-height: 1.45;
}

.pipeline {
  margin: 0;
  padding: 0;
  list-style: none;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  gap: 8px;
}

.stage {
  display: grid;
  grid-template-columns: 34px 1fr;
  gap: 12px;
  align-items: center;
  padding: 10px 12px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: #f8fafc;
}

.stage-index {
  width: 32px;
  height: 32px;
  display: grid;
  place-items: center;
  border-radius: 50%;
  color: #64748b;
  background: #e2e8f0;
  font-weight: 800;
}

.stage strong {
  display: block;
  color: #172033;
  font-size: 14px;
}

.stage p {
  margin: 3px 0 0;
  color: #64748b;
  font-size: 12px;
}

.stage.completed {
  border-color: #bbf7d0;
  background: #f0fdf4;
}

.stage.completed .stage-index {
  color: #ffffff;
  background: #16a34a;
}

.stage.active {
  border-color: #99f6e4;
  background: #f0fdfa;
}

.stage.active .stage-index {
  color: #ffffff;
  background: #0f766e;
}

.stage.failed {
  border-color: #fecdd3;
  background: #fff1f2;
}

.stage.failed .stage-index {
  color: #ffffff;
  background: #e11d48;
}
</style>
