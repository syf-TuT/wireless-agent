<template>
  <div class="upload-container">
    <el-upload
      ref="uploadRef"
      class="upload-area"
      drag
      :auto-upload="false"
      :on-change="handleFileChange"
      :limit="1"
      accept=".csv"
      :file-list="fileList"
    >
      <el-icon class="upload-icon"><upload-filled /></el-icon>
      <div class="el-upload__text">拖拽 CSV 文件到此处，或 <em>点击选择</em></div>
      <template #tip>
        <div class="el-upload__tip">用于接入用户请求、CQI 与射线追踪结果，启动后端切片分配流程。</div>
      </template>
    </el-upload>

    <div v-if="selectedFile" class="file-info">
      <div>
        <span>文件名</span>
        <strong>{{ selectedFile.name }}</strong>
      </div>
      <div>
        <span>文件大小</span>
        <strong>{{ formatFileSize(selectedFile.size) }}</strong>
      </div>
    </div>

    <div class="action-buttons">
      <el-button type="primary" size="large" :loading="processing" :disabled="!selectedFile || processing" @click="handleProcess">
        <el-icon v-if="!processing"><video-play /></el-icon>
        {{ processing ? '处理中' : '启动处理' }}
      </el-button>
      <el-button size="large" :disabled="processing || !selectedFile" @click="handleClear">
        <el-icon><delete /></el-icon>
        清除文件
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { Delete, UploadFilled, VideoPlay } from '@element-plus/icons-vue'
import type { UploadFile, UploadUserFile } from 'element-plus'

const emit = defineEmits<{
  process: [file: File]
  clear: []
}>()

const fileList = ref<UploadUserFile[]>([])
const selectedFile = ref<File | null>(null)
const processing = ref(false)

const handleFileChange = (file: UploadFile) => {
  if (file.raw) {
    selectedFile.value = file.raw
  }
}

const handleProcess = () => {
  if (!selectedFile.value) return
  processing.value = true
  emit('process', selectedFile.value)
}

const handleClear = () => {
  fileList.value = []
  selectedFile.value = null
  processing.value = false
  emit('clear')
}

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${Math.round((bytes / Math.pow(k, i)) * 100) / 100} ${sizes[i]}`
}

defineExpose({
  setProcessing: (value: boolean) => {
    processing.value = value
  }
})
</script>

<style scoped>
.upload-container {
  min-height: 300px;
  padding: 14px 18px 18px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.upload-area {
  flex: 1;
  min-height: 190px;
}

:deep(.el-upload),
:deep(.el-upload-dragger) {
  width: 100%;
  height: 100%;
}

:deep(.el-upload-dragger) {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  border-color: #b6c6d8;
  background: #f8fafc;
}

:deep(.el-upload-list) {
  display: none;
}

.upload-icon {
  margin-bottom: 12px;
  color: #0f766e;
  font-size: 50px;
}

:deep(.el-upload__text) {
  color: #172033;
  font-size: 15px;
}

:deep(.el-upload__text em) {
  color: #0f766e;
  font-style: normal;
  font-weight: 700;
}

:deep(.el-upload__tip) {
  margin-top: 10px;
  color: #64748b;
  font-size: 12px;
  line-height: 1.6;
}

.file-info {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}

.file-info div {
  padding: 10px 12px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
}

.file-info span,
.file-info strong {
  display: block;
}

.file-info span {
  margin-bottom: 4px;
  color: #64748b;
  font-size: 12px;
}

.file-info strong {
  overflow: hidden;
  color: #172033;
  font-size: 13px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.action-buttons {
  display: flex;
  gap: 10px;
}

.action-buttons .el-button {
  flex: 1;
}

@media (max-width: 760px) {
  .upload-container {
    min-height: 0;
  }

  .upload-area {
    min-height: 230px;
  }

  .file-info {
    grid-template-columns: 1fr;
  }

  .action-buttons {
    flex-direction: column;
  }
}
</style>
