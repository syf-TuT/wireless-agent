<template>
  <div class="header-actions">
    <span class="status-pill" :class="{ online: backendOnline }">
      <span class="status-dot"></span>
      {{ backendOnline ? '后端在线' : '后端未连接' }}
    </span>
    <button class="mode-toggle" :class="{ active: useKnowledgeBase }" type="button" @click="toggleKnowledgeBase">
      <el-icon>
        <collection-tag v-if="useKnowledgeBase" />
        <info-filled v-else />
      </el-icon>
      {{ useKnowledgeBase ? '知识库增强' : '基础模式' }}
    </button>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { CollectionTag, InfoFilled } from '@element-plus/icons-vue'

defineProps<{
  backendOnline: boolean
}>()

const emit = defineEmits<{
  (e: 'update:useKnowledgeBase', value: boolean): void
}>()

const useKnowledgeBase = ref(true)

const toggleKnowledgeBase = () => {
  useKnowledgeBase.value = !useKnowledgeBase.value
  emit('update:useKnowledgeBase', useKnowledgeBase.value)
}
</script>

<style scoped>
.header-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.status-pill,
.mode-toggle {
  height: 36px;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 0 13px;
  border-radius: 999px;
  font-size: 13px;
  font-weight: 700;
}

.status-pill {
  color: #9f1239;
  background: #fff1f2;
  border: 1px solid #fecdd3;
}

.status-pill.online {
  color: #166534;
  background: #f0fdf4;
  border-color: #bbf7d0;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: currentColor;
}

.mode-toggle {
  border: 1px solid #cbd5e1;
  color: #334155;
  background: #ffffff;
  cursor: pointer;
}

.mode-toggle.active {
  border-color: #0f766e;
  color: #ffffff;
  background: #0f766e;
}
</style>
