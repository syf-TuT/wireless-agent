<template>
  <div class="version-tag" :class="{ 'with-kb': useKnowledgeBase }" @click="toggleKnowledgeBase">
    <el-icon class="tag-icon">
      <info-filled v-if="!useKnowledgeBase" />
      <collection-tag v-else />
    </el-icon>
    <span class="tag-text">{{ useKnowledgeBase ? '有知识库版本' : '无知识库版本' }}</span>
    <div class="tag-shine"></div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { InfoFilled, CollectionTag } from '@element-plus/icons-vue'

const emit = defineEmits<{
  (e: 'update:useKnowledgeBase', value: boolean): void
}>()

const useKnowledgeBase = ref(false)

const toggleKnowledgeBase = () => {
  useKnowledgeBase.value = !useKnowledgeBase.value
  emit('update:useKnowledgeBase', useKnowledgeBase.value)
}
</script>

<style scoped>
.version-tag {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 20px;
  border-radius: 28px;
  background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%) !important;
  background-color: transparent !important;
  border: 2px solid #cbd5e1 !important;
  color: #64748b !important;
  font-weight: 700;
  font-size: 13px;
  letter-spacing: 0.5px;
  backdrop-filter: blur(12px);
  position: relative;
  overflow: hidden;
  box-shadow: 0 4px 16px rgba(203, 213, 225, 0.3);
  transition: all 0.3s ease;
  cursor: pointer;
  z-index: 10;
}

.version-tag.with-kb {
  background: linear-gradient(135deg, #c084fc 0%, #a78bfa 100%) !important;
  background-color: transparent !important;
  border: 2px solid #a78bfa !important;
  color: #ffffff !important;
  box-shadow: 0 4px 16px rgba(167, 139, 250, 0.4);
}

.version-tag:hover {
  transform: translateY(-2px) scale(1.05);
  box-shadow: 0 8px 24px rgba(203, 213, 225, 0.4);
  background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%) !important;
}

.version-tag.with-kb:hover {
  transform: translateY(-2px) scale(1.05);
  box-shadow: 0 8px 24px rgba(167, 139, 250, 0.5);
  background: linear-gradient(135deg, #c084fc 0%, #a78bfa 100%) !important;
}

.tag-icon {
  font-size: 16px;
  animation: iconBounce 2s ease-in-out infinite;
  position: relative;
  z-index: 2;
}

.tag-text {
  position: relative;
  z-index: 2;
}

.tag-shine {
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
  transition: left 0.6s ease;
  z-index: 1;
}

.version-tag:hover .tag-shine {
  left: 100%;
}

@keyframes iconBounce {
  0%,
  100% {
    transform: translateY(0);
  }

  50% {
    transform: translateY(-4px);
  }
}
</style>
