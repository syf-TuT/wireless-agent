# WirelessAgent 分层Prompt系统实现文档

## 概述

本文档描述了针对 `no_knowledge_base/WA_DS_V3_NKB.py` 实现的专业化分层Prompt系统。该系统将原来简单的固定Prompt升级为动态上下文感知的分层Prompt管理架构。

---

## 修改内容

### 1. 新增文件

**文件位置**: `no_knowledge_base/prompt_manager.py`

该文件实现了完整的分层Prompt管理系统，包含以下功能模块：

- **角色定义层** (Role Definition)
- **专业知识层** (Domain Knowledge)
- **决策框架层** (Decision Framework)
- **动态上下文层** (Dynamic Context)
- **输出格式层** (Output Schema)

### 2. 修改的文件

**文件位置**: `no_knowledge_base/WA_DS_V3_NKB.py`

修改内容：
- 添加导入: `from prompt_manager import prompt_manager, get_prompt_for_context`
- 重构 `initialize()` 函数，根据当前CQI和切片负载动态生成Prompt

---

## Prompt架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    分层Prompt架构                            │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: 角色定义层 (Role Definition)                      │
│  "你是5G/6G无线网络切片管理智能助手"                        │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: 专业知识层 (Domain Knowledge)                    │
│  - 切片类型定义 (eMBB/URLLC/mMTC)                          │
│  - CQI-MCS映射表                                          │
│  - 资源计算公式                                           │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: 决策框架层 (Decision Framework)                  │
│  - 输入解析 → 意图分类 → 资源计算 → 验证输出                │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: 动态上下文层 (Dynamic Context)                   │
│  - 当前CQI状态 / 切片负载 / 资源可用情况                   │
├─────────────────────────────────────────────────────────────┤
│  Layer 5: 输出格式层 (Output Schema)                       │
│  - JSON结构化输出 + 推理过程                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心优势

### 1. 动态上下文感知

| 维度 | 旧版本 | 新版本 |
|------|--------|--------|
| CQI处理 | 固定Prompt | 根据CQI(1-15)动态调整调制方式建议 |
| 负载感知 | 无 | 自动检测切片负载，触发负载均衡 |
| 资源状态 | 无 | 根据资源可用性调整决策策略 |
| 优先级 | 无 | 支持URLLC等业务优先级注入 |

### 2. 结构化输出

**旧版本**:
```
"I recommend using eMBB slice, because user needs high bandwidth for video streaming..."
```

**新版本**:
```json
{
    "reasoning": {
        "step1_intent": "识别关键词'video streaming'，判断为高带宽业务",
        "step2_cqi_analysis": "CQI=12为优秀信道，可支持64QAM-256QAM调制",
        "step3_resource_calc": "4K视频需至少30MHz带宽，当前可用90MHz",
        "step4_slice_selection": "选择eMBB切片，带宽需求匹配且负载适中"
    },
    "decision": {
        "slice_type": "eMBB",
        "bandwidth_mhz": 40,
        "estimated_rate_mbps": 180,
        "modulation": "64QAM",
        "confidence": 0.92,
        "reason": "高带宽视频业务匹配eMBB切片特性"
    },
    "warnings": []
}
```

### 3. 完整专业知识库

新增内容：
- **CQI-MCS映射表**: CQI 1-15 与调制方式的对应关系
- **Shannon公式**: 吞吐量计算公式详解
- **资源块计算**: RB与MHz的转换规则
- **负载均衡策略**: 切片利用率差异>20%时的分配策略

### 4. 推理透明度

4步推理框架：
1. **Step 1 - 意图识别**: 根据用户请求关键词判断业务类型
2. **Step 2 - 信道分析**: 根据CQI确定可用调制方式和最大吞吐量
3. **Step 3 - 资源计算**: 根据业务需求和信道条件计算所需带宽
4. **Step 4 - 切片选择**: 结合负载均衡和资源可用性确定最终切片

---

## 具体案例对比

### 案例1: 远程手术请求 (CQI=3)

**场景**: 用户请求 "remote surgery"，CQI=3

| 维度 | 旧版本 | 新版本 |
|------|--------|--------|
| 调制方式 | 可能推荐64QAM | 自动建议QPSK保证可靠性 |
| 风险 | 信道条件差时连接不稳定 | 适配低阶调制，手术安全 |
| 输出 | 简单文本 | 含推理过程的结构化JSON |

**新版本动态上下文**:
> 当前信道质量较差 (CQI<5)，建议采用低阶调制(BPSK-QPSK)保证传输可靠性
>
> 检测到URLLC业务请求，优先保障低延迟需求

### 案例2: 高负载网络下的视频请求

**场景**: eMBB切片负载85%，新用户请求视频业务

| 维度 | 旧版本 | 新版本 |
|------|--------|--------|
| 切片选择 | 直接分配eMBB | 考虑负载均衡 |
| 结果 | 可能导致拥塞 | 引导至其他切片或压缩带宽 |
| 决策 | 忽略全局状态 | 综合考量网络状态 |

**新版本动态上下文**:
> ⚠️ eMBB切片负载较重(>70%)，需优化资源分配或考虑向其他切片引导

### 案例3: 优秀信道下的高速传输

**场景**: CQI=12，网络资源充足

| 维度 | 旧版本 | 新版本 |
|------|--------|--------|
| 调制方式 | 基础建议 | 自动识别高阶调制 |
| 吞吐量 | 一般 | 最大化利用信道 |
| 输出 | 无置信度 | 包含confidence字段 |

**新版本动态上下文**:
> 当前信道质量优秀 (CQI≥10)，可采用高阶调制(64QAM-256QAM)获得高吞吐量

---

## 使用方法

### 基础使用

```python
from prompt_manager import get_prompt_for_context

# 根据当前网络状态生成动态Prompt
prompt = get_prompt_for_context(
    cqi=12,                                    # 信道质量 (1-15)
    slice_loads={'embb': 45, 'urllc': 30, 'mmtc': 20},  # 各切片负载(%)
    priority='normal',                         # 业务优先级
    resource_status='sufficient'              # 资源状态
)
```

### 在Workflow中使用

系统已自动集成到 `WA_DS_V3_NKB.py` 的 `initialize()` 函数中：

```python
def initialize(state: NetworkState) -> NetworkState:
    # ... 获取网络状态
    dynamic_prompt = get_prompt_for_context(
        cqi=state["cqi"],
        slice_loads=slice_loads,
        priority="normal",
        resource_status=resource_status
    )
    state["history"].append({"role": "system", "content": dynamic_prompt})
    return state
```

---

## 动态上下文模板

### CQI状态相关

| CQI范围 | 注入内容 |
|---------|----------|
| CQI ≥ 10 | 当前信道质量优秀，可采用高阶调制(64QAM-256QAM) |
| CQI 7-9 | 当前信道质量良好，建议采用中阶调制(16QAM-64QAM) |
| CQI 5-6 | 当前信道质量中等，建议采用16QAM调制 |
| CQI < 5 | 当前信道质量较差，建议采用低阶调制(BPSK-QPSK)保证可靠性 |

### 负载状态相关

| 负载情况 | 注入内容 |
|----------|----------|
| eMBB > 70% | ⚠️ eMBB切片负载较重，需优化资源分配 |
| URLLC > 70% | ⚠️ URLLC切片负载较重，需注意低延迟保障 |
| 全部 < 70% | 各切片负载均衡，可正常分配 |

### 资源状态相关

| 资源状态 | 注入内容 |
|----------|----------|
| sufficient | 当前网络资源充足，可满足业务需求 |
| limited | ⚠️ 网络资源有限，可能需要调整带宽分配 |
| critical | 🔴 网络资源紧张，需要启用压缩策略 |

---

## 文件清单

| 文件路径 | 说明 |
|----------|------|
| `no_knowledge_base/prompt_manager.py` | 新增：分层Prompt管理系统 |
| `no_knowledge_base/WA_DS_V3_NKB.py` | 修改：集成动态Prompt |

---

## 测试验证

```bash
cd no_knowledge_base

# 测试1: 导入验证
python -c "from prompt_manager import prompt_manager; print('OK')"

# 测试2: 端到端测试
python -c "
from WA_DS_V3_NKB import process_user_request
result = process_user_request(
    user_id='test_001',
    location='building_A',
    request='I need to stream 4K video',
    cqi=12,
    ground_truth='eMBB'
)
print('Test passed!')
"
```

---

## 后续优化方向

1. **RAG集成**: 结合历史决策经验优化Prompt
2. **多模态扩展**: 支持信道图像输入
3. **反馈学习**: 基于决策结果持续优化Prompt模板
4. **性能监控**: 添加Prompt效果评估指标
