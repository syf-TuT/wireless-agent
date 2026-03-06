"""
Wireless Agent Prompt Manager - 分层Prompt管理系统

提供完整的分层Prompt设计：
- Layer 1: 角色定义 (Role Definition)
- Layer 2: 专业知识 (Domain Knowledge)
- Layer 3: 决策框架 (Decision Framework)
- Layer 4: 动态上下文 (Dynamic Context)
- Layer 5: 输出格式 (Output Schema)
"""

from typing import Dict, Any, Optional
import json


# =============================================================================
# Layer 1: 角色定义 (Role Definition)
# =============================================================================

ROLE_DEFINITION = """你是5G/6G无线网络切片管理智能助手，专注于网络资源优化分配。

你的核心能力：
1. 用户意图分类：精准识别业务类型 (eMBB/URLLC/mMTC)
2. 信道质量分析：基于CQI计算最优资源分配方案
3. 网络切片决策：生成切片类型和带宽分配策略
4. 推理透明度：输出详细的决策推理过程"""


# =============================================================================
# Layer 2: 专业知识 (Domain Knowledge)
# =============================================================================

SLICE_TYPE_KNOWLEDGE = """## 网络切片类型定义

### eMBB (增强移动宽带)
- 带宽范围: 90 MHz (总容量)
- 适用场景: 视频流媒体、AR/VR、高清视频通话、大文件下载
- 延迟要求: 40-50ms
- 典型业务: 4K/8K视频、在线游戏、云游戏

### URLLC (超可靠低延迟通信)
- 带宽范围: 30 MHz (总容量)
- 适用场景: 工业控制、自动驾驶、远程医疗、实时监控
- 延迟要求: <5ms
- 典型业务: 远程手术、车联网V2X、工业自动化

### mMTC (大规模机器类通信)
- 带宽范围: 10 MHz (总容量)
- 适用场景: 物联网传感器、智能城市、环境监测
- 延迟要求: 可变 (100-1000ms)
- 典型业务: 智能电表、环境传感器、资产追踪"""


CQI_MCS_MAPPING = """## CQI与调制方式映射表

| CQI范围 | 调制方式 | 编码效率 | 适用信道 |
|---------|----------|----------|----------|
| CQI 1-2 | BPSK | 0.15 | 极差 |
| CQI 3-4 | QPSK | 0.38 | 较差 |
| CQI 5-6 | 16QAM | 0.60 | 中等 |
| CQI 7-9 | 64QAM | 0.75 | 良好 |
| CQI 10-12 | 64QAM | 0.90 | 优秀 |
| CQI 13-15 | 256QAM | 0.95 | 极佳 |

说明: CQI (Channel Quality Indicator) 范围1-15，数值越高表示信道质量越好。
高CQI值允许使用高阶调制方式，在相同带宽下获得更高吞吐量。"""


RESOURCE_CALCULATION = """## 资源计算公式

### 吞吐量计算 (Shannon公式)
rate = bandwidth_MHz * log2(1 + 10^(CQI/10)) * 10

### 资源块(Resource Block)计算
- 每RB带宽: 180 kHz
- 所需RB数 = ceil(所需带宽 / 0.18)

### 带宽分配规则
- eMBB: 最小6 MHz, 最大90 MHz
- URLLC: 最小1 MHz, 最大30 MHz
- mMTC: 最小1 MHz, 最大10 MHz

### 负载均衡策略
当切片利用率差异>20%时，优先分配到利用率较低的切片"""


# =============================================================================
# Layer 3: 决策框架 (Decision Framework)
# =============================================================================

DECISION_FRAMEWORK = """## 决策框架

### 输入处理流程
1. 解析用户业务需求，提取关键词
2. 提取信道质量指标 (CQI)
3. 识别用户位置和服务类型
4. 获取当前网络负载状态

### 决策流程 (4步推理)
Step 1 - 意图识别: 根据用户请求中的业务关键词判断业务类型
Step 2 - 信道分析: 根据CQI值确定可用的调制方式和最大吞吐量
Step 3 - 资源计算: 根据业务需求和信道条件计算所需带宽
Step 4 - 切片选择: 结合负载均衡和资源可用性确定最终切片类型

### 约束检查
- 总带宽不超过切片可用容量
- 切片数量不超过3个
- 优先保障高优先级业务需求
- 考虑资源利用率最大化"""


# =============================================================================
# Layer 4: 动态上下文模板 (Dynamic Context Templates)
# =============================================================================

CONTEXT_TEMPLATES = {
    # CQI状态相关
    "cqi_excellent": "当前信道质量优秀 (CQI≥10)，可采用高阶调制(64QAM-256QAM)获得高吞吐量",
    "cqi_good": "当前信道质量良好 (CQI 7-9)，建议采用中阶调制(16QAM-64QAM)",
    "cqi_fair": "当前信道质量中等 (CQI 5-6)，建议采用16QAM调制，平衡效率与可靠性",
    "cqi_poor": "当前信道质量较差 (CQI<5)，建议采用低阶调制(BPSK-QPSK)保证传输可靠性",

    # 负载状态相关
    "embb_heavy": "⚠️ eMBB切片负载较重(>70%)，需优化资源分配或考虑向其他切片引导",
    "urllc_heavy": "⚠️ URLLC切片负载较重(>70%)，需注意低延迟保障",
    "mmtc_heavy": "⚠️ mMTC切片负载较重(>70%)，需注意连接数限制",
    "balanced": "各切片负载均衡，可正常分配",

    # 优先级相关
    "urllc_priority": "🔴 检测到URLLC业务请求，优先保障低延迟需求",
    "embb_high_priority": "🟠 检测到高带宽需求，优先保障吞吐量",
    "mmtc_low_priority": "🟢 检测到mMTC业务，可在资源紧张时压缩",

    # 资源相关
    "resource_sufficient": "当前网络资源充足，可满足业务需求",
    "resource_limited": "⚠️ 网络资源有限，可能需要调整带宽分配",
    "resource_critical": "🔴 网络资源紧张，需要启用压缩策略"
}


# =============================================================================
# Layer 5: 输出格式 (Output Schema)
# =============================================================================

OUTPUT_SCHEMA = """## 输出格式要求

### 结构化JSON输出格式
请严格按照以下JSON格式输出决策结果：

```json
{
    "reasoning": {
        "step1_intent": "意图识别结果: ...",
        "step2_cqi_analysis": "信道分析结果: ...",
        "step3_resource_calc": "资源计算结果: ...",
        "step4_slice_selection": "切片选择结果: ..."
    },
    "decision": {
        "slice_type": "eMBB|URLLC|mMTC",
        "bandwidth_mhz": 数值,
        "estimated_rate_mbps": 数值,
        "modulation": "调制方式",
        "confidence": 0.0-1.0,
        "reason": "简要决策理由"
    },
    "warnings": ["警告信息列表(如有)"]
}
```

### 推理过程要求
1. 每个推理步骤必须有明确的依据
2. 引用具体的参数值(CQI、带宽、容量等)
3. 说明选择该切片类型的具体原因

### 示例输出
```json
{
    "reasoning": {
        "step1_intent": "识别到关键词'video streaming'，判断为高带宽业务",
        "step2_cqi_analysis": "CQI=12为优秀信道，可支持64QAM-256QAM调制",
        "step3_resource_calc": "4K视频需至少30MHz带宽，当前可用90MHz，满足需求",
        "step4_slice_selection": "选择eMBB切片，带宽需求匹配且当前负载适中(45%)"
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
```"""


# =============================================================================
# PromptManager 类
# =============================================================================

class PromptManager:
    """分层Prompt管理器"""

    def __init__(self):
        self.layers = {
            "role": ROLE_DEFINITION,
            "knowledge": SLICE_TYPE_KNOWLEDGE + "\n\n" + CQI_MCS_MAPPING + "\n\n" + RESOURCE_CALCULATION,
            "framework": DECISION_FRAMEWORK,
            "output": OUTPUT_SCHEMA
        }
        self.context_templates = CONTEXT_TEMPLATES

    def get_base_prompt(self) -> str:
        """获取基础Prompt(不含动态上下文)"""
        return "\n\n".join([
            self.layers["role"],
            self.layers["knowledge"],
            self.layers["framework"],
            self.layers["output"]
        ])

    def build_context_aware_prompt(self, context: Dict[str, Any]) -> str:
        """
        构建带动态上下文的完整Prompt

        Args:
            context: 包含以下键的字典:
                - cqi: int, 信道质量指示(1-15)
                - slice_loads: dict, 各切片负载情况 {"embb": float, "urllc": float, "mmtc": float}
                - priority: str, 业务优先级 "urllc"|"embb"|"mmtc"|"normal"
                - resource_status: str, 资源状态 "sufficient"|"limited"|"critical"

        Returns:
            str: 完整的Prompt文本
        """
        # 构建动态上下文
        context_sections = []

        # CQI状态
        cqi = context.get("cqi", 7)
        if cqi >= 10:
            context_sections.append(self.context_templates["cqi_excellent"])
        elif cqi >= 7:
            context_sections.append(self.context_templates["cqi_good"])
        elif cqi >= 5:
            context_sections.append(self.context_templates["cqi_fair"])
        else:
            context_sections.append(self.context_templates["cqi_poor"])

        # 负载状态
        slice_loads = context.get("slice_loads", {})
        if slice_loads.get("embb", 0) > 70:
            context_sections.append(self.context_templates["embb_heavy"])
        if slice_loads.get("urllc", 0) > 70:
            context_sections.append(self.context_templates["urllc_heavy"])
        if slice_loads.get("mmtc", 0) > 70:
            context_sections.append(self.context_templates["mmtc_heavy"])
        if not any(load > 70 for load in slice_loads.values()):
            context_sections.append(self.context_templates["balanced"])

        # 优先级
        priority = context.get("priority", "normal")
        if priority == "urllc":
            context_sections.append(self.context_templates["urllc_priority"])
        elif priority == "embb":
            context_sections.append(self.context_templates["embb_high_priority"])
        elif priority == "mmtc":
            context_sections.append(self.context_templates["mmtc_low_priority"])

        # 资源状态
        resource_status = context.get("resource_status", "sufficient")
        if resource_status == "limited":
            context_sections.append(self.context_templates["resource_limited"])
        elif resource_status == "critical":
            context_sections.append(self.context_templates["resource_critical"])
        else:
            context_sections.append(self.context_templates["resource_sufficient"])

        # 构建完整Prompt
        base_prompt = self.get_base_prompt()
        context_section = "## 当前网络状态\n" + "\n".join(context_sections)

        return f"{base_prompt}\n\n{context_section}"

    def extract_decision(self, llm_response: str) -> Dict[str, Any]:
        """
        从LLM响应中提取结构化决策

        Args:
            llm_response: LLM返回的文本响应

        Returns:
            dict: 包含reasoning和decision的字典
        """
        import re

        result = {
            "reasoning": {},
            "decision": {},
            "warnings": [],
            "raw_response": llm_response
        }

        # 尝试提取JSON
        json_match = re.search(r'\{[\s\S]*\}', llm_response)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                result.update(parsed)
                return result
            except json.JSONDecodeError:
                pass

        # 尝试从文本中提取关键信息
        slice_match = re.search(r'(eMBB|URLLC|mMTC)', llm_response, re.IGNORECASE)
        if slice_match:
            result["decision"]["slice_type"] = slice_match.group().upper()

        bandwidth_match = re.search(r'(\d+)\s*MHz', llm_response, re.IGNORECASE)
        if bandwidth_match:
            result["decision"]["bandwidth_mhz"] = int(bandwidth_match.group(1))

        return result


# 创建全局Prompt管理器实例
prompt_manager = PromptManager()


# =============================================================================
# 便捷函数
# =============================================================================

def get_prompt_for_context(cqi: int, slice_loads: Dict[str, float],
                           priority: str = "normal",
                           resource_status: str = "sufficient") -> str:
    """
    获取指定上下文下的Prompt

    Args:
        cqi: 信道质量指示(1-15)
        slice_loads: 各切片负载百分比
        priority: 业务优先级
        resource_status: 资源状态

    Returns:
        str: 完整的Prompt文本
    """
    context = {
        "cqi": cqi,
        "slice_loads": slice_loads,
        "priority": priority,
        "resource_status": resource_status
    }
    return prompt_manager.build_context_aware_prompt(context)


def get_base_prompt() -> str:
    """获取基础Prompt(不含动态上下文)"""
    return prompt_manager.get_base_prompt()
