# RAG优势验证实验设计

## 1. 实验目标与约束

| 项目 | 说明 |
|------|------|
| **实验目标** | 验证使用RAG（检索增强生成）相比不使用RAG在意图分类和资源利用方面的优势 |
| **响应变量 (Y)** | Y1: Intent Correct Rate (意图正确率) <br> Y2: 资源利用率 <br> Y3: Token消耗 |
| **实验预算** | 4组实验，每组≥30个请求 |
| **成功标准** | RAG在Y1显著优于非RAG (p<0.05) |

---

## 2. 因素与水平

### 控制因素

| 因素 | 符号 | 水平 |
|------|------|------|
| RAG使用 | A | [有, 无] |
| LLM模型 | B | [DeepSeek-V3.2, MiniMax-M2.5] |

### 实验组设计 (2×2 = 4组)

| 组别 | A(RAG) | B(模型) | 样本来源 |
|------|--------|---------|----------|
| 1 | ✓ | DeepSeek-V3.2 | `with_knowledge_base/DSv3KB.csv` |
| 2 | ✓ | MiniMax-M2.5 | `run_results/with_kb/...minimax.csv` |
| 3 | ✗ | DeepSeek-V3.2 | `no_knowledge_base/DSv3NKB.csv` |
| 4 | ✗ | MiniMax-M2.5 | 需运行实验 |

---

## 3. 实验设计选择

**完全因子设计 (Full Factorial Design)**
- 2因素 × 2水平 = 4组
- 每组最小样本量: 30个请求
- 总计最小实验量: 120个请求

---

## 4. 响应变量定义

| 响应 | 符号 | 测量方式 | 预期改进方向 |
|------|------|----------|--------------|
| 意图正确率 | Y1 | `Intent Correct`列 Yes/Total | RAG > 非RAG |
| 资源利用率 | Y2 | `Avg Resource Util After %` 均值 | 对比分析 |
| Token消耗 | Y3 | API调用token总数 | RAG < 非RAG |

---

## 5. 统计分析方法

| 响应变量 | 统计检验方法 | 效应量 |
|----------|--------------|--------|
| Y1: 意图正确率 | 卡方检验 / Fisher精确检验 | Cohen's h |
| Y2: 资源利用率 | 独立样本t检验 | Cohen's d |
| Y3: Token消耗 | 独立样本t检验 | Cohen's d |

### 假设检验

```
H0 (零假设): RAG与非RAG在意图正确率上无显著差异
H1 (备择假设): RAG在意图正确率上显著优于非RAG

显著性水平: α = 0.05
检验功效: 1 - β ≥ 0.80
```

---

## 6. 实验设计矩阵

| Run ID | A(RAG) | B(模型) | 预期Y1 | 预期Y2 | 预期Y3 | 数据状态 |
|--------|--------|---------|--------|--------|--------|----------|
| 1 | 1 (Yes) | 1 (DeepSeek) | 高 | 待分析 | 待分析 | 已有 |
| 2 | 1 (Yes) | 2 (MiniMax) | 高 | 待分析 | 待分析 | 已有 |
| 3 | 0 (No) | 1 (DeepSeek) | 中 | 待分析 | 待分析 | 已有 |
| 4 | 0 (No) | 2 (MiniMax) | 中 | 待分析 | 待分析 | 需运行 |

---

## 7. 执行计划

### Phase 1: 现有数据分析 (已完成部分)

- [x] 组1: 读取 `with_knowledge_base/network_slicing_results_DSv3KB.csv`
- [x] 组2: 读取 `run_results/with_kb/network_slicing_results_TJU_north_minimax-M2.5.csv`
- [x] 组3: 读取 `no_knowledge_base/network_slicing_results_DSv3NKB.csv`

### Phase 2: 补充实验 (待执行)

- [ ] 组4: 运行无RAG + MiniMax-M2.5模型
- [ ] 所有组: 添加token消耗记录

### Phase 3: 统计分析与报告

- [ ] 计算各组Y1, Y2, Y3描述统计
- [ ] 执行假设检验
- [ ] 计算效应量
- [ ] 撰写实验报告

---

## 8. 预期结论

| 假设 | 预期 | 理论依据 |
|------|------|----------|
| H1: Y1 (意图正确率) | RAG > 非RAG | RAG提供领域知识示例，减少LLM推理错误 |
| H2: Y2 (资源利用率) | 对比分析 | 可能提高资源分配合理性 |
| H3: Y3 (Token消耗) | RAG < 非RAG | RAG只检索相关知识，而非加载全部知识库 |

---

## 9. 实验执行脚本

实验执行与分析使用 `experiment_rag_comparison.py` 脚本:

```bash
# 运行实验分析
python experiment_rag_comparison.py

# 参数说明
# --rag-data: RAG模式结果CSV路径
# --no-rag-data: 非RAG模式结果CSV路径
# --output: 输出报告路径
```

---

## 10. 参考文献

- Montgomery, D. C. (2017). Design and Analysis of Experiments. Wiley.
- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
