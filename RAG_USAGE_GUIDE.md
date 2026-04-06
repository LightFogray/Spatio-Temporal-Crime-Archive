# RAG增强环境犯罪学语义生成 使用指南

## 概述

本系统通过构建**纯净的环境犯罪学知识库**（排除个体/群体偏见），使用RAG（检索增强生成）技术增强LLM的语义描述生成。

## 核心组件

| 文件 | 功能 |
|------|------|
| `src/env_criminology_kb.py` | 环境犯罪学知识库，包含9类场所类型的环境风险机制 |
| `src/rag_semantic_generator.py` | RAG增强生成器，支持检索-生成-审查流程 |
| `src/generate_semantic_embedding.py` | 主程序，集成RAG和基础两种模式 |

## 知识库内容

知识库包含以下场所类型的**纯环境视角**风险分析：

1. **高密度商业区** - 目标集中、人流混杂、暴露窗口
2. **夜间娱乐场所** - 时间真空、照明不足、冲突氛围
3. **交通枢纽** - 匿名性、监控盲区、地下空间
4. **纯住宅区** - 日间真空、绿化遮挡、巡逻不足
5. **混合用途住宅** - 边界模糊、过境人流
6. **工业区** - 周末真空、高价值目标、围墙封闭
7. **公园/开放空间** - 视线遮挡、入口过多、设计-安全张力
8. **学校周边** - 时段集中、边界盲区、快速真空
9. **高连通性区域** - 近重复效应、网络传播

**所有条目已通过偏见审查，排除人口统计、社会经济刻板印象。**

## 使用方法

### 1. 首次使用：构建知识库

```bash
cd src
python env_criminology_kb.py
```

这将创建 `data/knowledge_base/env_criminology_kb.json`

### 2. 生成RAG增强语义描述

```bash
# RAG模式（推荐）
python generate_semantic_embedding.py --mode rag

# 基础模式（原有逻辑）
python generate_semantic_embedding.py --mode basic

# 强制重新生成（忽略缓存）
python generate_semantic_embedding.py --mode rag --force
```

### 3. 输出文件

| 模式 | 文本文件 | 嵌入文件 |
|------|---------|---------|
| RAG | `semantic_texts_rag.json` | `semantic_embedding_rag.npy` |
| Basic | `semantic_texts_v2.json` | `semantic_embedding_v2.npy` |

## RAG流程详解

```
输入区域特征
    ↓
[1] 检索知识库
    - 结构化匹配：特征相似度计算
    - 语义匹配：BGE-M3嵌入相似度
    ↓
[2] 构建约束Prompt
    - 注入最匹配的环境风险机制
    - 提供监护策略示例
    - 明确禁止偏见词汇
    ↓
[3] LLM生成
    - 在知识约束下生成描述
    - 温度降至0.3降低随机性
    ↓
[4] 偏见审查
    - 检查禁止词汇（种族、阶层等）
    - 验证环境术语存在性
    ↓
[5] 输出/回退
    - 通过审查：输出描述
    - 未通过：使用保守模板生成
```

## 偏见审查机制

### 禁止的词汇/概念

```python
PROHIBITED_PATTERNS = {
    "demographic": ["black", "white", "hispanic", "asian", "immigrant", ...],
    "socioeconomic": ["poor people", "ghetto", "broken home", "welfare", ...],
    "pathology": ["criminal mind", "deviant", "anti-social", ...],
    "spatial_proxy": ["bad neighborhood because of people", ...]
}
```

### 必须通过的环境术语

```python
REQUIRED_ENVIRONMENTAL_TERMS = [
    "lighting", "surveillance", "guardianship", "density",
    "access", "design", "exposure", "environment"
]
```

## 对比：RAG vs Basic

| 维度 | RAG模式 | 基础模式 |
|------|---------|---------|
| **理论基础** | 结构化环境犯罪学知识 | 开放式LLM推断 |
| **幻觉风险** | 低（知识约束） | 较高 |
| **偏见控制** | 严格审查+模板回退 | 依赖Prompt约束 |
| **生成速度** | 较慢（检索+审查） | 较快 |
| **描述一致性** | 高（知识模板化） | 较低 |
| **适用场景** | 期刊论文（严谨性优先） | 快速实验 |

## 集成到训练流程

修改训练脚本以使用RAG嵌入：

```python
# train_stgcn_trans.py 或你的训练脚本

# 加载RAG语义嵌入
semantic_embed_path = "data/processed/semantic_embedding_rag.npy"
if os.path.exists(semantic_embed_path):
    semantic_embed = np.load(semantic_embed_path)
    print(f"Loaded RAG semantic embedding: {semantic_embed.shape}")
else:
    # 回退到基础版本
    semantic_embed = np.load("data/processed/semantic_embedding_v2.npy")
```

## 验证RAG效果

### 1. 检查知识库匹配质量

```python
from rag_semantic_generator import RAGSemanticGenerator

gen = RAGSemanticGenerator(use_rag=True)

# 测试区域
test_features = {
    'poi_commercial': 0.85,
    'camera_coverage': 0.2,
    'nightlight': 0.3
}

result = gen.generate_semantic_description(0, test_features, "mild")

print("匹配的场所类型:", result['rag_context']['matched_place_type'])
print("相似度分数:", result['rag_context']['similarity_score'])
print("偏见审查:", "通过" if result['bias_audit']['passed'] else "失败")
```

### 2. 对比RAG与基础模式输出

```bash
# 生成两种模式的描述
python generate_semantic_embedding.py --mode rag
python generate_semantic_embedding.py --mode basic

# 对比同一区域的描述
python -c "
import json
with open('data/processed/semantic_texts_rag.json') as f:
    rag = json.load(f)['texts']
with open('data/processed/semantic_texts_v2.json') as f:
    basic = json.load(f)['texts']

# 打印前3个对比
for i in range(3):
    print(f'=== Region {i} ===')
    print(f'RAG: {rag[i]}')
    print(f'Basic: {basic[i]}')
    print()
"
```

## 扩展知识库

如需添加新的场所类型，编辑 `env_criminology_kb.py`：

```python
default_entries.append({
    "entry_id": "your_new_type",
    "place_type": "descriptive_name",
    "environmental_features": {
        "poi_commercial": 0.5,
        # ... 特征归一化值
    },
    "risk_mechanism": "纯环境视角的风险机制描述...",
    "guardianship_strategies": [
        "策略1",
        "策略2"
    ]
})
```

重新运行 `env_criminology_kb.py` 更新知识库。

## 故障排除

| 问题 | 解决方案 |
|------|---------|
| RAG生成太慢 | 减少`top_k`参数（默认3），或增加`MAX_WORKERS` |
| 知识库匹配差 | 检查特征归一化是否一致 |
| 偏见审查失败率高 | 降低LLM温度，或检查知识库模板 |
| Ollama连接失败 | 确保Ollama服务在`localhost:11434`运行 |

## 学术引用建议

使用RAG系统时，论文可表述为：

> "为增强语义生成的理论一致性与偏见控制，本研究构建了基于环境犯罪学CPTED理论的结构化知识库，采用RAG（Retrieval-Augmented Generation）框架约束LLM输出。知识库严格限定于物理环境因素（照明、监控、可达性），排除人口统计与社会经济偏见，所有生成结果经自动化偏见审查，未通过审查的样本回退至保守模板生成。"

## 伦理声明

本知识库遵循以下原则：

1. **环境决定论边界**：只讨论物理环境对犯罪机会的影响
2. **无个体归因**：不将犯罪风险归因于特定人群特征
3. **政策导向**：建议的环境干预措施聚焦于设计改进
4. **透明审查**：所有知识条目经偏见审计，拒绝流程可追溯
