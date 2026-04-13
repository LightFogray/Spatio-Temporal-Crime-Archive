# 期刊论文实验设计方案 (保守改进版)

> **架构版本**: Adaptive Environment-Complexity Routing (ACR-ST)
> **核心创新**: 自适应专家融合 + CPTED知识约束 + RAG增强语义

---

## 一、Baseline设计（6-8个）

### 1.1 传统统计/机器学习方法（2个）

| 模型 | 简称 | 选择理由 |
|------|------|---------|
| Historical Average | HA | 最简单的时间序列基线，验证序列模式是否可被学习 |
| Random Forest | RF | 经典的非线性集成方法，犯罪预测领域的常用baseline |

### 1.2 深度时空学习方法（4-5个）

| 模型 | 简称 | 类型 | 选择理由 |
|------|------|------|---------|
| ST-GCN | STGCN | 时空图卷积 | 犯罪预测领域最主流的方法 |
| DCRNN | DCRNN | 图扩散RNN | 扩散卷积更适合犯罪传播建模 |
| Graph WaveNet | GWN | 自适应图 | 可学习自适应邻接矩阵，与你的工作相关 |
| ST-Transformer | STT | 纯Transformer | 验证去掉GCN仅用注意力的效果 |
| ConvLSTM | ConvLSTM | 纯CNN-RNN | 验证图结构是否必要 |

### 1.3 语义增强与差异化策略方法（3个）

| 模型 | 简称 | 选择理由 |
|------|------|---------|
| ST-GCN + Basic LLM | STGCN-LLM | 对比开放式LLM生成 vs RAG约束生成 |
| Hard-Partition by Density | HP-ST | 基于犯罪密度硬划分的差异化策略（对比软路由） |
| Uniform Fusion | UF-ST | 三层专家等权重融合（验证自适应加权的必要性） |
| **Ours (ACR-ST)** | **ACR-ST** | 自适应环境复杂度路由 + CPTED知识约束 |

---

## 二、消融实验设计（Ablation Study）

### 2.1 主消融实验（验证自适应专家融合各组件）

| 变体 | 配置 | 验证问题 |
|------|------|---------|
| **Full Model (ACR-ST)** | 完整自适应专家融合 + CPTED知识 | 基准性能 |
| w/o Adaptive Routing | 硬划分替代软路由（低/中/高阈值） | 软路由是否优于硬划分？ |
| w/o CPTED Knowledge | 移除Expert 2（仅基础+语义两层） | CPTED知识增强是否必要？ |
| w/o Semantic | 仅Expert 0（基础编码器） | LLM语义增强是否必要？ |
| Single Expert Only | 单一专家无路由机制 | 自适应路由机制本身的价值？ |
| w/o Near-Repeat | 移除近重复效应模块 | 犯罪学理论建模是否必要？ |
| w/o Hypergraph | 移除超图注意力 | 功能区聚类是否提升性能？ |

### 2.2 自适应路由机制消融（验证软路由设计）

| 变体 | 路由权重设计 | 验证问题 |
|------|-------------|---------|
| **Ours (Quadratic)** | $w_0=(1-c)^2, w_1=2c(1-c), w_2=c^2$ | 抛物线权重（中复杂度侧重语义） |
| Linear Routing | $w_0=1-c, w_1=0.5, w_2=c$ | 线性插值 vs 非线性 |
| Sharp Threshold | $w \in \{[1,0,0], [0,1,0], [0,0,1]\}$ | 硬切换 vs 软融合 |
| Uniform Weight | $w_0=w_1=w_2=1/3$ | 等权重 vs 自适应 |

### 2.3 组件替换消融（验证设计选择）

| 变体 | 替换为 | 验证问题 |
|------|--------|---------|
| Complexity Scorer → Random | 随机分配复杂度分数 | 环境复杂度评估是否必要？ |
| CPTED → POI-only | 仅使用POI密度替代CPTED四维度 | CPTED理论框架的价值？ |
| Near-Repeat → Fixed | 固定参数（α=0.001, β=0.3） | 可学习参数 vs 理论固定值 |
| ZINB → MSE | 均方误差损失 | 零膨胀分布是否必要？ |

---

## 三、对照实验设计（Controlled Experiments）

### 3.1 差异化策略对照（核心对照）

验证自适应环境复杂度路由 vs 其他差异化策略：

| 变体 | 策略设计 | 验证问题 |
|------|---------|---------|
| **Ours (ACR-ST)** | 环境复杂度软路由（$c \in [0,1]$连续） | 完整系统 |
| Hard-Partition (Density) | 基于犯罪密度硬划分（低/中/高三类） | 环境复杂度 vs 犯罪密度作为划分依据？ |
| Hard-Partition (Random) | 随机划分三类区域 | 硬划分本身是否有价值？ |
| Single-Scale | 所有区域使用相同增强级别 | 差异化策略是否必要？ |

### 3.2 RAG语义生成对照

验证RAG知识库约束的有效性：

| 变体 | 语义生成方式 | 验证问题 |
|------|-------------|---------|
| **Ours (RAG-Adaptive)** | RAG检索 + 知识约束 + 自适应路由 | 完整系统 |
| Basic LLM (Open) | 开放式LLM生成，无知识约束 | RAG约束是否减少幻觉？ |
| Template-Only | 固定模板，无LLM | LLM是否优于固定模板？ |
| RAG w/o Audit | RAG检索但无偏见审查 | 偏见审查是否必要？ |
| Generic Prompt | 通用描述（"这是一个有X个POI的区域"） | 犯罪学理论术语是否有价值？ |
| Short Description | 极简描述（1句话） | 描述详细程度的影响 |

### 3.3 CPTED知识嵌入对照

| 变体 | CPTED嵌入方式 | 验证问题 |
|------|--------------|---------|
| **Ours (CPTED-4D)** | 四维度得分（监护/入口/领域/活动） | 完整设计 |
| CPTED-Binary | 仅高风险/低风险二分类 | 细粒度四维度 vs 粗粒度二分类 |
| POI-Density Only | 用POI密度替代CPTED | 理论框架 vs 简单特征 |
| No Environmental Knowledge | 移除所有环境设计知识 | 环境知识本身的价值？ |

### 3.4 近重复效应建模对照

| 变体 | 建模方式 | 验证问题 |
|------|---------|---------|
| Ours | 可学习时空衰减参数 | 你的设计 |
| Fixed Gaussian | 固定高斯核（σ_space=400m, σ_time=7d） | 经典近重复理论参数是否最优？ |
| Exponential Decay | 指数衰减替代高斯 | 衰减函数形式的影响 |
| OD-modulated | 移除OD流调制 | 人流因素是否必要？ |
| Separate Effects | 空间/时间效应分别建模后相加 | 联合建模 vs 分离建模 |

### 3.5 图结构对照

| 变体 | 图结构 | 验证问题 |
|------|--------|---------|
| Ours | 空间+距离+犯罪+超图 | 完整设计 |
| Spatial Only | 仅空间邻接图 | 距离衰减是否必要？ |
| Distance Only | 仅距离图 | 拓扑结构是否必要？ |
| No Hypergraph | 移除超图（功能区聚类） | 土地利用语义聚类是否有价值？ |

---

## 四、评价指标体系

### 4.1 预测精度指标

| 指标 | 符号 | 说明 |
|------|------|------|
| Root Mean Square Error | RMSE | 整体误差 |
| Mean Absolute Error | MAE | 绝对误差 |
| Mean Absolute Percentage Error | MAPE | 百分比误差 |

### 4.2 实战应用指标（犯罪预测领域核心）

| 指标 | 符号 | 说明 |
|------|------|------|
| Prediction Accuracy Index | PAI | 预测效率 = (犯罪在Top-K热点中被捕获的比例) / (热点面积占比) |
| Hit Rate | HR@K% | Top-K命中率 = 预测Top-K ∩ 真实犯罪网格 / K |
| Jaccard Index | Jaccard | 集合相似度 = |Pred ∩ True| / |Pred ∪ True| |
| Adjusted Rand Index | ARI | 聚类相似度 |

### 4.3 可解释性指标（新增）

| 指标 | 符号 | 说明 |
|------|------|------|
| Mean Complexity Score | $\bar{c}$ | 平均环境复杂度（0-1） |
| Expert Weight Entropy | $H(w)$ | 专家权重分布的熵（反映路由确定性） |
| CPTED Coverage | - | CPTED四维度得分的覆盖率 |
| Bias Audit Pass Rate | - | RAG生成通过偏见审查的比例 |

### 4.4 效率指标

| 指标 | 说明 |
|------|------|
| Inference Time | 单次预测耗时（毫秒） |
| Parameter Count | 模型参数量 |
| Memory Usage | GPU显存占用 |
| FLOPs | 浮点运算次数 |

---

## 五、统计显著性检验

### 5.1 配对t检验
```python
from scipy import stats
t_stat, p_value = stats.ttest_rel(our_results, baseline_results)
# p < 0.05 表示显著优于baseline
```

### 5.2 效应量（Effect Size）
计算Cohen's d：
- d = 0.2 小效应
- d = 0.5 中等
- d = 0.8 大效应

### 5.3 多重比较校正
使用Bonferroni校正或FDR控制，应对多重假设检验。

---

## 六、实验结果表格模板

### 表1：与Baseline对比（主要结果表）

| Model | RMSE↓ | MAE↓ | PAI↑ | HR@10%↑ | Jaccard↑ | $\bar{c}$ | Params |
|-------|-------|------|------|---------|----------|----------|--------|
| HA | - | - | - | - | - | - | - |
| RF | - | - | - | - | - | - | - |
| STGCN | - | - | - | - | - | - | - |
| DCRNN | - | - | - | - | - | - | - |
| GWN | - | - | - | - | - | - | - |
| STT | - | - | - | - | - | - | - |
| HP-ST | - | - | - | - | - | - | - |
| **ACR-ST (Ours)** | **-** | **-** | **-** | **-** | **-** | **-** | - |
| Improv. | - | - | - | - | - | - | - |

### 表2：消融实验结果

| Variant | RMSE | MAE | PAI | HR@10% | $\Delta$PAI | Expert Weight Entropy |
|---------|------|-----|-----|--------|--------------|----------------------|
| Full Model (ACR-ST) | - | - | - | - | - | - |
| w/o Adaptive Routing | - | - | - | - | - | N/A |
| w/o CPTED Knowledge | - | - | - | - | - | - |
| w/o Semantic | - | - | - | - | - | N/A |
| Single Expert Only | - | - | - | - | - | N/A |
| w/o Near-Repeat | - | - | - | - | - | - |
| w/o Hypergraph | - | - | - | - | - | - |

### 表3：自适应路由机制对照

| Routing Design | RMSE | MAE | PAI | Avg $w_0$ | Avg $w_1$ | Avg $w_2$ |
|----------------|------|-----|-----|-----------|-----------|-----------|
| Quadratic (Ours) | - | - | - | - | - | - |
| Linear | - | - | - | - | - | - |
| Sharp Threshold | - | - | - | - | - | - |
| Uniform Weight | - | - | - | 0.33 | 0.33 | 0.33 |

### 表4：RAG语义生成对照

| Generation Method | RMSE | MAE | PAI | Bias Pass Rate | Avg Description Length |
|-------------------|------|-----|-----|----------------|------------------------|
| RAG-Adaptive (Ours) | - | - | - | - | - |
| Basic LLM (Open) | - | - | - | - | - |
| Template-Only | - | - | - | - | - |
| RAG w/o Audit | - | - | - | - | - |

### 表5：差异化策略对照

| Strategy | RMSE | MAE | PAI | Complexity Correlation | Notes |
|----------|------|-----|-----|------------------------|-------|
| ACR-ST (Ours) | - | - | - | - | 环境复杂度软路由 |
| Hard-Partition (Density) | - | - | - | - | 犯罪密度硬划分 |
| Single-Scale | - | - | - | - | 无差异化 |

---

## 七、实验执行建议

### 7.1 重复实验设置
```python
num_runs = 5  # 或10，用于统计检验
seeds = [42, 123, 456, 789, 2023]

# 报告均值±标准差
# 表格中写：2.15 ± 0.08
```

### 7.2 显著性标记
在表格中用星号标注显著性：
- * p < 0.05
- ** p < 0.01
- *** p < 0.001

示例：2.15*** 表示显著优于所有baseline

### 7.3 可视化建议

| 图表类型 | 展示内容 | 目的 |
|---------|---------|------|
| 复杂度-性能散点图 | x轴：环境复杂度，y轴：PAI | 验证复杂度评估的合理性 |
| 专家权重热力图 | 空间分布的三层专家权重 | 展示自适应路由的实际效果 |
| CPTED雷达图 | 四维度得分的区域对比 | 环境设计诊断工具 |
| 消融瀑布图 | 各组件贡献的累积效应 | 组件重要性排序 |

---

## 八、论文结构建议

### 推荐章节组织

1. **Introduction** - 研究背景与动机（差异化策略的必要性）
2. **Related Work** - 犯罪预测、时空图神经网络、LLM应用、CPTED理论
3. **Methodology**
   - 3.1 问题定义与符号
   - 3.2 RAG增强语义生成（知识库构建）
   - 3.3 自适应环境复杂度路由（核心创新）
   - 3.4 CPTED知识嵌入
   - 3.5 时空Transformer架构
4. **Experiments**
   - 4.1 Experimental Setup
   - 4.2 Main Results (与baseline对比)
   - 4.3 Ablation Study (自适应路由各组件)
   - 4.4 Controlled Experiments (差异化策略对照)
   - 4.5 Case Study (典型区域的复杂度分析和政策建议)
5. **Discussion**
   - 结果讨论、伦理考量（偏见控制）、局限与未来工作
6. **Conclusion**

### 关键图表清单

| 图表编号 | 内容 | 放置章节 |
|---------|------|---------|
| Figure 1 | 整体架构图（含RAG和自适应路由） | Methodology |
| Figure 2 | 自适应专家融合机制详解 | Methodology |
| Figure 3 | 复杂度-专家权重关系曲线 | Methodology |
| Figure 4 | Baseline性能雷达图 | Results |
| Figure 5 | 消融实验瀑布图 | Ablation Study |
| Figure 6 | 专家权重空间分布热力图 | Case Study |
| Figure 7 | CPTED四维度诊断示例 | Case Study |
| Table 1 | Baseline对比表 | Results |
| Table 2 | 消融实验结果表 | Ablation Study |
| Table 3 | 差异化策略对照表 | Controlled Experiments |

---

## 九、伦理声明模板

```markdown
## 伦理声明

本研究严格遵循以下原则确保算法的公平性和可解释性：

1. **环境纯净性**：所有输入特征仅包含物理环境因素（POI、道路、照明、监控），
   明确排除人口统计、社会经济地位等可能引入偏见的变量。

2. **知识库审查**：RAG知识库中所有条目经自动化偏见审查，排除种族、阶层相关表述，
   仅保留基于CPTED理论的环境设计知识。

3. **生成约束**：LLM Prompt明确限定讨论范围于物理环境，自动化审查未通过的生成
   结果将被回退至保守模板。

4. **可解释性问责**：系统输出环境复杂度分数、专家路由权重、CPTED四维度得分等
   可解释指标，支持人工审计和政策制定。

5. **政策导向**：建议的干预措施聚焦于环境设计改进（照明、视线、门禁等），
   避免针对特定人群的监控或执法建议。
```

---

**文档版本**: v2.0 (保守改进版)
**最后更新**: 2026-04-08
