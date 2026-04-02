# 期刊论文实验设计方案

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

### 1.3 语义增强方法（1-2个，如果有相关文献）

| 模型 | 简称 | 选择理由 |
|------|------|---------|
| ST-GCN + POI Embedding | STGCN-POI | 对比传统POI嵌入 vs 你的LLM语义 |
| ST-GCN + BERT | STGCN-BERT | 对比通用文本嵌入 vs 犯罪学Prompt设计的LLM语义 |

---

## 二、消融实验设计（Ablation Study）

### 2.1 主消融实验（验证各模块贡献）

| 变体 | 配置 | 验证问题 |
|------|------|---------|
| Full Model | 完整模型 | 基准性能 |
| w/o Semantic | 移除LLM语义嵌入 | LLM语义是否有价值？ |
| w/o Semantic Gate | 语义直接拼接（无门控） | 自适应门控是否优于简单拼接？ |
| w/o Near-Repeat | 移除近重复效应模块 | 近重复理论建模是否必要？ |
| w/o Hypergraph | 移除超图注意力 | 功能区聚类是否提升性能？ |
| w/o Cross-Fusion | 静态/动态特征直接相加 | 交叉注意力融合是否更优？ |

### 2.2 组件替换消融（验证设计选择）

| 变体 | 替换为 | 验证问题 |
|------|--------|---------|
| Gating → Concat+MLP | 拼接后接MLP | 门控机制 vs 传统融合 |
| Gating → Attention | 多头注意力融合 | 门控 vs 注意力 |
| Near-Repeat → Fixed | 固定参数（α=0.001, β=0.3） | 可学习参数 vs 理论固定值 |
| ZINB → MSE | 均方误差损失 | 零膨胀分布是否必要？ |

---

## 三、对照实验设计（Controlled Experiments）

### 3.1 LLM语义设计对照

验证Prompt工程是否有效：

| 变体 | Prompt设计 | 验证问题 |
|------|-----------|---------|
| Ours | 犯罪学理论驱动（Crime Generator/Guardianship） | 你的设计 |
| Generic | 通用描述（"这是一个有X个POI的区域"） | 犯罪学术语是否有价值？ |
| No Theory | 仅列出特征数值，无理论框架 | 理论框架是否必要？ |
| Short | 极简描述（1句话） | 描述详细程度的影响 |
| Long | 详细分析（5-6句话） | 描述长度是否存在饱和？ |

### 3.2 近重复效应建模对照

| 变体 | 建模方式 | 验证问题 |
|------|---------|---------|
| Ours | 可学习时空衰减参数 | 你的设计 |
| Fixed Gaussian | 固定高斯核（σ_space=400m, σ_time=7d） | 经典近重复理论参数是否最优？ |
| Exponential Decay | 指数衰减替代高斯 | 衰减函数形式的影响 |
| OD-modulated | 移除OD流调制 | 人流因素是否必要？ |
| Separate Effects | 空间/时间效应分别建模后相加 | 联合建模 vs 分离建模 |

### 3.3 图结构对照

| 变体 | 图结构 | 验证问题 |
|------|--------|---------|
| Ours | 空间+距离+犯罪+超图 | 完整设计 |
| Spatial Only | 仅空间邻接图 | 距离衰减是否必要？ |
| Distance Only | 仅距离图 | 拓扑结构是否必要？ |
| No Hypergraph | 移除超图（功能区聚类） | 土地利用语义聚类是否有价值？ |

### 3.4 解耦策略对照

| 变体 | 静态/动态处理方式 | 验证问题 |
|------|------------------|---------|
| Ours | 解耦编码+交叉融合 | 你的设计 |
| Early Fusion | 输入层直接拼接 | 早期融合 vs 晚期融合 |
| Late Fusion | 分别预测后加权 | 预测层融合 |
| No Decoupling | 单一编码器处理 | 解耦是否必要？ |

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

### 4.3 可解释性/效率指标

| 指标 | 说明 |
|------|------|
| Inference Time | 单次预测耗时（毫秒） |
| Parameter Count | 模型参数量 |
| Memory Usage | GPU显存占用 |

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

---

## 六、实验结果表格模板

### 表1：与Baseline对比（主要结果表）

| Model | RMSE↓ | MAE↓ | PAI↑ | HR@10%↑ | Jaccard↑ | Params |
|-------|-------|------|------|---------|----------|--------|
| HA | - | - | - | - | - | - |
| RF | - | - | - | - | - | - |
| STGCN | - | - | - | - | - | - |
| DCRNN | - | - | - | - | - | - |
| GWN | - | - | - | - | - | - |
| STT | - | - | - | - | - | - |
| **Ours** | **-** | **-** | **-** | **-** | **-** | - |
| Improv. | - | - | - | - | - | - |

### 表2：消融实验结果

| Variant | RMSE | MAE | PAI | HR@10% | ΔPAI |
|---------|------|-----|-----|--------|------|
| Full Model | - | - | - | - | - |
| w/o Semantic | - | - | - | - | - |
| w/o Semantic Gate | - | - | - | - | - |
| w/o Near-Repeat | - | - | - | - | - |
| w/o Hypergraph | - | - | - | - | - |
| w/o Cross-Fusion | - | - | - | - | - |

### 表3：语义设计对照实验

| Prompt Design | RMSE | MAE | PAI | Semantic Gate Mean |
|--------------|------|-----|-----|-------------------|
| Ours (Theory-driven) | - | - | - | - |
| Generic Description | - | - | - | - |
| No Theory Framework | - | - | - | - |
| Short (1 sentence) | - | - | - | - |
| Long (5-6 sentences) | - | - | - | - |

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

---

## 八、论文结构建议

### 推荐章节组织

1. **Introduction** - 研究背景与动机
2. **Related Work** - 犯罪预测、时空图神经网络、LLM应用
3. **Methodology** - 模型架构详解
4. **Experiments**
   - 4.1 Experimental Setup (数据、指标、baseline)
   - 4.2 Main Results (与baseline对比)
   - 4.3 Ablation Study (消融实验)
   - 4.4 Controlled Experiments (对照实验)
   - 4.5 Case Study (典型案例分析)
5. **Discussion** - 结果讨论与局限
6. **Conclusion**

### 推荐图表

| 图表 | 内容 | 放在哪里 |
|------|------|---------|
| 雷达图 | 多指标综合对比 | Results开头 |
| 提升幅度条形图 | 相对最强baseline的改进 | Results |
| 消融瀑布图 | 从Full Model逐个移除组件的性能变化 | Ablation Study |
| 参数敏感性 | 门控维度、近重复初始化等对性能的影响 | Parameter Study |
