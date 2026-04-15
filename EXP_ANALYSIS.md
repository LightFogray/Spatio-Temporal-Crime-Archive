# EP-STD (Environment-Prompted Spatio-Temporal Diffusion) 架构分析

**项目名称**: EP-STD - 环境引导的时空扩散犯罪预测模型  
**版本**: 2.0  
**日期**: 2026-04-13  
**核心目标**: 解决犯罪预测中的冷启动问题（Cold Start）

---

## 一、整体架构设计

### 1.1 系统概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EP-STD 整体架构                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  数据层     │ -> │  特征层     │ -> │  模型层     │ -> │  应用层     │  │
│  │  Data Layer │    │ Feature Layer│   │ Model Layer │    │ Application │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│        │                  │                  │                  │          │
│   Raw Data          Static/Temporal      EP-STD Core       Prediction     │
│   (Crime/Env/OD)    Features             (3 Stages)        & Explanation  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 详细数据流图

```mermaid
flowchart TB
    subgraph DataSource["数据源"]
        C1[Chicago Crime API<br/>2022-2023]
        C2[POI/Landuse/Green<br/>OSM Data]
        C3[OD Flow Data<br/>Divvy/Taxi]
        C4[Weather Data]
    end

    subgraph Preprocessing["数据预处理"]
        P1[chicago_crime_downloader.py<br/>分类下载: Violent/Property]
        P2[build_static_features.py<br/>静态特征: POI/Road/Landuse/Green]
        P3[build_od_flow.py<br/>OD流特征构建]
        P4[filter_od_by_date.py<br/>日期过滤对齐]
    end

    subgraph FeatureEngineering["特征工程"]
        F1[build_crime_timeseries.py<br/>犯罪时间序列: (730, 1246, 2)]
        F2[build_stgcn_input.py<br/>构建X.npy/Y.npy<br/>X:(723,1246,44) Y:(723,1246)]
        F3[build_graph.py<br/>构建5张图结构<br/>Spatial/Distance/CrimeV/CrimeP/OD]
    end

    subgraph EPSTD_Model["EP-STD 三阶段模型"]
        S1[Stage 1: 环境编码器<br/>epstd_stage1.py<br/>Input: (N,24) Static<br/>Output: (N,64) EnvEmb]
        S2[Stage 2: 原型学习<br/>epstd_stage2.py<br/>Input: EnvEmb<br/>Output: Prototypes + Labels]
        S3[Stage 3: 多图扩散<br/>epstd_stage3_enhanced_multigraph.py<br/>Input: X+5Graphs+EnvEmb<br/>Output: Risk+Uncertainty]
    end

    subgraph Evaluation["评估与应用"]
        E1[masking_experiment.py<br/>冷启动压力测试]
        E2[policy_recommender.py<br/>策略推荐]
        E3[explainability.py<br/>可解释性分析]
    end

    C1 --> P1 --> F1
    C2 --> P2
    C3 --> P3 --> P4
    C4 --> F2
    
    P2 --> F2
    P4 --> F2
    F1 --> F2
    
    F2 --> S1
    F2 --> S3
    F1 --> F3
    F3 --> S3
    
    S1 --> S2
    S1 --> S3
    S2 --> S3
    
    S3 --> E1
    S3 --> E2
    S3 --> E3
```

### 1.3 各模块详细输入输出

#### 1.3.1 数据预处理模块

| 文件 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `chicago_crime_downloader.py` | Socrata API | `chicago_crime_data/cleaned/*.csv` | 下载暴力/财产犯罪数据 |
| `build_static_features.py` | OSM数据/遥感数据 | `data/processed/*_features.npy` | 构建静态环境特征 |
| `build_od_flow.py` | Divvy/Taxi CSV | `*_inflow_daily.npy` 等 | 日级OD流聚合 |
| `grid_mapping.py` | 10000网格OD + 1246网格 | `*_1246.npy` | 网格对齐映射 |
| `build_crime_timeseries.py` | Cleaned crime CSV | `crime_combined_timeseries.npy` | (730, 1246, 2) 双犯罪类型 |

#### 1.3.2 特征工程模块

| 文件 | 输入 | 输出 | 形状 |
|------|------|------|------|
| `build_stgcn_input.py` | 静态特征+天气+犯罪滞后 | `X.npy`, `Y.npy` | X:(723,1246,44), Y:(723,1246) |
| `build_graph.py` | Crime时间序列+Grid | 5张邻接矩阵 | (1246, 1246) 或 (700, 1246, 1246) |

**X.npy 特征维度分解**:
```
X.shape = (T-T_lag, N, F)
  - T-T_lag: 723 天 (滞后7天)
  - N: 1246 个网格
  - F: 44 维特征
    * Spatial static: 24 维 (POI+Landuse+Road+Green+Nightlight+Camera)
    * Weather: 8 维 (7天滞后)
    * Crime lag: 14 维 (Violent/Property × 7天滞后)
```

#### 1.3.3 EP-STD 三阶段模型

**Stage 1: 环境编码器 (Environment Encoder)**
```python
# epstd_stage1.py
Input:  static_features (N, 24)  # 静态环境特征
Output: env_embeddings (N, 64)   # 环境嵌入
        checkpoints/env_encoder_best.pt
```

**Stage 2: 原型学习 (Prototype Learning)**
```python
# epstd_stage2.py
Input:  env_embeddings (N, 64)
Output: prototype_library.pkl     # 原型库
        prototype_labels.npy      # (N,) 每个网格的原型标签
        prototype_centers.npy     # (K, 64) K个原型中心
```

**Stage 3: 多图扩散模型 (Multi-Graph Diffusion)**
```python
# epstd_stage3_enhanced_multigraph.py
Input:  
  - X_train: (samples, N, F)           # 环境特征
  - Y_train: (samples, N)              # 犯罪标签
  - adj_list: [A_spatial, A_distance, A_crime_v, A_crime_p, A_od]  # 5张图
  - env_emb: (samples, N, 64)          # Stage 1输出
  - prototype_ids: (N,)                # Stage 2输出

Output: 
  - risk_mean: (N,)                    # 预测风险均值
  - risk_std: (N,)                     # 预测不确定性
  - pi: (N,)                           # 零膨胀概率
  - graph_weights: (5,)                # 图融合权重（可解释性）
  - crime_gate: (2,)                   # 犯罪类型门控
```

---

## 二、创新方法与技术细节

### 2.1 核心创新点总览

| 创新点 | 传统方法 | EP-STD改进 | 解决的问题 |
|--------|----------|------------|------------|
| **1. 三阶段生成式框架** | 判别式(STGCN) | 扩散模型+环境编码 | 数据稀疏、零膨胀 |
| **2. 环境自适应图融合** | 固定图权重 | w = MLP(E_env) | 环境异质性 |
| **3. 双犯罪交叉反馈** | 单一犯罪类型 | Gate(Violent, Property) | 冷启动转移学习 |
| **4. T-Norm逻辑引导** | 无领域知识 | 神经符号约束 | 专家知识融入 |
| **5. 稀疏犯罪传播图** | 全连接N²计算 | 空间掩码稀疏化 | 计算效率 |

### 2.2 技术细节

#### 2.2.1 环境自适应图融合

**问题**: 固定权重 $w = [0.5, 0.3, 0.2]$ 无法适应不同区域的图重要性差异

**解决方案**: 让每个网格根据环境特征学习个性化图权重

**数学公式**:
$$
w_i = 	ext{Softmax}(	ext{MLP}(E_{env}^{(i)}) / \tau)
$$

其中:
- $E_{env}^{(i)} \in \mathbb{R}^{64}$: 网格 $i$ 的环境嵌入
- MLP: $64 \rightarrow 32 \rightarrow 5$ 的神经网络
- $\tau$: 可学习的温度参数，控制softmax锐度
- $w_i \in \mathbb{R}^5$: 5张图的融合权重

**为什么有效**:
- 商业区: OD流图权重高（人流密集）
- 居民区: 空间邻接图权重高（近邻效应）
- 工业区: 夜间灯光图权重高（活动模式）

**适用条件**: 
- 静态环境特征能区分不同功能区
- 网格数量 N > 100（保证MLP有足够样本学习）

---

#### 2.2.2 双犯罪交叉反馈门控

**问题**: 暴力犯罪数据稀疏（零膨胀90%+），难以学习可靠的空间模式

**解决方案**: 利用财产犯罪（高频）辅助暴力犯罪（低频）预测，通过门控动态切换

**数学公式**:
$$
\begin{aligned}
g_i &= \text{Softmax}(\text{MLP}([E_{env}^{(i)}, \bar{c}_v^{(i)}, \bar{c}_p^{(i)}])) \\
\bar{c}_v^{(i)} &= \frac{1}{T}\sum_t c_v(t, i) \quad \text{(暴力犯罪历史均值)} \\
\bar{c}_p^{(i)} &= \frac{1}{T}\sum_t c_p(t, i) \quad \text{(财产犯罪历史均值)} \\
A_{crime}^{(i)} &= g_i^{(v)} \cdot A_{crime_v} + g_i^{(p)} \cdot A_{crime_p}
\end{aligned}
$$

**冷启动增强**:
当检测到暴力犯罪稀疏时（$\bar{c}_v^{(i)} < \epsilon$），强制门控偏向财产犯罪:
$$
g_i = [0.3, 0.7] \quad \text{if } \bar{c}_v^{(i)} < 0.1
$$

**为什么有效**:
- 财产犯罪与暴力犯罪在空间分布上有相关性（如商业区两者都高）
- 财产犯罪数据密集，空间模式更稳定
- 门控机制允许模型学习"何时依赖哪种犯罪类型"

**假设条件**:
- 两种犯罪类型在空间上不是完全独立（存在相关性）
- 财产犯罪的时空模式对暴力犯罪有预测价值

---

#### 2.2.3 T-Norm软逻辑引导

**问题**: 纯数据驱动模型在冷启动区域预测趋近于0（历史均值），忽略环境风险信号

**解决方案**: 将犯罪学专家规则转化为可微损失函数，在去噪过程中提供梯度引导

**核心规则**:
$$
\text{商业高} \land \text{监控低} \rightarrow \text{高风险}
$$

**T-Norm转换**:
- 合取（AND）: $T(a, b) = a \cdot b$ (乘积T-Norm)
- 蕴涵（→）: $I(a, b) = \min(1, b/a)$ (Goguen蕴涵)

**逻辑损失**:
$$
\mathcal{L}_{logic} = \underbrace{I(f_{risk}, \hat{y})}_{\text{漏报惩罚}} + \underbrace{I(1-f_{risk}, 1-\hat{y})}_{\text{误报惩罚}}
$$

其中 $f_{risk}$ 是从环境特征计算的风险先验:
$$
f_{risk} = \sigma(w_{comm} \cdot f_{comm} + w_{road} \cdot f_{road} - w_{light} \cdot f_{light} - w_{cam} \cdot f_{cam})
$$

**Classifier Guidance去噪**:
$$
x_{t-1} = x_t - \eta[\epsilon_\theta(x_t, t, c) + \omega \nabla_{x_t} \mathcal{L}_{logic}]
$$

**为什么有效**:
- 将"被动惩罚结果"变为"主动引导去噪方向"
- 在零数据区域提供确定性梯度（不是随机噪声）
- 双向约束防止过度保守（全城低报）或过度敏感（全城高报）

**退火策略**:
前20epoch $\omega = 0$，让模型先学习基本时空模式；之后线性增加到1.0，逐步引入逻辑约束。

---

#### 2.2.4 稀疏犯罪传播图

**问题**: 动态图计算 $O(T \cdot N^2 \cdot window)$，当 $N=1246$ 时计算量过大

**解决方案**: 利用空间邻近性假设，仅计算邻居间的高斯相似度

**稀疏化策略**:
$$
A_{crime}^{sparse}[i, j] = \begin{cases}
\exp(-\frac{\|c_i - c_j\|^2}{2\sigma^2}) & \text{if } A_{spatial}[i, j] > 0 \\
0 & \text{otherwise}
\end{cases}
$$

**计算复杂度**:
- 稠密: $O(N^2) = O(1.5M)$
- 稀疏: $O(N \cdot k) = O(7.5K)$ (平均度 $k \approx 6$)
- **加速比: ~200x**

**物理合理性**:
犯罪传播遵循距离衰减定律，远距离（>1km）的犯罪相关性主要来自环境协变量而非真实传播。

---

### 2.3 扩散模型核心公式

**前向扩散过程**（加噪）:
$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)
$$

**反向去噪过程**（预测噪声）:
$$
p_\theta(x_{t-1} | x_t, c) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t, c), \Sigma_t)
$$

**训练目标**:
$$
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}[\|\epsilon - \epsilon_\theta(x_t, t, c)\|^2] + \lambda_{zi} \mathcal{L}_{zi} + \lambda_{logic} \mathcal{L}_{logic}
$$

其中:
- $\epsilon_\theta$: 神经网络预测的噪声
- $\mathcal{L}_{zi}$: 零膨胀损失 (BCE)
- $\mathcal{L}_{logic}$: 逻辑约束损失

---

## 三、运行步骤

### 3.1 完整运行流程

```bash
# ============================================================
# Phase 1: 数据准备 (Data Preparation)
# ============================================================

# 1. 下载犯罪数据（分类下载: Violent/Property）
python src/chicago_crime_downloader.py
# 输出: chicago_crime_data/cleaned/{violent,property}_{2022,2023}_cleaned.csv

# 2. 构建静态环境特征
python src/build_static_features.py
# 输出: data/processed/{poi,landuse,green,road,nightlight,camera}_features.npy

# 3. 构建OD流特征（在另一台机器运行后传输）
python src/od_process/build_od_flow.py        # 生成日级OD
python src/od_process/grid_mapping.py         # 映射到1246网格
python src/od_process/merge_od_flow.py        # 合并为dynamic_od_flow.npy
# 输出: data/processed/dynamic_od_flow.npy (730, 1246, 4)

# 4. 构建犯罪时间序列
python src/build_crime_timeseries.py
# 输出: data/processed/crime_combined_timeseries.npy (730, 1246, 2)

# 5. 构建训练输入
python src/build_stgcn_input.py
# 输出: data/processed/X.npy, Y.npy

# 6. 构建所有图结构（5张图）
python src/build_graph.py
# 输出: 
#   - adj_adaptive.npy       (空间邻接)
#   - adj_distance.npy       (距离衰减)
#   - adj_crime_violent.npy  (暴力传播)
#   - adj_crime_property.npy (财产传播)
#   - adj_od.npy             (OD功能相似)


# ============================================================
# Phase 2: EP-STD 三阶段训练
# ============================================================

# Stage 1: 环境编码器训练（对比学习）
python src/epstd_stage1.py
# 输入: static_features (N, 24)
# 输出: checkpoints/env_encoder_best.pt, data/processed/env_embeddings.npy

# Stage 2: 原型学习（K-Means + 模糊风险先验）
python src/epstd_stage2.py
# 输入: env_embeddings (N, 64)
# 输出: checkpoints/prototype_library.pkl, prototype_labels.npy

# Stage 3: 多图扩散模型训练（核心）
python src/epstd_stage3_enhanced_multigraph.py
# 输入: X.npy, Y.npy, 5张图, env_embeddings, prototype_labels
# 输出: checkpoints/dual_multigraph_diffusion_best.pt


# ============================================================
# Phase 3: 评估与应用
# ============================================================

# 压力测试（冷启动性能）
python experiments/masking_experiment.py
# 评估指标: MAE, Correlation, Hotspot Recall @ masked grids

# 策略推荐
python src/policy_recommender.py

# 可解释性分析
python src/explainability.py
```

### 3.2 快速复现（使用已有检查点）

```bash
# 如果已有训练好的检查点，直接预测
python -c "
from src.epstd_stage3_enhanced_multigraph import *

# 加载模型
model = load_model('checkpoints/dual_multigraph_diffusion_best.pt')

# 预测
predictor = MultiGraphPredictor(model, scheduler, adj_list, env_encoder, prototype_library)
risk_mean, risk_std, pi = predictor.predict(X_test[0], return_graph_weights=True)
"
```

### 3.3 文件依赖关系

```
build_static_features.py
    ↓
build_stgcn_input.py ← build_crime_timeseries.py ← chicago_crime_downloader.py
    ↓
epstd_stage1.py → env_embeddings.npy
    ↓
epstd_stage2.py → prototype_library.pkl
    ↓
epstd_stage3_enhanced_multigraph.py
    ↓
experiments/masking_experiment.py
```

### 3.4 关键检查点文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `checkpoints/env_encoder_best.pt` | ~5MB | Stage 1环境编码器 |
| `checkpoints/prototype_library.pkl` | ~1MB | Stage 2原型库 |
| `checkpoints/multigraph_diffusion_best.pt` | ~50MB | Stage 3多图扩散模型 |
| `data/processed/X.npy` | ~170MB | 训练特征 (723, 1246, 44) |
| `data/processed/Y.npy` | ~3.5MB | 训练标签 (723, 1246) |

---

## 四、超参数配置

### 4.1 Stage 1 环境编码器

```python
config_stage1 = {
    'input_dim': 24,           # 静态特征维度
    'hidden_dim': 128,
    'output_dim': 64,          # 环境嵌入维度
    'temperature': 0.07,       # InfoNCE温度系数
    'batch_size': 256,
    'epochs': 100,
    'lr': 1e-3
}
```

### 4.2 Stage 2 原型学习

```python
config_stage2 = {
    'n_prototypes': 10,        # 原型数量（功能区类别）
    'fuzzy_threshold': 0.3,    # 模糊隶属度阈值
    'risk_bins': 5             # 风险分箱数
}
```

### 4.3 Stage 3 多图扩散

```python
config_stage3 = {
    'hidden_dim': 128,
    'num_layers': 4,           # 图注意力层数
    'num_timesteps': 1000,     # 扩散步数
    'beta_schedule': 'linear', # beta调度策略
    'beta_start': 1e-4,
    'beta_end': 0.02,
    'graph_fusion_temp': 0.5,  # 图融合温度
    'logic_warmup_epochs': 20, # 逻辑引导预热轮数
    'guidance_scale': 1.0,     # 逻辑引导强度
    'task_weights': {          # 多任务权重
        'violent': 1.0,
        'property': 0.5
    }
}
```

---

## 五、性能指标预期

### 5.1 主要评估指标

| 指标 | 基线(STGCN) | EP-STD(无逻辑) | L-EPSTD(有逻辑) |
|------|------------|----------------|-----------------|
| MAE | 2.5 | 2.1 | 1.8 |
| Correlation | 0.65 | 0.72 | 0.78 |
| Hotspot Recall@10% | 45% | 58% | 68% |
| **Cold Start Recall** | **5%** | **25%** | **55%** |

### 5.2 消融实验设计

```
Ablation Study:
  1. Base: 无环境编码 + 无逻辑引导
  2. +Env: 加入环境编码器
  3. +MultiGraph: 加入多图结构
  4. +AdaptiveFusion: 加入自适应图融合
  5. +CrossCrime: 加入双犯罪交叉反馈
  6. +Logic: 加入T-Norm逻辑引导 (Full Model)
```

### 5.3 评估指标设计说明

#### 为什么放弃传统指标（RMSE/MAE/Hit Rate/PAI/Jaccard）？

| 传统指标 | 问题 | EP-STD替代方案 |
|----------|------|----------------|
| **RMSE** | 对异常值敏感，零膨胀数据下会被大量零值主导 | 仅用于非零区域评估，配合零膨胀损失 |
| **MAE** | 无法区分"过度预测"和"预测不足"，警务部署中漏报代价更高 | 分位数损失 + 热点召回率组合 |
| **Hit Rate** | 只关心是否命中热点，不关心预测置信度 | `risk_std`不确定性量化 |
| **PAI** | 假设犯罪均匀分布，不适用于高度聚集的犯罪数据 | 原型内PAI（按功能区分组评估） |
| **Jaccard** | 对空间边界敏感，网格微移会导致指标大幅波动 | 距离加权重叠度（考虑空间衰减） |

**核心差异**：
1. **零膨胀感知**：传统指标将零犯罪日视为普通样本，EP-STD使用零膨胀概率`pi`显式建模
2. **不确定性量化**：传统指标点估计，EP-STD输出`risk_std`支持风险排序决策
3. **冷启动评估**：传统指标在全数据上计算，EP-STD专门设计"屏蔽实验"评估冷启动网格
4. **可解释性耦合**：传统指标黑盒，EP-STD指标与图权重、门控机制联动解释

---

## 六、模型输出与评估体系

### 6.1 基础预测输出

| 输出 | 符号 | 维度 | 说明 |
|------|------|------|------|
| `risk_mean` | μ | (N,) | 预测风险均值（期望犯罪数） |
| `risk_std` | σ | (N,) | 预测不确定性（标准差） |
| `pi` | π | (N,) | 零膨胀概率（该网格无犯罪的概率） |
| `graph_weights` | w | (B, N, 5) | 5张图的自适应融合权重 |
| `crime_gate` | g | (B, N, 2) | 暴力/财产犯罪交叉门控权重 |

### 6.2 核心评估指标

#### 6.2.1 冷启动专项指标（Masking Experiment）
```python
{
    'mae': 0.15,                    # 被屏蔽网格的平均绝对误差
    'correlation': 0.72,            # 预测与真实值的相关系数
    'hotspot_recall': 0.65,         # Top 10%热点召回率
    'n_masked': 249                 # 被屏蔽网格数
}
```

#### 6.2.2 全局性能指标
| 指标 | 含义 | 典型值 | 与传统指标对比 |
|------|------|--------|----------------|
| **PAI@10%** | 预测准确度指数 | 1.5-2.5 | 相同计算方式，但仅在原型内计算 |
| **Recall@10%** | 热点召回率 | 55%-75% | 等同于Hit Rate |
| **Pearson r** | 整体相关性 | 0.65-0.85 | 等同于传统Correlation |
| **Cold Start Recall** | 冷启动热点召回 | 25%-55% | 传统指标无此细分 |

---

## 七、可解释性输出

### 7.1 多图自适应融合权重（Adaptive Graph Fusion）

每个网格根据环境特征学习个性化的图融合权重，可解释该区域的主导影响因素:

```python
# 网格123的图权重示例
{
    'Spatial': 0.03,      # 空间邻接图权重 - 低
    'Distance': 0.15,     # 距离衰减图权重 - 中
    'CrimeV': 0.05,       # 暴力犯罪传播图 - 低
    'CrimeP': 0.75,       # 财产犯罪传播图 - 高（主导）
    'OD': 0.02            # OD流功能相似图 - 低
}
```

**可解释性解读**：
- **高Spatial权重** → 居民区（近邻效应主导）
- **高OD权重** → 商业区/交通枢纽（人流驱动犯罪）
- **高CrimeV权重** → 高风险区（历史暴力犯罪聚集）
- **高CrimeP权重** → 商业密集区（盗窃抢劫高发）

### 7.2 交叉犯罪门控（Cross-Crime Gate）

```python
{
    'violent_weight': 0.30,   # 暴力犯罪图贡献
    'property_weight': 0.70   # 财产犯罪图贡献
}
```

**可解释性解读**：
- 当暴力犯罪数据稀疏时（冷启动），门控自动提高财产犯罪图权重进行"迁移学习"
- 财产犯罪与暴力犯罪在空间分布上存在相关性（商业区两者都高）
- 门控机制允许模型学习"何时依赖哪种犯罪类型"

### 7.3 原型聚类解释（Stage 2）

| 原型ID | 网格数 | 平均风险 | 功能区类型 | 特征描述 |
|--------|--------|----------|------------|----------|
| 0 | 145 | 0.186 | 高密度商业区 | 商业POI密集、人流大 |
| 1 | 14 | 0.571 | 夜间娱乐区 | 酒吧夜店聚集 |
| 4 | 25 | 0.860 | 高风险商业区 | 商业+低监护 |
| 6 | 780 | 0.076 | 低密度住宅区 | 安静居住区 |
| 9 | 96 | 1.026 | 交通枢纽区 | 车站+商业混合 |

### 7.4 扩散过程可视化

- **去噪轨迹图**：展示从纯噪声到最终风险预测的全过程
- **时间注意力热图**：不同时间步的注意力权重分布（近重复效应窗口）
- **环境-原型映射**：T-SNE可视化环境嵌入空间中的原型分布

---

## 八、政策建议生成系统

### 8.1 网格风险画像

系统为每个高风险网格生成结构化画像：

```python
GridProfile {
    grid_id: 123,
    risk_score: 0.85,           # 预测风险分数
    risk_level: '极高',          # 风险等级（极高/高/中/低）
    cpted_scores: {             # CPTED四维评估
        'natural_surveillance': 0.2,   # 自然监护不足
        'access_control': 0.3,         # 入口控制薄弱
        'territorial_reinforcement': 0.2,
        'target_hardening': 0.1
    },
    environmental_features: {
        'green_ratio': 0.7,     # 绿化率高（可能遮挡视线）
        'nightlight': 0.2,      # 夜间照明不足
        'camera_count': 0.1     # 摄像头覆盖率低
    },
    nearby_crimes_7d: 3         # 近7天周边犯罪数
}
```

### 8.2 分级干预措施

#### 立即执行（24小时内）
| 措施 | 优先级 | 成本 | 实施主体 | 预期效果 |
|------|--------|------|----------|----------|
| 启动近重复预警响应 | 极高 | 低 | 公安分局 | 案发后7天内周边犯罪率降30% |
| 增加定点巡逻 | 高 | 低 | 辖区派出所 | 提升见警率，震慑潜在犯罪 |
| 推广社区守望计划 | 高 | 低 | 社区居委会 | 建立商户居民即时通报机制 |

#### 短期措施（1周内）
| 措施 | 优先级 | 成本 | 实施细节 |
|------|--------|------|----------|
| 升级LED路灯 | 高 | 中 | 更换为200W LED，照度提升至50lux以上 |
| 增补AI智能摄像头 | 高 | 中 | 在网格四角安装行为识别摄像头 |
| 修剪植被高度 | 高 | 低 | 灌木修剪至0.6米以下，树枝至2.5米以上 |
| 封闭无用巷道 | 高 | 中 | 用铁栅栏封闭犯罪高发的小巷和捷径 |

#### 中期措施（1月内）
| 措施 | 优先级 | 成本 | 预期效果 |
|------|--------|------|----------|
| 引入便民商业 | 中 | 中 | 税收优惠吸引便利店入驻，增加自然监护 |
| 举办社区活动 | 中 | 低 | 周末集市、文艺演出，增加正当人流 |
| 增设口袋公园 | 中 | 高 | 利用闲置空地建设小型公园 |

### 8.3 巡逻部署方案

```python
{
    'total_officers_deployed': 10,
    'coverage_grids': 15,
    'shift_recommendation': '三班制(06:00-14:00, 14:00-22:00, 22:00-06:00)',
    'grid_assignments': [
        {
            'grid_id': 123,
            'officers_assigned': 2,
            'patrol_hours': ['20:00-02:00'],  # 高风险时段
            'patrol_mode': '步行巡逻',         # 道路密度高
            'focus_areas': ['商业街区出入口', '监控盲区', '公交站点']
        }
    ]
}
```

### 8.4 预算估算

```python
{
    'immediate_usd': 5000,       # 立即行动（警力调度）
    'short_term_usd': 150000,    # 短期措施（照明、摄像头）
    'medium_term_usd': 300000,   # 中期措施（基础设施）
    'total_usd': 455000,
    'manpower_priority': '高',
    'roi_estimate': '每投入$1可降低犯罪损失$3-5'
}
```

---

## 九、城市设计建议（按功能区）

### 9.1 高密度商业区
**典型特征**：CrimeP权重高、商业POI密集、人流混杂

| 问题诊断 | 设计建议 | 责任主体 |
|----------|----------|----------|
| 犯罪目标丰富 | 推广商铺卷帘门、入侵报警器补贴 | 商务局+公安 |
| 人流混杂 | 建立50米范围商户联防机制 | 街道办 |
| 便于逃离 | 优化路口布局，减少潜在逃逸路径 | 规划局 |

### 9.2 低密度住宅区
**典型特征**：Spatial权重高、CPTED监护评分低

| 问题诊断 | 设计建议 | 责任主体 |
|----------|----------|----------|
| 自然监护不足 | 组织社区守望计划微信群 | 社区居委会 |
| 照明死角 | 安装太阳能感应灯 | 城管局 |
| 门禁管理松懈 | 修复损坏单元门，加装闭门器 | 物业公司 |

### 9.3 交通枢纽区
**典型特征**：OD权重高、人流密集

| 问题诊断 | 设计建议 | 责任主体 |
|----------|----------|----------|
| 站内摄像头盲区 | 优化摄像头布局，接入公安平台 | 交通局+公安 |
| 人员徘徊 | 设置清晰的方向指示系统 | 地铁公司 |
| 夜间疏散难 | 增加出租车/网约车接驳点 | 交通局 |

---

**文档结束**
