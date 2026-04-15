# EP-STD 实验说明文档

**文档版本**: 1.0  
**日期**: 2026-04-15  
**适用项目**: EP-STD (Environment-Prompted Spatio-Temporal Diffusion)

---

## 目录

1. [实验概述](#1-实验概述)
2. [环境准备](#2-环境准备)
3. [消融实验](#3-消融实验)
4. [对比实验](#4-对比实验)
5. [冷启动压力测试](#5-冷启动压力测试)
6. [结果分析](#6-结果分析)
7. [常见问题](#7-常见问题)

---

## 1. 实验概述

### 1.1 实验类型

| 实验类型 | 脚本路径 | 目的 |
|---------|---------|------|
| 消融实验 | `experiments/epstd_ablation_study.py` | 验证各组件贡献 |
| 对比实验 | `experiments/epstd_baseline_comparison.py` | 与基线模型比较 |
| 压力测试 | `experiments/masking_experiment.py` | 冷启动场景性能 |

### 1.2 实验依赖

**前置条件**:
- 已完成 Stage 1-3 的训练
- 检查点文件存在于 `checkpoints/` 目录
- 处理后的数据存在于 `data/processed/` 目录

**必需文件**:
```
checkpoints/
├── env_encoder_best.pt          # Stage 1 环境编码器
├── prototype_library.pkl        # Stage 2 原型库
└── multigraph_diffusion_best.pt # Stage 3 扩散模型

data/processed/
├── X.npy                        # 输入特征
├── Y.npy                        # 标签
├── adj_adaptive.npy             # 空间邻接图
├── adj_distance.npy             # 距离衰减图
├── adj_crime_violent.npy        # 暴力犯罪传播图
├── adj_crime_property.npy       # 财产犯罪传播图
├── adj_od.npy                   # OD流功能图
└── prototype_labels.npy         # 原型标签
```

---

## 2. 环境准备

### 2.1 安装依赖

```bash
# 基础依赖
pip install torch torchvision torchaudio
pip install numpy pandas scipy scikit-learn
pip install matplotlib seaborn
pip install tqdm

# 可选（用于高级可视化）
pip install networkx plotly
```

### 2.2 验证环境

```bash
python experiments/verify_env_prototype.py
```

**预期输出**:
```
[OK] Environment encoder loaded: env_encoder_best.pt
[OK] Prototype library loaded: prototype_library.pkl
[OK] X.npy shape: (723, 1246, 44)
[OK] Y.npy shape: (723, 1246)
[OK] All adjacency matrices found
```

---

## 3. 消融实验

### 3.1 实验设计

**消融变体** (8个):

| 变体名称 | 描述 | 禁用的组件 |
|---------|------|-----------|
| Full_L-EPSTD | 完整模型 | 无 |
| w/o_Logic_Guidance | 无逻辑引导 | T-Norm约束 |
| w/o_Adaptive_Fusion | 无自适应融合 | MLP权重 → 固定权重 |
| w/o_CrossCrime_Gate | 无交叉门控 | 只用Violent图 |
| w/o_MultiGraph | 无多图 | 只用Spatial图 |
| w/o_Environment_Encoder | 无环境编码 | 随机初始化 |
| w/o_ZeroInflation | 无零膨胀 | MSE损失 |
| w/o_Prototype | 无原型学习 | 禁用原型嵌入 |
| Base_Diffusion_Only | 最简基线 | 所有创新组件 |

### 3.2 运行命令

```bash
# 运行完整消融实验
python experiments/epstd_ablation_study.py
```

**参数配置** (在脚本内修改):
```python
class AblationConfig:
    EPOCHS = 50          # 消融实验训练轮数
    BATCH_SIZE = 16      # 批量大小
    LR = 1e-4           # 学习率
    SEEDS = [42, 123, 456]  # 随机种子列表
```

### 3.3 输出结果

**结果文件**:
```
experiments/results/
├── epstd_ablation_20250415_143022.csv    # 数值结果
├── epstd_ablation_20250415_143022.md     # Markdown报告
└── epstd_ablation_20250415_143022.json   # 原始数据
```

**报告内容示例**:
```markdown
# EP-STD Ablation Study Results

## Results Summary

| Variant | MAE | Correlation | PAI | CS_MAE | CS_Recall |
|---------|-----|-------------|-----|--------|-----------|
| Full_L-EPSTD | 0.12±0.01 | 0.78±0.02 | 2.5±0.1 | 0.18±0.02 | 0.65±0.03 |
| w/o_Logic_Guidance | 0.15±0.01 | 0.72±0.02 | 2.1±0.1 | 0.25±0.03 | 0.55±0.04 |
| ... | ... | ... | ... | ... | ... |

## Component Contribution Analysis

- **w/o_Logic_Guidance**: PAI drops by 16.0%, Cold-start Recall drops by 15.4%
- **w/o_Adaptive_Fusion**: PAI drops by 8.0%, Cold-start Recall drops by 10.8%
```

### 3.4 关键指标解读

| 指标 | 说明 | 期望趋势 |
|------|------|---------|
| MAE | 平均绝对误差 | 越低越好 |
| Correlation | Pearson相关系数 | 越高越好 |
| PAI | 预测准确度指数 | 越高越好 |
| CS_MAE | 冷启动MAE | 越低越好 |
| CS_Recall | 冷启动热点召回率 | 越高越好 |

---

## 4. 对比实验

### 4.1 对比模型

| 模型 | 类型 | 特点 |
|------|------|------|
| HA | 统计基线 | 历史平均 |
| RF | 机器学习 | 随机森林 |
| ConvLSTM | 深度学习 | 时空卷积LSTM |
| STGCN | 图神经网络 | 时空图卷积 |
| DCRNN | 图神经网络 | 扩散卷积RNN |
| GraphWaveNet | 图神经网络 | 自适应图卷积 |
| ST-Transformer | Transformer | 无扩散版本 |
| EP-STD_noLogic | 消融版本 | 无逻辑引导 |
| EP-STD (Ours) | 完整模型 | 所有组件 |

### 4.2 运行命令

```bash
# 运行对比实验
python experiments/epstd_baseline_comparison.py
```

**运行时间预估**:
- 单轮运行: ~2-4小时 (GPU)
- 完整5轮: ~10-20小时

### 4.3 输出结果

```
experiments/results/
├── comparison_run_1_seed42.json
├── comparison_run_2_seed123.json
├── ...
├── baseline_comparison_20250415_143022.csv
├── baseline_comparison_20250415_143022.md
└── comparison_full_results_20250415_143022.csv
```

### 4.4 结果解读

**性能对比表**:
```markdown
| Model | MAE | Correlation | PAI | CS_MAE | CS_Recall |
|-------|-----|-------------|-----|--------|-----------|
| HA | 0.35±0.02 | 0.45±0.03 | 1.2±0.1 | 0.42±0.03 | 0.25±0.03 |
| RF | 0.28±0.01 | 0.58±0.02 | 1.5±0.1 | 0.35±0.02 | 0.35±0.03 |
| STGCN | 0.20±0.01 | 0.68±0.02 | 2.0±0.1 | 0.28±0.02 | 0.45±0.03 |
| EP-STD | 0.12±0.01 | 0.78±0.02 | 2.5±0.1 | 0.18±0.02 | 0.65±0.03 |

## Improvement over Best Baseline (STGCN)
- MAE: +40.0%
- Correlation: +14.7%
- PAI: +25.0%
- CS_MAE: +35.7%
- CS_Recall: +44.4%
```

---

## 5. 冷启动压力测试

### 5.1 实验设计

**屏蔽策略**:
- 随机屏蔽20%的高犯罪网格历史数据
- 模拟新区域无历史数据的冷启动场景

**评估维度**:
1. 全局性能 vs 冷启动性能对比
2. 不同屏蔽比例的影响 (10%, 20%, 30%, 40%)
3. 逻辑引导开启/关闭对比

### 5.2 运行命令

```bash
# 运行压力测试
python experiments/masking_experiment.py
```

### 5.3 输出结果

```
experiments/results/
├── masking_experiment_20pct.npz
├── masking_experiment_30pct.npz
└── masking_experiment_visualization.png
```

### 5.4 可视化解读

生成的可视化包含4个子图:
1. **散点图**: 真实值 vs 预测值（masked网格）
2. **柱状图**: 各模型性能对比
3. **热力图**: 被屏蔽网格的空间分布
4. **残差分布**: 预测误差分布

---

## 6. 结果分析

### 6.1 统计显著性检验

运行5次不同种子后，可计算p-value:

```python
from scipy import stats

# 比较EP-STD和STGCN的MAE
epstd_mae = [0.12, 0.11, 0.13, 0.12, 0.11]
stgcn_mae = [0.20, 0.19, 0.21, 0.20, 0.20]

t_stat, p_value = stats.ttest_ind(stgcn_mae, epstd_mae)
print(f"p-value: {p_value:.4f}")  # p < 0.05 表示显著差异
```

### 6.2 消融实验瀑布图

使用结果CSV生成瀑布图:

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('epstd_ablation_*.csv')
full_pai = df.loc[df['Variant'] == 'Full_L-EPSTD', 'PAI'].values[0]

fig, ax = plt.subplots(figsize=(10, 6))
variants = df['Variant'].tolist()[1:]  # 排除Full
pai_drops = [full_pai - df.loc[df['Variant'] == v, 'PAI'].values[0] for v in variants]

ax.barh(variants, pai_drops)
ax.set_xlabel('PAI Drop')
ax.set_title('Component Contribution Analysis')
plt.tight_layout()
plt.savefig('ablation_waterfall.png')
```

### 6.3 对比实验雷达图

```python
import numpy as np
import matplotlib.pyplot as plt

models = ['HA', 'RF', 'STGCN', 'EP-STD']
metrics = ['Correlation', 'PAI', 'CS_Recall', 'HitRate']
values = [
    [0.45, 1.2, 0.25, 0.30],  # HA
    [0.58, 1.5, 0.35, 0.40],  # RF
    [0.68, 2.0, 0.45, 0.50],  # STGCN
    [0.78, 2.5, 0.65, 0.68],  # EP-STD
]

# 归一化到0-1
values_norm = np.array(values) / np.max(values, axis=0)

angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
for i, (model, vals) in enumerate(zip(models, values_norm)):
    vals = vals.tolist() + vals[:1].tolist()
    ax.plot(angles, vals, 'o-', linewidth=2, label=model)
    ax.fill(angles, vals, alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics)
ax.legend()
plt.savefig('comparison_radar.png')
```

---

## 7. 常见问题

### Q1: 消融实验运行时间过长怎么办？

**解决方案**:
```python
# 减少epoch数
AblationConfig.EPOCHS = 30  # 默认50

# 减少变体数量
ABLATION_VARIANTS = {
    'Full_L-EPSTD': {...},
    'w/o_Logic_Guidance': {...},
    'w/o_Adaptive_Fusion': {...},
    # 只保留关键变体
}

# 使用单轮而非多轮
AblationConfig.SEEDS = [42]  # 只运行一次
```

### Q2: 显存不足怎么办？

**解决方案**:
```python
# 减小批量大小
AblationConfig.BATCH_SIZE = 8  # 默认16

# 减小模型规模
AblationConfig.HIDDEN_DIM = 64  # 默认128
AblationConfig.NUM_LAYERS = 2   # 默认4

# 使用CPU运行
AblationConfig.DEVICE = 'cpu'
```

### Q3: 基线模型报错找不到模块？

**解决方案**:
```bash
# 确保baselines模块存在
ls src/baselines.py

# 如果不存在，创建简化版基线
python -c "
import sys
sys.path.insert(0, 'src')
from baselines import HistoricalAverage, RandomForestPredictor
print('Baselines imported successfully')
"
```

### Q4: 如何添加新的消融变体？

**示例**: 添加"w/o_TimeEncoding"变体

```python
ABLATION_VARIANTS = {
    # ... 现有变体 ...
    
    'w/o_TimeEncoding': {
        'description': '移除时间编码',
        'use_logic_guidance': True,
        'use_adaptive_fusion': True,
        'use_cross_crime_gate': True,
        'use_multi_graph': True,
        'use_env_encoder': True,
        'use_zero_inflation': True,
        'use_prototype': True,
        'use_time_encoding': False,  # 新添加的参数
    }
}

# 在AblationModelWrapper中处理
if not config.get('use_time_encoding', True):
    # 修改模型以禁用时间编码
    pass
```

### Q5: 结果与预期不符怎么办？

**排查步骤**:
1. 检查数据加载是否正确
2. 验证检查点文件是否加载成功
3. 检查损失函数是否收敛
4. 对比单轮结果与多轮结果的一致性
5. 检查是否有数据泄露（训练集/测试集划分）

---

## 附录

### A. 快速测试命令

```bash
# 测试消融实验（快速版，单变体）
python -c "
from experiments.epstd_ablation_study import ABLATION_VARIANTS
print('Ablation variants:', list(ABLATION_VARIANTS.keys()))
"

# 测试对比实验（快速版）
python -c "
from experiments.epstd_baseline_comparison import ComparisonConfig
print('Device:', ComparisonConfig.DEVICE)
print('Epochs:', ComparisonConfig.EPOCHS)
"
```

### B. 实验结果归档结构

```
experiments/results/
├── ablation/
│   ├── epstd_ablation_*.csv
│   ├── epstd_ablation_*.md
│   └── epstd_ablation_*.json
├── comparison/
│   ├── baseline_comparison_*.csv
│   ├── baseline_comparison_*.md
│   └── comparison_full_results_*.csv
└── masking/
    ├── masking_experiment_*.npz
    └── masking_experiment_*.png
```

### C. 引用格式

**消融实验**:
```bibtex
@article{epstd2025,
  title={Cold-Start Crime Prediction via Environment-Guided Spatio-Temporal Diffusion},
  author={...},
  journal={...},
  year={2025}
}
```

**实验代码**:
```
Code available at: https://github.com/.../experiments/
```

---

**文档维护**: 如有问题请更新此文档
