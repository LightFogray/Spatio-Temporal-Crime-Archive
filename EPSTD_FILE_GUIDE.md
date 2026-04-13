# EP-STD 文件指南

本指南汇总了EP-STD (Environment-Prompted Spatio-Temporal Diffusion) 项目的所有新增文件及其功能。

## 一、核心架构文件 (src/)

### Stage 1: 环境编码器
| 文件 | 功能 | 关键类/函数 |
|------|------|------------|
| `src/epstd_stage1.py` | 对比学习环境编码器 | `EnvironmentEncoder`, `ContrastiveDataset`, `info_nce_loss`, `train_contrastive_encoder` |

**输入输出**:
- 输入: 静态环境特征 (N, 24) - POI、CPTED、道路等
- 输出: 环境嵌入 (N, 64) + 可视化PCA图

---

### Stage 2: 原型学习
| 文件 | 功能 | 关键类/函数 |
|------|------|------------|
| `src/epstd_stage2.py` | 基于PyTorch的K-Means原型学习 | `PrototypeLibrary`, `FuzzyRiskPrior`, `test_cold_start_prediction` |

**输入输出**:
- 输入: 环境嵌入 (N, 64)
- 输出: 
  - `checkpoints/prototype_library.pkl` - 原型库
  - `data/processed/prototype_labels.npy` - 网格原型标签
  - `data/processed/prototype_centers.npy` - 原型中心
  - 冷启动预测MAE和相关性

---

### Stage 3: 扩散模型 (基础版)
| 文件 | 功能 | 关键类/函数 |
|------|------|------------|
| `src/epstd_stage3.py` | 条件扩散模型 | `ConditionalRiskDiffusion`, `DiffusionScheduler`, `GraphAttentionLayer`, `EPSTDPredictor` |

**输入输出**:
- 输入: 噪声风险 + 环境条件 + 原型ID
- 输出: 去噪后的风险预测 + 不确定性 + 零膨胀概率

---

### Stage 3 Enhanced: 逻辑引导扩散模型 (核心创新)
| 文件 | 功能 | 关键类/函数 |
|------|------|------------|
| `src/epstd_stage3_enhanced.py` | **带逻辑引导的增强扩散模型** | `SoftLogic`, `LogicConstraintCalculator`, `LogicGuidedDiffusionScheduler`, `LogicGuidedPredictor` |

**核心创新**:
1. **T-Norm软逻辑**: `SoftLogic` 类实现Goguen/Reichenbach蕴涵，将布尔规则转化为可微损失
2. **Classifier Guidance**: `denoise_step_with_guidance()` 在去噪过程中引入逻辑约束梯度引导
3. **双向约束**: 惩罚漏报(环境高风险但预测低)和误报(环境低风险但预测高)
4. **退火策略**: `logic_warmup_epochs` 参数控制逻辑约束的渐进引入

**数学公式**:
```
# T-Norm软逻辑蕴涵
I(a,b) = min(1, b/a)  # Goguen蕴涵

# Classifier Guidance去噪
x_{t-1} = x_t - η[ε_θ(x_t, t, c) + ω·∇_{x_t}L_logic]

# 逻辑损失（双向）
L_logic = L_under(pred_low | env_high) + L_over(pred_high | env_low)
```

---

### 完整流程
| 文件 | 功能 |
|------|------|
| `src/epstd_full_pipeline.py` | 三阶段串联的命令行工具，支持`--stage all/1/2/3` |

---

## 二、实验诊断文件 (experiments/)

### 冷启动能力诊断
| 文件 | 功能 | 输出指标 |
|------|------|---------|
| `experiments/cold_start_simple.py` | 简化版冷启动诊断（纯NumPy） | 环境-风险相关性、冷启动网格检测、环境相似一致性 |
| `experiments/cold_start_analysis.py` | 完整版冷启动诊断（需加载ST模型） | 零历史热点召回率、跨网格对齐分析、高风险低历史识别 |

---

### 压力测试 (核心实验)
| 文件 | 功能 | 实验设计 |
|------|------|---------|
| `experiments/masking_experiment.py` | **人为屏蔽实验** | 1. 屏蔽20%高犯罪网格历史<br>2. 对比L-EPSTD vs 基线<br>3. 评估冷启动性能(MAE/Corr/Recall) |

**预期结果**:
- 基线STGCN: 在masked网格上预测→0，完全失效
- L-EPSTD(无逻辑): 部分识别能力
- L-EPSTD(有逻辑): 保持识别能力，证明环境信号有效

---

## 三、生成文件 (运行后)

### 检查点
| 文件 | 说明 |
|------|------|
| `checkpoints/env_encoder_best.pt` | Stage 1环境编码器 |
| `checkpoints/prototype_library.pkl` | Stage 2原型库 |
| `checkpoints/epstd_diffusion_best.pt` | Stage 3基础扩散模型 |
| `checkpoints/logic_guided_diffusion_best.pt` | **Stage 3增强版** (带逻辑引导) |

### 数据文件
| 文件 | 说明 |
|------|------|
| `data/processed/env_embeddings.npy` | (1246, 64) 环境嵌入 |
| `data/processed/prototype_labels.npy` | (1246,) 网格原型标签 |
| `data/processed/prototype_centers.npy` | (10, 64) 原型中心 |
| `data/processed/epstd_predictions.npy` | EP-STD预测结果 |
| `data/processed/epstd_uncertainty.npy` | 预测不确定性 |
| `data/processed/epstd_zero_prob.npy` | 零犯罪概率 |

### 可视化
| 文件 | 说明 |
|------|------|
| `env_embeddings_pca.png` | 环境嵌入PCA可视化 |
| `prototypes_visualization.png` | 原型聚类可视化 |
| `experiments/results/masking_experiment_*.npz` | 压力测试结果 |
| `experiments/results/masking_experiment_visualization.png` | 压力测试可视化 |

---

## 四、使用示例

### 完整训练流程
```bash
# 运行全部三阶段
python src/epstd_full_pipeline.py --stage all --epochs 100

# 仅运行增强版Stage 3（假设1、2已完成）
python src/epstd_stage3_enhanced.py
```

### 压力测试
```bash
# 运行人为屏蔽实验
python experiments/masking_experiment.py
```

### 使用L-EPSTD进行预测
```python
from src.epstd_stage3_enhanced import LogicGuidedPredictor

predictor = LogicGuidedPredictor(
    model, scheduler, logic_calculator,
    env_encoder, prototype_library
)

# 带逻辑引导预测
risk_mean, risk_std, pi = predictor.predict(
    static_features,
    use_logic_guidance=True,
    guidance_scale=1.0
)
```

---

## 五、技术要点总结

### 1. T-Norm软逻辑
- **解决的问题**: 将布尔规则(如"商业高且监控低→高风险")转化为可微损失
- **实现**: `SoftLogic.evaluate_rule()` 使用Goguen蕴涵
- **优势**: 二阶可导，利于神经网络训练

### 2. Classifier Guidance
- **解决的问题**: 逻辑约束从"被动惩罚结果"变为"主动引导去噪"
- **实现**: `denoise_step_with_guidance()` 计算∇_{x_t}L_logic
- **优势**: 在冷启动区域"无中生有"，赋予确定性方向

### 3. 双向约束
- **漏报惩罚**: 环境高风险但预测低 → 大惩罚
- **误报惩罚**: 环境低风险但预测高 → 大惩罚
- **效果**: 防止"狼来了"效应(全城高报)或过度保守(全城低报)

### 4. 退火策略
- **Warmup**: 前20epoch λ_logic=0，让模型先学基本时空模式
- **渐进**: 之后线性增加到1.0，逐步引入逻辑约束
- **目的**: 防止初期逻辑冲突导致训练崩溃

---

## 六、后续优化方向

### 潜在空间扩散 (Latent Diffusion)
- **当前**: 在1246维网格空间做扩散
- **优化**: 在10维原型空间做扩散，再解码回网格
- **收益**: 显存占用降低70%，收敛更快
- **风险**: 可能损失细粒度空间信息

### 多任务学习
- **当前**: 单任务(暴力犯罪)
- **优化**: 同时输出暴力+财产犯罪
- **收益**: 利用财产犯罪高频数据增强环境编码器

---

*最后更新: 2026-04-12*
