"""
论文图表可视化脚本
生成雷达图、消融瀑布图、性能对比图等期刊级图表
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# 期刊常用配色
COLORS = {
    'primary': '#2E86AB',      # 主色
    'secondary': '#A23B72',    # 辅助色
    'accent': '#F18F01',       # 强调色
    'neutral': '#C73E1D',      # 中性色
    'light': '#E8F1F2',        # 浅色背景
    'dark': '#1B2631',         # 深色文字
    'baseline': '#95A5A6',     # baseline灰色
    'ours': '#E74C3C'          # 我们的方法红色
}


# ================================
# 1. 雷达图 (多维度综合对比)
# ================================

def plot_radar_comparison(results_df: pd.DataFrame,
                          models: List[str],
                          metrics: List[str] = ['PAI', 'HR@10%', 'Jaccard'],
                          save_path: Optional[str] = None):
    """
    绘制多模型雷达对比图

    Args:
        results_df: 包含各模型指标的DataFrame
        models: 要对比的模型列表
        metrics: 要显示的指标（越高越好）
        save_path: 保存路径
    """
    # 提取数据并归一化
    data = results_df.loc[models, metrics].values

    # 归一化到0-1范围
    data_normalized = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-8)

    # 角度
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    # 绘制每个模型
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for i, (model, color) in enumerate(zip(models, colors)):
        values = data_normalized[i].tolist()
        values += values[:1]  # 闭合

        # 我们的方法用特殊颜色
        if 'ours' in model.lower() or 'our' in model.lower():
            color = COLORS['ours']
            linewidth = 2.5
            alpha = 1.0
        else:
            linewidth = 1.5
            alpha = 0.7

        ax.plot(angles, values, 'o-', linewidth=linewidth,
                label=model, color=color, alpha=alpha)
        ax.fill(angles, values, alpha=0.15, color=color)

    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title('Multi-metric Performance Comparison\n(Normalized)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Radar plot saved to: {save_path}")

    plt.show()


# ================================
# 2. 消融瀑布图
# ================================

def plot_ablation_waterfall(ablation_df: pd.DataFrame,
                            metric: str = 'PAI',
                            save_path: Optional[str] = None):
    """
    绘制消融实验瀑布图
    展示从Full Model逐个移除组件的性能变化

    Args:
        ablation_df: 消融实验结果DataFrame
        metric: 要可视化的指标
        save_path: 保存路径
    """
    # 定义顺序：从完整模型开始，逐个移除
    ablation_order = [
        'Full_Model',
        'w/o_Cross_Fusion',
        'w/o_Hypergraph',
        'w/o_Near_Repeat',
        'w/o_Semantic_Gate',
        'w/o_Semantic'
    ]

    # 过滤存在的变体
    available_variants = [v for v in ablation_order if v in ablation_df.index]

    # 提取值
    values = ablation_df.loc[available_variants, metric].values

    # 计算变化量
    changes = np.diff(values)
    changes = np.insert(changes, 0, 0)  # Full Model没有变化

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))

    # 颜色：性能下降用红色，提升用绿色
    colors = [COLORS['primary'] if i == 0 else
              COLORS['neutral'] if changes[i] > 0 and metric in ['RMSE', 'MAE'] else
              COLORS['accent'] if changes[i] < 0 and metric in ['PAI', 'HR@10%', 'Jaccard'] else
              '#27AE60'
              for i in range(len(available_variants))]

    # 绘制条形
    bars = ax.bar(range(len(available_variants)), values, color=colors,
                   edgecolor='black', linewidth=1.2, alpha=0.8)

    # 添加数值标签
    for i, (bar, val, change) in enumerate(zip(bars, values, changes)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 添加变化箭头
        if i > 0:
            is_degradation = (change > 0 and metric in ['RMSE', 'MAE']) or \
                            (change < 0 and metric not in ['RMSE', 'MAE'])
            arrow = '↓' if is_degradation else '↑'
            color = 'red' if is_degradation else 'green'
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{arrow} {abs(change):.4f}',
                    ha='center', va='bottom', fontsize=8, color=color)

    # 设置标签
    labels = [v.replace('_', '\n') for v in available_variants]
    ax.set_xticks(range(len(available_variants)))
    ax.set_xticklabels(labels, fontsize=9, rotation=0)
    ax.set_ylabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'Ablation Study: Component Contribution\n({metric} Performance)',
                 fontsize=14, fontweight='bold')

    # 添加网格
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['primary'], label='Full Model'),
        Patch(facecolor=COLORS['neutral'], label='Performance Drop'),
        Patch(facecolor='#27AE60', label='Performance Gain')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Ablation waterfall plot saved to: {save_path}")

    plt.show()


# ================================
# 3. 性能对比柱状图
# ================================

def plot_performance_comparison(results_df: pd.DataFrame,
                                metrics: List[str] = ['RMSE', 'MAE', 'PAI'],
                                highlight_model: str = 'Ours',
                                save_path: Optional[str] = None):
    """
    绘制多指标性能对比柱状图

    Args:
        results_df: 结果DataFrame
        metrics: 要对比的指标
        highlight_model: 要高亮的模型（我们的方法）
        save_path: 保存路径
    """
    n_metrics = len(metrics)
    models = results_df.index.tolist()

    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]

    for idx, (metric, ax) in enumerate(zip(metrics, axes)):
        values = results_df[metric].values

        # 颜色设置
        colors = [COLORS['ours'] if model == highlight_model else COLORS['baseline']
                  for model in models]

        # 绘制柱状图
        bars = ax.bar(range(len(models)), values, color=colors,
                     edgecolor='black', linewidth=1, alpha=0.85)

        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom', fontsize=9)

        # 设置标签
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # 最佳值高亮
        if metric in ['RMSE', 'MAE', 'MAPE']:
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)

        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)

    plt.suptitle('Baseline Performance Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance comparison plot saved to: {save_path}")

    plt.show()


# ================================
# 4. 改进幅度条形图
# ================================

def plot_improvement_bars(baseline_metrics: Dict[str, float],
                          our_metrics: Dict[str, float],
                          save_path: Optional[str] = None):
    """
    绘制相对于baseline的改进幅度

    Args:
        baseline_metrics: 最佳baseline的各项指标
        our_metrics: 我们方法的各项指标
        save_path: 保存路径
    """
    metrics = list(baseline_metrics.keys())

    # 计算改进百分比
    improvements = []
    for metric in metrics:
        base_val = baseline_metrics[metric]
        our_val = our_metrics[metric]

        if metric in ['RMSE', 'MAE', 'MAPE']:
            # 越低越好
            imp = (base_val - our_val) / base_val * 100
        else:
            # 越高越好
            imp = (our_val - base_val) / base_val * 100

        improvements.append(imp)

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))

    # 颜色
    colors = [COLORS['accent'] if imp > 0 else COLORS['neutral'] for imp in improvements]

    # 绘制水平条形图
    bars = ax.barh(metrics, improvements, color=colors,
                   edgecolor='black', linewidth=1.2, alpha=0.85)

    # 添加数值标签
    for bar, imp in zip(bars, improvements):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'{imp:+.2f}%',
               ha='left' if imp > 0 else 'right',
               va='center', fontsize=10, fontweight='bold')

    # 添加零线
    ax.axvline(x=0, color='black', linewidth=1.5, linestyle='-')

    # 设置标签
    ax.set_xlabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Improvement over Best Baseline',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['accent'], label='Improvement'),
        Patch(facecolor=COLORS['neutral'], label='Degradation')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Improvement plot saved to: {save_path}")

    plt.show()


# ================================
# 5. 时间注意力可视化
# ================================

def plot_temporal_attention(attention_weights: np.ndarray,
                            time_labels: Optional[List[str]] = None,
                            save_path: Optional[str] = None):
    """
    可视化时间注意力热力图

    Args:
        attention_weights: (T, T) 注意力权重矩阵
        time_labels: 时间标签
        save_path: 保存路径
    """
    T = attention_weights.shape[0]

    if time_labels is None:
        time_labels = [f't-{T-i}' for i in range(T)]

    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制热力图
    mask = np.triu(np.ones_like(attention_weights, dtype=bool), k=1)  # 上三角掩码
    sns.heatmap(attention_weights, mask=mask, cmap='YlOrRd',
                square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                xticklabels=time_labels, yticklabels=time_labels,
                ax=ax, vmin=0, vmax=attention_weights.max())

    ax.set_xlabel('Key Time Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Query Time Step', fontsize=12, fontweight='bold')
    ax.set_title('Causal Temporal Self-Attention Pattern\n(Lower Triangular: Past Information Only)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Temporal attention plot saved to: {save_path}")

    plt.show()


# ================================
# 6. 特征重要性条形图
# ================================

def plot_feature_importance(feature_names: List[str],
                            importance_values: np.ndarray,
                            top_k: int = 15,
                            save_path: Optional[str] = None):
    """
    绘制特征重要性条形图（基于SHAP或梯度）

    Args:
        feature_names: 特征名称列表
        importance_values: 重要性值
        top_k: 显示前k个特征
        save_path: 保存路径
    """
    # 排序
    indices = np.argsort(importance_values)[-top_k:]

    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制水平条形图
    colors = plt.cm.RdYlBu(np.linspace(0.2, 0.8, top_k))
    bars = ax.barh(range(top_k), importance_values[indices], color=colors,
                   edgecolor='black', linewidth=1, alpha=0.85)

    # 设置标签
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Top-{top_k} Feature Importance (SHAP)',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # 添加数值标签
    for i, (bar, idx) in enumerate(zip(bars, indices)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'{width:.3f}',
               ha='left', va='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to: {save_path}")

    plt.show()


# ================================
# 7. 语义门控空间分布图
# ================================

def plot_semantic_gate_distribution(gate_values: np.ndarray,
                                    grid_coords: Optional[np.ndarray] = None,
                                    save_path: Optional[str] = None):
    """
    绘制语义门控权重的分布和空间热力图

    Args:
        gate_values: (N,) 每个网格的门控权重
        grid_coords: (N, 2) 网格坐标
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 子图1：直方图
    ax1 = axes[0]
    ax1.hist(gate_values, bins=30, color=COLORS['primary'],
             edgecolor='black', alpha=0.7)
    ax1.axvline(gate_values.mean(), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {gate_values.mean():.3f}')
    ax1.axvline(np.median(gate_values), color='orange', linestyle='--',
                linewidth=2, label=f'Median: {np.median(gate_values):.3f}')
    ax1.set_xlabel('Semantic Gate Weight', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax1.set_title('Distribution of Semantic Gate Weights',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # 子图2：空间分布
    ax2 = axes[1]
    if grid_coords is not None:
        scatter = ax2.scatter(grid_coords[:, 0], grid_coords[:, 1],
                             c=gate_values, cmap='RdYlBu_r', s=30,
                             alpha=0.7, edgecolors='black', linewidth=0.5)
        cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8)
        cbar.set_label('Gate Weight', fontsize=10)
        ax2.set_xlabel('X Coordinate', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Y Coordinate', fontsize=11, fontweight='bold')
        ax2.set_title('Spatial Distribution of Semantic Gate Weights\n(High=Heavy reliance on LLM semantics)',
                      fontsize=12, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'Grid coordinates not provided',
                ha='center', va='center', fontsize=12)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Semantic gate distribution plot saved to: {save_path}")

    plt.show()


# ================================
# 8. 近重复效应衰减曲线
# ================================

def plot_near_repeat_decay(spatial_decay: float,
                           temporal_decay: float,
                           max_dist: float = 1000,
                           max_time: float = 30,
                           save_path: Optional[str] = None):
    """
    绘制近重复效应的时空衰减曲线

    Args:
        spatial_decay: 空间衰减参数
        temporal_decay: 时间衰减参数
        max_dist: 最大距离（米或网格单位）
        max_time: 最大时间（天）
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 空间衰减
    distances = np.linspace(0, max_dist, 100)
    spatial_effect = np.exp(-spatial_decay * distances)

    ax1 = axes[0]
    ax1.fill_between(distances, spatial_effect, alpha=0.3, color=COLORS['primary'])
    ax1.plot(distances, spatial_effect, linewidth=2.5, color=COLORS['primary'])
    ax1.axhline(y=0.1, color='red', linestyle='--', linewidth=1.5,
                label='10% threshold')

    # 标记关键距离
    radius_50 = -np.log(0.5) / spatial_decay
    radius_10 = -np.log(0.1) / spatial_decay
    ax1.axvline(x=radius_50, color='orange', linestyle=':', alpha=0.7)
    ax1.axvline(x=radius_10, color='red', linestyle=':', alpha=0.7)

    ax1.set_xlabel('Distance (meters)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Effect Intensity', fontsize=12, fontweight='bold')
    ax1.set_title(f'Spatial Decay (α={spatial_decay:.4f})\n'
                  f'50% radius: {radius_50:.0f}m, 10% radius: {radius_10:.0f}m',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max_dist)
    ax1.set_ylim(0, 1.05)

    # 时间衰减
    times = np.linspace(0, max_time, 100)
    temporal_effect = np.exp(-temporal_decay * times)

    ax2 = axes[1]
    ax2.fill_between(times, temporal_effect, alpha=0.3, color=COLORS['secondary'])
    ax2.plot(times, temporal_effect, linewidth=2.5, color=COLORS['secondary'])
    ax2.axhline(y=0.1, color='red', linestyle='--', linewidth=1.5,
                label='10% threshold')

    # 标记关键时间
    window_50 = -np.log(0.5) / temporal_decay
    window_10 = -np.log(0.1) / temporal_decay
    ax2.axvline(x=window_50, color='orange', linestyle=':', alpha=0.7)
    ax2.axvline(x=window_10, color='red', linestyle=':', alpha=0.7)

    ax2.set_xlabel('Time (days)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Effect Intensity', fontsize=12, fontweight='bold')
    ax2.set_title(f'Temporal Decay (β={temporal_decay:.4f})\n'
                  f'50% window: {window_50:.1f}d, 10% window: {window_10:.1f}d',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_time)
    ax2.set_ylim(0, 1.05)

    plt.suptitle('Learned Near-Repeat Effect Decay Parameters',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Near-repeat decay plot saved to: {save_path}")

    plt.show()


# ================================
# 9. 预测vs真实对比热力图
# ================================

def plot_prediction_heatmap(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            grid_coords: np.ndarray,
                            sample_idx: int = 0,
                            save_path: Optional[str] = None):
    """
    绘制预测vs真实的空间热力图对比

    Args:
        y_true: (T, N) 真实值
        y_pred: (T, N) 预测值
        grid_coords: (N, 2) 网格坐标
        sample_idx: 要展示的时间步
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    vmax = max(y_true[sample_idx].max(), y_pred[sample_idx].max())

    # 真实值
    ax1 = axes[0]
    scatter1 = ax1.scatter(grid_coords[:, 0], grid_coords[:, 1],
                          c=y_true[sample_idx], cmap='YlOrRd', s=50,
                          alpha=0.8, vmin=0, vmax=vmax,
                          edgecolors='black', linewidth=0.5)
    ax1.set_title('Ground Truth', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    plt.colorbar(scatter1, ax=ax1, shrink=0.8, label='Crime Count')

    # 预测值
    ax2 = axes[1]
    scatter2 = ax2.scatter(grid_coords[:, 0], grid_coords[:, 1],
                          c=y_pred[sample_idx], cmap='YlOrRd', s=50,
                          alpha=0.8, vmin=0, vmax=vmax,
                          edgecolors='black', linewidth=0.5)
    ax2.set_title('Prediction', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    plt.colorbar(scatter2, ax=ax2, shrink=0.8, label='Crime Count')

    # 残差
    ax3 = axes[2]
    residual = y_pred[sample_idx] - y_true[sample_idx]
    scatter3 = ax3.scatter(grid_coords[:, 0], grid_coords[:, 1],
                          c=residual, cmap='RdBu_r', s=50,
                          alpha=0.8, vmin=-residual.std()*3, vmax=residual.std()*3,
                          edgecolors='black', linewidth=0.5)
    ax3.set_title('Residual (Pred - True)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('X Coordinate')
    ax3.set_ylabel('Y Coordinate')
    plt.colorbar(scatter3, ax=ax3, shrink=0.8, label='Residual')

    plt.suptitle(f'Spatial Prediction Quality (Sample {sample_idx})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction heatmap saved to: {save_path}")

    plt.show()


# ================================
# 10. 综合图表生成函数
# ================================

def generate_all_figures(results_csv: str, ablation_csv: Optional[str] = None,
                         output_dir: str = "experiments/figures"):
    """
    根据结果文件生成所有图表

    Args:
        results_csv: baseline结果CSV文件路径
        ablation_csv: 消融实验结果CSV文件路径
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    results_df = pd.read_csv(results_csv, index_col=0)

    print("Generating figures...")

    # 1. 雷达图
    models = results_df.index.tolist()
    plot_radar_comparison(
        results_df, models,
        save_path=f"{output_dir}/radar_comparison.png"
    )

    # 2. 性能对比柱状图
    plot_performance_comparison(
        results_df,
        metrics=['RMSE', 'MAE', 'PAI', 'HR@10%', 'Jaccard'],
        save_path=f"{output_dir}/performance_bars.png"
    )

    # 3. 改进幅度图
    # 假设最佳baseline是除Ours外的最低RMSE
    baseline_candidates = [m for m in results_df.index if 'ours' not in m.lower()]
    best_baseline = results_df.loc[baseline_candidates, 'RMSE'].idxmin()

    baseline_metrics = results_df.loc[best_baseline].to_dict()
    our_metrics = results_df.loc['Ours'].to_dict() if 'Ours' in results_df.index else results_df.iloc[-1].to_dict()

    plot_improvement_bars(
        baseline_metrics, our_metrics,
        save_path=f"{output_dir}/improvement_bars.png"
    )

    # 4. 消融实验图
    if ablation_csv and os.path.exists(ablation_csv):
        ablation_df = pd.read_csv(ablation_csv, index_col=0)
        plot_ablation_waterfall(
            ablation_df, metric='PAI',
            save_path=f"{output_dir}/ablation_waterfall.png"
        )

    print(f"\nAll figures saved to: {output_dir}/")


# ================================
# 示例用法
# ================================

if __name__ == "__main__":
    print("=" * 60)
    print("Visualization Tools for Crime Prediction Paper")
    print("=" * 60)

    # 示例：生成示例图表
    print("\nGenerating example visualizations...")

    # 模拟数据
    np.random.seed(42)

    # 雷达图示例
    example_df = pd.DataFrame({
        'RMSE': [12.5, 10.3, 8.5, 8.2, 7.8, 7.5, 7.2],
        'MAE': [8.5, 7.2, 6.0, 5.8, 5.5, 5.3, 4.9],
        'PAI': [1.1, 1.4, 1.8, 1.9, 2.0, 2.1, 2.3],
        'HR@10%': [0.45, 0.52, 0.62, 0.64, 0.67, 0.70, 0.74],
        'Jaccard': [0.28, 0.35, 0.43, 0.45, 0.48, 0.51, 0.55]
    }, index=['HA', 'RF', 'ConvLSTM', 'STGCN', 'DCRNN', 'GraphWaveNet', 'Ours'])

    os.makedirs("experiments/figures", exist_ok=True)

    plot_radar_comparison(example_df, example_df.index.tolist(),
                         save_path="experiments/figures/example_radar.png")

    print("\nExample figures generated!")
    print("Use generate_all_figures() with your actual results.")
    print("=" * 60)
