"""
可解释性分析模块 - 支持注意力导出、SHAP、Integrated Gradients
结合环境犯罪学理论的可解释性分析
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not installed. Some explanation functions will be limited. pip install shap")


# ================================
# 1. SHAP解释器封装
# ================================
class CrimePredictionExplainer:
    """
    犯罪预测模型的综合解释器
    支持SHAP、注意力可视化、特征重要性分析
    """

    def __init__(self, model: nn.Module, background_data: torch.Tensor,
                 feature_names: Optional[Dict[str, List[str]]] = None,
                 device: str = 'cuda'):
        """
        Args:
            model: 训练好的时空Transformer模型
            background_data: 背景数据用于SHAP的base值计算 (建议50-100个样本)
            feature_names: 特征名称字典，如 {'static': ['poi_commercial', ...], 'dynamic': [...]}
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.background_data = background_data.to(device)
        self.feature_names = feature_names or self._default_feature_names()
        self.forward_kwargs = {}  # 额外的模型前向参数

        # 图数据占位符（通过set_graph_data设置）
        self.A_spatial = None
        self.A_distance = None
        self.A_hypergraph = None
        self.semantic_embed = None
        self.OD_placeholder = None

        # 初始化SHAP解释器
        if SHAP_AVAILABLE:
            self.shap_explainer = self._init_shap_explainer()
        else:
            self.shap_explainer = None

    def _default_feature_names(self) -> Dict[str, List[str]]:
        """默认特征命名"""
        return {
            'static': [
                'poi_commercial', 'poi_transport', 'poi_public',
                'road_density', 'landuse_res', 'landuse_com', 'landuse_ind',
                'green_ratio', 'nightlight', 'camera_count',
                # 扩展更多静态特征...
            ],
            'dynamic': [
                'crime_lag_1d', 'crime_lag_2d', 'crime_lag_3d',
                'crime_lag_4d', 'crime_lag_5d', 'crime_lag_6d', 'crime_lag_7d',
                'temp_avg', 'rain_freq', 'snow_flag'
            ],
            'semantic': ['llm_semantic_embedding']  # LLM语义特征
        }

    def _init_shap_explainer(self):
        """初始化DeepExplainer (适用于PyTorch)"""
        # 使用DeepExplainer需要模型支持
        # 由于我们的模型有多个输入，需要包装
        return shap.DeepExplainer(self._wrap_model_for_shap(), self.background_data)

    def _wrap_model_for_shap(self):
        """
        包装模型以适配SHAP的输入格式
        SHAP期望单一tensor输入，我们需要将多输入打包
        """
        class WrappedModel(nn.Module):
            def __init__(self, original_model, A_spatial, A_distance, A_hypergraph,
                         semantic_embed, OD_placeholder):
                super().__init__()
                self.model = original_model
                self.A_spatial = A_spatial
                self.A_distance = A_distance
                self.A_hypergraph = A_hypergraph
                self.semantic_embed = semantic_embed
                self.OD_placeholder = OD_placeholder

            def forward(self, X_packed):
                """
                X_packed: 包含所有输入特征的tensor
                需要解包并调用原始模型
                """
                # 从packed tensor中提取各个部分
                # 假设前static_dim维是静态特征，后面是动态特征
                B, T, N, F = X_packed.shape

                # 创建占位符
                A_crime = torch.eye(N, device=X_packed.device).unsqueeze(0).expand(B, -1, -1)

                # 提取犯罪历史用于近重复效应
                crime_history = X_packed[:, :, :, -7:]
                crime_history = crime_history[:, :, :, 0]

                pi, mu, theta, _, _ = self.model(
                    X_packed, self.A_spatial, self.A_distance,
                    A_crime, self.A_hypergraph, self.OD_placeholder[:B],
                    semantic_embed=self.semantic_embed,
                    crime_history=crime_history
                )

                # 返回期望犯罪数作为预测值
                return (1 - pi) * mu

        return WrappedModel(
            self.model,
            self.A_spatial, self.A_distance, self.A_hypergraph,
            self.semantic_embed, self.OD_placeholder
        )

    def set_graph_data(self, A_spatial, A_distance, A_hypergraph,
                       semantic_embed=None, OD_placeholder=None):
        """
        设置图结构和辅助数据
        用于explain_feature_importance_global等批量分析方法
        """
        self.A_spatial = A_spatial.to(self.device)
        self.A_distance = A_distance.to(self.device)
        self.A_hypergraph = A_hypergraph.to(self.device)
        self.semantic_embed = semantic_embed.to(self.device) if semantic_embed is not None else None
        self.OD_placeholder = OD_placeholder.to(self.device) if OD_placeholder is not None else None

    def set_model_forward_kwargs(self, **kwargs):
        """
        设置模型前向传播的额外参数
        用于适配不同版本的模型API
        """
        self.forward_kwargs = kwargs

    def explain_with_shap(self, X_sample: torch.Tensor,
                         grid_ids: Optional[List[int]] = None) -> Dict:
        """
        使用SHAP解释预测结果

        Args:
            X_sample: 待解释的样本 (B, T, N, F)
            grid_ids: 指定关注的网格ID，None则解释所有网格

        Returns:
            shap_values: SHAP值字典
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            raise RuntimeError("SHAP not available. Install with: pip install shap")

        X_sample = X_sample.to(self.device)

        # 计算SHAP值
        shap_values = self.shap_explainer.shap_values(X_sample)

        # 整理结果
        results = {
            'shap_values': shap_values,  # (B, T, N, F)
            'base_values': self.shap_explainer.expected_value,
            'feature_names': self._flatten_feature_names(),
            'input_data': X_sample.cpu().numpy()
        }

        # 如果指定了grid_ids，提取对应结果
        if grid_ids is not None:
            results['target_grids'] = grid_ids
            results['grid_shap_values'] = shap_values[:, :, grid_ids, :]

        return results

    def explain_feature_importance_global(self, test_loader,
                                          max_samples: int = 100) -> Dict[str, np.ndarray]:
        """
        全局特征重要性分析 (基于梯度)

        Args:
            test_loader: 测试数据加载器
            max_samples: 最大分析样本数

        Returns:
            importance_dict: 各特征维度的重要性分数
        """
        self.model.eval()
        importance_accum = []
        sample_count = 0
        epsilon = 0.01  # 扰动幅度

        print(f"Computing feature importance using perturbation method (epsilon={epsilon})...")

        for X_batch, A_crime_batch, OD_batch, Y_batch in test_loader:
            if sample_count >= max_samples:
                break

            X_batch = X_batch.to(self.device)
            B, T, N, F = X_batch.shape

            with torch.no_grad():
                # 原始预测
                pi_orig, mu_orig, _, _, _ = self.model(
                    X_batch, self.A_spatial, self.A_distance,
                    A_crime_batch.to(self.device), self.A_hypergraph, OD_batch.to(self.device),
                    semantic_embed=self.semantic_embed
                )
                pred_orig = ((1 - pi_orig) * mu_orig).sum(dim=(0, 1))  # 对每个样本和时间聚合

                # 对每个特征维度进行扰动
                batch_importance = []
                for i in range(F):
                    X_perturbed = X_batch.clone()
                    X_perturbed[..., i] += epsilon

                    pi_new, mu_new, _, _, _ = self.model(
                        X_perturbed, self.A_spatial, self.A_distance,
                        A_crime_batch.to(self.device), self.A_hypergraph, OD_batch.to(self.device),
                        semantic_embed=self.semantic_embed
                    )
                    pred_new = ((1 - pi_new) * mu_new).sum(dim=(0, 1))

                    # 绝对变化作为重要性
                    importance_i = (pred_new - pred_orig).abs().mean()  # 对batch平均
                    batch_importance.append(importance_i.item())

                importance_accum.append(np.array(batch_importance))

            sample_count += B
            if sample_count % 50 == 0:
                print(f"  Processed {sample_count} samples...")

        # 平均重要性
        mean_importance = np.stack(importance_accum).mean(axis=0)
        print("Feature importance computation complete!")

        # 分类整理
        static_dim = len(self.feature_names['static'])
        dynamic_dim = len(self.feature_names['dynamic'])

        results = {
            'static': mean_importance[:static_dim],
            'dynamic': mean_importance[static_dim:static_dim+dynamic_dim],
            'overall': mean_importance
        }

        return results

    def explain_semantic_contribution(self, X_sample: torch.Tensor,
                                      semantic_embed: torch.Tensor) -> Dict:
        """
        分析LLM语义特征对预测的贡献
        环境犯罪学理论的可解释性分析

        Args:
            X_sample: 输入样本
            semantic_embed: 语义嵌入

        Returns:
            语义贡献分析结果
        """
        self.model.eval()

        with torch.no_grad():
            # 有语义的预测
            pi_with, mu_with, theta, _, _ = self.model(
                X_sample, self.A_spatial, self.A_distance,
                torch.eye(X_sample.size(2), device=self.device).unsqueeze(0).expand(X_sample.size(0), -1, -1),
                self.A_hypergraph, None,
                semantic_embed=semantic_embed
            )
            pred_with = ((1 - pi_with) * mu_with).cpu().numpy()

            # 无语义的预测 (使用零向量)
            semantic_zeros = torch.zeros_like(semantic_embed)
            pi_without, mu_without, _, _, _ = self.model(
                X_sample, self.A_spatial, self.A_distance,
                torch.eye(X_sample.size(2), device=self.device).unsqueeze(0).expand(X_sample.size(0), -1, -1),
                self.A_hypergraph, None,
                semantic_embed=semantic_zeros
            )
            pred_without = ((1 - pi_without) * mu_without).cpu().numpy()

        # 计算语义贡献
        semantic_contribution = pred_with - pred_without

        return {
            'prediction_with_semantic': pred_with,
            'prediction_without_semantic': pred_without,
            'semantic_contribution': semantic_contribution,
            'relative_contribution': semantic_contribution / (pred_without + 1e-6)
        }

    def explain_near_repeat_params(self) -> Optional[Dict]:
        """
        获取近重复效应的学习参数 (可解释性)
        """
        if not hasattr(self.model, 'near_repeat_module') or self.model.near_repeat_module is None:
            return None

        params = self.model.near_repeat_module.get_decay_params()

        # 计算理论近重复效应范围
        spatial_decay = params['spatial_decay']
        temporal_decay = params['temporal_decay']

        # 空间影响半径（衰减到10%的距离）
        spatial_radius = -np.log(0.1) / spatial_decay if spatial_decay > 0 else float('inf')

        # 时间影响窗口（衰减到10%的时间）
        temporal_window = -np.log(0.1) / temporal_decay if temporal_decay > 0 else float('inf')

        return {
            **params,
            'spatial_influence_radius': spatial_radius,  # 米或网格单位
            'temporal_influence_window': temporal_window,  # 天
            'interpretation': f"犯罪事件发生后，在 {spatial_radius:.1f} 距离单位内、"
                            f"{temporal_window:.1f} 天内，发生近重复犯罪的概率显著升高"
        }

    def _flatten_feature_names(self) -> List[str]:
        """将特征名称展平为列表"""
        names = []
        names.extend(self.feature_names['static'])
        names.extend(self.feature_names['dynamic'])
        return names


# ================================
# 2. 可视化工具
# ================================
class ExplanationVisualizer:
    """可解释性结果可视化"""

    @staticmethod
    def plot_temporal_attention(attention_weights: np.ndarray,
                                 time_labels: Optional[List[str]] = None,
                                 save_path: Optional[str] = None):
        """
        可视化时间注意力权重

        Args:
            attention_weights: (T, T) 注意力矩阵
            time_labels: 时间标签
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(attention_weights, cmap='viridis', ax=ax,
                   xticklabels=time_labels, yticklabels=time_labels,
                   cbar_kws={'label': 'Attention Weight'})

        ax.set_xlabel('Time Step (Past → Future)')
        ax.set_ylabel('Query Time Step')
        ax.set_title('Temporal Self-Attention Pattern\n(Causal Mask Applied)')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_feature_importance(importance_dict: Dict[str, np.ndarray],
                                 top_k: int = 15,
                                 save_path: Optional[str] = None):
        """
        可视化特征重要性

        Args:
            importance_dict: 来自explain_feature_importance_global的结果
            top_k: 显示前k个重要特征
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 静态特征
        static_imp = importance_dict['static']
        static_idx = np.argsort(static_imp)[-top_k:]

        axes[0].barh(range(top_k), static_imp[static_idx])
        axes[0].set_yticks(range(top_k))
        axes[0].set_yticklabels([f'Static_{i}' for i in static_idx])
        axes[0].set_xlabel('Importance Score')
        axes[0].set_title('Top Static Features')

        # 动态特征
        dynamic_imp = importance_dict['dynamic']
        dynamic_idx = np.argsort(dynamic_imp)[-top_k:]

        axes[1].barh(range(top_k), dynamic_imp[dynamic_idx])
        axes[1].set_yticks(range(top_k))
        axes[1].set_yticklabels([f'Dynamic_{i}' for i in dynamic_idx])
        axes[1].set_xlabel('Importance Score')
        axes[1].set_title('Top Dynamic Features')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_semantic_gate_distribution(gate_values: np.ndarray,
                                         grid_coords: Optional[np.ndarray] = None,
                                         save_path: Optional[str] = None):
        """
        可视化语义门控权重的空间分布

        Args:
            gate_values: (N,) 每个网格的语义门控权重
            grid_coords: (N, 2) 网格坐标用于空间映射
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 直方图
        axes[0].hist(gate_values, bins=30, edgecolor='black')
        axes[0].axvline(gate_values.mean(), color='red', linestyle='--',
                       label=f'Mean: {gate_values.mean():.3f}')
        axes[0].set_xlabel('Semantic Gate Weight')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Distribution of Semantic Gate Weights')
        axes[0].legend()

        # 空间分布 (如果有坐标)
        if grid_coords is not None:
            scatter = axes[1].scatter(grid_coords[:, 0], grid_coords[:, 1],
                                     c=gate_values, cmap='RdYlBu_r',
                                     s=20, alpha=0.6)
            axes[1].set_xlabel('X Coordinate')
            axes[1].set_ylabel('Y Coordinate')
            axes[1].set_title('Spatial Distribution of Semantic Gate Weights')
            plt.colorbar(scatter, ax=axes[1], label='Gate Weight')
        else:
            axes[1].text(0.5, 0.5, 'Grid coordinates not provided',
                        ha='center', va='center')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_near_repeat_decay(spatial_decay: float, temporal_decay: float,
                                max_dist: float = 1000, max_time: float = 30,
                                save_path: Optional[str] = None):
        """
        可视化近重复效应的时空衰减曲线

        Args:
            spatial_decay: 空间衰减参数
            temporal_decay: 时间衰减参数
            max_dist: 最大距离（米）
            max_time: 最大时间（天）
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        distances = np.linspace(0, max_dist, 100)
        spatial_kernel = np.exp(-spatial_decay * distances)

        axes[0].plot(distances, spatial_kernel, 'b-', linewidth=2)
        axes[0].fill_between(distances, spatial_kernel, alpha=0.3)
        axes[0].axhline(y=0.1, color='r', linestyle='--', label='10% threshold')
        axes[0].set_xlabel('Distance (meters)')
        axes[0].set_ylabel('Effect Intensity')
        axes[0].set_title('Spatial Decay of Near-Repeat Effect')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        times = np.linspace(0, max_time, 100)
        temporal_kernel = np.exp(-temporal_decay * times)

        axes[1].plot(times, temporal_kernel, 'g-', linewidth=2)
        axes[1].fill_between(times, temporal_kernel, alpha=0.3, color='green')
        axes[1].axhline(y=0.1, color='r', linestyle='--', label='10% threshold')
        axes[1].set_xlabel('Time (days)')
        axes[1].set_ylabel('Effect Intensity')
        axes[1].set_title('Temporal Decay of Near-Repeat Effect')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ================================
# 3. 环境犯罪学政策建议生成
# ================================
class PolicyAdvisor:
    """
    基于模型解释结果生成环境改善政策建议
    将可解释性转化为可操作的警务/城市规划建议
    """

    # 环境犯罪学干预措施库
    INTERVENTION_CATALOG = {
        'guardianship': {
            'low_surveillance': {
                'problem': '监护能力不足（摄像头覆盖率低）',
                'interventions': [
                    '增加CCTV摄像头部署，特别是公共区域',
                    '推行社区守望计划（Neighborhood Watch）',
                    '增加巡警频率，特别是夜间时段',
                    '安装智能路灯系统，提高照明覆盖'
                ]
            },
            'low_activity': {
                'problem': '自然监护不足（人流量少）',
                'interventions': [
                    '举办社区活动增加街道人流量',
                    '吸引商户入驻，增加商业活动',
                    '改善公共交通接驳，增加通勤人流',
                    '设置共享单车停放点，增加街道活跃度'
                ]
            }
        },
        'target_hardening': {
            'high_commercial': {
                'problem': '商业区域犯罪目标丰富',
                'interventions': [
                    '推广商铺防盗设施（卷帘门、报警器）',
                    '建立商户联防机制',
                    '设置贵重商品展示限制',
                    '增加夜间安保巡逻'
                ]
            },
            'high_residential_vulnerability': {
                'problem': '住宅区易受害性高',
                'interventions': [
                    '推广家庭安防系统',
                    '改善建筑门禁管理',
                    '修剪遮挡视线的植被',
                    '增加街道照明'
                ]
            }
        },
        'environmental_design': {
            'poor_lighting': {
                'problem': '照明不足',
                'interventions': [
                    '升级LED路灯，提高亮度',
                    '消除照明死角',
                    '鼓励商户夜间亮灯',
                    '安装太阳能应急照明'
                ]
            },
            'poor_visibility': {
                'problem': '视线受阻（植被/建筑）',
                'interventions': [
                    '修剪过高的树木和灌木',
                    '重新设计围栏高度（不超过1米）',
                    '消除建筑死角',
                    '增加镜面反射设施改善视野'
                ]
            },
            'escape_routes': {
                'problem': '便于逃离的路径设计',
                'interventions': [
                    '封闭不必要的后巷和捷径',
                    '在关键路径设置障碍物',
                    '增加通道的自然监护',
                    '设置电子门禁控制'
                ]
            }
        },
        'crime_generator': {
            'bar_club_area': {
                'problem': '娱乐场所聚集（犯罪生成器）',
                'interventions': [
                    '规范酒吧营业时间',
                    '增加夜间出租车/网约车接驳点',
                    '设置临时醒酒室',
                    '增加散场时段的警力部署'
                ]
            },
            'transit_hub': {
                'problem': '交通枢纽人流混杂',
                'interventions': [
                    '优化站内摄像头布局',
                    '增加便衣巡逻',
                    '改善站内照明',
                    '设置清晰的方向指示减少徘徊'
                ]
            }
        }
    }

    def __init__(self, explainer: CrimePredictionExplainer):
        self.explainer = explainer

    def generate_hotspot_recommendations(self, grid_id: int,
                                          shap_values: np.ndarray,
                                          feature_values: np.ndarray,
                                          semantic_desc: Optional[str] = None) -> Dict:
        """
        针对特定高风险网格生成政策建议

        Args:
            grid_id: 网格ID
            shap_values: 该网格的SHAP值
            feature_values: 该网格的特征值
            semantic_desc: LLM生成的语义描述（如果有）

        Returns:
            政策建议字典
        """
        recommendations = {
            'grid_id': grid_id,
            'risk_level': self._assess_risk_level(shap_values),
            'primary_factors': self._identify_key_factors(shap_values, feature_values),
            'interventions': [],
            'priority': None
        }

        # 根据关键因素匹配干预措施
        for factor in recommendations['primary_factors']:
            interventions = self._match_interventions(factor)
            recommendations['interventions'].extend(interventions)

        # 去重并排序
        recommendations['interventions'] = list(set(recommendations['interventions']))
        recommendations['priority'] = self._calculate_priority(recommendations)

        return recommendations

    def _assess_risk_level(self, shap_values: np.ndarray) -> str:
        """评估风险等级"""
        total_contribution = np.sum(np.abs(shap_values))
        if total_contribution > 2.0:
            return 'Very High'
        elif total_contribution > 1.0:
            return 'High'
        elif total_contribution > 0.5:
            return 'Medium'
        else:
            return 'Low'

    def _identify_key_factors(self, shap_values: np.ndarray,
                               feature_values: np.ndarray,
                               top_k: int = 5) -> List[Dict]:
        """识别关键风险因素"""
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[-top_k:]

        factors = []
        for idx in top_indices:
            factors.append({
                'feature_index': idx,
                'shap_value': shap_values[idx],
                'feature_value': feature_values[idx],
                'direction': 'increases_risk' if shap_values[idx] > 0 else 'decreases_risk'
            })

        return factors

    def _match_interventions(self, factor: Dict) -> List[str]:
        """根据风险因素匹配干预措施"""
        interventions = []

        # 简化版匹配逻辑，实际可根据特征索引细化
        idx = factor['feature_index']

        # 示例映射（需要根据实际特征定义调整）
        if idx in [0, 1, 2]:  # POI相关
            if factor['feature_value'] > 0.7:
                interventions.extend(self.INTERVENTION_CATALOG['crime_generator']['bar_club_area']['interventions'])

        elif idx in [8]:  # 照明
            if factor['feature_value'] < 0.3:
                interventions.extend(self.INTERVENTION_CATALOG['environmental_design']['poor_lighting']['interventions'])

        elif idx in [9]:  # 摄像头
            if factor['feature_value'] < 0.3:
                interventions.extend(self.INTERVENTION_CATALOG['guardianship']['low_surveillance']['interventions'])

        return interventions

    def _calculate_priority(self, recommendations: Dict) -> int:
        """计算干预优先级分数"""
        risk_score = {'Very High': 4, 'High': 3, 'Medium': 2, 'Low': 1}
        base_score = risk_score.get(recommendations['risk_level'], 1)
        intervention_count = len(recommendations['interventions'])

        return base_score * 10 + min(intervention_count, 10)

    def generate_citywide_report(self, all_grid_results: List[Dict]) -> Dict:
        """
        生成全市层面的政策建议报告

        Args:
            all_grid_results: 所有网格的分析结果

        Returns:
            汇总报告
        """
        risk_distribution = {'Very High': 0, 'High': 0, 'Medium': 0, 'Low': 0}
        all_interventions = []

        for result in all_grid_results:
            risk_distribution[result['risk_level']] += 1
            all_interventions.extend(result['interventions'])

        # 统计最常见的干预措施
        from collections import Counter
        intervention_counter = Counter(all_interventions)
        top_interventions = intervention_counter.most_common(10)

        return {
            'risk_distribution': risk_distribution,
            'total_hotspots': sum([risk_distribution[k] for k in ['Very High', 'High']]),
            'recommended_interventions': [
                {'action': action, 'target_grids': count}
                for action, count in top_interventions
            ],
            'budget_priority': self._prioritize_budget(top_interventions, risk_distribution)
        }

    def _prioritize_budget(self, top_interventions, risk_distribution):
        """基于风险分布给出预算分配建议"""
        total_high_risk = risk_distribution['Very High'] + risk_distribution['High']

        return {
            'immediate_action': f"优先处理 {risk_distribution['Very High']} 个极高风险区域",
            'estimated_budget_tiers': {
                'low_cost': '增加巡逻频率、社区宣传 (约$10k-50k)',
                'medium_cost': '照明改善、摄像头安装 (约$100k-500k)',
                'high_cost': '基础设施改造、商业引入 (约$1M+)'
            },
            'roi_focus': '建议优先投资中等成本的基础设施，可实现最大犯罪率降幅'
        }


# ================================
# 4. 使用示例
# ================================
def demo_explanation():
    """
    演示如何使用解释性分析模块
    """
    print("""
    使用示例:

    # 1. 初始化解释器
    explainer = CrimePredictionExplainer(
        model=trained_model,
        background_data=X_background,
        feature_names={'static': [...], 'dynamic': [...]}
    )
    explainer.set_graph_data(A_spatial, A_distance, A_hypergraph,
                             semantic_embed, OD)

    # 2. SHAP解释
    shap_results = explainer.explain_with_shap(X_sample, grid_ids=[100, 200, 300])

    # 3. 全局特征重要性
    importance = explainer.explain_feature_importance_global(test_loader)

    # 4. 语义贡献分析
    semantic_impact = explainer.explain_semantic_contribution(X_sample, semantic_embed)

    # 5. 近重复效应参数
    nr_params = explainer.explain_near_repeat_params()

    # 6. 可视化
    ExplanationVisualizer.plot_feature_importance(importance)
    ExplanationVisualizer.plot_near_repeat_decay(
        nr_params['spatial_decay'],
        nr_params['temporal_decay']
    )

    # 7. 生成政策建议
    advisor = PolicyAdvisor(explainer)
    for grid_id in high_risk_grids:
        rec = advisor.generate_hotspot_recommendations(
            grid_id, shap_values[grid_id], features[grid_id]
        )
        print(f"网格 {grid_id}: {rec['interventions']}")
    """)


if __name__ == "__main__":
    demo_explanation()
