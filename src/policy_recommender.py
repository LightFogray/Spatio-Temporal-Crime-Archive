"""
政策建议生成模块 - 基于特征值直接生成警务/环境干预建议
无需SHAP解释性分析，直接基于CPTED评分和环境特征生成可操作建议
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GridProfile:
    """网格风险画像"""
    grid_id: int
    risk_score: float
    risk_level: str
    cpted_scores: Dict[str, float]  # 自然监护、入口控制、活动支持、目标强化
    environmental_features: Dict[str, float]  # 绿化率、照明、摄像头等
    poi_features: Dict[str, float]  # POI分布
    nearby_crimes_7d: int  # 近7天周边犯罪数


class PolicyRecommender:
    """
    基于网格特征直接生成政策建议
    无需SHAP分析，适用于实时预警系统
    """

    # 干预措施知识库 - 结构化建议
    INTERVENTIONS = {
        'surveillance': {
            'low_natural': {
                'indicator': '自然监护不足 (人流量低 + CPTED监护评分低)',
                'measures': [
                    {'action': '增加定点巡逻', 'priority': '高', 'cost': '低',
                     'detail': '每日高峰时段(18:00-02:00)安排警车定点停靠2小时'},
                    {'action': '部署移动警务站', 'priority': '中', 'cost': '中',
                     'detail': '在风险网格中心设置临时警务站，配备2名警员'},
                    {'action': '推广社区守望计划', 'priority': '中', 'cost': '低',
                     'detail': '组织商户和居民建立微信群，实行"可疑情况即时通报"机制'},
                ]
            },
            'low_technical': {
                'indicator': '技术监护不足 (摄像头覆盖率低)',
                'measures': [
                    {'action': '增补AI智能摄像头', 'priority': '高', 'cost': '中',
                     'detail': '在网格四角安装具备行为识别功能的摄像头，覆盖盲区'},
                    {'action': '接入商户监控系统', 'priority': '中', 'cost': '低',
                     'detail': '协调沿街商户将监控接入公安联网平台'},
                    {'action': '部署无人机巡逻', 'priority': '低', 'cost': '高',
                     'detail': '夜间时段使用无人机进行空中巡逻'},
                ]
            }
        },
        'lighting': {
            'poor_lighting': {
                'indicator': '照明不足 (夜间灯光指数低)',
                'measures': [
                    {'action': '升级LED路灯', 'priority': '高', 'cost': '中',
                     'detail': '将现有路灯更换为200W LED灯，提升照度至50lux以上'},
                    {'action': '消除照明死角', 'priority': '高', 'cost': '低',
                     'detail': '修剪遮挡灯光的树木枝叶，清洁灯罩'},
                    {'action': '商户亮灯工程', 'priority': '中', 'cost': '低',
                     'detail': '鼓励商户夜间保留橱窗照明，给予电费补贴'},
                    {'action': '安装太阳能感应灯', 'priority': '中', 'cost': '中',
                     'detail': '在背街小巷安装人体感应太阳能灯'},
                ]
            }
        },
        'greening': {
            'poor_visibility': {
                'indicator': '视线遮挡 (绿化覆盖过高 + CPTED入口控制评分低)',
                'measures': [
                    {'action': '修剪植被高度', 'priority': '高', 'cost': '低',
                     'detail': '将灌木修剪至0.6米以下，树枝修剪至2.5米以上'},
                    {'action': '开辟通透视线廊道', 'priority': '中', 'cost': '中',
                     'detail': '重新设计绿化布局，确保行人视线可达15米'},
                    {'action': '减少密集植被', 'priority': '中', 'cost': '低',
                     'detail': '移除可能藏匿人员的密集灌木丛'},
                ]
            },
            'green_deficit': {
                'indicator': '绿化率过低 (缺乏积极活动空间)',
                'measures': [
                    {'action': '增设口袋公园', 'priority': '中', 'cost': '高',
                     'detail': '利用闲置空地建设小型公园，增加正当活动人群'},
                    {'action': '沿街绿化美化', 'priority': '低', 'cost': '中',
                     'detail': '种植行道树，设置花坛，提升环境品质吸引人流'},
                    {'action': '设置休憩设施', 'priority': '中', 'cost': '低',
                     'detail': '安装长椅、遮阳棚，鼓励居民停留增加监护'},
                ]
            }
        },
        'access_control': {
            'poor_access_control': {
                'indicator': '入口控制薄弱 (CPTED入口控制评分低 + 巷道密集)',
                'measures': [
                    {'action': '封闭无用巷道', 'priority': '高', 'cost': '中',
                     'detail': '用铁栅栏封闭犯罪高发的小巷和捷径'},
                    {'action': '设置电子门禁', 'priority': '中', 'cost': '中',
                     'detail': '在居民区入口安装刷卡门禁系统'},
                    {'action': '优化道路设计', 'priority': '低', 'cost': '高',
                     'detail': ' redesign路口布局，减少潜在逃逸路径'},
                ]
            }
        },
        'activity_support': {
            'low_activity': {
                'indicator': '活动支持不足 (商业POI少 + CPTED活动支持评分低)',
                'measures': [
                    {'action': '引入便民商业', 'priority': '中', 'cost': '中',
                     'detail': '提供税收优惠吸引便利店、咖啡店入驻'},
                    {'action': '举办社区活动', 'priority': '中', 'cost': '低',
                     'detail': '周末举办集市、文艺演出，增加正当人流'},
                    {'action': '延长商业营业时间', 'priority': '中', 'cost': '低',
                     'detail': '协调商户延长营业时间至22:00'},
                ]
            }
        },
        'target_hardening': {
            'high_commercial_risk': {
                'indicator': '商业目标丰富 (商业POI密度高 + 犯罪率上升)',
                'measures': [
                    {'action': '推广技防设施', 'priority': '高', 'cost': '中',
                     'detail': '为商户补贴安装卷帘门、入侵报警器'},
                    {'action': '建立商户联防', 'priority': '高', 'cost': '低',
                     'detail': '组织50米范围内商户建立联防机制，一店有事十店响应'},
                    {'action': '限制贵重品展示', 'priority': '中', 'cost': '低',
                     'detail': '建议珠宝店、电子产品店夜间收起贵重商品展示'},
                ]
            },
            'high_residential_risk': {
                'indicator': '住宅易受害性高 (住宅区密集 + 近重复效应显著)',
                'measures': [
                    {'action': '推广家庭安防', 'priority': '中', 'cost': '中',
                     'detail': '为居民补贴智能门锁、窗磁报警器'},
                    {'action': '改善门禁管理', 'priority': '高', 'cost': '低',
                     'detail': '修复损坏单元门，加装闭门器'},
                    {'action': '清理占用公共空间', 'priority': '中', 'cost': '低',
                     'detail': '清理楼道杂物，消除潜在藏匿点'},
                ]
            }
        },
        'near_repeat_response': {
            'active_near_repeat': {
                'indicator': '近重复效应活跃 (周边7天内发生多起同类犯罪)',
                'measures': [
                    {'action': '启动近重复预警响应', 'priority': '极高', 'cost': '低',
                     'detail': '在已有犯罪点周边500米、7天内实施超常规巡逻'},
                    {'action': '派发预防传单', 'priority': '高', 'cost': '低',
                     'detail': '向周边居民发放防盗防抢提示单'},
                    {'action': '组织社区会议', 'priority': '中', 'cost': '低',
                     'detail': '召集居民通报近期案情，教授防范技巧'},
                    {'action': '临时技防强化', 'priority': '高', 'cost': '中',
                     'detail': '在案发周边临时加装移动摄像头和感应灯'},
                ]
            }
        }
    }

    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Args:
            feature_names: 特征名称列表，用于定位CPTED和环境特征索引
        """
        self.feature_names = feature_names or self._default_feature_names()
        self.feature_idx = {name: i for i, name in enumerate(self.feature_names)}

    def _default_feature_names(self) -> List[str]:
        """默认特征命名 (与train_stgcn_trans.py保持一致)"""
        return [
            # Static features (0-15)
            'poi_commercial', 'poi_transport', 'poi_public',
            'road_density', 'landuse_res', 'landuse_com', 'landuse_ind',
            'green_ratio', 'nightlight', 'camera_count',
            'cpted_natural_surveillance', 'cpted_access_control',
            'cpted_territorial_reinforcement', 'cpted_target_hardening',
            'building_density', 'population_density',
            # Dynamic features (16+)
            'crime_lag_1d', 'crime_lag_2d', 'crime_lag_3d',
            'crime_lag_4d', 'crime_lag_5d', 'crime_lag_6d', 'crime_lag_7d',
            'temp_avg', 'rain_freq', 'snow_flag'
        ]

    def analyze_grid(self, grid_id: int, risk_score: float,
                     static_features: np.ndarray,
                     recent_crimes: int = 0) -> GridProfile:
        """
        分析单个网格的特征画像

        Args:
            grid_id: 网格ID
            risk_score: 预测风险分数
            static_features: 静态特征向量
            recent_crimes: 近7天该网格犯罪数

        Returns:
            GridProfile对象
        """
        # 确定风险等级
        if risk_score >= 0.8:
            risk_level = '极高'
        elif risk_score >= 0.6:
            risk_level = '高'
        elif risk_score >= 0.4:
            risk_level = '中等'
        else:
            risk_level = '低'

        # 提取CPTED评分 (假设索引10-13)
        cpted_scores = {
            'natural_surveillance': static_features[self.feature_idx.get('cpted_natural_surveillance', 10)],
            'access_control': static_features[self.feature_idx.get('cpted_access_control', 11)],
            'territorial_reinforcement': static_features[self.feature_idx.get('cpted_territorial_reinforcement', 12)],
            'target_hardening': static_features[self.feature_idx.get('cpted_target_hardening', 13)]
        }

        # 提取环境特征
        env_features = {
            'green_ratio': static_features[self.feature_idx.get('green_ratio', 7)],
            'nightlight': static_features[self.feature_idx.get('nightlight', 8)],
            'camera_count': static_features[self.feature_idx.get('camera_count', 9)],
            'road_density': static_features[self.feature_idx.get('road_density', 3)],
            'building_density': static_features[self.feature_idx.get('building_density', 14)]
        }

        # 提取POI特征
        poi_features = {
            'commercial': static_features[self.feature_idx.get('poi_commercial', 0)],
            'transport': static_features[self.feature_idx.get('poi_transport', 1)],
            'public': static_features[self.feature_idx.get('poi_public', 2)],
            'residential': static_features[self.feature_idx.get('landuse_res', 4)]
        }

        return GridProfile(
            grid_id=grid_id,
            risk_score=risk_score,
            risk_level=risk_level,
            cpted_scores=cpted_scores,
            environmental_features=env_features,
            poi_features=poi_features,
            nearby_crimes_7d=recent_crimes
        )

    def generate_recommendations(self, profile: GridProfile) -> Dict:
        """
        基于网格画像生成具体政策建议

        Returns:
            包含诊断结果和建议措施的字典
        """
        recommendations = {
            'grid_id': profile.grid_id,
            'risk_assessment': {
                'level': profile.risk_level,
                'score': round(profile.risk_score, 4),
                'primary_diagnosis': []
            },
            'immediate_actions': [],  # 立即执行（24小时内）
            'short_term_actions': [],  # 短期（1周内）
            'medium_term_actions': [],  # 中期（1月内）
            'budget_estimate': {}
        }

        problems = []

        # 1. 检查近重复效应 (最优先)
        if profile.nearby_crimes_7d >= 2:
            problems.append('near_repeat')
            recs = self.INTERVENTIONS['near_repeat_response']['active_near_repeat']
            recommendations['risk_assessment']['primary_diagnosis'].append(recs['indicator'])
            for m in recs['measures']:
                if m['priority'] == '极高':
                    recommendations['immediate_actions'].append(m)
                elif m['priority'] == '高':
                    recommendations['short_term_actions'].append(m)

        # 2. 检查照明状况
        if profile.environmental_features.get('nightlight', 0.5) < 0.3:
            problems.append('lighting')
            recs = self.INTERVENTIONS['lighting']['poor_lighting']
            recommendations['risk_assessment']['primary_diagnosis'].append(recs['indicator'])
            for m in recs['measures']:
                if m['priority'] == '高':
                    recommendations['short_term_actions'].append(m)
                else:
                    recommendations['medium_term_actions'].append(m)

        # 3. 检查自然监护 (CPTED评分 + POI活动)
        cpted_surveillance = profile.cpted_scores.get('natural_surveillance', 0.5)
        commercial_activity = profile.poi_features.get('commercial', 0)

        if cpted_surveillance < 0.4 and commercial_activity < 0.3:
            problems.append('surveillance_low')
            recs = self.INTERVENTIONS['surveillance']['low_natural']
            recommendations['risk_assessment']['primary_diagnosis'].append(recs['indicator'])
            for m in recs['measures']:
                if m['priority'] == '高':
                    recommendations['immediate_actions'].append(m)
                else:
                    recommendations['short_term_actions'].append(m)

        # 4. 检查绿化/视线遮挡
        green_ratio = profile.environmental_features.get('green_ratio', 0.5)
        cpted_access = profile.cpted_scores.get('access_control', 0.5)

        if green_ratio > 0.6 and cpted_access < 0.4:
            problems.append('greening_visibility')
            recs = self.INTERVENTIONS['greening']['poor_visibility']
            recommendations['risk_assessment']['primary_diagnosis'].append(recs['indicator'])
            recommendations['short_term_actions'].extend(recs['measures'][:2])
        elif green_ratio < 0.1 and profile.risk_score > 0.5:
            # 绿化过少，缺乏积极空间
            problems.append('green_deficit')
            recs = self.INTERVENTIONS['greening']['green_deficit']
            recommendations['risk_assessment']['primary_diagnosis'].append(recs['indicator'])
            recommendations['medium_term_actions'].extend(recs['measures'])

        # 5. 检查技术监护
        camera_count = profile.environmental_features.get('camera_count', 0)
        if camera_count < 0.2 and profile.risk_score > 0.6:
            problems.append('camera_low')
            recs = self.INTERVENTIONS['surveillance']['low_technical']
            recommendations['risk_assessment']['primary_diagnosis'].append(recs['indicator'])
            recommendations['short_term_actions'].extend(recs['measures'][:2])

        # 6. 检查活动支持
        cpted_activity = profile.cpted_scores.get('territorial_reinforcement', 0.5)
        if cpted_activity < 0.3 and commercial_activity < 0.3:
            problems.append('activity_low')
            recs = self.INTERVENTIONS['activity_support']['low_activity']
            recommendations['risk_assessment']['primary_diagnosis'].append(recs['indicator'])
            recommendations['short_term_actions'].append(recs['measures'][1])  # 社区活动
            recommendations['medium_term_actions'].extend([recs['measures'][0], recs['measures'][2]])

        # 7. 检查入口控制
        if cpted_access < 0.3:
            problems.append('access_control')
            recs = self.INTERVENTIONS['access_control']['poor_access_control']
            recommendations['risk_assessment']['primary_diagnosis'].append(recs['indicator'])
            recommendations['short_term_actions'].append(recs['measures'][0])
            recommendations['medium_term_actions'].extend(recs['measures'][1:])

        # 8. 目标强化 (商业/住宅)
        if commercial_activity > 0.7:
            problems.append('commercial_risk')
            recs = self.INTERVENTIONS['target_hardening']['high_commercial_risk']
            recommendations['risk_assessment']['primary_diagnosis'].append(recs['indicator'])
            recommendations['immediate_actions'].append(recs['measures'][0])
            recommendations['short_term_actions'].extend(recs['measures'][1:])

        if profile.poi_features.get('residential', 0) > 0.6 and profile.nearby_crimes_7d > 0:
            problems.append('residential_risk')
            recs = self.INTERVENTIONS['target_hardening']['high_residential_risk']
            recommendations['risk_assessment']['primary_diagnosis'].append(recs['indicator'])
            recommendations['short_term_actions'].extend(recs['measures'])

        # 去重
        recommendations['immediate_actions'] = self._deduplicate(recommendations['immediate_actions'])
        recommendations['short_term_actions'] = self._deduplicate(recommendations['short_term_actions'])
        recommendations['medium_term_actions'] = self._deduplicate(recommendations['medium_term_actions'])

        # 预算估算
        recommendations['budget_estimate'] = self._estimate_budget(recommendations)

        return recommendations

    def _deduplicate(self, actions: List[Dict]) -> List[Dict]:
        """去重措施列表"""
        seen = set()
        result = []
        for a in actions:
            key = a['action']
            if key not in seen:
                seen.add(key)
                result.append(a)
        return result

    def _estimate_budget(self, recommendations: Dict) -> Dict:
        """估算预算"""
        cost_map = {'低': 5000, '中': 50000, '高': 200000}

        immediate_cost = sum(cost_map.get(a['cost'], 10000)
                            for a in recommendations['immediate_actions'])
        short_cost = sum(cost_map.get(a['cost'], 10000)
                        for a in recommendations['short_term_actions'])
        medium_cost = sum(cost_map.get(a['cost'], 10000)
                         for a in recommendations['medium_term_actions'])

        return {
            'immediate_usd': immediate_cost,
            'short_term_usd': short_cost,
            'medium_term_usd': medium_cost,
            'total_usd': immediate_cost + short_cost + medium_cost,
            'manpower_priority': '高' if len(recommendations['immediate_actions']) > 2 else '中'
        }

    def generate_patrol_plan(self, high_risk_grids: List[GridProfile],
                             available_officers: int = 10) -> Dict:
        """
        生成巡逻部署方案

        Args:
            high_risk_grids: 高风险网格列表
            available_officers: 可用警力数量

        Returns:
            巡逻计划字典
        """
        # 按风险分数排序
        sorted_grids = sorted(high_risk_grids, key=lambda x: x.risk_score, reverse=True)

        # 分配警力 (风险越高，分配越多)
        total_risk = sum(g.risk_score for g in sorted_grids)
        patrol_plan = []

        for grid in sorted_grids:
            officer_ratio = grid.risk_score / total_risk if total_risk > 0 else 0
            officers = max(1, round(available_officers * officer_ratio))

            plan = {
                'grid_id': grid.grid_id,
                'officers_assigned': officers,
                'patrol_hours': ['20:00-02:00'] if grid.risk_score > 0.7 else ['14:00-22:00'],
                'patrol_mode': '步行巡逻' if grid.environmental_features.get('road_density', 0) > 0.5 else '车巡+步巡',
                'focus_areas': self._identify_focus_areas(grid)
            }
            patrol_plan.append(plan)

        return {
            'total_officers_deployed': sum(p['officers_assigned'] for p in patrol_plan),
            'coverage_grids': len(patrol_plan),
            'shift_recommendation': '双班制(06:00-14:00, 14:00-22:00, 22:00-06:00)',
            'grid_assignments': patrol_plan
        }

    def _identify_focus_areas(self, profile: GridProfile) -> List[str]:
        """识别该网格需要重点关注的区域类型"""
        focus = []

        if profile.poi_features.get('commercial', 0) > 0.5:
            focus.append('商业街区出入口')
        if profile.poi_features.get('transport', 0) > 0.3:
            focus.append('公交/地铁站周边')
        if profile.environmental_features.get('green_ratio', 0) > 0.4:
            focus.append('公园/绿地小径')
        if profile.environmental_features.get('camera_count', 0) < 0.3:
            focus.append('监控盲区')

        return focus if focus else ['主干道交叉口']

    def export_to_web_format(self, recommendations: Dict) -> Dict:
        """
        转换为Web可视化可用的格式
        """
        return {
            'grid_id': recommendations['grid_id'],
            'risk_level': recommendations['risk_assessment']['level'],
            'risk_score': recommendations['risk_assessment']['score'],
            'diagnosis': '; '.join(recommendations['risk_assessment']['primary_diagnosis']),
            'top_3_actions': (
                recommendations['immediate_actions'][:1] +
                recommendations['short_term_actions'][:2]
            ),
            'manpower_required': recommendations['budget_estimate']['manpower_priority'],
            'estimated_cost_usd': recommendations['budget_estimate']['total_usd']
        }


# ================================
# 批量处理函数
# ================================

def generate_recommendations_for_predictions(
    predictions: np.ndarray,
    static_features: np.ndarray,
    grid_metadata: List[Dict],
    recent_crime_counts: Optional[np.ndarray] = None,
    top_k: int = 50
) -> List[Dict]:
    """
    为预测结果批量生成建议

    Args:
        predictions: (N,) 各网格预测风险值
        static_features: (N, F) 静态特征矩阵
        grid_metadata: 网格元数据列表
        recent_crime_counts: (N,) 近7天犯罪数
        top_k: 只为前K高风险网格生成详细建议

    Returns:
        建议列表
    """
    recommender = PolicyRecommender()

    # 获取高风险网格索引
    high_risk_indices = np.argsort(predictions)[-top_k:][::-1]

    results = []
    for idx in high_risk_indices:
        profile = recommender.analyze_grid(
            grid_id=int(idx),
            risk_score=float(predictions[idx]),
            static_features=static_features[idx],
            recent_crimes=int(recent_crime_counts[idx]) if recent_crime_counts is not None else 0
        )

        recs = recommender.generate_recommendations(profile)
        web_format = recommender.export_to_web_format(recs)
        results.append(web_format)

    return results


if __name__ == "__main__":
    # 演示用法
    print("PolicyRecommender 演示")
    print("="*60)

    recommender = PolicyRecommender()

    # 模拟一个高风险网格的特征
    test_features = np.array([
        0.8,   # poi_commercial - 商业密集
        0.2,   # poi_transport
        0.3,   # poi_public
        0.6,   # road_density
        0.2,   # landuse_res
        0.7,   # landuse_com
        0.1,   # landuse_ind
        0.7,   # green_ratio - 绿化高但可能遮挡视线
        0.2,   # nightlight - 照明不足
        0.1,   # camera_count - 摄像头少
        0.2,   # cpted_natural_surveillance - 监护不足
        0.3,   # cpted_access_control - 入口控制差
        0.2,   # cpted_territorial_reinforcement
        0.1,   # cpted_target_hardening
        0.6,   # building_density
        0.5,   # population_density
    ] + [0.0] * 10)  # 动态特征占位

    profile = recommender.analyze_grid(
        grid_id=123,
        risk_score=0.85,
        static_features=test_features,
        recent_crimes=3  # 近7天有3起犯罪
    )

    recommendations = recommender.generate_recommendations(profile)

    print(f"\n网格 {recommendations['grid_id']} 风险评估:")
    print(f"  风险等级: {recommendations['risk_assessment']['level']}")
    print(f"  风险分数: {recommendations['risk_assessment']['score']}")
    print(f"  主要诊断: {recommendations['risk_assessment']['primary_diagnosis']}")

    print(f"\n立即执行措施 ({len(recommendations['immediate_actions'])}项):")
    for i, action in enumerate(recommendations['immediate_actions'], 1):
        print(f"  {i}. [{action['priority']}] {action['action']}")
        print(f"     详情: {action['detail']}")

    print(f"\n短期措施 ({len(recommendations['short_term_actions'])}项):")
    for i, action in enumerate(recommendations['short_term_actions'][:5], 1):
        print(f"  {i}. [{action['priority']}] {action['action']} (成本:{action['cost']})")

    print(f"\n预算估算: ${recommendations['budget_estimate']['total_usd']:,}")
    print(f"人力优先级: {recommendations['budget_estimate']['manpower_priority']}")
