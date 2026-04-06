"""
环境犯罪学知识库 (Environmental Criminology Knowledge Base)
纯净版：只包含环境因素，排除个体/群体偏见
用于RAG增强的犯罪学先验知识
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class KnowledgeEntry:
    """知识库条目"""
    entry_id: str
    place_type: str  # 场所类型
    environmental_features: Dict[str, float]  # 环境特征（归一化）
    risk_mechanism: str  # 风险机制描述（纯环境视角）
    guardianship_strategies: List[str]  # 监护策略
    bias_audit_passed: bool = True


class EnvironmentalCriminologyKB:
    """
    环境犯罪学知识库
    基于CPTED、日常活动理论、近重复理论构建
    """

    def __init__(self, kb_path: str = "data/knowledge_base/env_criminology_kb.json"):
        self.kb_path = kb_path
        self.entries: List[KnowledgeEntry] = []
        self.bias_auditor = BiasAuditor()

        # 初始化或加载知识库
        if os.path.exists(kb_path):
            self.load_kb()
        else:
            self.build_default_kb()

    def build_default_kb(self):
        """构建默认环境犯罪学知识库"""

        # 场所类型定义（纯环境视角）
        default_entries = [
            # ========== 商业场所 ==========
            {
                "entry_id": "commercial_high_density",
                "place_type": "high_density_commercial",
                "environmental_features": {
                    "poi_commercial": 0.9,
                    "poi_transport": 0.7,
                    "road_density": 0.8,
                    "nightlight": 0.8,
                    "camera_coverage": 0.6,
                    "landuse_mix": 0.9,
                    "green_ratio": 0.2
                },
                "risk_mechanism": (
                    "高商业密度产生犯罪生成器效应："
                    "1) 大量潜在目标聚集（财物、车辆）；"
                    "2) 人流混杂降低匿名性成本；"
                    "3) 营业时间延长增加暴露窗口。"
                    "风险由环境机会结构驱动，非特定人群。"
                ),
                "guardianship_strategies": [
                    "增加CCTV覆盖至80%以上",
                    "商户联防机制（共享监控资源）",
                    "优化夜间照明至lux 50+",
                    "设计通透橱窗增加自然监护"
                ]
            },

            {
                "entry_id": "entertainment_nighttime",
                "place_type": "nighttime_entertainment",
                "environmental_features": {
                    "poi_commercial": 0.8,
                    "nightlight": 0.4,  # 照明可能不足
                    "road_density": 0.6,
                    "camera_coverage": 0.3,
                    "landuse_mix": 0.5,
                    "green_ratio": 0.1
                },
                "risk_mechanism": (
                    "夜间娱乐场所形成犯罪吸引器："
                    "1) 酒精消费增加冲突概率（环境氛围因素）；"
                    "2) 深夜时段正式监护（警力、保安）减少；"
                    "3) 闭店时段人流骤减形成监护真空。"
                    "风险由时间-环境交互产生。"
                ),
                "guardianship_strategies": [
                    "闭店时段增配安保人员",
                    "设置夜间出租车接驳点（减少街道等待）",
                    "安装智能照明随人流调节",
                    "与周边商户建立夜间联防"
                ]
            },

            # ========== 交通枢纽 ==========
            {
                "entry_id": "transit_hub",
                "place_type": "transit_hub",
                "environmental_features": {
                    "poi_transport": 0.95,
                    "poi_commercial": 0.5,
                    "road_density": 0.9,
                    "nightlight": 0.7,
                    "camera_coverage": 0.5,
                    "landuse_mix": 0.6,
                    "green_ratio": 0.1
                },
                "risk_mechanism": (
                    "交通枢纽产生流动性风险："
                    "1) 人流快速更替降低社会凝聚力（匿名性）；"
                    "2) 多模式换乘创造复杂空间（监控盲区）；"
                    "3) 地下/半地下空间自然监护受限。"
                    "风险由空间-人流特性交互产生。"
                ),
                "guardianship_strategies": [
                    "消除监控盲区（全景摄像头覆盖）",
                    "改善自然视线（减少视线遮挡结构）",
                    "增加工作人员可见性（制服、标识）",
                    "设置紧急呼叫点"
                ]
            },

            # ========== 住宅区 ==========
            {
                "entry_id": "residential_pure",
                "place_type": "pure_residential",
                "environmental_features": {
                    "poi_commercial": 0.1,
                    "poi_transport": 0.3,
                    "road_density": 0.4,
                    "nightlight": 0.5,
                    "camera_coverage": 0.2,
                    "landuse_mix": 0.2,
                    "green_ratio": 0.6
                },
                "risk_mechanism": (
                    "纯住宅区风险特征："
                    "1) 日间人口外流导致监护真空（通勤效应）；"
                    "2) 绿化过高可能遮挡视线（自然监护 vs 隐私的平衡）；"
                    "3) 道路密度低导致巡逻覆盖不足。"
                    "风险由人口时间分布与空间设计交互产生。"
                ),
                "guardianship_strategies": [
                    "推广社区守望计划（增强非正式监护）",
                    "修剪过高植被（改善视线通透性）",
                    "安装 motion-sensor 照明",
                    "建立邻里安全群组"
                ]
            },

            {
                "entry_id": "residential_mixed",
                "place_type": "mixed_use_residential",
                "environmental_features": {
                    "poi_commercial": 0.5,
                    "poi_transport": 0.5,
                    "road_density": 0.6,
                    "nightlight": 0.6,
                    "camera_coverage": 0.4,
                    "landuse_mix": 0.7,
                    "green_ratio": 0.4
                },
                "risk_mechanism": (
                    "混合用途住宅区："
                    "1) 功能混合增加全天候人流（降低真空时段）；"
                    "2) 商业-住宅边界可能存在监护责任模糊；"
                    "3) 过境人流增加目标暴露。"
                    "风险由功能混合的空间边界效应产生。"
                ),
                "guardianship_strategies": [
                    "明确商业-住宅边界管理责任",
                    "增加边界区域照明",
                    "设计通透的底层商业（自然监护）",
                    "建立商户-居民安全联盟"
                ]
            },

            # ========== 工业区 ==========
            {
                "entry_id": "industrial_zone",
                "place_type": "industrial_zone",
                "environmental_features": {
                    "poi_commercial": 0.1,
                    "poi_transport": 0.6,
                    "road_density": 0.5,
                    "nightlight": 0.3,
                    "camera_coverage": 0.3,
                    "landuse_mix": 0.1,
                    "green_ratio": 0.2
                },
                "risk_mechanism": (
                    "工业区风险特征："
                    "1) 夜间/周末完全真空（无人口活动）；"
                    "2) 高价值设备/材料（目标丰富）；"
                    "3) 围墙封闭阻碍自然监护。"
                    "风险由时间真空与目标暴露产生。"
                ),
                "guardianship_strategies": [
                    " perimeter 围栏+入侵检测",
                    "夜间照明全覆盖（lux 30+）",
                    "CCTV 24小时监控",
                    "雇佣专业安保巡逻"
                ]
            },

            # ========== 开放空间 ==========
            {
                "entry_id": "park_open_space",
                "place_type": "park_open_space",
                "environmental_features": {
                    "poi_commercial": 0.1,
                    "poi_transport": 0.2,
                    "road_density": 0.3,
                    "nightlight": 0.2,
                    "camera_coverage": 0.1,
                    "landuse_mix": 0.1,
                    "green_ratio": 0.9
                },
                "risk_mechanism": (
                    "公园/开放空间风险："
                    "1) 设计导向美学而非安全（视线遮挡）；"
                    "2) 入口/出口过多导致访问控制困难；"
                    "3) 植被提供隐蔽（双面性：既是 amenity 也是风险）。"
                    "风险由设计-功能张力产生。"
                ),
                "guardianship_strategies": [
                    "设计活动节点（吸引合法使用）",
                    "修剪植被至视线通透（CPTED原则）",
                    "增加活动区域照明",
                    "设置清晰的空间分区"
                ]
            },

            # ========== 学校/教育机构周边 ==========
            {
                "entry_id": "school_zone",
                "place_type": "school_zone",
                "environmental_features": {
                    "poi_public": 0.9,
                    "poi_commercial": 0.3,
                    "road_density": 0.5,
                    "nightlight": 0.4,
                    "camera_coverage": 0.4,
                    "landuse_mix": 0.3,
                    "green_ratio": 0.5
                },
                "risk_mechanism": (
                    "教育机构周边："
                    "1) 上下学时段人流骤增（目标集中暴露）；"
                    "2) 放学后迅速真空；"
                    "3) 校园边界区域可能存在监控盲区。"
                    "风险由时间集中性与空间边界产生。"
                ),
                "guardianship_strategies": [
                    "上下学时段临时交通管控",
                    "增加边界照明与监控",
                    "设计接送区域避免街道混乱",
                    "与社区警务建立联动"
                ]
            },

            # ========== 近重复效应场景 ==========
            {
                "entry_id": "near_repeat_vulnerable",
                "place_type": "high_connectivity_area",
                "environmental_features": {
                    "poi_commercial": 0.6,
                    "poi_transport": 0.7,
                    "road_density": 0.8,
                    "nightlight": 0.5,
                    "camera_coverage": 0.3,
                    "landuse_mix": 0.6,
                    "green_ratio": 0.3
                },
                "risk_mechanism": (
                    "高连通性区域的近重复风险："
                    "1) 犯罪事件后信息通过空间网络传播；"
                    "2) 道路连通性便利犯罪人移动；"
                    "3) 监护资源固定，犯罪人可选择最弱点。"
                    "风险由空间拓扑与犯罪学习产生。"
                ),
                "guardianship_strategies": [
                    "事件后48小时增强巡逻（时间窗口关键）",
                    "在连通路径设置监控",
                    "快速修复破坏（破窗理论）",
                    "向周边区域发出预警"
                ]
            }
        ]

        # 审核并添加条目
        for entry_data in default_entries:
            if self.bias_auditor.audit_entry(entry_data):
                entry = KnowledgeEntry(**entry_data)
                self.entries.append(entry)

        self.save_kb()
        print(f"Built knowledge base with {len(self.entries)} entries")

    def save_kb(self):
        """保存知识库"""
        os.makedirs(os.path.dirname(self.kb_path), exist_ok=True)

        kb_dict = {
            "version": "1.0",
            "principles": [
                "Environment-only: No demographic references",
                "CPTED-based: Focus on design and opportunity",
                "Bias-audited: All entries reviewed for neutrality"
            ],
            "entries": [
                {
                    "entry_id": e.entry_id,
                    "place_type": e.place_type,
                    "environmental_features": e.environmental_features,
                    "risk_mechanism": e.risk_mechanism,
                    "guardianship_strategies": e.guardianship_strategies,
                    "bias_audit_passed": e.bias_audit_passed
                }
                for e in self.entries
            ]
        }

        with open(self.kb_path, 'w', encoding='utf-8') as f:
            json.dump(kb_dict, f, ensure_ascii=False, indent=2)

    def load_kb(self):
        """加载知识库"""
        with open(self.kb_path, 'r', encoding='utf-8') as f:
            kb_dict = json.load(f)

        self.entries = [
            KnowledgeEntry(**entry)
            for entry in kb_dict["entries"]
        ]
        print(f"Loaded {len(self.entries)} entries from {self.kb_path}")

    def query_similar(self, features: Dict[str, float], top_k: int = 3) -> List[KnowledgeEntry]:
        """
        基于环境特征相似度检索相关知识

        Args:
            features: 区域环境特征字典
            top_k: 返回最相似的k条

        Returns:
            相似的知识条目列表
        """
        similarities = []

        for entry in self.entries:
            sim = self._compute_similarity(features, entry.environmental_features)
            similarities.append((entry, sim))

        # 排序并返回top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in similarities[:top_k]]

    def _compute_similarity(self, features1: Dict[str, float],
                           features2: Dict[str, float]) -> float:
        """计算特征相似度（余弦相似度）"""
        keys = set(features1.keys()) & set(features2.keys())

        if not keys:
            return 0.0

        vec1 = np.array([features1.get(k, 0) for k in keys])
        vec2 = np.array([features2.get(k, 0) for k in keys])

        # 余弦相似度
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    def get_risk_mechanism_library(self) -> List[str]:
        """获取所有风险机制描述（用于分析模式）"""
        return [entry.risk_mechanism for entry in self.entries]


class BiasAuditor:
    """
    偏见审查器
    确保知识库条目符合环境犯罪学的"纯净"原则
    """

    # 禁止的词汇/概念（可能引入个体/群体偏见）
    PROHIBITED_PATTERNS = {
        "demographic": [
            r"\b(black|white|hispanic|asian|african|latino|immigrant|minority)\b",
            r"\b(race|racial|ethnic|ethnicity)\b",
            r"\b(young\s+male|teenager|youth)\s+(?:offender|criminal|suspect)",
            r"\b(unemployed|uneducated|illiterate)\b"
        ],
        "socioeconomic_stereotype": [
            r"\b(poor|poverty|low-income)\s+(?:people|population|community|neighborhood)",
            r"\b(ghetto|slum|hood)\b",
            r"\b(broken\s+home|single\s+parent|fatherless)\b",
            r"\b(welfare|food\s+stamp|section\s+8)\b",
            r"\b(culture\s+of\s+poverty|underclass)\b"
        ],
        "individual_pathology": [
            r"\b(criminal\s+mind|criminal\s+tendency|deviant)\b",
            r"\b(anti-social|psychopath|sociopath)\b",
            r"\b(moral\s+decay|family\s+values)\b"
        ],
        "spatial_proxy": [
            r"\b(bad|dangerous|rough)\s+neighborhood\s+(?:because|due\s+to).*?(?:people|residents)",
            r"\b(high\s+crime)\s+area\s+(?:because|due\s+to).*?(?:population|demographics)"
        ]
    }

    # 必须包含的环境相关词汇（确保焦点正确）
    REQUIRED_ENVIRONMENTAL_TERMS = [
        "lighting", "surveillance", "guardianship", "access", "visibility",
        "design", "density", "mix", "opportunity", "exposure", "space",
        "place", "environment", "physical", "landscape"
    ]

    def audit_entry(self, entry: Dict) -> bool:
        """
        审查单个条目

        Returns:
            True if passed, False if rejected
        """
        import re

        # 合并待检查文本
        text_to_check = ""
        if "risk_mechanism" in entry:
            text_to_check += entry["risk_mechanism"] + " "
        if "guardianship_strategies" in entry:
            text_to_check += " ".join(entry["guardianship_strategies"])

        text_lower = text_to_check.lower()

        # 检查1: 禁止词汇
        violations = []
        for category, patterns in self.PROHIBITED_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    violations.append(f"{category}: {pattern}")

        if violations:
            print(f"  [REJECTED] Entry {entry.get('entry_id', 'unknown')}")
            for v in violations:
                print(f"    - {v}")
            return False

        # 检查2: 必须包含环境术语（至少2个）
        env_terms_found = sum(1 for term in self.REQUIRED_ENVIRONMENTAL_TERMS
                            if term in text_lower)

        if env_terms_found < 2:
            print(f"  [WARNING] Entry {entry.get('entry_id', 'unknown')}: "
                  f"Only {env_terms_found} environmental terms found")
            # 警告但不拒绝，允许灵活性

        return True


# ================================
# 快捷使用接口
# ================================

def get_kb() -> EnvironmentalCriminologyKB:
    """获取知识库单例"""
    kb = EnvironmentalCriminologyKB()
    return kb


if __name__ == "__main__":
    # 测试知识库构建
    kb = EnvironmentalCriminologyKB()

    print(f"\nKnowledge base contains {len(kb.entries)} entries")
    print("\nExample query:")

    # 模拟查询
    query_features = {
        "poi_commercial": 0.85,
        "poi_transport": 0.6,
        "road_density": 0.75,
        "nightlight": 0.7,
        "camera_coverage": 0.5,
        "landuse_mix": 0.8,
        "green_ratio": 0.2
    }

    similar_entries = kb.query_similar(query_features, top_k=2)

    for i, entry in enumerate(similar_entries):
        print(f"\n[{i+1}] {entry.place_type}")
        print(f"    Risk mechanism: {entry.risk_mechanism[:100]}...")
        print(f"    Strategies: {entry.guardianship_strategies[0]}")
