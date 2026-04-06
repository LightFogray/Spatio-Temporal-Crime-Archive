"""
RAG增强的语义生成器
结合环境犯罪学知识库进行检索增强生成
用于替代/增强纯LLM的开放式生成
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import requests
from dataclasses import dataclass

from src.env_criminology_kb import EnvironmentalCriminologyKB, KnowledgeEntry


@dataclass
class RAGContext:
    """RAG检索上下文"""
    retrieved_entries: List[KnowledgeEntry]
    similarity_scores: List[float]
    place_type_match: str  # 最匹配的场所类型
    risk_mechanism_template: str  # 风险机制模板
    guardianship_suggestions: List[str]  # 监护策略建议


class RAGSemanticGenerator:
    """
    RAG增强的环境犯罪学语义生成器

    流程：
    1. 特征向量化 -> 检索知识库 -> 获取相似案例
    2. 使用知识库模板约束LLM生成
    3. 偏见审查输出
    """

    def __init__(self,
                 kb: Optional[EnvironmentalCriminologyKB] = None,
                 llm_model: str = "qwen3:4b",
                 embedding_model: str = "BAAI/bge-m3",
                 use_rag: bool = True,
                 rag_weight: float = 0.7):
        """
        Args:
            kb: 环境犯罪学知识库
            llm_model: 本地LLM模型名称
            embedding_model: 用于检索的嵌入模型
            use_rag: 是否启用RAG
            rag_weight: RAG约束强度（0-1）
        """
        self.kb = kb or EnvironmentalCriminologyKB()
        self.llm_model = llm_model
        self.use_rag = use_rag
        self.rag_weight = rag_weight

        # 初始化嵌入模型
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # 预计算知识库嵌入
        self._precompute_kb_embeddings()

        # 偏见审查器
        self.bias_auditor = BiasAuditor()

    def _precompute_kb_embeddings(self):
        """预计算知识库条目的嵌入向量"""
        self.kb_embeddings = []
        self.kb_texts = []

        for entry in self.kb.entries:
            # 构建检索文本
            text = f"{entry.place_type}. {entry.risk_mechanism}"
            self.kb_texts.append(text)

        if self.kb_texts:
            self.kb_embeddings = self.embedding_model.encode(
                self.kb_texts,
                convert_to_tensor=True,
                show_progress_bar=True
            )
            print(f"Precomputed {len(self.kb_texts)} KB embeddings")

    def retrieve_context(self, features: Dict[str, float],
                         top_k: int = 3) -> RAGContext:
        """
        基于特征检索相关知识

        Args:
            features: 区域环境特征
            top_k: 检索数量

        Returns:
            RAG上下文
        """
        # 方法1: 基于特征相似度（结构化匹配）
        structured_matches = self.kb.query_similar(features, top_k=top_k)

        # 方法2: 基于语义相似度（可选，用于更灵活匹配）
        # 构建特征描述文本
        feature_desc = self._features_to_text(features)
        query_embedding = self.embedding_model.encode(feature_desc, convert_to_tensor=True)

        # 计算相似度
        if len(self.kb_embeddings) > 0:
            similarities = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0),
                self.kb_embeddings
            )
            top_indices = torch.topk(similarities, min(top_k, len(similarities))).indices.cpu().numpy()
            semantic_matches = [self.kb.entries[i] for i in top_indices]
        else:
            semantic_matches = []

        # 合并结果（去重）
        seen_ids = set()
        merged_entries = []
        for entry in structured_matches + semantic_matches:
            if entry.entry_id not in seen_ids:
                merged_entries.append(entry)
                seen_ids.add(entry.entry_id)
            if len(merged_entries) >= top_k:
                break

        # 计算相似度分数
        scores = [self.kb._compute_similarity(features, e.environmental_features)
                 for e in merged_entries]

        # 构建上下文
        context = RAGContext(
            retrieved_entries=merged_entries,
            similarity_scores=scores,
            place_type_match=merged_entries[0].place_type if merged_entries else "unknown",
            risk_mechanism_template=merged_entries[0].risk_mechanism if merged_entries else "",
            guardianship_suggestions=merged_entries[0].guardianship_strategies if merged_entries else []
        )

        return context

    def _features_to_text(self, features: Dict[str, float]) -> str:
        """将特征转换为描述文本（用于语义检索）"""
        descriptions = []

        if features.get('poi_commercial', 0) > 0.7:
            descriptions.append("high commercial density")
        if features.get('poi_transport', 0) > 0.7:
            descriptions.append("transit hub")
        if features.get('camera_coverage', 0) < 0.3:
            descriptions.append("low surveillance")
        if features.get('nightlight', 0) < 0.3:
            descriptions.append("poor lighting")
        if features.get('landuse_mix', 0) > 0.7:
            descriptions.append("mixed land use")

        return "Area with " + ", ".join(descriptions) if descriptions else "urban area"

    def generate_semantic_description(self, region_id: int,
                                     features: Dict[str, float],
                                     weather_desc: str) -> Dict:
        """
        生成RAG增强的语义描述

        Args:
            region_id: 区域ID
            features: 环境特征
            weather_desc: 天气描述

        Returns:
            包含描述和元信息的字典
        """
        # 步骤1: 检索相关知识
        rag_context = self.retrieve_context(features, top_k=2)

        # 步骤2: 构建RAG约束的Prompt
        if self.use_rag and rag_context.retrieved_entries:
            prompt = self._build_rag_prompt(
                region_id, features, weather_desc, rag_context
            )
        else:
            # 回退到基础Prompt
            prompt = self._build_basic_prompt(region_id, features, weather_desc)

        # 步骤3: 调用LLM生成
        description = self._query_llm(prompt)

        # 步骤4: 偏见审查
        audit_result = self.bias_auditor.audit_description(description)

        # 步骤5: 如果审查失败，使用知识库模板生成保守描述
        if not audit_result['passed']:
            print(f"  [Region {region_id}] LLM output failed bias audit, using conservative template")
            description = self._generate_conservative_description(rag_context, features)

        return {
            'region_id': region_id,
            'description': description,
            'rag_context': {
                'matched_place_type': rag_context.place_type_match,
                'similarity_score': rag_context.similarity_scores[0] if rag_context.similarity_scores else 0,
                'retrieved_entries': [e.entry_id for e in rag_context.retrieved_entries],
                'guardianship_suggestions': rag_context.guardianship_suggestions[:2]
            },
            'bias_audit': audit_result,
            'prompt_type': 'RAG' if self.use_rag else 'Basic'
        }

    def _build_rag_prompt(self, region_id: int, features: Dict[str, float],
                         weather_desc: str, context: RAGContext) -> str:
        """构建RAG增强的Prompt"""

        # 提取特征值
        poi_com = features.get('poi_commercial', 0)
        poi_trans = features.get('poi_transport', 0)
        road = features.get('road_density', 0)
        light = features.get('nightlight', 0)
        camera = features.get('camera_coverage', 0)
        landuse_mix = features.get('landuse_mix', 0)
        green = features.get('green_ratio', 0)

        # 最匹配的知识条目
        best_match = context.retrieved_entries[0]

        prompt = f"""You are an environmental criminologist analyzing urban space.

## REGION CHARACTERISTICS
- Region ID: {region_id}
- Commercial activity: {self._level(poi_com)} ({poi_com:.2f})
- Transport accessibility: {self._level(poi_trans)} ({poi_trans:.2f})
- Road density: {self._level(road)} ({road:.2f})
- Night lighting: {self._level(light)} ({light:.2f})
- Camera coverage: {self._level(camera)} ({camera:.2f})
- Land use mix: {self._level(landuse_mix)} ({landuse_mix:.2f})
- Green space: {self._level(green)} ({green:.2f})
- Weather context: {weather_desc}

## REFERENCE CASE (Most Similar Environment)
Place type: {best_match.place_type}
Environmental risk mechanism: {best_match.risk_mechanism}

Note: This reference describes how similar PHYSICAL ENVIRONMENTS create crime opportunities.
DO NOT copy the reference text. Use it as a framework to analyze the current region's specific features.

## YOUR TASK
Based SOLELY on the environmental characteristics above, provide:

1. **Crime Generator/Attractor Assessment** (1 sentence)
   - Does this environment generate crime opportunities through target concentration, poor guardianship, or easy access?
   - Use environmental terms only (lighting, density, visibility, access control)

2. **Guardianship Analysis** (1 sentence)
   - Evaluate formal surveillance (cameras) and natural surveillance (lighting, visibility)
   - Identify specific environmental weaknesses

3. **Risk Summary** (1 sentence)
   - Synthesize the environmental risk profile
   - Focus on how physical design creates opportunities

## CRITICAL CONSTRAINTS
- ONLY discuss physical environment factors (lighting, design, density, access)
- NEVER mention race, ethnicity, socioeconomic status, or "types of people"
- NEVER use phrases like "bad neighborhood" or "high-crime area" without explaining the SPECIFIC environmental mechanisms
- Base your analysis STRICTLY on the numerical characteristics provided

## OUTPUT FORMAT
Respond in exactly 3 sentences, following the structure above.
Be specific, concise, and strictly environmental in your analysis.
"""
        return prompt

    def _build_basic_prompt(self, region_id: int, features: Dict[str, float],
                           weather_desc: str) -> str:
        """基础Prompt（无RAG）"""
        return f"""Analyze Region {region_id} as an environmental criminologist.

Features:
- Commercial: {features.get('poi_commercial', 0):.2f}
- Transport: {features.get('poi_transport', 0):.2f}
- Lighting: {features.get('nightlight', 0):.2f}
- Cameras: {features.get('camera_coverage', 0):.2f}

Task: 3 sentences on (1) crime generator/attractor, (2) guardianship, (3) environmental risk.

Constraint: Environment-only. No demographics.
"""

    def _generate_conservative_description(self, context: RAGContext,
                                          features: Dict[str, float]) -> str:
        """
        当LLM生成失败时，使用知识库模板生成保守描述
        """
        if not context.retrieved_entries:
            return "Urban area with moderate environmental risk factors. Standard surveillance and lighting conditions."

        best_match = context.retrieved_entries[0]

        # 基于特征匹配程度调整描述
        sim_score = context.similarity_scores[0] if context.similarity_scores else 0.5

        if sim_score > 0.8:
            # 高度匹配，直接使用知识库描述
            return best_match.risk_mechanism[:200] + "..."
        else:
            # 部分匹配，提取关键短语
            risk_phrases = [
                "target exposure due to commercial concentration",
                "guardianship gaps from limited surveillance",
                "access convenience via transport connectivity",
                "lighting deficiencies reducing visibility"
            ]

            # 根据特征选择相关短语
            selected = []
            if features.get('poi_commercial', 0) > 0.5:
                selected.append(risk_phrases[0])
            if features.get('camera_coverage', 0) < 0.4:
                selected.append(risk_phrases[1])
            if features.get('poi_transport', 0) > 0.5:
                selected.append(risk_phrases[2])
            if features.get('nightlight', 0) < 0.4:
                selected.append(risk_phrases[3])

            if selected:
                return f"Region shows {' and '.join(selected[:2])}. Environmental factors suggest moderate opportunity structure for street-level crime."
            else:
                return "Environmentally typical urban area with standard crime opportunity profile."

    def _query_llm(self, prompt: str, max_retries: int = 3) -> str:
        """查询本地LLM"""
        url = "http://localhost:11434/api/generate"

        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,  # 降低随机性
                "num_predict": 150,
                "stop": ["\n\n", "##"]
            }
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, timeout=30)
                response.raise_for_status()
                return response.json()['response'].strip()
            except Exception as e:
                print(f"  LLM query failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return "Environmental risk assessment pending due to system error."

        return "Environmental risk assessment pending."

    def _level(self, x: float) -> str:
        """数值转等级描述"""
        if x < 0.33:
            return "low"
        elif x < 0.66:
            return "moderate"
        else:
            return "high"


class BiasAuditor:
    """
    生成描述的偏见审查器
    """

    PROHIBITED_PATTERNS = [
        r"\b(black|white|hispanic|asian|african|latino|immigrant)s?\b",
        r"\b(race|racial|ethnic|ethnicity)\b",
        r"\b(poor|poverty|low-income)\s+(?:people|population|residents)\b",
        r"\b(ghetto|slum|hood)\b",
        r"\b(unemployed|uneducated)\s+(?:population|people)\b",
        r"\b(culture|cultural)\s+(?:of\s+poverty|problem|factor)\b",
        r"\b(broken\s+home|single\s+parent)\b",
        r"\b(neighborhood|area)\s+(?:is|has|because).*?(?:people|population|residents)",
        r"\b(dangerous|bad|rough)\s+(?:area|neighborhood)\s+(?:due\s+to|because\s+of).*?(?:people|residents)"
    ]

    REQUIRED_ENVIRONMENTAL_TERMS = [
        "lighting", "surveillance", "guardianship", "density",
        "access", "design", "exposure", "environment"
    ]

    def audit_description(self, description: str) -> Dict:
        """审查描述"""
        import re

        text_lower = description.lower()

        # 检查禁止模式
        violations = []
        for pattern in self.PROHIBITED_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                violations.append(pattern)

        # 检查是否包含环境术语
        env_terms = sum(1 for term in self.REQUIRED_ENVIRONMENTAL_TERMS
                       if term in text_lower)

        return {
            'passed': len(violations) == 0 and env_terms >= 1,
            'violations': violations,
            'environmental_terms': env_terms,
            'description_length': len(description)
        }


# ================================
# 批量生成接口
# ================================

def generate_rag_semantic_embeddings(features_dict: Dict[int, Dict],
                                    weather_global: Dict,
                                    output_path: str = "data/processed/semantic_texts_rag.json",
                                    use_rag: bool = True) -> List[str]:
    """
    批量生成RAG增强的语义描述

    Args:
        features_dict: {region_id: features} 字典
        weather_global: 全局天气信息
        output_path: 输出路径
        use_rag: 是否使用RAG

    Returns:
        描述文本列表
    """
    print("=" * 60)
    print(f"Generating RAG-enhanced semantic descriptions")
    print(f"RAG enabled: {use_rag}")
    print("=" * 60)

    # 初始化生成器
    generator = RAGSemanticGenerator(use_rag=use_rag)

    # 天气描述
    weather_desc = "cold climate" if weather_global.get('temp_avg', 15) < 10 else "mild climate"

    # 批量生成
    results = []
    texts = []

    for i, (region_id, features) in enumerate(features_dict.items()):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(features_dict)}...")

        result = generator.generate_semantic_description(
            region_id, features, weather_desc
        )

        results.append(result)
        texts.append(result['description'])

    # 保存结果
    output = {
        'generation_method': 'RAG' if use_rag else 'Basic',
        'total_regions': len(texts),
        'rag_statistics': {
            'avg_similarity': np.mean([r['rag_context']['similarity_score']
                                      for r in results if 'rag_context' in r]),
            'bias_audit_pass_rate': np.mean([r['bias_audit']['passed']
                                            for r in results if 'bias_audit' in r])
        },
        'texts': texts,
        'detailed_results': results
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {output_path}")
    print(f"RAG avg similarity: {output['rag_statistics']['avg_similarity']:.3f}")
    print(f"Bias audit pass rate: {output['rag_statistics']['bias_audit_pass_rate']*100:.1f}%")

    return texts


if __name__ == "__main__":
    # 测试RAG生成器
    print("Testing RAG Semantic Generator...")

    generator = RAGSemanticGenerator(use_rag=True)

    # 测试特征
    test_features = {
        'poi_commercial': 0.85,
        'poi_transport': 0.70,
        'road_density': 0.75,
        'nightlight': 0.40,  # 照明不足
        'camera_coverage': 0.25,  # 监控不足
        'landuse_mix': 0.80,
        'green_ratio': 0.15
    }

    result = generator.generate_semantic_description(
        region_id=0,
        features=test_features,
        weather_desc="mild climate"
    )

    print("\n" + "="*60)
    print("Generated Description:")
    print("="*60)
    print(result['description'])
    print("\nRAG Context:")
    print(f"  Matched type: {result['rag_context']['matched_place_type']}")
    print(f"  Similarity: {result['rag_context']['similarity_score']:.3f}")
    print(f"  Bias audit: {'PASSED' if result['bias_audit']['passed'] else 'FAILED'}")
