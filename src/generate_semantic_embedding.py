import os
import json
import numpy as np
import requests
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import argparse

# RAG增强模块
from rag_semantic_generator import RAGSemanticGenerator, generate_rag_semantic_embeddings
from env_criminology_kb import EnvironmentalCriminologyKB

# =========================
# 0. 初始化与配置
# =========================
SAVE_DIR = "data/processed/"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 必须在导入transformers前设置！
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.makedirs(SAVE_DIR, exist_ok=True)

CACHE_FILE = os.path.join(SAVE_DIR, "semantic_texts_v2.json")
CACHE_FILE_RAG = os.path.join(SAVE_DIR, "semantic_texts_rag.json")
EMBEDDING_FILE = os.path.join(SAVE_DIR, "semantic_embedding_v2.npy")
EMBEDDING_FILE_RAG = os.path.join(SAVE_DIR, "semantic_embedding_rag.npy")

# 配置并发数：取决于你的显存和本地LLM的并发处理能力
# Ollama 默认通常支持一定的并发，建议设为 4-8
MAX_WORKERS = 4

# RAG配置
USE_RAG = True  # 默认启用RAG
RAG_WEIGHT = 0.7 

# =========================
# 1. 加载特征 (保持原样)
# =========================
def load_features():
    files = {
        'poi': 'poi_features.npy',
        'road': 'road_density.npy',
        'light': 'nightlight_features.npy',
        'landuse': 'landuse_features.npy',
        'green': 'green_features.npy',
        'weather': 'weather_features.npy',
        'camera': 'camera_features.npy'
    }
    features = {}
    for k, f in files.items():
        path = os.path.join(SAVE_DIR, f)
        assert os.path.exists(path), f"缺失文件: {path}"
        features[k] = np.load(path)
        print(f"{k} shape: {features[k].shape}")

    N = features['poi'].shape[0]
    weather = features['weather']
    weather_global = {
        'rain_freq': weather[:,0].mean(),
        'temp_avg': weather[:,4].mean()
    }
    return features, N, weather_global

# =========================
# 2. 工具函数 (增强描述力)
# =========================
def level(x):
    val = x.mean() if isinstance(x, np.ndarray) else x
    return "low" if val < 0.33 else "moderate" if val < 0.66 else "high"

def describe_landuse(vec):
    res, com, ind = vec[:3]
    if max(res, com, ind) < 0.7:
        mix = "mixed-use urban fabric"
    else:
        mix = ["residential area", "commercial hub", "industrial zone"][np.argmax([res, com, ind])]
    return f"{mix} (res:{res:.2f}, com:{com:.2f}, ind:{ind:.2f})"

def describe_poi(vec):
    commercial, transport, public = vec[0], vec[1], vec[2]
    return f"commercial activity index: {commercial:.2f}, transport accessibility: {transport:.2f}, public services: {public:.2f}"

def describe_camera(vec):
    """摄像头监护能力描述"""
    
    val = vec.mean() if isinstance(vec, np.ndarray) else vec
    
    if val < 0.33:
        return "limited surveillance coverage"
    elif val < 0.66:
        return "moderate camera coverage"
    else:
        return "dense surveillance network"
# =========================
# 3. Prompt 构建 (注入犯罪学理论)
# =========================
def build_prompt(i, features, weather_desc):
    # 提取当前区域特征
    lu = describe_landuse(features['landuse'][i])
    po = describe_poi(features['poi'][i])
    rd = level(features['road'][i])
    nl = "well-lit" if level(features['light'][i]) == "high" else "dimly lit"
    gs = "rich green space" if level(features['green'][i]) == "high" else "lacking vegetation"
    cam = describe_camera(features['camera'][i])

    return f"""
Act as an environmental criminologist. Analyze the urban micro-environment of Region {i}:

- Land Use: {lu}
- POI Activity: {po}
- Infrastructure: {rd} road density, {nl} at night.
- Surveillance: {cam}
- Natural Environment: {gs}, {weather_desc}.

Task: Briefly identify:

1. Whether this area acts as a 'Crime Generator' or 'Attractor'.
2. The level of 'Capable Guardianship' (formal/informal surveillance).
3. Potential target suitability for street-level crime.

Constraint: 2-3 concise sentences. Focus on environmental risk factors.
"""

# =========================
# 4. 优化后的 LLM 调用 (增加健壮性)
# =========================
def query_llm(prompt, region_id):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen3:4b", 
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2, # 降低随机性
            "num_predict": 100,  # 严格限制长度，加速生成
            "stop": ["\n\n"]
        }
    }
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()['response'].strip()
    except Exception:
        return "Typical urban area with moderate environmental risk factors."

# =========================
# 5. RAG增强模式生成
# =========================
def generate_with_rag(features, N, weather_global):
    """
    使用RAG增强生成语义描述
    """
    print("=" * 60)
    print("使用RAG增强模式生成语义描述")
    print("=" * 60)

    # 构建特征字典
    features_dict = {}
    for i in range(N):
        features_dict[i] = {
            'poi_commercial': features['poi'][i][0] if len(features['poi'][i]) > 0 else 0,
            'poi_transport': features['poi'][i][1] if len(features['poi'][i]) > 1 else 0,
            'poi_public': features['poi'][i][2] if len(features['poi'][i]) > 2 else 0,
            'road_density': features['road'][i].mean() if isinstance(features['road'][i], np.ndarray) else features['road'][i],
            'nightlight': features['light'][i].mean() if isinstance(features['light'][i], np.ndarray) else features['light'][i],
            'camera_coverage': features['camera'][i].mean() if isinstance(features['camera'][i], np.ndarray) else features['camera'][i],
            'landuse_mix': np.std(features['landuse'][i][:3]) if len(features['landuse'][i]) >= 3 else 0,
            'green_ratio': features['green'][i].mean() if isinstance(features['green'][i], np.ndarray) else features['green'][i]
        }

    # 使用RAG生成
    texts = generate_rag_semantic_embeddings(
        features_dict=features_dict,
        weather_global=weather_global,
        output_path=CACHE_FILE_RAG,
        use_rag=True
    )

    return texts

# =========================
# 6. 执行主流程
# =========================
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Generate semantic embeddings for crime prediction')
    parser.add_argument('--mode', type=str, default='rag', choices=['rag', 'basic'],
                       help='Generation mode: rag (with knowledge base) or basic (original)')
    parser.add_argument('--force', action='store_true',
                       help='Force regeneration even if cache exists')
    args = parser.parse_args()

    global USE_RAG
    USE_RAG = (args.mode == 'rag')

    features, N, weather_global = load_features()
    weather_desc = "cold climate" if weather_global['temp_avg'] < 10 else "mild climate"

    # 选择生成模式
    if USE_RAG:
        # RAG增强模式
        cache_file = CACHE_FILE_RAG
        embedding_file = EMBEDDING_FILE_RAG

        if os.path.exists(cache_file) and not args.force:
            print(f"从RAG缓存加载文本: {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                texts = json.load(f)["texts"]
        else:
            texts = generate_with_rag(features, N, weather_global)
    else:
        # 基础模式（原有逻辑）
        cache_file = CACHE_FILE
        embedding_file = EMBEDDING_FILE

        if os.path.exists(cache_file) and not args.force:
            print(f"从缓存加载文本: {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                texts = json.load(f)["texts"]
        else:
            print(f"开始并发生成语义描述 (Threads: {MAX_WORKERS})...")
            texts = [None] * N

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_id = {
                    executor.submit(query_llm, build_prompt(i, features, weather_desc), i): i
                    for i in range(N)
                }

                for future in tqdm(as_completed(future_to_id), total=N):
                    idx = future_to_id[future]
                    texts[idx] = future.result()

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"texts": texts}, f, indent=2, ensure_ascii=False)

    # 生成语义嵌入
    print("\n正在生成语义嵌入 (BGE-M3)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("BAAI/bge-m3", device=device)

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    np.save(embedding_file, embeddings)
    print(f"完成！特征已保存至 {embedding_file}")
    print(f"生成模式: {'RAG增强' if USE_RAG else '基础模式'}")

if __name__ == "__main__":
    main()