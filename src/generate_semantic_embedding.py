import os
import json
import time
import numpy as np
import requests
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# =========================
# 0. 初始化
# =========================
SAVE_DIR = "data/processed/"
CACHE_FILE = os.path.join(SAVE_DIR, "semantic_texts.json")

# =========================
# 1. 加载特征
# =========================
poi = np.load("data/processed/poi_features.npy")
road = np.load("data/processed/road_density.npy")
light = np.load("data/processed/nightlight_features.npy")
landuse = np.load("data/processed/landuse_features.npy")
green = np.load("data/processed/green_ratio.npy")

N = poi.shape[0]

# =========================
# 2. 工具函数（保持不变）
# =========================
def level(x):
    if isinstance(x, np.ndarray):
        x = x.mean()
    if x < 0.33:
        return "low"
    elif x < 0.66:
        return "moderate"
    else:
        return "high"

def describe_landuse(vec):
    if isinstance(vec, np.ndarray) and vec.ndim > 1:
        vec = vec.mean(axis=0)
    res, com, ind = vec[:3]
    return f"residential {res:.2f}, commercial {com:.2f}, industrial {ind:.2f}"

def describe_poi(vec):
    return f"overall intensity {np.round(vec.mean(),2)}"

# =========================
# 3. Prompt构造（保持不变）
# =========================
def build_prompt(i):
    return f"""
You are an expert in urban studies and crime analysis.

Region characteristics:
- POI: {describe_poi(poi[i])}
- Road density: {level(road[i])}
- Nighttime light: {level(light[i])}
- Land use: {describe_landuse(landuse[i])}
- Green space: {level(green[i])}

Task:
1. Describe the functional type of this region
2. Describe human activity patterns
3. Infer potential violent crime risk factors

Keep it within 2-3 sentences.
Do NOT use actual crime data.
"""

# =========================
# 4. 调用LLM（修改为 Ollama 原生 API）
# =========================
def query_llm(prompt, retries=3):
    """使用 Ollama 原生 API"""
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "qwen3:4b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 500  # 限制输出长度
        }
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.Timeout:
            print(f"超时，重试 {attempt+1}/{retries}")
            time.sleep(2)
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
            if attempt == retries - 1:
                raise
            time.sleep(2)
    
    raise Exception("所有重试都失败")

# =========================
# 5. 加载缓存
# =========================
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        texts = json.load(f)
else:
    texts = [""] * N

# =========================
# 6. 批量生成语义文本
# =========================
print("🚀 Generating semantic descriptions...")

for i in tqdm(range(N)):
    if texts[i] != "":
        continue

    prompt = build_prompt(i)

    try:
        text = query_llm(prompt)
        texts[i] = text

        # 每生成10条保存一次
        if i % 10 == 0:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(texts, f, ensure_ascii=False, indent=2)

        time.sleep(0.2)  # 防止过载

    except Exception as e:
        print(f"Error at {i}: {e}")
        time.sleep(2)

# 最终保存
with open(CACHE_FILE, "w", encoding="utf-8") as f:
    json.dump(texts, f, ensure_ascii=False, indent=2)

print("✅ 文本生成完成")

# =========================
# 7. 文本 → embedding（保持不变）
# =========================
print("🔄 Encoding embeddings...")

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True
)

print("embedding shape:", embeddings.shape)

# =========================
# 8. 保存
# =========================
np.save(os.path.join(SAVE_DIR, "semantic_embedding.npy"), embeddings)

print("✅ semantic_embedding.npy 已生成！")