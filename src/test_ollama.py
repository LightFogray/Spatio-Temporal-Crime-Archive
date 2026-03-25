import requests

url = "http://localhost:11434/api/generate"
payload = {
    "model": "qwen3:4b",
    "prompt": "你好，请用一句话介绍自己",
    "stream": False
}

response = requests.post(url, json=payload)
result = response.json()
print(result.get("response"))