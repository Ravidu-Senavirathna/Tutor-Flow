import os
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
INVOKE_URL     = "https://integrate.api.nvidia.com/v1/chat/completions"
NIM_MODEL      = "google/diffusiongemma-26b-a4b-it"
MAX_TOKENS = 500
TEMPERATURE = 0.3
# ─────────────────────────────────────────────────────────

stream = False

headers = {
    "Authorization": f"Bearer {NVIDIA_API_KEY}",
    "Accept": "text/event-stream" if stream else "application/json"
}

question = "what is q learning in reinforcement learning"

payload = {
  "model": NIM_MODEL,
  "messages": [{"role":"user","content":question}],
  "max_tokens": MAX_TOKENS,
  "temperature": TEMPERATURE,
  "top_p": 0.95,
  "stream": stream,
  "chat_template_kwargs": {"enable_thinking":True},
}

response = requests.post(INVOKE_URL, headers=headers, json=payload, stream=stream)
response.raise_for_status()

answer =  response.json()["choices"][0]["message"]["reasoning"].strip()

if stream:
    for line in response.iter_lines():
        if line:
            print(line.decode("utf-8"))
else:
    print(f'Question: {question}\n')
    print(f'Answer: \n{answer}')