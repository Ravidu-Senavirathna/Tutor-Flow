import os
import requests
from dotenv import load_dotenv
from retriever import load_retriever, retrieve

load_dotenv()

# ── Config ────────────────────────────────────────────────
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
INVOKE_URL     = "https://integrate.api.nvidia.com/v1/chat/completions"
NIM_MODEL      = "google/diffusiongemma-26b-a4b-it"
MAX_TOKENS = 1000
TEMPERATURE = 0.3
stream = False
# ─────────────────────────────────────────────────────────

headers = {
    "Authorization": f"Bearer {NVIDIA_API_KEY}",
    "Accept": "text/event-stream" if stream else "application/json"
}


"""Build a prompt with retrieved context + question."""
def build_prompt(role: str, question: str, chunks: list[dict]) -> str:

    parts = []
    for chunk in chunks:
        parts.append(f"[Source: {chunk['source']}]\n{chunk['text']}")

    context = "\n\n".join(parts)

    prompt = f"""
        ROLE:{role}
        CONTEXT:{context}
        QUESTION:{question}
        ANSWER:
        """

    return prompt


def ask_question(role: str, question: str):

    model , collection = load_retriever()
    data = retrieve(question, model, collection, top_k=2)
    good_data = [c for c in data if c["score"] > 0.5]

    prompt = build_prompt(role, question, good_data)

    payload = {
    "model": NIM_MODEL,
    "messages": [{"role":"user","content":prompt}],
    "max_tokens": MAX_TOKENS,
    "temperature": TEMPERATURE,
    "top_p": 0.95,
    "stream": stream,
    "chat_template_kwargs": {"enable_thinking":True},
    }

    response = requests.post(INVOKE_URL, headers=headers, json=payload, stream=stream)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["reasoning"].strip()


if __name__ == "__main__":
    role = "A teacher"
    question = "what is model based in reinforcement learning"
    answer = ask_question(role, question)
    print(answer)