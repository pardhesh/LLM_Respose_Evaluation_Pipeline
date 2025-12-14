import json
import time
import re
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

EMBED_CACHE = {}

def embed(text: str) -> np.ndarray:
    if text not in EMBED_CACHE:
        EMBED_CACHE[text] = model.encode(text, normalize_embeddings=True)
    return EMBED_CACHE[text]

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def similarity(a: np.ndarray, b: np.ndarray) -> float:
    return cosine_similarity([a], [b])[0][0]

def relevance_score(user_query: str, ai_response: str) -> float:
    return float(similarity(embed(user_query), embed(ai_response)))

def hallucination_check(
    ai_response: str,
    contexts: List[str],
    threshold: float = 0.65 
):
    ai_response = re.sub(r"http\S+", "", ai_response)

    sentences = [
        s.strip() for s in ai_response.split(".")
        if len(s.strip().split()) >= 4  
    ]

    context_embeddings = [embed(c) for c in contexts]
    hallucinated = []

    for sentence in sentences:
        sent_emb = embed(sentence)
        max_sim = float(max(
            similarity(sent_emb, ctx_emb)
            for ctx_emb in context_embeddings
        ))

        if max_sim < threshold:
            hallucinated.append({
                "claim": sentence,
                "similarity": round(max_sim, 3)
            })

    rate = len(hallucinated) / max(len(sentences), 1)

    return round(rate, 2), hallucinated

def estimate_cost(prompt_tokens: int, completion_tokens: int):
    COST_PER_1K = 0.002
    total_tokens = prompt_tokens + completion_tokens
    return round((total_tokens / 1000) * COST_PER_1K, 6)

def evaluate(chat_json, context_json):
    turns = chat_json["conversation_turns"]

    user_query = next(
        t["message"] for t in reversed(turns)
        if t["role"] == "User"
    )
    ai_response = next(
        t["message"] for t in reversed(turns)
        if t["role"] == "AI/Chatbot"
    )

    contexts = [
        v["text"]
        for v in context_json["data"]["vector_data"][:5]
    ]

    start = time.time()

    relevance = relevance_score(user_query, ai_response)
    hallucination_rate, hallucinated_claims = hallucination_check(
        ai_response, contexts
    )

    latency = round(time.time() - start, 3)

    prompt_tokens = len(user_query.split()) + sum(len(c.split()) for c in contexts)
    completion_tokens = len(ai_response.split())

    cost = estimate_cost(prompt_tokens, completion_tokens)

    reliability_score = float(round(
        0.5 * relevance + 0.5 * (1 - hallucination_rate), 3
    ))

    return {
        "relevance_score": round(relevance, 3),
        "hallucination_rate": hallucination_rate,
        "hallucinated_claims": hallucinated_claims,
        "reliability_score": reliability_score,
        "latency_seconds": latency,
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        },
        "estimated_cost_usd": cost
    }


if __name__ == "__main__":
    chat = load_json("samples/sample-chat-conversation-02.json")
    context = load_json("samples/sample_context_vectors-02.json")

    report = evaluate(chat, context)
    print(json.dumps(report, indent=2))
