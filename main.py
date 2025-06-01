from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import httpx
import os

# FastAPI app
app = FastAPI()

# CORS config (allow all)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# Input format
class SimilarityRequest(BaseModel):
    docs: List[str]
    query: str

# Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b)

# Call Jina AI Embedding API
async def get_jina_embeddings(texts: List[str]):
    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer jina_f4ad6c34ec8e4a54b69eecef970e92b514LN270Z26uqxJrH6bxqPuRHsiC5",
        "Content-Type": "application/json"
    }
    json_data = {
        "input": texts,
        "model": "jina-embeddings-v2-base-en",  # or multilingual if needed
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=json_data)
        response.raise_for_status()
        data = response.json()
        return [item['embedding'] for item in data["data"]]

# Endpoint
@app.post("/similarity")
async def similarity(req: SimilarityRequest):
    all_texts = req.docs + [req.query]
    embeddings = await get_jina_embeddings(all_texts)

    doc_embeddings = embeddings[:-1]
    query_embedding = embeddings[-1]

    # Compute similarities
    scores = [cosine_similarity(doc_emb, query_embedding) for doc_emb in doc_embeddings]

    # Get top 3 indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]

    # Return top 3 matching docs
    top_docs = [req.docs[i] for i in top_indices]

    return {"matches": top_docs}
