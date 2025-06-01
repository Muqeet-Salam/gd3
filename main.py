from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import httpx
import os

app = FastAPI(
    title="InfoCore Semantic Search API",
    description="API for semantic search across technical documents using text embeddings",
    version="1.0.0"
)

# Configure CORS to allow all origins (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-API-Version"]
)

class SimilarityRequest(BaseModel):
    docs: List[str]
    query: str
    top_k: Optional[int] = 3  # Allow customization of number of results

class SimilarityResponse(BaseModel):
    matches: List[str]
    scores: Optional[List[float]] = None  # Include similarity scores if needed
    positions: Optional[List[int]] = None  # Include original positions

async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings using OpenAI's text-embedding-3-small model"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API key not configured")
    
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    json_data = {
        "input": texts,
        "model": "text-embedding-3-small",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=json_data)
            response.raise_for_status()
            data = response.json()
            return [item['embedding'] for item in data["data"]]
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail="Embedding API error")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors"""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    return np.dot(a, b) / (a_norm * b_norm)

@app.post("/similarity", response_model=SimilarityResponse)
async def compute_similarity(request: SimilarityRequest):
    """
    Compute semantic similarity between query and documents.
    
    Args:
        request: Contains 'docs' (list of document texts) and 'query' (search string)
    
    Returns:
        Response with top matching documents ordered by similarity
    """
    if not request.docs:
        return {"matches": []}
    
    if len(request.docs) > 100:  # Prevent too large requests
        raise HTTPException(status_code=400, detail="Too many documents (max 100)")
    
    # Get embeddings for all texts (documents + query)
    all_texts = request.docs + [request.query]
    embeddings = await get_embeddings(all_texts)
    
    # Separate document embeddings from query embedding
    doc_embeddings = embeddings[:-1]
    query_embedding = embeddings[-1]
    
    # Compute similarity scores
    scores = [cosine_similarity(doc_emb, query_embedding) 
              for doc_emb in doc_embeddings]
    
    # Get top K indices and their scores
    top_k = min(request.top_k, len(request.docs))
    top_indices = np.argsort(scores)[-top_k:][::-1]  # Descending order
    
    # Prepare response
    top_docs = [request.docs[i] for i in top_indices]
    top_scores = [float(scores[i]) for i in top_indices]  # Convert numpy float to Python float
    
    return {
        "matches": top_docs,
        "scores": top_scores,
        "positions": [int(i) for i in top_indices]  # Convert numpy int to Python int
    }
