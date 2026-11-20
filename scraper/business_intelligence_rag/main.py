#python -m uvicorn main:app --reload

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import numpy as np
import json
import requests
import os

app = FastAPI(title="Market Intelligence Hybrid RAG API", version="1.0")

# Enable CORS for future React/Next.js frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FAISS_PATH = "business_faiss.index"
DOCS_PATH = "business_docs.json"
OLLAMA_EMB_URL = "http://localhost:11434/api/embeddings"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

class ChatRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"status": "online", "message": "Ollama + FAISS Hybrid RAG API is running."}

# CRITICAL CHANGE: We switch from 'async def' to standard 'def' 
# This forces FastAPI to run this route inside a separate thread pool,
# preventing the synchronous 'requests' package from crashing Uvicorn's async event loop on Windows.
@app.post("/api/chat")
@app.post("/api/chat/")
def chat_endpoint(payload: ChatRequest):
    user_query = payload.question

    if not (os.path.exists(FAISS_PATH) and os.path.exists(DOCS_PATH)):
        raise HTTPException(status_code=500, detail="Database files missing! Run Scrapy first.")

    # 1. Load FAISS database and text logs
    index = faiss.read_index(FAISS_PATH)
    with open(DOCS_PATH, 'r', encoding='utf-8') as f:
        stored_documents = json.load(f)

    # 2. Extract keyword triggers
    words = user_query.split()
    triggers = [w.strip("?,.!") for w in words if any(c.isdigit() for c in w) or len(w) > 6]

    # --- PROCESS 1: EXACT MATCH (WEB2 KEYWORD LOOKUP) ---
    exact_matches = []
    if triggers:
        for doc in stored_documents:
            if any(trigger.lower() in doc['content'].lower() for trigger in triggers):
                exact_matches.append(doc)

    # --- PROCESS 2: SEMANTIC SEARCH (WEB3 VECTOR LOOKUP) ---
    try:
        emb_res = requests.post(OLLAMA_EMB_URL, json={"model": "nomic-embed-text", "prompt": user_query}, timeout=15)
        if emb_res.status_code != 200:
            raise HTTPException(status_code=502, detail="Ollama embeddings service failed.")
        
        question_vector = np.array([emb_res.json().get('embedding')]).astype('float32')
        
        k = min(5, index.ntotal)
        distances, indices = index.search(question_vector, k)
        
        semantic_matches = []
        # Robust flat matrix loop adaptation
        for row in indices:
            for idx in row:
                if idx < len(stored_documents) and idx != -1:
                    semantic_matches.append(stored_documents[idx])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding compilation error: {str(e)}")

    # --- PROCESS 3: HYBRID FUSION & DEDUPLICATION ---
    combined_matches = exact_matches.copy()
    seen_urls = {doc['url'] for doc in combined_matches}
    for doc in semantic_matches:
        if doc['url'] not in seen_urls:
            combined_matches.append(doc)
            seen_urls.add(doc['url'])

    # 4. Compile context payload
    context = ""
    for doc in combined_matches:
        context += f"{doc['content']} | URL: {doc['url']}\n"

    # --- PROCESS 4: OLLAMA CHAT COMPLETION ---
    system_prompt = (
        "You are an expert Hybrid E-commerce Assistant. Analyze the provided product pool "
        "and answer the user question accurately. Use the exact keyword data if present."
    )
    
    chat_payload = {
        "model": "llama3",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\nQuestion: {user_query}"}
        ],
        "stream": False,
        "options": {"temperature": 0.1}
    }

    try:
        # Increased timeout to 180 seconds to account for slower local disk generation loads
        chat_res = requests.post(OLLAMA_CHAT_URL, json=chat_payload, timeout=None)
        if chat_res.status_code == 200:
            ai_answer = chat_res.json().get('message', {}).get('content', '')
            return {
                "question": user_query,
                "answer": ai_answer,
                "sources_scanned": len(combined_matches)
            }
        else:
            raise HTTPException(status_code=502, detail="Ollama chat completion failed.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM execution error: {str(e)}")
