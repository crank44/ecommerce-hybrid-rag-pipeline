import faiss
import numpy as np
import json
import requests
import os

def hybrid_search_rag(question, keyword_triggers=None):
    faiss_path = "business_faiss.index"
    docs_path = "business_docs.json"
    ollama_emb_url = "http://localhost:11434/api/embeddings"
    ollama_chat_url = "http://localhost:11434/api/chat"

    # 1. Verificare fișiere pe disc
    if not (os.path.exists(faiss_path) and os.path.exists(docs_path)):
        print("[-] Error: Index files not found! Run Scrapy first.")
        return

    # 2. Încărcare date
    index = faiss.read_index(faiss_path)
    with open(docs_path, 'r', encoding='utf-8') as f:
        stored_documents = json.load(f)

    print(f"[+] Loaded database with {index.ntotal} products.")
    
    # --- PROCESUL 1: CĂUTARE EXACTĂ (KEYWORD / WEB2) ---
    exact_matches = []  # CORECTAT: Inițializată ca listă goală
    if keyword_triggers:
        print(f"[*] Running Exact Keyword Search for: {keyword_triggers}...")
        for doc in stored_documents:
            if any(trigger.lower() in doc['content'].lower() for trigger in keyword_triggers):
                exact_matches.append(doc)
    
    print(f"[+] Keyword search found {len(exact_matches)} exact matches.")

    # --- PROCESUL 2: CĂUTARE SEMANTICĂ (VECTOR / FAISS) ---
    print(f"[*] Running Semantic Vector Search for: '{question}'...")
    payload = {"model": "nomic-embed-text", "prompt": question}
    emb_res = requests.post(ollama_emb_url, json=payload, timeout=10)
    
    if emb_res.status_code != 200:
        print("[-] Error generating embeddings.")
        return
        
    question_vector = np.array([emb_res.json().get('embedding')]).astype('float32')
    
    k = min(5, index.ntotal)
    distances, indices = index.search(question_vector, k)
    
    semantic_matches = []
    # Corectat parcurgerea pentru matricea 2D returnată de FAISS
    for row in indices:
        for idx in row:
            if idx < len(stored_documents) and idx != -1:
                semantic_matches.append(stored_documents[idx])

    # --- PROCESUL 3: FUZIUNEA HYBRIDĂ & ELIMINAREA DUPLICATELOR ---
    combined_matches = exact_matches.copy()
    seen_urls = {doc['url'] for doc in combined_matches}
    
    for doc in semantic_matches:
        if doc['url'] not in seen_urls:
            combined_matches.append(doc)
            seen_urls.add(doc['url'])

    # --- PROCESUL 4: COMPILAREA CONTEXTULUI PENTRU LLM ---
    context = ""
    print("\n=== HYBRID SEARCH RESULTS POOL ===")
    for doc in combined_matches:
        print(f" -> POOL ITEM: {doc['title']}")
        context += f"{doc['content']} | URL: {doc['url']}\n"
    print("==================================\n")

    # 5. Interogare Llama3 cu contextul hibrid optimizat
    system_prompt = (
        "You are an expert Hybrid E-commerce Assistant. Analyze the provided product pool "
        "and answer the user question accurately. Use the exact keyword data if present."
    )
    user_prompt = f"Context:\n{context}\nQuestion: {question}"

    chat_payload = {
        "model": "llama3",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {"temperature": 0.1}
    }
    
    print("[+] Asking Llama3 to analyze the hybrid context...")
    try:
        chat_res = requests.post(ollama_chat_url, json=chat_payload, timeout=180)
        if chat_res.status_code == 200:
            print("\n🤖 HYBRID AI RESPONSE:")
            print("-" * 60)
            print(chat_res.json().get('message', {}).get('content', ''))
            print("-" * 60)
    except Exception as e:
        print(f"[-] Error: {str(e)}")

if __name__ == "__main__":
    user_query = input("Ask your hybrid bot a question: ")
    words = user_query.split()
    triggers = [w.strip("?,.!") for w in words if any(c.isdigit() for c in w) or len(w) > 6]
    
    hybrid_search_rag(user_query, keyword_triggers=triggers)
