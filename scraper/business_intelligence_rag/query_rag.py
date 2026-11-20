import faiss
import numpy as np
import json
import requests

def query_business_intelligence(question):
    faiss_path = "business_faiss.index"
    docs_path = "business_docs.json"
    ollama_emb_url = "http://localhost:11434/api/embeddings"
    ollama_chat_url = "http://localhost:11434/api/chat"

    # 1. Safety Check: Ensure data index files exist
    if not (faiss.os.path.exists(faiss_path) and faiss.os.path.exists(docs_path)):
        print("[-] Error: Local Vector Database not found! Run your Scrapy spider first.")
        return

    # 2. Load the FAISS Index and text documents from disk
    index = faiss.read_index(faiss_path)
    with open(docs_path, 'r', encoding='utf-8') as f:
        stored_documents = json.load(f)

    print(f"\n[+] Loaded database with {index.ntotal} commercial documents.")
    print(f"[+] Embedding user question: '{question}'...")

    try:
        # 3. Vectorize the user question using the same embedding model
        payload = {"model": "nomic-embed-text", "prompt": question}
        emb_res = requests.post(emb_emb_url := ollama_emb_url, json=payload, timeout=10)
        
        if emb_res.status_code != 200:
            print("[-] Error communicating with local Ollama server for embeddings.")
            return
            
        question_vector = np.array([emb_res.json().get('embedding')]).astype('float32')

        # 4. Perform vector search in FAISS (Retrieve Top 2 closest matches)
        k = k = min(15, index.ntotal) #index.ntotal  #2
        distances, indices = index.search(question_vector, k)

        # 5. Build the Context block from matching articles
        # context = ""
        # print(f"\n[+] Processing ALL {index.ntotal} products from vector index...")
        
        # # Folosim indices[0] pentru a parcurge corect toți vectorii returnați de FAISS
        # for idx in indices[0]:  
            # if idx < len(stored_documents) and idx != -1:
                # doc = stored_documents[idx]
                
                # # Salvăm conținutul curat (care conține deja Titlul, Prețul și Stocul liniar)
                # context += f"{doc['content']} | URL: {doc['url']}\n"
        
        # print(f"[+] Context compiled successfully ({len(context)} characters sent to AI).")
        # print("=========================================================\n")
        context = ""
        print("\n=== TOP ARTICLES RETRIEVED ===")
        print(f"\n[+] Processing ALL {index.ntotal} products from vector index...")
        for idx in indices[0]:
            if idx < len(stored_documents) and idx != -1:
                doc = stored_documents[idx]
                print(f" -> MATCHED: {doc['title']} ({doc['url']})")
                #context += f"Source: {doc['title']}\nContent: {doc['content']}\n\n"
                context += f"{doc['content']} | URL: {doc['url']}\n"
        
        print(f"[+] Context compiled successfully ({len(context)} characters sent to AI).")
        print("=========================================================\n")
        print("==============================\n")
        

        # 6. Construct the Advanced prompt for the LLM
        # system_prompt = (
            # "You are an expert Business Intelligence Agent. Analyze the provided news articles "
            # "and answer the user question accurately based ONLY on the context provided. Keep it concise."
        # )
        system_prompt = (
            "You are an expert E-commerce Assistant. Analyze the provided product data "
            "and answer the user question accurately based ONLY on the context provided. "
            "Include prices and availability in your answer. Keep it concise."
        )
        user_prompt = f"Context:\n{context}\nQuestion: {question}"

        # 7. Query Llama3 locally to generate the final analytical response
        print("[+] Asking Llama3 to generate business insights...")
        chat_payload = {
            "model": "llama3", # Change to "mistral" if your VRAM is limited
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
               "temperature": 0.2  # 0 strcit - 1 creativ
            }
        }
        
        chat_res = requests.post(ollama_chat_url, json=chat_payload, timeout=None)
        
        if chat_res.status_code == 200:
            ai_response = chat_res.json().get('message', {}).get('content', '')
            print("\n🤖 BUSINESS AI ANALYSIS RESPONSE:")
            print("-" * 50)
            print(ai_response)
            print("-" * 50)
        else:
            print(f"[-] Llama3 model failed to respond. Code: {chat_res.status_code}")

    except Exception as e:
        print(f"[-] Execution Error: {str(e)}")

if __name__ == "__main__":
    # Prompt the user directly in the terminal
    user_query = input("Ask your local Market Intelligence bot a question: ")
    query_business_intelligence(user_query)
