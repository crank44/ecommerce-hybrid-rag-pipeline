#
import json
import requests
import sys
import os

def run_map_reduce_rag(question):
    docs_path = "business_docs.json"
    ollama_chat_url = "http://localhost:11434/api/chat"

    # 1. Safety Check: Verify raw data file exists
    if not os.path.exists(docs_path):
        print(f"[-] Error: Raw data file '{docs_path}' not found! Run Scrapy first.")
        return

    # 2. Load all 1000 raw documents extracted by Scrapy
    with open(docs_path, 'r', encoding='utf-8') as f:
        stored_documents = json.load(f)
    
    print(f"[+] Loaded {len(stored_documents)} products from raw JSON data.")
    
    # 3. Compile all documents into a single text block
    full_text = "\n".join([f"{doc['content']} | URL: {doc['url']}" for doc in stored_documents])
    
    # 4. Split text into chunks of roughly 5,000 characters (~1,200 tokens)
    chunk_size = 3000
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    print(f"[+] Split entire catalog into {len(chunks)} fragments of {chunk_size} characters each.\n")

    partial_summaries = []
    print("[*] Starting Phase 1: MAP (Processing fragments individually)...")
    print("-" * 60)
    
    # 5. MAP PHASE: Query Llama3 for each chunk with an extraction prompt
    for idx, chunk in enumerate(chunks):
        sys.stdout.write(f"\r-> Processing fragment {idx + 1}/{len(chunks)}... ")
        sys.stdout.flush()
        
        system_prompt = (
            "You are a strict Data Extraction Bot. Analyze the provided product list "
            "and extract ONLY relevant data points (such as titles, prices, or counts) "
            "that help answer the final question. Be direct. Do not write intros or prose."
        )
        user_prompt = f"Product List Fragment:\n{chunk}\n\nQuestion to solve: {question}"
        
        payload = {
            "model": "llama3",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {"temperature": 0.0} # Absolute precision for data extraction
        }
        
        try:
            res = requests.post(ollama_chat_url, json=payload, timeout=None)
            if res.status_code == 200:
                answer = res.json().get('message', {}).get('content', '').strip()
                partial_summaries.append(f"Fragment {idx+1} Data:\n{answer}")
            else:
                print(f"\n[-] Fragment {idx+1} failed with status code {res.status_code}")
        except Exception as e:
            print(f"\n[-] Request error on fragment {idx+1}: {str(e)}")
            return

    print("\n\n[*] Starting Phase 2: REDUCE (Aggregating partial data into final answer)...")
    print("-" * 60)
    
    # 6. REDUCE PHASE: Compile the 40 parial responses and send them for final resolution
    compiled_partial_context = "\n\n".join(partial_summaries)
    
    final_system_prompt = (
        "You are an expert E-commerce Business Intelligence Agent. Analyze the compiled "
        "partial data extracted from the full store catalog and provide a final, "
        "comprehensive, and perfectly accurate answer to the user's question."
    )
    final_user_prompt = f"Compiled Data from all fragments:\n{compiled_partial_context}\n\nFinal Question: {question}"
    
    final_payload = {
        "model": "llama3",
        "messages": [
            {"role": "system", "content": final_system_prompt},
            {"role": "user", "content": final_user_prompt}
        ],
        "stream": False,
        "options": {"temperature": 0.2}
    }
    
    try:
        # We give the final prompt a safe timeout window
        print("[+] Asking Llama3 to generate the final analytical response...")
        final_res = requests.post(ollama_chat_url, json=final_payload, timeout=None)
        
        if final_res.status_code == 200:
            final_response = final_res.json().get('message', {}).get('content', '')
            print("\n🤖 MAP-REDUCE GLOBAL AI RESPONSE:")
            print("=" * 60)
            print(final_response)
            print("=" * 60)
        else:
            print(f"[-] Final Reduce phase failed. Code: {final_res.status_code}")
    except Exception as e:
        print(f"[-] Final Execution Error: {str(e)}")

if __name__ == "__main__":
    user_query = input("Ask a global store analysis question (Map-Reduce): ")
    run_map_reduce_rag(user_query)
