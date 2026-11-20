import faiss
import numpy as np
import json
import os
import requests

class BusinessRagPipeline:
    def __init__(self):
        self.dimension = 768
        self.index = faiss.IndexFlatL2(self.dimension)
        self.stored_documents = []
        
        self.faiss_path = "business_faiss.index"
        self.docs_path = "business_docs.json"
        
        # URL-ul local nativ al serverului Ollama
        self.ollama_url = "http://localhost:11434/api/embeddings"

    def process_item(self, item, spider):
        text_content = item['content']
        
        if text_content:
            try:
                # Folosim un request HTTP nativ, imun la problemele de asyncio din Python 3.13
                payload = {
                    "model": "nomic-embed-text",
                    "prompt": text_content
                }
                response = requests.post(self.ollama_url, json=payload, timeout=10)
                
                if response.status_code == 200:
                    embedding = response.json().get('embedding')
                    
                    if embedding:
                        vector = np.array([embedding]).astype('float32')
                        self.index.add(vector)
                        
                        self.stored_documents.append({
                            "title": item['title'],
                            "url": item['source_url'],
                            "content": text_content,
                            "date": item['date_extracted']
                        })
                        spider.logger.info(f"==> VECTORIZAT CU SUCCES: {item['title']} (Total în RAM: {self.index.ntotal})")
                else:
                    spider.logger.error(f"Ollama server a raspuns cu eroarea: {response.status_code}")
                    
            except Exception as e:
                spider.logger.error(f"Eroare la generarea embedding-ului: {str(e)}")
                
        return item

    def close_spider(self, spider):
        spider.logger.info(f"=== INCHIDERE SPIDER. Incerc salvare pe disc... Total vectori acumulati: {self.index.ntotal} ===")
        
        # Forțăm scrierea pe disc chiar dacă, dintr-un motiv anume, indexul a rămas gol, ca să verificăm dacă se apelează funcția
        try:
            faiss.write_index(self.index, self.faiss_path)
            with open(self.docs_path, 'w', encoding='utf-8') as f:
                json.dump(self.stored_documents, f, ensure_ascii=False, indent=4)
            spider.logger.info("=== FIȘIERELE AU FOST SCRISE CU SUCCES PE DISC! ===")
        except Exception as e:
            spider.logger.error(f"Eroare critica la scrierea pe disc: {str(e)}")
