#### E-Commerce & Market Intelligence Hybrid RAG System

An end-to-end data extraction and local retrieval-augmented generation (RAG) pipeline built for e-commerce product intelligence. 

The system automates product data collection using **Scrapy**, indexes the structured data into a hybrid search engine (**FAISS vector search + exact keyword matching**), and serves natural-language queries locally via **Ollama (Llama3)** and **FastAPI**—ensuring zero external API costs and complete data privacy. 

#### Architecture Overview

1. **Data Ingestion (Scrapy):** Custom spiders extract, clean, and normalize product titles, specifications, pricing, and availability into structured formats.
2. **Hybrid Retrieval Engine (FAISS + Keyword Search):** Combines dense vector embeddings (nomic-embed-text) for semantic understanding with exact keyword filtering to prevent hallucinations on specific SKUs, model numbers, or prices.
3. **Local LLM Inference (Ollama + Llama3):** Generates grounded answers strictly from retrieved catalog context without sending proprietary data to third-party APIs.
4. **Backend API (FastAPI / Uvicorn):** Exposes clean, asynchronous REST endpoints for querying and data ingestion.
5. **Frontend Interfaces:** Includes both a rapid-prototyping dashboard (**Streamlit**) and a lightweight, dependency-free web interface (**HTML / Vanilla JS**).

#### Tech Stack

* **Scraping & Data Pipelines:** Python, Scrapy
* **AI & Search:** Ollama (Llama3, nomic-embed-text), FAISS, Hybrid Search (Semantic + Lexical)
* **Backend & Databases:** FastAPI, Uvicorn, MySQL / SQLite
* **Frontend:** Streamlit, HTML5, CSS3, Vanilla JavaScript

#### Key Capabilities

* **Anti-Hallucination Grounding:** Hybrid scoring ensures exact product codes and prices are matched accurately alongside natural-language queries.
* **Zero Ongoing Token Costs:** Runs 100% locally or on private infrastructure.
* **Modular Design:** The scraping pipeline and FastAPI layer can be easily connected to existing relational databases (MySQL/PostgreSQL) or legacy PHP backends.

*Available for custom web scraping pipelines, data extraction, and private RAG/backend integrations.*
