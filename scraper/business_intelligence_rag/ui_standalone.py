#streamlit run ui_standalone.py
import streamlit as st
import requests

st.set_page_config(page_title="Market Intelligence Bot", page_icon="🤖", layout="centered")

st.title("🤖 Market Intelligence Bot")
st.subheader("Interfață de chat conectată la FastAPI RAG")

# Inițializăm istoricul chat-ului în memoria sesiunii Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afișăm mesajele anterioare din istoric
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Căsuța de input pentru utilizator
if user_query := st.chat_input("Pune o întrebare magazinului (ex: prețuri, stocuri)..."):
    
    # 1. Afișăm întrebarea utilizatorului în UI
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # 2. Trimitem cererea către API-ul tău local FastAPI
    with st.chat_message("assistant"):
        with st.spinner("Llama3 analizează catalogul hibrid..."):
            try:
                # Trimitem payload-ul JSON exact cum cere endpoint-ul tău din main.py
                api_url = "http://127.0.0.1:8000/api/chat"
                response = requests.post(api_url, json={"question": user_query}, timeout=300)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "Eroare: Nu am primit un răspuns valid.")
                    sources = data.get("sources_scanned", 0)
                    
                    # Adăugăm un mic badge pentru sursele scanate
                    full_response = f"{answer}\n\n*🔍 [Surse scanate în FAISS + Web2: {sources}]*"
                    st.write(full_response)
                    
                    # Salvăm în istoric
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error(f"Eroare API (Cod {response.status_code})")
            except Exception as e:
                st.error(f"Nu s-a putut conecta la serverul FastAPI: {str(e)}")
