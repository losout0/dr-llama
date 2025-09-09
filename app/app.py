import streamlit as st
from pathlib import Path
import sys
import os

src_path = str(Path(__file__).resolve().parent.parent)
src_path_2 = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)
if src_path_2 not in sys.path:
    sys.path.append(src_path_2)

try:
    from src.graph import build_graph
except ImportError:
    st.error("Erro ao importar o grafo. Certifique-se que o 'src/graph.py' existe.")
    st.stop()

AVATAR_PATH = "assets/avatar-fav.png"
ASSISTANT_AVATAR = AVATAR_PATH if os.path.exists(AVATAR_PATH) else "⚖️"

# --- Configuração da Página do Streamlit ---
st.set_page_config(
    page_title="Dr. Llama - Assistente Jurídico",
    page_icon=AVATAR_PATH,
    layout="wide"
)

st.title("⚖️ Dr. Llama: Seu Assistente Jurídico (PoC)")
st.markdown("Faça uma pergunta sobre a Legislação Brasileira (Constituição, Código do Consumidor, Código Penal).")

# --- Inicialização do Grafo (em cache) ---
@st.cache_resource
def load_graph():
    """
    Carrega e compila o grafo LangGraph apenas uma vez, usando o cache do Streamlit.
    """
    print("Carregando e compilando o grafo... (isso só deve acontecer uma vez)")
    graph = build_graph()
    print("Grafo carregado com sucesso.")
    return graph

app = load_graph()

# --- Interface de Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostra o histórico de mensagens
for message in st.session_state.messages:
    avatar_to_use = ASSISTANT_AVATAR if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Fontes Utilizadas"):
                for doc in message["sources"]:
                    pretty_name = doc.metadata.get('pretty_name', 'Fonte Desconhecida')
                    page = doc.metadata.get('page', 'N/A')
                    st.info(f"**Fonte:** {pretty_name} (página aprox. {page})\n\n"
                            f"> \"...{doc.page_content[:250]}...\"")

if prompt := st.chat_input("Qual é a sua dúvida?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Analisando documentos e gerando resposta..."):
            
            final_state = app.invoke({"question": prompt})
            
            answer = final_state.get("answer", "Desculpe, ocorreu um erro ao processar sua resposta.")
            documents = final_state.get("documents", [])

            st.markdown(answer)
            
            if documents:
                with st.expander("Fontes Utilizadas para esta resposta"):
                    for doc in documents:
                        pretty_name = doc.metadata.get('pretty_name', 'Fonte Desconhecida')
                        page = doc.metadata.get('page', 'N/A')
                        st.info(f"**Fonte:** {pretty_name} (página aprox. {page + 1})\n\n"
                                f"> \"...{doc.page_content[:250]}...\"")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer, 
                "sources": documents
            })