# app/app.py

import streamlit as st
from pathlib import Path
import sys
import os

# --- Bloco de Importação Robusto ---
# Garante que a aplicação consegue encontrar o pacote 'src'
try:
    # Abordagem 1: Tenta a importação direta
    from src.graph import build_graph
except ImportError:
    # Abordagem 2: Se falhar, adiciona os paths e tenta de novo
    try:
        src_path = str(Path(__file__).resolve().parent.parent)
        sys.path.append(src_path)
        from src.graph import build_graph
    except ImportError as e:
        st.error(f"Erro Crítico: Não foi possível encontrar o módulo 'src.graph'. Verifique a sua estrutura de pastas e a instalação. Detalhes: {e}")
        st.stop()

# --- Definição de Avatares ---
AVATAR_SUCCESS_PATH = "assets/avatar.png"
AVATAR_FAIL_PATH = "assets/avatar_fail.png"

AVATAR_SUCCESS = AVATAR_SUCCESS_PATH if os.path.exists(AVATAR_SUCCESS_PATH) else "⚖️"
AVATAR_FAIL = AVATAR_FAIL_PATH if os.path.exists(AVATAR_FAIL_PATH) else "⚠️"

# --- Configuração da Página ---
st.set_page_config(
    page_title="Dr. Llama - Assistente Jurídico",
    page_icon=AVATAR_SUCCESS,
    layout="wide"
)

# --- Bloco do Título ---
if os.path.exists(AVATAR_SUCCESS_PATH):
    col1, col2 = st.columns([1, 5], vertical_alignment="center")
    with col1:
        st.image(AVATAR_SUCCESS_PATH, width=90)
    with col2:
        st.title("Dr. Llama: Seu Assistente Jurídico")
        st.markdown("Faça uma pergunta sobre os direitos e deveres do consumidor.")
else:
    st.title("⚖️ Dr. Llama: Seu Assistente Jurídico")
    st.markdown("Faça uma pergunta sobre os direitos e deveres do consumidor.")
st.divider()

# --- Inicialização do Grafo (em cache) ---
@st.cache_resource
def load_graph():
    print("A carregar e a compilar o grafo... (isto só deve acontecer uma vez)")
    graph = build_graph()
    print("Grafo carregado com sucesso.")
    return graph

app = load_graph()

# --- Interface de Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostra o histórico de mensagens
for message in st.session_state.messages:
    avatar_to_use = message.get("avatar", AVATAR_SUCCESS if message["role"] == "assistant" else "👤")
    with st.chat_message(message["role"], avatar=avatar_to_use):
        st.markdown(message["content"])
        # Mostra as fontes se for uma resposta do assistente e elas existirem
        if message.get("sources"):
            with st.expander("Fontes Utilizadas"):
                for doc in message["sources"]:
                    # --- CORREÇÃO APLICADA AQUI (Histórico) ---
                    pretty_name = doc.metadata.get('pretty_name', 'Fonte Desconhecida')
                    article = doc.metadata.get('article', 'Artigo N/A')
                    st.info(f"**Fonte:** {pretty_name}, (Art. {article})\n\n"
                            f"> \"...{doc.page_content[:250]}...\"")

# Input do utilizador
if prompt := st.chat_input("Qual é a sua dúvida?"):
    # Adiciona a mensagem do utilizador e redesenha
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "👤"})
    st.rerun()

# Lógica para processar a última mensagem do utilizador
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_message = st.session_state.messages[-1]
    print(user_message)
    
    with st.chat_message("assistant", avatar=AVATAR_SUCCESS): # Avatar temporário
        with st.spinner("Analisando documentos, gerando resposta e fazendo verificação..."):
            final_state = app.invoke({"question": user_message["content"]})

    answer = final_state.get("answer", "Desculpe, ocorreu um erro.")
    documents = final_state.get("documents", [])
    verdict_obj = final_state.get("verdict")

    # Escolhe o avatar com base no veredito
    if verdict_obj and verdict_obj.verdict == "nao_fiel":
        avatar_to_display = AVATAR_FAIL
    else:
        avatar_to_display = AVATAR_SUCCESS
    
    # Adiciona a resposta do assistente ao estado da sessão
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer, 
        "sources": documents,
        "avatar": avatar_to_display
    })
    
    # Redesenha a página para mostrar a nova resposta do assistente
    st.rerun()