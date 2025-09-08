import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

def create_llm() -> BaseChatModel:
    """
    Fábrica de LLMs.
    Lê a variável de ambiente LLM_PROVIDER para decidir qual LLM instanciar.
    Retorna uma instância de um modelo de chat (Ollama ou Gemini).
    """
    project_root = Path(__file__).parent.parent.parent
    dotenv_path = project_root / "config" / ".env"
    
    if not dotenv_path.exists():
        print(f"Aviso: arquivo .env não encontrado em {dotenv_path}. Usando variáveis de ambiente globais.")
    
    load_dotenv(dotenv_path=dotenv_path)
    
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    model=os.getenv("LLM_MODEL", "llama3.2:1b" ).lower()
    
    print(f"--- Utilizando o provedor de LLM: {provider} | Modelo: {model} ---")
    
    if provider == "gemini":
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("A chave GOOGLE_API_KEY não foi encontrada no arquivo .env")
        
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=google_api_key,
            temperature=0
        )
        
    elif provider == "ollama":
        return ChatOllama(model=model, temperature=0) 
    
    else:
        raise ValueError(
            f"Provedor de LLM '{provider}' não suportado. "
            "Use 'ollama' ou 'gemini'."
        )