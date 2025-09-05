from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document

class RetrieverAgent:
    
    def __init__(self):
        # --- CONFIGURAÇÃO ---
        project_root = Path(__file__).parent.parent.parent
        db_faiss_path = str(project_root / "vectorstores" / "db_faiss")
        embedding_model_name = 'thenlper/gte-small'
        # --------------------
        
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'}
        )
        
        self.db = FAISS.load_local(
            db_faiss_path, 
            self.embeddings_model, 
            allow_dangerous_deserialization=True
        )
        
        self.retriever = self.db.as_retriever(search_kwargs={'k': 4})

    def get_relevant_documents(self, question: str) -> List[Document]:
        # Busca e retorna a lista de objetos Document
        documents = self.retriever.invoke(question)
        return documents

# --- Singleton ---    
retriever_agent = RetrieverAgent()
