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
        
        self.retriever = self.db.as_retriever(search_kwargs={'k': 2})

    def get_relevant_documents(self, queries: List[str]) -> List[Document]:
        """
        Busca documentos para uma LISTA de consultas, junta os resultados e remove duplicados.
        Esta operação é rápida, pois os modelos já estão carregados.
        """
        all_doc_lists = self.retriever.batch(queries)
        
        final_docs_map = {}
        for doc_list in all_doc_lists:
            for doc in doc_list:
                if doc.page_content not in final_docs_map:
                    final_docs_map[doc.page_content] = doc
        
        return list(final_docs_map.values())

# --- Singleton ---    
retriever_agent = RetrieverAgent()
