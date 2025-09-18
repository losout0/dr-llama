# agents/rephrase_agent.py
import sys
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List
from langchain_core.documents import Document

src_path = str(Path(__file__).resolve().parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from utils import create_llm

class RephraseSuggestionAgent:
    """
    Agente inteligente que gera sugestões de reformulação baseadas no contexto
    da pergunta original, documentos recuperados e motivo da falha.
    """
    
    def __init__(self):
        self.llm = create_llm()
        
    def generate_suggestions(self, question: str, documents: List[Document], 
                           verdict_reason: str, intent: str) -> List[str]:
        """Gera sugestões inteligentes de reformulação"""
        
        # Extrair contexto dos documentos
        doc_context = self._extract_document_context(documents)
        
        prompt_template = """
Você é um especialista em direito do consumidor brasileiro que ajuda usuários a reformular perguntas jurídicas.

SITUAÇÃO:
- O usuário fez uma pergunta, mas a resposta gerada não foi fiel aos documentos
- Você precisa sugerir 3-5 reformulações que tenham maior chance de sucesso

PERGUNTA ORIGINAL: {question}

CONTEXTO DOS DOCUMENTOS ENCONTRADOS:
{doc_context}

MOTIVO DA FALHA: {verdict_reason}

INTENÇÃO DETECTADA: {intent}

INSTRUÇÕES:
1. Analise por que a pergunta falhou (terminologia inadequada, muito vaga, etc.)
2. Gere 3-5 reformulações usando:
   - Terminologia jurídica mais precisa
   - Termos que aparecem nos documentos recuperados
   - Especificidade maior se a pergunta era vaga
   - Sinônimos jurídicos apropriados

3. Retorne APENAS as sugestões, uma por linha, sem numeração ou explicações

REFORMULAÇÕES SUGERIDAS:
"""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            result = chain.invoke({
                "question": question,
                "doc_context": doc_context,
                "verdict_reason": verdict_reason,
                "intent": intent
            })
            
            # Processar resposta
            suggestions = [line.strip() for line in result.strip().split('\n') 
                          if line.strip() and not line.startswith('-')]
            
            return suggestions[:5]  # Máximo 5 sugestões
            
        except Exception as e:
            print(f"Erro no RephraseSuggestionAgent: {e}")
            return self._fallback_suggestions(question)
    
    def _extract_document_context(self, documents: List[Document]) -> str:
        """Extrai contexto relevante dos documentos"""
        if not documents:
            return "Nenhum documento específico foi recuperado"
        
        context_parts = []
        seen_articles = set()
        
        for doc in documents[:5]:  # Primeiros 5 documentos
            # Extrair artigos mencionados
            article = doc.metadata.get('article', '')
            if article and article != 'Não especificado' and article not in seen_articles:
                seen_articles.add(article)
                
            # Extrair palavras-chave do conteúdo
            content_preview = doc.page_content[:200].lower()
            doc_name = doc.metadata.get('pretty_name', 'Documento')
            
            context_parts.append(f"- {doc_name}: {article} - {content_preview}...")
        
        if seen_articles:
            context_parts.insert(0, f"Artigos encontrados: {', '.join(list(seen_articles)[:5])}")
        
        return '\n'.join(context_parts[:10])  # Limitar contexto
    
    def _fallback_suggestions(self, question: str) -> List[str]:
        """Sugestões de fallback simples se a LLM falhar"""
        q_lower = question.lower()
        
        if "venda casada" in q_lower:
            return [
                "O que é venda condicionada?",
                "Práticas abusivas no código do consumidor",
                "Artigo 39 do CDC sobre venda casada"
            ]
        elif "propaganda" in q_lower or "publicidade" in q_lower:
            return [
                "O que é publicidade enganosa?",
                "Artigo 37 do código do consumidor",
                "Tipos de propaganda enganosa no CDC"
            ]
        else:
            return [
                "Reformule usando termos mais específicos",
                "Tente usar vocabulário jurídico",
                "Seja mais específico sobre o contexto"
            ]

# Instância singleton
rephrase_agent = RephraseSuggestionAgent()
