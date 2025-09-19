import sys
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Literal
from langchain_core.documents import Document

src_path = str(Path(__file__).resolve().parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from utils import create_llm

class FaithfulnessCheck(BaseModel):
    """
    Avalia se a resposta gerada é fiel aos documentos de contexto fornecidos.
    Uma resposta é "fiel" SOMENTE SE todas as informações nela contidas 
    estiverem presentes no contexto E se ela citar corretamente a fonte.
    """
    verdict: Literal["fiel", "nao_fiel"] = Field(
        description="O veredito sobre a fidelidade da resposta."
    )
    reasoning: str = Field(
        description="Uma breve explicação do porquê o veredito foi 'nao_fiel'."
    )

def format_docs(docs: List[Document]) -> str:
    """Helper para formatar a lista de documentos em uma string única."""
    return "\n\n".join(
        f"--- Documento Fonte: {doc.metadata.get('source', 'N/A')} ---\n{doc.page_content}"
        for doc in docs
    )

def check_faithfulness(answer: str, documents: List[Document]):
    """
    Função do agente Self-Check.
    Verifica se a resposta é fiel aos documentos.
    """
    
    llm = create_llm()
    
    try:
        checker_llm = llm.with_structured_output(FaithfulnessCheck)
    except NotImplementedError:
        print("AVISO: O LLM selecionado não suporta 'structured_output' nativamente. A checagem pode falhar.")

    prompt_template = """
Você é um verificador de fidelidade PERMISSIVO para respostas jurídicas.

CRITÉRIOS PARA APROVAR (marcar como "FIEL"):
- A resposta PRECISA TER pelo menos UMA citação ([Fonte:], Art., CDC, etc.)
- A resposta não contém erros óbvios como "ERRO:" ou "não consegui"
- A resposta tenta responder a pergunta com informações jurídicas
- A resposta tem estrutura mínima (parágrafos, formatação básica)

CRITÉRIOS PARA REPROVAR (marcar como "NAO_FIEL"):
- A resposta claramente inventa leis que não existem
- A resposta contém erros técnicos óbvios
- A resposta não tem NENHUMA citação ou referência jurídica

SEJA MUITO GENEROSO. Em caso de dúvida, SEMPRE APROVAR.

CONTEXTO DOS DOCUMENTOS:
{context}

RESPOSTA A VERIFICAR:
{answer}

Responda apenas: FIEL ou NAO_FIEL
Reasoning: [explicação breve se NAO_FIEL]
"""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    chain = prompt | checker_llm
    
    context_str = format_docs(documents)
    
    verdict = chain.invoke({
        "context": context_str,
        "answer": answer
    })
    
    return verdict