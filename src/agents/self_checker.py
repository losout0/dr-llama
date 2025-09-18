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
Sua tarefa é verificar se a resposta é fiel aos documentos, com tolerância para variações normais de formatação e sinônimos.

CRITÉRIOS PARA "FIEL":
1. **FIDELIDADE SEMÂNTICA**: O SIGNIFICADO da informação está nos documentos, mesmo que com redação ligeiramente diferente
2. **CITAÇÕES VÁLIDAS**: Se menciona "Art. 37" e há "Art. 37º" ou "Artigo 37" no documento = VÁLIDO
3. **CORREÇÕES PERMITIDAS**: Pode corrigir hifenização, pontuação, formatação do PDF
4. **SINÔNIMOS ACEITOS**: "publicidade enganosa" = "propaganda enganosa" = VÁLIDO

CRITÉRIOS PARA "NAO_FIEL":
- Inventa informações factuais que NÃO existem nos documentos
- Cita artigos que realmente não existem (ex: Art. 999 do CDC)
- Contradiz explicitamente o texto dos documentos

SEJA FLEXÍVEL com formatação e sinônimos. SEJA RIGOROSO apenas com fatos inventados.

DOCUMENTOS_DE_EVIDENCIA:
{context}

RESPOSTA_GERADA:
{answer}

Analise com TOLERÂNCIA SEMÂNTICA e forneça seu veredito:
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    chain = prompt | checker_llm
    
    context_str = format_docs(documents)
    
    verdict = chain.invoke({
        "context": context_str,
        "answer": answer
    })
    
    return verdict