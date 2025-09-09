from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Literal
from langchain_core.documents import Document

from ..utils import create_llm

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
    Sua tarefa é atuar como um auditor rigoroso, mas com bom senso. Você deve comparar a RESPOSTA_GERADA com os DOCUMENTOS_DE_EVIDENCIA e determinar se a resposta é 100% fiel às evidências.

    Uma resposta é considerada "fiel" SE E SOMENTE SE:
    1.  Todas as afirmações factuais na resposta estão DIRETAMENTE E SEMANTICAMENTE suportadas por trechos nos documentos de evidência.
    2.  A resposta NÃO contém nenhuma informação factual que não esteja nos documentos.
    3.  A resposta CITA a fonte (o nome do arquivo) de onde a informação foi retirada.

    Uma resposta é "nao_fiel" se:
    1.  Ela inventa qualquer informação factual (alucinação).
    2.  Ela esquece de citar a fonte.
    3.  Ela faz afirmações que contradizem as evidências.

    --- REGRAS DE FLEXIBILIDADE (IMPORTANTE) ---
    -   **PERMITIDO:** É permitido e correto que a resposta corrija erros óbvios de formatação do documento de origem, tais como desifenização.
    -   **EXEMPLO DE CORREÇÃO PERMITIDA:** Se o documento de evidência mostrar "serviço sem a prévia elabo -", uma resposta que diga "serviço sem a prévia elaboração" é considerada FIEL, pois está a corrigir um artefacto de formatação do PDF.
    -   A verificação deve ser sobre a **FIDELIDADE SEMÂNTICA (significado)**, não sobre uma correspondência literal de caracteres.

    DOCUMENTOS_DE_EVIDENCIA:
    {context}

    RESPOSTA_GERADA:
    {answer}

    Analise a resposta gerada com base em TODAS as regras acima e forneça seu veredito e o motivo.
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    chain = prompt | checker_llm
    
    context_str = format_docs(documents)
    
    verdict = chain.invoke({
        "context": context_str,
        "answer": answer
    })
    
    return verdict