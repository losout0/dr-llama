from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List
from langchain_core.documents import Document

from ..utils import create_llm

def format_docs_for_answerer(docs: List[Document]) -> str:
    """
    Helper para formatar documentos, usando o 'pretty_name' dos metadados.
    """
    output = []
    for doc in docs:
        # Tenta buscar o 'pretty_name'; se falhar, usa o nome do ficheiro 'source'
        pretty_name = doc.metadata.get('pretty_name', Path(doc.metadata.get('source', 'N/A')).name)
        output.append(
            f"--- Documento Fonte: {pretty_name} ---\n{doc.page_content}"
        )
    return "\n\n".join(output)


def generate_answer(question: str, documents: List[Document]) -> str:
    llm = create_llm()

    prompt_template = """
    Você é um assistente especializado em legislação brasileira. Sua tarefa é responder à pergunta do usuário.

    REGRAS OBRIGATÓRIAS (VOCÊ DEVE SEGUIR):
    1.  BASE EXCLUSIVA: Sua resposta deve ser baseada **EXCLUSIVAMENTE** nos trechos de documentos fornecidos no CONTEXTO.
    2.  NÃO INVENTE: **Não adicione nenhuma informação** que não esteja explicitamente no CONTEXTO.
    3.  CITAÇÃO OBRIGATÓRIA E DETALHADA: Para CADA informação que você fornecer, você **DEVE** citar a fonte. O LLM deve LER o número do artigo no texto do contexto. O formato deve ser [Fonte: Nome do Documento, Art. XX].
    4.  RESPOSTA CURTA: Seja direto e responda apenas ao que foi perguntado.
    5.  SEM CONTEXTO = SEM RESPOSTA: Se a resposta não estiver no CONTEXTO, responda: "Com base nos documentos fornecidos, não encontrei informações sobre este tópico."

    CONTEXTO:
    {context}

    PERGUNTA:
    {question}

    RESPOSTA (Precisa, fiel e com citações detalhadas [Fonte: Nome, Art. XX]):
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    context_string = format_docs_for_answerer(documents)
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "question": question,
        "context": context_string
    })
    
    return response