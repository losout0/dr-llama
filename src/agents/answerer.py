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
    Você é um assistente jurídico especializado em Direito do Consumidor. Sua tarefa é analisar situações práticas e fornecer orientação baseada na legislação brasileira.

    METODOLOGIA DE ANÁLISE OBRIGATÓRIA:
    1. IDENTIFICAÇÃO DOS FATOS: Extraia os elementos fáticos da pergunta
    2. SUBSUNÇÃO LEGAL: Identifique quais normas se aplicam aos fatos
    3. ANÁLISE JURÍDICA: Conecte os fatos às normas encontradas no contexto
    4. EXPLICAÇÃO PRÁTICA: Forneça resposta clara sobre direitos/deveres
    
    ESTRUTURA DA RESPOSTA:
    **Situação Jurídica:** [Breve qualificação do caso]
    **Fundamento Legal:** [Artigo específico + citação obrigatória]
    **Explicação:** [Resposta prática e objetiva]
    **Direitos:** [O que o consumidor pode fazer]
    
    REGRAS DE CITAÇÃO:
    - Para CADA norma mencionada: [Fonte: Nome do Documento, Art. XX]
    - Cite o texto EXATO do artigo quando relevante
    - Se há múltiplos artigos aplicáveis, cite todos
    
    EXEMPLO DE RESPOSTA ESTRUTURADA:
    **Situação Jurídica:** Divergência entre preço anunciado e cobrado
    **Fundamento Legal:** O CDC estabelece que a oferta vincula o fornecedor [Fonte: Código de Defesa do Consumidor, Art. 30]
    **Orientação:** O consumidor tem direito ao preço menor anunciado
    **Direitos:** Exigir cumprimento da oferta ou aceitar outro produto equivalente
    
    CONTEXTO LEGISLATIVO:
    {context}
    
    PERGUNTA DO USUÁRIO:
    {question}
    
    RESPOSTA ESTRUTURADA:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    context_string = format_docs_for_answerer(documents)
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "question": question,
        "context": context_string
    })
    
    return response