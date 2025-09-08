from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import List
from langchain_core.documents import Document

from ..utils import create_llm

def generate_answer(question: str, documents: List[Document]) -> str:
    llm = create_llm()

    prompt_template = """
    Você é um assistente especializado em legislação brasileira. Sua tarefa é responder à pergunta do usuário com base exclusivamente nos trechos de documentos fornecidos no contexto.

    Instruções:
    1.  Responda de forma clara e objetiva.
    2.  Ao usar uma informação de um documento, **cite a fonte** (o nome do arquivo) e, se possível, o artigo ou seção relevante.
    3.  Se a informação não estiver no contexto, responda: "Com base nos documentos fornecidos, não encontrei informações sobre este tópico."

    CONTEXTO:
    {context}

    PERGUNTA:
    {input}

    RESPOSTA INFORMATIVA E COM CITAÇÕES:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Cria a cadeia que sabe como formatar os documentos no prompt
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # Invoca a cadeia passando os documentos e a pergunta
    response = question_answer_chain.invoke({
        "input": question,
        "context": documents
    })
    
    return response