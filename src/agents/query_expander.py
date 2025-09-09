import sys
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List

src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)
from utils import create_llm

def expand_query(question: str) -> List[str]:
    """
    Pega na pergunta original do utilizador e gera 3 consultas de busca alternativas
    para melhorar a recuperação de documentos, incluindo termos legais relacionados.
    """
    llm = create_llm()

    prompt_template = """
    Você é um assistente especialista em direito brasileiro e um tradutor de "linguagem popular" para "jargão jurídico".
    A sua tarefa é reescrever a pergunta do utilizador para gerar 3 consultas de busca otimizadas para um motor de busca de vetores.
    A sua principal função é traduzir termos populares para os seus equivalentes jurídicos formais que aparecem na Lei.
    Retorne APENAS as 3 consultas, cada uma numa nova linha, sem qualquer outro texto.

    --- EXEMPLO DE COMPORTAMENTO ESPERADO ---
    PERGUNTA ORIGINAL:
    Me fale sobre venda casada

    CONSULTAS DE BUSCA ALTERNATIVAS:
    o que é venda condicionada no código do consumidor
    Me fale sobre venda condicionada
    proibição de venda condicionada lei
    -------------------------------------------

    PERGUNTA ORIGINAL:
    {question}

    CONSULTAS DE BUSCA ALTERNATIVAS:
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Esta cadeia gera uma única string com 3 linhas
    expansion_chain = prompt | llm | StrOutputParser()
    
    result_string = expansion_chain.invoke({"question": question})
    
    
    # Divide a string em uma lista de consultas
    queries = result_string.strip().split('\n')
    cleaned_queries = [q.lstrip("0123456789. \t") for q in queries if q]
    
    # Adiciona a pergunta original à lista também, por segurança
    final_queries = list(set(cleaned_queries))
    
    print(f"--- CONSULTAS EXPANDIDAS: {final_queries} ---")
    return final_queries # Remove duplicados e retorna