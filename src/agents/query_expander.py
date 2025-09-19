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
Você é um gerador de consultas para busca densa em legislação brasileira (CDC).
Regras obrigatórias:
- Retorne exatamente 3 consultas curtas (3–6 palavras), uma por linha.
- Sem numeração, sem aspas, sem asteriscos, sem cabeçalhos, sem explicações.
- Use minúsculas e termos jurídicos precisos (ex.: prática abusiva, oferta vinculante, art. 39 cdc).
- Não repita consultas nem varie apenas por plurais/sinais.
- Saída deve conter apenas as 3 linhas de consultas.

Pergunta:
{question}

Saída:
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