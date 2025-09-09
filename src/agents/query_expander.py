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
    Você é um assistente especializado em Direito Brasileiro, com domínio da legislação vigente e da terminologia jurídica formal. Sua função é atuar como tradutor de expressões populares para seus correspondentes técnicos e legais, conforme utilizados em normas, códigos e jurisprudência.
    Sua tarefa consiste em reescrever a pergunta original do usuário — formulada em linguagem informal — para gerar três consultas de busca otimizadas para motores de busca baseados em vetores jurídicos.
    Cada consulta deve:
    - Utilizar vocabulário jurídico preciso
    - Refletir os termos legais presentes na legislação brasileira
    - Ser redigida de forma clara, objetiva e tecnicamente adequada
    Retorne apenas as três consultas, cada uma em uma nova linha, sem qualquer texto adicional.

    EXEMPLO DE COMPORTAMENTO ESPERADO
    PERGUNTA ORIGINAL:
    Me fale sobre venda casada
    CONSULTAS DE BUSCA ALTERNATIVAS:
    o que é venda condicionada no código do consumidor
    Me fale sobre venda condicionada
    proibição de venda condicionada lei

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