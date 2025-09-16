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
    Você é um especialista em Direito do Consumidor Brasileiro. Sua função é transformar situações práticas do cotidiano em consultas jurídicas precisas para busca em legislação.
    
    CONTEXTO: O usuário descreverá uma situação real. Você deve:
    1. Identificar os CONCEITOS JURÍDICOS envolvidos
    2. Mapear para os ARTIGOS/INSTITUTOS LEGAIS correspondentes  
    3. Gerar APENAS 3 consultas otimizadas para recuperação de documentos
    
    TÉCNICA DE EXPANSÃO:
    - Consulta 1: Situação direta com termos técnicos
    - Consulta 2: Instituto/conceito jurídico principal
    - Consulta 3: Direitos e obrigações específicos
    
    EXEMPLO PRÁTICO:
    SITUAÇÃO: "Vi uma placa de um produto com um preço mas na prateleira está com um preço diferente. Qual dos valores eu devo considerar?"
    
    CONCEITOS IDENTIFICADOS: Publicidade enganosa, oferta vinculante, preço anunciado
    ARTIGOS RELEVANTES: Art. 30, 35, 37 CDC
    
    CONSULTAS GERADAS:
    divergência preço anunciado e cobrado código consumidor
    oferta vinculante publicidade preços CDC artigo 30
    direito consumidor preço menor anunciado estabelecimento
    
    SITUAÇÃO DO USUÁRIO:
    {question}
    
    CONSULTAS DE BUSCA (3 linhas, sem numeração):
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