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
    Você é um revisor jurídico especializado. Sua função é verificar se a orientação jurídica está 100% fundamentada na legislação fornecida.

    CRITÉRIOS DE FIDELIDADE JURÍDICA:
    
    ✅ RESPOSTA FIEL quando:
    1. Cada afirmação jurídica tem base textual nos documentos
    2. Os artigos citados existem e dizem exatamente o que foi afirmado
    3. A interpretação jurídica está alinhada com o texto legal
    4. Não há invenção de direitos ou obrigações não previstos
    
    ❌ RESPOSTA NÃO FIEL quando:
    1. Cita artigos inexistentes ou com conteúdo diferente
    2. Inventa interpretações não baseadas no texto legal
    3. Omite exceções importantes presentes na lei
    4. Faz afirmações jurídicas sem base documental
    
    ATENÇÃO ESPECIAL PARA:
    - Numeração correta de artigos, incisos e parágrafos
    - Condições e exceções previstas na lei
    - Prazos e procedimentos específicos
    - Diferença entre direitos e meras faculdades
    
    VERIFICAÇÃO DE QUALIDADE JURÍDICA:
    - A resposta conecta corretamente fatos → norma → consequência?
    - Os artigos citados realmente regulam a situação apresentada?
    - Há contradições entre a resposta e o texto legal?
    
    DOCUMENTOS DE REFERÊNCIA:
    {context}
    
    RESPOSTA A VERIFICAR:
    {answer}
    
    Analise com rigor jurídico e forneça seu veredito:
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    chain = prompt | checker_llm
    
    context_str = format_docs(documents)
    
    verdict = chain.invoke({
        "context": context_str,
        "answer": answer
    })
    
    return verdict