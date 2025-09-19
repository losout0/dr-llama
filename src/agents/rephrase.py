# src/agents/rephrase.py

from pathlib import Path
import sys
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

src_path = str(Path(__file__).resolve().parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from utils import create_llm

class SimpleRephraser:
    """
    Reescreve a pergunta em português jurídico claro, em 1 linha curta.
    Rápido: baixa temperatura e poucos tokens.
    """

    def __init__(self):
        self.llm = create_llm()
        # Tornar a geração mais rápida e estável
        
        self.prompt = ChatPromptTemplate.from_template(
            """
Reescreva a pergunta abaixo em português jurídico claro e objetivo, mantendo o mesmo sentido.
Regras:
- 1 frase, até 25 palavras.
- Sem listas, sem explicações, sem aspas.
Pergunta: {question}
Saída:
""".strip()
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def rephrase(self, question: str) -> str:
        text = self.chain.invoke({"question": question}).strip()
        # Pega a primeira linha e higieniza
        line = text.splitlines()[0].strip().strip('"').strip("'")
        if not line.endswith("?"):
            line += "?"
        return line

# Instância singleton compatível com import existente
rephrase_agent = SimpleRephraser()
