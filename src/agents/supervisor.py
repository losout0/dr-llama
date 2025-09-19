import sys
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, List
import re

src_path = str(Path(__file__).resolve().parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from utils import create_llm

class SupervisorAgent:
    def __init__(self):
        self.llm = create_llm()
        
    def supervise(self, question: str) -> Dict:
        """Análise híbrida: heurísticas + LLM"""
        
        # PASSO 1: Verificação determinística rápida
        deterministic_result = self._deterministic_check(question)
        
        if deterministic_result is not None:
            # Se as heurísticas são conclusivas, usa elas
            intent = self._classify_intent_simple(question)
            
            return {
                "intent": intent,
                "needs_clarification": deterministic_result,
                "confidence": "alta" if not deterministic_result else "baixa",
                "method": "deterministic"
            }
        
        # PASSO 2: Se heurísticas não são conclusivas, usa LLM
        return self._llm_analysis(question)
    
    def _deterministic_check(self, question: str) -> bool | None:
        """
        Regras determinísticas para casos óbvios.
        Retorna:
        - False: NÃO precisa esclarecimento (pergunta clara)
        - True: precisa esclarecimento (pergunta vaga)
        - None: não consegue decidir (passa para LLM)
        """
        q_lower = question.lower()

        # CASOS QUE NÃO PRECISAM ESCLARECIMENTO (claros)
        clear_indicators = [
            # ✅ ADICIONADO: Casos conceituais/educativos
            (r'o que é.*\?', []),
            (r'me fale sobre.*\?', []),
            (r'defin.*\?', []),
            (r'como funciona.*\?', []),
            (r'.*é.*prática abusiva\?', []),
            (r'.*é.*permitido\?', []),
            (r'.*propaganda enganosa.*\?', []),
            (r'.*venda casada.*\?', []),
            (r'.*dois preços.*\?', []),
            (r'quais.*direitos.*consumidor\?', ['básicos', 'fundamentais', 'principais']),

            # Casos situacionais específicos (já existentes)
            (r'preço.*diferente|diferente.*preço', ['placa', 'caixa', 'r\\$', 'real']),
            (r'placa.*r\\$\\d+.*caixa.*r\\$\\d+', []),
            (r'anunciado.*r\\$\\d+.*cobr.*r\\$\\d+', []),

            # Casos de produto defeituoso
            (r'produto.*defeituoso.*dias', ['comprei', 'prazo']),
            (r'comprei.*defeito.*trocar', []),

            # Casos de promoção/oferta
            (r'promoção.*recusa|oferta.*descumprir', ['loja', 'estabelecimento']),
        ]

        for pattern, additional_terms in clear_indicators:
            if re.search(pattern, q_lower):
                # Se tem o padrão principal + termos adicionais (ou sem termos) = caso claro
                if not additional_terms or any(term in q_lower for term in additional_terms):
                    return False  # NÃO precisa esclarecimento

        # CASOS QUE PRECISAM ESCLARECIMENTO (vagos)
        vague_patterns = [
            r'^posso processar\?*$',
            r'^tenho direito\?*$',
            r'^o que fazer\?*$',
            r'^é legal\?*$',
            r'^quais são meus direitos\?*$',  # ✅ ADICIONADO: Sem contexto específico
            r'^isso pode dar problema\?*$',   # ✅ ADICIONADO: Pergunta muito vaga
        ]

        for pattern in vague_patterns:
            if re.search(pattern, q_lower.strip()):
                return True  # precisa esclarecimento

        # Se é muito curto e sem detalhes = vago
        if len(question.strip()) < 15 and '?' in question:  # ✅ AJUSTADO: de 20 para 15
            return True

        # Não consegue decidir = passa para LLM
        return None

    def _classify_intent_simple(self, question: str) -> str:
        """Classificação simples baseada em palavras-chave"""
        q_lower = question.lower()

        consumer_keywords = ['preço', 'produto', 'loja', 'compra', 'venda', 'defeito', 'promoção', 'mercado']
        constitutional_keywords = ['direito fundamental', 'constituição', 'liberdade', 'igualdade']
        
        if any(keyword in q_lower for keyword in consumer_keywords):
            return "consumidor"
        elif any(keyword in q_lower for keyword in constitutional_keywords):
            return "constitucional"
        else:
            return "consumidor"  # default
    
    def _llm_analysis(self, question: str) -> Dict:
        """Análise via LLM para casos não-determinísticos"""
        
        # Prompt MUITO mais simples e direto
        prompt_template = """
Pergunta: "{question}"

Esta pergunta tem FATOS SUFICIENTES para uma resposta jurídica?

Responda apenas: SIM ou NAO

Resposta:"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            result = chain.invoke({"question": question}).strip().upper()
            needs_clarification = "NAO" in result or "NÃO" in result
            
            return {
                "intent": "consumidor",
                "needs_clarification": needs_clarification,
                "expanded_queries": [question],
                "confidence": "media",
                "method": "llm"
            }
        except:
            # Fallback: assume que não precisa esclarecimento
            return {
                "intent": "consumidor", 
                "needs_clarification": False,
                "expanded_queries": [question],
                "confidence": "baixa",
                "method": "fallback"
            }

# Instância singleton
supervisor_agent = SupervisorAgent()

def supervise_question(question: str) -> Dict:
    return supervisor_agent.supervise(question)