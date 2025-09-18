print("Iniciando: importando as bibliotecas e instanciando os agentes...")

from typing import List, TypedDict, Literal
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from agents import retriever_agent
from agents import generate_answer
from agents import check_faithfulness, FaithfulnessCheck
from agents import expand_query
from agents import apply_disclaimer
from agents import supervisor_agent, supervise_question
from agents import rephrase_agent

# --- Definição do Estado do Grafo ---

class GraphState(TypedDict):
    question: str
    intent: str
    needs_clarification: bool
    expanded_queries: List[str] 
    confidence: str
    documents: List[Document]
    answer: str
    verdict: FaithfulnessCheck

# --- NÓS DO GRAFO ---
def supervisor_node(state: GraphState):
    """Nó supervisor que classifica e decide próximos passos"""
    print(" --- EXECUTANDO NÓ: SUPERVISOR ---")
    question = state['question']
    
    supervision_result = supervise_question(question)
    
    return {
        "intent": supervision_result["intent"],
        "needs_clarification": supervision_result["needs_clarification"],
        "expanded_queries": supervision_result["expanded_queries"],
        "confidence": supervision_result["confidence"]
    }

def query_expander_node(state: GraphState):
    "Nó que executa o agente Query Expander"
    print(" --- EXECUTANDO NÓ: QUERY EXPANDER ---")
    question = state['question']
    question = expand_query(question)
    return {"question": question}

def retrieve_node(state: GraphState):
    """Nó que executa o agente Retriever."""
    print("--- EXECUTANDO NÓ: RETRIEVER ---")
    question = state["question"]
    documents = retriever_agent.get_relevant_documents(question)
    return {"documents": documents}

def answer_node(state: GraphState):
    """Nó que executa o agente Answerer."""
    print("--- EXECUTANDO NÓ: ANSWERER ---")
    question = state["question"]
    documents = state["documents"]
    answer = generate_answer(question, documents)
    return {"answer": answer}

def self_check_node(state: GraphState):
    """
    Nó que executa o agente Self-Check.
    Compara a resposta gerada com os documentos de evidência.
    """
    print("--- EXECUTANDO NÓ: SELF-CHECK ---")
    answer = state["answer"]
    documents = state["documents"]
    
    verdict_obj = check_faithfulness(answer, documents)
    print(f"--- VEREDITO DO SELF-CHECK: {verdict_obj.verdict} ---")
    return {"verdict": verdict_obj}

def clarification_node(state: GraphState):
    """Nó que pede esclarecimentos quando a pergunta é incompleta"""
    print(" --- EXECUTANDO NÓ: CLARIFICATION ---")
    question = state["question"]
    intent = state["intent"]
    
    clarification_response = f"""
🤔 **Preciso entender melhor sua situação para te ajudar adequadamente.**

Você perguntou sobre: {question}

Para dar uma resposta jurídica precisa, preciso de mais detalhes:

📋 **Informações que me ajudariam:**
- Contexto específico da situação
- Local onde ocorreu (se aplicável)
- Documentos ou evidências disponíveis
- Qual resultado você espera alcançar

💡 **Exemplo:** Se é sobre preços diferentes, me diga onde viu cada preço (placa, etiqueta, sistema do caixa) e qual a diferença entre eles.

❓ **Pode reformular sua pergunta com mais detalhes?**
"""
    
    return {"answer": clarification_response}

def safety_node(state: GraphState):
    """Nó que aplica disclaimer de segurança"""
    print(" --- EXECUTANDO NÓ: SAFETY CHECK ---")
    answer = state["answer"]
    final_answer_with_disclaimer = apply_disclaimer(answer)
    return {"answer": final_answer_with_disclaimer}

# --- NÓ DE ROTEAMENTO CONDICIONAL ---

def route_after_supervisor(state: GraphState) -> Literal["clarification", "retrieve"]:
    """Decide se pede esclarecimento ou segue para recuperação"""
    needs_clarification = state.get("needs_clarification", False)
    
    # DEBUG: Imprimir para verificar
    print(f"DEBUG Route: needs_clarification = {needs_clarification}")
    
    # DECISÃO SIMPLES E CLARA
    if needs_clarification is True:  # Verificação explícita
        print("Roteando para: clarification")
        return "clarification"
    else:
        print("Roteando para: retrieve")
        return "retrieve"

def route_after_check(state: GraphState) -> Literal["end_safe", "retry_or_fail"]:
    """
    Decide para onde ir após a checagem de fidelidade.
    """
    verdict = state["verdict"].verdict
    if verdict == "fiel":
        return "end_safe"
    else:
        return "retry_or_fail"

def fail_node(state: GraphState):
    print(" --- EXECUTANDO NÓ: FAIL ---")

    question = state["question"]
    intent = state.get("intent", "consumidor")
    verdict_reason = state["verdict"].reasoning
    documents = state.get("documents", [])
    # Gerar sugestões usando o agente inteligente
    suggestions = rephrase_agent.generate_suggestions(
        question=question,
        documents=documents,
        verdict_reason=verdict_reason,
        intent=intent
    )
    # Formatar sugestões para exibição
    formatted_suggestions = '\n'.join([f'• "{sugg}"' for sugg in suggestions])
    
    intelligent_response = f"""
    🤔 **Não consegui dar uma resposta totalmente precisa, mas tenho sugestões!**

    Você perguntou: "{question}"

    📝 **Tente reformular assim:**
    {formatted_suggestions}

    💡 **Por que essas sugestões?** Baseei-me nos documentos que encontrei e na terminologia jurídica mais adequada para sua pergunta.

    ❓ **Escolha uma das sugestões acima** ou reformule usando termos similares.
    """

    return {"answer": intelligent_response}

# --- CONSTRUÇÃO DO GRAFO ---

def build_graph():
    """
    Constrói o grafo LangGraph conectando os nós com lógica condicional.
    """
    workflow = StateGraph(GraphState)
    
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("answerer", answer_node)
    workflow.add_node("self_check", self_check_node)
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("fail_node", fail_node)
    workflow.add_node("safety_node", safety_node)
    
    workflow.set_entry_point("supervisor")
    
    workflow.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "clarification": "clarification",
            "retrieve": "retriever"
        }
    )
    
    workflow.add_edge("retriever", "answerer")
    workflow.add_edge("answerer", "self_check")
    
    workflow.add_conditional_edges(
        "self_check",
        route_after_check,
        {
            "end_safe": "safety_node",
            "retry_or_fail": "fail_node"
        }
    )
    
    workflow.add_edge("clarification", "safety_node")
    workflow.add_edge("fail_node", "safety_node")
    workflow.add_edge("safety_node", END)

    app = workflow.compile()
    return app

# --- Bloco de Teste Interativo (REPL) ---
if __name__ == '__main__':
    print("Iniciando o Dr. Llama com Supervisor...")
    graph = build_graph()
    
    print("Digite a sua pergunta ou '/bye' para sair.")
    while True:
        try:
            question = input("\nPrompt: ")
            if question.lower().strip() == "/bye":
                print("Até logo! Encerrando...")
                break
            
            if not question:
                continue
            
            inputs = {"question": question}
            print("Processando com supervisor...")
            
            final_state = graph.invoke(inputs)
            final_answer = final_state.get("answer", "Erro: O grafo não produziu uma resposta.")
            
            print("\nResposta:")
            print(final_answer)
            
            # Debug info
            print(f"\n--- DEBUG INFO ---")
            print(f"Intent: {final_state.get('intent', 'N/A')}")
            print(f"Confidence: {final_state.get('confidence', 'N/A')}")
            print(f"Needed Clarification: {final_state.get('needs_clarification', 'N/A')}")
            
            documents = final_state.get("documents", [])
            if documents:
                print(f"Documents Retrieved: {len(documents)}")
                sources = set()
                for doc in documents:
                    pretty_name = doc.metadata.get('pretty_name', 'Desconhecida')
                    sources.add(pretty_name)
                print(f"Sources: {', '.join(sources)}")
                
        except KeyboardInterrupt:
            print("\nEncerrando...")
            break
        except Exception as e:
            print(f"\nOcorreu um erro inesperado: {e}")
            print("Reiniciando o loop...")