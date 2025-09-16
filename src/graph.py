print("Iniciando: importando as bibliotecas e instanciando os agentes...")

from typing import List, TypedDict, Literal
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from agents import retriever_agent
from agents import generate_answer
from agents import check_faithfulness, FaithfulnessCheck
from agents import expand_query

# --- Definição do Estado do Grafo ---

class GraphState(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    verdict: FaithfulnessCheck

# --- NÓS DO GRAFO ---

def query_expander_node(state: GraphState):
    "Nó que executa o agente Query Expander"
    print(" --- EXECUTANDO NÓ: QUERY EXPANDER ---")
    question = state['question']
    question = expand_query(question)
    return {"question": question}

def retrieve_node(state: GraphState):
    """Nó que executa o agente Retriever."""
    question = state["question"]
    documents = retriever_agent.get_relevant_documents(question)
    return {"documents": documents}

def answer_node(state: GraphState):
    """Nó que executa o agente Answerer."""
    question = state["question"]
    documents = state["documents"]
    answer = generate_answer(question, documents)
    return {"answer": answer}

def self_check_node(state: GraphState):
    """Nó que executa o agente Self-Check."""
    answer = state["answer"]
    documents = state["documents"]
    
    verdict_obj = check_faithfulness(answer, documents)
    return {"verdict": verdict_obj}

def fail_node(state: GraphState):
    """Nó de falha. Retorna uma resposta segura."""
    verdict_reason = state["verdict"].reasoning
    safe_answer = (
        "Não consegui gerar uma resposta confiável com base nos documentos disponíveis. "
        f"(Motivo da falha interna: {verdict_reason})"
    )
    return {"answer": safe_answer}

# --- ROTEAMENTO CONDICIONAL (sem alterações) ---

def route_after_check(state: GraphState) -> Literal["end_safe", "retry_or_fail"]:
    """Decide para onde ir após a checagem."""
    verdict = state["verdict"].verdict
    if verdict == "fiel":
        return "end_safe"
    else:
        return "retry_or_fail"

# --- CONSTRUÇÃO DO GRAFO (sem alterações) ---

def build_graph():
    """Constrói o grafo LangGraph."""
    workflow = StateGraph(GraphState)
    
    workflow.add_node("query_expander", query_expander_node)
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("answerer", answer_node)
    workflow.add_node("self_check", self_check_node)
    workflow.add_node("fail_node", fail_node)
    
    workflow.set_entry_point("query_expander")
    
    workflow.add_edge("query_expander", "retriever")
    workflow.add_edge("retriever", "answerer")
    workflow.add_edge("answerer", "self_check")

    workflow.add_conditional_edges(
        "self_check",                   
        route_after_check,              
        {
            "end_safe": END,
            "retry_or_fail": "fail_node"
        }
    )
    
    workflow.add_edge("fail_node", END)

    app = workflow.compile()
    return app

# --- Bloco de Teste Interativo (REPL) ---
if __name__ == '__main__':
    print("Iniciando o Dr. Llama (Modo de Teste Interativo)...")
    
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
            
            print("Processando...")
            final_state = graph.invoke(inputs)
            
            final_answer = final_state.get("answer", "Erro: O grafo não produziu uma resposta.")
            print("\nResposta:")
            print(final_answer)
            
            documents = final_state.get("documents", [])
            if documents:
                print("\n--- Fontes Recuperadas ---")
                sources = set()
                for doc in documents:
                    pretty_name = doc.metadata.get('pretty_name', 'Desconhecida')
                    sources.add(pretty_name)
                print(", ".join(sources))
            
        except KeyboardInterrupt:
            # Permite sair com Ctrl+C
            print("\nEncerrando...")
            break
        except Exception as e:
            print(f"\nOcorreu um erro inesperado: {e}")
            print("Reiniciando o loop...")