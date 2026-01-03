from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field 


from models.llm_provider import get_llm
from schemas import interpret_candidate_answer_structured, RequirementMatchResult

from langgraph.graph import StateGraph, END


@dataclass
class ConversationState:
    # Requisitos por preguntar
    pending_requirements: List[str] = field(default_factory=list)
    # Requisitos que el candidato dice cumplir
    additional_fulfilled: List[str] = field(default_factory=list)
    # Historial de mensajes (corto plazo)
    history: List[Dict[str, str]] = field(default_factory=list)
    # Resumen de contexto (largo plazo)
    long_term_summary: str = ""
    # Último requisito preguntado
    current_requirement: Optional[str] = None
    # Flag para terminar
    finished: bool = False
    # Última respuesta del candidato (opcional)
    last_candidate_answer: str = ""

def node_select_next_requirement(state: ConversationState) -> ConversationState:
    if not state.pending_requirements:
        state.finished = True
        state.current_requirement = None
        return state

    state.current_requirement = state.pending_requirements.pop(0)
    return state


def node_ask_candidate(state: ConversationState) -> ConversationState:
    """
    Pregunta al candidato por el requisito actual (interacción por terminal).
    Añade el intercambio al historial de corto plazo.
    """
    req = state.current_requirement
    if req is None:
        state.finished = True
        return state

    # Saludo solo si es la primera pregunta (historial vacío)
    if len(state.history) == 0:
        print("\n--- INICIO DE CONVERSACIÓN CON EL CANDIDATO ---\n")
        print("Hola, gracias por tu tiempo. Vamos a preguntarte por algunos requisitos específicos.\n")

    print(f"¿Tienes experiencia o cumples con este requisito?\n- {req}")
    resp = input("Tu respuesta: ")

    # Guardamos en historial
    state.history.append(
        {
            "role": "agent",
            "content": f"¿Tienes experiencia o cumples con este requisito?\n- {req}",
        }
    )
    state.history.append(
        {
            "role": "candidate",
            "content": resp,
        }
    )

    # Guardamos temporalmente la última respuesta en el estado
    state.last_candidate_answer = resp  # atributo dinámico
    return state


def node_evaluate_answer(state: ConversationState) -> ConversationState:
    """
    Usa el LLM (structured output) para decidir si el candidato cumple el requisito actual.
    """
    req = state.current_requirement
    resp = getattr(state, "last_candidate_answer", "")

    if not req:
        state.finished = True
        return state

    # Llamamos a la función estructurada
    result: RequirementMatchResult = interpret_candidate_answer_structured(req, resp)

    if result.cumple:
        state.additional_fulfilled.append(req)

    # Podrías guardar la justificación en el historial si quieres
    state.history.append(
        {
            "role": "system",
            "content": f"Evaluación requisito '{req}': cumple={result.cumple}, justificación={result.justificacion}",
        }
    )

    return state


def node_update_long_term_summary(state: ConversationState) -> ConversationState:
    """
    Actualiza la memoria a largo plazo (resumen del contexto) usando el historial reciente
    + resumen previo si existe.
    """
    llm = get_llm(temperature=0.1)

    # Preparamos un pequeño prompt para resumir
    summary_prompt = """
Eres un asistente que resume el contexto de una conversación con un candidato.

Dispones de:
- Un resumen previo (puede estar vacío).
- Un nuevo bloque de historial reciente.

Tu tarea:
- Actualizar el resumen para que refleje la información relevante acumulada.
- Debe ser breve pero informativo (3-6 frases).

Resumen previo:
{prev_summary}

Historial reciente:
{history}

Devuelve SOLO el nuevo resumen.
"""

    history_text = "\n".join(
        [f"{m['role']}: {m['content']}" for m in state.history[-6:]]  # solo últimos turnos
    )

    messages = [
        {"role": "system", "content": summary_prompt.format(
            prev_summary=state.long_term_summary,
            history=history_text
        )}
    ]

    resp = llm.invoke(messages)
    new_summary = resp.content.strip()

    state.long_term_summary = new_summary
    return state


def should_continue(state: ConversationState) -> str:
    if state.finished:
        return END
    return "loop"

# ==========================
# CONSTRUCCIÓN DEL GRAFO
# ==========================


def build_conversation_graph():
    graph = StateGraph(ConversationState)

    # Nodos
    graph.add_node("select_next_requirement", node_select_next_requirement)
    graph.add_node("ask_candidate", node_ask_candidate)
    graph.add_node("evaluate_answer", node_evaluate_answer)
    graph.add_node("update_long_term_summary", node_update_long_term_summary)

    # Flujo básico
    graph.set_entry_point("select_next_requirement")
    graph.add_edge("select_next_requirement", "ask_candidate")
    graph.add_edge("ask_candidate", "evaluate_answer")
    graph.add_edge("evaluate_answer", "update_long_term_summary")

    # Desde el resumen, decidimos si seguimos o terminamos
    graph.add_conditional_edges(
        "update_long_term_summary",
        should_continue,
        {
            "loop": "select_next_requirement",
            END: END,
        }
    )

    # Checkpointing de memoria (opcional pero útil)
    app = graph.compile()

    return app

def ask_candidate_about_requirements_with_graph(
    not_found_requirements: List[str],
    initial_long_term_summary: str = "",
) -> List[str]:
    if not not_found_requirements:
        return []

    app = build_conversation_graph()

    init_state = ConversationState(
        pending_requirements=list(not_found_requirements),
        additional_fulfilled=[],
        history=[],
        long_term_summary=initial_long_term_summary,
        current_requirement=None,
        finished=False,
    )

    final_state = app.invoke(init_state)

   
    print("\nGracias, hemos registrado tus respuestas.\n")
    print("Resumen de contexto (memoria a largo plazo):")
    print(final_state.get("long_term_summary", ""))  
    print()

    return final_state.get("additional_fulfilled", [])

