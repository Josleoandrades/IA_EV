# schemas.py
from typing import List, Literal
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from models.llm_provider import get_llm


# ==========================
# 1. MODELOS Pydantic
# ==========================

class RequirementItem(BaseModel):
    texto: str = Field(...)
    tipo: Literal["obligatorio", "opcional"] = Field(...)
    group: str | None = Field(
        default=None,
        description="Identificador del grupo lógico (por ejemplo 'formacion_minima')."
    )
    operator: Literal["AND", "OR"] | None = Field(
        default=None,
        description="Operador lógico entre requisitos del mismo grupo."
    )

class RequirementItemsResponse(BaseModel):
    requirements: List[RequirementItem] = Field(
        ...,
        description="Lista de requisitos atómicos de la oferta."
    )


class RequirementMatchResult(BaseModel):
    """Resultado de comparar un requisito contra un CV o respuesta."""
    cumple: bool = Field(
        ...,
        description="Indica si se cumple el requisito."
    )
    justificacion: str = Field(
        ...,
        description="Breve explicación de por qué se considera que cumple o no."
    )


class PromptLouderResult(BaseModel):
    """
    Ejemplo genérico de salida estructurada para la función promptlouder.
    Ajusta campos según la lógica que quieras.
    """
    message: str = Field(
        ...,
        description="Mensaje generado por el modelo."
    )
    score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Puntuación entre 0 y 100."
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Etiquetas asociadas al resultado."
    )

class RequirementEvalItem(BaseModel):
    requisito: str = Field(..., description="Texto del requisito.")
    cumple: bool = Field(..., description="Indica si se cumple el requisito.")
    justificacion: str = Field(
        ...,
        description="Breve explicación de por qué se considera que cumple o no."
    )
class RequirementEvalList(BaseModel):
    items: List[RequirementEvalItem] = Field(
        ...,
        description="Lista de requisitos evaluados contra el CV."
    )

# ==========================
# 2. PROMPTS (SYSTEM PROMPTS)
# ==========================

# Prompt para extraer requisitos desde el texto de la oferta
PARSE_REQUIREMENTS_SYSTEM_PROMPT = """
#Role
Eres parte del personal de recursos humanos de una empresa. Para esta tarea en particular, recibirar una oferta de trabajo
y tienes como objetivo extraer los requisitos contenidos en ella. Ojo que en una misma linea pueden aparecer varios requisitos
por lo que tendrás que analizar el texto cuidadosamente y distinguir los requisitos contenidos.

#Input

- Texto libre con la descripción de de una oferta de trabajo.


#Instructions

1 Recibes la oferta de trabajo. 
2 Analiza el texto y extrae los requisitos atómicos.
3 Para cada requisito, determina si es obligatorio o opcional según las pistas del texto.
4 Devuelve SOLO un JSON con la lista de requisitos y su tipo.



#Reducing hallucinations
1. Evita la ambigüedad y la vaguedad
2. Evita fusionar conceptos no relacionados
3. Evita describir escenarios imposibles
4. Evite contradecir los hechos conocidos
5. Evite asignar propiedades poco características

#Objetivo
Cada requisito tiene estos campos:
- texto: requisito atómico en español.
- tipo: "obligatorio" o "opcional".
- group: identificador de grupo lógico cuando varios requisitos están conectados por un "o" (disyunción lógica).
- operator: "AND" o "OR" para indicar cómo se combinan los requisitos dentro del mismo grupo.

Reglas generales:
- Si en la frase se usa "valorable", "deseable", "opcional" => tipo = "opcional".
- Si se usa "mínimo", "requerido", "obligatorio" o no se indica => tipo = "obligatorio".
- Si una línea contiene varios requisitos unidos por "y" => sepáralos en varios requisitos atómicos,
  normalmente sin agruparlos (operator AND por defecto si pertenecen a un mismo grupo claro).
- Si una línea contiene varios requisitos unidos por "o" (por ejemplo "Ingeniería Informática o Máster en IA")
  debes:
  - crear un grupo lógico (por ejemplo "formacion_minima" o un identificador similar),
  - crear un requisito para cada opción,
  - asignar el mismo valor en 'group' a esos requisitos,
  - asignar operator = "OR" a esos requisitos.

Reglas específicas para 'group' y 'operator':
- Si un requisito no pertenece a ningún grupo lógico => group = null, operator = null.
- Si varios requisitos pertenecen al mismo grupo lógico unido por "o" => 
  - todos comparten el mismo group (por ejemplo "formacion_minima"),
  - operator = "OR" en todos ellos.
- Puedes inventar el nombre del grupo, pero debe ser corto y consistente, por ejemplo:
  - "formacion_minima" para requisitos de formación mínima,
  - "experiencia_stack" para requisitos que son alternativas de tecnología, etc.

No devuelvas explicaciones en texto libre, solo rellena correctamente los campos definidos en el modelo.

"""

# Prompt para comparar una  lista de requisitos con un CV
MATCH_REQUIREMENT_SYSTEM_PROMPT = """
#Role
Eres parte del personal de recursos humanos de una empresa. Para esta tarea en particular, recibirar una CV y una lista de requisitos clasificados como obligatorios, opcionales o deseables
y tienes como objetivo evaluar el CV y generar un score. Para cada requisito, debes determinar si el candidato lo cumple o no. Debes tener en cuenta que aunque no venga explicitamente
escrita el requisito puede que haya realizado trabajos que usan esas tecnologías o habilidades. Debes analiza el trabajo que se realiza en el puesto y trazarlo con los requisitos. Por ejemplo
para el requisito Experiencia en Python, si en el CV viene que ha trabajado en desarrollo backend y no se menciona Python pero si Django, puedes inferir que conoce Python.

#Input

- Texto libre con el contenido del CV.
- lista de dicts ["texto":..., "tipo":"obligatorio"/"opcional", ...]

#Instructions

1 Recibes el CV y la lista de requisitos.
2 Analiza el texto y te apoyas en el contenido de rules para generar definir si cumple o no el requisito. 


#Rules 

- No inventes cosas, limitate a evaluar el candidato teniento en cuenta cuantos requisitos cumple su CV.
- Piensa como un profesional de recursos humanos que debe filtrar candidatos.
- "cumple" solo será true si el CV deja claro que el requisito se cumple.
- Si no está claro, considera que NO se cumple (cumple: false).


#Reducing hallucinations
1. Evita la ambigüedad y la vaguedad
2. Evita fusionar conceptos no relacionados
3. Evita describir escenarios imposibles
4. Evite contradecir los hechos conocidos
5. Evite asignar propiedades poco características
"""

# Prompt para interpretar la respuesta de un candidato sobre un requisito concreto
INTERPRET_CANDIDATE_ANSWER_SYSTEM_PROMPT = """
#Role
Eres parte del personal de recursos humanos de una empresa. Para esta tarea en particular interactuas con un candidato. Tienes como objetivo preguntar al candidato por ciertos requisitos que no aparecen en su CV
y determinar si los cumple o no, basándote en sus respuestas.

#Input

- Texto libre con el contenido respuesta del candidato.

#Instructions

1 Inicias una conversación con el candidato, debe saludar al candidato y preguntar por los
requisitos no encontrados.
2 Analiza el texto y te apoyas en el contenido de rules para generar definir si cumple o no el requisito. 
3 Devuelve SOLO un JSON definiendo si cumple o no el requisito.

#Rules 

- No inventes cosas, limitate a evaluar el candidato teniento en cuenta cuantos requisitos cumple su CV.
- Piensa como un profesional de recursos humanos que debe filtrar candidatos.
- "cumple" solo será true si la respuesta del candidato indica que tiene el requisito.
- Si no está claro, considera que NO se cumple (cumple: false).




#Reducing hallucinations
1. Evita la ambigüedad y la vaguedad
2. Evita fusionar conceptos no relacionados
3. Evita describir escenarios imposibles
4. Evite contradecir los hechos conocidos
5. Evite asignar propiedades poco características
"""

# Prompt genérico para promptlouder
PROMPTLOUDER_SYSTEM_PROMPT = """
Eres un asistente experto en evaluación de candidatos.
Debes devolver una salida estructurada con:
- message: resumen breve de la evaluación,
- score: puntuación de 0 a 100,
- tags: lista de etiquetas relevantes (por ejemplo ["python", "senior"]).
"""



def parse_requirements_structured(oferta_texto: str) -> List[RequirementItem]:
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(RequirementItemsResponse)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PARSE_REQUIREMENTS_SYSTEM_PROMPT),
            ("user", "Requisitos de la oferta:\n\n{oferta}"),
        ]
    )

    chain = prompt | structured_llm
    result: RequirementItemsResponse = chain.invoke({"oferta": oferta_texto})
    return result.requirements


def check_requirement_structured(
    requisitos: list[str],
    cv_text: str,
) -> list[RequirementEvalItem]:
    llm = get_llm(temperature=0.0)

    # ✅ Pasamos el modelo contenedor, NO List[...]
    structured_llm = llm.with_structured_output(RequirementEvalList)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", MATCH_REQUIREMENT_SYSTEM_PROMPT),
            ("user", "Requisitos:\n{reqs}\n\nCV:\n{cv}"),
        ]
    )

    reqs_str = "\n".join(f"- {r}" for r in requisitos)

    chain = prompt | structured_llm
    result: RequirementEvalList = chain.invoke({"reqs": reqs_str, "cv": cv_text})

    # Devolvemos la lista pura
    return result.items



def interpret_candidate_answer_structured(requisito: str, respuesta: str) -> RequirementMatchResult:
    """
    Interpreta si el candidato cumple el requisito a partir de su respuesta libre.
    """
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(RequirementMatchResult)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", INTERPRET_CANDIDATE_ANSWER_SYSTEM_PROMPT),
            ("user", "Requisito: {req}\nRespuesta del candidato: {resp}"),
        ]
    )

    chain = prompt | structured_llm
    result: RequirementMatchResult = chain.invoke({"req": requisito, "resp": respuesta})
    return result


def promptlouder(user_prompt: str) -> PromptLouderResult:
    """
    Función genérica que:
    - Crea el LLM,
    - Usa PROMPTLOUDER_SYSTEM_PROMPT como system,
    - Devuelve PromptLouderResult validado.
    """
    llm = get_llm(temperature=0.2)
    structured_llm = llm.with_structured_output(PromptLouderResult)

    result: PromptLouderResult = structured_llm.invoke(
        [
            {"role": "system", "content": PROMPTLOUDER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )
    return result

