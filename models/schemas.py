from typing import List, Literal
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from models.llm_provider import get_llm

class RequirementItem(BaseModel):
    """Requisito atómico de la oferta."""
    texto: str = Field(..., description="Requisito atómico en texto claro.")
    tipo: Literal["obligatorio", "opcional"] = Field(
        ...,
        description='Tipo de requisito: "obligatorio" o "opcional".'
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




## Prompt Templates

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

#few-shot examples

<Agent - 1>
 Hola, gracias por tu tiempo. Vamos a preguntarte por algunos requisitos específicos.
 Podrías indicarnos si tienes experiencia o cumples con este requisito?
 - Conocimientos en FastAPI
</Agent - 1>

<Input - Candidato>
    Hola, si durante mi experiencia he trabajado con FastAPI en varios proyectos. Desde la creación de APIs RESTful hasta la integración con bases de datos.
</Input - Candidato>

<output>
{
    "cumple": true,
}

</output>

<Agent - 2>
 Hola, gracias por tu tiempo. Vamos a preguntarte por algunos requisitos específicos.
 Podrías indicarnos si tienes experiencia o cumples con este requisito?
 - Conocimientos en LangChain
</Agent>

<Input - Candidato>
    No, nunca he trabajado con LangChain.
</Input - Candidato>

<output>
{
    "cumple": false,
}

</output>


#Reducing hallucinations
1. Evita la ambigüedad y la vaguedad
2. Evita fusionar conceptos no relacionados
3. Evita describir escenarios imposibles
4. Evite contradecir los hechos conocidos
5. Evite asignar propiedades poco características

"""

MATCH_REQUIREMENT_SYSTEM_PROMPT = """
#Role
Eres parte del personal de recursos humanos de una empresa. Para esta tarea en particular, recibirar una CV y una lista de requisitos clasificados como obligatorios, opcionales o deseables
y tienes como objetivo evaluar el CV y generar un score. Para cada requisito, debes determinar si el candidato lo cumple o no y generar la salida en formato JSON.

#Input

- Texto libre con el contenido del CV.
- lista de dicts [{"texto":..., "tipo":"obligatorio"/"opcional"}, ...]

#Instructions

1 Recibes el CV y la lista de requisitos.
2 Analiza el texto y te apoyas en el contenido de rules para generar definir si cumple o no el requisito. 
3 Devuelve SOLO un JSON definiendo si cumple o no el requisito.

#Rules 

- No inventes cosas, limitate a evaluar el candidato teniento en cuenta cuantos requisitos cumple su CV.
- Piensa como un profesional de recursos humanos que debe filtrar candidatos.
- "cumple" solo será true si el CV deja claro que el requisito se cumple.
- Si no está claro, considera que NO se cumple (cumple: false).

#few-shot examples

<Input - CV>
    Experiencia:
    Desarrollador de IA Generativa - EMPRESA A (Abril 2023 - Actualidad)
    Encargado de desarrollar sistemas de IA generativa en Python, diseñando prompts eficientes y
    sistemas escalables
    Data Science / LLM - EMPRESA B (Enero 2022 - Abril 2023)
    Analista de datos para el entrenamiento de modelos LLM. Entre mis funciones reentrenamiento
    y validación con prompt diseñados para validar su correcto funcionamiento
    Formación:
    Ingeniería Informática (2017 - 2021
</Input - CV>

<Input - Requisito>
[
  {"Requisito1": "Experiencia mínima de 3 años en Python", "tipo": "obligatorio"},
]
</Input - Requisito>
<output>
{
    "cumple": true,
}

</output>

<Input - CV>
    Experiencia:
    Desarrollador de IA Generativa - EMPRESA A (Abril 2023 - Actualidad)
    Encargado de desarrollar sistemas de IA generativa en Python, diseñando prompts eficientes y
    sistemas escalables
    Data Science / LLM - EMPRESA B (Enero 2022 - Abril 2023)
    Analista de datos para el entrenamiento de modelos LLM. Entre mis funciones reentrenamiento
    y validación con prompt diseñados para validar su correcto funcionamiento
    Formación:
    Ingeniería Informática (2017 - 2021
</Input - CV>

<Input - Requisito>
[
  {"Requisito2": "Formación mínima requerida: Ingeniería/Grado en informática o Master en IA", "tipo": "obligatorio"}
]
</Input - Requisito>
<output>
{
    "cumple": true,
}

</output>

<Input - CV>
    Experiencia:
    Desarrollador de IA Generativa - EMPRESA A (Abril 2023 - Actualidad)
    Encargado de desarrollar sistemas de IA generativa en Python, diseñando prompts eficientes y
    sistemas escalables
    Data Science / LLM - EMPRESA B (Enero 2022 - Abril 2023)
    Analista de datos para el entrenamiento de modelos LLM. Entre mis funciones reentrenamiento
    y validación con prompt diseñados para validar su correcto funcionamiento
    Formación:
    Ingeniería Informática (2017 - 2021
</Input - CV>

<Input - Requisito>
[
  {"Requisito3": "Conocimientos en FastAPI", "tipo": "deseable"}
]
</Input - Requisito>
<output>
{
    "cumple": false,
}

</output>


#Reducing hallucinations
1. Evita la ambigüedad y la vaguedad
2. Evita fusionar conceptos no relacionados
3. Evita describir escenarios imposibles
4. Evite contradecir los hechos conocidos
5. Evite asignar propiedades poco características


"""

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

#Rules 

- No inventes cosas, limitate a identificar los requisitos tecnicos asociados al puesto de trabajo.
- Apoyate por la descripcion del puesto te dará pistas sobre los posibles requisitos.
- Piensa como un profesional de recursos humanos que debe filtrar candidatos.
- Cada requisito puede venir con algún indicador de obligatorio/mínimo o deseable/valorable/opcional, o sin indicador (considera obligatorio si no se indica).
- Si en la frase se usa "valorable", "deseable", "opcional" => tipo = "opcional".
- Si se usa "mínimo", "requerido", "obligatorio" o no se indica => tipo = "obligatorio".
- Si una línea contiene varios requisitos unidos por "y" se deben separar en requisitos distintos.

#few-shot examples

<Input>
Requisitos de la oferta:
- Experiencia mínima de 3 años en Python
- Formación mínima requerida: Ingeniería/Grado en informática o Master en IA
- Valorable conocimientos en FastAPI y LangChain
</Input>

<output>
[
  {"Requisito1": "Experiencia mínima de 3 años en Python", "tipo": "obligatorio"},
  {"Requisito2": "Formación mínima requerida: Ingeniería/Grado en informática o Master en IA", "tipo": "obligatorio"},
  {"Requisito3": "Conocimientos en FastAPI", "tipo": "deseable"},
  {"Requisito4": "Conocimientos en LangChain", "tipo": "deseable"}
]
</output>

#Reducing hallucinations
1. Evita la ambigüedad y la vaguedad
2. Evita fusionar conceptos no relacionados
3. Evita describir escenarios imposibles
4. Evite contradecir los hechos conocidos
5. Evite asignar propiedades poco características


#Objetivo:
1. Devuelve una lista JSON con TODOS los requisitos atómicos (sin "y" / "o" mezclados).
2. Para cada requisito indica:
   - "texto": el requisito atómico.
   - "tipo": "obligatorio" o "opcional".

"""

PROMPTLOUDER_SYSTEM_PROMPT = """
Eres un asistente experto en evaluación de candidatos.
Debes devolver una salida estructurada con:
- message: resumen breve de la evaluación,
- score: puntuación de 0 a 100,
- tags: lista de etiquetas relevantes (por ejemplo ["python", "senior"]).
"""

def parse_requirements_structured(oferta_texto: str) -> List[RequirementItem]:
    """
    Usa structured output para extraer y validar requisitos (RequirementItem)
    desde el texto de la oferta.
    """
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(List[RequirementItem])  # tipo: lista de RequirementItem

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PARSE_REQUIREMENTS_SYSTEM_PROMPT),
            ("user", "Requisitos de la oferta:\n\n{oferta}"),
        ]
    )

    chain = prompt | structured_llm
    result: List[RequirementItem] = chain.invoke({"oferta": oferta_texto})
    return result


def check_requirement_structured(requisito: str, cv_text: str) -> RequirementMatchResult:
    """
    Compara un requisito con el CV usando structured output y RequirementMatchResult.
    """
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(RequirementMatchResult)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", MATCH_REQUIREMENT_SYSTEM_PROMPT),
            ("user", "Requisito: {req}\n\nCV:\n{cv}"),
        ]
    )

    chain = prompt | structured_llm
    result: RequirementMatchResult = chain.invoke({"req": requisito, "cv": cv_text})
    return result

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