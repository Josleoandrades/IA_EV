- IA Eval – Evaluación de Candidatos con IA -
  Este proyecto es una herramienta de línea de comandos para evaluar candidatos frente a una oferta de trabajo, usando modelos de lenguaje (OpenAI) y LangChain.
  Permite:
    Extraer y clasificar requisitos de una oferta (obligatorios / opcionales, con soporte para grupos lógicos tipo OR).
    Evaluar un CV contra todos los requisitos en una sola llamada al LLM (structured output).
    Preguntar al candidato por requisitos no encontrados mediante una conversación guiada, con memoria de contexto (LangGraph).
    Recalcular la puntuación final del candidato integrando sus respuestas.
  
1. Estructura del proyecto
  ia_eval/
  ├─ main.py
  ├─ config.py
  ├─ requirements.txt
  ├─ models/
  │  └─ llm_provider.py
  ├─ schemas.py
  ├─ services/
  │  ├─ requirement_parser.py
  │  ├─ cv_evaluator.py
  │  └─ conversation_agent.py
  └─ Dockerfile
  Archivos principales
  main.py


Contiene:
  Modelos Pydantic para structured output, por ejemplo:
  RequirementItem (requisito atómico con campos texto, tipo, group, operator).
  RequirementItemsResponse (lista de requisitos).
  RequirementEvalItem y RequirementEvalList (evaluación batch de requisitos contra el CV).
  RequirementMatchResult (evaluación de un requisito o respuesta del candidato).
  PromptLouderResult (ejemplo genérico para otros outputs).
  Prompts bien definidos para:
  Parsear requisitos de la oferta.
  Evaluar requisitos vs CV.
  Interpretar respuestas del candidato.
  Funciones de alto nivel con structured output, por ejemplo:
  parse_requirements_structured(oferta_texto) -> List[RequirementItem]
  check_requirement_structured(requisitos: List[str], cv_text: str) -> List[RequirementEvalItem]
  interpret_candidate_answer_structured(requisito, respuesta) -> RequirementMatchResult
  promptlouder(user_prompt) -> PromptLouderResult
  services/requirement_parser.py

Usa parse_requirements_structured para:
  Recibir el texto de requisitos de la oferta.
  Devolver una lista de dict con cada requisito, su tipo, y (cuando aplique) su group y operator para lógica OR/AND.
  services/cv_evaluator.py

Implementa la lógica de evaluación:
  Llama a check_requirement_structured para evaluar todos los requisitos en una sola llamada al LLM.
  Construye grupos lógicos:
  Cada requisito puede tener group y operator (AND / OR).
  Ejemplo: “Ingeniería informática o Máster en IA” se modela como 2 requisitos con el mismo group y operator="OR".
  Calcula:
  matching_requirements (cumplidos).
  unmatching_requirements (obligatorios no cumplidos).
  not_found_requirements (opcionales no cumplidos / no encontrados).
  discarded (si falla algún obligatorio, considerado a nivel de grupo).
  score (porcentaje de requisitos cumplidos sobre el total, ajustado a la regla de descarte).
  Función adicional:
  reevaluate_with_additional_info(...) para recalcular puntuación tras la conversación con el candidato.
  services/conversation_agent.py

Implementa el agente de conversación con LangGraph:
  Usa un ConversationState (dataclass) con:
  pending_requirements: lista de requisitos a preguntar.
  additional_fulfilled: requisitos que el candidato afirma cumplir.
  history: historial de turnos (memoria de corto plazo).
  long_term_summary: resumen de contexto (memoria de largo plazo).
  current_requirement, finished, etc.
  Nodos del grafo:
    select_next_requirement: toma el siguiente requisito o marca fin.
    ask_candidate: pregunta por el requisito actual (interfaz consola).
    evaluate_answer: llama a interpret_candidate_answer_structured para decidir si cumple.
    update_long_term_summary: genera/actualiza un resumen corto de la conversación con el LLM.
    Función pública:
    ask_candidate_about_requirements_with_graph(not_found_requirements, initial_long_term_summary="") -> List[str]
    Devuelve los requisitos adicionales que el candidato dice cumplir.

- Dockerfile:
  Permite construir una imagen Docker reproducible con todas las dependencias.

2. Requisitos previos
  Python 3.10+ (se recomienda 3.11).
  Cuenta de OpenAI y una API Key válida.
  Docker instalado para ejecutar en contenedor.

4. Instalación y ejecución local (sin Docker)
  Clonar el repo o copiar el proyecto a tu máquina.
  Crear entorno virtual:
  bash
  
  python -m venv .venv
  # Linux / macOS
  source .venv/bin/activate
  # Windows (PowerShell)
  # .venv\Scripts\Activate.ps1
  Instalar dependencias:
  bash
  
  pip install -r requirements.txt
  Definir la API Key de OpenAI:
  bash
  
  # Linux / macOS
  export OPENAI_API_KEY="TU_API_KEY_AQUI"
  
  # Windows (PowerShell)
  $env:OPENAI_API_KEY="TU_API_KEY_AQUI"
  Ejecutar la aplicación:
  bash


- python main.py -
  El flujo será:
    Introduces los requisitos de la oferta (texto), finalizando con una línea vacía.
    Introduces el CV del candidato (texto), finalizando con una línea vacía.
  El sistema:
    Parseará requisitos.
    Evaluará el CV (fase 1).
    Si hay requisitos opcionales no encontrados y el candidato no está descartado:
    Iniciará una conversación por consola para preguntar por esos requisitos (fase 2).
    Mostrará la puntuación final y si el candidato queda descartado.
  
4. Uso en Docker

  4.1. Construir la imagen
    Desde la raíz del proyecto (donde está el Dockerfile):
    bash
    
    docker build -t ia-eval:latest .
  4.2. Ejecutar el contenedor
    Con API key en línea de comandos:
    bash
    
    docker run -it --rm \
      -e OPENAI_API_KEY="TU_API_KEY_AQUI" \
      ia-eval:latest
    O usando un fichero .env en la raíz con:
    env
    
    OPENAI_API_KEY=TU_API_KEY_AQUI
    y luego:
    bash
    
    docker run -it --rm --env-file .env ia-eval:latest
    
5. Modelo de datos 
  5.1. Requisitos de la oferta – RequirementItem
  
    RequirementItem(
        texto="Formación mínima requerida: Ingeniería/Grado en informática",
        tipo="obligatorio",
        group="formacion_minima",   # opcional
        operator="OR",              # opcional ("AND" o "OR")
    )
    group + operator permiten expresar lógica:
    Grupo formacion_minima con operator="OR" → basta que se cumpla uno de los requisitos del grupo.
    Si group es None, el requisito se trata como independiente.
  5.2. Evaluación batch – RequirementEvalItem
  RequirementEvalItem(
      requisito="Experiencia mínima de 3 años en Python",
      cumple=True,
      justificacion="El CV indica varios años de experiencia desarrollando en Python."
  )
  El evaluador agrupa y aplica la lógica de descarte y puntuación a partir de estos resultados.
