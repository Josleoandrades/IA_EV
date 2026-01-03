from typing import List, Dict, Tuple
from langchain_core.prompts import ChatPromptTemplate
from models.llm_provider import get_llm
from schemas import check_requirement_structured, RequirementEvalItem

def check_requirement_against_cv(requisito: str, cv_text: str) -> bool:
    result = check_requirement_structured(requisito, cv_text)
    # Si quieres, puedes loguear result.justificacion
    return result.cumple


def evaluate_cv_against_requirements(
    requisitos: List[Dict],   # [{"texto":..., "tipo":..., "group":..., "operator":...}, ...]
    cv_text: str,
) -> Dict:
    if not requisitos:
        return {
            "score": 0.0,
            "discarded": False,
            "matching_requirements": [],
            "unmatching_requirements": [],
            "not_found_requirements": [],
        }

    textos = [r["texto"] for r in requisitos]

    
    eval_items: List[RequirementEvalItem] = check_requirement_structured(textos, cv_text)

    cumple_map = {item.requisito: item.cumple for item in eval_items}

    
    groups: Dict[str, Dict] = {}
    for r in requisitos:
        texto = r["texto"]
        tipo = r.get("tipo", "obligatorio")
        group = r.get("group")  # puede ser None
        operator = r.get("operator", "AND")  # por defecto

        cumple = bool(cumple_map.get(texto, False))

        if group is None:
            # tratamos cada uno como grupo unitario
            gid = f"__single__::{texto}"
            groups[gid] = {
                "tipo": tipo,
                "operator": "AND",
                "requirements": [
                    {"texto": texto, "cumple": cumple}
                ],
            }
        else:
            if group not in groups:
                groups[group] = {
                    "tipo": tipo,
                    "operator": operator,
                    "requirements": [],
                }
            groups[group]["requirements"].append(
                {"texto": texto, "cumple": cumple}
            )

    matching = []
    unmatching = []
    not_found = []

    descartado = False

    # 3) Evaluar cada grupo según su operador
    for gid, gdata in groups.items():
        tipo = gdata["tipo"]          
        operator = gdata["operator"]  
        reqs = gdata["requirements"]  

        if operator == "OR":
            group_cumple = any(r["cumple"] for r in reqs)
        else:  # AND o por defecto
            group_cumple = all(r["cumple"] for r in reqs)

        # Distribuimos matching / unmatching / not_found a nivel de requisito
        if group_cumple:
            for r in reqs:
                matching.append(r["texto"])
        else:
            # el grupo no se cumple
            if tipo == "obligatorio":
                descartado = True
                for r in reqs:
                    unmatching.append(r["texto"])
            else:
                for r in reqs:
                    not_found.append(r["texto"])

    total_requisitos = len(requisitos)
    num_cumplidos = len(set(matching))  # por si algún texto se repite

    score = (num_cumplidos / total_requisitos) * 100.0 if total_requisitos > 0 else 0.0
    if descartado:
        score = 0.0

    return {
        "score": round(score, 2),
        "discarded": descartado,
        "matching_requirements": matching,
        "unmatching_requirements": unmatching,
        "not_found_requirements": not_found,
    }

def reevaluate_with_additional_info(
    requisitos: List[Dict],
    initial_matching: List[str],
    additional_fulfilled: List[str],
) -> Dict:
    """
    Recalcula la puntuación después de preguntar al candidato por 
    requisitos no encontrados en el CV.

    - requisitos: lista completa de requisitos
    - initial_matching: requisitos cumplidos detectados en el CV
    - additional_fulfilled: requisitos que el candidato dice cumplir en la conversación
    """
    total_requisitos = len(requisitos)
    if total_requisitos == 0:
        return {"score": 0.0, "discarded": False}

    # un requisito cuenta como cumplido si:
    #   - estaba en initial_matching, o
    #   - el candidato ha dicho en conversación que lo cumple
    all_matching = set(initial_matching) | set(additional_fulfilled)

    # si no cumple algún obligatorio (ni en CV ni en conversación) => descartado
    descartado = False
    for r in requisitos:
        texto = r["texto"]
        tipo = r.get("tipo", "obligatorio")
        if tipo == "obligatorio" and texto not in all_matching:
            descartado = True
            break

    num_cumplidos = len(
        [r for r in requisitos if r["texto"] in all_matching]
    )
    score = (num_cumplidos / total_requisitos) * 100.0

    if descartado:
        score = 0.0

    return {
        "score": round(score, 2),
        "discarded": descartado,
        "matching_requirements": list(all_matching),
    }
