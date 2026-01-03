from services.requirement_parser import parse_requirements
from services.cv_evaluator import (
    evaluate_cv_against_requirements,
    reevaluate_with_additional_info,
)

from services.conversation_agent import ask_candidate_about_requirements_with_graph




def main():
    print("=== Sistema de Evaluación de Candidatos con IA (Prueba Técnica) ===\n")

    # 1) Leer requisitos de la oferta
    print("Introduce el texto de los requisitos de la oferta (finaliza con una línea vacía):")
    lines = []
    while True:
        line = input()
        if not line.strip():
            break
        lines.append(line)
    oferta_texto = "\n".join(lines)

    # 2) Leer CV del candidato
    print("\nIntroduce el texto completo del CV del candidato (finaliza con una línea vacía):")
    cv_lines = []
    while True:
        line = input()
        if not line.strip():
            break
        cv_lines.append(line)
    cv_text = "\n".join(cv_lines)

    # 3) Parseo de requisitos con el LLM
    print("\nAnalizando requisitos de la oferta...\n")
    requisitos = parse_requirements(oferta_texto)

    if not requisitos:
        print("No se han podido extraer requisitos de la oferta. Revisa el texto de entrada.")
        return

    print("Requisitos interpretados:")
    for r in requisitos:
        print(f"- [{r['tipo']}] {r['texto']}")
    print()

    # 4) Evaluar CV contra los requisitos (Fase 1)
    print("Evaluando CV contra la oferta...\n")
    eval_result = evaluate_cv_against_requirements(requisitos, cv_text)

    print("Resultado de la primera fase:")
    print(eval_result)

    if eval_result["discarded"]:
        print("\nEl candidato ha sido descartado por no cumplir un requisito obligatorio.")
        print(f"Puntuación final: {eval_result['score']}%")
        return

    # 5) Si no está descartado, iniciamos conversación para requisitos no encontrados
    not_found = eval_result["not_found_requirements"]

    print(not_found)
    
    seen = set()
    not_found_unique = []
    for r in not_found:
        if r not in seen:
            seen.add(r)
            not_found_unique.append(r)

    
    additional_fulfilled = ask_candidate_about_requirements_with_graph(
    not_found_requirements=not_found_unique,
    initial_long_term_summary="",
)


    # 6) Recalcular puntuación con la nueva información
    print("Recalculando puntuación con la información adicional obtenida...\n")
    reeval = reevaluate_with_additional_info(
        requisitos,
        initial_matching=eval_result["matching_requirements"],
        additional_fulfilled=additional_fulfilled,
    )

    print("Resultado final tras la conversación:")
    print(reeval)
    print(f"\nPuntuación final: {reeval['score']}%")
    if reeval["discarded"]:
        print("El candidato ha sido descartado por no cumplir algún requisito obligatorio.")
    else:
        print("El candidato NO ha sido descartado.")


if __name__ == "__main__":
    main()
