from typing import List
from schemas import RequirementItem, parse_requirements_structured

def parse_requirements(oferta_texto: str) -> List[dict]:
    items: List[RequirementItem] = parse_requirements_structured(oferta_texto)
    return [item.dict() for item in items]