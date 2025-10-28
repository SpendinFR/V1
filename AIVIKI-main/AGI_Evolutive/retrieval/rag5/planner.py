from typing import List
def expand_queries(question: str) -> List[str]:
    q = question.strip().rstrip('?')
    candidates=[q, q+' contexte', q+' définition']
    if 'quel' in q: candidates.append(q.replace('quel','quelle'))
    out=[]; seen=set()
    for c in candidates:
        if c not in seen: seen.add(c); out.append(c)
    return out
def decompose(question: str) -> List[str]:
    parts=[p.strip() for p in question.replace(' et ', ',').split(',') if p.strip()]
    return parts or [question]
