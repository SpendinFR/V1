from typing import Dict, List, Optional
def should_refuse(citations: List[Dict], min_docs=2, min_score=0.15, min_top1_score=0.25) -> bool:
    if not citations or len(citations) < min_docs: return True
    sorted_c = sorted(citations, key=lambda c: c.get('score',0.0), reverse=True)
    top1 = sorted_c[0].get('score', 0.0)
    good = [c for c in citations if c.get('score',0.0) >= min_score]
    if len(good) < min_docs: return True
    if top1 < min_top1_score: return True
    return False
def refusal_message(msg: str, diagnostics: Optional[Dict] = None) -> Dict:
    payload = {'answer': None, 'reason': msg, 'citations': [], 'status': 'refused'}
    if diagnostics:
        payload['diagnostics'] = diagnostics
    return payload
