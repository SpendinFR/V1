from typing import List, Tuple, Dict
import importlib
def _token_count(text: str, tok_name: str = None) -> int:
    if tok_name:
        try:
            tok = importlib.import_module('transformers').AutoTokenizer.from_pretrained(tok_name)
            return len(tok.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return max(1, len(text)//4)
def find_snippet(text: str, query: str, max_chars=400) -> Tuple[int,int,str]:
    terms=[t for t in query.lower().split() if len(t)>2]; low=text.lower()
    for t in terms:
        idx=low.find(t)
        if idx!=-1:
            start=max(0, idx - max_chars//4)
            end=min(len(text), idx + max_chars - (idx - start))
            return start,end,text[start:end]
    return 0, min(len(text), max_chars), text[:max_chars]
def assemble_citations(query: str, doc_texts: Dict[str,str], ranked: List[Tuple[str,float]],
                       budget_tokens=1200, snippet_chars=400, tokenizer: str = None) -> List[Dict]:
    used_tokens=0; out=[]
    for doc_id, score in ranked:
        text=doc_texts.get(doc_id,''); s,e,snip=find_snippet(text, query, max_chars=snippet_chars)
        snip_tokens=_token_count(snip, tokenizer)
        if used_tokens + snip_tokens > budget_tokens: break
        out.append({'doc_id':doc_id, 'score':score, 'start':s, 'end':e, 'snippet':snip, 'tokens':snip_tokens})
        used_tokens += snip_tokens
    return out
