from typing import Dict, List, Tuple
import math
def recall_at_k(groundtruth: Dict[str, List[str]], results: Dict[str, List[str]], k=50) -> float:
    num=0; den=0
    for qid, gt in groundtruth.items():
        den += len(set(gt))
        pred=set(results.get(qid,[])[:k])
        num += len(set(gt) & pred)
    return num/den if den else 0.0
def dcg(scores: List[float]) -> float:
    return sum((2**s - 1)/math.log2(i+2) for i,s in enumerate(scores))
def ndcg_at_k(groundtruth: Dict[str, List[str]], scores: Dict[str, List[Tuple[str,float]]], k=10) -> float:
    vals=[]
    for qid, gt in groundtruth.items():
        ranked=scores.get(qid,[])[:k]
        rel=[1.0 if doc in gt else 0.0 for doc,_ in ranked]
        ideal=sorted(rel, reverse=True); idcg=dcg(ideal)
        vals.append(dcg(rel)/idcg if idcg>0 else 1.0)
    return sum(vals)/len(vals) if vals else 0.0
def citation_precision(gold_citations: List[str], produced: List[str]) -> float:
    if not produced: return 0.0
    prod=set(produced); gold=set(gold_citations)
    return len(prod & gold)/len(prod)
def groundedness(answer_sentences: List[Tuple[str, List[str]]], support_map: Dict[str, List[str]]) -> float:
    good=0; total=0
    for sid, gold_docs in answer_sentences:
        total+=1
        pred=set(support_map.get(sid, []))
        if pred & set(gold_docs):
            good+=1
    return good/total if total else 0.0
