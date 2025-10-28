from typing import Dict, Any
import json, os, time
_LOG = os.getenv("RAG_LOG_PATH", "").strip() or None
def log_event(event: Dict[str, Any]):
    event = dict(event); event['ts'] = time.time()
    if _LOG:
        try:
            os.makedirs(os.path.dirname(_LOG), exist_ok=True)
            with open(_LOG, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')
        except Exception:
            pass
    return event
