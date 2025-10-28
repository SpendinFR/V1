import datetime
import pathlib
from typing import Any


def json_sanitize(x: Any):
    try:
        import numpy as np  # type: ignore

        if isinstance(x, (np.bool_,)):
            return bool(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
    except Exception:
        pass

    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (list, tuple, set)):
        return [json_sanitize(i) for i in x]
    if isinstance(x, dict):
        return {str(k): json_sanitize(v) for k, v in x.items()}
    if isinstance(x, (datetime.date, datetime.datetime)):
        return x.isoformat()
    if isinstance(x, pathlib.Path):
        return str(x)
    if hasattr(x, "__dict__"):
        return json_sanitize(vars(x))
    return str(x)
