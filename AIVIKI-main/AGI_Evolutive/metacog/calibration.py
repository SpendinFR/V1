from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple, Iterable, Callable
import os, json, time, uuid, math, re, logging
from collections import deque
from statistics import mean

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    if x > 60.0:
        return 1.0
    if x < -60.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


CalibFeatureFn = Callable[[float, Optional[Dict[str, Any]]], Tuple[float, ...]]


class OnlineLogisticRegressor:
    """Tiny SGD-based logistic regression for calibration purposes."""

    def __init__(self, lr: float = 0.35, l2: float = 1e-3, epochs: int = 40):
        self.lr = float(lr)
        self.l2 = float(l2)
        self.epochs = int(max(1, epochs))

    def fit(self, features: List[Tuple[float, ...]], labels: List[float]) -> Tuple[float, ...]:
        if not features:
            return tuple()
        dims = len(features[0])
        weights = [0.0] * dims

        for _ in range(self.epochs):
            for vec, target in zip(features, labels):
                score = sum(w * x for w, x in zip(weights, vec))
                pred = _sigmoid(score)
                error = pred - target
                for i in range(dims):
                    grad = error * vec[i] + self.l2 * weights[i]
                    weights[i] -= self.lr * grad

        return tuple(weights)


class LogisticCalibrator:
    def __init__(self, weights: Tuple[float, ...], means: Tuple[float, ...], scales: Tuple[float, ...]):
        self.weights = tuple(weights)
        self.means = tuple(means)
        self.scales = tuple(scales)

    def _transform(self, features: Tuple[float, ...]) -> Tuple[float, ...]:
        if not self.weights:
            return tuple()
        transformed = [features[0]]
        for i in range(1, len(features)):
            scale = self.scales[i] if i < len(self.scales) else 1.0
            mean = self.means[i] if i < len(self.means) else 0.0
            if scale == 0:
                transformed.append(0.0)
            else:
                transformed.append((features[i] - mean) / scale)
        return tuple(transformed)

    def predict(self, features: Tuple[float, ...]) -> float:
        transformed = self._transform(features)
        score = sum(w * x for w, x in zip(self.weights, transformed))
        return max(0.0, min(1.0, _sigmoid(score)))


class IsotonicCalibrator:
    """Piecewise-constant isotonic regression using the PAV algorithm."""

    def __init__(self, points: List[Tuple[float, float]]):
        self.points = sorted(points)

    @staticmethod
    def fit(confidences: Iterable[float], labels: Iterable[float]) -> Optional["IsotonicCalibrator"]:
        pairs = sorted((max(0.0, min(1.0, float(c))), max(0.0, min(1.0, float(y)))) for c, y in zip(confidences, labels))
        if not pairs:
            return None

        blocks: List[Dict[str, float]] = []
        for x, y in pairs:
            block = {"weight": 1.0, "avg": y, "min": x, "max": x}
            blocks.append(block)
            while len(blocks) >= 2 and blocks[-2]["avg"] > blocks[-1]["avg"]:
                last = blocks.pop()
                prev = blocks.pop()
                total_w = prev["weight"] + last["weight"]
                merged = {
                    "weight": total_w,
                    "avg": (prev["avg"] * prev["weight"] + last["avg"] * last["weight"]) / total_w,
                    "min": prev["min"],
                    "max": last["max"],
                }
                blocks.append(merged)

        points: List[Tuple[float, float]] = []
        for block in blocks:
            points.append(((block["min"] + block["max"]) / 2.0, block["avg"]))

        if len(points) == 1:
            # Degenerate case: everything identical → constant probability
            points.append((points[0][0], points[0][1]))

        return IsotonicCalibrator(points)

    def predict(self, confidence: float) -> float:
        if not self.points:
            return max(0.0, min(1.0, float(confidence)))
        x = max(0.0, min(1.0, float(confidence)))
        points = self.points
        if x <= points[0][0]:
            return points[0][1]
        if x >= points[-1][0]:
            return points[-1][1]
        for i in range(1, len(points)):
            left_x, left_y = points[i - 1]
            right_x, right_y = points[i]
            if left_x <= x <= right_x:
                if math.isclose(left_x, right_x):
                    return (left_y + right_y) / 2.0
                ratio = (x - left_x) / (right_x - left_x)
                return left_y + ratio * (right_y - left_y)
        return points[-1][1]


class CompositeCalibrator:
    """Blend of logistic + isotonic calibration."""

    def __init__(
        self,
        logistic: Optional[LogisticCalibrator],
        isotonic: Optional[IsotonicCalibrator],
        feature_fn: "CalibFeatureFn",
    ) -> None:
        self.logistic = logistic
        self.isotonic = isotonic
        self._feature_fn = feature_fn

    def predict(self, confidence: float, meta: Optional[Dict[str, Any]] = None) -> float:
        outputs: List[float] = []
        if self.logistic:
            features = self._feature_fn(confidence, meta)
            outputs.append(self.logistic.predict(features))
        if self.isotonic:
            outputs.append(self.isotonic.predict(confidence))
        if not outputs:
            outputs.append(max(0.0, min(1.0, float(confidence))))
        return max(0.0, min(1.0, sum(outputs) / len(outputs)))

    def threshold(self, base: float = 0.45) -> float:
        target = 0.5
        step = 0.005
        samples = [i * step for i in range(int(1.0 / step) + 1)]
        best_sample = base
        best_diff = float("inf")

        for sample in samples:
            pred = self.predict(sample, None)
            diff = abs(pred - target)
            if diff + 1e-6 < best_diff:
                best_diff = diff
                best_sample = sample
            elif abs(diff - best_diff) <= 1e-6 and abs(sample - base) < abs(best_sample - base):
                best_sample = sample

        if best_diff > 0.25 or not math.isfinite(best_sample):
            return max(0.05, min(0.95, base))

        return max(0.05, min(0.95, best_sample))


RE_TEXT_TOKEN = re.compile(r"\w+", re.UNICODE)
RE_HAS_EMOJI = re.compile(r"[\U0001F300-\U0001FAFF]")
_PUNCTUATION = "?!;:…"


def _extract_text(meta: Optional[Dict[str, Any]]) -> str:
    if not isinstance(meta, dict):
        return ""
    for key in ("text", "utterance", "prompt", "content", "message", "raw_text"):
        value = meta.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _calibration_features(confidence: float, meta: Optional[Dict[str, Any]]) -> Tuple[float, ...]:
    conf = max(1e-6, min(1.0 - 1e-6, _safe_float(confidence, 0.0)))
    meta = meta or {}
    text = _extract_text(meta)
    char_len = len(text)
    tokens = RE_TEXT_TOKEN.findall(text.lower()) if text else []
    token_len = len(tokens)
    punctuation_ratio = sum(1 for ch in text if ch in _PUNCTUATION) / max(1, char_len)
    emoji_flag = 1.0 if text and RE_HAS_EMOJI.search(text) else 0.0
    uncertainty = _safe_float(meta.get("uncertainty") or meta.get("uncertainty_score"), 0.0)
    novelty = _safe_float(meta.get("novelty_score"), 0.0)
    hedging = 1.0 if bool(meta.get("hedge")) else 0.0
    mean_token_len = (sum(len(t) for t in tokens) / max(1, token_len)) if token_len else 0.0

    return (
        1.0,
        conf,
        conf * conf,
        math.log1p(char_len) / 6.0,
        math.log1p(token_len) / 5.0,
        punctuation_ratio,
        emoji_flag,
        math.log1p(mean_token_len) / 3.0,
        max(0.0, min(1.0, uncertainty)),
        max(0.0, min(1.0, novelty)),
        hedging,
    )


def _normalise_feature_vectors(vectors: List[Tuple[float, ...]]) -> Tuple[List[Tuple[float, ...]], Tuple[float, ...], Tuple[float, ...]]:
    if not vectors:
        return [], tuple(), tuple()
    dims = len(vectors[0])
    means = [0.0] * dims
    scales = [1.0] * dims
    n = len(vectors)
    for i in range(1, dims):
        mean = sum(vec[i] for vec in vectors) / n
        variance = sum((vec[i] - mean) ** 2 for vec in vectors) / n
        scale = math.sqrt(variance) if variance > 1e-9 else 1.0
        means[i] = mean
        scales[i] = scale
    normed: List[Tuple[float, ...]] = []
    for vec in vectors:
        row = [vec[0]]
        for i in range(1, dims):
            scale = scales[i]
            row.append((vec[i] - means[i]) / scale if scale else 0.0)
        normed.append(tuple(row))
    return normed, tuple(means), tuple(scales)


class OnlineTextNoveltyClassifier:
    def __init__(self, lr: float = 0.18, l2: float = 1e-4) -> None:
        self.lr = float(lr)
        self.l2 = float(l2)
        self.bias = 0.0
        self.weights: Dict[str, float] = {}

    def _features(self, text: str) -> Dict[str, float]:
        text = text or ""
        lowered = text.lower()
        char_len = len(lowered)
        features: Dict[str, float] = {}

        tokens = re.findall(r"[\wàâäéèêëîïôöùûüç']{2,}", lowered)
        if tokens:
            norm = 1.0 / len(tokens)
            for tok in tokens[:60]:
                key = f"tok:{tok}"
                features[key] = features.get(key, 0.0) + norm

        for n in (2, 3):
            if char_len < n:
                continue
            norm = 1.0 / max(1, char_len - n + 1)
            for i in range(char_len - n + 1):
                gram = lowered[i : i + n]
                if not gram.strip():
                    continue
                key = f"ng:{gram}"
                features[key] = features.get(key, 0.0) + norm

        if char_len:
            punctuation_ratio = sum(c in _PUNCTUATION for c in lowered) / char_len
            upper_ratio = sum(1 for c in text if c.isupper()) / char_len
        else:
            punctuation_ratio = 0.0
            upper_ratio = 0.0
        features["char_len"] = min(1.0, char_len / 160.0)
        features["punctuation"] = punctuation_ratio
        features["upper"] = upper_ratio
        features["emoji"] = 1.0 if RE_HAS_EMOJI.search(text) else 0.0

        return features

    def _score(self, feats: Dict[str, float]) -> float:
        score = self.bias
        for key, value in feats.items():
            score += self.weights.get(key, 0.0) * value
        return score

    def predict(self, text: str) -> float:
        feats = self._features(text)
        return max(0.0, min(1.0, _sigmoid(self._score(feats))))

    def update(self, text: str, label: float, sample_weight: float = 1.0) -> None:
        feats = self._features(text)
        pred = _sigmoid(self._score(feats))
        error = (pred - max(0.0, min(1.0, label))) * sample_weight
        self.bias -= self.lr * (error + self.l2 * self.bias)
        for key, value in feats.items():
            w = self.weights.get(key, 0.0)
            grad = error * value + self.l2 * w
            new_weight = w - self.lr * grad
            if abs(new_weight) < 1e-6:
                self.weights.pop(key, None)
            else:
                self.weights[key] = new_weight


from AGI_Evolutive.utils.jsonsafe import json_sanitize

class CalibrationMeter:
    def __init__(self, path: str = "data/calibration.jsonl", max_history: Optional[int] = 2048) -> None:
        self.path = path
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._calibrator_cache: Dict[str, Dict[str, Any]] = {}
        self.max_history = max_history if (isinstance(max_history, int) and max_history > 0) else None
        self._last_llm_assessment: Optional[Dict[str, Any]] = None

    def _invalidate_cache(self) -> None:
        self._calibrator_cache.clear()

    def _read_rows(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def _write_rows(self, rows: Iterable[Dict[str, Any]]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(json_sanitize(row), ensure_ascii=False) + "\n")

    def _enforce_max_history(self) -> None:
        if not self.max_history:
            return
        rows = self._read_rows()
        if len(rows) <= self.max_history:
            return
        self._write_rows(rows[-self.max_history:])

    def log_prediction(self, domain: str, p: float, meta: Optional[Dict[str, Any]] = None) -> str:
        eid = str(uuid.uuid4())
        row = {"id": eid, "t": time.time(), "domain": domain, "p": float(p), "meta": meta or {}}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(json_sanitize(row), ensure_ascii=False) + "\n")
        self._enforce_max_history()
        self._invalidate_cache()
        return eid

    def log_outcome(self, event_id: str, success: bool) -> None:
        items = self._read_rows()
        for it in items:
            if it.get("id") == event_id:
                it["success"] = bool(success)
                break
        self._write_rows(items)
        self._enforce_max_history()
        self._invalidate_cache()

    def _iter(self) -> List[Dict[str, Any]]:
        return self._read_rows()

    def _recent_rows(self, domain: Optional[str] = None, limit: int = 512) -> List[Dict[str, Any]]:
        rows = [
            r
            for r in self._iter()
            if "success" in r and (domain is None or r.get("domain") == domain)
        ]
        if limit and len(rows) > limit:
            rows = rows[-limit:]
        return rows

    def _build_calibrator(self, domain: Optional[str]) -> Optional[CompositeCalibrator]:
        rows = self._recent_rows(domain)
        if len(rows) < 6:
            return None
        labels = [1.0 if r.get("success") else 0.0 for r in rows]
        if all(l == labels[0] for l in labels):
            return None
        features = [_calibration_features(_safe_float(r.get("p"), 0.0), r.get("meta")) for r in rows]
        normed, means, scales = _normalise_feature_vectors(features)
        logistic_weights = OnlineLogisticRegressor().fit(normed, labels)
        logistic = LogisticCalibrator(logistic_weights, means, scales) if logistic_weights else None
        isotonic = IsotonicCalibrator.fit([_safe_float(r.get("p"), 0.0) for r in rows], labels)
        if logistic is None and isotonic is None:
            return None
        return CompositeCalibrator(logistic, isotonic, _calibration_features)

    def _get_calibrator(self, domain: Optional[str]) -> Optional[CompositeCalibrator]:
        key = domain or "__global__"
        rows = self._recent_rows(domain)
        cache = self._calibrator_cache.get(key)
        current_count = len(rows)
        if cache and cache.get("count") == current_count:
            return cache.get("calibrator")
        calibrator = self._build_calibrator(domain)
        self._calibrator_cache[key] = {"count": current_count, "calibrator": calibrator}
        return calibrator

    def _llm_calibration_assessment(
        self,
        domain: Optional[str],
        confidence: float,
        calibrated: float,
        meta: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload = {
            "domain": domain or "global",
            "raw_confidence": confidence,
            "calibrated_confidence": calibrated,
            "meta": meta or {},
            "stats": self.report(domain),
        }

        llm_result = try_call_llm_dict(
            "metacog_calibration",
            input_payload=payload,
            logger=logger,
        )

        if isinstance(llm_result, Dict):
            try:
                perceived = max(0.0, min(1.0, float(llm_result.get("perceived_confidence", calibrated))))
            except (TypeError, ValueError):
                perceived = calibrated
            bias = str(llm_result.get("calibration_bias") or "").strip()
            advice = str(llm_result.get("adjustment_advice") or "").strip()
            notes = llm_result.get("notes")
            result = {
                "perceived_confidence": round(perceived, 3),
                "calibration_bias": bias or None,
                "adjustment_advice": advice or None,
            }
            if isinstance(notes, str) and notes.strip():
                result["notes"] = notes.strip()
            return result

        delta = calibrated - confidence
        if delta > 0.05:
            bias = "sous-confiance"
            advice = "Tu peux affirmer avec un peu plus d'assurance."
        elif delta < -0.05:
            bias = "surconfiance"
            advice = "Réduis légèrement la certitude exprimée."
        else:
            bias = "alignée"
            advice = "La confiance exprimée est cohérente."
        return {
            "perceived_confidence": round(max(0.0, min(1.0, calibrated)), 3),
            "calibration_bias": bias,
            "adjustment_advice": advice,
        }

    def report(self, domain: Optional[str] = None) -> Dict[str, Any]:
        rows = [r for r in self._iter() if "success" in r and (domain is None or r.get("domain")==domain)]
        if not rows:
            return {"count": 0, "brier": None, "ece": None, "bins": []}
        brier = mean([(float(r["p"]) - (1.0 if r["success"] else 0.0))**2 for r in rows])
        # ECE: 10 bins
        bins = [[] for _ in range(10)]
        for r in rows:
            i = min(9, max(0, int(float(r["p"])*10)))
            bins[i].append(r)
        ece_terms = []
        for i, b in enumerate(bins):
            if not b: continue
            conf = mean([float(x["p"]) for x in b])
            acc = mean([1.0 if x["success"] else 0.0 for x in b])
            ece_terms.append(abs(conf - acc) * len(b)/len(rows))
        ece = sum(ece_terms) if ece_terms else None
        return {"count": len(rows), "brier": brier, "ece": ece, "bins": [len(b) for b in bins]}

    def suggested_hedging_delta(self, domain: Optional[str] = None) -> float:
        r = self.report(domain)
        if r["ece"] is None: return 0.0
        if r["ece"] > 0.25:  return +0.15
        if r["ece"] > 0.15:  return +0.08
        if r["ece"] < 0.05:  return -0.04
        return 0.0

    # --- Extensions multi-domaines ---
    def domain_reports(self) -> Dict[str, Dict[str, Any]]:
        domains = sorted({row.get("domain", "global") for row in self._iter() if "success" in row})
        return {d: self.report(d if d != "global" else None) for d in domains}

    def dynamic_threshold(self, domain: Optional[str] = None, base: float = 0.45) -> float:
        info = self.report(domain)
        count = info.get("count", 0) or 0
        ece = info.get("ece")
        threshold = base
        if ece is not None:
            threshold += max(-0.15, min(0.20, ece * 0.6))
        if count > 80:
            threshold -= 0.05
        elif count < 15:
            threshold += 0.05
        return max(0.1, min(0.85, threshold))

    def calibrated_threshold(self, domain: Optional[str] = None, base: float = 0.45) -> float:
        calibrator = self._get_calibrator(domain)
        if calibrator is None:
            return self.dynamic_threshold(domain, base)
        return calibrator.threshold(base)

    def calibrate_probability(
        self,
        domain: Optional[str],
        confidence: float,
        meta: Optional[Dict[str, Any]] = None,
    ) -> float:
        calibrator = self._get_calibrator(domain)
        if calibrator is None:
            calibrated = max(0.0, min(1.0, _safe_float(confidence, 0.0)))
        else:
            calibrated = calibrator.predict(confidence, meta)
        self._last_llm_assessment = self._llm_calibration_assessment(
            domain,
            max(0.0, min(1.0, _safe_float(confidence, 0.0))),
            calibrated,
            meta,
        )
        return calibrated

    @property
    def last_llm_assessment(self) -> Optional[Dict[str, Any]]:
        return self._last_llm_assessment

    def should_abstain(
        self,
        domain: Optional[str],
        confidence: float,
        margin: float = 0.05,
        meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        threshold = self.calibrated_threshold(domain)
        calibrated = self.calibrate_probability(domain, confidence, meta)
        if calibrated + margin < 0.5:
            return True
        return float(confidence) + margin < threshold


class NoveltyDetector:
    """Very light-weight detector for out-of-distribution user inputs."""

    def __init__(self, window: int = 128, threshold: float = 0.45) -> None:
        self.window = deque(maxlen=max(8, window))
        self.threshold = max(0.05, min(0.95, threshold))
        self.classifier = OnlineTextNoveltyClassifier()

    # Feature extraction intentionally simple (token stats + punctuation balance)
    def _features(self, text: str) -> Tuple[float, ...]:
        text = text or ""
        tokens = re.findall(r"\w+", text.lower())
        token_count = len(tokens)
        unique_ratio = len(set(tokens)) / max(1, token_count)
        char_count = len(text)
        digit_ratio = sum(c.isdigit() for c in text) / max(1, char_count)
        upper_ratio = sum(c.isupper() for c in text) / max(1, char_count)
        punctuation_ratio = sum(c in "?!;:" for c in text) / max(1, char_count)
        long_token_ratio = sum(len(t) >= 10 for t in tokens) / max(1, token_count)
        question_tokens = sum(1 for t in tokens if t in {"pourquoi", "comment", "qui", "quoi", "où", "how", "why"})
        question_density = question_tokens / max(1, token_count)
        length_norm = min(1.0, token_count / 160.0)
        return (
            length_norm,
            unique_ratio,
            digit_ratio,
            upper_ratio,
            punctuation_ratio,
            long_token_ratio,
            question_density,
        )

    def _avg_vector(self) -> Optional[Tuple[float, ...]]:
        if not self.window:
            return None
        dims = len(self.window[0])
        sums = [0.0] * dims
        for vec in self.window:
            for i, value in enumerate(vec):
                sums[i] += value
        return tuple(s / len(self.window) for s in sums)

    def novelty_score(self, text: str) -> float:
        vec = self._features(text)
        avg = self._avg_vector()
        if avg is None:
            return 0.0
        dist = math.sqrt(sum((vec[i] - avg[i]) ** 2 for i in range(len(vec))))
        return max(0.0, min(1.0, dist * 2.2))

    def assess(self, text: str, update: bool = True) -> Tuple[float, bool]:
        score = self.novelty_score(text)
        classifier_score = self.classifier.predict(text) if self.classifier else 0.0
        combined_score = max(score, classifier_score)
        flagged = combined_score >= self.threshold
        if update:
            self.window.append(self._features(text))
            if self.classifier:
                if score >= self.threshold + 0.15:
                    self.classifier.update(text, 1.0)
                elif score <= max(0.0, self.threshold - 0.15) and len(self.window) >= 4:
                    self.classifier.update(text, 0.0)
        return combined_score, flagged

    def feedback(self, text: str, novel: bool, weight: float = 1.0) -> None:
        if self.classifier:
            self.classifier.update(text, 1.0 if novel else 0.0, sample_weight=weight)
        if not novel:
            self.window.append(self._features(text))
