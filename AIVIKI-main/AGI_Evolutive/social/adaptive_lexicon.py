# AGI_Evolutive/social/adaptive_lexicon.py
# Lexique adaptatif des marqueurs (positifs/négatifs), avec rétention 2-couches :
# - Couche ACTIVE : postérieurs Beta avec décroissance douce (priorise l'actualité)
# - Couche ARCHIVE : totaux stables (sans decay), jamais oubliés
# Réactivation automatique d'anciens marqueurs quand ils réapparaissent.

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import re, json, os, time, math, unicodedata, random, logging

from AGI_Evolutive.utils.llm_service import try_call_llm_dict

LOGGER = logging.getLogger(__name__)

# ----------------- utilitaires -----------------
def _now(): return time.time()
def clamp(x,a=0.0,b=1.0): return max(a, min(b, x))

_STOPWORDS = set("""
le la les un une des de du au aux et ou mais donc car que qui quoi dont où
je tu il elle on nous vous ils elles ne pas plus moins très trop ce cette ces
mon ton son ma ta sa mes tes ses est es suis êtes sont c'est ça ok d' l'
""".split())


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = _strip_accents(s)
    s = s.strip().lower()
    s = re.sub(r"[^\w\s'"+"]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

_EMOJI_RE = re.compile(
    "["                       # basic emoji ranges
    "\U0001F300-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "]+", flags=re.UNICODE)

def _ngrams(tokens: List[str], nmin=1, nmax=3):
    for n in range(nmin, nmax+1):
        for i in range(len(tokens)-n+1):
            yield " ".join(tokens[i:i+n])

def _tokenize(s: str) -> List[str]:
    # split emojis into separate tokens and words
    em_split = _EMOJI_RE.split(s)
    emojis = _EMOJI_RE.findall(s)
    parts: List[str] = []
    for i, chunk in enumerate(em_split):
        words = [w for w in re.split(r"\s+", chunk) if w]
        parts += words
        if i < len(emojis):
            parts.append(emojis[i])
    out: List[str] = []
    for t in parts:
        if len(t) <= 1 and t not in {"❤️","👍","👌","👏","🔥","🤣","😂","😅","😆","😍"}:
            continue
        out.append(t)
    return out


class OnlineLogisticCalibrator:
    """Logistic calibration mise à jour en ligne pour ajuster les incréments."""

    def __init__(self, lr: float = 0.05, l2: float = 1e-4, clip: float = 6.0):
        self.lr = lr
        self.l2 = l2
        self.clip = clip
        self.n_features = 10
        self.weights: List[float] = [0.0 for _ in range(self.n_features)]

    def _sigmoid(self, x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def _features(
        self,
        entry: "LexEntry",
        reward: float,
        confidence: float,
        now_ts: float,
        prev_last_ts: Optional[float],
        was_dormant: bool,
    ) -> List[float]:
        phrase = entry.phrase
        ttl = max(1.0, float(getattr(entry, "ttl_days", 0.0) or 1.0))
        recency = 0.0
        if prev_last_ts:
            recency = math.tanh((now_ts - float(prev_last_ts)) / (ttl * 86400.0))
        uses_signal = math.tanh(math.log1p(max(0.0, float(entry.uses))))
        total = float(entry.total_pos + entry.total_neg)
        balance = 0.0
        if total > 0:
            balance = math.tanh((entry.total_pos - entry.total_neg) / total)
        features = [
            1.0,
            reward,
            reward * reward,
            confidence,
            uses_signal,
            1.0 if was_dormant else 0.0,
            recency,
            math.tanh(len(phrase.split()) / 4.0),
            1.0 if _EMOJI_RE.search(phrase) else 0.0,
            balance,
        ]
        return features

    def predict(self, features: List[float]) -> float:
        z = sum(w * f for w, f in zip(self.weights, features))
        return self._sigmoid(z)

    def update(self, features: List[float], target: float) -> None:
        pred = self.predict(features)
        error = pred - clamp(target, 0.0, 1.0)
        for i, f in enumerate(features):
            grad = error * f + self.l2 * self.weights[i]
            self.weights[i] -= self.lr * grad
            self.weights[i] = clamp(self.weights[i], -self.clip, self.clip)

    def score(
        self,
        entry: "LexEntry",
        reward: float,
        confidence: float,
        now_ts: float,
        prev_last_ts: Optional[float],
        was_dormant: bool,
    ) -> float:
        features = self._features(entry, reward, confidence, now_ts, prev_last_ts, was_dormant)
        prob = self.predict(features)
        self.update(features, reward)
        return prob


class DiscreteThompsonSampler:
    """Bandit Thompson Sampling discret pour choisir la demi-vie/TTL."""

    def __init__(self, candidates: List[float]):
        self.candidates = list(candidates)
        if not self.candidates:
            raise ValueError("Need at least one TTL candidate")
        self.alpha: List[float] = [1.0 for _ in self.candidates]
        self.beta: List[float] = [1.0 for _ in self.candidates]

    def sample(self) -> Tuple[int, float]:
        best_idx = 0
        best_val = -1.0
        for i, (a, b) in enumerate(zip(self.alpha, self.beta)):
            val = random.betavariate(a, b)
            if val > best_val:
                best_val = val
                best_idx = i
        return best_idx, self.candidates[best_idx]

    def update(self, idx: int, success: bool) -> None:
        if idx < 0 or idx >= len(self.candidates):
            return
        if success:
            self.alpha[idx] += 1.0
        else:
            self.beta[idx] += 1.0

# ----------------- entrées -----------------
@dataclass
class LexEntry:
    phrase: str
    # Beta posteriors (couche ACTIVE)
    alpha_pos: float = 1.0
    beta_pos : float = 1.0
    alpha_neg: float = 1.0
    beta_neg : float = 1.0
    uses: int = 0
    last_ts: float = field(default_factory=_now)
    # Per-user léger
    per_user: Dict[str, Dict[str, float]] = field(default_factory=dict) # {user_id: {"pos":a, "neg":b, "uses":n, "last":ts}}
    tags: List[str] = field(default_factory=list)
    # --- ARCHIVE (stables, sans decay) ---
    total_pos: int = 0
    total_neg: int = 0
    first_seen_ts: float = field(default_factory=_now)
    dormant: bool = False
    ttl_days: float = 60.0
    ttl_arm: int = -1

    def p_pos(self) -> float:
        return self.alpha_pos / (self.alpha_pos + self.beta_pos)

    def p_neg(self) -> float:
        return self.alpha_neg / (self.alpha_neg + self.beta_neg)

# ----------------- archive (structure simple) -----------------
@dataclass
class ArchiveEntry:
    phrase: str
    total_pos: int = 0
    total_neg: int = 0
    uses: int = 0
    first_seen_ts: float = field(default_factory=_now)
    last_seen_ts: float = field(default_factory=_now)

# ----------------- lexique adaptatif -----------------
class AdaptiveLexicon:
    """
    Lexique adaptatif global + par utilisateur.
    - observe_message(...) : apprend depuis les n-grams & emojis avec reward multi-source
    - top_markers(...) : retourne les meilleurs marqueurs ACTIFS (dormants exclus)
    - match(...) : détecte la présence d’un marqueur appris (actif ou dormant) dans un message
    Rétention 2-couches : ACTIVE (decay doux) + ARCHIVE (sans decay, réactivation).
    """

    def __init__(self, arch, path: str = "data/lexicon.json", cfg: Optional[Dict[str,Any]] = None):
        self.arch = arch
        self.path = getattr(arch, "lexicon_path", path)
        self.cfg = cfg or self._default_cfg(getattr(arch, "social_critic_cfg_path", None))
        self.archive_path = self.cfg["lexicon_retention"].get("archive_path", "data/lexicon_archive.json")

        self.entries: Dict[str, LexEntry] = {}
        self.archive: Dict[str, ArchiveEntry] = {}

        retention_cfg = self.cfg.get("lexicon_retention", {})
        default_days = float(retention_cfg.get("dormant_after_days", 60) or 60)
        ttl_candidates = retention_cfg.get("ttl_candidates_days") or [
            max(3.0, default_days / 4.0),
            max(5.0, default_days / 2.0),
            default_days,
            default_days * 1.5,
        ]
        self.ttl_candidates = sorted({float(max(1.0, c)) for c in ttl_candidates})
        self.ttl_bandits: Dict[str, DiscreteThompsonSampler] = {}
        self.calibrator = OnlineLogisticCalibrator(
            lr=float(retention_cfg.get("calibrator_lr", 0.05)),
            l2=float(retention_cfg.get("calibrator_l2", 1e-4)),
            clip=float(retention_cfg.get("calibrator_clip", 6.0)),
        )

        self._load_active()
        self._load_archive()

        # seeds : injecte POS/NEG statiques comme priors (sans figer)
        seeds = getattr(arch, "lexicon_seeds", None)
        if seeds and isinstance(seeds, dict):
            for p in seeds.get("pos", []):
                e = self._ensure_active(p)
                e.alpha_pos += 2.0
            for p in seeds.get("neg", []):
                e = self._ensure_active(p)
                e.alpha_neg += 2.0
        # housekeeping dormant au chargement
        self._refresh_dormant_flags()

    # ------------- config par défaut -------------
    def _default_cfg(self, critic_cfg_path: Optional[str]) -> Dict[str, Any]:
        # essaie de lire social_critic_config.json si présent
        if critic_cfg_path and os.path.exists(critic_cfg_path):
            try:
                data = json.load(open(critic_cfg_path, "r", encoding="utf-8")) or {}
                if "lexicon_retention" in data:
                    return {"lexicon_retention": data["lexicon_retention"]}
            except Exception:
                pass
        # défauts raisonnables
        return {"lexicon_retention": {
            "decay": 0.995,                 # douce priorisation
            "floor_alpha_beta": 1.0,        # plancher (rien n'est effacé)
            "dormant_after_days": 60,       # au-delà → dormant (si pas revu)
            "revive_boost": 0.4,            # boost à la réactivation
            "archive_path": "data/lexicon_archive.json",
            "ttl_candidates_days": [7, 14, 30, 60, 90],
            "calibrator_lr": 0.05,
            "calibrator_l2": 1e-4,
            "calibrator_clip": 6.0,
        }}

    # ------------- I/O -------------
    def _load_active(self):
        try:
            if os.path.exists(self.path):
                raw = json.load(open(self.path, "r", encoding="utf-8")) or {}
                for phrase, d in raw.items():
                    # compat: anciennes versions n’ont pas tous les champs
                    entry = LexEntry(phrase=phrase, **{k: v for k, v in d.items() if k != "phrase"})
                    self._ensure_entry_ttl(entry)
                    self.entries[phrase] = entry
        except Exception:
            self.entries = {}

    def _load_archive(self):
        try:
            if os.path.exists(self.archive_path):
                raw = json.load(open(self.archive_path, "r", encoding="utf-8")) or {}
                for phrase, d in raw.items():
                    self.archive[phrase] = ArchiveEntry(phrase=phrase, **{k:v for k,v in d.items() if k != "phrase"})
        except Exception:
            self.archive = {}

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            data = {k: asdict(v) for k, v in self.entries.items()}
            json.dump(data, open(self.path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        except Exception:
            pass

    def save_archive(self):
        try:
            os.makedirs(os.path.dirname(self.archive_path), exist_ok=True)
            data = {k: asdict(v) for k, v in self.archive.items()}
            json.dump(data, open(self.archive_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ------------- helpers -------------
    def _ensure_active(self, phrase: str) -> LexEntry:
        if phrase not in self.entries:
            entry = LexEntry(phrase=phrase)
            self._assign_ttl(entry)
            self.entries[phrase] = entry
        else:
            entry = self.entries[phrase]
            self._ensure_entry_ttl(entry)
        return entry

    def _ensure_archive(self, phrase: str) -> ArchiveEntry:
        if phrase not in self.archive:
            self.archive[phrase] = ArchiveEntry(phrase=phrase)
        return self.archive[phrase]

    def _marker_family(self, phrase: str, entry: Optional[LexEntry] = None) -> str:
        if entry and entry.tags:
            for tag in entry.tags:
                if tag.startswith("family:"):
                    return tag.split(":", 1)[1]
        if _EMOJI_RE.search(phrase):
            return "emoji"
        length = len(phrase.split())
        if length <= 1:
            return "unigram"
        if length == 2:
            return "bigram"
        return "trigram_plus"

    def _get_bandit(self, family: str) -> DiscreteThompsonSampler:
        if family not in self.ttl_bandits:
            self.ttl_bandits[family] = DiscreteThompsonSampler(self.ttl_candidates)
        return self.ttl_bandits[family]

    def _assign_ttl(self, entry: LexEntry) -> None:
        family = self._marker_family(entry.phrase, entry)
        sampler = self._get_bandit(family)
        arm_idx, ttl = sampler.sample()
        entry.ttl_arm = arm_idx
        entry.ttl_days = float(ttl)

    def _ensure_entry_ttl(self, entry: LexEntry) -> None:
        family = self._marker_family(entry.phrase, entry)
        sampler = self._get_bandit(family)
        if getattr(entry, "ttl_days", None) in (None, 0):
            self._assign_ttl(entry)
            return
        if getattr(entry, "ttl_arm", None) is None or entry.ttl_arm < 0 or entry.ttl_arm >= len(sampler.candidates):
            self._assign_ttl(entry)
            return
        ttl_value = float(entry.ttl_days)
        if ttl_value <= 0 or not any(abs(ttl_value - c) < 1e-6 for c in sampler.candidates):
            self._assign_ttl(entry)

    def _record_ttl_feedback(
        self,
        entry: LexEntry,
        success: bool,
        prev_last_ts: Optional[float] = None,
        now_ts: Optional[float] = None,
    ) -> None:
        if entry.ttl_days is None:
            return
        family = self._marker_family(entry.phrase, entry)
        sampler = self._get_bandit(family)
        if entry.ttl_arm is None or entry.ttl_arm < 0 or entry.ttl_arm >= len(sampler.candidates):
            return
        if success:
            if prev_last_ts is None or now_ts is None:
                return
            ttl_seconds = float(entry.ttl_days) * 86400.0
            if ttl_seconds <= 0:
                return
            if (now_ts - float(prev_last_ts)) < 0.25 * ttl_seconds:
                return
        sampler.update(entry.ttl_arm, success)

    def _refresh_dormant_flags(self):
        now = _now()
        for e in self.entries.values():
            self._ensure_entry_ttl(e)
            days = float(e.ttl_days or self.cfg["lexicon_retention"].get("dormant_after_days", 60))
            if days <= 0:
                continue
            horizon = days * 86400.0
            was_dormant = e.dormant
            e.dormant = (now - float(e.last_ts)) > horizon
            if e.dormant and not was_dormant:
                self._record_ttl_feedback(e, success=False)

    def _llm_extract_markers(
        self,
        user_msg: str,
        reward01: float,
        confidence: float,
        user_id: Optional[str],
        candidate_ngrams: List[str],
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], Optional[float], Optional[Dict[str, Any]]]:
        """Ask the LLM to highlight salient markers present in ``user_msg``.

        Returns the normalized markers along with their metadata, an optional
        reward hint, and the raw response payload.  When the LLM integration is
        disabled or fails, empty defaults are returned so the caller can rely
        purely on the heuristic n-gram extraction.
        """

        if not user_msg:
            return ([], None, None)

        payload = {
            "message": str(user_msg),
            "reward": clamp(float(reward01), 0.0, 1.0),
            "confidence": clamp(float(confidence), 0.0, 1.0),
            "user_id": user_id or "default",
            "candidate_markers": candidate_ngrams[:12],
        }

        response = try_call_llm_dict(
            "social_adaptive_lexicon",
            input_payload=payload,
            logger=LOGGER,
        )
        if not isinstance(response, dict):
            return ([], None, None)

        markers: List[Tuple[str, Dict[str, Any]]] = []
        raw_markers = response.get("markers")
        if isinstance(raw_markers, list):
            for entry in raw_markers:
                if not isinstance(entry, dict):
                    continue
                phrase = _normalize(str(entry.get("phrase") or ""))
                if not phrase or phrase in _STOPWORDS:
                    continue
                markers.append((phrase, entry))

        reward_hint = response.get("reward_hint")
        try:
            reward_hint_value: Optional[float]
            if reward_hint is None:
                reward_hint_value = None
            else:
                reward_hint_value = clamp(float(reward_hint), 0.0, 1.0)
        except (TypeError, ValueError):
            reward_hint_value = None

        try:
            setattr(self.arch, "_lexicon_last_llm_response", response)
        except Exception:
            pass

        return markers, reward_hint_value, response

    @staticmethod
    def _llm_marker_reward(markers: List[Tuple[str, Dict[str, Any]]]) -> Optional[float]:
        if not markers:
            return None
        weight_sum = 0.0
        score_sum = 0.0
        for _phrase, meta in markers:
            polarity = str(meta.get("polarity") or "").strip().lower()
            if polarity not in {"positive", "negative", "neutral", "mixed"}:
                continue
            try:
                confidence = clamp(float(meta.get("confidence", 0.6)), 0.0, 1.0)
            except (TypeError, ValueError):
                confidence = 0.6
            base = 0.5
            if polarity == "positive":
                base = 0.85
            elif polarity == "negative":
                base = 0.15
            elif polarity in {"neutral", "mixed"}:
                base = 0.5
            score_sum += base * confidence
            weight_sum += confidence
        if weight_sum <= 0.0:
            return None
        return clamp(score_sum / weight_sum, 0.0, 1.0)

    def _apply_decay_and_floor(self, e: LexEntry):
        decay = float(self.cfg["lexicon_retention"].get("decay", 0.995))
        floor = float(self.cfg["lexicon_retention"].get("floor_alpha_beta", 1.0))
        e.alpha_pos = floor + (e.alpha_pos - floor) * decay
        e.beta_pos  = floor + (e.beta_pos  - floor) * decay
        e.alpha_neg = floor + (e.alpha_neg - floor) * decay
        e.beta_neg  = floor + (e.beta_neg  - floor) * decay

    def _reactivate_if_resurfaced(self, e: LexEntry):
        if not e.dormant:
            return
        boost = float(self.cfg["lexicon_retention"].get("revive_boost", 0.4))
        # re-sème la couche ACTIVE à partir des totaux d’archive
        e.alpha_pos = max(e.alpha_pos, 1.0 + boost * float(e.total_pos))
        e.alpha_neg = max(e.alpha_neg, 1.0 + 0.2  * float(e.total_neg))
        e.beta_pos  = max(e.beta_pos,  1.0)
        e.beta_neg  = max(e.beta_neg,  1.0)
        e.dormant = False

    # ------------- API principale -------------
    def observe_message(self, user_msg: str, reward01: float, confidence: float = 0.5, user_id: Optional[str]=None):
        """
        Observe un message utilisateur et met à jour:
        - Couche ACTIVE : posteriors Beta (avec decay & plancher)
        - Couche ARCHIVE : totaux (sans decay), last_seen_ts
        - Réactivation si le marqueur était dormant
        """
        s = _normalize(user_msg or "")
        if not s:
            return
        toks = _tokenize(s)
        grams = list(_ngrams(toks, 1, 3))
        # évite unigrams stopwords
        grams = [g for g in grams if not (len(g.split())==1 and g in _STOPWORDS)]

        llm_markers, reward_hint, llm_response = self._llm_extract_markers(
            user_msg,
            reward01,
            confidence,
            user_id,
            grams,
        )
        marker_reward = self._llm_marker_reward(llm_markers)
        llm_confidence = None
        if isinstance(llm_response, dict):
            try:
                llm_confidence = clamp(float(llm_response.get("confidence", 0.0)), 0.0, 1.0)
            except (TypeError, ValueError):
                llm_confidence = None

        merged: List[str] = []
        seen = set()
        for candidate in grams + [phrase for phrase, _ in llm_markers]:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            merged.append(candidate)
        grams = merged

        marker_meta: Dict[str, Dict[str, Any]] = {phrase: meta for phrase, meta in llm_markers}

        now = _now()
        for g in grams:
            e = self._ensure_active(g)
            a = self._ensure_archive(g)

            # réactivation éventuelle si dormant
            was_dormant = e.dormant
            prev_last_ts = e.last_ts
            self._reactivate_if_resurfaced(e)
            self._ensure_entry_ttl(e)

            # decay doux + plancher
            self._apply_decay_and_floor(e)

            # update ACTIVE selon reward
            r = clamp(float(reward01), 0.0, 1.0)
            conf = clamp(float(confidence), 0.0, 1.0)
            if llm_confidence is not None:
                conf = clamp(0.6 * conf + 0.4 * llm_confidence, 0.0, 1.0)
            blended_hint = reward_hint if reward_hint is not None else marker_reward
            if blended_hint is not None:
                r = clamp(0.65 * r + 0.35 * blended_hint, 0.0, 1.0)

            prob = self.calibrator.score(
                entry=e,
                reward=r,
                confidence=conf,
                now_ts=now,
                prev_last_ts=prev_last_ts,
                was_dormant=was_dormant,
            )
            pos_factor = 0.6 + 0.4 * prob
            neg_factor = 0.6 + 0.4 * (1.0 - prob)

            meta = marker_meta.get(g)
            if meta:
                tags = list(e.tags or [])
                if "source:llm" not in tags:
                    tags.append("source:llm")
                polarity = str(meta.get("polarity") or "").strip().lower()
                if polarity:
                    tags = [t for t in tags if not t.startswith("polarity:")]
                    tags.append(f"polarity:{polarity}")
                rationale = str(meta.get("rationale") or "").strip()
                if rationale:
                    short = rationale[:60]
                    if short and f"hint:{short}" not in tags:
                        tags.append(f"hint:{short}")
                e.tags = tags

            if r >= 0.6:
                e.alpha_pos += conf * pos_factor
                e.total_pos += 1  # ARCHIVE: totaux stables
                if not was_dormant:
                    self._record_ttl_feedback(
                        e,
                        success=True,
                        prev_last_ts=prev_last_ts,
                        now_ts=now,
                    )
            elif r <= 0.4:
                e.beta_pos  += conf * neg_factor
                e.alpha_neg += conf * (0.3 + 0.5 * (1.0 - prob))
                e.total_neg += 1  # ARCHIVE
                if was_dormant:
                    self._record_ttl_feedback(e, success=False)
            else:
                # neutre: micro stabilisation côté "neg" pour éviter sur-confiance
                e.beta_neg  += conf * 0.1 * (0.5 + 0.5 * (1.0 - prob))

            e.uses += 1
            e.last_ts = now

            # per-user
            if user_id:
                u = e.per_user.setdefault(user_id, {"pos":1.0,"neg":1.0,"uses":0,"last":now})
                if r >= 0.6:
                    u["pos"] += conf * pos_factor
                elif r <= 0.4:
                    u["neg"] += conf * neg_factor
                u["uses"] += 1; u["last"] = now

            # ARCHIVE : totaux & timestamps sans decay
            a.uses += 1
            a.last_seen_ts = now
            if a.first_seen_ts <= 0:
                a.first_seen_ts = now

        # maj dormant flags (si longue inactivité sur d'autres entrées)
        self._refresh_dormant_flags()
        self.save()
        self.save_archive()

    def top_markers(self, polarity: str = "pos", k: int = 20, user_id: Optional[str] = None) -> List[str]:
        """
        Classement pour usage ACTIF (réutilisation) — on EXCLUT les dormants.
        """
        scored: List[Tuple[float, str]] = []
        for phrase, e in self.entries.items():
            if e.dormant:
                continue  # on n'encourage pas un dormant
            if polarity == "pos":
                p = e.p_pos()
                if user_id and user_id in e.per_user:
                    u = e.per_user[user_id]
                    bonus = (u["pos"] / (u["pos"] + u["neg"])) - 0.5
                    p += 0.15 * bonus
            else:
                p = e.p_neg()
            if e.uses < 3:
                continue
            scored.append((float(p), phrase))
        scored.sort(reverse=True)
        return [ph for _, ph in scored[:k]]

    def match(self, user_msg: str, polarity: str = "pos", user_id: Optional[str]=None) -> bool:
        """
        Détection PASSIVE (pour le Social Critic) — on doit VOIR un marqueur même s'il est dormant.
        => on ignore le flag dormant ici (ne pas rater un "vieux tic" qui revient).
        """
        s = _normalize(user_msg or "")
        if not s:
            return False
        # set des phrases connues (actives + dormantes). L’archive garde les anciennes aussi.
        phrases = set(self.entries.keys()) | set(self.archive.keys())
        if not phrases:
            return False
        # recherche simple par substring (rapide, robuste aux emojis)
        return any(ph in s for ph in phrases)
