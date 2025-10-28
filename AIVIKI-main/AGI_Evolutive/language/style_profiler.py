from dataclasses import asdict, dataclass
from typing import Dict, Any, Iterable, List
from collections import Counter, defaultdict
import logging
import math
import os
import json
import re
import unicodedata
from datetime import datetime

from AGI_Evolutive.utils.jsonsafe import json_sanitize
from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)
LLM_SPEC_KEY = "language_style_profiler"


@dataclass
class UserStyleProfile:
    user_id: str
    avg_sentence_len: float = 14.0
    emoji_rate: float = 0.05
    exclam_rate: float = 0.03
    question_rate: float = 0.04
    formality: float = 0.6
    fav_lexicon: Dict[str, float] = None
    prefers_bullets: bool = False
    uses_caps: float = 0.05
    language: str = "fr"
    samples_seen: int = 0
    personal_names: Dict[str, int] = None
    associative_memory: List[Dict[str, str]] = None


class OnlineTextClassifier:
    """Very small online softmax classifier relying on bag-of-ngrams features."""

    def __init__(self, labels: Iterable[str], lr: float = 0.05):
        self.labels = tuple(labels)
        self.lr = lr
        self.weights = {label: defaultdict(float) for label in self.labels}
        self.bias = {label: 0.0 for label in self.labels}
        self.total_seen = 0

    def _tokenize(self, text: str) -> Iterable[str]:
        return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", text.lower())

    def _features(self, text: str) -> Counter:
        tokens = list(self._tokenize(text))
        feats: Counter = Counter()
        for tok in tokens:
            feats[f"tok:{tok}"] += 1.0
            if len(tok) > 3:
                feats[f"suffix:{tok[-3:]}"] += 1.0
        for i in range(len(tokens) - 1):
            bigram = tokens[i] + "_" + tokens[i + 1]
            feats[f"bi:{bigram}"] += 1.0
        punctuation = Counter(
            {
                "exclam": text.count("!"),
                "question": text.count("?"),
                "ellipsis": text.count("..."),
            }
        )
        feats.update({f"punct:{k}": float(v) for k, v in punctuation.items() if v})
        emoji_count = sum(1 for ch in text if ch in StyleProfiler.EMOJI_CHARS)
        if emoji_count:
            feats["has_emoji"] = float(emoji_count)
        feats["bias"] = 1.0
        return feats

    def predict_proba(self, text: str) -> Dict[str, float]:
        if not self.labels:
            return {}
        feats = self._features(text)
        scores = {}
        for label in self.labels:
            score = self.bias[label]
            weights = self.weights[label]
            for feat, value in feats.items():
                if feat in weights:
                    score += weights[feat] * value
            scores[label] = score
        max_score = max(scores.values()) if scores else 0.0
        exp_scores = {label: math.exp(score - max_score) for label, score in scores.items()}
        total = sum(exp_scores.values()) or 1.0
        return {label: val / total for label, val in exp_scores.items()}

    def update(self, text: str, label: str):
        if label not in self.labels:
            return
        feats = self._features(text)
        probs = self.predict_proba(text)
        for lbl in self.labels:
            error = (1.0 if lbl == label else 0.0) - probs.get(lbl, 0.0)
            for feat, value in feats.items():
                self.weights[lbl][feat] += self.lr * error * value
            self.bias[lbl] += self.lr * error
        self.total_seen += 1

    def state(self) -> Dict[str, Any]:
        return {
            "labels": self.labels,
            "lr": self.lr,
            "weights": {lbl: dict(weights) for lbl, weights in self.weights.items()},
            "bias": dict(self.bias),
            "total_seen": self.total_seen,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]):
        classifier = cls(state.get("labels", ()), lr=state.get("lr", 0.05))
        classifier.weights = {
            lbl: defaultdict(float, weights)
            for lbl, weights in state.get("weights", {}).items()
        }
        classifier.bias = dict(state.get("bias", {})) or {lbl: 0.0 for lbl in classifier.labels}
        classifier.total_seen = state.get("total_seen", 0)
        return classifier


class StyleProfiler:
    EMOJI_CHARS = set(
        list(
            "👍👌👏💯✨🔥😄😁😊🤝❤️💪🤩🙌🌟✅🆗🙂🤗👎😡😠😞😔💢❌🛑🤬😤🙄😒😂🤣😭😅😉😎🤔🤨😍😘😉🥲🥳😇🤖"
        )
    )
    PERSONAL_NAME_PATTERNS = [
        re.compile(
            r"\b(?:je m'appelle|je me nomme|mon prénom est|mon prenom est|appelle[- ]moi|moi c'est)\s+([A-Za-zÀ-ÖØ-öø-ÿ'\-]{2,60})",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bje suis\s+([A-Za-zÀ-ÖØ-öø-ÿ'\-]{2,60})",
            re.IGNORECASE,
        ),
    ]
    NAME_STOPWORDS = {
        "toi",
        "ton",
        "ta",
        "tes",
        "mon",
        "ma",
        "mes",
        "son",
        "sa",
        "ses",
        "notre",
        "votre",
        "leur",
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "ce",
        "cette",
        "cet",
        "nul",
        "créateur",
        "createur",
    }
    MEMORY_PATTERNS_FR = [
        re.compile(r"\bj(?:'aime|e préfère|e prefere|e déteste|e deteste|[' ]?adore)\b", re.IGNORECASE),
        re.compile(r"\bmon\s+[^.?!]*préféré", re.IGNORECASE),
        re.compile(r"\bma\s+[^.?!]*préférée", re.IGNORECASE),
        re.compile(r"\bmes\s+[^.?!]*préférés", re.IGNORECASE),
        re.compile(r"\bje me souviens\b", re.IGNORECASE),
        re.compile(r"\bje me rappelle\b", re.IGNORECASE),
        re.compile(r"\btu m'avais dit\b", re.IGNORECASE),
        re.compile(r"\bil y a\s+\d+\s+(?:jours?|semaines?|mois|ans)\b", re.IGNORECASE),
    ]
    FAMILIAR_MARKERS_FR = {
        "wesh",
        "frerot",
        "frérot",
        "mdr",
        "ptdr",
        "ouais",
        "tkt",
        "dsl",
        "bg",
        "lol",
        "salut",
        "coucou",
    }
    FORMAL_MARKERS_FR = {
        "cependant",
        "toutefois",
        "ainsi",
        "par conséquent",
        "par consequent",
        "tandis que",
        "néanmoins",
        "neanmoins",
        "cordialement",
        "madame",
        "monsieur",
    }
    EN_MARKERS = {
        "the",
        "and",
        "is",
        "are",
        "you",
        "i",
        "but",
        "however",
        "therefore",
        "gonna",
        "wanna",
        "though",
        "although",
    }
    FR_LANGUAGE_REGEX = re.compile(r"\best\s+(?:un|une|le|la|l['’])", re.IGNORECASE)
    BULLET_PATTERN = re.compile(r"^\s*(?:[-•*]|\d+[.)])\s+", re.MULTILINE)

    def __init__(self, persist_path: str = "data/style_profiles.json"):
        self.persist_path = persist_path
        directory = os.path.dirname(self.persist_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.profiles: Dict[str, UserStyleProfile] = {}
        self.global_doc_freq: Counter = Counter()
        self.total_samples = 0
        self.formality_classifier = OnlineTextClassifier(labels=("formal", "familiar"))
        self._load()

    def observe(self, user_id: str, text: str):
        p = self.profiles.get(user_id) or UserStyleProfile(user_id=user_id, fav_lexicon={})
        p = self._ensure_profile_defaults(p)
        tokens = self._simple_tokens(text)
        token_counts = Counter(tok.lower() for tok in tokens if len(tok) > 1)
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_len = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        emojis = [ch for ch in text if ch in self.EMOJI_CHARS]
        exclam = text.count("!")
        quest = text.count("?")
        caps_ratio = self._caps_ratio(text)

        lang = self._detect_language(text, tokens)

        fam = sum(1 for t in tokens if t.lower() in self.FAMILIAR_MARKERS_FR)
        formal = sum(1 for t in tokens if t.lower() in self.FORMAL_MARKERS_FR)
        formality_delta = (formal - fam) * 0.02

        classifier_adjustment = 0.0
        confident_label = None
        if abs(formal - fam) >= 2:
            confident_label = "formal" if formal > fam else "familiar"
        elif self.formality_classifier.total_seen > 10:
            proba = self.formality_classifier.predict_proba(text)
            classifier_adjustment = (proba.get("formal", 0.5) - 0.5) * 0.2

        if classifier_adjustment:
            formality_delta += classifier_adjustment

        emoji_ratio = len(emojis) / max(1, len(tokens)) if tokens else 0.0

        for tok in token_counts.keys():
            self.global_doc_freq[tok] += 1

        self.total_samples += 1

        if p.fav_lexicon is None:
            p.fav_lexicon = {}

        fav_lex = {}
        for tok, freq in token_counts.items():
            tf = freq / max(1, len(tokens))
            idf = math.log((1 + self.total_samples) / (1 + self.global_doc_freq.get(tok, 0))) + 1.0
            previous = p.fav_lexicon.get(tok, 0.0) * 0.85
            fav_lex[tok] = previous + tf * idf

        p.fav_lexicon.update(fav_lex)
        if len(p.fav_lexicon) > 60:
            top_tokens = dict(sorted(p.fav_lexicon.items(), key=lambda x: x[1], reverse=True)[:60])
            p.fav_lexicon = top_tokens

        n = p.samples_seen
        p.avg_sentence_len = (p.avg_sentence_len * n + avg_len) / (n + 1)
        p.emoji_rate = (p.emoji_rate * n + emoji_ratio) / (n + 1)
        p.exclam_rate = (p.exclam_rate * n + (exclam / max(1, len(text)))) / (n + 1)
        p.question_rate = (p.question_rate * n + (quest / max(1, len(text)))) / (n + 1)
        p.uses_caps = (p.uses_caps * n + caps_ratio) / (n + 1)
        p.formality = min(1.0, max(0.0, p.formality + formality_delta))
        p.language = lang
        p.samples_seen = n + 1

        p.prefers_bullets = p.prefers_bullets or bool(self.BULLET_PATTERN.search(text))

        self._update_personal_context(p, text)

        llm_update = self._llm_analyse(
            user_id,
            text,
            profile=p,
            heuristics={
                "avg_sentence_len": avg_len,
                "emoji_ratio": emoji_ratio,
                "exclam_density": exclam / max(1, len(text)),
                "question_density": quest / max(1, len(text)),
                "formality_score": p.formality,
                "language": lang,
            },
        )
        if llm_update:
            self._apply_llm_update(p, llm_update)

        self.profiles[user_id] = p

        if confident_label:
            self.formality_classifier.update(text, confident_label)

        self._save()

    def style_of(self, user_id: str) -> UserStyleProfile:
        profile = self.profiles.get(user_id) or UserStyleProfile(user_id=user_id, fav_lexicon={})
        return self._ensure_profile_defaults(profile)

    def rewrite_to_match(self, base_text: str, user_id: str) -> str:
        p = self.style_of(user_id)

        if p.language == "fr":
            base_text = self._adjust_formality_fr(base_text, p.formality)

        base_text = self._apply_personal_address(base_text, p)
        base_text = self._shape_punctuation(base_text, p)

        if p.emoji_rate > 0.01:
            base_text = self._sprinkle_emojis(base_text, p)

        base_text = self._append_associative_memory(base_text, p)
        base_text = self._shape_sentence_length(base_text, p)

        if p.uses_caps > 0.12:
            base_text = self._emphasize_some_words(base_text)

        return base_text

    def _ensure_profile_defaults(self, profile: UserStyleProfile) -> UserStyleProfile:
        if profile.fav_lexicon is None:
            profile.fav_lexicon = {}
        if profile.personal_names is None:
            profile.personal_names = {}
        if profile.associative_memory is None:
            profile.associative_memory = []
        else:
            cleaned: List[Dict[str, str]] = []
            for entry in profile.associative_memory:
                if isinstance(entry, dict) and "text" in entry:
                    cleaned.append({
                        "text": entry.get("text", ""),
                        "timestamp": entry.get("timestamp", ""),
                    })
                elif isinstance(entry, str):
                    cleaned.append({"text": entry, "timestamp": ""})
            profile.associative_memory = [
                item for item in cleaned if item.get("text")
            ]
        return profile

    def _save(self):
        try:
            with open(self.persist_path, "w", encoding="utf-8") as f:
                data = {
                    "profiles": {uid: asdict(p) for uid, p in self.profiles.items()},
                    "_meta": {
                        "global_doc_freq": dict(self.global_doc_freq),
                        "total_samples": self.total_samples,
                        "formality_classifier": self.formality_classifier.state(),
                    },
                }
                json.dump(json_sanitize(data), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _llm_analyse(
        self,
        user_id: str,
        text: str,
        *,
        profile: UserStyleProfile,
        heuristics: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = {
            "user_id": user_id,
            "text": text[-4000:],
            "language": profile.language,
            "samples_seen": profile.samples_seen,
            "heuristics": heuristics,
            "recent_preferences": sorted(
                list((profile.fav_lexicon or {}).items()),
                key=lambda item: item[1],
                reverse=True,
            )[:10],
        }
        response = try_call_llm_dict(LLM_SPEC_KEY, input_payload=payload, logger=LOGGER)
        if not isinstance(response, dict):
            return {}
        return response

    def _apply_llm_update(self, profile: UserStyleProfile, llm_data: Dict[str, Any]) -> None:
        tone = llm_data.get("tone")
        if isinstance(tone, str):
            tone_lower = tone.lower()
            if "formel" in tone_lower or "formal" in tone_lower:
                profile.formality = min(1.0, profile.formality + 0.08)
            elif any(key in tone_lower for key in ("casual", "familial", "décontracté", "relaxed")):
                profile.formality = max(0.0, profile.formality - 0.08)

        preferences = llm_data.get("preferences")
        if isinstance(preferences, list):
            for pref in preferences:
                if not isinstance(pref, dict):
                    continue
                trait = str(pref.get("trait", "")).lower()
                try:
                    strength = float(pref.get("strength", 0.5))
                except (TypeError, ValueError):
                    strength = 0.5
                if trait in {"emoji", "emoji_usage"}:
                    profile.emoji_rate = max(
                        0.0, min(0.5, profile.emoji_rate + (strength - 0.5) * 0.2)
                    )
                elif trait in {"exclamation", "exclamations"}:
                    profile.exclam_rate = max(
                        0.0, min(0.3, profile.exclam_rate + (strength - 0.5) * 0.15)
                    )
                elif trait in {"questions", "question"}:
                    profile.question_rate = max(
                        0.0, min(0.3, profile.question_rate + (strength - 0.5) * 0.15)
                    )
                elif trait in {"caps", "uppercase"}:
                    profile.uses_caps = max(
                        0.0, min(0.6, profile.uses_caps + (strength - 0.5) * 0.2)
                    )
                elif trait in {"bullet_lists", "bullets", "listes"} and strength >= 0.55:
                    profile.prefers_bullets = True

        lexicon_suggestions = llm_data.get("lexicon" )
        if isinstance(lexicon_suggestions, list):
            for suggestion in lexicon_suggestions:
                if isinstance(suggestion, dict):
                    token = suggestion.get("token")
                    weight = suggestion.get("weight", 0.0)
                else:
                    token = suggestion
                    weight = 0.1
                if not token:
                    continue
                try:
                    weight_val = float(weight)
                except (TypeError, ValueError):
                    weight_val = 0.1
                profile.fav_lexicon[token] = profile.fav_lexicon.get(token, 0.0) + weight_val

        personal_facts = llm_data.get("personal_facts")
        if isinstance(personal_facts, list):
            for fact in personal_facts:
                if isinstance(fact, dict):
                    summary = fact.get("summary")
                    confidence = fact.get("confidence")
                else:
                    summary = fact
                    confidence = None
                if not summary:
                    continue
                entry = {
                    "text": str(summary),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
                if confidence is not None:
                    entry["confidence"] = float(confidence)
                profile.associative_memory.append(entry)

        detected_names = llm_data.get("names")
        if isinstance(detected_names, list):
            for name_info in detected_names:
                if isinstance(name_info, dict):
                    name = name_info.get("name") or name_info.get("text")
                    count = name_info.get("count", 1)
                else:
                    name = name_info
                    count = 1
                if not name:
                    continue
                name_norm = unicodedata.normalize("NFKD", str(name)).strip()
                if not name_norm:
                    continue
                try:
                    increment = int(count)
                except (TypeError, ValueError):
                    increment = 1
                profile.personal_names[name_norm] = profile.personal_names.get(name_norm, 0) + increment

    def _load(self):
        if not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                profiles = data
                meta = {}
                if "profiles" in data:
                    profiles = data.get("profiles", {})
                    meta = data.get("_meta", {})
                for uid, d in profiles.items():
                    self.profiles[uid] = UserStyleProfile(**d)
                self.global_doc_freq = Counter(meta.get("global_doc_freq", {}))
                self.total_samples = meta.get("total_samples", len(self.profiles))
                clf_state = meta.get("formality_classifier")
                if clf_state:
                    self.formality_classifier = OnlineTextClassifier.from_state(clf_state)
        except Exception:
            self.profiles = {}
            self.global_doc_freq = Counter()
            self.total_samples = 0
            self.formality_classifier = OnlineTextClassifier(labels=("formal", "familiar"))

    def _simple_tokens(self, text: str):
        normalized = unicodedata.normalize("NFKC", text)
        return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", normalized)

    def _update_personal_context(self, profile: UserStyleProfile, text: str):
        names = self._extract_personal_names(text)
        if names:
            for name in names:
                profile.personal_names[name] = profile.personal_names.get(name, 0) + 1
        statements = self._extract_associative_statements(text)
        if statements:
            timestamp = datetime.utcnow().isoformat()
            existing = {
                entry.get("text")
                for entry in profile.associative_memory
                if isinstance(entry, dict)
            }
            for sentence in statements:
                if sentence not in existing:
                    profile.associative_memory.append({
                        "text": sentence,
                        "timestamp": timestamp,
                    })
            if len(profile.associative_memory) > 30:
                profile.associative_memory = profile.associative_memory[-30:]

    def _extract_personal_names(self, text: str) -> List[str]:
        candidates: List[str] = []
        for pattern in self.PERSONAL_NAME_PATTERNS:
            for match in pattern.finditer(text):
                raw = match.group(1).strip()
                if not raw:
                    continue
                token = re.split(r"[\s,.;:!?]", raw)[0].strip("'\" ")
                if not token:
                    continue
                if not re.match(r"^[A-Za-zÀ-ÖØ-öø-ÿ'\-]+$", token):
                    continue
                lowered = token.lower()
                if lowered in self.NAME_STOPWORDS:
                    continue
                normalized = token.title()
                candidates.append(normalized)
        unique: Dict[str, None] = {}
        for name in candidates:
            unique[name] = None
        return list(unique.keys())

    def _extract_associative_statements(self, text: str) -> List[str]:
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        selected: List[str] = []
        for sentence in sentences:
            for pattern in self.MEMORY_PATTERNS_FR:
                if pattern.search(sentence):
                    selected.append(sentence)
                    break
        return selected

    def _apply_personal_address(self, text: str, profile: UserStyleProfile) -> str:
        if not profile.personal_names:
            return text
        name = max(profile.personal_names.items(), key=lambda item: item[1])[0]
        if not name:
            return text
        if name.lower() in text.lower():
            return text
        stripped = text.lstrip()
        leading = text[: len(text) - len(stripped)]
        lowered = stripped.lower()
        greetings = ("salut", "bonjour", "coucou", "hey")
        if lowered.startswith(greetings):
            parts = stripped.split(" ", 1)
            if len(parts) == 2:
                greeting, rest = parts
                rest = rest.lstrip(", ")
                if not rest.lower().startswith(name.lower()):
                    return f"{leading}{greeting} {name}, {rest}"
            return text
        if profile.language == "fr":
            greeting = "Salut" if profile.formality < 0.6 else "Bonjour"
        else:
            greeting = "Hey" if profile.formality < 0.6 else "Hello"
        if stripped:
            return f"{leading}{greeting} {name}, {stripped}"
        return f"{leading}{greeting} {name}"

    def _append_associative_memory(self, text: str, profile: UserStyleProfile) -> str:
        memories = [
            entry
            for entry in profile.associative_memory
            if isinstance(entry, dict) and entry.get("text")
        ]
        if not memories:
            return text
        def memory_score(entry: Dict[str, str]) -> tuple:
            text_lower = entry.get("text", "").lower()
            score = 0
            if any(keyword in text_lower for keyword in ("adore", "aime", "préf", "prefe")):
                score += 5
            if "tu m'avais dit" in text_lower:
                score += 3
            if "il y a" in text_lower:
                score += 1
            return (score, entry.get("timestamp", ""))

        memory = max(memories, key=memory_score)
        memory_text = memory.get("text", "").strip()
        if not memory_text:
            return text
        if memory_text.lower() in text.lower():
            return text
        if len(text) > 220:
            return text
        if profile.language == "fr":
            addition = f" Je me souviens que tu m'avais dit : \"{memory_text}\"."
        else:
            addition = f" I remember you told me: \"{memory_text}\"."
        return text.rstrip() + addition

    def _caps_ratio(self, text: str) -> float:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0.0
        upper = sum(1 for c in letters if c.isupper())
        return upper / max(1, len(letters))

    def _detect_language(self, text: str, tokens: Iterable[str]) -> str:
        lower_text = text.lower()
        en_hits = sum(1 for t in tokens if t.lower() in self.EN_MARKERS)
        fr_hits = sum(1 for t in tokens if t.lower() in self.FORMAL_MARKERS_FR)
        if self.FR_LANGUAGE_REGEX.search(lower_text):
            fr_hits += 2
        if en_hits > fr_hits + 1:
            return "en"
        return "fr"

    def _adjust_formality_fr(self, txt: str, formality: float) -> str:
        replacements_familiar = {
            "bonjour": "salut",
            "je vais": "j'vais",
            "je ne": "j'",
            "ne pas": "pas",
        }
        replacements_formal = {
            "salut": "bonjour",
            "ok": "d'accord",
            "ouais": "oui",
            "t'inquiète": "ne vous en faites pas",
        }
        if formality < 0.45:
            for k, v in replacements_familiar.items():
                txt = re.sub(rf"\b{k}\b", v, txt, flags=re.I)
        elif formality > 0.65:
            for k, v in replacements_formal.items():
                txt = re.sub(rf"\b{k}\b", v, txt, flags=re.I)
        return txt

    def _shape_punctuation(self, txt: str, p: UserStyleProfile) -> str:
        if p.exclam_rate > 0.01 and not txt.strip().endswith("!"):
            txt = txt.rstrip(". ") + "!"
        if p.question_rate > 0.02 and not txt.strip().endswith("?"):
            if len(txt) < 140:
                txt += " (tu vois ?)" if p.language == "fr" else " (you see?)"
        return txt

    def _sprinkle_emojis(self, txt: str, p: UserStyleProfile) -> str:
        emojis_pos = ["🙂", "😊", "🔥", "✨", "💯", "👌", "👍", "🤝", "🙌"]
        if p.emoji_rate > 0.04 and len(txt) < 200:
            return txt + " " + emojis_pos[0]
        if p.emoji_rate > 0.08 and len(txt) < 220:
            return txt + " " + emojis_pos[min(2, len(emojis_pos) - 1)]
        return txt

    def _shape_sentence_length(self, txt: str, p: UserStyleProfile) -> str:
        sents = re.split(r"(?<=[.!?])\s+", txt.strip())
        if not sents:
            return txt
        sents = [s if len(s.split()) <= 28 else self._split_long(s) for s in sents]
        return " ".join(sents)

    def _split_long(self, s: str) -> str:
        words = s.split()
        if len(words) <= 1:
            return s
        mid = len(words) // 2
        return " ".join(words[:mid]) + ". " + " ".join(words[mid:])

    def _emphasize_some_words(self, txt: str) -> str:
        tokens = self._simple_tokens(txt)
        if len(tokens) < 6:
            return txt
        idxs = [i for i, t in enumerate(tokens) if len(t) > 4]
        if not idxs:
            return txt
        for i in idxs[:2]:
            tokens[i] = tokens[i].upper()
        return " ".join(tokens)
