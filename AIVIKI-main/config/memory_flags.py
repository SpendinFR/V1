"""Feature flags et poids (tout optionnel). Activer/désactiver sans casser l'existant."""

ENABLE_SALIENCE_SCORER: bool = True
ENABLE_SUMMARIZER: bool = True
ENABLE_RERANKING: bool = True
ENABLE_PREFS_BRIDGE: bool = True


SALIENCE_WEIGHTS = {
    "recency": 0.25,
    "affect": 0.20,
    "reward": 0.15,
    "goal_rel": 0.15,
    "prefs": 0.15,
    "novelty": 0.07,
    "usage": 0.03,
}


# Âges (secondes) pour recency (demi-vies par type)
HALF_LIVES = {
    "default": 3 * 24 * 3600,  # 3 jours
    "interaction": 2 * 24 * 3600,
    "episode": 7 * 24 * 3600,
    "digest.daily": 14 * 24 * 3600,
    "digest.weekly": 30 * 24 * 3600,
    "digest.monthly": 90 * 24 * 3600,
}
