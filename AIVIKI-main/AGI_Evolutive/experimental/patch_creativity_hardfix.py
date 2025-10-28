import re, sys, io, os

PATH = os.path.join("creativity","__init__.py")
if not os.path.exists(PATH):
    print("[ERR] creativity/__init__.py introuvable.")
    sys.exit(1)

with io.open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

original = src

# --- A. Corriger les lignes corrompues de type self.<expression> ---
# 1) self.(processing_threads["insight_detection"] <noise>) = insight_thread
src = re.sub(
    r'self\.\(processing_threads\["insight_detection"\][^)]*\)\s*=\s*insight_thread',
    'self.processing_threads["insight_detection"] = insight_thread',
    src
)

# 2) self.(creative_processes["current_phase"] <noise>) = "preparation"
src = re.sub(
    r'self\.\(creative_processes\["current_phase"\][^)]*\)\s*=\s*"preparation"',
    'self.creative_processes["current_phase"] = "preparation"',
    src
)

# 3) self.(creative_history["learning_trajectory"] <noise>).append(
src = re.sub(
    r'self\.\(creative_history\["learning_trajectory"\][^)]*\)\.append\(',
    'self.creative_history["learning_trajectory"].append(',
    src
)

# 4) generation_strategies accès
src = re.sub(
    r'self\.\(idea_generation\["generation_strategies"\][^)]*\)\.get\(',
    'self.idea_generation["generation_strategies"].get(',
    src
)

# 5) idea_pool append
src = re.sub(
    r'self\.\(idea_generation\["idea_pool"\][^)]*\)\.append\(',
    'self.idea_generation["idea_pool"].append(',
    src
)

# 6) creative_states["creative_flow"] et ["creative_confidence"]
src = re.sub(
    r'self\.\(creative_states\["creative_flow"\][^)]*\)',
    'self.creative_states["creative_flow"]',
    src
)
src = re.sub(
    r'self\.\(creative_states\["creative_confidence"\][^)]*\)',
    'self.creative_states["creative_confidence"]',
    src
)

# 7) conceptual_blending["blend_history"]
src = re.sub(
    r'self\.\(conceptual_blending\["blend_history"\][^)]*\)\.append\(',
    'self.conceptual_blending["blend_history"].append(',
    src
)
# 8) creative_history["blends_created"]
src = re.sub(
    r'self\.\(creative_history\["blends_created"\][^)]*\)\.append\(',
    'self.creative_history["blends_created"].append(',
    src
)

# 9) conceptual_space["concept_network"] accès
src = re.sub(
    r'self\.\(conceptual_space\["concept_network"\][^)]*\)',
    'self.conceptual_space["concept_network"]',
    src
)

# --- B. Ajouter une normalisation silencieuse et l'appeler ---
guard_func = r'''
def __crea__normalize(self):
    # Garantir les bons types
    if not isinstance(getattr(self, "creative_states", None), dict):
        self.creative_states = {}
    if not isinstance(getattr(self, "creative_history", None), dict):
        self.creative_history = {}
    if not isinstance(getattr(self, "idea_generation", None), dict):
        self.idea_generation = {}
    if not isinstance(getattr(self, "conceptual_blending", None), dict):
        self.conceptual_blending = {}
    if not isinstance(getattr(self, "insight_detection", None), dict):
        self.insight_detection = {}
    if not isinstance(getattr(self, "innovation_engine", None), dict):
        self.innovation_engine = {}
    if not isinstance(getattr(self, "creative_processes", None), dict):
        self.creative_processes = {}
    if not isinstance(getattr(self, "processing_threads", None), dict):
        self.processing_threads = {}
    if not isinstance(getattr(self, "creative_knowledge", None), dict):
        self.creative_knowledge = {}
    if not isinstance(getattr(self, "conceptual_space", None), dict):
        self.conceptual_space = {}

    # Clés attendues
    self.creative_history.setdefault("learning_trajectory", [])
    self.creative_history.setdefault("ideas_generated", [])
    self.creative_history.setdefault("blends_created", [])
    self.creative_history.setdefault("insights_experienced", [])
    self.creative_states.setdefault("creative_flow", 0.3)
    self.creative_states.setdefault("creative_confidence", 0.5)
    self.creative_states.setdefault("inspiration_level", 0.4)

    self.idea_generation.setdefault("idea_pool", [])
    self.idea_generation.setdefault("generation_strategies", {})

    self.conceptual_blending.setdefault("blend_history", [])
    self.conceptual_space.setdefault("concept_network", {})

    # Convertir d'éventuelles chaînes JSON / strings brutes en structures neutres
    # et convertir les idées brutes en objets dict-like si nécessaire
    pool = self.idea_generation.get("idea_pool", [])
    fixed = []
    for it in pool:
        if isinstance(it, str):
            fixed.append({"id": f"idea_fix_{int(time.time())}", "concept_core": it, "domains": [], "components": [], "associations": [], "state": "brute", "novelty": 0.5, "usefulness": 0.5, "feasibility": 0.5, "elaboration": 0.3, "created_time": time.time(), "last_modified": time.time(), "development_history": [] , "emotional_affinity": 0.3, "activation_level": 0.1})
        else:
            fixed.append(it)
    self.idea_generation["idea_pool"] = fixed
'''

# Injecter la fonction si absente
if "__crea__normalize(" not in src:
    # après la classe CreativitySystem: on essaie d'injecter au début de la classe (après __init__)
    # Si pas trivial, on l'ajoutera juste après la déclaration de CreativitySystem
    src = src.replace("class CreativitySystem:", "class CreativitySystem:\n" + guard_func)

# Appeler la normalisation au début des points d'entrée critiques
def add_call(block_name: str, code: str):
    pattern = rf"(def\s+{block_name}\s*\([^\)]*\):\s*\n)"
    if re.search(pattern, code):
        code = re.sub(pattern, r"\1        try:\n            __crea__normalize(self)\n        except Exception:\n            pass\n", code, count=1)
    return code

for fn in ["_start_creative_monitoring","_start_insight_detection","_start_incubation_process",
           "_initiate_first_creative_cycle","generate_ideas","_monitor_creative_state",
           "_process_incubation_phase","_monitor_insight_conditions","_update_conceptual_space"]:
    src = add_call(fn, src)

# Sauvegarde
if src != original:
    with io.open(PATH, "w", encoding="utf-8") as f:
        f.write(src)
    print("[OK] Patch appliqué à creativity/__init__.py")
else:
    print("[INFO] Aucun changement nécessaire (déjà patché ?)")
