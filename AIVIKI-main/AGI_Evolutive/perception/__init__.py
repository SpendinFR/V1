# perception/__init__.py
"""
Système de Perception Complet de l'AGI Évolutive
Traitement multi-modal des entrées sensorielles et formation de représentations
"""

import hashlib
import logging
import math
import time
from collections import deque
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from AGI_Evolutive.utils.llm_service import try_call_llm_dict


LOGGER = logging.getLogger(__name__)


class OnlineLinear:
    """Simple régression linéaire online avec régularisation."""

    def __init__(self, dimension: int, alpha: float = 1.0, bounds: Optional[Tuple[float, float]] = None):
        self.dimension = dimension
        self.alpha = alpha
        self.bounds = bounds
        self.A = np.eye(dimension) * alpha  # Matrice de corrélation régularisée
        self.b = np.zeros(dimension)

    def predict(self, context: List[float]) -> float:
        context_vec = np.asarray(context, dtype=float)
        if context_vec.shape[0] != self.dimension:
            raise ValueError(f"Contexte de dimension {context_vec.shape[0]} incompatible avec le modèle ({self.dimension}).")

        theta = np.linalg.solve(self.A, self.b)
        prediction = float(theta.dot(context_vec))

        if self.bounds is not None:
            prediction = float(np.clip(prediction, self.bounds[0], self.bounds[1]))

        return prediction

    def update(self, context: List[float], target: float):
        context_vec = np.asarray(context, dtype=float)
        if context_vec.shape[0] != self.dimension:
            raise ValueError(f"Contexte de dimension {context_vec.shape[0]} incompatible avec le modèle ({self.dimension}).")

        self.A += np.outer(context_vec, context_vec)
        self.b += context_vec * float(target)


class DiscreteThompsonSampler:
    """Thompson Sampling discret sur un ensemble fini de paramètres."""

    def __init__(self, candidates: List[float]):
        unique_candidates = sorted(set(float(c) for c in candidates))
        if not unique_candidates:
            raise ValueError("La liste de candidats pour le Thompson Sampling ne peut pas être vide.")

        self.candidates = unique_candidates
        self.alpha: Dict[float, float] = {candidate: 1.0 for candidate in unique_candidates}
        self.beta: Dict[float, float] = {candidate: 1.0 for candidate in unique_candidates}

    def sample(self) -> float:
        best_candidate = self.candidates[0]
        best_score = -np.inf

        for candidate in self.candidates:
            score = np.random.beta(self.alpha[candidate], self.beta[candidate])
            if score > best_score:
                best_candidate = candidate
                best_score = score

        return best_candidate

    def update(self, candidate: float, reward: float):
        if candidate not in self.alpha:
            return

        if reward >= 0:
            self.alpha[candidate] += reward
        else:
            self.beta[candidate] += abs(reward)


@dataclass
class AdaptiveParameter:
    """Paramètre adaptatif combinant GLM online et bandit discret."""

    name: str
    value: float
    bounds: Tuple[float, float]
    max_step: float
    model: OnlineLinear
    bandit: DiscreteThompsonSampler
    history: deque

    def propose_update(self, context: List[float], reward: float) -> float:
        """Calcule une nouvelle valeur candidate bornée et limitée en vitesse."""

        # Mise à jour du modèle linéaire sur la cible actuelle enrichie du signal de récompense
        target = np.clip(self.value + reward, self.bounds[0], self.bounds[1])
        self.model.update(context, target)
        linear_proposal = self.model.predict(context)

        # Echantillonnage Thompson discret pour favoriser l'exploration
        bandit_choice = self.bandit.sample()

        # Combinaison prudente des deux propositions
        blended = 0.6 * linear_proposal + 0.4 * bandit_choice

        # Application des contraintes
        bounded = float(np.clip(blended, self.bounds[0], self.bounds[1]))
        delta = bounded - self.value
        if abs(delta) > self.max_step:
            bounded = self.value + np.sign(delta) * self.max_step

        return float(bounded)

    def commit(self, new_value: float, reward: float):
        self.value = float(np.clip(new_value, self.bounds[0], self.bounds[1]))
        # Mise à jour du bandit autour de la meilleure option discrète proche
        closest_candidate = min(self.bandit.candidates, key=lambda c: abs(c - self.value))
        self.bandit.update(closest_candidate, reward)
        self.history.append(self.value)
        if len(self.history) > 100:
            self.history.popleft()


class AdaptiveParameterManager:
    """Gestion centralisée des paramètres adaptatifs du module de perception."""

    def __init__(self, parameter_store: Dict[str, float]):
        self.parameter_store = parameter_store
        self.parameters: Dict[str, AdaptiveParameter] = {}
        self.last_context: Optional[List[float]] = None

        definitions = {
            "sensitivity_threshold": {
                "bounds": (0.02, 0.5),
                "candidates": [0.05, 0.1, 0.15, 0.25, 0.35],
            },
            "discrimination_threshold": {
                "bounds": (0.01, 0.3),
                "candidates": [0.02, 0.05, 0.1, 0.18, 0.25],
            },
            "integration_window": {
                "bounds": (0.05, 0.5),
                "candidates": [0.05, 0.1, 0.2, 0.35, 0.5],
            },
            "object_persistence": {
                "bounds": (0.5, 5.0),
                "candidates": [0.5, 1.0, 2.0, 3.5, 5.0],
            },
            "change_blindness_threshold": {
                "bounds": (0.05, 0.8),
                "candidates": [0.1, 0.2, 0.4, 0.6, 0.75],
            },
        }

        for name, value in parameter_store.items():
            meta = definitions.get(name)
            if not meta:
                continue

            bounds = meta["bounds"]
            candidates = meta["candidates"]
            dimension = 5  # biais + 4 métriques principales
            model = OnlineLinear(dimension=dimension, alpha=1.0, bounds=bounds)
            bandit = DiscreteThompsonSampler(candidates)
            max_step = (bounds[1] - bounds[0]) * 0.2
            self.parameters[name] = AdaptiveParameter(
                name=name,
                value=float(np.clip(value, bounds[0], bounds[1])),
                bounds=bounds,
                max_step=max_step,
                model=model,
                bandit=bandit,
                history=deque(maxlen=100)
            )
            self.parameter_store[name] = self.parameters[name].value

    def context_vector(self, metrics: Dict[str, float]) -> List[float]:
        context = [
            1.0,
            float(metrics.get("average_salience", 0.0)),
            float(metrics.get("average_confidence", 0.0)),
            float(metrics.get("object_density", 0.0)),
            float(metrics.get("drift_magnitude", 0.0)),
        ]
        self.last_context = context
        return context

    def update(self, metrics: Dict[str, float]):
        if not self.parameters:
            return

        context = self.context_vector(metrics)
        reward = float(metrics.get("reward_signal", 0.0))

        for name, adaptive_param in self.parameters.items():
            proposed_value = adaptive_param.propose_update(context, reward)
            adaptive_param.commit(proposed_value, reward)
            self.parameter_store[name] = adaptive_param.value

    def get_history(self, name: str) -> List[float]:
        param = self.parameters.get(name)
        return list(param.history) if param else []


def _label_components(edge_data: np.ndarray) -> Tuple[np.ndarray, int]:
    """Label connected components in a binary array without heavy dependencies."""
    if edge_data.size == 0:
        return np.zeros_like(edge_data, dtype=int), 0

    mask = edge_data.astype(bool)
    labeled = np.zeros(mask.shape, dtype=int)
    visited = np.zeros(mask.shape, dtype=bool)
    current_label = 0

    if mask.ndim != 2:
        # Fallback for non 2D data: treat everything as one component if any true values exist
        if mask.any():
            labeled = mask.astype(int)
            labeled[labeled > 0] = 1
            return labeled, 1
        return labeled, 0

    neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for coord in np.argwhere(mask):
        r, c = int(coord[0]), int(coord[1])
        if visited[r, c] or not mask[r, c]:
            continue

        current_label += 1
        stack = [(r, c)]

        while stack:
            sr, sc = stack.pop()
            if visited[sr, sc]:
                continue
            visited[sr, sc] = True
            if not mask[sr, sc]:
                continue

            labeled[sr, sc] = current_label

            for dr, dc in neighbours:
                nr, nc = sr + dr, sc + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]:
                    if not visited[nr, nc] and mask[nr, nc]:
                        stack.append((nr, nc))

    return labeled, current_label

class Modality(Enum):
    """Modalités sensorielles"""
    VISUAL = "visuel"
    AUDITORY = "auditif"
    TACTILE = "tactile"
    PROPRIOCEPTIVE = "proprioceptif"
    TEMPORAL = "temporel"

class FeatureType(Enum):
    """Types de caractéristiques perceptives"""
    EDGE = "contour"
    COLOR = "couleur"
    TEXTURE = "texture"
    SHAPE = "forme"
    MOTION = "mouvement"
    PITCH = "hauteur"
    RHYTHM = "rythme"
    PRESSURE = "pression"
    TEMPERATURE = "température"


class StructuralEvolutionManager:
    """Supervise l'évolution lente des structures perceptives."""

    def __init__(self):
        self.performance_window: deque = deque(maxlen=20)
        self.last_innovation_step: int = 0
        self.iteration: int = 0

    def observe(self, metrics: Dict[str, float], perceptual_learning: Dict[str, Any]):
        self.iteration += 1
        confidence = float(metrics.get("average_confidence", 0.0))
        self.performance_window.append(confidence)

        if len(self.performance_window) < self.performance_window.maxlen:
            return

        rolling_mean = float(np.mean(self.performance_window))
        drift = float(metrics.get("drift_magnitude", 0.0))

        # Introduit de nouveaux détecteurs si la confiance moyenne reste faible
        if rolling_mean < 0.55 and (self.iteration - self.last_innovation_step) > 10:
            self._introduce_new_detector(perceptual_learning, drift)
            self.last_innovation_step = self.iteration

        # Allège les détecteurs inutilisés si la confiance est élevée mais stable
        if rolling_mean > 0.8 and drift < 0.05:
            self._prune_detectors(perceptual_learning)

    def _introduce_new_detector(self, perceptual_learning: Dict[str, Any], drift: float):
        detectors = perceptual_learning.setdefault("feature_detectors", {})
        candidate_name = f"evolved_detector_{int(time.time())}"
        if candidate_name in detectors:
            return

        detectors[candidate_name] = {
            "type": "evolved_detector",
            "feature_type": FeatureType.EDGE,
            "sensitivity": 0.6 + min(drift, 0.3),
            "specificity": 0.6,
            "learning_examples": 0,
        }

    def _prune_detectors(self, perceptual_learning: Dict[str, Any]):
        detectors = perceptual_learning.get("feature_detectors", {})
        removable = [name for name, meta in detectors.items()
                     if meta.get("learning_examples", 0) == 0 and meta.get("type") == "learned_detector"]

        for name in removable[:2]:
            detectors.pop(name, None)

@dataclass
class PerceptualObject:
    """Objet perceptif unifié"""
    id: str
    features: Dict[FeatureType, Any]
    modality: Modality
    confidence: float
    spatial_position: Tuple[float, float, float]
    temporal_position: float
    salience: float
    stability: float
    associations: List[str]

@dataclass
class PerceptualScene:
    """Scène perceptive complète"""
    timestamp: float
    objects: List[PerceptualObject]
    background: Dict[str, Any]
    gist: str  # Impression générale de la scène
    attention_focus: Optional[str]
    emotional_tone: float

class PerceptionSystem:
    """
    Système de perception multi-modal inspiré du traitement sensoriel humain
    Transforme les entrées brutes en représentations structurées
    """
    
    def __init__(self, cognitive_architecture=None, memory_system=None):
        self.cognitive_architecture = cognitive_architecture
        self.memory_system = memory_system
        self.creation_time = time.time()

        # --- LIAISONS INTER-MODULES ---
        if self.cognitive_architecture is not None:
            self.reasoning = getattr(self.cognitive_architecture, "reasoning", None)
            self.goals = getattr(self.cognitive_architecture, "goals", None)
            self.emotions = getattr(self.cognitive_architecture, "emotions", None)
            self.world_model = getattr(self.cognitive_architecture, "world_model", None)
            self.metacognition = getattr(self.cognitive_architecture, "metacognition", None)
        try:
            if hasattr(self, "attention_system") and "top_down_guidance" in self.attention_system and hasattr(self, "goals"):
                self.attention_system["top_down_guidance"].connect(self.goals)
        except Exception:
            pass

        
        # === MODULES DE TRAITEMENT PAR MODALITÉ ===
        self.modality_processors = {
            Modality.VISUAL: VisualProcessor(),
            Modality.AUDITORY: AuditoryProcessor(),
            Modality.TACTILE: TactileProcessor(),
            Modality.PROPRIOCEPTIVE: ProprioceptiveProcessor(),
            Modality.TEMPORAL: TemporalProcessor()
        }
        
        # === INTÉGRATION MULTI-MODALE ===
        self.multisensory_integration = {
            "binding_mechanism": CrossModalBinder(),
            "temporal_synchronization": TemporalSync(),
            "spatial_alignment": SpatialAligner(),
            "confidence_calibration": ConfidenceCalibrator()
        }
        
        # === ATTENTION PERCEPTIVE ===
        self.attention_system = {
            "bottom_up_salience": BottomUpSalience(),
            "top_down_guidance": TopDownGuidance(),
            "attention_spotlight": None,
            "inhibition_of_return": InhibitionOfReturn(),
            "attentional_blink": AttentionalBlink()
        }
        
        # === REPRÉSENTATIONS PERCEPTIVES ===
        self.perceptual_representations = {
            "object_files": {},      # Fichiers d'objets persistants
            "scene_representations": [],  # Représentations de scènes
            "feature_maps": {},      # Cartes de caractéristiques
            "gestalt_grouping": GestaltGrouper()  # Regroupements Gestalt
        }
        
        # === APPRENTISSAGE PERCEPTIF ===
        self.perceptual_learning = {
            "feature_detectors": {},
            "categorical_perception": {},
            "perceptual_expertise": {},
            "adaptation_mechanisms": {}
        }
        self.auto_signal_registry: Dict[str, Dict[str, Any]] = {}
        
        # === PARAMÈTRES PERCEPTIFS ===
        self.perceptual_parameters = {
            "sensitivity_threshold": 0.1,
            "discrimination_threshold": 0.05,
            "integration_window": 0.1,  # 100ms pour l'intégration
            "object_persistence": 2.0,  # Persistance des objets
            "change_blindness_threshold": 0.3
        }

        # Gestion adaptative des paramètres perceptifs (niveau rapide)
        self.adaptive_parameters = AdaptiveParameterManager(self.perceptual_parameters)

        # Evolution lente de la structure perceptive (niveau lent)
        self.structural_evolution = StructuralEvolutionManager()
        self._last_scene_metrics: Optional[Dict[str, float]] = None
        self._last_llm_summary: Optional[Dict[str, Any]] = None

        # === HISTORIQUE PERCEPTIF ===
        self.perceptual_history = {
            "recent_scenes": [],
            "object_tracking": {},
            "change_detection": {},
            "prediction_errors": [],
            "parameter_drifts": [],
            "llm_annotations": [],
        }
        
        # === INNÉS PERCEPTIFS ===
        self._initialize_innate_perceptions()
        
        print("👁️ Système de perception initialisé")
    
    def _initialize_innate_perceptions(self):
        """Initialise les capacités perceptives innées"""
        
        # Détecteurs de caractéristiques innés
        innate_detectors = {
            "edge_detector": {
                "type": "feature_detector",
                "modality": Modality.VISUAL,
                "sensitivity": 0.8,
                "specificity": 0.7
            },
            "motion_detector": {
                "type": "feature_detector", 
                "modality": Modality.VISUAL,
                "sensitivity": 0.9,
                "specificity": 0.6
            },
            "pitch_detector": {
                "type": "feature_detector",
                "modality": Modality.AUDITORY,
                "sensitivity": 0.7,
                "specificity": 0.8
            }
        }
        
        self.perceptual_learning["feature_detectors"] = innate_detectors
        
        # Catégories perceptives innées
        innate_categories = {
            "animate_vs_inanimate": {
                "features": ["motion", "contingency", "agency"],
                "discrimination_threshold": 0.6
            },
            "near_vs_far": {
                "features": ["size", "clarity", "parallax"],
                "discrimination_threshold": 0.5
            }
        }
        
        self.perceptual_learning["categorical_perception"] = innate_categories
    
    def process_sensory_input(self, sensory_data: Dict[Modality, Any]) -> PerceptualScene:
        """
        Traite les entrées sensorielles multi-modales et forme une scène perceptive
        """
        processing_start = time.time()
        
        # === PHASE 1: TRAITEMENT PAR MODALITÉ ===
        modality_results = {}
        for modality, data in sensory_data.items():
            if modality in self.modality_processors:
                processor = self.modality_processors[modality]
                modality_results[modality] = processor.process(data)
        
        # === PHASE 2: EXTRACTION DE CARACTÉRISTIQUES ===
        feature_maps = self._extract_features(modality_results)
        
        # === PHASE 3: REGROUPEMENT GESTALT ===
        perceptual_objects = self._gestalt_grouping(feature_maps)
        
        # === PHASE 4: INTÉGRATION MULTI-MODALE ===
        integrated_objects = self._multisensory_integration(perceptual_objects)
        
        # === PHASE 5: CALCUL DE SAillance ===
        salience_map = self._compute_salience_map(integrated_objects, feature_maps)
        
        # === PHASE 6: ATTENTION ET SÉLECTION ===
        attention_focus = self._attention_selection(salience_map, integrated_objects)
        
        # === PHASE 7: FORMATION DE LA SCÈNE ===
        perceptual_scene = self._form_perceptual_scene(
            integrated_objects, 
            attention_focus,
            processing_start
        )
        
        # === PHASE 8: MISE À JOUR DES REPRÉSENTATIONS ===
        self._update_perceptual_representations(perceptual_scene)
        
        # === PHASE 9: APPRENTISSAGE PERCEPTIF ===
        self._perceptual_learning_update(perceptual_scene)

        # === PHASE 10: ADAPTATIONS META-PERCEPTIVES ===
        processing_time = time.time() - processing_start
        self._meta_adaptation_cycle(perceptual_scene, processing_time=processing_time)

        metrics_snapshot = dict(self._last_scene_metrics or {})
        llm_annotation = self._llm_review_scene(perceptual_scene, metrics_snapshot)
        if llm_annotation:
            perceptual_scene.background.setdefault("llm_analysis", llm_annotation)

        print(f"🔄 Scène perceptive formée en {processing_time:.3f}s - {len(integrated_objects)} objets")

        return perceptual_scene
    
    def _extract_features(self, modality_results: Dict[Modality, Any]) -> Dict[Modality, Dict[FeatureType, Any]]:
        """Extrait les caractéristiques de chaque modalité"""
        feature_maps = {}
        
        for modality, result in modality_results.items():
            feature_maps[modality] = {}
            
            # Extraction basée sur le type de modalité
            if modality == Modality.VISUAL:
                visual_features = self._extract_visual_features(result)
                feature_maps[modality].update(visual_features)
            
            elif modality == Modality.AUDITORY:
                auditory_features = self._extract_auditory_features(result)
                feature_maps[modality].update(auditory_features)
            
            elif modality == Modality.TACTILE:
                tactile_features = self._extract_tactile_features(result)
                feature_maps[modality].update(tactile_features)
        
        return feature_maps
    
    def _extract_visual_features(self, visual_data: Any) -> Dict[FeatureType, Any]:
        """Extrait les caractéristiques visuelles"""
        features = {}
        
        # Simulation de détection de contours
        if isinstance(visual_data, np.ndarray) and visual_data.ndim == 2:
            # Traitement d'image simulé
            edges = self._simulate_edge_detection(visual_data)
            features[FeatureType.EDGE] = edges
        
        # Caractéristiques de couleur (simulées)
        features[FeatureType.COLOR] = {
            "dominant_hue": 0.5,
            "saturation": 0.8,
            "brightness": 0.7
        }
        
        # Caractéristiques de texture (simulées)
        features[FeatureType.TEXTURE] = {
            "roughness": 0.3,
            "regularity": 0.6,
            "orientation": 45.0
        }
        
        return features
    
    def _extract_auditory_features(self, auditory_data: Any) -> Dict[FeatureType, Any]:
        """Extrait les caractéristiques auditives"""
        features = {}
        
        # Caractéristiques de hauteur (simulées)
        features[FeatureType.PITCH] = {
            "fundamental_frequency": 440.0,  # La 440Hz
            "harmonic_structure": [440, 880, 1320],
            "pitch_strength": 0.8
        }
        
        # Caractéristiques de rythme (simulées)
        features[FeatureType.RHYTHM] = {
            "tempo": 120.0,  # BPM
            "regularity": 0.9,
            "accent_pattern": [1, 0, 0, 1]
        }
        
        return features
    
    def _extract_tactile_features(self, tactile_data: Any) -> Dict[FeatureType, Any]:
        """Extrait les caractéristiques tactiles"""
        features = {}
        
        # Caractéristiques de pression (simulées)
        features[FeatureType.PRESSURE] = {
            "intensity": 0.5,
            "distribution": "localized",
            "duration": 0.2
        }
        
        # Caractéristiques de température (simulées)
        features[FeatureType.TEMPERATURE] = {
            "value": 30.5,  # °C
            "contrast": 2.0,
            "stability": 0.8
        }
        
        return features
    
    def _simulate_edge_detection(self, image_data: np.ndarray) -> np.ndarray:
        """Simule la détection de contours (version simplifiée)"""
        # Pour une vraie implémentation, utiliser cv2.Canny ou similar
        # Ici, simulation basique
        if image_data.ndim == 2:
            edges = np.zeros_like(image_data)
            edges[1:, 1:] = np.abs(image_data[1:, 1:] - image_data[:-1, :-1])
            return edges > 0.1  # Seuil arbitraire
        return np.array([])
    
    def _gestalt_grouping(self, feature_maps: Dict) -> List[PerceptualObject]:
        """Applique les principes Gestalt pour regrouper les caractéristiques"""
        objects = []
        
        # Regroupement par proximité
        proximity_groups = self._group_by_proximity(feature_maps)
        
        # Regroupement par similarité
        similarity_groups = self._group_by_similarity(proximity_groups)
        
        # Regroupement par bonne continuation
        continuity_groups = self._group_by_continuity(similarity_groups)
        
        # Création des objets perceptifs
        for group_id, group_features in enumerate(continuity_groups):
            perceptual_object = PerceptualObject(
                id=f"obj_{group_id}_{int(time.time()*1000)}",
                features=group_features,
                modality=self._determine_primary_modality(group_features),
                confidence=self._calculate_object_confidence(group_features),
                spatial_position=self._calculate_spatial_center(group_features),
                temporal_position=time.time(),
                salience=self._calculate_object_salience(group_features),
                stability=0.7,  # Stabilité initiale
                associations=[]
            )
            objects.append(perceptual_object)
        
        return objects
    
    def _group_by_proximity(self, feature_maps: Dict) -> List[Dict]:
        """Regroupe les caractéristiques par proximité spatiale"""
        groups = []
        
        # Implémentation simplifiée
        for modality, features in feature_maps.items():
            if modality == Modality.VISUAL:
                # Regroupement spatial basique
                visual_groups = self._group_visual_features(features)
                groups.extend(visual_groups)
        
        return groups
    
    def _group_visual_features(self, visual_features: Dict) -> List[Dict]:
        """Regroupe les caractéristiques visuelles"""
        groups = []
        
        # Simulation de regroupement
        if FeatureType.EDGE in visual_features:
            edge_data = visual_features[FeatureType.EDGE]
            if isinstance(edge_data, np.ndarray):
                # Regroupement basé sur la connectivité des contours
                labeled_array, num_features = _label_components(edge_data)
                
                for i in range(1, num_features + 1):
                    group_features = {
                        FeatureType.EDGE: (labeled_array == i),
                        FeatureType.COLOR: visual_features.get(FeatureType.COLOR, {}),
                        FeatureType.TEXTURE: visual_features.get(FeatureType.TEXTURE, {})
                    }
                    groups.append(group_features)
        
        if not groups:
            # Retourner au moins un groupe avec toutes les caractéristiques
            groups.append(visual_features)
        
        return groups
    
    def _group_by_similarity(self, groups: List[Dict]) -> List[Dict]:
        """Regroupe les caractéristiques par similarité"""
        merged_groups = []
        
        for group in groups:
            # Vérifier la similarité avec les groupes existants
            merged = False
            for existing_group in merged_groups:
                if self._calculate_group_similarity(group, existing_group) > 0.7:
                    # Fusionner les groupes similaires
                    self._merge_feature_groups(existing_group, group)
                    merged = True
                    break
            
            if not merged:
                merged_groups.append(group)
        
        return merged_groups
    
    def _group_by_continuity(self, groups: List[Dict]) -> List[Dict]:
        """Regroupe les caractéristiques par continuité"""
        # Implémentation simplifiée - bonne continuation des contours
        continuous_groups = []
        
        for group in groups:
            if self._check_continuity(group):
                continuous_groups.append(group)
            else:
                # Diviser le groupe s'il manque de continuité
                split_groups = self._split_discontinuous_group(group)
                continuous_groups.extend(split_groups)
        
        return continuous_groups
    
    def _check_continuity(self, group: Dict) -> bool:
        """Vérifie la continuité d'un groupe de caractéristiques"""
        # Vérification simplifiée
        if FeatureType.EDGE in group:
            edge_data = group[FeatureType.EDGE]
            if isinstance(edge_data, np.ndarray):
                # Vérifier la connectivité
                labeled, num_components = _label_components(edge_data)
                return num_components == 1
        
        return True  # Par défaut, supposer la continuité
    
    def _split_discontinuous_group(self, group: Dict) -> List[Dict]:
        """Divise un groupe discontinu en groupes continus"""
        split_groups = []
        
        if FeatureType.EDGE in group:
            edge_data = group[FeatureType.EDGE]
            if isinstance(edge_data, np.ndarray):
                labeled, num_components = _label_components(edge_data)
                
                for i in range(1, num_components + 1):
                    component_mask = (labeled == i)
                    component_features = {
                        FeatureType.EDGE: component_mask,
                        FeatureType.COLOR: group.get(FeatureType.COLOR, {}),
                        FeatureType.TEXTURE: group.get(FeatureType.TEXTURE, {})
                    }
                    split_groups.append(component_features)
        
        return split_groups if split_groups else [group]
    
    def _calculate_group_similarity(self, group1: Dict, group2: Dict) -> float:
        """Calcule la similarité entre deux groupes de caractéristiques"""
        similarity_scores = []
        
        for feature_type in [FeatureType.COLOR, FeatureType.TEXTURE]:
            if feature_type in group1 and feature_type in group2:
                feat1 = group1[feature_type]
                feat2 = group2[feature_type]
                
                if isinstance(feat1, dict) and isinstance(feat2, dict):
                    feature_similarity = self._calculate_feature_similarity(feat1, feat2)
                    similarity_scores.append(feature_similarity)
        
        return np.mean(similarity_scores) if similarity_scores else 0.0
    
    def _calculate_feature_similarity(self, feat1: Dict, feat2: Dict) -> float:
        """Calcule la similarité entre deux ensembles de caractéristiques"""
        common_keys = set(feat1.keys()) & set(feat2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1 = feat1[key]
            val2 = feat2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Similarité numérique
                similarity = 1.0 - min(abs(val1 - val2), 1.0)
                similarities.append(similarity)
            elif isinstance(val1, str) and isinstance(val2, str):
                # Similarité textuelle
                similarity = 1.0 if val1 == val2 else 0.0
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _merge_feature_groups(self, target_group: Dict, source_group: Dict):
        """Fusionne deux groupes de caractéristiques"""
        for feature_type, features in source_group.items():
            if feature_type not in target_group:
                target_group[feature_type] = features
            else:
                # Fusion des caractéristiques existantes
                if isinstance(features, dict) and isinstance(target_group[feature_type], dict):
                    target_group[feature_type].update(features)
    
    def _determine_primary_modality(self, features: Dict) -> Modality:
        """Détermine la modalité principale d'un objet"""
        # Basé sur les caractéristiques présentes
        if any(feature_type in features for feature_type in 
               [FeatureType.EDGE, FeatureType.COLOR, FeatureType.TEXTURE]):
            return Modality.VISUAL
        elif any(feature_type in features for feature_type in 
                 [FeatureType.PITCH, FeatureType.RHYTHM]):
            return Modality.AUDITORY
        elif any(feature_type in features for feature_type in 
                 [FeatureType.PRESSURE, FeatureType.TEMPERATURE]):
            return Modality.TACTILE
        
        return Modality.VISUAL  # Par défaut
    
    def _calculate_object_confidence(self, features: Dict) -> float:
        """Calcule la confiance dans la détection d'un objet"""
        confidence_factors = []
        
        # Nombre de caractéristiques détectées
        feature_count = len(features)
        confidence_factors.append(min(feature_count / 5.0, 1.0))
        
        # Clarté des caractéristiques
        for feature_data in features.values():
            if isinstance(feature_data, dict) and 'confidence' in feature_data:
                confidence_factors.append(feature_data['confidence'])
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _calculate_spatial_center(self, features: Dict) -> Tuple[float, float, float]:
        """Calcule la position spatiale centrale d'un objet"""
        # Simulation - à remplacer par un calcul réel basé sur les données
        return (0.5, 0.5, 0.0)  # Position normalisée (x, y, z)
    
    def _calculate_object_salience(self, features: Dict) -> float:
        """Calcule la saillance d'un objet perceptif"""
        salience_factors = []
        
        # Contraste des caractéristiques
        if FeatureType.COLOR in features:
            color_data = features[FeatureType.COLOR]
            if isinstance(color_data, dict) and 'brightness' in color_data:
                salience_factors.append(color_data['brightness'])
        
        # Mouvement
        if FeatureType.MOTION in features:
            salience_factors.append(0.8)  # Les objets en mouvement sont saillants
        
        # Nouveauté (à intégrer avec la mémoire)
        salience_factors.append(0.6)  # Supposition basique
        
        return np.mean(salience_factors) if salience_factors else 0.5
    
    def _multisensory_integration(self, objects: List[PerceptualObject]) -> List[PerceptualObject]:
        """Intègre les informations multi-modales"""
        integrated_objects = []
        
        # Regroupement des objets par similarité spatiale et temporelle
        for obj in objects:
            # Vérifier s'il peut être intégré avec un objet existant
            integrated = False
            for existing_obj in integrated_objects:
                if self._should_integrate_objects(obj, existing_obj):
                    integrated_obj = self._integrate_two_objects(existing_obj, obj)
                    integrated_objects.remove(existing_obj)
                    integrated_objects.append(integrated_obj)
                    integrated = True
                    break
            
            if not integrated:
                integrated_objects.append(obj)
        
        return integrated_objects
    
    def _should_integrate_objects(self, obj1: PerceptualObject, obj2: PerceptualObject) -> bool:
        """Détermine si deux objets doivent être intégrés"""
        # Vérification de la proximité spatiale
        spatial_distance = self._calculate_spatial_distance(
            obj1.spatial_position, 
            obj2.spatial_position
        )
        
        # Vérification de la proximité temporelle
        temporal_distance = abs(obj1.temporal_position - obj2.temporal_position)
        
        # Vérification de la cohérence des modalités
        modality_compatibility = self._check_modality_compatibility(obj1.modality, obj2.modality)
        
        return (spatial_distance < 0.2 and 
                temporal_distance < self.perceptual_parameters["integration_window"] and
                modality_compatibility > 0.7)
    
    def _calculate_spatial_distance(self, pos1: Tuple, pos2: Tuple) -> float:
        """Calcule la distance spatiale entre deux positions"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))
    
    def _check_modality_compatibility(self, mod1: Modality, mod2: Modality) -> float:
        """Vérifie la compatibilité entre deux modalités"""
        compatibility_matrix = {
            (Modality.VISUAL, Modality.VISUAL): 1.0,
            (Modality.AUDITORY, Modality.AUDITORY): 1.0,
            (Modality.VISUAL, Modality.AUDITORY): 0.8,  # Vision et audition peuvent s'intégrer
            (Modality.VISUAL, Modality.TACTILE): 0.9,   # Vision et toucher fortement liés
            (Modality.AUDITORY, Modality.TACTILE): 0.6   # Audition et toucher moins liés
        }
        
        key = (mod1, mod2) if mod1.value <= mod2.value else (mod2, mod1)
        return compatibility_matrix.get(key, 0.3)
    
    def _integrate_two_objects(self, obj1: PerceptualObject, obj2: PerceptualObject) -> PerceptualObject:
        """Intègre deux objets en un seul"""
        # Fusion des caractéristiques
        merged_features = obj1.features.copy()
        merged_features.update(obj2.features)
        
        # Calcul de la position moyenne
        avg_position = (
            (obj1.spatial_position[0] + obj2.spatial_position[0]) / 2,
            (obj1.spatial_position[1] + obj2.spatial_position[1]) / 2,
            (obj1.spatial_position[2] + obj2.spatial_position[2]) / 2
        )
        
        # Calcul de la confiance moyenne pondérée
        total_confidence = (obj1.confidence + obj2.confidence) / 2
        
        # Détermination de la modalité principale
        primary_modality = obj1.modality if obj1.confidence > obj2.confidence else obj2.modality
        
        return PerceptualObject(
            id=f"integrated_{obj1.id}_{obj2.id}",
            features=merged_features,
            modality=primary_modality,
            confidence=total_confidence,
            spatial_position=avg_position,
            temporal_position=(obj1.temporal_position + obj2.temporal_position) / 2,
            salience=max(obj1.salience, obj2.salience),
            stability=(obj1.stability + obj2.stability) / 2,
            associations=obj1.associations + obj2.associations
        )
    
    def _compute_salience_map(self, objects: List[PerceptualObject], feature_maps: Dict) -> Dict[str, float]:
        """Calcule une carte de saillance pour guider l'attention"""
        salience_map = {}
        
        for obj in objects:
            # Saillance basée sur les caractéristiques de l'objet
            object_salience = obj.salience
            
            # Renforcement pour les objets nouveaux ou importants
            if self.memory_system:
                novelty = self._calculate_novelty(obj)
                object_salience *= (1.0 + novelty)
            
            salience_map[obj.id] = object_salience
        
        return salience_map
    
    def _calculate_novelty(self, obj: PerceptualObject) -> float:
        """Calcule la nouveauté d'un objet (intégration avec la mémoire)"""
        # Simulation basique - à intégrer avec le système de mémoire
        return 0.3  # Supposition
    
    def _attention_selection(self, salience_map: Dict, objects: List[PerceptualObject]) -> Optional[str]:
        """Sélectionne l'objet qui recevra l'attention focale"""
        if not salience_map:
            return None
        
        # Sélection basée sur la saillance
        most_salient_id = max(salience_map.items(), key=lambda x: x[1])[0]
        
        # Vérification du seuil d'attention
        if salience_map[most_salient_id] > self.perceptual_parameters["sensitivity_threshold"]:
            self.attention_system["attention_spotlight"] = most_salient_id
            return most_salient_id
        
        return None
    
    def _form_perceptual_scene(self, objects: List[PerceptualObject], 
                             attention_focus: Optional[str],
                             timestamp: float) -> PerceptualScene:
        """Forme une représentation unifiée de la scène perceptive"""
        
        # Calcul de l'impression générale (gist)
        scene_gist = self._extract_scene_gist(objects)
        
        # Calcul du ton émotionnel
        emotional_tone = self._calculate_emotional_tone(objects)
        
        # Construction de l'arrière-plan
        background = self._construct_background(objects)
        
        return PerceptualScene(
            timestamp=timestamp,
            objects=objects,
            background=background,
            gist=scene_gist,
            attention_focus=attention_focus,
            emotional_tone=emotional_tone
        )
    
    def _extract_scene_gist(self, objects: List[PerceptualObject]) -> str:
        """Extrait l'impression générale de la scène"""
        object_count = len(objects)
        
        if object_count == 0:
            return "scène vide"
        elif object_count == 1:
            return "objet isolé"
        elif object_count < 5:
            return "scène simple"
        else:
            return "scène complexe"
    
    def _calculate_emotional_tone(self, objects: List[PerceptualObject]) -> float:
        """Calcule le ton émotionnel de la scène"""
        if not objects:
            return 0.0
        
        # Basé sur les caractéristiques des objets (couleurs vives, mouvements rapides, etc.)
        valence_scores = []
        
        for obj in objects:
            # Score de valence basé sur les caractéristiques
            if obj.modality == Modality.VISUAL:
                if FeatureType.COLOR in obj.features:
                    color_data = obj.features[FeatureType.COLOR]
                    if isinstance(color_data, dict) and 'brightness' in color_data:
                        valence_scores.append(color_data['brightness'] - 0.5)
            
            # Objets en mouvement peuvent être excitants
            if FeatureType.MOTION in obj.features:
                valence_scores.append(0.3)
        
        return np.mean(valence_scores) if valence_scores else 0.0
    
    def _construct_background(self, objects: List[PerceptualObject]) -> Dict[str, Any]:
        """Construit la représentation d'arrière-plan"""
        return {
            "spatial_extent": self._calculate_spatial_extent(objects),
            "temporal_stability": 0.8,
            "feature_statistics": self._compute_feature_statistics(objects)
        }
    
    def _calculate_spatial_extent(self, objects: List[PerceptualObject]) -> Dict[str, float]:
        """Calcule l'étendue spatiale de la scène"""
        if not objects:
            return {"width": 1.0, "height": 1.0, "depth": 1.0}
        
        x_positions = [obj.spatial_position[0] for obj in objects]
        y_positions = [obj.spatial_position[1] for obj in objects]
        z_positions = [obj.spatial_position[2] for obj in objects]
        
        return {
            "width": max(x_positions) - min(x_positions),
            "height": max(y_positions) - min(y_positions),
            "depth": max(z_positions) - min(z_positions)
        }
    
    def _compute_feature_statistics(self, objects: List[PerceptualObject]) -> Dict[str, Any]:
        """Calcule les statistiques des caractéristiques de la scène"""
        statistics = {
            "object_count": len(objects),
            "modality_distribution": {},
            "average_salience": 0.0,
            "average_confidence": 0.0
        }
        
        if objects:
            # Distribution des modalités
            for obj in objects:
                modality = obj.modality.value
                statistics["modality_distribution"][modality] = \
                    statistics["modality_distribution"].get(modality, 0) + 1
            
            # Moyennes
            statistics["average_salience"] = np.mean([obj.salience for obj in objects])
            statistics["average_confidence"] = np.mean([obj.confidence for obj in objects])
        
        return statistics
    
    def _update_perceptual_representations(self, scene: PerceptualScene):
        """Met à jour les représentations perceptives persistantes"""
        
        # Mise à jour des fichiers d'objets
        for obj in scene.objects:
            if obj.id not in self.perceptual_representations["object_files"]:
                self.perceptual_representations["object_files"][obj.id] = {
                    "first_seen": scene.timestamp,
                    "last_seen": scene.timestamp,
                    "stability_history": [obj.stability],
                    "feature_consistency": 1.0
                }
            else:
                # Mise à jour de l'objet existant
                obj_file = self.perceptual_representations["object_files"][obj.id]
                obj_file["last_seen"] = scene.timestamp
                obj_file["stability_history"].append(obj.stability)
                
                # Calcul de la consistance des caractéristiques
                obj_file["feature_consistency"] = self._calculate_feature_consistency(obj)
        
        # Ajout de la scène à l'historique
        self.perceptual_history["recent_scenes"].append(scene)
        
        # Limite de l'historique
        if len(self.perceptual_history["recent_scenes"]) > 10:
            self.perceptual_history["recent_scenes"].pop(0)
    
    def _calculate_feature_consistency(self, obj: PerceptualObject) -> float:
        """Calcule la consistance des caractéristiques d'un objet"""
        # Simulation basique
        return 0.8
    
    def _perceptual_learning_update(self, scene: PerceptualScene):
        """Met à jour les mécanismes d'apprentissage perceptif"""

        # Apprentissage des détecteurs de caractéristiques
        for obj in scene.objects:
            for feature_type, features in obj.features.items():
                self._update_feature_detector(feature_type, features, obj.confidence)

        # Apprentissage des catégories perceptives
        self._update_categorical_perception(scene.objects)

    def _meta_adaptation_cycle(self, scene: PerceptualScene, processing_time: float):
        """Orchestre les boucles d'adaptation rapide et lente."""

        metrics = self._collect_scene_metrics(scene, processing_time)
        self.adaptive_parameters.update(metrics)
        self.structural_evolution.observe(metrics, self.perceptual_learning)

        drift_entry = {
            "timestamp": scene.timestamp,
            "drift": metrics.get("drift_magnitude", 0.0),
            "parameters": {k: float(v) for k, v in self.perceptual_parameters.items()},
        }
        self.perceptual_history["parameter_drifts"].append(drift_entry)
        if len(self.perceptual_history["parameter_drifts"]) > 50:
            self.perceptual_history["parameter_drifts"].pop(0)

        self._last_scene_metrics = metrics

    def _llm_review_scene(
        self,
        scene: PerceptualScene,
        metrics: Mapping[str, float],
    ) -> Optional[Dict[str, Any]]:
        """Demande au LLM une analyse synthétique de la scène."""

        payload = self._build_llm_payload(scene, metrics)
        if not payload:
            return None

        try:
            response = try_call_llm_dict("perception_module", input_payload=payload)
        except Exception:  # pragma: no cover - prudence
            LOGGER.debug("Analyse LLM du module perception indisponible", exc_info=True)
            return None

        if not isinstance(response, Mapping):
            return None

        summary = dict(response)
        applied = self._apply_llm_recommendations(summary.get("recommended_settings"))
        if applied:
            summary["applied_settings"] = applied

        self._last_llm_summary = summary
        history = self.perceptual_history.get("llm_annotations")
        if isinstance(history, list):
            history.append({"timestamp": scene.timestamp, "analysis": summary})
            if len(history) > 20:
                del history[0]

        return summary

    def _build_llm_payload(
        self,
        scene: PerceptualScene,
        metrics: Mapping[str, float],
    ) -> Dict[str, Any]:
        objects: List[Dict[str, Any]] = []
        for obj in scene.objects[:8]:
            objects.append(
                {
                    "id": obj.id,
                    "modality": obj.modality.value,
                    "confidence": round(float(obj.confidence), 4),
                    "salience": round(float(obj.salience), 4),
                    "stability": round(float(obj.stability), 4),
                    "features": sorted(ft.value for ft in obj.features.keys()),
                }
            )

        metrics_payload = {
            key: round(float(value), 4) for key, value in metrics.items()
            if isinstance(value, (int, float))
        }

        parameters_payload = {
            key: round(float(value), 4)
            for key, value in self.perceptual_parameters.items()
            if isinstance(value, (int, float))
        }

        payload = {
            "timestamp": datetime.fromtimestamp(scene.timestamp).isoformat(),
            "attention_focus": scene.attention_focus,
            "gist": scene.gist,
            "emotional_tone": round(float(scene.emotional_tone), 4),
            "object_count": len(scene.objects),
            "objects": objects,
            "metrics": metrics_payload,
            "parameters": parameters_payload,
        }

        drifts = self.perceptual_history.get("parameter_drifts") or []
        if drifts:
            payload["recent_drifts"] = drifts[-3:]

        return payload

    def _apply_llm_recommendations(self, settings: Any) -> Dict[str, float]:
        """Applique les réglages proposés par le LLM tout en respectant les bornes locales."""

        if not isinstance(settings, Mapping):
            return {}

        applied: Dict[str, float] = {}

        sensitivity = settings.get("sensibility")
        if isinstance(sensitivity, (int, float)):
            param = self.adaptive_parameters.parameters.get("sensitivity_threshold")
            if param:
                bounded = float(np.clip(sensitivity, param.bounds[0], param.bounds[1]))
                param.commit(bounded, reward=0.0)
                self.perceptual_parameters["sensitivity_threshold"] = param.value
                applied["sensitivity_threshold"] = param.value
            else:
                bounded = float(max(0.01, min(1.0, sensitivity)))
                self.perceptual_parameters["sensitivity_threshold"] = bounded
                applied["sensitivity_threshold"] = bounded

        window_seconds = settings.get("window_seconds")
        if isinstance(window_seconds, (int, float)):
            param = self.adaptive_parameters.parameters.get("integration_window")
            if param:
                bounded = float(np.clip(window_seconds, param.bounds[0], param.bounds[1]))
                param.commit(bounded, reward=0.0)
                self.perceptual_parameters["integration_window"] = param.value
                applied["integration_window"] = param.value
            else:
                bounded = float(max(0.01, window_seconds))
                self.perceptual_parameters["integration_window"] = bounded
                applied["integration_window"] = bounded

        return applied

    def _collect_scene_metrics(self, scene: PerceptualScene, processing_time: float) -> Dict[str, float]:
        """Calcule les métriques nécessaires aux mécanismes d'adaptation."""

        feature_stats = self._compute_feature_statistics(scene.objects)
        average_confidence = float(feature_stats.get("average_confidence", 0.0))
        average_salience = float(feature_stats.get("average_salience", 0.0))
        object_count = float(feature_stats.get("object_count", 0))

        previous = self._last_scene_metrics or {
            "average_confidence": average_confidence,
            "average_salience": average_salience,
            "object_count": object_count,
        }

        drift_components = [
            abs(average_confidence - previous.get("average_confidence", average_confidence)),
            abs(average_salience - previous.get("average_salience", average_salience)),
            abs(object_count - previous.get("object_count", object_count)) * 0.05,
        ]
        drift_magnitude = float(np.mean(drift_components))

        prediction_error = max(0.0, 1.0 - average_confidence)
        reward_signal = average_confidence - prediction_error - 0.1 * drift_magnitude

        metrics = {
            "average_confidence": average_confidence,
            "average_salience": average_salience,
            "object_count": object_count,
            "object_density": object_count / max(1.0, processing_time * 10.0),
            "processing_time": float(processing_time),
            "prediction_error": prediction_error,
            "reward_signal": reward_signal,
            "drift_magnitude": drift_magnitude,
        }

        return metrics

    def _update_feature_detector(self, feature_type: FeatureType, features: Any, confidence: float):
        """Met à jour un détecteur de caractéristiques"""
        detector_key = f"{feature_type.value}_detector"

        if detector_key not in self.perceptual_learning["feature_detectors"]:
            # Création d'un nouveau détecteur
            self.perceptual_learning["feature_detectors"][detector_key] = {
                "type": "learned_detector",
                "feature_type": feature_type,
                "sensitivity": confidence,
                "specificity": 0.7,
                "learning_examples": 1
            }
        else:
            # Mise à jour du détecteur existant
            detector = self.perceptual_learning["feature_detectors"][detector_key]
            detector["learning_examples"] += 1
            
            # Ajustement de la sensibilité
            learning_rate = 0.1
            detector["sensitivity"] = (1 - learning_rate) * detector["sensitivity"] + learning_rate * confidence
    
    def _update_categorical_perception(self, objects: List[PerceptualObject]):
        """Met à jour la perception catégorielle"""
        # Apprentissage des catégories basé sur la similarité des objets
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                similarity = self._calculate_object_similarity(obj1, obj2)
                
                if similarity > 0.8:
                    # Les objets similaires renforcent une catégorie
                    category_features = self._extract_category_features(obj1, obj2)
                    self._update_category("learned_category", category_features, similarity)
    
    def _calculate_object_similarity(self, obj1: PerceptualObject, obj2: PerceptualObject) -> float:
        """Calcule la similarité entre deux objets"""
        similarity_scores = []
        
        # Similarité des caractéristiques
        for feature_type in set(obj1.features.keys()) & set(obj2.features.keys()):
            feat_similarity = self._calculate_feature_similarity(
                obj1.features[feature_type],
                obj2.features[feature_type]
            )
            similarity_scores.append(feat_similarity)
        
        # Similarité spatiale
        spatial_similarity = 1.0 - self._calculate_spatial_distance(
            obj1.spatial_position, obj2.spatial_position
        )
        similarity_scores.append(spatial_similarity)
        
        return np.mean(similarity_scores) if similarity_scores else 0.0
    
    def _extract_category_features(self, obj1: PerceptualObject, obj2: PerceptualObject) -> List[str]:
        """Extrait les caractéristiques communes pour former une catégorie"""
        common_features = []
        
        for feature_type in set(obj1.features.keys()) & set(obj2.features.keys()):
            common_features.append(feature_type.value)
        
        return common_features
    
    def _update_category(self, category_name: str, features: List[str], strength: float):
        """Met à jour une catégorie perceptive"""
        if category_name not in self.perceptual_learning["categorical_perception"]:
            self.perceptual_learning["categorical_perception"][category_name] = {
                "features": features,
                "discrimination_threshold": 0.6,
                "learning_strength": strength
            }
        else:
            category = self.perceptual_learning["categorical_perception"][category_name]
            category["learning_strength"] = max(category["learning_strength"], strength)
    
    def get_perception_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du système de perception"""
        return {
            "object_files_count": len(self.perceptual_representations["object_files"]),
            "feature_detectors_count": len(self.perceptual_learning["feature_detectors"]),
            "categories_learned": len(self.perceptual_learning["categorical_perception"]),
            "current_attention": self.attention_system["attention_spotlight"],
            "recent_scenes_count": len(self.perceptual_history["recent_scenes"])
        }

    def on_auto_intention_promoted(
        self,
        event: Mapping[str, Any],
        evaluation: Optional[Mapping[str, Any]] = None,
        self_assessment: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not isinstance(event, Mapping):
            return
        signals = event.get("signals")
        if isinstance(signals, list):
            for signal in signals:
                if not isinstance(signal, Mapping):
                    continue
                name = str(signal.get("name") or "").strip()
                if not name:
                    continue
                metric = str(signal.get("metric") or name)
                detector = {
                    "type": "auto_signal",
                    "metric": metric,
                    "target": float(signal.get("target", 0.6) or 0.6),
                    "source": event.get("action_type"),
                    "description": event.get("description"),
                }
                self.auto_signal_registry[name] = detector
                feature_bank = self.perceptual_learning.setdefault("feature_detectors", {})
                feature_bank.setdefault(
                    name,
                    {
                        "type": "auto_feature",
                        "modality": Modality.TEMPORAL,
                        "sensitivity": max(0.2, min(0.95, float((evaluation or {}).get("alignment", 0.5)) + 0.2)),
                        "specificity": 0.6,
                    },
                )
        trace = {
            "ts": time.time(),
            "action_type": event.get("action_type"),
            "score": (evaluation or {}).get("score"),
            "signals": len(self.auto_signal_registry),
        }
        history = self.perceptual_history.setdefault("auto_intentions", deque(maxlen=120))
        history.append(trace)

# Classes auxiliaires pour les processeurs de modalité
class VisualProcessor:
    """Processeur pour la modalité visuelle"""
    def process(self, visual_data: Any) -> Dict[str, Any]:
        # Simulation du traitement visuel
        return {
            "raw_data": visual_data,
            "processing_stage": "early_vision",
            "features_available": ["edges", "colors", "motion"]
        }

class AuditoryProcessor:
    """Processeur pour la modalité auditive"""
    def process(self, auditory_data: Any) -> Dict[str, Any]:
        return {
            "raw_data": auditory_data,
            "processing_stage": "cochlear_processing",
            "features_available": ["pitch", "timbre", "rhythm"]
        }

class TactileProcessor:
    """Processeur pour la modalité tactile"""
    def process(self, tactile_data: Any) -> Dict[str, Any]:
        return {
            "raw_data": tactile_data,
            "processing_stage": "somatosensory",
            "features_available": ["pressure", "temperature", "vibration"]
        }

class ProprioceptiveProcessor:
    """Processeur pour la proprioception"""
    def process(self, proprio_data: Any) -> Dict[str, Any]:
        return {
            "raw_data": proprio_data,
            "processing_stage": "body_schema",
            "features_available": ["position", "movement", "force"]
        }

class TemporalProcessor:
    """Processeur pour la perception temporelle"""
    def process(self, temporal_data: Any) -> Dict[str, Any]:
        return {
            "raw_data": temporal_data,
            "processing_stage": "temporal_processing",
            "features_available": ["duration", "rhythm", "timing"]
        }

# Classes pour l'intégration multi-modale
class CrossModalBinder:
    """Liaison inter-modale"""
    def bind_features(self, features_dict: Dict) -> List[PerceptualObject]:
        return []

class TemporalSync:
    """Synchronisation temporelle"""
    def synchronize(self, objects: List[PerceptualObject]) -> List[PerceptualObject]:
        return objects

class SpatialAligner:
    """Alignement spatial"""
    def align_positions(self, objects: List[PerceptualObject]) -> List[PerceptualObject]:
        return objects

class ConfidenceCalibrator:
    """Calibration des confiances"""
    def calibrate(self, objects: List[PerceptualObject]) -> List[PerceptualObject]:
        return objects

# Classes pour l'attention
class BottomUpSalience:
    """Saillance bottom-up"""
    def compute_salience(self, features: Dict) -> float:
        return 0.5

class TopDownGuidance:
    """Guidage top-down"""
    def guide_attention(self, goals: List[str], objects: List[PerceptualObject]) -> Optional[str]:
        return None

class InhibitionOfReturn:
    """Inhibition de retour"""
    def should_inhibit(self, object_id: str) -> bool:
        return False

class AttentionalBlink:
    """Clignement attentionnel"""
    def is_blinking(self) -> bool:
        return False

# Classe pour le regroupement Gestalt
class GestaltGrouper:
    """Regroupement selon les principes Gestalt"""
    def group_features(self, features: Dict) -> List[Dict]:
        return [features]

# Test du système de perception
if __name__ == "__main__":
    print("👁️ TEST DU SYSTÈME DE PERCEPTION")
    print("=" * 50)
    
    # Création du système
    perception_system = PerceptionSystem()
    
    # Données sensorielles simulées
    test_sensory_data = {
        Modality.VISUAL: np.random.rand(100, 100),  # Image simulée
        Modality.AUDITORY: {"sound_wave": [0.1, 0.5, -0.2, 0.8]},
        Modality.TACTILE: {"pressure": 0.7, "temperature": 25.0}
    }
    
    print("\n🎯 Traitement des entrées sensorielles en cours")
    perceptual_scene = perception_system.process_sensory_input(test_sensory_data)
    
    print(f"Scène perceptive créée avec {len(perceptual_scene.objects)} objets")
    print(f"Gist de la scène: {perceptual_scene.gist}")
    print(f"Focus attentionnel: {perceptual_scene.attention_focus}")
    print(f"Ton émotionnel: {perceptual_scene.emotional_tone:.2f}")
    
    # Affichage des objets détectés
    print("\n🔍 Objets perceptifs détectés:")
    for i, obj in enumerate(perceptual_scene.objects):
        print(f"  {i+1}. {obj.modality.value} - Confiance: {obj.confidence:.2f} - Saillance: {obj.salience:.2f}")
    
    # Statistiques
    print("\n📊 Statistiques de perception:")
    stats = perception_system.get_perception_stats()
    for key, value in stats.items():
        print(f"  - {key}: {value}")