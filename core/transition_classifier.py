"""
CORAL ASTROBIO — Capa 2: Planetary Transition State Classifier (PTSC)
Clasificador de estados transicionales de complejidad planetaria.

Modelo de 4 transiciones universales (Copilot/Kimi):
T0→T1: Química compleja → Prebiótica
T1→T2: Química prebiótica → Metabolismo primitivo
T2→T3: Metabolismo → Ecosistema activo
T3→T4: Ecosistema → Tecnosfera

No busca vida como estado binario sino como trayectoria.
Reduce falsos negativos al no depender de O2.

Basado en:
- copilot_resumen_enfoque_transicional_astrobiologia
- herramienta_transicionales_ml_deteccion_estados_intermedios (Kimi)
- firmas_transicionales_no_moleculares (Copilot)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple


TRANSITION_DEFINITIONS = [
    {
        "id": "T0_T1",
        "label": "T0→T1",
        "name": "Química Compleja → Prebiótica",
        "description": (
            "Moléculas orgánicas no aleatorias, gradientes energéticos estables, "
            "ciclos redox activos. Detectable sin biofirmas clásicas."
        ),
        "key_indicators": ["h2o", "ch4", "surface"],
        "color": "#5a6d8a",
    },
    {
        "id": "T1_T2",
        "label": "T1→T2",
        "name": "Prebiótica → Metabolismo Primitivo",
        "description": (
            "Ciclos autocatalíticos, desequilibrios químicos persistentes, "
            "producción estable de gases reductores/oxidantes."
        ),
        "key_indicators": ["ch4", "seasonal", "o2proxy"],
        "color": "#7b8fff",
    },
    {
        "id": "T2_T3",
        "label": "T2→T3",
        "name": "Metabolismo → Ecosistema Activo",
        "description": (
            "Variabilidad estacional coherente, pigmentos no minerales, "
            "ritmos de crecimiento y decadencia biológica. Independiente de O2."
        ),
        "key_indicators": ["o2proxy", "seasonal", "dms", "surface"],
        "color": "#a8e6cf",
    },
    {
        "id": "T3_T4",
        "label": "T3→T4",
        "name": "Ecosistema → Tecnosfera",
        "description": (
            "Contaminantes industriales (CFCs), emisiones térmicas no naturales, "
            "alteraciones planetarias deliberadas. Solo siglos de persistencia."
        ),
        "key_indicators": ["techno", "o2proxy"],
        "color": "#ffd166",
    },
]


@dataclass
class TransitionResult:
    """Resultado del clasificador PTSC."""
    planet_name: str
    scores: List[float]               # P(en transición Ti) para i=0..3
    dominant_idx: int                 # Índice de la transición dominante
    dominant_score: float
    dominant_label: str
    dominant_name: str
    continuous_state: float           # Estado continuo [0, 4]
    false_negative_reduction: float   # % reducción vs paradigma O2
    detectability_window_ma: float    # Ventana temporal estimada (Ma)
    uncertainty: float                # Incertidumbre del clasificador


def classify_transitions(
    evidences: Dict[str, float],
    planet_name: str = "Desconocido",
) -> TransitionResult:
    """
    Clasifica el estado transicional de un planeta.

    Los scores son probabilidades independientes (no suman 1)
    porque un planeta puede estar simultáneamente en múltiples
    transiciones (pensamiento de gradiente, no binario).

    Args:
        evidences: Dict con evidencias observacionales [0,1]
        planet_name: Nombre del planeta

    Returns:
        TransitionResult con clasificación completa
    """
    e = evidences

    # T0→T1: Química Compleja → Prebiótica
    # Impulsada por: agua, complejidad química, actividad superficial
    # Suprimida por: vulcanismo intenso que destruye moléculas
    s0 = np.clip(
        0.38
        + e.get("h2o", 0) * 0.20
        + e.get("ch4", 0) * 0.15
        + e.get("surface", 0) * 0.10
        - e.get("volcanic", 0) * 0.04,
        0.0, 0.99
    )

    # T1→T2: Prebiótica → Metabolismo
    # Impulsada por: CH4 (metabolismo anoxigénico), variación estacional
    # Requiere energía estable → suprimida por vulcanismo caótico
    s1 = np.clip(
        0.18
        + e.get("ch4", 0) * 0.26
        + e.get("seasonal", 0) * 0.20
        + e.get("o2proxy", 0) * 0.10
        - e.get("volcanic", 0) * 0.10,
        0.0, 0.99
    )

    # T2→T3: Metabolismo → Ecosistema
    # Firma dinámica: borde rojo + variación estacional + DMS
    # Independiente de O2 (detecta biosferas anoxigénicas)
    s2 = np.clip(
        0.08
        + e.get("o2proxy", 0) * 0.32
        + e.get("seasonal", 0) * 0.30
        + e.get("dms", 0) * 0.20
        + e.get("surface", 0) * 0.10,
        0.0, 0.99
    )

    # T3→T4: Ecosistema → Tecnosfera
    # Solo detectable por tecnosignaturas (CFCs duran siglos, no milenios)
    # Por tanto: implica civilización activa o muy reciente
    s3 = np.clip(
        0.01
        + e.get("techno", 0) * 0.82
        + e.get("o2proxy", 0) * 0.04,
        0.0, 0.99
    )

    scores = [float(s0), float(s1), float(s2), float(s3)]
    dominant_idx = int(np.argmax(scores))
    dominant_score = scores[dominant_idx]

    # Estado continuo: interpolación ponderada
    weights = np.array(scores)
    indices = np.array([0.5, 1.5, 2.5, 3.5])  # Punto medio de cada transición
    continuous = float(np.average(indices, weights=weights + 0.01))

    # Reducción de falsos negativos vs paradigma O2 clásico
    # Si el ecosistema tiene vida no oxigénica (seasonal alto, surface alto, o2proxy bajo)
    # el paradigma clásico lo perdería. Nuestro enfoque lo captura.
    o2_paradigm_miss = max(0.0,
        (e.get("seasonal", 0) * 0.4 + e.get("surface", 0) * 0.3
         - e.get("o2proxy", 0) * 0.5) * dominant_score
    )
    fn_reduction = np.clip(o2_paradigm_miss * 80 + dominant_score * 15, 0, 60)

    # Ventana de detectabilidad (Ma) — depende del estado
    # Tecnosfera: muy corta (siglos); Ecosistema: larga (>1 Ga)
    if dominant_idx == 3:
        window = 0.001  # ~1000 años
    elif dominant_idx == 2:
        window = 400 + dominant_score * 800
    elif dominant_idx == 1:
        window = 200 + dominant_score * 600
    else:
        window = 100 + dominant_score * 400

    # Incertidumbre del clasificador
    # Alta si los scores están muy igualados, baja si hay uno dominante
    sorted_scores = sorted(scores, reverse=True)
    uncertainty = 1.0 - (sorted_scores[0] - sorted_scores[1]) / max(sorted_scores[0], 0.01)
    uncertainty = np.clip(uncertainty, 0, 1)

    td = TRANSITION_DEFINITIONS[dominant_idx]

    return TransitionResult(
        planet_name=planet_name,
        scores=scores,
        dominant_idx=dominant_idx,
        dominant_score=dominant_score,
        dominant_label=td["label"],
        dominant_name=td["name"],
        continuous_state=continuous,
        false_negative_reduction=float(fn_reduction),
        detectability_window_ma=float(window),
        uncertainty=float(uncertainty),
    )


def rank_by_transition(
    planets: List[Dict],
    target_transition: int = 2,  # Default: buscar ecosistemas
) -> List[Tuple[Dict, TransitionResult]]:
    """
    Ordena planetas por probabilidad de estar en una transición específica.

    Args:
        planets: Lista de dicts con 'evidences' y 'name'
        target_transition: 0=química, 1=metabolismo, 2=ecosistema, 3=tecnosfera

    Returns:
        Lista ordenada (planeta, resultado)
    """
    ranked = []
    for p in planets:
        result = classify_transitions(p["evidences"], p["name"])
        ranked.append((p, result))

    ranked.sort(key=lambda x: x[1].scores[target_transition], reverse=True)
    return ranked


def get_transition_trajectory(
    evidences_history: List[Dict[str, float]],
    planet_name: str = "Target",
) -> List[TransitionResult]:
    """
    Analiza la trayectoria temporal de un planeta a través de
    múltiples snapshots de evidencias.

    Args:
        evidences_history: Lista de dicts de evidencias en orden temporal
        planet_name: Nombre del planeta

    Returns:
        Lista de TransitionResult, uno por snapshot
    """
    return [
        classify_transitions(ev, planet_name)
        for ev in evidences_history
    ]
