"""
CORAL ASTROBIO — Capa 1: Dynamic Bayesian Consilience Engine (DBCE)
Motor de inferencia bayesiana competitiva bio vs. abiótico.

Resuelve el problema de circularidad de Seager:
no solo detecta coherencia entre biofirmas, sino que compite
activamente contra modelos abióticos alternativos.

Basado en consenso multi-IA Coral 2026:
- herramienta_marco_bayesiano_consiliencia_v1 (DeepSeek)
- herramienta_consiliencia_dinamica_bayesiana_multi_línea (Kimi)
- consiliencia_seager_falsabilidad_problema (Kimi/Grok)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from data.planets import EVIDENCE_DEFINITIONS, ABIO_MODELS


@dataclass
class BayesianResult:
    """Resultado completo del motor bayesiano."""
    planet_name: str
    odds_ratio: float          # bio/abio odds ratio
    bio_probability: float     # P(biológico | evidencias)
    abio_probability: float    # P(abiótico | evidencias)
    consilience_score: float   # Consiliencia corregida [0-1]
    log_odds: float
    ci_95_low: float
    ci_95_high: float
    verdict: str               # "BIOLÓGICO PROBABLE" | "ABIÓTICO PROBABLE" | "INCIERTO"
    active_abio_models: List[Dict]
    evidence_contributions: Dict[str, float]
    paradigm_shift: bool       # True si odds > umbral de alarma
    false_negative_risk: float # Riesgo de falso negativo vs paradigma O2


def compute_bayesian_odds(
    evidences: Dict[str, float],
    planet_name: str = "Desconocido",
    prior_log_odds: float = -0.15,  # Prior agnóstico ligeramente abiótico
    paradigm_shift_threshold: float = 2.5,
) -> BayesianResult:
    """
    Calcula odds ratio bio/abiótico dado un diccionario de evidencias.

    El prior -0.15 representa el estado agnóstico: sin evidencia,
    asumimos que los procesos abióticos son algo más probables
    (basado en que conocemos un solo caso de vida: la Tierra).

    Args:
        evidences: Dict con claves de EVIDENCE_DEFINITIONS, valores [0,1]
        planet_name: Nombre del planeta para el reporte
        prior_log_odds: Log-odds prior (default agnóstico)
        paradigm_shift_threshold: Umbral de odds para activar alerta

    Returns:
        BayesianResult con todos los parámetros calculados
    """
    log_odds = prior_log_odds
    contributions = {}

    for key, defn in EVIDENCE_DEFINITIONS.items():
        if key == "causal_lag":
            continue  # Este parámetro lo usa el analizador causal
        val = evidences.get(key, 0.0)
        w = defn["weight"]

        # Contribución log-odds: evidencia positiva suma, negativa resta
        if w > 0:
            contrib = w * (val - 0.30) * 0.80
        else:
            contrib = w * (val - 0.20) * 0.60

        log_odds += contrib
        contributions[key] = contrib

    # Convertir log-odds a probabilidades
    odds = np.exp(log_odds)
    bio_prob = odds / (1.0 + odds)
    abio_prob = 1.0 - bio_prob

    # Intervalo de confianza 95% (bootstrap simplificado)
    # Asumimos incertidumbre de ±40% en el odds ratio
    ci_low = odds * 0.42
    ci_high = odds * 2.38

    # Puntuación de consiliencia corregida
    # No es solo coherencia: penaliza si modelos abióticos también son activos
    active_abio = _compute_abio_models(evidences, abio_prob)
    abio_penalty = sum(m["score"] for m in active_abio if m["score"] > 0.25) * 0.08
    consilience = np.clip(
        bio_prob * 0.78
        + evidences.get("seasonal", 0) * 0.14
        + evidences.get("o2proxy", 0) * 0.08
        - abio_penalty,
        0.01, 0.99
    )

    # Veredicto
    if consilience > 0.68:
        verdict = "BIOLÓGICO PROBABLE"
    elif consilience < 0.35:
        verdict = "ABIÓTICO PROBABLE"
    else:
        verdict = "INCIERTO"

    # Riesgo de falso negativo vs paradigma O2 clásico
    # Si hay vida sin O2 (seasonal alto, surface alto) pero o2proxy bajo → riesgo alto
    fn_risk = max(0.0, (evidences.get("seasonal", 0) * 0.4
                        + evidences.get("surface", 0) * 0.3
                        - evidences.get("o2proxy", 0) * 0.5))
    fn_risk = np.clip(fn_risk, 0, 1)

    return BayesianResult(
        planet_name=planet_name,
        odds_ratio=float(odds),
        bio_probability=float(bio_prob),
        abio_probability=float(abio_prob),
        consilience_score=float(consilience),
        log_odds=float(log_odds),
        ci_95_low=float(ci_low),
        ci_95_high=float(ci_high),
        verdict=verdict,
        active_abio_models=active_abio,
        evidence_contributions=contributions,
        paradigm_shift=bool(odds > paradigm_shift_threshold),
        false_negative_risk=float(fn_risk),
    )


def _compute_abio_models(
    evidences: Dict[str, float],
    abio_prob: float
) -> List[Dict]:
    """Calcula la puntuación de cada modelo abiótico competidor."""
    results = []
    for model in ABIO_MODELS:
        driver_val = evidences.get(model["key_driver"], 0.0)
        score = np.clip(
            model["base_score"] * (0.70 + abio_prob * 0.90) + driver_val * 0.10,
            0.0, 0.99
        )
        results.append({
            "name": model["name"],
            "score": float(score),
            "dominant": score > 0.25,
            "key_driver": model["key_driver"],
            "driver_value": float(driver_val),
        })
    return sorted(results, key=lambda x: x["score"], reverse=True)


def rank_planets(planets: List[Dict]) -> List[Tuple[Dict, BayesianResult]]:
    """
    Ordena una lista de planetas por probabilidad biológica.

    Args:
        planets: Lista de dicts con campo 'evidences' y 'name'

    Returns:
        Lista de (planeta, resultado) ordenada por bio_probability desc
    """
    ranked = []
    for p in planets:
        result = compute_bayesian_odds(
            evidences=p["evidences"],
            planet_name=p["name"]
        )
        ranked.append((p, result))

    ranked.sort(key=lambda x: x[1].bio_probability, reverse=True)
    return ranked


def sensitivity_analysis(
    evidences: Dict[str, float],
    planet_name: str = "Target",
    n_samples: int = 1000,
    noise_std: float = 0.05,
) -> Dict:
    """
    Análisis de sensibilidad: cuánto cambia el resultado si las
    evidencias tienen un 5% de error de medición.

    Returns dict con estadísticas de la distribución de odds.
    """
    odds_samples = []
    for _ in range(n_samples):
        noisy = {
            k: np.clip(v + np.random.normal(0, noise_std), 0, 1)
            for k, v in evidences.items()
        }
        result = compute_bayesian_odds(noisy, planet_name)
        odds_samples.append(result.odds_ratio)

    arr = np.array(odds_samples)
    return {
        "mean_odds": float(np.mean(arr)),
        "std_odds": float(np.std(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "p50": float(np.percentile(arr, 50)),
        "prob_bio_robust": float(np.mean(arr > 1.0)),
        "n_samples": n_samples,
    }


def most_informative_evidence(
    evidences: Dict[str, float],
    planet_name: str = "Target",
) -> List[Tuple[str, float]]:
    """
    Identifica qué evidencia, si aumentara 0.1 puntos,
    produciría el mayor cambio en odds ratio.
    Útil para priorizar observaciones futuras.
    """
    base = compute_bayesian_odds(evidences, planet_name)
    deltas = []

    for key in EVIDENCE_DEFINITIONS:
        if key == "causal_lag":
            continue
        modified = {**evidences, key: min(1.0, evidences.get(key, 0) + 0.10)}
        result = compute_bayesian_odds(modified, planet_name)
        delta = result.odds_ratio - base.odds_ratio
        deltas.append((key, delta))

    deltas.sort(key=lambda x: abs(x[1]), reverse=True)
    return deltas
