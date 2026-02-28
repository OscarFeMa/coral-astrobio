"""
CORAL ASTROBIO — Capa 3: Causal Network Analyzer
Análisis de series temporales para detectar firmas dinámicas no geológicas.

Si la variación en reflectancia superficial PRECEDE sistemáticamente
a cambios en gases atmosféricos → firma de metabolismo planetario.

La vida como régimen dinámico, no como sustancia.
(DeepSeek: cambio_paradigma_vida_como_propiedad_emergente_redes)

Detecta:
- Desfase causal entre señales (cross-correlation lag)
- Sincronía de fase no trivial entre frecuencias
- Dimensionalidad efectiva (grados de libertad acoplados)
- Entropía de información fuera del equilibrio
"""

import math
import numpy as np
from scipy import signal, stats
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class CausalSignal:
    """Series temporales de un planeta."""
    planet_name: str
    time: np.ndarray           # Tiempo en días (fracción de año orbital)
    surface_reflectance: np.ndarray   # Señal de reflectancia superficial
    atmospheric_gas: np.ndarray       # Proxy gas atmosférico (CH4/O3)
    stellar_flux: np.ndarray          # Flujo estelar normalizado
    period_days: float                # Período orbital


@dataclass
class CausalResult:
    """Resultado del análisis causal."""
    planet_name: str
    causal_lag_days: float        # Desfase causal estimado en días
    causal_lag_fraction: float    # Desfase como fracción del período
    phase_synchrony: float        # Sincronía de fase [0-1]
    info_entropy: float           # Entropía de información [0-1]
    effective_dim: float          # Dimensionalidad efectiva
    granger_bio: float            # Estadístico Granger (superficie→gas)
    granger_abio: float           # Estadístico Granger (gas→superficie)
    causal_direction: str         # "BIO" | "ABIO" | "BIDIRECCIONAL" | "NONE"
    non_geological: float         # Probabilidad de firma no geológica [0-1]
    spectral_complexity: float    # Complejidad espectral [0-1]


def generate_synthetic_timeseries(
    planet: Dict,
    n_periods: float = 3.0,
    points_per_period: int = 120,
    noise_level: float = 0.05,
    seed: Optional[int] = None,
) -> CausalSignal:
    """
    Genera series temporales sintéticas pero físicamente informadas
    a partir de las evidencias de un planeta.

    La amplitud y el desfase de las señales reflejan las evidencias:
    - seasonal alto → mayor amplitud estacional
    - causal_lag → desfase superficie/gas
    - volcanic → ruido de alta frecuencia (geológico)

    Args:
        planet: Dict con 'evidences', 'name', 'period_days'
        n_periods: Número de períodos a simular
        points_per_period: Puntos de datos por período orbital
        noise_level: Nivel base de ruido
        seed: Semilla aleatoria para reproducibilidad
    """
    if seed is not None:
        np.random.seed(seed)

    ev = planet.get("evidences", {})
    period = planet.get("period_days", 30.0)
    name = planet.get("name", "Unknown")

    n_points = int(n_periods * points_per_period)
    t = np.linspace(0, n_periods * period, n_points)
    omega = 2 * np.pi / period  # Frecuencia orbital

    # ---- SEÑAL DE REFLECTANCIA SUPERFICIAL ----
    # Componente biológica: modulada por estación (fotosíntesis/metabolismo)
    seasonal_amp = 0.15 + ev.get("seasonal", 0.3) * 0.25
    surface_amp = 0.10 + ev.get("surface", 0.3) * 0.15

    surface = (
        seasonal_amp * np.sin(omega * t)                           # Ciclo anual principal
        + surface_amp * np.sin(2 * omega * t + 0.3)               # Armónico 2do
        + 0.04 * np.sin(3 * omega * t + 0.8)                     # Armónico 3ro
        + ev.get("volcanic", 0.2) * 0.08 * np.random.randn(n_points)  # Ruido geológico
        + noise_level * np.random.randn(n_points)                  # Ruido instrumental
    )

    # ---- SEÑAL ATMOSFÉRICA (GAS PROXY) ----
    # La clave: en vida real, la señal de gas viene DESPUÉS de la superficial
    # porque el metabolismo produce los gases con un desfase causal
    lag_fraction = ev.get("causal_lag", 0.25)
    lag_days = lag_fraction * period
    lag_points = int(lag_fraction * points_per_period)

    gas_amp = 0.12 + ev.get("ch4", 0.2) * 0.18
    gas_o2_amp = 0.08 + ev.get("o2proxy", 0.1) * 0.12
    dms_amp = ev.get("dms", 0.05) * 0.10

    # Gas base: correlacionado con superficie pero retrasado
    gas_base = (
        gas_amp * np.sin(omega * t - omega * lag_days)
        + gas_o2_amp * np.sin(2 * omega * t - omega * lag_days * 0.7)
        + dms_amp * np.sin(omega * t * 1.3)
    )

    # Añadir componente abiótica no correlacionada
    volcanic_noise = ev.get("volcanic", 0.2) * 0.15 * np.random.randn(n_points)
    photochem_noise = 0.03 * np.sin(omega * t * 0.7 + np.random.uniform(0, np.pi))

    gas = gas_base + volcanic_noise + photochem_noise + noise_level * np.random.randn(n_points)

    # ---- FLUJO ESTELAR ----
    stellar = 1.0 + 0.005 * np.sin(omega * t * 0.1) + 0.002 * np.random.randn(n_points)

    return CausalSignal(
        planet_name=name,
        time=t,
        surface_reflectance=surface,
        atmospheric_gas=gas,
        stellar_flux=stellar,
        period_days=period,
    )


def analyze_causal_structure(sig: CausalSignal) -> CausalResult:
    """
    Analiza la estructura causal de las series temporales.

    Métricas calculadas:
    1. Desfase causal: cross-correlation máximo surface → gas
    2. Sincronía de fase: coherencia espectral normalizada
    3. Entropía de información: complejidad fuera del equilibrio
    4. Causalidad de Granger (simplificada): ¿surface predice gas?
    5. Dimensionalidad efectiva: grados de libertad acoplados

    Returns:
        CausalResult con todas las métricas
    """
    s = sig.surface_reflectance
    g = sig.atmospheric_gas
    t = sig.time
    period = sig.period_days

    # Normalizar señales
    s_norm = (s - np.mean(s)) / (np.std(s) + 1e-10)
    g_norm = (g - np.mean(g)) / (np.std(g) + 1e-10)

    # ---- 1. DESFASE CAUSAL ----
    # Cross-correlation completa
    n = len(s_norm)
    corr = np.correlate(g_norm, s_norm, mode='full')
    lags = np.arange(-(n-1), n)
    dt = t[1] - t[0]  # Paso temporal en días

    # Buscamos el lag máximo en rango físicamente razonable [0, 0.5 período]
    max_lag_points = int(0.5 * period / dt)
    valid_mask = (lags >= 0) & (lags <= max_lag_points)
    valid_lags = lags[valid_mask]
    valid_corr = corr[valid_mask]

    if len(valid_corr) > 0:
        best_idx = np.argmax(np.abs(valid_corr))
        causal_lag_points = int(valid_lags[best_idx])
        causal_lag_days = float(causal_lag_points * dt)
    else:
        causal_lag_days = 0.0
        causal_lag_points = 0

    causal_lag_fraction = causal_lag_days / max(period, 1.0)

    # ---- 2. SINCRONÍA DE FASE ----
    # Coherencia espectral en la frecuencia orbital
    f, Cxy = signal.coherence(s_norm, g_norm, fs=1.0/dt, nperseg=min(256, n//4))
    f_orbital = 1.0 / period
    if len(f) > 0:
        idx_orbital = np.argmin(np.abs(f - f_orbital))
        # Sincronía = promedio de coherencia en banda orbital ±20%
        band = (f >= f_orbital * 0.8) & (f <= f_orbital * 1.2)
        phase_synchrony = float(np.mean(Cxy[band])) if np.any(band) else float(Cxy[idx_orbital])
    else:
        phase_synchrony = 0.0

    phase_synchrony = float(np.clip(phase_synchrony, 0, 1))

    # ---- 3. ENTROPÍA DE INFORMACIÓN ----
    # Entropía de permutación (Bandt-Pompe) como proxy de complejidad
    info_entropy = _permutation_entropy(s_norm, order=3)
    info_entropy = float(np.clip(info_entropy, 0, 1))

    # ---- 4. CAUSALIDAD DE GRANGER (simplificada) ----
    # Comparamos varianza residual: ¿añadir superficie mejora predicción del gas?
    # Versión simple: regresión lineal AR
    lag_order = max(1, causal_lag_points)
    granger_bio, granger_abio = _simplified_granger(s_norm, g_norm, lag_order)

    # Dirección causal
    if granger_bio > 0.15 and granger_bio > granger_abio * 1.5:
        causal_direction = "BIO"        # Superficie → Gas (metabolismo)
    elif granger_abio > 0.15 and granger_abio > granger_bio * 1.5:
        causal_direction = "ABIO"       # Gas → Superficie (fotoquímica)
    elif granger_bio > 0.10 and granger_abio > 0.10:
        causal_direction = "BIDIRECCIONAL"
    else:
        causal_direction = "NONE"

    # ---- 5. DIMENSIONALIDAD EFECTIVA ----
    # PCA de la matriz [surface, gas, stellar_flux]
    matrix = np.column_stack([s_norm, g_norm, (sig.stellar_flux - 1) * 100])
    cov = np.cov(matrix.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0)
    total = np.sum(eigenvalues)
    if total > 0:
        normalized = eigenvalues / total
        # Entropía de los eigenvalores = dimensionalidad efectiva
        effective_dim = float(np.exp(-np.sum(normalized * np.log(normalized + 1e-10))))
    else:
        effective_dim = 1.0

    # ---- 6. COMPLEJIDAD ESPECTRAL ----
    # Número de picos espectrales significativos normalizado
    freqs, psd = signal.welch(s_norm, fs=1.0/dt, nperseg=min(256, n//4))
    peaks, _ = signal.find_peaks(psd, height=np.percentile(psd, 75))
    spectral_complexity = float(np.clip(len(peaks) / 10.0, 0, 1))

    # ---- PUNTUACIÓN FINAL: ¿FIRMA NO GEOLÓGICA? ----
    non_geological = np.clip(
        phase_synchrony * 0.30
        + (1.0 if causal_direction == "BIO" else 0.3 if causal_direction == "BIDIRECCIONAL" else 0.0) * 0.25
        + info_entropy * 0.20
        + spectral_complexity * 0.15
        + np.clip(effective_dim / 3.0, 0, 1) * 0.10,
        0.0, 1.0
    )

    return CausalResult(
        planet_name=sig.planet_name,
        causal_lag_days=causal_lag_days,
        causal_lag_fraction=causal_lag_fraction,
        phase_synchrony=phase_synchrony,
        info_entropy=info_entropy,
        effective_dim=effective_dim,
        granger_bio=float(granger_bio),
        granger_abio=float(granger_abio),
        causal_direction=causal_direction,
        non_geological=float(non_geological),
        spectral_complexity=spectral_complexity,
    )


def _permutation_entropy(x: np.ndarray, order: int = 3) -> float:
    """
    Entropía de permutación de Bandt-Pompe.
    Mide complejidad y aleatoriedad de una serie temporal.
    Alta entropía + estructura periódica = firma biológica.
    """
    n = len(x)
    patterns = {}
    for i in range(n - order + 1):
        pattern = tuple(np.argsort(x[i:i+order]))
        patterns[pattern] = patterns.get(pattern, 0) + 1

    total = sum(patterns.values())
    if total == 0:
        return 0.0

    probs = np.array([v / total for v in patterns.values()])
    max_entropy = np.log(math.factorial(order))
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def _simplified_granger(
    x: np.ndarray,
    y: np.ndarray,
    lag: int,
) -> Tuple[float, float]:
    """
    Test de causalidad de Granger simplificado.

    bio_score: reducción de varianza al añadir x (superficie) para predecir y (gas)
    abio_score: reducción de varianza al añadir y para predecir x

    Returns:
        (bio_score, abio_score) en [0, 1]
    """
    lag = max(1, min(lag, len(x)//4))

    def var_reduction(predictor, target, l):
        n = len(target) - l
        if n < 4:
            return 0.0
        # Modelo restringido: autoregresión de target
        y_t = target[l:]
        y_lag = np.column_stack([target[l-k-1:len(target)-k-1] for k in range(min(l, 3))])
        try:
            _, res_restricted, _, _ = np.linalg.lstsq(
                np.column_stack([np.ones(n), y_lag]), y_t, rcond=None
            )
            # Modelo completo: + predictor
            x_lag = np.column_stack([predictor[l-k-1:len(predictor)-k-1] for k in range(min(l, 3))])
            _, res_full, _, _ = np.linalg.lstsq(
                np.column_stack([np.ones(n), y_lag, x_lag]), y_t, rcond=None
            )
            if len(res_restricted) > 0 and len(res_full) > 0:
                reduction = float((res_restricted[0] - res_full[0]) / (res_restricted[0] + 1e-10))
                return float(np.clip(reduction, 0, 1))
        except Exception:
            pass
        return 0.0

    bio_score = var_reduction(x, y, lag)   # ¿superficie predice gas?
    abio_score = var_reduction(y, x, lag)  # ¿gas predice superficie?
    return bio_score, abio_score
