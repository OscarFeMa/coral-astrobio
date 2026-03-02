"""
CORAL ASTROBIO — Catálogo dinámico desde NASA Exoplanet Archive
Carga todos los exoplanetas confirmados con parámetros suficientes.
Las evidencias se estiman automáticamente a partir de parámetros físicos.
"""

import os
import json
import math
import hashlib
import urllib.request
import urllib.parse
import ssl
from typing import List, Dict, Optional

# ── Parámetros de filtrado ───────────────────────────────────────
MIN_RADIUS_EARTH = 0.5    # mínimo radio
MAX_RADIUS_EARTH = 4.0    # excluir Júpiter y gigantes gaseosos grandes
MAX_DIST_PC      = 200    # distancia máxima en parsecs
MAX_TEQ_K        = 2000   # temperatura de equilibrio máxima
MIN_TEQ_K        = 150    # temperatura mínima

# Cache en disco para no llamar a NASA en cada reinicio
CACHE_FILE = os.path.join(os.path.dirname(__file__), "_nasa_cache.json")
CACHE_MAX_AGE_HOURS = 24

# ── NASA TAP Query ───────────────────────────────────────────────
NASA_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

NASA_QUERY = """
SELECT pl_name, hostname, pl_rade, pl_bmasse, pl_eqt, 
       sy_dist, st_spectype, pl_orbper, pl_tsm,
       disc_facility
FROM pscomppars
WHERE pl_rade IS NOT NULL
  AND pl_bmasse IS NOT NULL
  AND pl_eqt IS NOT NULL
  AND sy_dist IS NOT NULL
  AND pl_rade >= 0.5
  AND pl_rade <= 4.0
  AND pl_eqt >= 150
  AND pl_eqt <= 2000
  AND sy_dist <= 200
ORDER BY pl_tsm DESC NULLS LAST
""".strip()


def _fetch_nasa() -> List[Dict]:
    """Descarga planetas desde NASA TAP API."""
    params = urllib.parse.urlencode({
        "query": NASA_QUERY,
        "format": "json",
    })
    url = f"{NASA_TAP}?{params}"
    ctx = ssl.create_default_context()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CoralAstrobio/3.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f"[NASA TAP] Error: {e}")
        return []


def _load_cache() -> Optional[List[Dict]]:
    """Carga caché si existe y no ha expirado."""
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE) as f:
            data = json.load(f)
        import time
        age_hours = (time.time() - data.get("ts", 0)) / 3600
        if age_hours > CACHE_MAX_AGE_HOURS:
            return None
        return data.get("planets", [])
    except Exception:
        return None


def _save_cache(planets: List[Dict]):
    """Guarda planetas en caché."""
    import time
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump({"ts": time.time(), "planets": planets}, f)
    except Exception:
        pass


def _classify_type(radius: float, mass: float, teq: float) -> str:
    """Clasifica el tipo de planeta."""
    if radius < 1.0:
        return "Tierra-análogo"
    elif radius < 1.6:
        if mass < 5:
            return "Super-Tierra"
        return "Super-Tierra masiva"
    elif radius < 2.0:
        return "Sub-Neptuno"
    elif radius < 3.5:
        if teq > 400:
            return "Mini-Neptuno caliente"
        return "Mini-Neptuno Hycean"
    else:
        return "Neptuno"


def _is_habitable_zone(teq: float, spectype: str) -> bool:
    """Estima si el planeta está en zona habitable."""
    return 180 <= teq <= 400


def _is_jwst_priority(tsm: float, dist: float, radius: float) -> bool:
    """Estima prioridad JWST."""
    if tsm and tsm > 30:
        return True
    if dist < 15 and radius < 2.0:
        return True
    return False


def _estimate_evidences(planet_data: Dict) -> Dict:
    """
    Estima las evidencias observacionales a partir de parámetros físicos.
    Todas las evidencias están en [0, 1].
    
    Lógica:
    - h2o: mayor si teq en rango habitable y radio bajo
    - ch4: mayor en mundos pequeños con temperatura moderada
    - o2proxy: mayor si en zona habitable y radio terrestre
    - seasonal: mayor si periodo orbital en rango estacional
    - surface: mayor en planetas terrestres con temperatura moderada
    - volcanic: mayor en planetas muy calientes o masivos
    - dms: solo detectable en mundos Hycean con JWST
    - techno: siempre 0 (sin evidencia)
    - causal_lag: basado en periodo orbital normalizado
    """
    teq = float(planet_data.get("pl_eqt") or 300)
    radius = float(planet_data.get("pl_rade") or 1.5)
    mass = float(planet_data.get("pl_bmasse") or 5.0)
    period = float(planet_data.get("pl_orbper") or 30.0)
    tsm = float(planet_data.get("pl_tsm") or 10.0)
    dist = float(planet_data.get("sy_dist") or 50.0)

    # Factor habitable: 1.0 en zona ideal, decrece fuera
    hz_center = 270  # K ideal
    hz_factor = max(0, 1.0 - abs(teq - hz_center) / 250.0)

    # Factor tamaño: favorece planetas pequeños para biosignatures
    size_factor = max(0, 1.0 - (radius - 1.0) / 3.5)

    # Factor observabilidad: TSM normalizado
    obs_factor = min(1.0, tsm / 100.0) if tsm else 0.1

    # Factor proximidad: planetas cercanos tienen mejor S/N
    prox_factor = max(0.1, 1.0 - dist / 200.0)

    # Factor masa: exceso de masa reduce probabilidad de vida
    mass_penalty = max(0.3, 1.0 - (mass - 1.0) / 20.0)

    # Calcular evidencias
    h2o = round(min(0.95, hz_factor * 0.8 + obs_factor * 0.15 + 0.05), 2)
    ch4 = round(min(0.9, hz_factor * 0.4 * size_factor + obs_factor * 0.1), 2)
    o2proxy = round(min(0.85, hz_factor * size_factor * 0.6 + obs_factor * 0.1), 2)
    seasonal = round(min(0.8, hz_factor * 0.5 * (1.0 if 5 < period < 400 else 0.3)), 2)
    surface = round(min(0.75, hz_factor * size_factor * mass_penalty * 0.7), 2)
    volcanic = round(min(0.95, max(0.05, (teq - 300) / 1200 + mass / 25.0 + 0.1)), 2)
    
    # DMS: solo en mundos Hycean con buena observabilidad
    dms = round(min(0.4, hz_factor * obs_factor * 0.3 if radius > 1.5 and radius < 2.8 else 0.02), 2)
    
    # Causal lag: basado en periodo orbital (normalizado a [0,1])
    causal_lag = round(min(0.9, max(0.05, math.log(period + 1) / 8.0)), 2)

    return {
        "o2proxy":    max(0.01, o2proxy),
        "ch4":        max(0.01, ch4),
        "h2o":        max(0.01, h2o),
        "dms":        max(0.0, dms),
        "seasonal":   max(0.01, seasonal),
        "surface":    max(0.01, surface),
        "volcanic":   max(0.05, volcanic),
        "techno":     0.00,
        "causal_lag": causal_lag,
    }


def _compute_tsm(radius, mass, teq, dist, star_radius=0.5) -> float:
    """Calcula TSM si no está disponible."""
    if not all([radius, mass, teq, dist]):
        return 0.0
    try:
        # TSM = scale_factor * (Rp^3 * Teq) / (Mp * Rs^2) * J
        # Aproximación simplificada
        scale = 0.190 if radius < 1.5 else (1.26 if radius < 2.75 else 1.28)
        tsm = scale * (radius**3 * teq) / (mass * star_radius**2)
        return round(tsm, 1)
    except Exception:
        return 0.0


def _nasa_to_planet(row: Dict) -> Optional[Dict]:
    """Convierte una fila NASA TAP al formato interno."""
    try:
        name = str(row.get("pl_name") or "").strip()
        host = str(row.get("hostname") or "").strip()
        if not name or not host:
            return None

        radius = float(row.get("pl_rade") or 0)
        mass = float(row.get("pl_bmasse") or 0)
        teq = float(row.get("pl_eqt") or 0)
        dist = float(row.get("sy_dist") or 0)
        period = float(row.get("pl_orbper") or 0) if row.get("pl_orbper") else None
        spectype = str(row.get("st_spectype") or "—").strip() or "—"
        tsm_raw = row.get("pl_tsm")
        tsm = float(tsm_raw) if tsm_raw else _compute_tsm(radius, mass, teq, dist)

        if not (radius and mass and teq and dist):
            return None

        hz = _is_habitable_zone(teq, spectype)
        jwst = _is_jwst_priority(tsm, dist, radius)
        ptype = _classify_type(radius, mass, teq)
        evidences = _estimate_evidences(row)

        return {
            "name": name,
            "host": host,
            "type": ptype,
            "mass_earth": round(mass, 3),
            "radius_earth": round(radius, 3),
            "teq_k": round(teq),
            "dist_pc": round(dist, 2),
            "spectype": spectype[:10],
            "tsm": round(tsm, 1),
            "period_days": round(period, 3) if period else None,
            "jwst": jwst,
            "hz": hz,
            "notes": f"Fuente: NASA Exoplanet Archive. Tipo: {ptype}.",
            "evidences": evidences,
        }
    except Exception as e:
        return None


def _load_planets_from_nasa() -> List[Dict]:
    """Carga planetas desde NASA o caché."""
    # Intentar caché primero
    cached = _load_cache()
    if cached:
        print(f"[NASA TAP] Usando caché: {len(cached)} planetas")
        return cached

    print("[NASA TAP] Descargando catálogo...")
    rows = _fetch_nasa()
    if not rows:
        print("[NASA TAP] Sin datos — usando catálogo curado de respaldo")
        return []

    planets = []
    for row in rows:
        p = _nasa_to_planet(row)
        if p:
            planets.append(p)

    # Deduplicar por nombre
    seen = set()
    unique = []
    for p in planets:
        if p["name"] not in seen:
            seen.add(p["name"])
            unique.append(p)

    print(f"[NASA TAP] {len(unique)} planetas cargados")
    _save_cache(unique)
    return unique


# ── Catálogo curado de respaldo (12 planetas originales) ─────────
PLANETS_CURATED = [
    {
        "name": "LHS 1140 b", "host": "LHS 1140", "type": "Super-Tierra",
        "mass_earth": 6.38, "radius_earth": 1.727, "teq_k": 235, "dist_pc": 14.99,
        "spectype": "M4.5V", "tsm": 18.3, "period_days": 24.74, "jwst": True, "hz": True,
        "notes": "Mejor candidato actual. Atmósfera probablemente retenida.",
        "evidences": {"o2proxy": 0.12, "ch4": 0.31, "h2o": 0.68, "dms": 0.05, "seasonal": 0.42, "surface": 0.38, "volcanic": 0.25, "techno": 0.01, "causal_lag": 0.34}
    },
    {
        "name": "TRAPPIST-1 e", "host": "TRAPPIST-1", "type": "Tierra-análogo",
        "mass_earth": 0.772, "radius_earth": 0.910, "teq_k": 251, "dist_pc": 12.43,
        "spectype": "M8V", "tsm": 14.1, "period_days": 6.10, "jwst": True, "hz": True,
        "notes": "Sistema compacto. 7 planetas. Candidato ideal para espectroscopía comparativa.",
        "evidences": {"o2proxy": 0.08, "ch4": 0.22, "h2o": 0.81, "dms": 0.03, "seasonal": 0.19, "surface": 0.51, "volcanic": 0.18, "techno": 0.00, "causal_lag": 0.21}
    },
    {
        "name": "K2-18 b", "host": "K2-18", "type": "Mini-Neptuno Hycean",
        "mass_earth": 8.63, "radius_earth": 2.372, "teq_k": 255, "dist_pc": 38.0,
        "spectype": "M2.5V", "tsm": 92.0, "period_days": 32.94, "jwst": True, "hz": True,
        "notes": "DMS tentativo detectado por JWST. Candidato Hycean.",
        "evidences": {"o2proxy": 0.15, "ch4": 0.35, "h2o": 0.72, "dms": 0.28, "seasonal": 0.31, "surface": 0.22, "volcanic": 0.19, "techno": 0.00, "causal_lag": 0.38}
    },
    {
        "name": "GJ 1214 b", "host": "GJ 1214", "type": "Mini-Neptuno",
        "mass_earth": 6.26, "radius_earth": 2.742, "teq_k": 556, "dist_pc": 14.64,
        "spectype": "M4.5V", "tsm": 46.4, "period_days": 1.58, "jwst": True, "hz": False,
        "notes": "Alta metalicidad atmosférica. Espectroscopía JWST en curso.",
        "evidences": {"o2proxy": 0.05, "ch4": 0.18, "h2o": 0.55, "dms": 0.02, "seasonal": 0.08, "surface": 0.15, "volcanic": 0.42, "techno": 0.00, "causal_lag": 0.12}
    },
    {
        "name": "TOI-700 d", "host": "TOI-700", "type": "Tierra-análogo",
        "mass_earth": 1.57, "radius_earth": 1.144, "teq_k": 246, "dist_pc": 31.1,
        "spectype": "M2V", "tsm": 4.8, "period_days": 37.42, "jwst": True, "hz": True,
        "notes": "Primera Tierra en zona habitable confirmada por TESS.",
        "evidences": {"o2proxy": 0.09, "ch4": 0.24, "h2o": 0.71, "dms": 0.03, "seasonal": 0.38, "surface": 0.44, "volcanic": 0.21, "techno": 0.00, "causal_lag": 0.29}
    },
    {
        "name": "TRAPPIST-1 f", "host": "TRAPPIST-1", "type": "Tierra-análogo",
        "mass_earth": 0.934, "radius_earth": 1.045, "teq_k": 219, "dist_pc": 12.43,
        "spectype": "M8V", "tsm": 12.8, "period_days": 9.21, "jwst": True, "hz": True,
        "notes": "Zona habitable exterior. Posible océano bajo capa de hielo.",
        "evidences": {"o2proxy": 0.06, "ch4": 0.18, "h2o": 0.75, "dms": 0.02, "seasonal": 0.22, "surface": 0.44, "volcanic": 0.16, "techno": 0.00, "causal_lag": 0.19}
    },
    {
        "name": "Proxima Cen b", "host": "Proxima Centauri", "type": "Tierra-análogo",
        "mass_earth": 1.07, "radius_earth": 1.03, "teq_k": 234, "dist_pc": 1.29,
        "spectype": "M5.5V", "tsm": 5.2, "period_days": 11.19, "jwst": False, "hz": True,
        "notes": "Planeta habitable más cercano. Alta actividad estelar de la estrella.",
        "evidences": {"o2proxy": 0.07, "ch4": 0.20, "h2o": 0.61, "dms": 0.02, "seasonal": 0.28, "surface": 0.35, "volcanic": 0.31, "techno": 0.00, "causal_lag": 0.22}
    },
    {
        "name": "LHS 3844 b", "host": "LHS 3844", "type": "Super-Tierra",
        "mass_earth": 2.25, "radius_earth": 1.32, "teq_k": 805, "dist_pc": 14.9,
        "spectype": "M5V", "tsm": 0.3, "period_days": 0.46, "jwst": True, "hz": False,
        "notes": "Control abiótico. Alta temperatura, sin atmósfera detectable.",
        "evidences": {"o2proxy": 0.02, "ch4": 0.05, "h2o": 0.12, "dms": 0.00, "seasonal": 0.05, "surface": 0.08, "volcanic": 0.71, "techno": 0.00, "causal_lag": 0.06}
    },
]


# ── Carga principal ──────────────────────────────────────────────
def _build_planet_list() -> List[Dict]:
    """
    Intenta cargar desde NASA. Si falla, usa catálogo curado.
    Los planetas curados se añaden al inicio si no están ya en la lista NASA.
    """
    nasa_planets = _load_planets_from_nasa()

    if not nasa_planets:
        print("[planets] Usando catálogo curado (12 planetas)")
        return PLANETS_CURATED

    # Fusionar: los planetas curados tienen precedencia (mejor calibración)
    nasa_names = {p["name"].lower() for p in nasa_planets}
    extra_curated = [p for p in PLANETS_CURATED if p["name"].lower() not in nasa_names]

    combined = extra_curated + nasa_planets
    print(f"[planets] Catálogo total: {len(combined)} planetas ({len(extra_curated)} curados + {len(nasa_planets)} NASA)")
    return combined


# Carga: arranca con curados, descarga NASA en background
import threading

PLANETS = list(PLANETS_CURATED)

def _background_load():
    global PLANETS
    try:
        nasa = _build_planet_list()
        if nasa and len(nasa) > len(PLANETS_CURATED):
            PLANETS[:] = nasa
            print(f"[planets] Background update: {len(PLANETS)} planetas")
    except Exception as e:
        print(f"[planets] Background error: {e}")

threading.Thread(target=_background_load, daemon=True).start()


# ── Definiciones estáticas ───────────────────────────────────────
EVIDENCE_DEFINITIONS = {
    "o2proxy":    {"label": "O₃ proxy O₂ (JWST)",     "weight": 2.1,  "bio": True,  "unit": "[0-1]"},
    "ch4":        {"label": "CH₄ desequilibrio",        "weight": 1.4,  "bio": True,  "unit": "[0-1]"},
    "h2o":        {"label": "H₂O vapor detectado",      "weight": 1.0,  "bio": True,  "unit": "[0-1]"},
    "dms":        {"label": "DMS tentativo",             "weight": 1.8,  "bio": True,  "unit": "[0-1]"},
    "seasonal":   {"label": "Variación estacional",      "weight": 2.5,  "bio": True,  "unit": "[0-1]"},
    "surface":    {"label": "Borde rojo superficial",    "weight": 1.6,  "bio": True,  "unit": "[0-1]"},
    "volcanic":   {"label": "Actividad volcánica",       "weight": -1.8, "bio": False, "unit": "[0-1]"},
    "techno":     {"label": "Tecnosignatura (CFC)",      "weight": 3.0,  "bio": True,  "unit": "[0-1]"},
    "causal_lag": {"label": "Desfase causal (fracción)", "weight": 0.0,  "bio": True,  "unit": "[0-1]"},
}

ABIO_MODELS = [
    {"name": "Serpentinización + UV fotólisis", "base_score": 0.31, "key_driver": "volcanic"},
    {"name": "Vulcanismo + reducción CO₂",      "base_score": 0.24, "key_driver": "volcanic"},
    {"name": "Fotoquímica H₂-rica",             "base_score": 0.19, "key_driver": "ch4"},
    {"name": "Impactos + síntesis orgánica",    "base_score": 0.11, "key_driver": "surface"},
]


if __name__ == "__main__":
    print(f"Total planetas: {len(PLANETS)}")
    print(f"Con JWST: {sum(1 for p in PLANETS if p['jwst'])}")
    print(f"En zona habitable: {sum(1 for p in PLANETS if p['hz'])}")
    print(f"Top 5 por TSM:")
    for p in sorted(PLANETS, key=lambda x: x.get('tsm') or 0, reverse=True)[:5]:
        print(f"  {p['name']} TSM={p['tsm']} hz={p['hz']}")
