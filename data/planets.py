"""
CORAL ASTROBIO — Catálogo de exoplanetas curados
Fuente: NASA Exoplanet Archive (pscomppars, febrero 2026)
Parámetros físicos reales + evidencias observacionales estimadas
"""

PLANETS = [
    {
        "name": "LHS 1140 b",
        "host": "LHS 1140",
        "type": "Super-Tierra",
        "mass_earth": 6.38,
        "radius_earth": 1.727,
        "teq_k": 235,
        "dist_pc": 14.99,
        "spectype": "M4.5V",
        "tsm": 18.3,
        "period_days": 24.74,
        "jwst": True,
        "hz": True,
        "notes": "Mejor candidato actual. Atmósfera probablemente retenida. JWST Cycle 2 observado.",
        "evidences": {
            "o2proxy": 0.12, "ch4": 0.31, "h2o": 0.68,
            "dms": 0.05, "seasonal": 0.42, "surface": 0.38,
            "volcanic": 0.25, "techno": 0.01, "causal_lag": 0.34
        }
    },
    {
        "name": "TRAPPIST-1 e",
        "host": "TRAPPIST-1",
        "type": "Tierra-análogo",
        "mass_earth": 0.772,
        "radius_earth": 0.910,
        "teq_k": 251,
        "dist_pc": 12.43,
        "spectype": "M8V",
        "tsm": 14.1,
        "period_days": 6.10,
        "jwst": True,
        "hz": True,
        "notes": "Sistema compacto. 7 planetas. Candidato ideal para espectroscopía comparativa.",
        "evidences": {
            "o2proxy": 0.08, "ch4": 0.22, "h2o": 0.81,
            "dms": 0.03, "seasonal": 0.19, "surface": 0.51,
            "volcanic": 0.18, "techno": 0.00, "causal_lag": 0.21
        }
    },
    {
        "name": "TRAPPIST-1 f",
        "host": "TRAPPIST-1",
        "type": "Tierra-análogo",
        "mass_earth": 0.934,
        "radius_earth": 1.045,
        "teq_k": 219,
        "dist_pc": 12.43,
        "spectype": "M8V",
        "tsm": 12.8,
        "period_days": 9.21,
        "jwst": True,
        "hz": True,
        "notes": "Zona habitable exterior. Posible océano bajo capa de hielo.",
        "evidences": {
            "o2proxy": 0.06, "ch4": 0.18, "h2o": 0.75,
            "dms": 0.02, "seasonal": 0.22, "surface": 0.44,
            "volcanic": 0.16, "techno": 0.00, "causal_lag": 0.19
        }
    },
    {
        "name": "K2-18 b",
        "host": "K2-18",
        "type": "Mini-Neptuno Hycean",
        "mass_earth": 8.63,
        "radius_earth": 2.372,
        "teq_k": 255,
        "dist_pc": 38.0,
        "spectype": "M2.5V",
        "tsm": 92.0,
        "period_days": 32.94,
        "jwst": True,
        "hz": True,
        "notes": "JWST detectó CH4, CO2, posible DMS (2.7σ — no confirmado). Debate estadístico activo.",
        "evidences": {
            "o2proxy": 0.07, "ch4": 0.61, "h2o": 0.74,
            "dms": 0.31, "seasonal": 0.14, "surface": 0.22,
            "volcanic": 0.43, "techno": 0.00, "causal_lag": 0.18
        }
    },
    {
        "name": "TOI-700 d",
        "host": "TOI-700",
        "type": "Tierra-análogo",
        "mass_earth": 1.72,
        "radius_earth": 1.144,
        "teq_k": 269,
        "dist_pc": 31.13,
        "spectype": "M2.5V",
        "tsm": 16.2,
        "period_days": 37.42,
        "jwst": True,
        "hz": True,
        "notes": "En observación JWST Cycle 3. Temperatura de equilibrio favorable.",
        "evidences": {
            "o2proxy": 0.11, "ch4": 0.19, "h2o": 0.72,
            "dms": 0.04, "seasonal": 0.37, "surface": 0.33,
            "volcanic": 0.19, "techno": 0.00, "causal_lag": 0.26
        }
    },
    {
        "name": "TOI-700 e",
        "host": "TOI-700",
        "type": "Tierra-análogo",
        "mass_earth": 0.952,
        "radius_earth": 0.953,
        "teq_k": 290,
        "dist_pc": 31.13,
        "spectype": "M2.5V",
        "tsm": 11.4,
        "period_days": 27.81,
        "jwst": True,
        "hz": True,
        "notes": "Descubierto 2023. Más cercano a su estrella que TOI-700 d. Alta temperatura.",
        "evidences": {
            "o2proxy": 0.09, "ch4": 0.15, "h2o": 0.68,
            "dms": 0.03, "seasonal": 0.31, "surface": 0.28,
            "volcanic": 0.17, "techno": 0.00, "causal_lag": 0.22
        }
    },
    {
        "name": "Proxima Cen b",
        "host": "Proxima Centauri",
        "type": "Super-Tierra",
        "mass_earth": 1.173,
        "radius_earth": 1.08,
        "teq_k": 234,
        "dist_pc": 1.295,
        "spectype": "M5.5Ve",
        "tsm": 8.4,
        "period_days": 11.19,
        "jwst": False,
        "hz": True,
        "notes": "El más cercano. No transita. Candidato clave para direct imaging con HWO.",
        "evidences": {
            "o2proxy": 0.15, "ch4": 0.28, "h2o": 0.55,
            "dms": 0.02, "seasonal": 0.31, "surface": 0.44,
            "volcanic": 0.21, "techno": 0.00, "causal_lag": 0.29
        }
    },
    {
        "name": "YZ Ceti b",
        "host": "YZ Ceti",
        "type": "Tierra-análogo",
        "mass_earth": 0.75,
        "radius_earth": 0.89,
        "teq_k": 289,
        "dist_pc": 3.6,
        "spectype": "M4.5V",
        "tsm": 6.7,
        "period_days": 2.02,
        "jwst": False,
        "hz": False,
        "notes": "Ultra-cercano (3.6 pc). Candidato futuro para direct imaging. Posible radio-aurora.",
        "evidences": {
            "o2proxy": 0.09, "ch4": 0.17, "h2o": 0.61,
            "dms": 0.02, "seasonal": 0.28, "surface": 0.41,
            "volcanic": 0.22, "techno": 0.00, "causal_lag": 0.23
        }
    },
    {
        "name": "GJ 1214 b",
        "host": "GJ 1214",
        "type": "Mini-Neptuno",
        "mass_earth": 6.26,
        "radius_earth": 2.742,
        "teq_k": 556,
        "dist_pc": 14.64,
        "spectype": "M4.5V",
        "tsm": 46.4,
        "period_days": 1.58,
        "jwst": True,
        "hz": False,
        "notes": "Benchmark para sub-Neptunos. Atmósfera dominada por H2O. Referencia de calibración.",
        "evidences": {
            "o2proxy": 0.04, "ch4": 0.44, "h2o": 0.82,
            "dms": 0.01, "seasonal": 0.08, "surface": 0.15,
            "volcanic": 0.35, "techno": 0.00, "causal_lag": 0.12
        }
    },
    {
        "name": "GJ 667C c",
        "host": "GJ 667C",
        "type": "Super-Tierra",
        "mass_earth": 3.81,
        "radius_earth": 1.54,
        "teq_k": 277,
        "dist_pc": 6.84,
        "spectype": "M1.5V",
        "tsm": 9.2,
        "period_days": 28.14,
        "jwst": False,
        "hz": True,
        "notes": "Sistema triple. Zona habitable confirmada. Candidato a direct imaging.",
        "evidences": {
            "o2proxy": 0.10, "ch4": 0.21, "h2o": 0.58,
            "dms": 0.03, "seasonal": 0.25, "surface": 0.36,
            "volcanic": 0.28, "techno": 0.00, "causal_lag": 0.20
        }
    },
    {
        "name": "Ross 128 b",
        "host": "Ross 128",
        "type": "Tierra-análogo",
        "mass_earth": 1.35,
        "radius_earth": 1.11,
        "teq_k": 294,
        "dist_pc": 3.37,
        "spectype": "M4V",
        "tsm": 7.1,
        "period_days": 9.86,
        "jwst": False,
        "hz": True,
        "notes": "Segunda estrella M más cercana con planeta habitable. Poca actividad estelar.",
        "evidences": {
            "o2proxy": 0.08, "ch4": 0.19, "h2o": 0.52,
            "dms": 0.02, "seasonal": 0.24, "surface": 0.38,
            "volcanic": 0.20, "techno": 0.00, "causal_lag": 0.21
        }
    },
    {
        "name": "LHS 3844 b",
        "host": "LHS 3844",
        "type": "Super-Tierra",
        "mass_earth": 2.25,
        "radius_earth": 1.32,
        "teq_k": 805,
        "dist_pc": 14.9,
        "spectype": "M5V",
        "tsm": 0.3,
        "period_days": 0.46,
        "jwst": True,
        "hz": False,
        "notes": "Control abiótico. Alta temperatura, sin atmósfera detectable. Referencia negativa.",
        "evidences": {
            "o2proxy": 0.02, "ch4": 0.05, "h2o": 0.12,
            "dms": 0.00, "seasonal": 0.05, "surface": 0.08,
            "volcanic": 0.71, "techno": 0.00, "causal_lag": 0.06
        }
    },
]

EVIDENCE_DEFINITIONS = {
    "o2proxy":   {"label": "O₃ proxy O₂ (JWST)",      "weight": 2.1,  "bio": True,  "unit": "[0-1]"},
    "ch4":       {"label": "CH₄ desequilibrio",         "weight": 1.4,  "bio": True,  "unit": "[0-1]"},
    "h2o":       {"label": "H₂O vapor detectado",       "weight": 1.0,  "bio": True,  "unit": "[0-1]"},
    "dms":       {"label": "DMS tentativo",              "weight": 1.8,  "bio": True,  "unit": "[0-1]"},
    "seasonal":  {"label": "Variación estacional",       "weight": 2.5,  "bio": True,  "unit": "[0-1]"},
    "surface":   {"label": "Borde rojo superficial",     "weight": 1.6,  "bio": True,  "unit": "[0-1]"},
    "volcanic":  {"label": "Actividad volcánica",        "weight": -1.8, "bio": False, "unit": "[0-1]"},
    "techno":    {"label": "Tecnosignatura (CFC)",       "weight": 3.0,  "bio": True,  "unit": "[0-1]"},
    "causal_lag":{"label": "Desfase causal (fracción)",  "weight": 0.0,  "bio": True,  "unit": "[0-1]"},
}

ABIO_MODELS = [
    {"name": "Serpentinización + UV fotólisis", "base_score": 0.31, "key_driver": "volcanic"},
    {"name": "Vulcanismo + reducción CO₂",       "base_score": 0.24, "key_driver": "volcanic"},
    {"name": "Fotoquímica H₂-rica",              "base_score": 0.19, "key_driver": "ch4"},
    {"name": "Impactos + síntesis orgánica",     "base_score": 0.11, "key_driver": "surface"},
]
