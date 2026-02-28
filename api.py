"""
CORAL ASTROBIO v2.0 — API REST
Servidor Flask que expone el backend Python como API.

Endpoints:
    GET  /api/planets              — Lista todos los planetas
    GET  /api/planet/<name>        — Análisis completo de un planeta
    POST /api/analyze              — Análisis con evidencias personalizadas
    GET  /api/rank                 — Ranking por probabilidad biológica
    GET  /api/sensitivity/<name>   — Análisis de sensibilidad
    GET  /api/status               — Estado del servidor
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Cargar .env automáticamente si existe
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
except ImportError:
    pass  # python-dotenv opcional

from flask import Flask, jsonify, request, make_response
from dataclasses import asdict
import traceback

from data.planets import PLANETS, EVIDENCE_DEFINITIONS, ABIO_MODELS
from core.bayesian_engine import (
    compute_bayesian_odds, rank_planets,
    sensitivity_analysis, most_informative_evidence
)
from core.transition_classifier import (
    classify_transitions, TRANSITION_DEFINITIONS
)
from core.causal_analyzer import (
    generate_synthetic_timeseries, analyze_causal_structure
)

app = Flask(__name__)

import numpy as np
class NumpyJSON(app.json_provider_class):
    def default(self, o):
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)
app.json_provider_class = NumpyJSON
app.json = NumpyJSON(app)
# ---- HELPERS ----

def planet_full_analysis(planet, custom_evidences=None):
    """Ejecuta las 3 capas y devuelve resultado serializable."""
    ev = custom_evidences if custom_evidences else planet["evidences"]

    bayes = compute_bayesian_odds(ev, planet["name"])
    transition = classify_transitions(ev, planet["name"])
    sig = generate_synthetic_timeseries(planet, seed=42)
    causal = analyze_causal_structure(sig)

    # Series temporales (muestra de 200 puntos para no sobrecargar)
    step = max(1, len(sig.time) // 200)
    timeseries = {
        "time": sig.time[::step].tolist(),
        "surface": sig.surface_reflectance[::step].tolist(),
        "gas": sig.atmospheric_gas[::step].tolist(),
        "stellar": sig.stellar_flux[::step].tolist(),
    }

    return {
        "planet": {
            "name": planet["name"],
            "host": planet.get("host", "—"),
            "type": planet.get("type", "—"),
            "mass_earth": planet.get("mass_earth"),
            "radius_earth": planet.get("radius_earth"),
            "teq_k": planet.get("teq_k"),
            "dist_pc": planet.get("dist_pc"),
            "spectype": planet.get("spectype", "—"),
            "tsm": planet.get("tsm"),
            "period_days": planet.get("period_days"),
            "jwst": planet.get("jwst", False),
            "hz": planet.get("hz", False),
            "notes": planet.get("notes", ""),
            "evidences": ev,
        },
        "bayesian": {
            "odds_ratio": round(bayes.odds_ratio, 4),
            "bio_probability": round(bayes.bio_probability, 4),
            "abio_probability": round(bayes.abio_probability, 4),
            "consilience_score": round(bayes.consilience_score, 4),
            "log_odds": round(bayes.log_odds, 4),
            "ci_95_low": round(bayes.ci_95_low, 4),
            "ci_95_high": round(bayes.ci_95_high, 4),
            "verdict": bayes.verdict,
            "paradigm_shift": bayes.paradigm_shift,
            "false_negative_risk": round(bayes.false_negative_risk, 4),
            "evidence_contributions": {
                k: round(v, 4) for k, v in bayes.evidence_contributions.items()
            },
            "active_abio_models": bayes.active_abio_models,
        },
        "transition": {
            "scores": [round(s, 4) for s in transition.scores],
            "dominant_idx": transition.dominant_idx,
            "dominant_score": round(transition.dominant_score, 4),
            "dominant_label": transition.dominant_label,
            "dominant_name": transition.dominant_name,
            "continuous_state": round(transition.continuous_state, 4),
            "false_negative_reduction": round(transition.false_negative_reduction, 2),
            "detectability_window_ma": round(transition.detectability_window_ma, 1),
            "uncertainty": round(transition.uncertainty, 4),
            "definitions": [
                {"label": td["label"], "name": td["name"], "description": td["description"]}
                for td in TRANSITION_DEFINITIONS
            ],
        },
        "causal": {
            "causal_lag_days": round(causal.causal_lag_days, 2),
            "causal_lag_fraction": round(causal.causal_lag_fraction, 4),
            "phase_synchrony": round(causal.phase_synchrony, 4),
            "info_entropy": round(causal.info_entropy, 4),
            "effective_dim": round(causal.effective_dim, 3),
            "granger_bio": round(causal.granger_bio, 4),
            "granger_abio": round(causal.granger_abio, 4),
            "causal_direction": causal.causal_direction,
            "non_geological": round(causal.non_geological, 4),
            "spectral_complexity": round(causal.spectral_complexity, 4),
        },
        "timeseries": timeseries,
    }


def find_planet(name):
    return next(
        (p for p in PLANETS if name.lower() in p["name"].lower()),
        None
    )


# ---- ENDPOINTS ----

@app.route("/api/status")
def status():
    return jsonify({
        "status": "online",
        "version": "2.0",
        "planet_count": len(PLANETS),
        "engines": ["DBCE", "PTSC", "CNA"],
        "data_source": "NASA Exoplanet Archive (curated)",
    })


@app.route("/api/planets")
def get_planets():
    """Lista todos los planetas con análisis rápido."""
    ranked = rank_planets(PLANETS)
    result = []
    for planet, bayes in ranked:
        transition = classify_transitions(planet["evidences"], planet["name"])
        result.append({
            "name": planet["name"],
            "host": planet.get("host", "—"),
            "type": planet.get("type", "—"),
            "mass_earth": planet.get("mass_earth"),
            "radius_earth": planet.get("radius_earth"),
            "teq_k": planet.get("teq_k"),
            "dist_pc": planet.get("dist_pc"),
            "spectype": planet.get("spectype", "—"),
            "tsm": planet.get("tsm"),
            "jwst": planet.get("jwst", False),
            "hz": planet.get("hz", False),
            "bio_probability": round(bayes.bio_probability, 4),
            "odds_ratio": round(bayes.odds_ratio, 4),
            "verdict": bayes.verdict,
            "dominant_transition": transition.dominant_label,
            "consilience_score": round(bayes.consilience_score, 4),
        })
    return jsonify({"planets": result, "count": len(result)})


@app.route("/api/planet/<path:name>")
def get_planet(name):
    """Análisis completo de un planeta."""
    planet = find_planet(name)
    if not planet:
        return jsonify({"error": f"Planeta '{name}' no encontrado"}), 404
    try:
        result = planet_full_analysis(planet)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze_custom():
    """
    Análisis con evidencias personalizadas.
    Body JSON: { "planet_name": "LHS 1140 b", "evidences": {...} }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Body JSON requerido"}), 400

    planet_name = data.get("planet_name", "Custom")
    custom_ev = data.get("evidences", {})

    planet = find_planet(planet_name) or {
        "name": planet_name,
        "type": "Desconocido",
        "period_days": 30.0,
        "evidences": custom_ev,
    }

    try:
        result = planet_full_analysis(planet, custom_evidences=custom_ev)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rank")
def get_rank():
    """Ranking de todos los planetas por probabilidad biológica."""
    ranked = rank_planets(PLANETS)
    return jsonify({
        "ranking": [
            {
                "rank": i + 1,
                "name": p["name"],
                "bio_probability": round(r.bio_probability, 4),
                "odds_ratio": round(r.odds_ratio, 4),
                "verdict": r.verdict,
                "consilience": round(r.consilience_score, 4),
                "paradigm_shift": r.paradigm_shift,
            }
            for i, (p, r) in enumerate(ranked)
        ]
    })


@app.route("/api/sensitivity/<path:name>")
def get_sensitivity(name):
    """Análisis de sensibilidad para un planeta."""
    planet = find_planet(name)
    if not planet:
        return jsonify({"error": f"Planeta '{name}' no encontrado"}), 404

    results = {}
    for noise in [0.02, 0.05, 0.10, 0.20]:
        s = sensitivity_analysis(
            planet["evidences"], planet["name"],
            n_samples=500, noise_std=noise
        )
        results[str(noise)] = s

    priorities = most_informative_evidence(planet["evidences"], planet["name"])

    return jsonify({
        "planet": planet["name"],
        "sensitivity_by_noise": results,
        "priority_observations": [
            {
                "evidence_key": k,
                "label": EVIDENCE_DEFINITIONS.get(k, {}).get("label", k),
                "delta_odds": round(d, 4),
                "direction": "BIO" if d > 0 else "ABIO",
            }
            for k, d in priorities[:6]
        ],
    })


@app.route("/api/evidences")
def get_evidences():
    """Definiciones de todas las evidencias disponibles."""
    return jsonify({
        "evidences": [
            {
                "key": k,
                "label": v["label"],
                "weight": v["weight"],
                "bio": v["bio"],
                "unit": v["unit"],
            }
            for k, v in EVIDENCE_DEFINITIONS.items()
        ]
    })



@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


if __name__ == "__main__":
    print("\n  CORAL ASTROBIO — API Server v2.0")
    print("  http://localhost:5050\n")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5050)), debug=False)
# Monkey-patch: añadir CORS headers a todas las respuestas

# ================================================================
# SUPABASE MEMORY ENDPOINTS
# ================================================================

_memory = None

def get_memory():
    global _memory
    if _memory is None:
        try:
            from memory.supabase_client import CoralMemory
            _memory = CoralMemory()
        except Exception:
            _memory = None
    return _memory


@app.route("/api/memory/status")
def memory_status():
    mem = get_memory()
    if not mem:
        return jsonify({"online": False, "error": "CoralMemory no disponible"})
    return jsonify(mem.get_stats())


@app.route("/api/memory/save/<path:name>", methods=["POST"])
def memory_save_planet(name):
    """Guarda el análisis de un planeta en Supabase namespace=astrobio."""
    planet = find_planet(name)
    if not planet:
        return jsonify({"error": f"Planeta '{name}' no encontrado"}), 404

    analysis = planet_full_analysis(planet)
    mem = get_memory()
    if not mem:
        return jsonify({"status": "offline", "message": "Supabase no configurado"}), 503

    result = mem.save_planet_analysis(name, analysis)
    return jsonify(result)


@app.route("/api/memory/list")
def memory_list():
    """Lista entradas del namespace astrobio."""
    namespace = request.args.get("namespace", "astrobio")
    limit = int(request.args.get("limit", 50))
    mem = get_memory()
    if not mem:
        return jsonify({"entries": [], "offline": True})
    entries = mem.list_entries(namespace, limit)
    return jsonify({"entries": entries, "count": len(entries), "namespace": namespace})


@app.route("/api/memory/search")
def memory_search():
    """Búsqueda semántica en memoria."""
    q = request.args.get("q", "")
    namespace = request.args.get("namespace", "astrobio")
    if not q:
        return jsonify({"error": "Parámetro q requerido"}), 400
    mem = get_memory()
    if not mem:
        return jsonify({"results": [], "offline": True})
    results = mem.search_similar(q, namespace)
    return jsonify({"results": results, "query": q, "namespace": namespace})


@app.route("/api/memory/save-coral", methods=["POST"])
def memory_save_coral():
    """Guarda una nota en el namespace coral."""
    data = request.get_json()
    if not data or not data.get("content"):
        return jsonify({"error": "content requerido"}), 400
    mem = get_memory()
    if not mem:
        return jsonify({"status": "offline"}), 503
    result = mem.save_coral_note(
        title=data.get("title", "Nota Coral"),
        content=data["content"],
        tags=data.get("tags", []),
    )
    return jsonify(result)


# ================================================================
# COMPARE ENDPOINT (multi-planeta)
# ================================================================

@app.route("/api/compare")
def compare_planets():
    """Compara múltiples planetas. ?names=LHS+1140+b,K2-18+b,TRAPPIST-1+e"""
    names_raw = request.args.get("names", "")
    names = [n.strip() for n in names_raw.split(",") if n.strip()]
    if not names:
        # Default: top 6 por TSM
        sorted_p = sorted(PLANETS, key=lambda p: p.get("tsm", 0), reverse=True)
        names = [p["name"] for p in sorted_p[:6]]

    results = []
    for name in names:
        planet = find_planet(name)
        if not planet:
            continue
        bayes = compute_bayesian_odds(planet["evidences"], planet["name"])
        transition = classify_transitions(planet["evidences"], planet["name"])
        sig = generate_synthetic_timeseries(planet, seed=42)
        causal = analyze_causal_structure(sig)
        results.append({
            "name": planet["name"],
            "type": planet.get("type", "—"),
            "teq_k": planet.get("teq_k"),
            "dist_pc": planet.get("dist_pc"),
            "tsm": planet.get("tsm"),
            "jwst": planet.get("jwst", False),
            "hz": planet.get("hz", False),
            "bio_probability": round(bayes.bio_probability, 4),
            "odds_ratio": round(bayes.odds_ratio, 4),
            "verdict": bayes.verdict,
            "consilience": round(bayes.consilience_score, 4),
            "paradigm_shift": bayes.paradigm_shift,
            "dominant_transition": transition.dominant_label,
            "transition_scores": [round(s, 3) for s in transition.scores],
            "causal_direction": causal.causal_direction,
            "non_geological": round(causal.non_geological, 4),
            "phase_synchrony": round(causal.phase_synchrony, 4),
            "evidences": planet["evidences"],
        })

    return jsonify({"comparison": results, "count": len(results)})


# ================================================================
# PDF EXPORT ENDPOINT
# ================================================================

@app.route("/api/export/pdf/<path:name>")
def export_pdf(name):
    """Genera y sirve un reporte PDF como descarga."""
    import tempfile
    from flask import send_file
    from reports.pdf_generator import generate_report

    planet = find_planet(name)
    if not planet:
        return jsonify({"error": f"'{name}' no encontrado"}), 404

    bayes = compute_bayesian_odds(planet["evidences"], planet["name"])
    transition = classify_transitions(planet["evidences"], planet["name"])
    sig = generate_synthetic_timeseries(planet, seed=42)
    causal = analyze_causal_structure(sig)

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    generate_report(planet, bayes, transition, causal, sig, tmp.name)
    tmp.close()

    safe = planet["name"].replace(" ", "_")
    return send_file(
        tmp.name,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"astrobio_{safe}.pdf",
    )

@app.route("/api/debug/routes")
def debug_routes():
    routes = [str(r) for r in app.url_map.iter_rules()]
    return jsonify({"routes": routes, "count": len(routes)})
