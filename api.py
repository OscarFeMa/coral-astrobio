"""
CORAL ASTROBIO v3.0 — API REST
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
except ImportError:
    pass

import traceback
import tempfile
import numpy as np
from flask import Flask, jsonify, request, send_file

from data.planets import PLANETS, EVIDENCE_DEFINITIONS, ABIO_MODELS
from core.bayesian_engine import compute_bayesian_odds, rank_planets, sensitivity_analysis, most_informative_evidence
from core.transition_classifier import classify_transitions, TRANSITION_DEFINITIONS
from core.causal_analyzer import generate_synthetic_timeseries, analyze_causal_structure

app = Flask(__name__)

class NumpyJSON(app.json_provider_class):
    def default(self, o):
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)

app.json_provider_class = NumpyJSON
app.json = NumpyJSON(app)

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

def find_planet(name):
    return next((p for p in PLANETS if name.lower() in p["name"].lower()), None)

def safe_float(v):
    try: return round(float(v), 4)
    except: return None

def planet_full_analysis(planet, custom_evidences=None):
    ev = custom_evidences if custom_evidences else planet["evidences"]
    bayes = compute_bayesian_odds(ev, planet["name"])
    transition = classify_transitions(ev, planet["name"])
    sig = generate_synthetic_timeseries(planet, seed=42)
    causal = analyze_causal_structure(sig)
    step = max(1, len(sig.time) // 200)
    return {
        "planet": {
            "name": planet["name"], "host": planet.get("host", "—"), "type": planet.get("type", "—"),
            "mass_e": planet.get("mass_earth"), "radius_e": planet.get("radius_earth"),
            "teq_k": planet.get("teq_k"), "dist_pc": planet.get("dist_pc"),
            "spectype": planet.get("spectype", "—"), "tsm": planet.get("tsm"),
            "period_days": planet.get("period_days"),
            "jwst": bool(planet.get("jwst", False)), "hz": bool(planet.get("hz", False)),
            "notes": planet.get("notes", ""), "evidences": ev,
        },
        "bayesian": {
            "odds_ratio": safe_float(bayes.odds_ratio),
            "bio_probability": safe_float(bayes.bio_probability),
            "abio_probability": safe_float(bayes.abio_probability),
            "consilience_score": safe_float(bayes.consilience_score),
            "log_odds": safe_float(bayes.log_odds),
            "ci_95_low": safe_float(bayes.ci_95_low),
            "ci_95_high": safe_float(bayes.ci_95_high),
            "verdict": str(bayes.verdict),
            "paradigm_shift": bool(bayes.paradigm_shift),
            "false_negative_risk": safe_float(bayes.false_negative_risk),
            "evidence_contributions": {k: safe_float(v) for k, v in bayes.evidence_contributions.items()},
            "active_abio_models": [
                {k: (bool(v) if isinstance(v, (bool, np.bool_)) else (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else str(v))) for k, v in m.items()}
                for m in bayes.active_abio_models
            ],
        },
        "transition": {
            "scores": [safe_float(s) for s in transition.scores],
            "transitions": [{"label": str(td["label"]), "name": str(td["name"]), "score": safe_float(s)} for td, s in zip(TRANSITION_DEFINITIONS, transition.scores)],
            "dominant_idx": int(transition.dominant_idx),
            "dominant_score": safe_float(transition.dominant_score),
            "dominant_label": str(transition.dominant_label),
            "dominant_name": str(transition.dominant_name),
            "continuous_state": safe_float(transition.continuous_state),
            "false_negative_reduction": safe_float(transition.false_negative_reduction),
            "detectability_window_ma": safe_float(transition.detectability_window_ma),
            "uncertainty": safe_float(transition.uncertainty),
            "definitions": [{"label": td["label"], "name": td["name"], "description": td["description"]} for td in TRANSITION_DEFINITIONS],
        },
        "causal": {
            "causal_lag_days": safe_float(causal.causal_lag_days),
            "causal_lag_fraction": safe_float(causal.causal_lag_fraction),
            "phase_synchrony": safe_float(causal.phase_synchrony),
            "info_entropy": safe_float(causal.info_entropy),
            "effective_dim": safe_float(causal.effective_dim),
            "granger_bio": safe_float(causal.granger_bio),
            "granger_abio": safe_float(causal.granger_abio),
            "causal_direction": str(causal.causal_direction),
            "non_geological": safe_float(causal.non_geological),
            "spectral_complexity": safe_float(causal.spectral_complexity),
        },
        "timeseries": {
            "time": sig.time[::step].tolist(),
            "surface": sig.surface_reflectance[::step].tolist(),
            "gas": sig.atmospheric_gas[::step].tolist(),
            "stellar": sig.stellar_flux[::step].tolist(),
        },
        "evidences": [{"key": k, "name": EVIDENCE_DEFINITIONS.get(k, {}).get("label", k), "value": float(v)} for k, v in ev.items()],
    }

@app.route("/api/status")
def status():
    return jsonify({"status": "online", "version": "3.0", "planet_count": len(PLANETS), "engines": ["DBCE", "PTSC", "CNA"], "data_source": "NASA Exoplanet Archive (curated)"})

@app.route("/api/planets")
def get_planets():
    ranked = rank_planets(PLANETS)
    result = []
    for planet, bayes in ranked:
        transition = classify_transitions(planet["evidences"], planet["name"])
        result.append({
            "name": planet["name"], "host": planet.get("host", "—"), "type": planet.get("type", "—"),
            "mass_e": planet.get("mass_earth"), "radius_e": planet.get("radius_earth"),
            "teq_k": planet.get("teq_k"), "dist_pc": planet.get("dist_pc"),
            "spectype": planet.get("spectype", "—"), "tsm": planet.get("tsm"),
            "jwst": bool(planet.get("jwst", False)), "hz": bool(planet.get("hz", False)),
            "bio_probability": safe_float(bayes.bio_probability),
            "odds_ratio": safe_float(bayes.odds_ratio),
            "verdict": str(bayes.verdict),
            "dominant_transition": str(transition.dominant_label),
            "consilience_score": safe_float(bayes.consilience_score),
        })
    return jsonify({"planets": result, "count": len(result)})

@app.route("/api/planet/<path:name>")
def get_planet(name):
    planet = find_planet(name)
    if not planet:
        return jsonify({"error": f"Planeta '{name}' no encontrado"}), 404
    try:
        return jsonify(planet_full_analysis(planet))
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route("/api/analyze", methods=["POST", "OPTIONS"])
def analyze_custom():
    if request.method == "OPTIONS": return jsonify({})
    data = request.get_json()
    if not data: return jsonify({"error": "Body JSON requerido"}), 400
    planet_name = data.get("planet_name", "Custom")
    custom_ev = data.get("evidences", {})
    planet = find_planet(planet_name) or {"name": planet_name, "type": "Desconocido", "period_days": 30.0, "evidences": custom_ev}
    try:
        return jsonify(planet_full_analysis(planet, custom_evidences=custom_ev))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/rank")
def get_rank():
    ranked = rank_planets(PLANETS)
    return jsonify({"ranking": [{"rank": i+1, "name": p["name"], "bio_probability": safe_float(r.bio_probability), "odds_ratio": safe_float(r.odds_ratio), "verdict": str(r.verdict), "consilience": safe_float(r.consilience_score)} for i, (p, r) in enumerate(ranked)]})

@app.route("/api/sensitivity/<path:name>")
def get_sensitivity(name):
    planet = find_planet(name)
    if not planet: return jsonify({"error": f"Planeta '{name}' no encontrado"}), 404
    results = {}
    for noise in [0.02, 0.05, 0.10, 0.20]:
        results[str(noise)] = sensitivity_analysis(planet["evidences"], planet["name"], n_samples=500, noise_std=noise)
    priorities = most_informative_evidence(planet["evidences"], planet["name"])
    return jsonify({"planet": planet["name"], "sensitivity_by_noise": results, "priority_observations": [{"evidence_key": k, "label": EVIDENCE_DEFINITIONS.get(k, {}).get("label", k), "delta_odds": safe_float(d), "direction": "BIO" if d > 0 else "ABIO"} for k, d in priorities[:6]]})

@app.route("/api/evidences")
def get_evidences():
    return jsonify({"evidences": [{"key": k, "label": v["label"], "weight": v["weight"], "bio": v["bio"], "unit": v["unit"]} for k, v in EVIDENCE_DEFINITIONS.items()]})

@app.route("/api/compare")
def compare_planets():
    names_raw = request.args.get("names", "")
    names = [n.strip() for n in names_raw.split(",") if n.strip()]
    if not names:
        names = [p["name"] for p in sorted(PLANETS, key=lambda p: p.get("tsm", 0), reverse=True)[:6]]
    results = []
    for name in names:
        planet = find_planet(name)
        if not planet: continue
        try:
            d = planet_full_analysis(planet)
            results.append({"planet": d["planet"], "bayesian": d["bayesian"], "transition": d["transition"], "causal": d["causal"]})
        except Exception: continue
    return jsonify({"planets": results, "count": len(results)})

@app.route("/api/export/pdf/<path:name>")
def export_pdf(name):
    from reports.pdf_generator import generate_report
    planet = find_planet(name)
    if not planet: return jsonify({"error": f"'{name}' no encontrado"}), 404
    try:
        bayes = compute_bayesian_odds(planet["evidences"], planet["name"])
        transition = classify_transitions(planet["evidences"], planet["name"])
        sig = generate_synthetic_timeseries(planet, seed=42)
        causal = analyze_causal_structure(sig)
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        generate_report(planet, bayes, transition, causal, sig, tmp.name)
        tmp.close()
        return send_file(tmp.name, mimetype="application/pdf", as_attachment=True, download_name=f"astrobio_{planet['name'].replace(' ','_')}.pdf")
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

_memory = None

def get_memory():
    global _memory
    if _memory is None:
        try:
            from memory.supabase_client import CoralMemory
            _memory = CoralMemory()
        except Exception as e:
            print(f"[API] CoralMemory error: {e}")
    return _memory

@app.route("/api/memory/status")
def memory_status():
    mem = get_memory()
    if not mem: return jsonify({"online": False, "error": "CoralMemory no disponible"})
    return jsonify(mem.get_stats())

@app.route("/api/memory/save/<path:name>", methods=["POST", "OPTIONS"])
def memory_save_planet(name):
    if request.method == "OPTIONS": return jsonify({})
    planet = find_planet(name)
    if not planet: return jsonify({"error": f"'{name}' no encontrado"}), 404
    mem = get_memory()
    if not mem: return jsonify({"status": "offline"}), 503
    return jsonify(mem.save_planet_analysis(name, planet_full_analysis(planet)))

@app.route("/api/memory/list")
def memory_list():
    namespace = request.args.get("namespace", "astrobio")
    limit = int(request.args.get("limit", 50))
    mem = get_memory()
    if not mem: return jsonify({"entries": [], "offline": True})
    entries = mem.list_by_tema(namespace, limit=limit)
    return jsonify({"entries": entries, "count": len(entries), "namespace": namespace})

@app.route("/api/memory/search")
def memory_search():
    q = request.args.get("q", "")
    namespace = request.args.get("namespace", "astrobio")
    if not q: return jsonify({"error": "Parámetro q requerido"}), 400
    mem = get_memory()
    if not mem: return jsonify({"results": [], "offline": True})
    return jsonify({"results": mem.search_similar(q, namespace), "query": q, "namespace": namespace})

@app.route("/api/memory/save-coral", methods=["POST", "OPTIONS"])
def memory_save_coral():
    if request.method == "OPTIONS": return jsonify({})
    data = request.get_json()
    if not data or not data.get("content"): return jsonify({"error": "content requerido"}), 400
    mem = get_memory()
    if not mem: return jsonify({"status": "offline"}), 503
    return jsonify(mem.save_coral_note(title=data.get("title", "Nota Coral"), content=data["content"], subtema=data.get("subtema", "general")))

@app.route("/api/debug/routes")
def debug_routes():
    return jsonify({"routes": sorted([str(r) for r in app.url_map.iter_rules()]), "count": len(list(app.url_map.iter_rules()))})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"\n  CORAL ASTROBIO — API Server v3.0")
    print(f"  http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
