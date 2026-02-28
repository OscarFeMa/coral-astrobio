"""
CORAL ASTROBIO v2.0 — Script Principal
Análisis integrado de exoplanetas para detección de vida.

Uso:
    python astrobio.py                          # Analiza todos los planetas
    python astrobio.py --planet "LHS 1140 b"   # Analiza un planeta específico
    python astrobio.py --rank                  # Ranking por probabilidad biológica
    python astrobio.py --report "LHS 1140 b"  # Genera reporte PDF
    python astrobio.py --all-reports           # Genera PDFs de todos los planetas
    python astrobio.py --sensitivity           # Análisis de sensibilidad
"""

import os
import sys
import argparse
from datetime import datetime

# Añadir directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.planets import PLANETS, EVIDENCE_DEFINITIONS
from core.bayesian_engine import (
    compute_bayesian_odds, rank_planets,
    sensitivity_analysis, most_informative_evidence
)
from core.transition_classifier import classify_transitions, rank_by_transition
from core.causal_analyzer import generate_synthetic_timeseries, analyze_causal_structure
from reports.pdf_generator import generate_report


# ---- COLORES ANSI ----
class C:
    BIO    = '\033[38;2;0;200;150m'
    ABIO   = '\033[38;2;255;77;109m'
    STAR   = '\033[38;2;76;201;240m'
    TECH   = '\033[38;2;255;209;102m'
    DIM    = '\033[38;2;58;90;122m'
    BRIGHT = '\033[38;2;232;244;255m'
    MUTED  = '\033[38;2;100;130;160m'
    RESET  = '\033[0m'
    BOLD   = '\033[1m'


def header():
    print(f"\n{C.STAR}{'='*70}{C.RESET}")
    print(f"{C.BRIGHT}{C.BOLD}  CORAL ASTROBIO v2.0 — Detección Agnóstica de Vida Extraterrestre{C.RESET}")
    print(f"{C.MUTED}  Motor Bayesiano · Clasificador PTSC · Analizador Causal{C.RESET}")
    print(f"{C.MUTED}  Proyecto Coral Multi-IA 2026 · {datetime.now().strftime('%Y-%m-%d %H:%M')}{C.RESET}")
    print(f"{C.STAR}{'='*70}{C.RESET}\n")


def print_planet_summary(planet, bayes, transition, causal):
    """Imprime resumen completo de un planeta."""
    name = planet["name"]
    v_color = C.BIO if "BIOLÓGICO" in bayes.verdict else C.ABIO if "ABIÓTICO" in bayes.verdict else C.TECH

    print(f"\n{C.STAR}{'─'*65}{C.RESET}")
    print(f"{C.BRIGHT}{C.BOLD}  {name}{C.RESET}  {C.MUTED}{planet.get('type','—')} · {planet.get('spectype','—')} · {planet.get('dist_pc','—')} pc{C.RESET}")
    print(f"{C.STAR}{'─'*65}{C.RESET}")

    # Bayesiano
    print(f"\n  {C.STAR}◈ CAPA 1 — MOTOR BAYESIANO (DBCE){C.RESET}")
    print(f"    Odds ratio bio/abio:  {C.BRIGHT}{bayes.odds_ratio:.3f}{C.RESET}")
    print(f"    P(biológico):         {C.BIO}{bayes.bio_probability:.3f}{C.RESET}  |  P(abiótico): {C.ABIO}{bayes.abio_probability:.3f}{C.RESET}")
    print(f"    IC 95%:               [{bayes.ci_95_low:.2f}, {bayes.ci_95_high:.2f}]")
    print(f"    Consiliencia:         {bayes.consilience_score:.3f}")
    print(f"    Veredicto:            {v_color}{C.BOLD}{bayes.verdict}{C.RESET}")

    if bayes.paradigm_shift:
        print(f"    {C.TECH}⚠  ALERTA: Cambio de paradigma detectado (odds > 2.5){C.RESET}")

    # Modelos abióticos
    print(f"\n    {C.DIM}Modelos abióticos competidores:{C.RESET}")
    for m in bayes.active_abio_models:
        status = f"{C.ABIO}DOMINANTE{C.RESET}" if m["dominant"] else f"{C.MUTED}suprimido{C.RESET}"
        print(f"    · {m['name'][:40]:<40} {m['score']:.2f}  {status}")

    # Transición
    print(f"\n  {C.STAR}◈ CAPA 2 — CLASIFICADOR PTSC{C.RESET}")
    for i, (score, td) in enumerate(zip(transition.scores, [
        {"label":"T0→T1","name":"Química → Prebiótica"},
        {"label":"T1→T2","name":"Prebiótica → Metabolismo"},
        {"label":"T2→T3","name":"Metabolismo → Ecosistema"},
        {"label":"T3→T4","name":"Ecosistema → Tecnosfera"},
    ])):
        bar_len = int(score * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        marker = " ◄ DOMINANTE" if i == transition.dominant_idx else ""
        col = C.BIO if i == transition.dominant_idx else C.MUTED
        print(f"    {col}{td['label']} {bar} {score:.2f}{marker}{C.RESET}")

    print(f"    Estado continuo: {C.BRIGHT}{transition.continuous_state:.2f}{C.RESET}  |  "
          f"FN reducidos: {C.BIO}{transition.false_negative_reduction:.1f}%{C.RESET}  |  "
          f"Ventana: {C.STAR}{transition.detectability_window_ma:.0f} Ma{C.RESET}")

    # Causal
    print(f"\n  {C.STAR}◈ CAPA 3 — ANALIZADOR CAUSAL{C.RESET}")
    dir_color = C.BIO if causal.causal_direction == "BIO" else C.ABIO if causal.causal_direction == "ABIO" else C.TECH
    print(f"    Dirección causal:     {dir_color}{C.BOLD}{causal.causal_direction}{C.RESET}")
    print(f"    Desfase superficie→gas: {causal.causal_lag_days:.1f} días ({causal.causal_lag_fraction:.2f}×período)")
    print(f"    Sincronía de fase:    {causal.phase_synchrony:.3f}")
    print(f"    Entropía información: {causal.info_entropy:.3f}")
    print(f"    Dim. efectiva:        {causal.effective_dim:.2f}")
    print(f"    Firma no geológica:   {C.BRIGHT}{causal.non_geological:.3f}{C.RESET}")


def cmd_analyze_all():
    """Analiza todos los planetas."""
    header()
    print(f"{C.BRIGHT}Analizando {len(PLANETS)} planetas...{C.RESET}\n")

    for p in PLANETS:
        bayes = compute_bayesian_odds(p["evidences"], p["name"])
        transition = classify_transitions(p["evidences"], p["name"])
        sig = generate_synthetic_timeseries(p, seed=42)
        causal = analyze_causal_structure(sig)
        print_planet_summary(p, bayes, transition, causal)


def cmd_analyze_one(planet_name: str):
    """Analiza un planeta específico con análisis completo."""
    header()
    planet = next((p for p in PLANETS if planet_name.lower() in p["name"].lower()), None)
    if not planet:
        print(f"{C.ABIO}Error: Planeta '{planet_name}' no encontrado.{C.RESET}")
        print(f"Planetas disponibles: {', '.join(p['name'] for p in PLANETS)}")
        return

    bayes = compute_bayesian_odds(planet["evidences"], planet["name"])
    transition = classify_transitions(planet["evidences"], planet["name"])
    sig = generate_synthetic_timeseries(planet, seed=42)
    causal = analyze_causal_structure(sig)

    print_planet_summary(planet, bayes, transition, causal)

    # Análisis de sensibilidad
    print(f"\n  {C.STAR}◈ ANÁLISIS DE SENSIBILIDAD (n=500, σ=0.05){C.RESET}")
    sens = sensitivity_analysis(planet["evidences"], planet["name"], n_samples=500)
    print(f"    Odds medio:      {sens['mean_odds']:.3f} ± {sens['std_odds']:.3f}")
    print(f"    P5/P50/P95:      {sens['p5']:.3f} / {sens['p50']:.3f} / {sens['p95']:.3f}")
    print(f"    P(bio robusta):  {C.BIO}{sens['prob_bio_robust']:.1%}{C.RESET}")

    # Prioridad de observaciones
    print(f"\n  {C.STAR}◈ PRIORIDAD DE OBSERVACIONES FUTURAS{C.RESET}")
    priorities = most_informative_evidence(planet["evidences"], planet["name"])
    for key, delta in priorities[:5]:
        label = EVIDENCE_DEFINITIONS.get(key, {}).get("label", key)
        direction = f"{C.BIO}↑BIO{C.RESET}" if delta > 0 else f"{C.ABIO}↑ABIO{C.RESET}"
        print(f"    · {label:<35} Δodds={delta:+.3f}  {direction}")


def cmd_rank():
    """Muestra ranking de planetas por probabilidad biológica."""
    header()
    ranked = rank_planets(PLANETS)

    print(f"{C.BRIGHT}{'RANKING':<4} {'PLANETA':<20} {'P(BIO)':>7} {'ODDS':>7} {'VEREDICTO':<20} {'TRANSICIÓN'}{C.RESET}")
    print(C.DIM + "─" * 80 + C.RESET)

    for i, (p, r) in enumerate(ranked):
        rank = i + 1
        v_color = C.BIO if "BIOLÓGICO" in r.verdict else C.ABIO if "ABIÓTICO" in r.verdict else C.TECH
        t = classify_transitions(p["evidences"], p["name"])
        t_labels = ["T0→T1", "T1→T2", "T2→T3", "T3→T4"]
        print(f"  {rank:<4} {p['name']:<20} {r.bio_probability:>7.3f} {r.odds_ratio:>7.3f} "
              f"{v_color}{r.verdict:<20}{C.RESET} {C.STAR}{t_labels[t.dominant_idx]}{C.RESET}")


def cmd_generate_report(planet_name: str, output_dir: str = "output"):
    """Genera reporte PDF de un planeta."""
    header()
    planet = next((p for p in PLANETS if planet_name.lower() in p["name"].lower()), None)
    if not planet:
        print(f"{C.ABIO}Error: Planeta '{planet_name}' no encontrado.{C.RESET}")
        return

    os.makedirs(output_dir, exist_ok=True)
    safe_name = planet["name"].replace(" ", "_").replace("/", "-")
    output_path = os.path.join(output_dir, f"astrobio_{safe_name}_{datetime.now().strftime('%Y%m%d')}.pdf")

    print(f"{C.BRIGHT}Generando reporte para {planet['name']}...{C.RESET}")

    bayes = compute_bayesian_odds(planet["evidences"], planet["name"])
    transition = classify_transitions(planet["evidences"], planet["name"])
    sig = generate_synthetic_timeseries(planet, seed=42)
    causal = analyze_causal_structure(sig)

    path = generate_report(planet, bayes, transition, causal, sig, output_path)
    print(f"{C.BIO}✓ Reporte generado: {path}{C.RESET}")
    return path


def cmd_all_reports(output_dir: str = "output"):
    """Genera reportes PDF para todos los planetas."""
    header()
    os.makedirs(output_dir, exist_ok=True)
    print(f"{C.BRIGHT}Generando {len(PLANETS)} reportes PDF...{C.RESET}\n")

    paths = []
    for p in PLANETS:
        safe_name = p["name"].replace(" ", "_").replace("/", "-")
        output_path = os.path.join(output_dir, f"astrobio_{safe_name}.pdf")
        print(f"  · {p['name']:<25}", end=" ", flush=True)

        bayes = compute_bayesian_odds(p["evidences"], p["name"])
        transition = classify_transitions(p["evidences"], p["name"])
        sig = generate_synthetic_timeseries(p, seed=42)
        causal = analyze_causal_structure(sig)

        path = generate_report(p, bayes, transition, causal, sig, output_path)
        paths.append(path)
        print(f"{C.BIO}✓{C.RESET}")

    print(f"\n{C.BIO}Reportes generados en: {output_dir}/{C.RESET}")
    return paths


def cmd_sensitivity(planet_name: str):
    """Análisis de sensibilidad profundo para un planeta."""
    header()
    planet = next((p for p in PLANETS if planet_name.lower() in p["name"].lower()), None)
    if not planet:
        print(f"{C.ABIO}Error: '{planet_name}' no encontrado.{C.RESET}")
        return

    print(f"{C.BRIGHT}Análisis de sensibilidad: {planet['name']}{C.RESET}\n")

    for noise in [0.02, 0.05, 0.10, 0.20]:
        sens = sensitivity_analysis(planet["evidences"], planet["name"],
                                    n_samples=1000, noise_std=noise)
        print(f"  σ={noise:.2f}: P(bio)={sens['prob_bio_robust']:.1%}  "
              f"odds={sens['mean_odds']:.2f}±{sens['std_odds']:.2f}  "
              f"IC[{sens['p5']:.2f},{sens['p95']:.2f}]")


# ---- MAIN ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CORAL ASTROBIO v2.0 — Análisis de exoplanetas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--planet",      type=str, help="Nombre del planeta a analizar")
    parser.add_argument("--rank",        action="store_true", help="Ranking por probabilidad biológica")
    parser.add_argument("--report",      type=str, help="Generar reporte PDF de un planeta")
    parser.add_argument("--all-reports", action="store_true", help="Generar PDFs de todos los planetas")
    parser.add_argument("--sensitivity", type=str, help="Análisis de sensibilidad de un planeta")
    parser.add_argument("--output-dir",  type=str, default="output", help="Directorio de salida")

    args = parser.parse_args()

    if args.rank:
        cmd_rank()
    elif args.planet:
        cmd_analyze_one(args.planet)
    elif args.report:
        cmd_generate_report(args.report, args.output_dir)
    elif args.all_reports:
        cmd_all_reports(args.output_dir)
    elif args.sensitivity:
        cmd_sensitivity(args.sensitivity)
    else:
        # Sin argumentos: análisis completo + ranking
        cmd_rank()
        print(f"\n{C.MUTED}Usa --planet 'nombre' para análisis detallado, --report para PDF{C.RESET}\n")
