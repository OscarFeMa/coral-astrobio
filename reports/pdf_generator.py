"""
CORAL ASTROBIO — Generador de Reportes PDF
Exporta análisis completo de un planeta a PDF profesional.
"""

import os
import io
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, KeepTogether
)
from reportlab.pdfgen import canvas as rl_canvas

from core.bayesian_engine import BayesianResult, sensitivity_analysis, most_informative_evidence
from core.transition_classifier import TransitionResult, TRANSITION_DEFINITIONS
from core.causal_analyzer import CausalResult, CausalSignal
from data.planets import EVIDENCE_DEFINITIONS

# ---- PALETA ----
COL_VOID    = colors.HexColor("#03050a")
COL_DEEP    = colors.HexColor("#070d1a")
COL_BIO     = colors.HexColor("#00c896")
COL_ABIO    = colors.HexColor("#ff4d6d")
COL_STELLAR = colors.HexColor("#4cc9f0")
COL_TECH    = colors.HexColor("#ffd166")
COL_MUTED   = colors.HexColor("#3a5a7a")
COL_TEXT    = colors.HexColor("#c8dff0")
COL_BRIGHT  = colors.HexColor("#e8f4ff")
COL_DIM     = colors.HexColor("#1a2a3a")
WHITE       = colors.white
BLACK       = colors.black


def generate_report(
    planet: Dict,
    bayes: BayesianResult,
    transition: TransitionResult,
    causal: CausalResult,
    signal: CausalSignal,
    output_path: str,
) -> str:
    """
    Genera un reporte PDF completo del análisis de un planeta.

    Args:
        planet: Dict con metadatos del planeta
        bayes: Resultado del motor bayesiano
        transition: Resultado del clasificador PTSC
        causal: Resultado del analizador causal
        signal: Series temporales
        output_path: Ruta de salida del PDF

    Returns:
        Ruta al PDF generado
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title=f"CORAL ASTROBIO — {planet['name']}",
        author="CORAL ASTROBIO v2.0 / Proyecto Coral Multi-IA 2026",
    )

    styles = _build_styles()
    story = []

    # ---- PORTADA ----
    story += _build_header(planet, bayes, styles)
    story.append(Spacer(1, 0.5*cm))

    # ---- RESUMEN EJECUTIVO ----
    story += _build_executive_summary(planet, bayes, transition, causal, styles)
    story.append(Spacer(1, 0.4*cm))

    # ---- GRÁFICOS ----
    charts_img = _build_charts_image(bayes, transition, causal, signal)
    story.append(RLImage(charts_img, width=17*cm, height=10*cm))
    story.append(Spacer(1, 0.4*cm))

    # ---- CAPA 1: BAYESIANO ----
    story += _build_bayesian_section(bayes, planet, styles)
    story.append(Spacer(1, 0.4*cm))

    # ---- CAPA 2: TRANSICIÓN ----
    story += _build_transition_section(transition, styles)
    story.append(Spacer(1, 0.4*cm))

    # ---- CAPA 3: CAUSAL ----
    story += _build_causal_section(causal, styles)
    story.append(Spacer(1, 0.4*cm))

    # ---- OBSERVACIONES PRIORITARIAS ----
    story += _build_priority_observations(planet, bayes, styles)
    story.append(Spacer(1, 0.4*cm))

    # ---- PIE ----
    story += _build_footer_section(styles)

    doc.build(story, onFirstPage=_add_background, onLaterPages=_add_background)
    return output_path


def _add_background(c, doc):
    """Fondo oscuro para cada página."""
    c.saveState()
    c.setFillColor(COL_VOID)
    c.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
    # Línea superior
    c.setStrokeColor(COL_MUTED)
    c.setLineWidth(0.5)
    c.line(2*cm, A4[1]-1.2*cm, A4[0]-2*cm, A4[1]-1.2*cm)
    # Número de página
    c.setFont("Courier", 7)
    c.setFillColor(COL_MUTED)
    c.drawRightString(A4[0]-2*cm, 1.2*cm, f"CORAL ASTROBIO v2.0 · {datetime.now().strftime('%Y-%m-%d')}")
    c.restoreState()


def _build_styles():
    """Estilos de texto."""
    base = getSampleStyleSheet()
    styles = {}

    def s(name, font, size, color, align=TA_LEFT, space_before=0, space_after=2):
        styles[name] = ParagraphStyle(
            name, fontName=font, fontSize=size,
            textColor=color, alignment=align,
            spaceAfter=space_after*mm, spaceBefore=space_before*mm,
            leading=size*1.4
        )

    s("eyebrow",    "Courier",        7,  COL_BIO,     TA_LEFT, 0, 2)
    s("title",      "Courier-Bold",   18, COL_BRIGHT,  TA_LEFT, 0, 4)
    s("subtitle",   "Courier",        9,  COL_TEXT,    TA_LEFT, 0, 2)
    s("section",    "Courier-Bold",   9,  COL_STELLAR, TA_LEFT, 4, 2)
    s("body",       "Courier",        8,  COL_TEXT,    TA_LEFT, 0, 2)
    s("small",      "Courier",        7,  COL_MUTED,   TA_LEFT, 0, 1)
    s("verdict_bio","Courier-Bold",   11, COL_BIO,     TA_CENTER, 2, 2)
    s("verdict_abio","Courier-Bold",  11, COL_ABIO,    TA_CENTER, 2, 2)
    s("verdict_unc","Courier-Bold",   11, COL_TECH,    TA_CENTER, 2, 2)
    s("note",       "Courier-Oblique",7,  COL_MUTED,   TA_LEFT, 0, 1)
    return styles


def _build_header(planet, bayes, styles):
    items = []
    items.append(Paragraph("◈ CORAL ASTROBIO v2.0 · ANÁLISIS DE EXOPLANETA", styles["eyebrow"]))
    items.append(Paragraph(planet["name"], styles["title"]))
    meta = (f"{planet.get('type','—')} · {planet.get('spectype','—')} · "
            f"{planet.get('dist_pc','—')} pc · T_eq: {planet.get('teq_k','—')} K · "
            f"TSM: {planet.get('tsm','—')} · "
            f"{'JWST ✓' if planet.get('jwst') else 'Sin datos JWST'}")
    items.append(Paragraph(meta, styles["subtitle"]))
    items.append(HRFlowable(width="100%", thickness=0.5, color=COL_MUTED, spaceAfter=4))

    # Tabla de parámetros físicos
    data = [
        ["PARÁMETRO", "VALOR", "PARÁMETRO", "VALOR"],
        ["Masa (M⊕)", f"{planet.get('mass_earth','—')}", "Radio (R⊕)", f"{planet.get('radius_earth','—')}"],
        ["T_eq (K)", f"{planet.get('teq_k','—')}", "Período (días)", f"{planet.get('period_days','—')}"],
        ["Distancia (pc)", f"{planet.get('dist_pc','—')}", "Tipo estelar", f"{planet.get('spectype','—')}"],
        ["Zona habitable", "SÍ" if planet.get('hz') else "NO", "JWST observado", "SÍ" if planet.get('jwst') else "NO"],
    ]
    t = Table(data, colWidths=[4.5*cm, 3.5*cm, 4.5*cm, 3.5*cm])
    t.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,0), 'Courier-Bold'),
        ('FONTNAME', (0,1), (-1,-1), 'Courier'),
        ('FONTSIZE', (0,0), (-1,-1), 7),
        ('TEXTCOLOR', (0,0), (-1,0), COL_STELLAR),
        ('TEXTCOLOR', (0,1), (-1,-1), COL_TEXT),
        ('BACKGROUND', (0,0), (-1,0), COL_DIM),
        ('BACKGROUND', (0,1), (-1,-1), COL_DEEP),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [COL_DEEP, COL_VOID]),
        ('GRID', (0,0), (-1,-1), 0.3, COL_DIM),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ]))
    items.append(t)
    return items


def _build_executive_summary(planet, bayes, transition, causal, styles):
    items = []
    items.append(Paragraph("RESUMEN EJECUTIVO", styles["section"]))

    # Tabla de veredictos
    v_color = COL_BIO if "BIOLÓGICO" in bayes.verdict else COL_ABIO if "ABIÓTICO" in bayes.verdict else COL_TECH
    data = [
        ["MOTOR BAYESIANO (DBCE)", "CLASIFICADOR PTSC", "ANALIZADOR CAUSAL"],
        [bayes.verdict, f"T{transition.dominant_idx} — {transition.dominant_label}", f"CAUSAL: {causal.causal_direction}"],
        [f"Odds: {bayes.odds_ratio:.2f} | P(bio): {bayes.bio_probability:.2f}",
         f"{transition.dominant_name[:28]}",
         f"No geológico: {causal.non_geological:.2f}"],
    ]
    t = Table(data, colWidths=[5.5*cm, 5.5*cm, 5.5*cm])
    t.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,0), 'Courier-Bold'),
        ('FONTNAME', (0,1), (-1,-1), 'Courier'),
        ('FONTSIZE', (0,0), (-1,0), 7),
        ('FONTSIZE', (0,1), (-1,-1), 8),
        ('TEXTCOLOR', (0,0), (-1,0), COL_MUTED),
        ('TEXTCOLOR', (0,1), (-1,1), v_color),
        ('TEXTCOLOR', (0,2), (-1,2), COL_TEXT),
        ('BACKGROUND', (0,0), (-1,0), COL_DIM),
        ('BACKGROUND', (0,1), (-1,-1), COL_DEEP),
        ('GRID', (0,0), (-1,-1), 0.3, COL_DIM),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    items.append(t)

    # Nota epistemológica
    if planet.get("notes"):
        items.append(Spacer(1, 2*mm))
        items.append(Paragraph(f"Nota: {planet['notes']}", styles["note"]))

    return items


def _build_charts_image(bayes, transition, causal, signal):
    """Genera imagen con los 3 gráficos principales."""
    fig = plt.figure(figsize=(11, 6.5), facecolor='#03050a')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    # ---- GRÁFICO 1: Odds ratio y distribución ----
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_odds_gauge(ax1, bayes)

    # ---- GRÁFICO 2: Transiciones ----
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_transitions(ax2, transition)

    # ---- GRÁFICO 3: Contribuciones de evidencia ----
    ax3 = fig.add_subplot(gs[0, 2])
    _plot_evidence_contributions(ax3, bayes)

    # ---- GRÁFICO 4: Series temporales causales (ancho completo) ----
    ax4 = fig.add_subplot(gs[1, :])
    _plot_timeseries(ax4, signal, causal)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor='#03050a', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def _plot_odds_gauge(ax, bayes):
    ax.set_facecolor('#070d1a')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')

    # Barra horizontal bio vs abio
    bio_w = bayes.bio_probability
    ax.barh(0.7, bio_w, height=0.12, color='#00c896', left=0, alpha=0.9)
    ax.barh(0.7, 1-bio_w, height=0.12, color='#ff4d6d', left=bio_w, alpha=0.9)
    ax.text(0.5, 0.55, f"ODDS RATIO: {bayes.odds_ratio:.2f}", color='#e8f4ff',
            ha='center', va='center', fontsize=11, fontfamily='monospace', fontweight='bold')
    ax.text(0.5, 0.42, bayes.verdict, color='#00c896' if 'BIOLÓGICO' in bayes.verdict
            else '#ff4d6d' if 'ABIÓTICO' in bayes.verdict else '#ffd166',
            ha='center', va='center', fontsize=8, fontfamily='monospace')
    ax.text(0.02, 0.7, f"BIO\n{bayes.bio_probability:.0%}", color='#00c896',
            ha='left', va='center', fontsize=7, fontfamily='monospace')
    ax.text(0.98, 0.7, f"ABIO\n{bayes.abio_probability:.0%}", color='#ff4d6d',
            ha='right', va='center', fontsize=7, fontfamily='monospace')
    ax.text(0.5, 0.25, f"Consiliencia: {bayes.consilience_score:.2f}", color='#c8dff0',
            ha='center', va='center', fontsize=7, fontfamily='monospace')
    ax.text(0.5, 0.12, f"IC 95%: [{bayes.ci_95_low:.2f}, {bayes.ci_95_high:.2f}]",
            color='#3a5a7a', ha='center', va='center', fontsize=6, fontfamily='monospace')
    ax.set_title("MOTOR BAYESIANO", color='#4cc9f0', fontsize=7, fontfamily='monospace', pad=4)


def _plot_transitions(ax, transition):
    ax.set_facecolor('#070d1a')
    ax.axis('off')
    colors_t = ['#5a6d8a', '#7b8fff', '#a8e6cf', '#ffd166']
    labels = ['T0→T1', 'T1→T2', 'T2→T3', 'T3→T4']

    for i, (s, c, l) in enumerate(zip(transition.scores, colors_t, labels)):
        y = 0.82 - i * 0.20
        ax.barh(y, s, height=0.10, color=c, alpha=0.85 if i == transition.dominant_idx else 0.45)
        ax.text(-0.02, y, l, color='#c8dff0', ha='right', va='center', fontsize=6.5, fontfamily='monospace')
        ax.text(s + 0.02, y, f"{s:.2f}", color=c, ha='left', va='center', fontsize=7, fontfamily='monospace')

    ax.set_xlim(-0.15, 1.15)
    ax.text(0.5, -0.04, f"Estado dominante: {transition.dominant_label}", color='#4cc9f0',
            ha='center', va='center', fontsize=6.5, fontfamily='monospace',
            transform=ax.transAxes)
    ax.set_title("CLASIFICADOR PTSC", color='#4cc9f0', fontsize=7, fontfamily='monospace', pad=4)


def _plot_evidence_contributions(ax, bayes):
    ax.set_facecolor('#070d1a')
    items = sorted(bayes.evidence_contributions.items(), key=lambda x: x[1])
    labels = [EVIDENCE_DEFINITIONS.get(k, {}).get("label", k)[:18] for k, _ in items]
    values = [v for _, v in items]
    bar_colors = ['#00c896' if v > 0 else '#ff4d6d' for v in values]

    y_pos = range(len(labels))
    ax.barh(list(y_pos), values, color=bar_colors, alpha=0.8, height=0.6)
    ax.axvline(0, color='#3a5a7a', linewidth=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=5.5, color='#c8dff0', fontfamily='monospace')
    ax.set_xlabel("Contribución log-odds", fontsize=6, color='#3a5a7a', fontfamily='monospace')
    ax.tick_params(axis='x', colors='#3a5a7a', labelsize=6)
    ax.spines[:].set_color('#1a2a3a')
    ax.set_facecolor('#070d1a')
    ax.set_title("CONTRIBUCIONES EVIDENCIA", color='#4cc9f0', fontsize=7, fontfamily='monospace', pad=4)


def _plot_timeseries(ax, signal, causal):
    ax.set_facecolor('#070d1a')
    t = signal.time
    # Normalizar para visualización
    s = (signal.surface_reflectance - np.mean(signal.surface_reflectance)) / (np.std(signal.surface_reflectance) + 1e-10)
    g = (signal.atmospheric_gas - np.mean(signal.atmospheric_gas)) / (np.std(signal.atmospheric_gas) + 1e-10)

    ax.plot(t, s, color='#00c896', linewidth=0.9, label='Reflectancia superficial', alpha=0.9)
    ax.plot(t, g, color='#4cc9f0', linewidth=0.9, label='Gas atmosférico', alpha=0.8)

    # Marcar desfase causal
    if causal.causal_lag_days > 0:
        ax.axvline(causal.causal_lag_days, color='#ffd166', linewidth=0.7,
                   linestyle='--', alpha=0.5, label=f"Δlag={causal.causal_lag_days:.1f}d")

    ax.set_xlabel("Tiempo (días)", fontsize=6.5, color='#3a5a7a', fontfamily='monospace')
    ax.set_ylabel("Señal normalizada", fontsize=6.5, color='#3a5a7a', fontfamily='monospace')
    ax.tick_params(colors='#3a5a7a', labelsize=6)
    ax.spines[:].set_color('#1a2a3a')
    ax.legend(fontsize=6, labelcolor='#c8dff0', facecolor='#070d1a',
              edgecolor='#1a2a3a', framealpha=0.8)
    ax.set_title(
        f"SERIES TEMPORALES CAUSALES · Dirección: {causal.causal_direction} · "
        f"Sincronía: {causal.phase_synchrony:.2f} · No-geológico: {causal.non_geological:.2f}",
        color='#4cc9f0', fontsize=7, fontfamily='monospace', pad=4
    )


def _build_bayesian_section(bayes, planet, styles):
    items = []
    items.append(HRFlowable(width="100%", thickness=0.3, color=COL_DIM))
    items.append(Paragraph("CAPA 1 — DYNAMIC BAYESIAN CONSILIENCE ENGINE (DBCE)", styles["section"]))

    # Tabla de evidencias
    data = [["EVIDENCIA", "VALOR", "CONTRIBUCIÓN LOG-ODDS", "IMPACTO"]]
    for key, defn in EVIDENCE_DEFINITIONS.items():
        if key == "causal_lag":
            continue
        # Acceder directamente a los valores de evidencia del planeta
        ev_val = planet.get("evidences", {}).get(key, 0.0)
        contrib = bayes.evidence_contributions.get(key, 0.0)
        impact = "+" if contrib > 0.05 else "−" if contrib < -0.05 else "≈"
        data.append([
            defn["label"],
            f"{ev_val:.2f}",
            f"{contrib:+.3f}",
            impact
        ])

    t = Table(data, colWidths=[6*cm, 2.5*cm, 5*cm, 2.5*cm])
    t.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,0), 'Courier-Bold'),
        ('FONTNAME', (0,1), (-1,-1), 'Courier'),
        ('FONTSIZE', (0,0), (-1,-1), 7),
        ('TEXTCOLOR', (0,0), (-1,0), COL_STELLAR),
        ('TEXTCOLOR', (0,1), (-1,-1), COL_TEXT),
        ('BACKGROUND', (0,0), (-1,0), COL_DIM),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [COL_DEEP, COL_VOID]),
        ('GRID', (0,0), (-1,-1), 0.3, COL_DIM),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
    ]))
    items.append(t)

    # Modelos abióticos
    items.append(Spacer(1, 2*mm))
    items.append(Paragraph("Modelos Abióticos Competidores:", styles["small"]))
    for m in bayes.active_abio_models:
        status = "DOMINANTE" if m["dominant"] else "suprimido"
        color_s = '<font color="#ff4d6d">' if m["dominant"] else '<font color="#3a5a7a">'
        items.append(Paragraph(
            f'  {m["name"]}: score={m["score"]:.2f} — {color_s}{status}</font>',
            styles["small"]
        ))

    if bayes.paradigm_shift:
        items.append(Spacer(1, 2*mm))
        items.append(Paragraph(
            f'⚠ ALERTA: Odds ratio {bayes.odds_ratio:.2f} supera umbral de cambio de paradigma (2.5). '
            'Recomendar observación adicional JWST.',
            styles["body"]
        ))

    return items


def _build_transition_section(transition, styles):
    items = []
    items.append(HRFlowable(width="100%", thickness=0.3, color=COL_DIM))
    items.append(Paragraph("CAPA 2 — PLANETARY TRANSITION STATE CLASSIFIER (PTSC)", styles["section"]))

    tnames = [td["name"] for td in TRANSITION_DEFINITIONS]
    data = [["TRANSICIÓN", "NOMBRE", "SCORE", "ESTADO"]]
    for i, (score, td) in enumerate(zip(transition.scores, TRANSITION_DEFINITIONS)):
        dominant = "◈ DOMINANTE" if i == transition.dominant_idx else ""
        data.append([
            td["label"],
            td["name"][:35],
            f"{score:.3f}",
            dominant
        ])

    t = Table(data, colWidths=[2*cm, 8*cm, 2.5*cm, 3.5*cm])
    t.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,0), 'Courier-Bold'),
        ('FONTNAME', (0,1), (-1,-1), 'Courier'),
        ('FONTSIZE', (0,0), (-1,-1), 7),
        ('TEXTCOLOR', (0,0), (-1,0), COL_STELLAR),
        ('TEXTCOLOR', (0,1), (-1,-1), COL_TEXT),
        ('BACKGROUND', (0,0), (-1,0), COL_DIM),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [COL_DEEP, COL_VOID]),
        ('GRID', (0,0), (-1,-1), 0.3, COL_DIM),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
    ]))
    items.append(t)

    items.append(Spacer(1, 2*mm))
    items.append(Paragraph(
        f"Estado continuo: {transition.continuous_state:.2f} · "
        f"Reducción falsos negativos vs O₂: {transition.false_negative_reduction:.1f}% · "
        f"Ventana detectabilidad: {transition.detectability_window_ma:.0f} Ma · "
        f"Incertidumbre clasificador: {transition.uncertainty:.2f}",
        styles["small"]
    ))
    return items


def _build_causal_section(causal, styles):
    items = []
    items.append(HRFlowable(width="100%", thickness=0.3, color=COL_DIM))
    items.append(Paragraph("CAPA 3 — CAUSAL NETWORK ANALYZER", styles["section"]))

    data = [
        ["MÉTRICA", "VALOR", "INTERPRETACIÓN"],
        ["Desfase causal", f"{causal.causal_lag_days:.1f} días ({causal.causal_lag_fraction:.2f} período)",
         "Tiempo que superficie precede al gas"],
        ["Sincronía de fase", f"{causal.phase_synchrony:.3f}",
         "Coherencia espectral en banda orbital"],
        ["Entropía de información", f"{causal.info_entropy:.3f}",
         "Complejidad fuera del equilibrio"],
        ["Dimensionalidad efectiva", f"{causal.effective_dim:.2f}",
         "Grados de libertad acoplados"],
        ["Granger BIO", f"{causal.granger_bio:.3f}",
         "Superficie → Gas (metabolismo)"],
        ["Granger ABIO", f"{causal.granger_abio:.3f}",
         "Gas → Superficie (fotoquímica)"],
        ["Dirección causal", causal.causal_direction,
         "Flujo de información dominante"],
        ["Firma no geológica", f"{causal.non_geological:.3f}",
         "Probabilidad de origen biológico dinámico"],
    ]

    t = Table(data, colWidths=[4*cm, 4.5*cm, 7.5*cm])
    t.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,0), 'Courier-Bold'),
        ('FONTNAME', (0,1), (-1,-1), 'Courier'),
        ('FONTSIZE', (0,0), (-1,-1), 7),
        ('TEXTCOLOR', (0,0), (-1,0), COL_STELLAR),
        ('TEXTCOLOR', (0,1), (-1,-1), COL_TEXT),
        ('BACKGROUND', (0,0), (-1,0), COL_DIM),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [COL_DEEP, COL_VOID]),
        ('GRID', (0,0), (-1,-1), 0.3, COL_DIM),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
    ]))
    items.append(t)
    return items


def _build_priority_observations(planet, bayes, styles):
    items = []
    items.append(HRFlowable(width="100%", thickness=0.3, color=COL_DIM))
    items.append(Paragraph("OBSERVACIONES PRIORITARIAS RECOMENDADAS", styles["section"]))

    most_info = most_informative_evidence(planet.get("evidences", {}), planet["name"])
    items.append(Paragraph(
        "Las siguientes evidencias, si aumentaran +0.10 puntos, producirían el mayor "
        "cambio en el odds ratio (prioridad de observación futura):",
        styles["small"]
    ))
    items.append(Spacer(1, 1*mm))

    for i, (key, delta) in enumerate(most_info[:4]):
        label = EVIDENCE_DEFINITIONS.get(key, {}).get("label", key)
        direction = "↑ BIO" if delta > 0 else "↑ ABIO"
        items.append(Paragraph(
            f"  {i+1}. {label}: Δodds = {delta:+.3f} ({direction})",
            styles["body"]
        ))

    return items


def _build_footer_section(styles):
    items = []
    items.append(Spacer(1, 3*mm))
    items.append(HRFlowable(width="100%", thickness=0.3, color=COL_DIM))
    items.append(Paragraph(
        "CORAL ASTROBIO v2.0 · Proyecto Coral Multi-IA 2026 · "
        "Paradigma agnóstico de astrobiología · "
        "Motor bayesiano: DBCE · Clasificador: PTSC · Analizador: CNA · "
        "Datos: NASA Exoplanet Archive · JWST MAST STScI",
        styles["small"]
    ))
    items.append(Paragraph(
        "Nota epistemológica: Este análisis no es una afirmación de vida. Es una cuantificación "
        "probabilística de la evidencia disponible. Los odds ratio son orientativos. "
        "La consiliencia corregida requiere exclusión activa de modelos abióticos alternativos "
        "(Kimi/Grok 2026: consiliencia_seager_falsabilidad_problema).",
        styles["note"]
    ))
    return items
