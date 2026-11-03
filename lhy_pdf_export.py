"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

import numpy as np
from fpdf import FPDF

import lhy_constants as C
from lhy_evaluation import Analysis, Settings, de, dn, verdict_main

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "€": "EUR", "„": '"', "“": '"', "²": "^2", "³": "3", "×": "x", "√": "Wurzel", "≈": "~",
            "σ": "Sigma", "ρ": "rho", "α": "alpha", "β": "beta", "θ": "theta", "∞": "unendlich", "Σ": "Summe", "δ": "delta", "Δ": "Delta", "½": "1/2", "̂": "", "ϱ": "rho"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(settings: Settings, a: Analysis) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Gleichmäßig abdecken: Schichtung und Latin Hypercube"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(
        f"{a.netz.name}, {a.netz.n} Knoten, {a.netz.m} Kanten; jede Kante hält mit Wahrscheinlichkeit {de(settings.p, 2)}; Ziel: das Netz hängt zusammen. "
        f"{de(settings.n, 0)} Punkte je Schätzung, {C.REPS} Wiederholungen, Strom {settings.seed}. Alles simuliert, keine Messdaten."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 9)
    widths = (70, 40, 40)
    for h, w in zip(("Schätzer", "Streuung", "Varianz gegen einfach"), widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for label, series, ratio in (("einfach", a.iid, 1.0), ("Latin Hypercube", a.lhs, a.f_lhs), ("Spiegelpaare", a.anti, a.f_anti), ("Schichtung (wirksamste Kante)", a.strat, a.f_strat)):
        for text, w in zip((label, de(float(np.std(series, ddof=1)), 5), dn(ratio, 3)), widths):
            pdf.cell(w, 6, _s(text), border=1)
        pdf.ln()
    pdf.ln(2)
    pdf.multi_cell(0, 5, _s(f"Wahrheit R (Aufzählung aller Kantenmengen): {de(a.truth, 5)}. Summe der Haupteffekte: {dn(a.add_share, 4)} der Varianz; Vorhersage für Latin Hypercube {dn(a.pred_lhs, 4)}, "
                            f"für die Schichtung nach der wirksamsten Kante {dn(a.pred_strat, 4)}. Naiver Standardfehler unter Latin Hypercube: {dn(a.se_ratio, 2)}-mal die wahre Streuung."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.multi_cell(0, 5, _s(verdict_main(a, settings)[1]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: latin-hypercube-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
