"""Kennzahlen, Verdikt-Logik und Experimente über feste Ströme (reine Funktionen, kein Streamlit).

Ziel X = 1, wenn das Netz zusammenhängt; gesucht ist R = E X. Vier Schätzer mit je N Punkten stehen nebeneinander: einfach, Latin Hypercube, Spiegelpaare und Schichtung nach der
wirksamsten Kante. Der Gewinn des Latin Hypercube ist für großes N vorhersagbar: Faktor 1 - (Summe der Haupteffekte) / Var X (asymptotisch); der der Schichtung nach einer Kante 1 - Anteil dieser Kante.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import lhy_algorithm as A
import lhy_constants as C
from lhy_scenario import Netz, named_netz


def de(x: float, nd: int = 2) -> str:
    """Zahl im deutschen Format: Punkt als Tausendertrenner, Komma als Dezimalzeichen, echtes Minuszeichen."""
    return f"{x:,.{nd}f}".translate(str.maketrans(",.", ".,")).replace("-", "−")


def dn(x: float, nd: int = 2) -> str:
    """Wie `de`, aber „nicht bestimmbar“ für NaN (z. B. ein Varianzverhältnis ohne Varianz im Nenner)."""
    return de(x, nd) if np.isfinite(x) else "nicht bestimmbar"


def de_e(x: float) -> str:
    """Wissenschaftliche Schreibweise mit Dezimalkomma, z. B. 4,2e-10."""
    return f"{x:.1e}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    netz: str = C.DEFAULT_NETZ
    p: float = C.DEFAULT_P
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    netz: Netz
    truth: float
    var0: float                   # Var X (exakt)
    shares: tuple[float, ...]     # Anteil der Haupteffekte je Kante an Var X (exakt)
    add_share: float              # Summe der Anteile: so viel entfernt Latin Hypercube
    top: int                      # wirksamste Kante
    top_share: float
    pred_lhs: float               # 1 - add_share
    pred_strat: float             # 1 - top_share
    nq: float                     # N * q mit q = 1 - p
    f_lhs: float                  # gemessen: Varianz Latin Hypercube / einfach
    f_anti: float
    f_strat: float
    sd_iid: float
    sd_lhs: float
    se_ratio: float               # mittlerer naiver Standardfehler unter Latin Hypercube / wahre Streuung der Latin-Hypercube-Mittel
    cover_iid: float
    cover_lhs: float
    iid: tuple[float, ...]
    lhs: tuple[float, ...]
    anti: tuple[float, ...]
    strat: tuple[float, ...]


# --- Wahrheit -----------------------------------------------------------------------------------------------------------------------------

def netz_table(netz: Netz) -> tuple[np.ndarray, np.ndarray]:
    """Aufzählung aller Kantenmengen (höchstens C.LIVE_EDGE_LIMIT Kanten): X und Zahl haltender Kanten je Menge."""
    if netz.m > C.LIVE_EDGE_LIMIT:
        raise ValueError(f"{netz.m} Kanten: aufgezählt wird nur bis {C.LIVE_EDGE_LIMIT}")
    return A.enumerate_x(netz.n, netz.edges)


def _ratio(a: np.ndarray, b: np.ndarray) -> float:
    vb = float(b.var(ddof=1))
    return float(a.var(ddof=1)) / vb if vb > 1e-18 else float("nan")


# --- Hauptanalyse -------------------------------------------------------------------------------------------------------------------------

def analyse(settings: Settings, netz: Netz | None = None, table=None) -> Analysis:
    """REPS Schätzungen aus je N Punkten mit vier Schätzern (einfach, Latin Hypercube, Spiegelpaare, Schichtung nach der wirksamsten Kante), feste Ströme."""
    netz = netz if netz is not None else named_netz(settings.netz)
    X, k = table if table is not None else netz_table(netz)
    me = A.main_effects(X, k, netz.m, settings.p)
    key = (1, netz.n, netz.m, int(settings.p * 1000), settings.n, settings.seed)
    r = A.estimates(X, settings.p, key, C.REPS, settings.n, netz.m, me["top"], me["EX"])
    e = r["est"]
    return Analysis(
        netz, me["EX"], me["Var"], tuple(float(x) for x in me["share"]), me["add_share"], me["top"], me["top_share"], 1.0 - me["add_share"], 1.0 - me["top_share"],
        settings.n * (1.0 - settings.p), _ratio(e["lhs"], e["iid"]), _ratio(e["anti"], e["iid"]), _ratio(e["strat"], e["iid"]), float(e["iid"].std(ddof=1)), float(e["lhs"].std(ddof=1)),
        float(r["se_lhs"].mean() / e["lhs"].std(ddof=1)) if e["lhs"].std(ddof=1) > 0 else float("nan"), r["coverage"]["iid"], r["coverage"]["lhs"],
        tuple(float(x) for x in e["iid"]), tuple(float(x) for x in e["lhs"]), tuple(float(x) for x in e["anti"]), tuple(float(x) for x in e["strat"]),
    )


def verdict_main(a: Analysis, settings: Settings) -> tuple[str, str]:
    """Lohnt das gleichmäßige Abdecken, und stimmt die Vorhersage aus den Haupteffekten?"""
    if not np.isfinite(a.add_share):
        return "info", "Das Netz hängt bei diesen Einstellungen nie oder immer zusammen: das Ziel hat keine Streuung, es gibt nichts zu senken."
    kopf = (f"Die Haupteffekte tragen {de(a.add_share * 100, 0)} % der Varianz (die wirksamste Kante {de(a.top_share * 100, 0)} %); Latin Hypercube entfernt diesen Anteil: Vorhersage für den Faktor {de(a.pred_lhs, 2)}, "
            f"gemessen wurde {dn(a.f_lhs, 2)} (Spiegelpaare {dn(a.f_anti, 2)}, Schichtung nach der wirksamsten Kante {dn(a.f_strat, 2)}).")
    if a.nq < 5.0:
        kopf += " Bei so seltenen Ausfällen (N·q unter 5) rauschen die gemessenen Zahlen stark."
    if a.nq < 1.0 and a.add_share > 0.2:
        return "warning", kopf + f" Mit N·q = {de(a.nq, 1)} enthält aber nur die oberste Schicht den Ausfallbereich, und davon nur den Anteil N·q: der volle Gewinn zeigt sich erst bei mehr Punkten (Experiment 4)."
    if a.add_share < 0.1:
        return "info", kopf + " Der additive Anteil ist winzig: gleichmäßiges Abdecken bringt hier praktisch nichts, die Varianz steckt im Zusammenspiel der Kanten."
    if a.se_ratio > 1.2:
        return "success", kopf + f" Der naive Standardfehler ist dabei {de(a.se_ratio, 2)}-mal zu groß (Experiment 5)."
    return "success", kopf


# --- Experiment 1: Varianz gegen N ----------------------------------------------------------------------------------------------------------

def n_sweep(settings: Settings, netz: Netz, table) -> dict:
    X, k = table
    me = A.main_effects(X, k, netz.m, settings.p)
    rows = []
    for N in C.SWEEP_NS:
        key = (2, netz.n, netz.m, int(settings.p * 1000), N, settings.seed)
        r = A.estimates(X, settings.p, key, C.SWEEP_REPS, N, netz.m, me["top"], me["EX"])
        e = r["est"]
        rows.append({"N": N, "f_lhs": _ratio(e["lhs"], e["iid"]), "f_anti": _ratio(e["anti"], e["iid"]), "f_strat": _ratio(e["strat"], e["iid"]), "nq": N * (1 - settings.p)})
    return {"rows": rows, "pred_lhs": 1.0 - me["add_share"], "pred_strat": 1.0 - me["top_share"]}


def verdict_n_sweep(exp: dict) -> tuple[str, str]:
    rows = exp["rows"]
    if not np.isfinite(exp["pred_lhs"]):
        return "info", "Das Ziel hat keine Streuung, es gibt nichts zu messen."
    noise = 100 / np.sqrt(C.SWEEP_REPS / 2)
    return "info", (f"Vorhersage aus den Haupteffekten: Latin Hypercube {de(exp['pred_lhs'], 3)}, Schichtung nach der wirksamsten Kante {de(exp['pred_strat'], 3)}. "
                    f"Gemessen liegt Latin Hypercube zwischen {de(min(r['f_lhs'] for r in rows), 2)} und {de(max(r['f_lhs'] for r in rows), 2)}, die Spiegelpaare zwischen "
                    f"{de(min(r['f_anti'] for r in rows), 2)} und {de(max(r['f_anti'] for r in rows), 2)}, die Schichtung zwischen {de(min(r['f_strat'] for r in rows), 2)} und {de(max(r['f_strat'] for r in rows), 2)}. "
                    f"Das Rauschen der Verhältnisse beträgt bei {C.SWEEP_REPS} Wiederholungen rund {de(noise, 0)} %, bei seltenen Ausfällen deutlich mehr.")


# --- Experiment 2: wer trägt den Gewinn (exakt) -------------------------------------------------------------------------------------------------

def share_table(settings: Settings, netz: Netz, table) -> dict:
    """Exakte Anteile der Haupteffekte je Kante bei p des Reglers und die Summe gegen p (Aufzählung, kein Zufall)."""
    X, k = table
    me = A.main_effects(X, k, netz.m, settings.p)
    rows = []
    for p in C.SHARE_PS:
        m2 = A.main_effects(X, k, netz.m, p)
        rows.append({"p": p, "EX": m2["EX"], "add_share": m2["add_share"], "top_share": m2["top_share"], "top": m2["top"]})
    return {"edges": list(netz.edges), "shares": [float(x) for x in me["share"]], "add_share": me["add_share"], "top": me["top"], "rows": rows}


def verdict_shares(exp: dict) -> tuple[str, str]:
    rows = exp["rows"]
    ok = [r for r in rows if np.isfinite(r["add_share"])]
    if not ok:
        return "info", "Das Ziel hat bei keinem p eine Streuung."
    best = max(ok, key=lambda r: r["add_share"])
    worst = min(ok, key=lambda r: r["add_share"])
    return "info", (f"Bei p = {de(best['p'], 2)} tragen die Haupteffekte zusammen {de(best['add_share'] * 100, 0)} % der Varianz, bei p = {de(worst['p'], 2)} nur {de(worst['add_share'] * 100, 0)} %; "
                    f"die wirksamste Kante trägt bei p = {de(best['p'], 2)} {de(best['top_share'] * 100, 0)} %. Der Gewinn von Latin Hypercube ist nicht monoton in p und hängt am Netz, nicht an der Zahl der Kanten.")


# --- Experiment 3: Dimension (exakt) --------------------------------------------------------------------------------------------------------------

KINDS = {"kreis": "Kreis (höchstens eine Kante fällt aus)", "reihe": "Reihe (alle halten)", "parallel": "Parallel (mindestens eine hält)", "mehrheit": "Mehrheit (mindestens die Hälfte hält)"}


def dimension() -> dict:
    """Additiver Anteil der Varianz gegen die Kantenzahl m für vier symmetrische Ziele und drei p (binomial exakt)."""
    blocks = []
    for p in C.DIM_PS:
        blocks.append({"p": p, "series": {kind: [A.symmetric_share(kind, m, p) for m in C.DIM_MS] for kind in KINDS}})
    return {"ms": list(C.DIM_MS), "blocks": blocks}


def verdict_dimension(exp: dict) -> tuple[str, str]:
    hi = next(b for b in exp["blocks"] if b["p"] == max(C.DIM_PS))
    lo = next(b for b in exp["blocks"] if b["p"] == min(C.DIM_PS))
    m = exp["ms"][-1]
    return "info", (f"Bei p = {de(hi['p'], 2)} und {m} gleich wichtigen Kanten trägt der additive Anteil bei der Reihe {de(hi['series']['reihe'][-1] * 100, 0)} % der Varianz (großer Gewinn trotz vieler Eingaben); "
                    f"bei der Mehrheit schon bei {exp['ms'][1]} Kanten nur {de(lo['series']['mehrheit'][1] * 100, 0)} % (p = {de(lo['p'], 2)}). Nicht die Zahl der Eingaben entscheidet, sondern ob die Funktion fast additiv ist.")


# --- Experiment 4: seltene Ausfälle ---------------------------------------------------------------------------------------------------------------

def rare_events(settings: Settings, netz: Netz, table) -> dict:
    """Faktor von Latin Hypercube und Schichtung gegen N bei p = RARE_P (Ausfallwahrscheinlichkeit q = 0,01): der Gewinn erscheint erst, wenn N·q etwa 1 erreicht."""
    X, k = table
    p = C.RARE_P
    me = A.main_effects(X, k, netz.m, p)
    rows = []
    for N in C.RARE_NS:
        key = (3, netz.n, netz.m, int(p * 1000), N, settings.seed)
        r = A.estimates(X, p, key, C.RARE_REPS, N if N % 2 == 0 else N + 1, netz.m, me["top"], me["EX"])
        e = r["est"]
        rows.append({"N": N, "nq": N * (1 - p), "f_lhs": _ratio(e["lhs"], e["iid"]), "f_strat": _ratio(e["strat"], e["iid"])})
    return {"rows": rows, "pred_lhs": 1.0 - me["add_share"], "pred_strat": 1.0 - me["top_share"], "add_share": me["add_share"], "q": 1.0 - p}


def verdict_rare(exp: dict) -> tuple[str, str]:
    rows = exp["rows"]
    if not np.isfinite(exp["pred_lhs"]):
        return "info", "Das Ziel hat keine Streuung, es gibt nichts zu messen."
    first, last = rows[0], rows[-1]
    return "warning", (f"Bei q = {de(exp['q'], 2)} tragen die Haupteffekte {de(exp['add_share'] * 100, 0)} % der Varianz (Vorhersage {de(exp['pred_lhs'], 3)}). Gemessen ist Latin Hypercube bei N = {de(first['N'], 0)} "
                       f"(N·q = {de(first['nq'], 1)}) bei {dn(first['f_lhs'], 2)}, bei N = {de(last['N'], 0)} (N·q = {de(last['nq'], 0)}) bei {dn(last['f_lhs'], 2)}. Solange N·q unter 1 liegt, enthält nur die oberste Schicht den Ausfallbereich, und davon nur den Anteil N·q; "
                       f"der Gewinn ist dann klein, und die Messung rauscht stark (nur wenige Wiederholungen enthalten überhaupt einen Ausfall).")


# --- Experiment 5: Standardfehler und Intervalle -------------------------------------------------------------------------------------------------

def intervals(settings: Settings, netz: Netz, table) -> dict:
    """Naiver Standardfehler unter Latin Hypercube gegen die wahre Streuung und Überdeckung der 95-%-Intervalle: einfach, naiv unter Latin Hypercube, zehn Latin-Hypercube-Blöcke."""
    X, k = table
    me = A.main_effects(X, k, netz.m, settings.p)
    rows = []
    for N in C.INT_NS:
        key = (4, netz.n, netz.m, int(settings.p * 1000), N, settings.seed)
        r = A.estimates(X, settings.p, key, C.INT_REPS, N, netz.m, me["top"], me["EX"], with_blocks=True)
        sd = float(r["est"]["lhs"].std(ddof=1))
        rows.append({"N": N, "se_ratio": float(r["se_lhs"].mean() / sd) if sd > 0 else float("nan"), "cover_iid": r["coverage"]["iid"], "cover_lhs": r["coverage"]["lhs"],
                     "cover_blocks": r["coverage"]["blocks"] if r["blocks"] else None})
    return {"rows": rows}


def verdict_intervals(exp: dict) -> tuple[str, str]:
    rows = exp["rows"]
    ratios = [r["se_ratio"] for r in rows if np.isfinite(r["se_ratio"])]
    if not ratios:
        return "info", "Das Ziel hat keine Streuung, es gibt nichts zu messen."
    last = rows[-1]
    blk = [r["cover_blocks"] for r in rows if r["cover_blocks"] is not None]
    text = (f"Der naive Standardfehler (Stichprobenstreuung durch √N) ist unter Latin Hypercube zwischen {de(min(ratios), 2)}- und {de(max(ratios), 2)}-mal so groß wie die wahre Streuung der Mittel. "
            f"Überdeckung des naiven 95-%-Intervalls bei N = {de(last['N'], 0)}: {de(last['cover_lhs'], 2)} (einfach gezogen: {de(last['cover_iid'], 2)})")
    if blk:
        text += f"; zehn Latin-Hypercube-Blöcke: {de(min(blk), 2)} bis {de(max(blk), 2)}."
    else:
        text += "."
    return "warning", text


# --- Spielwiese ---------------------------------------------------------------------------------------------------------------------------

def custom_result(settings: Settings, netz: Netz) -> dict:
    """Ein eigenes Netz: Analyse und die exakten Anteile der Haupteffekte je Kante."""
    table = netz_table(netz)
    a = analyse(settings, netz=netz, table=table)
    return {"analysis": a, "shares": a.shares}
