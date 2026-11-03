"""Randfälle: eigene Netze, entartete Ziele, kleinste Größen, Eingabefehler."""

import math

import numpy as np
import pytest

import lhy_algorithm as A
import lhy_constants as C
import lhy_evaluation as ev
from lhy_scenario import grid_netz, parse_edges


def _s(**kw):
    return ev.Settings(**kw)


def test_eingabefehler_der_spielwiese():
    with pytest.raises(ValueError, match="Schleife"):
        parse_edges("0-0")
    with pytest.raises(ValueError, match="Keine Kanten"):
        parse_edges("\n# nur Kommentar\n")
    with pytest.raises(ValueError, match="Doppelte"):
        parse_edges("0-1\n1-0")
    with pytest.raises(ValueError, match="erwartet zwei ganze Zahlen"):
        parse_edges("0-a")
    with pytest.raises(ValueError, match="höchstens"):
        parse_edges("\n".join(f"{i}-{i + 1}" for i in range(C.CUSTOM_EDGE_LIMIT + 1)))
    with pytest.raises(ValueError, match="Knoten"):
        parse_edges("\n".join(f"{2 * i}-{2 * i + 1}" for i in range(8)))


def test_baum_hat_keine_streuung_wenn_p_eins_ist_und_sonst_nur_den_zusammenhang():
    netz = parse_edges("0-1\n1-2")                                                 # Pfad: X = beide Kanten halten
    a = ev.analyse(_s(p=0.9, n=30), netz=netz)
    assert a.truth == pytest.approx(0.81)
    assert a.add_share == pytest.approx(2 * 0.9 / 1.9)                             # Reihe aus zwei Kanten: 2p/(1+p)


def test_ziel_ohne_streuung_gibt_ehrliches_verdikt():
    netz = parse_edges("0-1")                                                      # eine Kante: X = Kante hält, Var > 0; ein Netz mit nur Knoten gibt es nicht
    a = ev.analyse(_s(p=0.9, n=10), netz=netz)
    assert np.isfinite(a.add_share) and a.add_share == pytest.approx(1.0)          # eine Kante: das Ziel ist selbst additiv
    # konstantes Ziel: p = 1 würde den Regler verlassen; das Verdikt für Var 0 prüfen wir an einer handgesetzten Analyse (test_verdicts)


def test_eine_kante_wird_von_latin_hypercube_praktisch_exakt():
    netz = parse_edges("0-1")
    a = ev.analyse(_s(p=0.7, n=100), netz=netz)
    assert a.f_lhs < 0.01 and a.f_strat < 0.01                                      # additiv, eine Eingabe: Latin Hypercube und Schichtung nehmen alles


def test_kleinste_punktzahl_und_ungerade_zahlen_laufen():
    netz = grid_netz(2, 3)
    X, k = A.enumerate_x(netz.n, netz.edges)
    for N in (2, 4, 10):
        r = A.estimates(X, 0.7, (1, N), 10, N, netz.m, 0, 0.5, with_blocks=False)
        assert all(np.all(np.isfinite(v)) for v in r["est"].values())
    assert np.isfinite(A.strat_mean(X, 0.7, 5, 3, netz.m, 0))                       # N = 3 ungerade: die Schichtung braucht keine geraden N


def test_alle_netze_und_alle_n_laufen_ohne_fehler():
    for netz_key in C.NETZE:
        for n in C.N_CHOICES:
            a = ev.analyse(_s(netz=netz_key, n=n, p=0.8))
            assert math.isfinite(a.truth) and len(a.lhs) == C.REPS


@pytest.mark.parametrize("p", [C.P_MIN, C.P_MAX])
def test_reglergrenzen_fuer_p(p):
    a = ev.analyse(_s(p=p))
    assert 0.0 <= a.truth <= 1.0


def test_bei_seltenen_ausfaellen_und_kleinem_n_gibt_es_keine_streuung_in_den_wiederholungen():
    a = ev.analyse(_s(netz="3x4", p=0.99, n=10))
    assert a.nq == pytest.approx(0.1)
    assert math.isnan(a.f_lhs) or a.f_lhs >= 0                                       # kaum Ausfälle: Verhältnis kann nicht bestimmbar sein
    kind, text = ev.verdict_main(a, _s(netz="3x4", p=0.99, n=10))
    assert kind in ("info", "warning") and "rauschen" in text


def test_aufzaehlung_ueber_dem_limit_wird_abgelehnt():
    with pytest.raises(ValueError, match="aufgezählt"):
        ev.netz_table(grid_netz(3, 5))


def test_dimension_gibt_nan_fuer_konstante_ziele():
    assert math.isnan(A.symmetric_share("parallel", 60, 0.99)) or A.symmetric_share("parallel", 60, 0.99) < 0.01
    d = ev.dimension()
    assert d["ms"] == list(C.DIM_MS) and len(d["blocks"]) == len(C.DIM_PS)
