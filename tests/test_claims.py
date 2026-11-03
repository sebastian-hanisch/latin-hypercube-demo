"""Jede Zahl der README, nachgerechnet über dieselben Auswertungsfunktionen wie in der App (DEMO-PLAYBOOK §4).

Exakte Zahlen (Aufzählung, Formeln) mit engen Bändern. Gemessene Zahlen stammen aus festen Strömen und sind reproduzierbar; sie stehen hier mit Bändern, die das Rauschen
(300 bis 400 Wiederholungen: 7 bis 10 %, bei seltenen Ausfällen mehr) und kleine Plattformunterschiede der Gleitkommasummen abdecken."""

import numpy as np
import pytest

import lhy_algorithm as A
import lhy_evaluation as ev
from lhy_scenario import named_netz

GITTER = ev.Settings("3x4", 0.7, 100)
HANTEL = ev.Settings("hantel", 0.9, 100)


@pytest.fixture(scope="module")
def g():
    netz = named_netz("3x4")
    return netz, ev.netz_table(netz)


@pytest.fixture(scope="module")
def h():
    netz = named_netz("hantel")
    return netz, ev.netz_table(netz)


def _me(nt, p):
    netz, (X, k) = nt
    return A.main_effects(X, k, netz.m, p)


def test_anteile_der_haupteffekte_gitter(g):
    soll = {0.5: 0.204, 0.7: 0.467, 0.8: 0.403, 0.9: 0.214, 0.95: 0.104, 0.99: 0.020}
    for p, v in soll.items():
        me = _me(g, p)
        assert me["add_share"] == pytest.approx(v, abs=6e-4), p
        assert me["top_share"] <= 0.0435
    assert _me(g, 0.7)["top_share"] == pytest.approx(0.043, abs=6e-4)
    assert 1 - _me(g, 0.7)["add_share"] == pytest.approx(0.533, abs=6e-4)
    assert 1 - _me(g, 0.99)["add_share"] == pytest.approx(0.980, abs=6e-4)


def test_anteile_der_haupteffekte_hantel(h):
    soll = {0.5: 0.357, 0.7: 0.597, 0.8: 0.669, 0.9: 0.736, 0.95: 0.812, 0.99: 0.946}
    bruecke = {0.5: 0.143, 0.7: 0.324, 0.8: 0.449, 0.9: 0.631, 0.95: 0.773, 0.99: 0.944}
    for p in soll:
        me = _me(h, p)
        assert me["add_share"] == pytest.approx(soll[p], abs=6e-4), p
        assert me["top_share"] == pytest.approx(bruecke[p], abs=6e-4), p
    me = _me(h, 0.9)
    assert 1 - me["add_share"] == pytest.approx(0.264, abs=6e-4) and 1 - me["top_share"] == pytest.approx(0.369, abs=6e-4)


def test_vorhersage_und_messung_gitter(g):
    netz, table = g
    ex = ev.n_sweep(GITTER, netz, table)
    assert ex["pred_lhs"] == pytest.approx(0.533, abs=6e-4) and ex["pred_strat"] == pytest.approx(0.957, abs=6e-4)
    lhs = [r["f_lhs"] for r in ex["rows"]]
    anti = [r["f_anti"] for r in ex["rows"]]
    strat = [r["f_strat"] for r in ex["rows"]]
    assert min(lhs) == pytest.approx(0.50, abs=0.07) and max(lhs) == pytest.approx(0.56, abs=0.07)
    assert min(anti) == pytest.approx(0.73, abs=0.1) and max(anti) == pytest.approx(0.86, abs=0.1)
    assert min(strat) == pytest.approx(0.96, abs=0.1) and max(strat) == pytest.approx(1.02, abs=0.1)
    a = ev.analyse(GITTER, netz, table)
    assert a.f_lhs == pytest.approx(0.63, abs=0.1) and a.f_lhs > a.pred_lhs and a.f_lhs < 0.85      # Messung über der Vorhersage, Rauschen
    assert a.truth == pytest.approx(0.456, abs=5e-4) and a.nq == pytest.approx(30.0)


def test_vorhersage_und_messung_hantel(h):
    netz, table = h
    ex = ev.n_sweep(HANTEL, netz, table)
    lhs = [r["f_lhs"] for r in ex["rows"]]
    strat = [r["f_strat"] for r in ex["rows"]]
    anti = [r["f_anti"] for r in ex["rows"]]
    assert min(lhs) == pytest.approx(0.25, abs=0.05) and max(lhs) == pytest.approx(0.28, abs=0.05)
    assert min(strat) == pytest.approx(0.33, abs=0.07) and max(strat) == pytest.approx(0.42, abs=0.07)
    assert min(anti) == pytest.approx(0.81, abs=0.12) and max(anti) == pytest.approx(1.02, abs=0.12)


def test_gitter_bei_seltenem_ausfall_hat_keinen_gewinn(g):
    netz, table = g
    ex = ev.rare_events(GITTER, netz, table)
    assert ex["pred_lhs"] == pytest.approx(0.980, abs=6e-4)
    rows = {r["N"]: r["f_lhs"] for r in ex["rows"]}
    assert np.isnan(rows[10])                                                         # N·q = 0,1: keine Wiederholung mit Ausfall, Verhältnis nicht bestimmbar
    assert rows[30] == pytest.approx(1.0, abs=0.35) and rows[300] == pytest.approx(1.02, abs=0.35) and rows[1000] == pytest.approx(1.11, abs=0.35)
    assert 0.3 < rows[100] < 1.0                                                      # Rauschzelle 0,58


def test_seltene_bruecke_gewinn_ab_nq_eins(h):
    netz, table = h
    ex = ev.rare_events(HANTEL, netz, table)
    assert ex["pred_lhs"] == pytest.approx(0.054, abs=6e-4) and ex["q"] == pytest.approx(0.01)
    lhs = {r["N"]: r["f_lhs"] for r in ex["rows"]}
    strat = {r["N"]: r["f_strat"] for r in ex["rows"]}
    nq = {r["N"]: r["nq"] for r in ex["rows"]}
    assert [nq[n] for n in (10, 30, 100, 300, 1000)] == pytest.approx([0.1, 0.3, 1.0, 3.0, 10.0])
    assert lhs[10] > 0.8 and lhs[30] > 0.7                                            # N·q unter 1: kein Gewinn (1,36 und 0,96)
    assert lhs[100] == pytest.approx(0.038, abs=0.03) and lhs[300] == pytest.approx(0.046, abs=0.03) and lhs[1000] == pytest.approx(0.055, abs=0.03)
    assert strat[10] == pytest.approx(0.075, abs=0.05) and strat[30] == pytest.approx(0.043, abs=0.03) and strat[10] < 0.2 < lhs[10]    # Schichtung robuster bei kleinem N


def test_dimension_exakt():
    d = ev.dimension()
    blocks = {b["p"]: b["series"] for b in d["blocks"]}
    assert d["ms"] == [4, 8, 16, 32, 60]
    assert blocks[0.99]["reihe"] == pytest.approx([0.985, 0.965, 0.926, 0.852, 0.732], abs=6e-4)
    assert blocks[0.9]["reihe"] == pytest.approx([0.848, 0.672, 0.404, 0.126, 0.012], abs=6e-4)
    assert blocks[0.9]["kreis"] == pytest.approx([0.429, 0.656, 0.679, 0.377, 0.068], abs=6e-4)
    assert blocks[0.7]["mehrheit"] == pytest.approx([0.391, 0.291, 0.162, 0.049, 0.005], abs=6e-4)
    assert blocks[0.7]["parallel"][0] == pytest.approx(0.076, abs=6e-4) and all((np.isnan(v) or v < 0.01) for v in blocks[0.7]["parallel"][1:])


def test_standardfehler_und_intervalle(g, h):
    for nt, settings, se_lo, se_hi in ((g, GITTER, 1.33, 1.41), (h, HANTEL, 1.91, 1.99)):
        netz, table = nt
        ex = ev.intervals(settings, netz, table)
        ratios = [r["se_ratio"] for r in ex["rows"]]
        assert min(ratios) == pytest.approx(se_lo, abs=0.1) and max(ratios) == pytest.approx(se_hi, abs=0.1)
        assert all(0.9 < r["cover_iid"] < 0.98 for r in ex["rows"])
        assert all(r["cover_lhs"] > 0.97 for r in ex["rows"])                         # naiv unter Latin Hypercube zu breit
        assert all(0.9 < r["cover_blocks"] < 0.98 for r in ex["rows"])
    exg = ev.intervals(GITTER, *g)
    assert min(r["cover_lhs"] for r in exg["rows"]) == pytest.approx(0.98, abs=0.02)
    exh = ev.intervals(HANTEL, *h)
    assert all(r["cover_lhs"] == pytest.approx(1.0, abs=0.01) for r in exh["rows"])
