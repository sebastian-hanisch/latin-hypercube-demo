"""Orakel-Tests: andere Rechenwege als die der Demo.

- Ziel X gegen networkx (Zusammenhang) auf allen Kantenmengen kleiner Netze.
- Haupteffekte gegen eine zweite Rechnung: direkte Bruchsumme über alle Kantenmengen (statt der gewichteten Vektorsummen), und die Zerlegung Var X = Summe der Haupteffekte + Rest mit Rest >= 0.
- Symmetrische Ziele: binomiale Formel gegen die Aufzählung (main_effects auf der Tabelle f(k)).
- Latin Hypercube in einer Dimension = Schichtung mit N gleichen Schichten: exakte Varianz (nur die Schicht, die p enthält, streut), gegen viele Wiederholungen.
- Schichtung nach einer Kante: Varianz exakt gleich (Var X - Haupteffekt)/N bei ganzzahligem N p, gegen die Aufzählung (bedingte Varianzen) und viele Wiederholungen.
- Latin Hypercube mit vielen Kanten: asymptotische Varianz gegen die exakte Vorhersage (Owen 1997) und Obergrenze N/(N-1).
"""

from fractions import Fraction

import numpy as np
import pytest

import lhy_algorithm as A
import lhy_evaluation as ev
from lhy_scenario import grid_netz, hantel_netz

nx = pytest.importorskip("networkx")


def x_networkx(n, edges, mask):
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(e for i, e in enumerate(edges) if (mask >> i) & 1)
    return nx.is_connected(g)


@pytest.mark.parametrize("netz", [hantel_netz(), grid_netz(2, 3), grid_netz(3, 3)], ids=["hantel", "2x3", "3x3"])
def test_x_gegen_networkx_auf_allen_kantenmengen(netz):
    X, k = ev.netz_table(netz)
    for mask in range(1 << netz.m):
        assert bool(X[mask]) == x_networkx(netz.n, netz.edges, mask), mask
        assert k[mask] == bin(mask).count("1")


def haupteffekte_brueche(netz, p: Fraction):
    """Direkte Rechnung mit Brüchen: E X, Var X und je Kante p (1-p) (a1 - a0)^2 aus Summen über alle Kantenmengen (ohne Vektorisierung)."""
    m = netz.m
    gew = []
    xs = []
    for mask in range(1 << m):
        k = bin(mask).count("1")
        gew.append(p**k * (1 - p) ** (m - k))
        xs.append(1 if x_networkx(netz.n, netz.edges, mask) else 0)
    ex = sum(w * x for w, x in zip(gew, xs))
    var = ex - ex * ex                                                            # X ist 0/1
    eff = []
    for e in range(m):
        s1 = sum(w * x for mask, (w, x) in enumerate(zip(gew, xs)) if (mask >> e) & 1)
        s0 = sum(w * x for mask, (w, x) in enumerate(zip(gew, xs)) if not (mask >> e) & 1)
        a1, a0 = s1 / p, s0 / (1 - p)
        eff.append(p * (1 - p) * (a1 - a0) ** 2)
    return ex, var, eff


@pytest.mark.parametrize("netz", [hantel_netz(), grid_netz(2, 3)], ids=["hantel", "2x3"])
@pytest.mark.parametrize("p", [Fraction(1, 2), Fraction(3, 4), Fraction(9, 10)])
def test_haupteffekte_gegen_bruchrechnung(netz, p):
    X, k = ev.netz_table(netz)
    me = A.main_effects(X, k, netz.m, float(p))
    ex, var, eff = haupteffekte_brueche(netz, p)
    assert me["EX"] == pytest.approx(float(ex), abs=1e-12) and me["Var"] == pytest.approx(float(var), abs=1e-12)
    assert me["eff"].tolist() == pytest.approx([float(x) for x in eff], abs=1e-12)
    assert sum(eff) <= var                                                         # die Haupteffekte erklären höchstens die ganze Varianz (Rest = Zusammenspiel >= 0)


def test_zerlegung_summe_haupteffekte_hoechstens_var_auf_dem_gitter(gitter34):
    netz, (X, k) = gitter34
    for p in (0.5, 0.7, 0.9, 0.99):
        me = A.main_effects(X, k, netz.m, p)
        assert 0.0 <= me["add_share"] <= 1.0 and np.all(me["share"] >= 0)


@pytest.mark.parametrize("kind", ["kreis", "reihe", "parallel", "mehrheit"])
@pytest.mark.parametrize("m,p", [(5, 0.7), (7, 0.9), (8, 0.5)])
def test_symmetrische_formel_gegen_die_aufzaehlung(kind, m, p):
    masks = np.arange(1 << m)
    k = np.array([bin(x).count("1") for x in masks])
    X = np.array([A.symmetric_target(kind, int(j), m) for j in k])
    me = A.main_effects(X, k, m, p)
    soll = me["add_share"]
    got = A.symmetric_share(kind, m, p)
    if np.isnan(soll):
        assert np.isnan(got)
    else:
        assert got == pytest.approx(soll, abs=1e-10)


def test_eine_dimension_latin_hypercube_ist_schichtung_mit_gleichen_schichten():
    # f(u) = 1[u < p]: nur die Schicht, die p enthält, streut; Var(Mittel) = f_s (1 - f_s) / N^2 mit f_s dem Anteil dieser Schicht unterhalb p
    N, p = 10, 0.37
    X = np.array([0.0, 1.0])
    fs = p * N - int(p * N)
    soll = fs * (1 - fs) / N**2
    est = np.array([A.lhs_mean(X, p, A.stream_seed(31, r), N, 1)[0] for r in range(8000)])
    assert est.mean() == pytest.approx(p, abs=4 * np.sqrt(soll) / np.sqrt(8000) + 1e-9)
    assert est.var(ddof=1) == pytest.approx(soll, rel=0.08)
    assert soll < p * (1 - p) / N / 10                                             # viel kleiner als beim einfachen Ziehen


def test_schichtung_varianz_exakt_gleich_bedingte_varianzen(hantel):
    netz, (X, k) = hantel
    p, N = 0.9, 100
    me = A.main_effects(X, k, netz.m, p)
    e = me["top"]
    w = p**k * (1 - p) ** (netz.m - k)
    masks = np.arange(len(X))
    up = ((masks >> e) & 1).astype(bool)
    def cond_var(sel):
        ws = w[sel] / w[sel].sum()
        mu = float(np.sum(ws * X[sel]))
        return float(np.sum(ws * X[sel] ** 2) - mu**2)
    v1, v0 = cond_var(up), cond_var(~up)
    n1 = round(N * p)
    exakt = (p**2 * v1 / n1 + (1 - p) ** 2 * v0 / (N - n1))
    assert exakt == pytest.approx((me["Var"] - me["eff"][e]) / N, rel=1e-9)         # (σ² - Haupteffekt)/N bei ganzzahligem N p
    est = np.array([A.strat_mean(X, p, A.stream_seed(41, r), N, netz.m, e) for r in range(6000)])
    assert est.var(ddof=1) == pytest.approx(exakt, rel=0.12)


def test_latin_hypercube_varianz_gegen_die_exakte_vorhersage_auf_der_hantel(hantel):
    netz, (X, k) = hantel
    p, N, R = 0.9, 100, 2500
    me = A.main_effects(X, k, netz.m, p)
    est = np.array([A.lhs_mean(X, p, A.stream_seed(51, r), N, netz.m)[0] for r in range(R)])
    pred = (me["Var"] - me["eff"].sum()) / N
    assert est.var(ddof=1) == pytest.approx(pred, rel=0.15)
    assert est.var(ddof=1) < 0.4 * me["Var"] / N


def test_latin_hypercube_nie_viel_schlechter_als_einfach_owen_schranke(gitter34):
    # Owen: Var(LHS-Mittel) <= N/(N-1) * sigma^2/N für jede Funktion; hier mit Messrauschen von rund 10 %
    netz, (X, k) = gitter34
    me = A.main_effects(X, k, netz.m, 0.7)
    N, R = 30, 1200
    est = np.array([A.lhs_mean(X, 0.7, A.stream_seed(61, r), N, netz.m)[0] for r in range(R)])
    assert est.var(ddof=1) <= 1.0 * me["Var"] / N * (N / (N - 1)) * 1.1
    assert est.var(ddof=1) == pytest.approx((me["Var"] - me["eff"].sum()) / N, rel=0.2)


def test_alle_schaetzer_erwartungstreu_gepoolt(gitter34):
    netz, (X, k) = gitter34
    p, N = 0.7, 30
    me = A.main_effects(X, k, netz.m, p)
    r = A.estimates(X, p, (71, 1), 800, N, netz.m, me["top"], me["EX"])
    for name, est in r["est"].items():
        z = (est.mean() - me["EX"]) / (est.std(ddof=1) / np.sqrt(len(est)))
        assert abs(z) < 4.0, name


def test_ein_zusammenhaengendes_ziel_wird_vom_modul_gerechnet_wie_vom_orakel(hantel):
    netz, (X, k) = hantel
    ex, var, eff = haupteffekte_brueche(netz, Fraction(7, 10))
    a = ev.analyse(ev.Settings(netz="hantel", p=0.7, n=30), netz=netz, table=(X, k))
    assert a.truth == pytest.approx(float(ex), abs=1e-12) and a.add_share == pytest.approx(float(sum(eff) / var), abs=1e-12)
