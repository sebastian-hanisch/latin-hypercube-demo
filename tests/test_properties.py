"""Eigenschaftstests (ORACLE-PLAYBOOK §5): Aussagen, die für jede zulässige Eingabe gelten müssen."""

import numpy as np
import pytest

import lhy_algorithm as A
import lhy_evaluation as ev
from lhy_scenario import grid_netz, hantel_netz


def test_additive_funktion_wird_von_latin_hypercube_exakt_gemittelt():
    # f(u) = Summe der Kanten, die halten, ist additiv: Var(LHS-Mittel) = (Rest = 0) / N, also nur der Zitter in je einer Schicht streut
    m, p, N = 6, 0.5, 20
    X = np.array([bin(mask).count("1") for mask in range(1 << m)], dtype=float)
    est = np.array([A.lhs_mean(X, p, A.stream_seed(81, r), N, m)[0] for r in range(2000)])
    iid = np.array([A.iid_mean(X, p, A.stream_seed(82, r), N, m)[0] for r in range(2000)])
    assert est.var(ddof=1) < 0.06 * iid.var(ddof=1)
    k = np.array([bin(mask).count("1") for mask in range(1 << m)])
    me = A.main_effects(X, k, m, p)
    assert me["add_share"] == pytest.approx(1.0)                                   # additiv: Haupteffekte erklären die ganze Varianz


@pytest.mark.parametrize("netz", [hantel_netz(), grid_netz(2, 3), grid_netz(3, 3)], ids=["hantel", "2x3", "3x3"])
def test_anteile_liegen_in_null_bis_eins_und_summe_hoechstens_eins(netz):
    X, k = ev.netz_table(netz)
    for p in (0.5, 0.7, 0.9, 0.99):
        me = A.main_effects(X, k, netz.m, p)
        assert np.all(me["share"] >= -1e-12) and me["add_share"] <= 1.0 + 1e-9
        assert me["top_share"] == pytest.approx(me["share"].max())


def test_symmetrische_kanten_haben_gleiche_haupteffekte_im_kreis():
    # ein Kreis aus 5 Kanten: alle Kanten sind austauschbar, die Haupteffekte gleich
    edges = tuple((i, (i + 1) % 5) for i in range(5))
    X, k = A.enumerate_x(5, edges)
    me = A.main_effects(X, k, 5, 0.8)
    assert me["eff"].tolist() == pytest.approx([me["eff"][0]] * 5, abs=1e-12)


def test_haupteffekt_einer_bruecke_waechst_mit_p_bis_nahe_eins(hantel):
    netz, (X, k) = hantel
    bruecke = netz.edges.index((2, 3))
    anteile = [A.main_effects(X, k, netz.m, p)["share"][bruecke] for p in (0.5, 0.7, 0.9, 0.99)]
    assert all(a < b for a, b in zip(anteile, anteile[1:]))


def test_gitter_gewinn_ist_nicht_monoton_in_p(gitter34):
    netz, (X, k) = gitter34
    add = [A.main_effects(X, k, netz.m, p)["add_share"] for p in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)]
    assert add[1] > add[0] and add[1] > add[-1] and add[-1] < 0.05                 # erst steigend, dann fallend bis fast null


def test_reihe_bei_hohem_p_bleibt_auch_bei_vielen_kanten_additiv_parallel_nicht():
    assert A.symmetric_share("reihe", 60, 0.99) > 0.7
    assert A.symmetric_share("mehrheit", 16, 0.9) < 0.01 or np.isnan(A.symmetric_share("mehrheit", 16, 0.9))
    assert A.symmetric_share("parallel", 8, 0.7) < 0.01
    assert A.symmetric_share("kreis", 60, 0.9) < A.symmetric_share("kreis", 16, 0.9)


def test_spiegelpaare_nutzen_die_bruecke_nicht_latin_hypercube_schon(hantel):
    netz, (X, k) = hantel
    p, N = 0.9, 100
    me = A.main_effects(X, k, netz.m, p)
    r = A.estimates(X, p, (91, 1), 600, N, netz.m, me["top"], me["EX"])
    v = {name: est.var(ddof=1) for name, est in r["est"].items()}
    assert v["lhs"] < 0.45 * v["iid"] and v["anti"] > 0.7 * v["iid"] and v["strat"] < 0.6 * v["iid"]


def test_naiver_standardfehler_unter_latin_hypercube_ist_zu_gross(hantel):
    netz, (X, k) = hantel
    me = A.main_effects(X, k, netz.m, 0.9)
    r = A.estimates(X, 0.9, (92, 1), 500, 100, netz.m, me["top"], me["EX"], with_blocks=True)
    assert r["se_lhs"].mean() > 1.5 * r["est"]["lhs"].std(ddof=1)
    assert r["coverage"]["lhs"] > 0.99 and 0.9 < r["coverage"]["blocks"] < 1.0 and 0.9 < r["coverage"]["iid"] < 0.98


def test_ohne_bloecke_unter_zwanzig_punkten(gitter34):
    netz, (X, k) = gitter34
    r = A.estimates(X, 0.7, (93, 1), 20, 10, netz.m, 0, 0.456, with_blocks=True)
    assert r["blocks"] is False and r["coverage"]["blocks"] == 0.0
