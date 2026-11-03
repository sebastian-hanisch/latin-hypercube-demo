"""Kern von Hand geprüft: Zufallsstrom, Latin-Hypercube-Eigenschaft, Aufzählung der Hantel, Haupteffekte mit Brüchen, Schätzer an kleinen Fällen."""

from fractions import Fraction

import numpy as np
import pytest

import lhy_algorithm as A


def test_uniforms_gegen_die_ganzzahl_referenz():
    seed = A.stream_seed(3, 7)
    soll = [(A.mix_int((seed + i * A.GAMMA) & A.MASK) >> 11) * (1.0 / (1 << 53)) for i in range(1, 6)]
    assert A.uniforms(seed, 5).tolist() == soll


def test_ausgabe_i_haengt_nur_von_seed_und_i_ab():
    assert A.uniforms(5, 100)[:10].tolist() == A.uniforms(5, 10).tolist()


@pytest.mark.parametrize("N,m", [(1, 3), (2, 2), (10, 5), (30, 17)])
def test_latin_hypercube_je_spalte_genau_ein_punkt_je_schicht(N, m):
    u = A.lhs_uniforms(A.stream_seed(11, N, m), N, m)
    assert u.shape == (N, m) and np.all((u >= 0) & (u < 1))
    for e in range(m):
        assert sorted((u[:, e] * N).astype(int).tolist()) == list(range(N))          # Schicht j enthält genau einen Punkt


def test_latin_hypercube_spalten_sind_verschieden_permutiert_und_der_zitter_liegt_in_der_schicht():
    N, m = 20, 6
    u = A.lhs_uniforms(7, N, m)
    perms = {tuple((u[:, e] * N).astype(int)) for e in range(m)}
    assert len(perms) == m                                                         # kaum eine Doppelung bei 20! Permutationen
    # Zitter: Nachkommastelle von N*u ist gleichverteilt (Mittel nahe 1/2 bei vielen Spalten)
    big = A.lhs_uniforms(9, 2000, 8)
    frac = (big * 2000) % 1.0
    assert abs(frac.mean() - 0.5) < 0.01


def test_latin_hypercube_aus_dem_strom_von_hand_n_gleich_zwei():
    # N = 2, m = 1: erste zwei Zahlen = Zitter v, nächste zwei = Sortierschlüssel; u_i = (Rang_i + v_i) / 2
    seed = 42
    z = A.uniforms(seed, 4)
    rang = np.argsort(z[2:4])
    soll = (rang + z[:2]) / 2
    assert A.lhs_uniforms(seed, 2, 1)[:, 0].tolist() == pytest.approx(soll.tolist())


def test_masken_bitpackung_von_hand():
    X = np.arange(8, dtype=float)                                                  # X[mask] = mask: das Nachschlagen gibt die Maske zurück
    u = np.array([[0.1, 0.9, 0.2], [0.9, 0.9, 0.9], [0.1, 0.1, 0.1]])
    assert A.lookup(X, u, 0.5).tolist() == [0b101, 0b000, 0b111]                   # Kante e hält (u_e < p): Bit e


def test_hantel_zaehlungen_von_hand(hantel):
    netz, (X, k) = hantel
    assert int(X.sum()) == 2**7 // 8                                              # p = 1/2: 16 zusammenhängende von 128 Kantenmengen (Dreieck: 4 von 8, zweimal, Brücke hält)
    assert int(X.sum()) == 4 * 4 * 1


def test_hantel_haupteffekt_der_bruecke_mit_bruechen(hantel):
    netz, (X, k) = hantel
    me = A.main_effects(X, k, netz.m, 0.5)
    bruecke = netz.edges.index((2, 3))
    a1 = Fraction(4, 8) ** 2                                                      # beide Dreiecke hängen zusammen (je mindestens zwei von drei Kanten)
    assert me["eff"][bruecke] == pytest.approx(float(Fraction(1, 2) * Fraction(1, 2) * a1**2))   # p (1-p) (a1 - 0)^2 = 1/64
    assert me["eff"][bruecke] == pytest.approx(1 / 64)
    assert me["top"] == bruecke and me["EX"] == pytest.approx(0.125)
    assert me["Var"] == pytest.approx(0.125 * 0.875)


def test_symmetrische_ziele_von_hand():
    assert A.symmetric_target("kreis", 5, 5) == 1.0 and A.symmetric_target("kreis", 3, 5) == 0.0
    assert A.symmetric_target("reihe", 5, 5) == 1.0 and A.symmetric_target("reihe", 4, 5) == 0.0
    assert A.symmetric_target("parallel", 0, 5) == 0.0 and A.symmetric_target("parallel", 1, 5) == 1.0
    assert A.symmetric_target("mehrheit", 3, 6) == 1.0 and A.symmetric_target("mehrheit", 2, 6) == 0.0


def test_symmetrischer_anteil_reihe_mit_zwei_kanten_von_hand():
    # X = u_1 < p und u_2 < p: E X = p^2, Var = p^2 (1 - p^2); Haupteffekt je Kante: p(1-p) (p - 0)^2; Anteil = 2 p^3 (1-p) / (p^2 (1 - p^2)) = 2 p (1-p) / (1-p^2) = 2p/(1+p)
    for p in (0.3, 0.5, 0.9):
        assert A.symmetric_share("reihe", 2, p) == pytest.approx(2 * p / (1 + p))


def test_gespiegelte_paare_von_hand():
    X = np.arange(8, dtype=float)
    seed, N, m, p = 5, 4, 3, 0.5
    u = A.iid_uniforms(seed, 2, m)
    soll = 0.5 * (A.lookup(X, u, p).mean() + A.lookup(X, 1 - u, p).mean())
    assert A.anti_mean(X, p, seed, N, m) == pytest.approx(soll)


def test_schichtung_zaehlt_die_punkte_der_kante_von_hand():
    X = np.arange(8, dtype=float)
    seed, N, m, p, e = 3, 10, 3, 0.7, 1
    nu = 7
    u = A.iid_uniforms(seed, N, m)
    u[:nu, e] *= p
    u[nu:, e] = p + u[nu:, e] * (1 - p)
    assert np.all(u[:nu, e] < p) and np.all(u[nu:, e] >= p)
    x = A.lookup(X, u, p)
    assert A.strat_mean(X, p, seed, N, m, e) == pytest.approx(p * x[:nu].mean() + (1 - p) * x[nu:].mean())


def test_schichtung_ist_exakt_erwartungstreu_bei_beliebigem_n_p(hantel):
    # Der nachgewichtete Schätzer ist auch bei nicht ganzzahligem N p erwartungstreu: Erwartungswert = p E[X | hält] + (1-p) E[X | fällt aus] = E X
    netz, (X, k) = hantel
    p = 0.9
    me = A.main_effects(X, k, netz.m, p)
    est = np.array([A.strat_mean(X, p, A.stream_seed(21, r), 25, netz.m, me["top"]) for r in range(3000)])
    z = (est.mean() - me["EX"]) / (est.std(ddof=1) / np.sqrt(len(est)))
    assert abs(z) < 4.0
