"""Verdikte (reine Funktionen in lhy_evaluation.py) an handgesetzten Werten: Schwellen und jede Zahl im Text."""

import dataclasses

import lhy_evaluation as ev
from lhy_scenario import hantel_netz

S = ev.Settings()


def _a(**kw) -> ev.Analysis:
    basis = ev.Analysis(
        netz=hantel_netz(), truth=0.85, var0=0.127, shares=(0.01,) * 6 + (0.63,), add_share=0.74, top=6, top_share=0.63, pred_lhs=0.26, pred_strat=0.37, nq=10.0,
        f_lhs=0.29, f_anti=1.09, f_strat=0.42, sd_iid=0.035, sd_lhs=0.018, se_ratio=1.98, cover_iid=0.93, cover_lhs=1.0, iid=(0.8,) * 3, lhs=(0.8,) * 3, anti=(0.8,) * 3, strat=(0.8,) * 3)
    return dataclasses.replace(basis, **kw)


def test_erfolg_mit_standardfehler_hinweis():
    kind, text = ev.verdict_main(_a(), S)
    assert kind == "success"
    for frag in ("74 %", "63 %", "Vorhersage für den Faktor 0,26", "gemessen wurde 0,29", "Spiegelpaare 1,09", "0,42", "1,98-mal zu groß"):
        assert frag in text


def test_erfolg_ohne_standardfehler_hinweis_wenn_der_fehler_klein_ist():
    kind, text = ev.verdict_main(_a(se_ratio=1.1), S)
    assert kind == "success" and "zu groß" not in text


def test_winziger_additiver_anteil_ist_ein_hinweis():
    kind, text = ev.verdict_main(_a(add_share=0.02, top_share=0.0025, pred_lhs=0.98), S)
    assert kind == "info" and "praktisch nichts" in text and "2 %" in text


def test_zu_wenige_punkte_warnung():
    kind, text = ev.verdict_main(_a(nq=0.3), S)
    assert kind == "warning" and "N·q = 0,3" in text and "Experiment 4" in text


def test_rauschhinweis_bei_sehr_seltenen_ausfaellen():
    _, text = ev.verdict_main(_a(nq=3.0, add_share=0.02, pred_lhs=0.98), S)
    assert "rauschen die gemessenen Zahlen stark" in text


def test_nicht_bestimmbar_statt_nan():
    _, text = ev.verdict_main(_a(f_lhs=float("nan")), S)
    assert "nicht bestimmbar" in text and "nan" not in text


def test_keine_streuung():
    kind, text = ev.verdict_main(_a(add_share=float("nan"), var0=0.0), S)
    assert kind == "info" and "keine Streuung" in text


def test_n_sweep_verdikt_zahlen():
    rows = [{"N": 10, "f_lhs": 0.30, "f_anti": 0.9, "f_strat": 0.40, "nq": 1}, {"N": 100, "f_lhs": 0.26, "f_anti": 0.95, "f_strat": 0.37, "nq": 10}]
    kind, text = ev.verdict_n_sweep({"rows": rows, "pred_lhs": 0.264, "pred_strat": 0.369})
    assert kind == "info" and "0,264" in text and "0,369" in text and "zwischen 0,26 und 0,30" in text and "zwischen 0,90 und 0,95" in text and "rund 7 %" in text
    assert "keine Streuung" in ev.verdict_n_sweep({"rows": rows, "pred_lhs": float("nan"), "pred_strat": float("nan")})[1]


def test_anteil_verdikt_zahlen():
    rows = [{"p": 0.7, "EX": 0.5, "add_share": 0.47, "top_share": 0.04, "top": 8}, {"p": 0.99, "EX": 0.99, "add_share": 0.02, "top_share": 0.0025, "top": 13}]
    kind, text = ev.verdict_shares({"rows": rows})
    assert kind == "info" and "p = 0,70 tragen die Haupteffekte zusammen 47 %" in text and "p = 0,99 nur 2 %" in text and "4 %" in text
    assert "bei keinem p eine Streuung" in ev.verdict_shares({"rows": [{"p": 0.5, "add_share": float("nan"), "top_share": float("nan")}]})[1]


def test_dimension_verdikt_zahlen():
    ms = [4, 8, 16, 32, 60]
    hi = {"p": 0.99, "series": {"reihe": [0.98, 0.96, 0.93, 0.85, 0.73], "mehrheit": [0.0] * 5, "kreis": [0.1] * 5, "parallel": [0.0] * 5}}
    lo = {"p": 0.7, "series": {"reihe": [0.5] * 5, "mehrheit": [0.39, 0.29, 0.16, 0.05, 0.005], "kreis": [0.7] * 5, "parallel": [0.07] * 5}}
    _, text = ev.verdict_dimension({"ms": ms, "blocks": [lo, hi]})
    assert "p = 0,99 und 60 gleich wichtigen Kanten" in text and "Reihe 73 %" in text and "bei 8 Kanten nur 29 % (p = 0,70)" in text


def test_seltene_ausfaelle_verdikt_zahlen():
    rows = [{"N": 10, "nq": 0.1, "f_lhs": float("nan"), "f_strat": 1.0}, {"N": 100, "nq": 1.0, "f_lhs": 0.06, "f_strat": 0.06}]
    kind, text = ev.verdict_rare({"rows": rows, "pred_lhs": 0.054, "add_share": 0.946, "q": 0.01})
    assert kind == "warning" and "95 %" in text and "0,054" in text and "N = 10 (N·q = 0,1) bei nicht bestimmbar" in text and "N = 100 (N·q = 1) bei 0,06" in text


def test_intervall_verdikt_zahlen():
    rows = [{"N": 30, "se_ratio": 1.4, "cover_iid": 0.94, "cover_lhs": 0.99, "cover_blocks": 0.95}, {"N": 300, "se_ratio": 1.37, "cover_iid": 0.94, "cover_lhs": 0.99, "cover_blocks": 0.94}]
    kind, text = ev.verdict_intervals({"rows": rows})
    assert kind == "warning" and "zwischen 1,37- und 1,40-mal" in text and "N = 300: 0,99 (einfach gezogen: 0,94)" in text and "0,94 bis 0,95" in text


def test_deutsche_zahlen():
    assert ev.de(1234.567, 1) == "1.234,6" and ev.de(-0.5, 2) == "−0,50" and ev.de_e(0.000042) == "4,2e-05" and ev.dn(float("nan")) == "nicht bestimmbar"
