"""AppTest der Experimente, der Spielwiese, des PDF-Exports und des Permalinks (Einzelheiten, die der allgemeine Rauchtest nicht trifft)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import lhy_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texte(at) -> str:
    return " ".join(e.value for kind in (at.success, at.info, at.warning, at.error) for e in kind)


@pytest.mark.parametrize("key,fragment", [
    ("run_sweep", "Vorhersage aus den Haupteffekten: Latin Hypercube 0,533"),
    ("run_rare", "Solange N·q unter 1 liegt"),
    ("run_intervals", "Der naive Standardfehler (Stichprobenstreuung durch √N)"),
])
def test_experiment_auf_klick_zeigt_sein_verdikt(key, fragment):
    at = _app()
    assert fragment not in _texte(at)                       # vor dem Klick steht noch nichts da
    at.button(key=key).click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert fragment in _texte(at)


def test_exakte_experimente_stehen_ohne_klick_da():
    t = _texte(_app())
    assert "Der Gewinn von Latin Hypercube ist nicht monoton in p" in t
    assert "Nicht die Zahl der Eingaben entscheidet" in t


def test_experiment_bleibt_nach_einer_aenderung_der_einstellungen_sichtbar():
    at = _app()
    at.button(key="run_sweep").click().run(timeout=TIMEOUT)
    at.sidebar.select_slider(key="n_select").set_value(300).run(timeout=TIMEOUT)
    assert not at.exception
    assert "Vorhersage aus den Haupteffekten" in _texte(at)


def test_kernfrage_verdikt_wechselt_mit_dem_preset():
    at = _app()
    assert "Latin Hypercube entfernt diesen Anteil" in _texte(at)
    at.sidebar.button(key="preset_Eine kritische Brücke").click().run(timeout=TIMEOUT)
    assert "Die Haupteffekte tragen 74 %" in _texte(at)
    at.sidebar.button(key="preset_Seltener Ausfall im Gitter").click().run(timeout=TIMEOUT)
    assert "praktisch nichts" in _texte(at)
    at.sidebar.button(key="preset_Seltene Brücke, zu wenige Punkte").click().run(timeout=TIMEOUT)
    assert "keine Schicht, die einen Ausfall enthält" in _texte(at)


def test_jedes_netz_laeuft_durch():
    at = _app()
    for netz in C.NETZE:
        at.sidebar.selectbox(key="netz_select").set_value(netz).run(timeout=TIMEOUT)
        assert not at.exception, (netz, [str(e) for e in at.exception])


def test_spielwiese_gueltiges_netz_und_fehlermeldungen():
    at = _app()
    assert not at.error
    at.text_area(key="eigene_kanten").set_value("0-0").run(timeout=TIMEOUT)
    assert not at.exception and "Schleife" in " ".join(e.value for e in at.error)
    at.text_area(key="eigene_kanten").set_value("").run(timeout=TIMEOUT)
    assert not at.exception and "Keine Kanten" in " ".join(e.value for e in at.error)


def test_spielwiese_pfad_laeuft_durch():
    at = _app()
    at.text_area(key="eigene_kanten").set_value("0-1\n1-2").run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert not at.error


def test_pdf_wird_erst_nach_klick_erzeugt():
    at = _app()
    assert not at.get("download_button")
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.get("download_button")) == 1
    at.sidebar.select_slider(key="n_select").set_value(300).run(timeout=TIMEOUT)         # andere Einstellung: altes PDF nicht mehr anbieten
    assert not at.get("download_button")


def test_permalink_wird_begrenzt_und_eingerastet():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["p"] = "5"
    at.query_params["n"] = "250"
    at.query_params["g"] = "9x9"            # unbekanntes Netz: verworfen
    at.query_params["seed"] = "-4"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.sidebar.slider(key="p_slider").value == C.P_MAX
    assert at.sidebar.select_slider(key="n_select").value == 300                           # 250 liegt näher an 300 als an 100
    assert at.sidebar.selectbox(key="netz_select").value == C.DEFAULT_NETZ
    assert at.sidebar.number_input(key="seed_input").value == 0
    at2 = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at2.query_params["g"] = "hantel"
    at2.run()
    assert at2.sidebar.selectbox(key="netz_select").value == "hantel"
