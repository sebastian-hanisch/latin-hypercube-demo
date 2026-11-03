"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset rechnet und hat das erwartete Verdikt."""

import pytest

import lhy_constants as C
import lhy_evaluation as ev
import lhy_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_options():
    for name, preset in C.PRESETS.items():
        assert C.P_MIN <= preset["p"] <= C.P_MAX and 0 <= preset["seed"] <= C.SEED_MAX, name
        assert preset["n"] in C.N_CHOICES and preset["netz"] in C.NETZE, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_jedes_preset_unterscheidet_sich_vom_standard():
    standard = C.PRESETS["Standardfall (Voreinstellung)"]
    for name, preset in C.PRESETS.items():
        if name != "Standardfall (Voreinstellung)":
            assert preset != standard, name


def test_preset_kritische_bruecke_hat_grossen_gewinn_und_eine_wirksame_kante():
    a = ev.analyse(ev.Settings(**C.PRESETS["Eine kritische Brücke"]))
    assert a.add_share > 0.7 and a.top_share > 0.6 and a.pred_lhs < 0.3 and a.f_lhs < 0.5


def test_preset_seltener_ausfall_im_gitter_hat_winzigen_additiven_anteil():
    a = ev.analyse(ev.Settings(**C.PRESETS["Seltener Ausfall im Gitter"]))
    assert a.add_share < 0.05
    assert ev.verdict_main(a, ev.Settings(**C.PRESETS["Seltener Ausfall im Gitter"]))[0] == "info"


def test_preset_seltene_bruecke_hat_nq_unter_eins_und_warnt():
    s = ev.Settings(**C.PRESETS["Seltene Brücke, zu wenige Punkte"])
    a = ev.analyse(s)
    assert a.nq < 1.0 and a.add_share > 0.9 and ev.verdict_main(a, s)[0] == "warning"
    gross = ev.analyse(ev.Settings(netz="hantel", p=0.99, n=300))
    assert gross.f_lhs < 0.2 < a.f_lhs                                              # mit 300 Punkten erscheint der Gewinn


def test_standard_preset_halbiert_ungefaehr_die_varianz():
    a = ev.analyse(ev.Settings())
    assert 0.4 < a.pred_lhs < 0.65 and a.f_lhs < 0.85


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_rechnet(name):
    a = ev.analyse(ev.Settings(**C.PRESETS[name]))
    assert 0.0 <= a.truth <= 1.0
