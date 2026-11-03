"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lhy_evaluation as ev  # noqa: E402
from lhy_scenario import grid_netz, hantel_netz  # noqa: E402


@pytest.fixture(scope="session")
def hantel():
    """Hantel: zwei Dreiecke, durch eine Brücke verbunden (7 Kanten). Handrechnung bei p = 1/2: 16 von 128 Kantenmengen hängen zusammen; Haupteffekt der Brücke 1/64."""
    netz = hantel_netz()
    return netz, ev.netz_table(netz)


@pytest.fixture(scope="session")
def gitter34():
    netz = grid_netz(3, 4)
    return netz, ev.netz_table(netz)
