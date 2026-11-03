# Gleichmäßig abdecken: Schichtung und Latin Hypercube – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-latin-hypercube-demo.streamlit.app/)** -->

Fünftes Stück der Linie **„Monte-Carlo-Schätzung und Varianzreduktion“** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (die Linie wird erst vollständig veröffentlicht). Nachfolger von [mehrere-modelle-demo](https://github.com/sebastian-hanisch/mehrere-modelle-demo) (Stück 4), [kontrollvariaten-demo](https://github.com/sebastian-hanisch/kontrollvariaten-demo) (Stück 3) und [gekoppelte-ziehungen-demo](https://github.com/sebastian-hanisch/gekoppelte-ziehungen-demo) (Stück 2, Spiegelpaare als einfachste Schichtung). Hier hebt das Stück die Annahme „die Punkte sind unabhängig gezogen“ auf: Die Punkte decken jede Eingabe **gleichmäßig** ab. Vehikel ist wie in Stück 1 bis 3 die Zuverlässigkeit eines Gitternetzes (Ziel: das Netz hängt zusammen), dazu eine **Hantel** (zwei Dreiecke, durch eine einzige Brücke verbunden). Die Wahrheit ist exakt aufgezählt.

## Warum dieses Problem

Latin Hypercube teilt den Wertebereich jeder Eingabe in N Schichten und zieht aus jeder genau einen Punkt, die Spalten unabhängig vertauscht. Die Frage: **Was bringt das, und woran erkennt man es vorher?** Die Antwort ist überraschend scharf: Latin Hypercube entfernt aus der Varianz **genau den additiven Anteil**, die Summe der Haupteffekte. Der Gewinn ist deshalb aus der Aufzählung **exakt vorhersagbar** und hängt an der **Additivität** der Funktion, nicht an der Zahl der wichtigen Eingaben.

## Befunde und Korrekturen gegenüber dem Plan

Vor dem Bau wurden die Hypothesen der Gliederung an Messreihen geprüft (Vormessung im Website-Repo, `docs/planung/monte-carlo-planung/vormessung/`). Was sich geändert hat:

- **„Gewinn groß bei wenigen wirksamen Eingaben, klein bei vielen gleich wichtigen“ trifft nur halb.** Eine Reihe aus 60 gleich wichtigen Kanten hat bei p = 0,99 einen additiven Anteil von 0,73 (großer Gewinn), Parallel und Mehrheit schon bei wenigen Kanten fast null, und das Gitter 3×4 liegt zwischen 0,47 (p = 0,7) und 0,02 (p = 0,99). Entscheidend ist, ob die Funktion fast additiv ist.
- **Der Gewinn ist nicht monoton in p** (Gitter, Kreis): erst steigend, dann fallend.
- **Neu: Der naive Standardfehler ist unter Latin Hypercube bis zum Zweifachen (Hantel) zu groß**, die Intervalle decken zu viel ab; zehn unabhängige Latin Hypercubes geben ein richtiges Intervall.
- **Neu: Bei seltenen Ausfällen kommt der Gewinn erst ab N·q ≈ 1**; die Schichtung nach der wirksamsten Kante ist dort robuster als Latin Hypercube.
- Die Spiegelung (Stück 2) ist der Vergleichsmaßstab: Latin Hypercube gewinnt dort, wo die Spiegelung nichts nutzt.

## Modell

- **Ziel:** X = 1, wenn das Netz zusammenhängt; Kante e hält, wenn ihre Gleichverteilte uₑ unter p liegt. Netze: Gitter 2×3, 3×3, 3×4 und die Hantel (bis 17 Kanten, alle Kantenmengen live aufgezählt).
- **Latin Hypercube:** uᵢₑ = (πₑ(i) + vᵢₑ)/N mit zufälliger Permutation πₑ der N Schichten und gleichverteiltem Zitter v. Der Schätzer ist das Mittel der N Auswertungen.
- **Gewinn (Owen 1997):** Var ≈ (σ² − Σₑ Var(E[f | uₑ]))/N; der Haupteffekt jeder Kante ist p(1−p)(a₁ − a₀)² mit a₁ = E[X | e hält], a₀ = E[X | e fällt aus]. Faktor gegen einfaches Ziehen: **1 − Σ der Haupteffekt-Anteile**.
- **Schichtung nach einer Kante:** round(N·p) Punkte mit haltender, der Rest mit ausgefallener Kante, nachgewichtet p·Mittel(hält) + (1−p)·Mittel(fällt aus); erwartungstreu auch bei nicht ganzzahligem N·p, Gewinn gleich dem Anteil dieser Kante. Mathematik im 📐-Expander der App.

## Methodik

Zufallszahlen aus festen SplitMix64-Strömen (nie ein einzelner Seed). Wahrheit und Haupteffekte aus der vollständigen Aufzählung. Je Messung 300 bis 400 Wiederholungen mit festen Strömen; Varianzverhältnisse rauschen um rund 7 bis 10 %, bei seltenen Ausfällen deutlich mehr. Vier Schätzer mit je N Auswertungen: einfach, Latin Hypercube, Spiegelpaare (N/2 Paare), Schichtung nach der wirksamsten Kante.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Standardfall: Gitter 3×4, p = 0,7, N = 100.

| Frage | Befund |
|---|---|
| Wie viel der Varianz steckt in den Haupteffekten? (exakt, 3×4, p = 0,5 / 0,7 / 0,8 / 0,9 / 0,95 / 0,99) | **0,204 / 0,467 / 0,403 / 0,214 / 0,104 / 0,020**; die wirksamste einzelne Kante trägt höchstens 0,043 |
| dasselbe für die Hantel (eine Brücke) | **0,357 / 0,597 / 0,669 / 0,736 / 0,812 / 0,946**; die Brücke allein 0,143 / 0,324 / 0,449 / **0,631** / 0,773 / **0,944** |
| Vorhersage und Messung, Gitter 3×4, p = 0,7 | Faktor Latin Hypercube vorhergesagt **0,533**; gemessen über N = 10 / 30 / 100 / 300 zwischen **0,50 und 0,56** (im Standardfall mit N = 100: 0,63, Rauschen), Spiegelpaare 0,73 bis 0,86, Schichtung nach der wirksamsten Kante 0,96 bis 1,02 (vorhergesagt 0,957) |
| Vorhersage und Messung, Hantel, p = 0,9 | Latin Hypercube vorhergesagt **0,264**, gemessen **0,25 bis 0,28**; Schichtung nach der Brücke vorhergesagt **0,369**, gemessen **0,33 bis 0,42**; Spiegelpaare **0,81 bis 1,02**: die Spiegelung nutzt die Brücke nicht |
| Gitter 3×4, p = 0,99 | Haupteffekte 0,020: vorhergesagter Faktor **0,980**, kein Gewinn |
| Viele gleich wichtige Kanten (exakt, m = 4 / 8 / 16 / 32 / 60) | p = 0,99: **Reihe** 0,985 / 0,965 / 0,926 / 0,852 / **0,732**; p = 0,9: Reihe 0,848 / 0,672 / 0,404 / 0,126 / 0,012, **Kreis** 0,429 / 0,656 / 0,679 / 0,377 / 0,068; p = 0,7: **Mehrheit** 0,391 / 0,291 / 0,162 / 0,049 / 0,005, Parallel 0,076 bei m = 4 und danach unter 0,01 |
| Seltene Brücke (Hantel, p = 0,99), N = 10 / 30 / 100 / 300 / 1000, N·q = 0,1 / 0,3 / 1 / 3 / 10 | Latin Hypercube **1,36 / 0,96 / 0,038 / 0,046 / 0,055** (vorhergesagt 0,054): der Gewinn erscheint erst ab N·q ≈ 1; Schichtung nach der Brücke 0,075 / 0,043 / 0,030 / 0,040 / 0,043: robuster bei kleinem N |
| Seltener Ausfall im Gitter (3×4, p = 0,99) | Latin Hypercube 1,0 / 0,58 / 1,02 / 1,11 für N = 30 / 100 / 300 / 1000 (vorhergesagt 0,980): Rauschen um 1, kein Gewinn |
| Naiver Standardfehler unter Latin Hypercube gegen die wahre Streuung | Gitter 3×4, p = 0,7: **1,33 bis 1,41**; Hantel, p = 0,9: **1,91 bis 1,99** |
| Überdeckung des 95-%-Intervalls | einfach gezogen 0,93 bis 0,96; naiv unter Latin Hypercube **0,98 bis 0,99** (3×4) und **1,00** (Hantel); zehn Latin-Hypercube-Blöcke **0,95 bis 0,96** (3×4) und 0,92 bis 0,96 (Hantel) |

## Ehrliche Grenzen

- **Die Vorhersage ist asymptotisch** (Owen 1997); bei kleinem N und seltenen Ausfällen weicht die Messung ab (Tabelle: N·q unter 1). Die exakten Anteile sind belastbar, einzelne gemessene Faktoren rauschen um 7 bis 10 %, im Standardfall (N = 100) liegt die Messung 0,63 über der Vorhersage 0,533.
- **Nur das Ziel „Zusammenhang“** und Netze bis 17 Kanten; unabhängige, gleiche Ausfälle.
- **Kein Vergleich mit gescramblten Sobol-Folgen** (Stück 9). Das Intervall aus zehn Blöcken verbraucht N/10 Punkte je Block und gibt dafür etwas Gewinn ab; wie viel, ist nicht ausgewertet.
- **Die Schichtung nach einer einzigen Kante** holt nur deren Haupteffekt; mehrdimensionale Schichten wachsen exponentiell mit der Dimension und sind nicht gebaut.
- **Literaturangaben im Mathematik-Expander** stammen aus dem Gedächtnis und sind nicht eingesehen.

## Verifikation

- **Kern** (`tests/test_algorithm.py`): Zufallsstrom gegen Ganzzahl-Referenz, Latin-Hypercube-Eigenschaft (je Spalte genau ein Punkt je Schicht, Zitter in der Schicht), Maskenpackung, Hantel von Hand (16 von 128 Kantenmengen; Haupteffekt der Brücke 1/64 mit Brüchen), symmetrische Ziele, Spiegelpaare und Schichtung an kleinen Fällen.
- **Orakel** (`tests/test_oracle_lhy.py`): Zusammenhang gegen `networkx` auf allen Kantenmengen von Hantel, 2×3 und 3×3; **Haupteffekte gegen eine zweite Rechnung mit Brüchen** und Zerlegung (Summe der Haupteffekte höchstens Var X); symmetrische Formel gegen die Aufzählung; **Latin Hypercube in einer Dimension gegen die exakte Schichtungsvarianz**; **Schichtung gegen die bedingten Varianzen** (Var X − Haupteffekt)/N; Latin-Hypercube-Varianz gegen die exakte Vorhersage und Owens Schranke N/(N−1); alle Schätzer erwartungstreu.
- **Eigenschaften** (`tests/test_properties.py`): additive Funktion wird fast exakt gemittelt, Anteile in [0, 1], Brücke wächst mit p, Gitter nicht monoton, Reihe additiv und Parallel nicht, Spiegelpaare nutzen die Brücke nicht, naiver Standardfehler zu groß.
- **Randfälle** (`tests/test_edge_cases.py`): Eingabefehler der Spielwiese, Pfad und einzelne Kante, kleinste Punktzahlen, jedes Netz mit jedem N, p an den Grenzen, seltene Ausfälle.
- **Verdikte** (`tests/test_verdicts.py`): jede Schwelle und jede genannte Zahl an von Hand gesetzten Werten, deutsches Zahlenformat.
- **Presets, Layout, Oberfläche:** Presets vollständig und jedes mit dem erwarteten Verdikt; Schnellstart oben, fester Kopf, aktives Preset hervorgehoben; AppTest für jeden Button, jeden Regler an Minimum und Maximum, die Experimente, die Spielwiese, den PDF-Export und den Permalink.
- **Laufzeit-Importe, Footer, Claims.**

Zahl der Tests: siehe `pytest --collect-only -q tests`.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `lhy_algorithm.py` | Kern: Strom, Latin Hypercube, Aufzählung, Haupteffekte, Schätzer |
| `lhy_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `lhy_evaluation.py` | Kennzahlen, Verdikt-Logik, fünf Experimente über feste Ströme |
| `lhy_layout.py` | Fester Kopf (Stylesheet) |
| `lhy_pdf_export.py` | PDF der aktuellen Einstellung |
| `lhy_presets.py` | Permalink (Begrenzen), Presets, Zufallsstrom-Knopf |
| `lhy_scenario.py` | Gitter, Hantel und eigene Netze |
| `lhy_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest und networkx |
| `requirements.txt` | Laufzeit-Bibliotheken |
| `tests/` | Kern von Hand, Orakel, Eigenschaften, Randfälle, Verdikte, Presets, Layout, AppTest, Footer, Laufzeit-Importe, Claims |

## Bewusst nicht umgesetzt

| Annahme | Verweis |
|---|---|
| Quasi-Zufallsfolgen mit Randomisierung | Stück 9 der Linie (randomisiertes Quasi-Monte-Carlo) |
| Welche Eingabe treibt die Varianz (Sobol-Indizes) | Stück 10 der Linie |
| Seltene Ereignisse und Gewichte | Stück 7 der Linie (Importance Sampling) |
| Eine Größe mit bekanntem Mittelwert | [kontrollvariaten-demo](https://github.com/sebastian-hanisch/kontrollvariaten-demo) (Stück 3) |
| Spiegelpaare | [gekoppelte-ziehungen-demo](https://github.com/sebastian-hanisch/gekoppelte-ziehungen-demo) (Stück 2) |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
