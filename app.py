"""Gleichmäßig abdecken: Schichtung und Latin Hypercube - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 5 der Linie "Monte-Carlo-Schätzung und Varianzreduktion": Latin Hypercube deckt jede Eingabe für sich gleichmäßig ab und entfernt für großes N den additiven Anteil der Varianz
(die Summe der Haupteffekte). Der Gewinn ist vorhersagbar und hängt an der Additivität, nicht an der Zahl der Eingaben. Vehikel ist die Zuverlässigkeit eines
Gitternetzes (und einer Hantel mit einer kritischen Brücke), die Wahrheit ist exakt aufgezählt.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import lhy_constants as C
import lhy_evaluation as ev
from lhy_evaluation import Settings, de, dn
from lhy_layout import dunkles_design, kopf_css
from lhy_pdf_export import build_pdf
from lhy_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    options,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from lhy_scenario import named_netz, parse_edges
from lhy_visualization import build_dimension, build_intervals, build_n_sweep, build_netz, build_rare, build_share_curve, build_shares, build_spread

st.set_page_config(page_title="Gleichmäßig abdecken: Schichtung und Latin Hypercube – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Alle Kantenmengen aufzählen …")
def _table(netz_key):
    return ev.netz_table(named_netz(netz_key))


@st.cache_data(show_spinner="Schätzungen wiederholen …")
def _analysis(settings, table):
    return ev.analyse(settings, table=table)


@st.cache_data(show_spinner="Varianz gegen N messen …")
def _n_sweep(settings, table):
    return ev.n_sweep(settings, named_netz(settings.netz), table)


@st.cache_data(show_spinner=False)
def _shares(settings, table):
    return ev.share_table(settings, named_netz(settings.netz), table)


@st.cache_data(show_spinner=False)
def _dimension():
    return ev.dimension()


@st.cache_data(show_spinner="Seltene Ausfälle messen …")
def _rare(settings, table):
    return ev.rare_events(settings, named_netz(settings.netz), table)


@st.cache_data(show_spinner="Intervalle messen …")
def _intervals(settings, table):
    return ev.intervals(settings, named_netz(settings.netz), table)


@st.cache_data(show_spinner=False)
def _custom(text, settings):
    netz = parse_edges(text)
    return netz, ev.custom_result(settings, netz)


def _netz_label(key):
    netz = named_netz(key)
    return f"{netz.name} ({netz.m} Kanten)"


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in lhy_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🎲 Gleichmäßig abdecken: Schichtung und Latin Hypercube")
st.markdown(
    """
**Latin Hypercube** teilt den Wertebereich jeder Eingabe in N gleich große Schichten und zieht aus jeder genau einen Punkt; die Spalten werden unabhängig vertauscht. Er entfernt für großes N **den additiven Anteil** der Varianz, die Summe der Haupteffekte:
Var ≈ (σ² − Σ Var(E[f | uₑ])) / N. Der Gewinn ist deshalb **vorhersagbar** und hängt **nicht an der Zahl der wichtigen Eingaben, sondern daran, wie additiv die Funktion ist**.
Bei einer kritischen Brücke ist er groß, wo erst das Zusammenspiel der Kanten zählt kaum, und bei seltenen Ausfällen erst, wenn N·q etwa 1 erreicht. Die **Schichtung nach einer einzigen Eingabe** holt den Haupteffekt dieser Eingabe heraus.
Eine Falle: Der **naive Standardfehler stimmt unter Latin Hypercube nicht** (bei der Hantel mit p = 0,9 ist er etwa 1,9-fach zu groß und die Intervalle decken zu viel ab; bei N·q unter 1 kann er auch zu klein sein).

Weiter unten: **🔬 fünf Experimente**, **🧪 Spielwiese**, **🚧 Wo die Annahmen enden** und die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert Latin Hypercube", expanded=True):
    st.markdown(
        """
1. **Ziehen:** Jede Kante hält unabhängig mit Wahrscheinlichkeit p, das Ziel X ist „das Netz hängt zusammen“. Eine Kante hält, wenn ihre Gleichverteilte uₑ unter p liegt.
2. **Latin Hypercube:** N Punkte; für jede Kante eine zufällige Reihenfolge der N Schichten [0, 1/N), …, [(N−1)/N, 1), darin je ein gleichverteilter Punkt. Jede Kante fällt dadurch in fast genau (1−p)·N Punkten aus, statt zufällig mehr oder weniger oft.
3. **Der Gewinn:** Zerlegt man X in einen additiven Teil (Summe der Haupteffekte E[X | uₑ] − E X) und den Rest (Zusammenspiel), nimmt Latin Hypercube den additiven Teil aus der Varianz. Der Faktor gegen einfaches Ziehen ist **1 − (Summe der Haupteffekt-Anteile)**.
4. **Schichtung nach einer Kante:** Man zieht n = round(N·p) Punkte mit haltender und den Rest mit ausgefallener Kante und gewichtet nach. Der Gewinn ist der Anteil dieser einen Kante, exakt bei ganzzahligem N·p, sonst näherungsweise.
5. **Standardfehler:** Die Streuung der Auswertungen durch √N überschätzt die Streuung des Mittels unter Latin Hypercube. Zehn unabhängige Latin Hypercubes der Größe N/10 geben ein brauchbares Intervall aus der Streuung ihrer Mittel, wenn jeder Block genug Ausfälle sieht.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Netz**")
    netz_key = st.selectbox("Netz", options("netz_select"), format_func=_netz_label, key="netz_select",
                            help="Gitter (Knoten in Zeilen und Spalten) oder die Hantel: zwei Dreiecke, durch eine einzige Brücke verbunden. Die Wahrheit wird live aufgezählt.")
    p = st.slider("Haltewahrscheinlichkeit p je Kante", *bounds("p_slider"), key="p_slider", step=0.01, help="Jede Kante hält unabhängig mit dieser Wahrscheinlichkeit; bei p nahe 1 sind Ausfälle selten.")
    st.markdown("**Schätzung**")
    n = st.select_slider("Punkte N", options("n_select"), key="n_select", help="Punkte je Schätzung; alle Schätzer benutzen dieselbe Zahl von Auswertungen. Die Spiegelpaare brauchen eine gerade Zahl.")
    seed = st.number_input("Zufallsstrom", *bounds("seed_input"), key="seed_input", step=1, help="Welcher feste Strom von Zufallszahlen benutzt wird (SplitMix64).")
    st.button("🎲 Neuen Strom wählen", width="stretch", on_click=randomize_seed)

sync_query_params({"netz_select": netz_key, "p_slider": float(p), "n_select": int(n), "seed_input": int(seed)})

settings = Settings(str(netz_key), float(p), int(n), int(seed))
table = _table(settings.netz)
a = _analysis(settings, table)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Wahrheit R", de(a.truth, 4), help=f"Exakte Zuverlässigkeit des {a.netz.name}s bei p = {de(settings.p, 2)} (Aufzählung aller Kantenmengen).")
m2.metric("Anteil der Haupteffekte", f"{de(a.add_share * 100, 0)} %" if a.add_share == a.add_share else "–", help="Summe der Haupteffekt-Anteile an der Varianz (exakt): so viel der Varianz entfernt Latin Hypercube, wenn N groß ist.")
m3.metric("Faktor Latin Hypercube", dn(a.f_lhs, 2) if a.f_lhs == a.f_lhs else "–", delta=f"Vorhersage {de(a.pred_lhs, 2)}" if a.pred_lhs == a.pred_lhs else None, delta_color="off",
          help=f"Varianz gegen einfaches Ziehen, gemessen aus {C.REPS} Wiederholungen mit festen Strömen. Rauschen etwa 7 %, bei seltenen Ausfällen mehr.")
m4.metric("Faktor Schichtung", dn(a.f_strat, 2) if a.f_strat == a.f_strat else "–", delta=f"Vorhersage {de(a.pred_strat, 2)}" if a.pred_strat == a.pred_strat else None, delta_color="off",
          help="Schichtung nach der wirksamsten Kante, nachgewichtet.")
st.caption(f"Wirksamste Kante: {a.netz.edges[a.top][0]}–{a.netz.edges[a.top][1]} mit {de(a.top_share * 100, 1)} % der Varianz. Spiegelpaare (Stück 2): Faktor {dn(a.f_anti, 2)}. "
           f"Naiver Standardfehler unter Latin Hypercube gegen die wahre Streuung: {dn(a.se_ratio, 2)}-mal. N·q = {de(a.nq, 1)}." if a.add_share == a.add_share else "Das Ziel hat bei diesen Einstellungen keine Streuung.")
st.plotly_chart(build_spread(a), width="stretch", key="main_chart")
st.caption("Kästen: Verteilung der Schätzungen über die Wiederholungen bei gleicher Zahl von Auswertungen (einfach, Latin Hypercube, Spiegelpaare, Schichtung nach der wirksamsten Kante; Linie: Median, Raute und Balken: Mittel und Streuung). Gestrichelt: die Wahrheit.")

if st.button("📄 PDF der Schätzung erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="latin-hypercube.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Was bringt gleichmäßiges Abdecken, und woran erkennt man es vorher?")
kind, text = ev.verdict_main(a, settings)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Varianz gegen die Zahl der Punkte")
st.caption(f"Varianzverhältnis gegen einfaches Ziehen bei N = {', '.join(str(x) for x in C.SWEEP_NS)}, je {C.SWEEP_REPS} Wiederholungen mit festen Strömen; gewähltes Netz und p. Gestrichelt: die Vorhersage aus den exakten Haupteffekten. Kostet einige Sekunden.")
if st.button("▶ Messung starten", key="run_sweep"):
    st.session_state["sweep_done"] = True
if st.session_state.get("sweep_done"):
    ex1 = _n_sweep(settings, table)
    if ex1["pred_lhs"] == ex1["pred_lhs"]:
        st.plotly_chart(build_n_sweep(ex1), width="stretch", key="sweep_chart")
    kind, text = ev.verdict_n_sweep(ex1)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 2: Wer trägt den Gewinn?")
st.caption("Der **exakte** Anteil, den jede Kante als Haupteffekt an der Varianz hat (aus der Aufzählung aller Kantenmengen, kein Zufall). Die Summe ist der Anteil, den Latin Hypercube entfernt; die größte Einzelkante der, den die Schichtung nach ihr entfernt.")
ex2 = _shares(settings, table)
if ex2["add_share"] == ex2["add_share"]:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(build_shares(ex2), width="stretch", key="shares_chart")
    with c2:
        st.plotly_chart(build_netz(named_netz(settings.netz), ex2["shares"]), width="stretch", key="shares_net")
    st.plotly_chart(build_share_curve(ex2, settings.p), width="stretch", key="share_curve")
    st.dataframe(
        {"p": [de(r["p"], 2) for r in ex2["rows"]], "E X": [de(r["EX"], 4) for r in ex2["rows"]], "Summe der Haupteffekte": [dn(r["add_share"], 3) for r in ex2["rows"]],
         "wirksamste Kante": [dn(r["top_share"], 3) for r in ex2["rows"]]},
        hide_index=True, width="stretch", key="shares_table",
    )
kind, text = ev.verdict_shares(ex2)
getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 3: Viele gleich wichtige Eingaben")
st.caption("Der additive Anteil der Varianz für vier symmetrische Ziele bei m gleich wichtigen Kanten (binomial exakt, kein Zufall): Kreis (hängt zusammen, wenn höchstens eine Kante ausfällt), Reihe (alle halten), Parallel (mindestens eine hält), Mehrheit (mindestens die Hälfte hält). Fehlende Punkte: das Ziel ist praktisch konstant.")
ex3 = _dimension()
cols = st.columns(len(ex3["blocks"]))
for col, block in zip(cols, ex3["blocks"]):
    with col:
        st.plotly_chart(build_dimension(block, ex3["ms"]), width="stretch", key=f"dim_chart_{block['p']}")
kind, text = ev.verdict_dimension(ex3)
getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 4: Seltene Ausfälle")
st.caption(f"Bei p = {str(C.RARE_P).replace('.', ',')} (Ausfallwahrscheinlichkeit q = 0,01) und N = {', '.join(str(x) for x in C.RARE_NS)}: Latin Hypercube und Schichtung, je {C.RARE_REPS} Wiederholungen. Der Gewinn kann erst erscheinen, wenn N·q etwa 1 erreicht. Kostet einige Sekunden.")
if st.button("▶ Messung starten", key="run_rare"):
    st.session_state["rare_done"] = True
if st.session_state.get("rare_done"):
    ex4 = _rare(settings, table)
    if ex4["pred_lhs"] == ex4["pred_lhs"]:
        st.plotly_chart(build_rare(ex4), width="stretch", key="rare_chart")
        st.dataframe(
            {"N": [r["N"] for r in ex4["rows"]], "N·q": [de(r["nq"], 1) for r in ex4["rows"]], "Faktor Latin Hypercube": [dn(r["f_lhs"], 2) for r in ex4["rows"]],
             "Faktor Schichtung": [dn(r["f_strat"], 2) for r in ex4["rows"]]},
            hide_index=True, width="stretch", key="rare_table",
        )
    kind, text = ev.verdict_rare(ex4)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 5: Standardfehler und Intervalle")
st.caption(f"Der naive Standardfehler (Streuung der Auswertungen durch √N) gegen die wahre Streuung der Latin-Hypercube-Mittel, und die Überdeckung der 95-%-Intervalle bei N = {', '.join(str(x) for x in C.INT_NS)}, je {C.INT_REPS} Wiederholungen. Ausweg: {C.BLOCKS} unabhängige Latin Hypercubes der Größe N/{C.BLOCKS}.")
if st.button("▶ Messung starten", key="run_intervals"):
    st.session_state["intervals_done"] = True
if st.session_state.get("intervals_done"):
    ex5 = _intervals(settings, table)
    if any(r["se_ratio"] == r["se_ratio"] for r in ex5["rows"]):
        st.plotly_chart(build_intervals(ex5), width="stretch", key="intervals_chart")
        st.dataframe(
            {"N": [r["N"] for r in ex5["rows"]], "Standardfehler naiv / wahr": [dn(r["se_ratio"], 2) for r in ex5["rows"]], "Überdeckung einfach": [de(r["cover_iid"], 2) for r in ex5["rows"]],
             "Überdeckung naiv unter Latin Hypercube": [de(r["cover_lhs"], 2) for r in ex5["rows"]], "Überdeckung zehn Blöcke": [de(r["cover_blocks"], 2) for r in ex5["rows"]]},
            hide_index=True, width="stretch", key="intervals_table",
        )
    kind, text = ev.verdict_intervals(ex5)
    getattr(st, kind)(text)

st.markdown("---")

# --- Spielwiese -----------------------------------------------------------------------------------------------------------------------------

with st.expander("🧪 Spielwiese: ein eigenes Netz"):
    st.caption(
        f"Eine Kante je Zeile, zwei ganze Zahlen (»0-1«). Höchstens {C.CUSTOM_EDGE_LIMIT} Kanten und {C.CUSTOM_NODE_LIMIT} Knoten; p, N und Strom stammen aus der Seitenleiste. "
        "Es zeigt die exakten Haupteffekt-Anteile (Strichstärke) und die Schätzer."
    )
    text_in = st.text_area("Kanten", "0-1\n1-2\n2-0\n3-4\n4-5\n5-3\n2-3", height=140, key="eigene_kanten")
    try:
        cnetz, cres = _custom(text_in, settings)
    except ValueError as err:
        st.error(str(err))
    else:
        ca = cres["analysis"]
        s1, s2, s3 = st.columns(3)
        s1.metric("Wahrheit R", de(ca.truth, 4))
        s2.metric("Anteil der Haupteffekte", f"{de(ca.add_share * 100, 0)} %" if ca.add_share == ca.add_share else "–")
        s3.metric("Faktor Latin Hypercube", dn(ca.f_lhs, 2) if ca.f_lhs == ca.f_lhs else "–")
        st.plotly_chart(build_netz(cnetz, list(cres["shares"]) if ca.add_share == ca.add_share else None), width="stretch", key="custom_net")
        kind, text = ev.verdict_main(ca, settings)
        getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Ziel ist fast additiv in den Eingaben** | Latin Hypercube entfernt nur den additiven Anteil; steckt die Varianz im Zusammenspiel (Gitter bei seltenem Ausfall, Parallel, Mehrheit), bringt er fast nichts (Experimente 2 und 3). | Stück 10 (Sobol-Indizes: welche Eingabe treibt die Varianz) |
| **Es gibt in den Schichten Ereignisse** | Bei seltenen Ausfällen mit N·q unter 1 enthält nur die oberste Schicht den Ausfallbereich, und davon nur den Anteil N·q; der Gewinn ist klein und wächst erst ab N·q von etwa 1 voll (Experiment 4). | Stück 7 (Importance Sampling) |
| **Der Standardfehler gilt wie beim einfachen Ziehen** | Unter Latin Hypercube ist er meist zu groß (bei der Hantel mit p = 0,9 etwa 1,9-fach), das Intervall zu breit; ein brauchbares Intervall braucht wiederholte Latin Hypercubes mit genug Ausfällen je Block (Experiment 5). | – |
| **Wenige, unabhängige Eingaben** | Latin Hypercube deckt nur die Ränder je Eingabe ab, nicht den Raum; bei vielen Eingaben bleibt die Abdeckung des Raums dünn. | Stück 9 (randomisierte Quasi-Monte-Carlo-Folgen) |
| **Eine Eingabe schichten** | Die Schichtung nach einer Kante holt nur deren Haupteffekt; mehrdimensionale Schichten wachsen mit der Dimension exponentiell. | – |
| **Eine Variable erklärt das Ergebnis** | Dann ist die Kontrollvariate (Stück 3) oft besser als das Abdecken. | Stück 3 |
"""
)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen.** Alle Netze und Ausfälle sind **erfunden**; die Wahrheit ist die exakte Aufzählung. Die Kanten fallen unabhängig und mit gleicher Wahrscheinlichkeit aus. Die Zufallszahlen kommen aus festen Strömen (SplitMix64), nie aus einem einzelnen Seed.

**Grenzen.** Aufgezählt wird live bis 17 Kanten. Ziel ist nur der Zusammenhang. Der Gewinn von Latin Hypercube ist die **asymptotische** Vorhersage (Owen 1997); bei kleinem N weicht die Messung ab und rauscht. Der Vergleich mit gescramblten Sobol-Folgen fehlt (Stück 9). Das Intervall aus Blöcken verbraucht N/10-Punkte je Block und gibt dafür etwas Gewinn ab; wie viel, ist nicht ausgewertet.

**Bekannte Probleme und Unsicherheiten.** Varianzverhältnisse aus 300 bis 400 Wiederholungen rauschen um rund 7 bis 10 %; bei p = 0,99 und kleinem N enthalten die meisten Wiederholungen keinen Ausfall, die Messung ist dort nicht belastbar (die exakte Vorhersage schon). Die Literaturangaben im Mathematik-Expander stammen aus dem Gedächtnis und sind nicht eingesehen. Die Oberfläche ist mit AppTest geprüft und im Browser angesehen, nicht mit echtem Mausklick auf jedem Gerät.
        """
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem.** Gesucht ist $R = \mathbb E\,f(U)$ mit $f(u) = \mathbf 1[\text{Netz hängt zusammen, wenn Kante } e \text{ hält} \Leftrightarrow u_e < p]$ und $U \sim \mathrm{Unif}[0,1)^m$.

**Latin Hypercube.** Für $N$ Punkte und jede Eingabe $e$ eine zufällige Permutation $\pi_e$ von $\{0,\dots,N-1\}$ und Zitter $v_{ie} \sim \mathrm{Unif}[0,1)$: $u_{ie} = (\pi_e(i) + v_{ie})/N$. Jede Randverteilung ist gleichverteilt, der Schätzer $\bar f = \frac1N \sum_i f(u_i)$ ist erwartungstreu.

**Varianz (Owen 1997).** Mit der ANOVA-Zerlegung $f = \mu + \sum_e f_e(u_e) + r(u)$, $f_e(u_e) = \mathbb E[f \mid u_e] - \mu$:
$$\operatorname{Var}\bar f = \frac{\sigma^2 - \sum_e \operatorname{Var} f_e}{N} + o(1/N),\qquad \operatorname{Var} f_e = p(1-p)(a_{1,e} - a_{0,e})^2,$$
mit $a_{1,e} = \mathbb E[f \mid e \text{ hält}]$ und $a_{0,e} = \mathbb E[f \mid e \text{ fällt aus}]$. Der Faktor gegen einfaches Ziehen ist $1 - \sum_e \operatorname{Var} f_e / \sigma^2$. Für jedes $f$ gilt $\operatorname{Var}\bar f \le \frac{N}{N-1}\cdot\sigma^2/N$.

**Schichtung nach einer Eingabe.** $n_1 = \operatorname{round}(Np)$ Punkte mit $u_e < p$, $N - n_1$ mit $u_e \ge p$, übrige Eingaben unabhängig; $\hat R = p\,\bar f_{\text{hält}} + (1-p)\,\bar f_{\text{fällt aus}}$. Bei ganzzahligem $n_1 = Np$ ist die Varianz $\sigma^2(1 - \mathrm{Anteil}_e)/N$, sonst ist das eine Näherung.

**Standardfehler.** Der naive Standardfehler $s/\sqrt N$ schätzt $\sqrt{\sigma^2/N}$, nicht $\sqrt{\operatorname{Var}\bar f}$ und ist deshalb unter Latin Hypercube meist zu groß (der Faktor ist etwa $\sqrt{1/(1-\text{Anteil der Haupteffekte})}$; bei $N\cdot q<1$ kann er auch zu klein sein). Zehn unabhängige Latin Hypercubes der Größe $N/10$ ergeben ein Intervall aus der Streuung ihrer Mittel ($t$-Quantil mit 9 Freiheitsgraden).

**Wahrheit.** Die Aufzählung aller $2^m$ Kantenmengen liefert $\mu$, $\sigma^2$ und alle Haupteffekte exakt, mit den Gewichten $p^k(1-p)^{m-k}$. Für symmetrische Ziele $f(k)$ genügt die Binomialverteilung.

**Literatur (aus dem Gedächtnis, nicht eingesehen).** McKay, Beckman und Conover 1979, „A comparison of three methods for selecting values of input variables in the analysis of output from a computer code“ (Technometrics); Stein 1987, „Large sample properties of simulations using Latin hypercube sampling“ (Technometrics); Owen 1997, „Monte Carlo variance of scrambled net quadrature“ (SIAM J. Numer. Anal.) und Owen 1992, „A central limit theorem for Latin hypercube sampling“ (JRSS B).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
