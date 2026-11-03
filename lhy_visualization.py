"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import plotly.graph_objects as go

import lhy_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_spread(a):
    """Verteilung der Schätzungen aus REPS Wiederholungen mit je N Punkten: einfach, Latin Hypercube, Spiegelpaare, Schichtung; gestrichelt die Wahrheit."""
    fig = go.Figure()
    for name, series, color in (("einfach", a.iid, C.COLOR_PLAIN), ("Latin Hypercube", a.lhs, C.COLOR_LHS), ("Spiegelpaare", a.anti, C.COLOR_ANTI), ("Schichtung", a.strat, C.COLOR_STRAT)):
        fig.add_trace(go.Box(y=list(series), name=name, marker_color=color, boxmean="sd", boxpoints="outliers"))
    fig.add_hline(y=a.truth, line=dict(color=C.COLOR_TRUTH, dash="dash"), annotation_text="Wahrheit", annotation_position="top right")
    fig.update_yaxes(title_text="Schätzung von R")
    fig = _base(fig, 340)
    fig.update_layout(showlegend=False)
    return fig


def build_n_sweep(exp):
    rows = exp["rows"]
    fig = go.Figure()
    ns = [str(r["N"]) for r in rows]
    for key, label, color in (("f_lhs", "Latin Hypercube", C.COLOR_LHS), ("f_anti", "Spiegelpaare", C.COLOR_ANTI), ("f_strat", "Schichtung (wirksamste Kante)", C.COLOR_STRAT)):
        fig.add_trace(go.Scatter(x=ns, y=[r[key] for r in rows], mode="markers+lines", name=label, line=dict(color=color)))
    fig.add_hline(y=exp["pred_lhs"], line=dict(color=C.COLOR_LHS, dash="dash"), annotation_text="Vorhersage Latin Hypercube", annotation_position="top right")
    fig.add_hline(y=exp["pred_strat"], line=dict(color=C.COLOR_STRAT, dash="dot"), annotation_text="Vorhersage Schichtung", annotation_position="bottom right")
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_OTHER, dash="dot"))
    fig.update_xaxes(title_text="Punkte N", type="category")
    fig.update_yaxes(title_text="Varianz gegen einfach gezogen")
    return _base(fig, 340)


def build_shares(exp):
    """Anteil der Haupteffekte je Kante an der Varianz (exakt), mit der Summe als Linie."""
    fig = go.Figure()
    labels = [f"{u}–{v}" for u, v in exp["edges"]]
    colors = [C.COLOR_STRAT if i == exp["top"] else C.COLOR_LHS for i in range(len(labels))]
    fig.add_trace(go.Bar(x=labels, y=exp["shares"], marker_color=colors, name="Anteil der Kante"))
    fig.update_xaxes(title_text="Kante", type="category")
    fig.update_yaxes(title_text="Anteil an der Varianz")
    fig = _base(fig, 300)
    fig.update_layout(showlegend=False)
    return fig


def build_share_curve(exp, p_now):
    rows = exp["rows"]
    fig = go.Figure()
    ps = [r["p"] for r in rows]
    fig.add_trace(go.Scatter(x=ps, y=[r["add_share"] for r in rows], mode="markers+lines", name="Summe der Haupteffekte (Latin Hypercube)", line=dict(color=C.COLOR_LHS)))
    fig.add_trace(go.Scatter(x=ps, y=[r["top_share"] for r in rows], mode="markers+lines", name="wirksamste Kante (Schichtung)", line=dict(color=C.COLOR_STRAT)))
    fig.add_vline(x=p_now, line=dict(color=C.COLOR_OTHER, dash="dot"), annotation_text="Regler p", annotation_position="bottom right")
    fig.update_xaxes(title_text="Haltewahrscheinlichkeit p")
    fig.update_yaxes(title_text="Anteil an der Varianz", range=[0, 1.02])
    return _base(fig, 320)


def build_dimension(block, ms):
    fig = go.Figure()
    colors = {"kreis": C.COLOR_LHS, "reihe": C.COLOR_PLAIN, "parallel": C.COLOR_BAD, "mehrheit": C.COLOR_ANTI}
    names = {"kreis": "Kreis", "reihe": "Reihe", "parallel": "Parallel", "mehrheit": "Mehrheit"}
    for kind, series in block["series"].items():
        fig.add_trace(go.Scatter(x=[str(m) for m in ms], y=series, mode="markers+lines", name=names[kind], line=dict(color=colors[kind]), connectgaps=False))
    fig.update_layout(title=dict(text=f"p = {block['p']:g}".replace(".", ","), font=dict(size=13)))
    fig.update_xaxes(title_text="Kanten m", type="category")
    fig.update_yaxes(title_text="additiver Anteil der Varianz", range=[0, 1.02])
    return _base(fig, 300)


def build_rare(exp):
    rows = exp["rows"]
    fig = go.Figure()
    xs = [str(r["N"]) for r in rows]
    fig.add_trace(go.Scatter(x=xs, y=[r["f_lhs"] for r in rows], mode="markers+lines", name="Latin Hypercube", line=dict(color=C.COLOR_LHS), connectgaps=False))
    fig.add_trace(go.Scatter(x=xs, y=[r["f_strat"] for r in rows], mode="markers+lines", name="Schichtung", line=dict(color=C.COLOR_STRAT), connectgaps=False))
    fig.add_hline(y=exp["pred_lhs"], line=dict(color=C.COLOR_LHS, dash="dash"), annotation_text="Vorhersage Latin Hypercube", annotation_position="top right")
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_OTHER, dash="dot"))
    fig.update_xaxes(title_text="Punkte N (N·q = N mal 0,01)", type="category")
    fig.update_yaxes(title_text="Varianz gegen einfach gezogen")
    return _base(fig, 330)


def build_intervals(exp):
    rows = exp["rows"]
    fig = go.Figure()
    ns = [str(r["N"]) for r in rows]
    fig.add_trace(go.Bar(x=ns, y=[r["cover_iid"] for r in rows], name="einfach gezogen, naiv", marker_color=C.COLOR_PLAIN))
    fig.add_trace(go.Bar(x=ns, y=[r["cover_lhs"] for r in rows], name="Latin Hypercube, naiv", marker_color=C.COLOR_BAD))
    fig.add_trace(go.Bar(x=ns, y=[r["cover_blocks"] for r in rows], name="Latin Hypercube, zehn Blöcke", marker_color=C.COLOR_LHS))
    fig.add_hline(y=0.95, line=dict(color=C.COLOR_TRUTH, dash="dash"), annotation_text="Soll 0,95", annotation_position="bottom right")
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Punkte N", type="category")
    fig.update_yaxes(title_text="Überdeckung des 95-%-Intervalls", range=[0, 1.02])
    return _base(fig, 320)


def build_netz(netz, shares=None):
    """Das Netz; mit `shares` (Anteil der Haupteffekte je Kante) wird die Strichstärke daran gemessen."""
    fig = go.Figure()
    xs, ys = [c[0] for c in netz.coords], [c[1] for c in netz.coords]
    top = max(shares) if shares and max(shares) > 0 else None
    for i, (u, v) in enumerate(netz.edges):
        width = 3 if top is None else 2 + 10 * shares[i] / top
        fig.add_trace(go.Scatter(x=[xs[u], xs[v]], y=[ys[u], ys[v]], mode="lines", line=dict(color="#666", width=width), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(size=22, color="white", line=dict(color="#222", width=2)), showlegend=False, hoverinfo="skip"))
    for i, (x, y) in enumerate(zip(xs, ys)):
        fig.add_annotation(x=x, y=y, text=str(i), showarrow=False, font=dict(size=10, color="#222"))
    pad = 0.5
    fig.update_xaxes(visible=False, range=[min(xs) - pad, max(xs) + pad])
    fig.update_yaxes(visible=False, range=[min(ys) - pad, max(ys) + pad])
    fig = _base(fig, 280)
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    return fig
